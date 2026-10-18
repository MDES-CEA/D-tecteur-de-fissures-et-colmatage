"""Stockage local versionne des zones et des traces de fissures."""

from __future__ import annotations

import sqlite3
import json
import math
import os
import re
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


ZONE_STATUSES = ("todo", "in_progress", "review", "done")
# « reopened » : gel d'alteration sur les parois d'une fissure aujourd'hui vide
# (rouverte par la preparation, la resine...). Elle a donc ete colmatee.
CRACK_CATEGORIES = ("unclassified", "open", "filled", "reopened", "mixed", "uncertain")
CRACK_STATES = ("proposed", "validated", "rejected")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ZoneConflict(RuntimeError):
    pass


class CrackConflict(RuntimeError):
    pass


def _same_path(first: str | Path, second: str | Path) -> bool:
    """Deux chemins designent-ils le meme fichier ? Windows ignore la casse."""
    return os.path.normcase(str(first)) == os.path.normcase(str(second))


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "mosaique"


def already_registered_message(owner_id: str, image_id: str, resolved: str | Path) -> str:
    """Le meme fichier deja enregistre sous un autre identifiant."""
    return (
        f"Cette image est deja enregistree sous l'identifiant "
        f"« {owner_id} », pas « {image_id} ».\n"
        f"  {resolved}\n"
        f"Relancer sans CRACK_IMAGE_ID pour reprendre le travail "
        f"existant, ou definir CRACK_IMAGE_ID={owner_id}."
    )


def resolve_among(images, image_path: Path, width: int, height: int,
                  reserved=()) -> tuple[str, object, str | None]:
    """Identifiant d'une image parmi des images deja enregistrees.

    `images` : lignes (id, path, width, height), d'une base ou de plusieurs.
    `reserved` : identifiants a ne jamais donner a une image nouvelle.
    Rend (identifiant, ligne reprise ou None, remarque a afficher) :

    1. le chemin est deja enregistre : on reprend son identifiant ;
    2. le fichier a ete deplace : une seule image enregistree a disparu de
       son emplacement et a exactement ces dimensions, on la reprend ;
    3. sinon un identifiant tire du nom, rendu unique par un suffixe.

    Le nom seul ne suffit jamais : deux mosaiques « RECALAGE.tif » de deux
    echantillons auraient partage leurs traces.
    """
    resolved = Path(image_path).resolve()
    known = next((row for row in images if _same_path(row["path"], resolved)), None)
    if known is not None:
        return known["id"], known, None
    moved = [
        row for row in images
        if (row["width"], row["height"]) == (width, height)
        and not Path(row["path"]).exists()
    ]
    if len(moved) == 1:
        return moved[0]["id"], moved[0], (
            f"Image deplacee : reprise des annotations de "
            f"« {moved[0]['id']} », auparavant a {moved[0]['path']}"
        )
    base = _slug(Path(image_path).stem)
    taken = {row["id"] for row in images} | set(reserved)
    candidate, suffix = base, 2
    while candidate in taken:
        candidate, suffix = f"{base}-{suffix}", suffix + 1
    notes = []
    if len(moved) > 1:
        notes.append(
            f"{len(moved)} images disparues ont ces dimensions "
            f"({', '.join(row['id'] for row in moved)}) : aucune n'est "
            f"reprise d'office. Definir CRACK_IMAGE_ID pour en choisir une."
        )
    if candidate != base:
        notes.append(
            f"L'identifiant « {base} » appartient deja a une autre image ; "
            f"celle-ci recoit « {candidate} »."
        )
    return candidate, None, "\n".join(notes) or None


def _length(points: list[list[float]]) -> float:
    return sum(
        math.hypot(second[0] - first[0], second[1] - first[1])
        for first, second in zip(points, points[1:])
    )


# Un trace et sa suggestion de categorie. La suggestion n'est jointe que si
# elle porte sur la revision ACTUELLE du trace : deplacer un noeud la rend
# caduque, jusqu'a ce qu'elle soit recalculee.
_CRACK_SELECT = """
    SELECT c.id, c.zone_id, c.network_index, c.anchors_json, c.geometry_json,
           c.category, c.state, c.source, c.created_by, c.updated_by, c.revision,
           c.created_at, c.updated_at,
           s.category AS suggested_category, s.confidence AS suggested_confidence
    FROM cracks c
    LEFT JOIN suggestions s ON s.crack_id = c.id AND s.revision = c.revision
"""


def _crack_dict(row: sqlite3.Row) -> dict:
    result = dict(row)
    result["anchors"] = json.loads(result.pop("anchors_json"))
    result["geometry"] = json.loads(result.pop("geometry_json"))
    result["length_px"] = round(_length(result["geometry"]), 2)
    category = result.pop("suggested_category", None)
    confidence = result.pop("suggested_confidence", None)
    result["suggestion"] = (
        {"category": category, "confidence": round(float(confidence), 3)} if category else None
    )
    return result


class AnnotationDatabase:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def initialize_image(
        self,
        image_id: str,
        image_path: Path,
        width: int,
        height: int,
        zone_size: int,
    ) -> None:
        with closing(self.connect()) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS images (
                    id TEXT PRIMARY KEY,
                    path TEXT NOT NULL UNIQUE,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    zone_size INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS zones (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
                    row_index INTEGER NOT NULL,
                    col_index INTEGER NOT NULL,
                    x INTEGER NOT NULL,
                    y INTEGER NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'todo',
                    assigned_to TEXT,
                    revision INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    UNIQUE(image_id, row_index, col_index)
                );

                CREATE INDEX IF NOT EXISTS zones_image_status
                ON zones(image_id, status);

                CREATE TABLE IF NOT EXISTS cracks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
                    zone_id INTEGER NOT NULL REFERENCES zones(id) ON DELETE CASCADE,
                    network_index INTEGER NOT NULL DEFAULT 0,
                    anchors_json TEXT NOT NULL,
                    geometry_json TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'unclassified',
                    state TEXT NOT NULL DEFAULT 'proposed',
                    source TEXT NOT NULL DEFAULT 'auto',
                    created_by TEXT,
                    updated_by TEXT,
                    revision INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS cracks_zone_state
                ON cracks(zone_id, state);

                CREATE TABLE IF NOT EXISTS crack_revisions (
                    crack_id INTEGER NOT NULL REFERENCES cracks(id) ON DELETE CASCADE,
                    revision INTEGER NOT NULL,
                    anchors_json TEXT NOT NULL,
                    geometry_json TEXT NOT NULL,
                    category TEXT NOT NULL,
                    state TEXT NOT NULL,
                    changed_by TEXT,
                    changed_at TEXT NOT NULL,
                    PRIMARY KEY(crack_id, revision)
                );

                CREATE TABLE IF NOT EXISTS suggestions (
                    crack_id INTEGER PRIMARY KEY REFERENCES cracks(id) ON DELETE CASCADE,
                    revision INTEGER NOT NULL,
                    category TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    probabilities_json TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )
            resolved = str(image_path.resolve())
            # Le chemin est unique en base. Une image deja enregistree sous un
            # autre identifiant ferait echouer l'insertion sur une contrainte
            # illisible ; on explique plutot quoi faire, car le cas arrive des
            # qu'un `CRACK_IMAGE_ID` traine dans l'environnement.
            images = connection.execute(
                "SELECT id, path, width, height FROM images"
            ).fetchall()
            owner = next(
                (row for row in images
                 if _same_path(row["path"], resolved) and row["id"] != image_id),
                None,
            )
            if owner is not None:
                raise ValueError(already_registered_message(owner["id"], image_id, resolved))
            # L'inverse est plus grave : un identifiant deja pris par une AUTRE
            # image. L'accepter rattachait silencieusement ses traces a la
            # nouvelle image, et retirait l'ancienne de la base. Seul cas
            # admis : le meme fichier deplace, reconnu a ce que son ancien
            # emplacement a disparu et que ses dimensions sont identiques.
            current = next((row for row in images if row["id"] == image_id), None)
            if current is not None and not _same_path(current["path"], resolved):
                if Path(current["path"]).exists():
                    raise ValueError(
                        f"L'identifiant « {image_id} » appartient deja a une "
                        f"autre image :\n  {current['path']}\n"
                        f"Le reprendre melangerait les traces des deux images. "
                        f"Relancer sans CRACK_IMAGE_ID pour obtenir un "
                        f"identifiant libre."
                    )
                if (current["width"], current["height"]) != (width, height):
                    raise ValueError(
                        f"L'image enregistree sous « {image_id} » "
                        f"({current['width']} x {current['height']} px) a "
                        f"disparu de\n  {current['path']}\nmais celle-ci mesure "
                        f"{width} x {height} px : ce n'est pas le meme fichier."
                    )
            connection.execute(
                """
                INSERT INTO images(id, path, width, height, zone_size, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    path=excluded.path,
                    width=excluded.width,
                    height=excluded.height,
                    zone_size=excluded.zone_size
                """,
                (image_id, resolved, width, height, zone_size, _now()),
            )
            for row_index, y in enumerate(range(0, height, zone_size)):
                for col_index, x in enumerate(range(0, width, zone_size)):
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO zones(
                            image_id, row_index, col_index, x, y, width, height,
                            status, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'todo', ?)
                        """,
                        (
                            image_id,
                            row_index,
                            col_index,
                            x,
                            y,
                            min(zone_size, width - x),
                            min(zone_size, height - y),
                            _now(),
                        ),
                    )

    def image_id_for_path(self, image_path: Path) -> str | None:
        """Identifiant déjà enregistré pour ce fichier, s'il y en a un.

        Permet d'ouvrir une mosaïque en la désignant simplement par son
        chemin : on retrouve ses annotations sans avoir à se souvenir de son
        identifiant, et sans risquer de les rattacher à une autre image.
        """
        resolved = Path(image_path).resolve()
        return next(
            (row["id"] for row in self._registered_images()
             if _same_path(row["path"], resolved)),
            None,
        )

    def image_zone_size(self, image_id: str) -> int | None:
        """Cote des zones d'une image deja enregistree, ou None."""
        with closing(self.connect()) as connection:
            try:
                row = connection.execute(
                    "SELECT zone_size FROM images WHERE id = ?", (image_id,)
                ).fetchone()
            except sqlite3.OperationalError:
                return None
        return int(row["zone_size"]) if row else None

    def _registered_images(self) -> list[sqlite3.Row]:
        with closing(self.connect()) as connection:
            try:
                return connection.execute(
                    "SELECT id, path, width, height FROM images ORDER BY id"
                ).fetchall()
            except sqlite3.OperationalError:
                return []        # base encore vide : les tables n'existent pas

    def resolve_image_id(
        self, image_path: Path, width: int, height: int
    ) -> tuple[str, str | None]:
        """Identifiant a utiliser pour cette image, et une remarque a afficher.

        Les regles sont celles de `resolve_among`, appliquees aux images de
        cette base.
        """
        image_id, _, note = resolve_among(self._registered_images(), image_path, width, height)
        return image_id, note

    def list_zones(self, image_id: str) -> list[dict]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                """
                SELECT id, row_index, col_index, x, y, width, height, status,
                       assigned_to, revision, updated_at
                FROM zones WHERE image_id = ? ORDER BY row_index, col_index
                """,
                (image_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_zone(self, image_id: str, zone_id: int) -> dict:
        with closing(self.connect()) as connection:
            row = connection.execute(
                """
                SELECT id, row_index, col_index, x, y, width, height, status,
                       assigned_to, revision, updated_at
                FROM zones WHERE id = ? AND image_id = ?
                """,
                (zone_id, image_id),
            ).fetchone()
        if row is None:
            raise KeyError(zone_id)
        return dict(row)

    def list_cracks(
        self,
        image_id: str,
        zone_id: int,
        include_rejected: bool = False,
    ) -> list[dict]:
        where_rejected = "" if include_rejected else "AND c.state != 'rejected'"
        with closing(self.connect()) as connection:
            rows = connection.execute(
                f"""{_CRACK_SELECT}
                WHERE c.image_id = ? AND c.zone_id = ? {where_rejected}
                ORDER BY c.state = 'rejected', c.network_index, c.id
                """,
                (image_id, zone_id),
            ).fetchall()
        return [_crack_dict(row) for row in rows]

    def list_image_cracks(
        self,
        image_id: str,
        include_rejected: bool = False,
    ) -> list[dict]:
        """Retourne les tracés de toutes les zones pour la vue globale."""
        where_rejected = "" if include_rejected else "AND c.state != 'rejected'"
        with closing(self.connect()) as connection:
            rows = connection.execute(
                f"""{_CRACK_SELECT}
                WHERE c.image_id = ? {where_rejected}
                ORDER BY c.zone_id, c.network_index, c.id
                """,
                (image_id,),
            ).fetchall()
        return [_crack_dict(row) for row in rows]

    def length_statistics(self, image_id: str, zone_id: int | None = None) -> dict:
        """Somme les polylignes actives, globalement ou pour une seule zone."""
        parameters: list[object] = [image_id]
        zone_filter = ""
        if zone_id is not None:
            zone_filter = "AND zone_id = ?"
            parameters.append(zone_id)
        with closing(self.connect()) as connection:
            rows = connection.execute(
                f"""
                SELECT category, geometry_json
                FROM cracks
                WHERE image_id = ? AND state != 'rejected' {zone_filter}
                """,
                parameters,
            ).fetchall()

        lengths = {category: 0.0 for category in CRACK_CATEGORIES}
        counts = {category: 0 for category in CRACK_CATEGORIES}
        for row in rows:
            category = row["category"]
            if category not in lengths:
                category = "unclassified"
            lengths[category] += _length(json.loads(row["geometry_json"]))
            counts[category] += 1

        lengths = {key: round(value, 2) for key, value in lengths.items()}
        lengths["total"] = round(sum(lengths.values()), 2)
        counts["total"] = sum(counts.values())
        return {"length_px": lengths, "crack_count": counts}

    def get_crack(self, image_id: str, crack_id: int) -> dict:
        with closing(self.connect()) as connection:
            row = connection.execute(
                f"{_CRACK_SELECT} WHERE c.id = ? AND c.image_id = ?",
                (crack_id, image_id),
            ).fetchone()
        if row is None:
            raise KeyError(crack_id)
        return _crack_dict(row)

    # -- Suggestions de categorie -----------------------------------------

    def set_suggestion(self, crack_id: int, revision: int, category: str,
                       confidence: float, probabilities: dict | None = None) -> None:
        with closing(self.connect()) as connection, connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO suggestions(
                    crack_id, revision, category, confidence, probabilities_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (crack_id, revision, category, float(confidence),
                 json.dumps(probabilities) if probabilities else None, _now()),
            )

    def labelled_cracks(self, image_id: str, categories) -> list[dict]:
        """Traces valides des categories donnees : les etiquettes d'apprentissage."""
        placeholders = ",".join("?" * len(categories))
        with closing(self.connect()) as connection:
            rows = connection.execute(
                f"""{_CRACK_SELECT}
                WHERE c.image_id = ? AND c.state = 'validated' AND c.category IN ({placeholders})
                ORDER BY c.id
                """,
                (image_id, *categories),
            ).fetchall()
        return [_crack_dict(row) for row in rows]

    def review_queue(self, image_id: str) -> list[dict]:
        """Propositions a verifier, avec une suggestion a jour, tuile par tuile.

        L'ordre suit la grille (ligne, colonne) : on verifie une tuile entiere
        avant de passer a la voisine, au lieu de sauter d'un bout a l'autre.
        """
        with closing(self.connect()) as connection:
            rows = connection.execute(
                f"""{_CRACK_SELECT}
                JOIN zones z ON z.id = c.zone_id
                WHERE c.image_id = ? AND c.state = 'proposed' AND s.category IS NOT NULL
                ORDER BY z.row_index, z.col_index, c.id
                """,
                (image_id,),
            ).fetchall()
        return [_crack_dict(row) for row in rows]

    @staticmethod
    def _insert_crack(
        connection: sqlite3.Connection,
        image_id: str,
        zone_id: int,
        anchors: list[list[float]],
        geometry: list[list[float]],
        network_index: int,
        category: str,
        state: str,
        source: str,
        annotator: str | None,
    ) -> int:
        timestamp = _now()
        anchors_json = json.dumps(anchors, separators=(",", ":"))
        geometry_json = json.dumps(geometry, separators=(",", ":"))
        cursor = connection.execute(
            """
            INSERT INTO cracks(
                image_id, zone_id, network_index, anchors_json, geometry_json,
                category, state, source, created_by, updated_by, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                image_id, zone_id, network_index, anchors_json, geometry_json,
                category, state, source, annotator, annotator, timestamp, timestamp,
            ),
        )
        crack_id = int(cursor.lastrowid)
        connection.execute(
            """
            INSERT INTO crack_revisions(
                crack_id, revision, anchors_json, geometry_json, category, state,
                changed_by, changed_at
            ) VALUES (?, 0, ?, ?, ?, ?, ?, ?)
            """,
            (crack_id, anchors_json, geometry_json, category, state, annotator, timestamp),
        )
        return crack_id

    def add_detected_cracks_if_empty(
        self,
        image_id: str,
        zone_id: int,
        cracks: list[dict],
        annotator: str | None,
    ) -> tuple[list[dict], bool]:
        """Ajoute une proposition atomiquement, sans ecraser un travail existant."""
        with closing(self.connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT 1 FROM cracks
                WHERE image_id = ? AND zone_id = ? AND state != 'rejected'
                LIMIT 1
                """,
                (image_id, zone_id),
            ).fetchone()
            if existing is not None:
                return self.list_cracks(image_id, zone_id), False
            for crack in cracks:
                self._insert_crack(
                    connection,
                    image_id,
                    zone_id,
                    crack["anchors"],
                    crack["geometry"],
                    int(crack.get("network_index", 0)),
                    "unclassified",
                    "proposed",
                    "auto",
                    annotator,
                )
        return self.list_cracks(image_id, zone_id), True

    def create_crack(
        self,
        image_id: str,
        zone_id: int,
        anchors: list[list[float]],
        geometry: list[list[float]],
        annotator: str | None,
    ) -> dict:
        with closing(self.connect()) as connection, connection:
            crack_id = self._insert_crack(
                connection,
                image_id,
                zone_id,
                anchors,
                geometry,
                0,
                "unclassified",
                "proposed",
                "manual",
                annotator,
            )
        return self.get_crack(image_id, crack_id)

    def update_crack(
        self,
        image_id: str,
        crack_id: int,
        expected_revision: int,
        anchors: list[list[float]],
        geometry: list[list[float]],
        category: str,
        state: str,
        annotator: str | None,
    ) -> dict:
        if category not in CRACK_CATEGORIES:
            raise ValueError(f"Categorie inconnue : {category}")
        if state not in CRACK_STATES:
            raise ValueError(f"Etat inconnu : {state}")
        with closing(self.connect()) as connection, connection:
            self._update_crack_row(connection, image_id, crack_id, expected_revision,
                                   anchors, geometry, category, state, annotator)
        return self.get_crack(image_id, crack_id)

    def replace_proposals(self, image_id: str, zone_ids, cracks: list[dict],
                          annotator: str | None) -> tuple[int, int]:
        """Remplace les propositions jamais verifiees (detecteur, analyse) des tuiles donnees.

        Une seule transaction : les anciennes propositions automatiques de ces
        tuiles disparaissent, les nouvelles (`cracks` : zone_id, anchors,
        geometry, category) arrivent en propositions de source « analyse ».
        Le reste (fissures validees, traces a la main) n'est pas touche.
        Rend (retirees, ajoutees).
        """
        removed = 0
        with closing(self.connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            for zone_id in zone_ids:
                cursor = connection.execute(
                    "DELETE FROM cracks WHERE image_id = ? AND zone_id = ? AND state = 'proposed' "
                    "AND source IN ('auto', 'analyse')", (image_id, zone_id))
                removed += cursor.rowcount
            for crack in cracks:
                self._insert_crack(connection, image_id, crack["zone_id"], crack["anchors"], crack["geometry"],
                                   0, crack["category"], "proposed", "analyse", annotator)
        return removed, len(cracks)

    def confirm_zone_cracks(self, image_id: str, zone_id: int, annotator: str | None,
                            state_from: str = "proposed", state_to: str = "validated",
                            only_ids=None) -> list[int]:
        """Passe les fissures d'une tuile d'un etat a l'autre (valider la tuile, ou l'annuler)."""
        changed = []
        with closing(self.connect()) as connection, connection:
            rows = connection.execute(
                "SELECT id, revision, anchors_json, geometry_json, category FROM cracks "
                "WHERE image_id = ? AND zone_id = ? AND state = ?", (image_id, zone_id, state_from)).fetchall()
            for row in rows:
                if only_ids is not None and row["id"] not in only_ids:
                    continue
                self._update_crack_row(connection, image_id, row["id"], row["revision"],
                                       json.loads(row["anchors_json"]), json.loads(row["geometry_json"]),
                                       row["category"], state_to, annotator)
                changed.append(int(row["id"]))
        return changed

    def split_crack(
        self,
        image_id: str,
        crack_id: int,
        expected_revision: int,
        pieces: list[tuple[list, list]],
        annotator: str | None,
    ) -> list[dict]:
        """Coupe un trace en morceaux (noeuds, geometrie), en une seule transaction.

        Le premier morceau garde l'identifiant et l'historique du trace ; les
        suivants sont de nouveaux traces de meme categorie, etat et origine.
        """
        if len(pieces) < 2:
            raise ValueError("Il faut au moins deux morceaux.")
        current = self.get_crack(image_id, crack_id)
        with closing(self.connect()) as connection, connection:
            anchors, geometry = pieces[0]
            self._update_crack_row(connection, image_id, crack_id, expected_revision, anchors, geometry,
                                   current["category"], current["state"], annotator)
            ids = [crack_id]
            for anchors, geometry in pieces[1:]:
                ids.append(self._insert_crack(
                    connection, image_id, current["zone_id"], anchors, geometry,
                    int(current.get("network_index") or 0), current["category"], current["state"],
                    current["source"], annotator))
        return [self.get_crack(image_id, piece) for piece in ids]

    @staticmethod
    def _update_crack_row(connection, image_id, crack_id, expected_revision,
                          anchors, geometry, category, state, annotator) -> None:
        anchors_json = json.dumps(anchors, separators=(",", ":"))
        geometry_json = json.dumps(geometry, separators=(",", ":"))
        timestamp = _now()
        cursor = connection.execute(
            """
            UPDATE cracks
            SET anchors_json = ?, geometry_json = ?, category = ?, state = ?,
                updated_by = ?, revision = revision + 1, updated_at = ?
            WHERE id = ? AND image_id = ? AND revision = ?
            """,
            (
                anchors_json, geometry_json, category, state, annotator,
                timestamp, crack_id, image_id, expected_revision,
            ),
        )
        if cursor.rowcount != 1:
            exists = connection.execute(
                "SELECT 1 FROM cracks WHERE id = ? AND image_id = ?",
                (crack_id, image_id),
            ).fetchone()
            if exists:
                raise CrackConflict(
                    "La fissure a ete modifiee dans une autre fenetre."
                )
            raise KeyError(crack_id)
        connection.execute(
            """
            INSERT INTO crack_revisions(
                crack_id, revision, anchors_json, geometry_json, category,
                state, changed_by, changed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                crack_id, expected_revision + 1, anchors_json, geometry_json, category,
                state, annotator, timestamp,
            ),
        )

    def update_zone(
        self,
        image_id: str,
        zone_id: int,
        status: str,
        assigned_to: str | None,
        expected_revision: int,
    ) -> dict:
        if status not in ZONE_STATUSES:
            raise ValueError(f"Statut inconnu : {status}")
        with closing(self.connect()) as connection, connection:
            cursor = connection.execute(
                """
                UPDATE zones
                SET status = ?, assigned_to = ?, revision = revision + 1,
                    updated_at = ?
                WHERE id = ? AND image_id = ? AND revision = ?
                """,
                (
                    status,
                    assigned_to or None,
                    _now(),
                    zone_id,
                    image_id,
                    expected_revision,
                ),
            )
            if cursor.rowcount != 1:
                exists = connection.execute(
                    "SELECT 1 FROM zones WHERE id = ? AND image_id = ?",
                    (zone_id, image_id),
                ).fetchone()
                if exists:
                    raise ZoneConflict("La zone a ete modifiee dans une autre fenetre.")
                raise KeyError(zone_id)
            row = connection.execute(
                """
                SELECT id, row_index, col_index, x, y, width, height, status,
                       assigned_to, revision, updated_at
                FROM zones WHERE id = ?
                """,
                (zone_id,),
            ).fetchone()
        return dict(row)
