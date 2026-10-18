"""Annotations de la base, en lecture seule, et decoupage apprentissage / test.

Le decoupage se fait par LIGNES ENTIERES de la grille : deux tuiles voisines se
ressemblent, et une fissure en traverse souvent plusieurs. Melanger tuiles
d'apprentissage et de test dans une meme ligne surestimerait les resultats. Le
decoupage est enregistre a la premiere utilisation puis toujours relu : une
tuile validee plus tard dans une ligne de test rejoint le test.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from math import hypot
from pathlib import Path

from annotation_app.paths import analysis_root

CATEGORIES = ("open", "filled", "reopened")


def connect_readonly(database: Path) -> sqlite3.Connection:
    """Connexion qui ne peut rien ecrire, meme par erreur."""
    connection = sqlite3.connect(f"file:{Path(database).resolve().as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def polyline_length(points) -> float:
    return sum(hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))


@dataclass
class Annotations:
    image_id: str
    image_path: Path
    zone_size: int
    zones: list[dict]                     # toutes les tuiles, avec statut, ligne, colonne
    validated: list[dict]                 # fissures classees par l'annotateur
    unverified: list[dict] = field(default_factory=list)   # propositions jamais verifiees
    rejected: list[dict] = field(default_factory=list)

    @property
    def done_ids(self) -> set[int]:
        return {z["id"] for z in self.zones if z["status"] == "done"}

    def zone(self, zone_id: int) -> dict:
        return next(z for z in self.zones if z["id"] == zone_id)

    def output_dir(self) -> Path:
        return analysis_root() / self.image_id


def load(database: Path, image_path: Path | None = None) -> Annotations:
    """Annotations d'une mosaique : celle de `image_path`, ou la seule de la base."""
    connection = connect_readonly(database)
    images = [dict(r) for r in connection.execute("SELECT * FROM images")]
    if image_path is not None:
        wanted = Path(image_path).resolve()
        images = [i for i in images if Path(i["path"]).resolve() == wanted] or images
    if len(images) != 1:
        raise ValueError(f"{len(images)} mosaiques possibles dans la base : preciser l'image.")
    image = images[0]
    zones = [dict(r) for r in connection.execute(
        "SELECT * FROM zones WHERE image_id=? ORDER BY row_index, col_index", (image["id"],))]
    groups = {"validated": [], "unverified": [], "rejected": []}
    for row in connection.execute(
            "SELECT id, zone_id, category, state, source, geometry_json FROM cracks WHERE image_id=?",
            (image["id"],)):
        crack = {"id": row["id"], "zone_id": row["zone_id"], "category": row["category"],
                 "source": row["source"], "geometry": json.loads(row["geometry_json"])}
        if len(crack["geometry"]) < 2:
            continue
        if row["state"] == "validated":
            groups["validated"].append(crack)
        elif row["state"] == "proposed":
            groups["unverified"].append(crack)
        else:
            groups["rejected"].append(crack)
    connection.close()
    return Annotations(image["id"], Path(image["path"]), int(image["zone_size"]), zones, **groups)


# -- Decoupage ---------------------------------------------------------------------

def _spread(rows: list[int], count: int, avoid: set[int] = frozenset()) -> list[int]:
    """`count` lignes regulierement reparties dans `rows`, hors `avoid`."""
    candidates = [r for r in rows if r not in avoid]
    if count <= 0 or not candidates:
        return []
    picks = []
    for k in range(count):
        index = min(len(candidates) - 1, int((k + 0.5) * len(candidates) / count))
        if candidates[index] not in picks:
            picks.append(candidates[index])
    return picks


def choose_split(annotations: Annotations, test_share: float = 0.25,
                 validation_share: float = 0.10) -> dict:
    """Lignes de test et de validation, reparties du haut au bas de l'echantillon."""
    done = [z for z in annotations.zones if z["status"] == "done"]
    per_row = {}
    for zone in done:
        per_row[zone["row_index"]] = per_row.get(zone["row_index"], 0) + 1
    rows = sorted(per_row)

    def pick(share, avoid):
        target = share * len(done)
        for count in range(1, len(rows) + 1):
            chosen = _spread(rows, count, avoid)
            if sum(per_row[r] for r in chosen) >= target:
                return chosen
        return _spread(rows, len(rows), avoid)

    test_rows = sorted(pick(test_share, set()))
    validation_rows = sorted(pick(validation_share, set(test_rows)))
    return {"image_id": annotations.image_id, "cree_le": datetime.now().isoformat(timespec="seconds"),
            "lignes_test": test_rows, "lignes_validation": validation_rows,
            "regle": "lignes entieres de la grille ; test jamais appris, validation pour les reglages"}


def split_for(annotations: Annotations, path: Path | None = None) -> dict:
    """Le decoupage enregistre de cette mosaique, cree a la premiere demande."""
    path = path or annotations.output_dir() / "decoupage.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    split = choose_split(annotations)
    if not annotations.done_ids:
        return split            # mosaique sans tuile validee : rien a figer pour l'instant
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8")
    return split


def role(zone: dict, split: dict) -> str:
    """« test », « validation » ou « apprentissage » selon la ligne de la tuile."""
    if zone["row_index"] in split["lignes_test"]:
        return "test"
    if zone["row_index"] in split["lignes_validation"]:
        return "validation"
    return "apprentissage"


def zones_by_role(annotations: Annotations, split: dict, done_only: bool = True) -> dict[str, list[dict]]:
    result = {"apprentissage": [], "validation": [], "test": []}
    for zone in annotations.zones:
        if done_only and zone["status"] != "done":
            continue
        result[role(zone, split)].append(zone)
    return result


# -- Fissures rechainees -----------------------------------------------------------

def chains(cracks: list[dict], tolerance: float = 0.6) -> list[dict]:
    """Rechaine les morceaux d'une meme fissure (coupes automatiques ou a la main).

    Deux morceaux se suivent quand une extremite de l'un est celle de l'autre,
    et d'eux seuls : a un carrefour (trois morceaux ou plus), la chaine
    s'arrete. Rend [{"geometry", "segments": [(debut, fin, categorie, id)], "zone_id"}],
    abscisses en pixels le long de la geometrie.
    """
    def key(point):
        return (round(point[0] / tolerance), round(point[1] / tolerance))

    ends: dict[tuple, list[int]] = {}
    for index, crack in enumerate(cracks):
        for point in (crack["geometry"][0], crack["geometry"][-1]):
            ends.setdefault(key(point), []).append(index)

    def partner(index, point):
        members = [m for m in ends[key(point)] if m != index]
        return members[0] if len(ends[key(point)]) == 2 and members else None

    used, result = set(), []
    order = sorted(range(len(cracks)), key=lambda i: (
        partner(i, cracks[i]["geometry"][0]) is not None and partner(i, cracks[i]["geometry"][-1]) is not None))
    for start in order:
        if start in used:
            continue
        # On part d'une extremite libre si le morceau en a une.
        geometry = list(cracks[start]["geometry"])
        if partner(start, geometry[0]) is not None and partner(start, geometry[-1]) is None:
            geometry.reverse()
        pieces = [(start, geometry)]
        used.add(start)
        current, tail = start, geometry[-1]
        while True:
            nxt = partner(current, tail)
            if nxt is None or nxt in used:
                break
            g = list(cracks[nxt]["geometry"])
            if key(g[0]) != key(tail):
                g.reverse()
            pieces.append((nxt, g))
            used.add(nxt)
            current, tail = nxt, g[-1]
        whole, segments, position = [], [], 0.0
        for index, g in pieces:
            piece_length = polyline_length(g)
            segments.append((position, position + piece_length, cracks[index]["category"], cracks[index]["id"]))
            position += piece_length
            whole.extend(g if not whole else g[1:])
        result.append({"geometry": whole, "segments": segments, "zone_id": cracks[start]["zone_id"]})
    return result
