"""Une mosaique ouverte : base, image, detecteur, automatisation.

Toute la logique de l'application, sans interface : la fenetre PySide6 l'appelle
directement, et les tests aussi. Ouvrir une session :

1. trouve la base de la mosaique, `data/bases/<image_id>.sqlite3` (meme
   deplacee, jamais confondue avec une autre de meme nom ; voir bases.py),
   apres avoir separe l'ancienne base unique si elle est encore la. Une base
   donnee explicitement (`database_path`, `CRACK_DATABASE_PATH`) est « forcee » :
   elle sert telle quelle et peut contenir plusieurs mosaiques ;
2. sauvegarde la base AVANT d'y ecrire quoi que ce soit, puis toutes les 4 h ;
3. cree la grille de tuiles de 3 mm ;
4. charge le detecteur a l'echelle de l'image et le modele de colmatage.

A la fermeture, le -wal est reporte dans la base : le fichier se suffit alors
a lui-meme et peut etre donne tel quel.

Les methodes peuvent etre appelees depuis des fils de fond : chaque acces a la
base ouvre sa propre connexion, le detecteur a ses propres verrous.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from pathlib import Path

from . import bases
from .automatisation import Automation
from .backup import backup_database, start_periodic_backup
from .database import AnnotationDatabase, CRACK_CATEGORIES, CRACK_STATES, ZONE_STATUSES
from .echelle import make_detector, pixel_size_um, resolution_factor, zone_size
from .paths import DATA_DIR, LAST_IMAGE_FILE, remember_image
from .tile_service import PyramidTileService


class SessionError(RuntimeError):
    """Erreur a montrer telle quelle a l'annotateur."""


class Session:
    def __init__(self, image_path: Path, database_path: Path | None = None,
                 image_id: str | None = None, log=print, data_dir: Path | None = None) -> None:
        """`data_dir` : l'etat du poste (bases, sauvegardes, cache), `data/` par defaut."""
        self.log = log
        self.data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
        self.image_path = Path(image_path).resolve()
        if not self.image_path.is_file():
            raise SessionError(f"Mosaique introuvable : {self.image_path}")
        self.cache_dir = Path(os.environ.get("CRACK_CACHE_DIR") or self.data_dir / "analysis_cache").resolve()
        self.tiles = PyramidTileService(self.image_path)
        self.pixel_um = pixel_size_um(self.image_path)
        self.factor = resolution_factor(self.pixel_um)

        self.image_id = image_id or os.environ.get("CRACK_IMAGE_ID")
        forced = database_path or os.environ.get("CRACK_DATABASE_PATH")
        # Base forcee : servie telle quelle, comme avant les bases par mosaique.
        self.database_forced = bool(forced)
        if forced:
            self.database_path = Path(forced).resolve()
        else:
            try:
                bases.migrate(self.data_dir, log)
                path, self.image_id = bases.database_for(self.image_path, self.tiles.width, self.tiles.height,
                                                         self.image_id, self.data_dir, log)
            except (bases.MigrationError, ValueError) as error:
                raise SessionError(str(error)) from error
            self.database_path = path.resolve()
        backups = bases.backup_dir_for(self.database_path, self.data_dir)
        self.backup_dir = backups

        # Avant toute ecriture dans la base, y compris l'enregistrement de l'image.
        try:
            saved = backup_database(self.database_path, backups)
            if saved:
                log(f"Sauvegarde : {saved}")
        except Exception as error:
            log(f"ATTENTION, sauvegarde impossible dans {backups} : {error}")
        self._backup_thread = start_periodic_backup(self.database_path, backups)

        self.database = AnnotationDatabase(self.database_path)
        if not self.image_id:
            self.image_id, note = self.database.resolve_image_id(
                self.image_path, self.tiles.width, self.tiles.height)
            if note:
                log(note)
        # Tuiles de meme surface physique quelle que soit la resolution ; une
        # image deja enregistree garde sa grille.
        self.zone_size = self.database.image_zone_size(self.image_id) or zone_size(self.factor)
        try:
            self.database.initialize_image(self.image_id, self.image_path, self.tiles.width,
                                           self.tiles.height, self.zone_size)
        except ValueError as error:
            raise SessionError(str(error)) from error
        if not self.database_forced:
            remember_image(self.image_path, self.data_dir / LAST_IMAGE_FILE.name)
            # Fichiers recalculables qui ne seront plus relus (voir menage.py).
            from .detector_model import default_model_path
            from .menage import tidy
            tidy(self.data_dir, self.cache_dir,
                 Path(os.environ.get("CRACK_ANALYSE_DIR") or self.data_dir / "analyse"), default_model_path(), log)

        self.detector, _ = make_detector(self.tiles, zone_size=self.zone_size)
        self._layers = None
        self.automation = Automation(self.database, self.image_id, self.tiles.source, self.pixel_um,
                                     self._proposals_for,
                                     model_path=os.environ.get("CRACK_COLMATAGE_MODEL") or None)

    # -- Description -----------------------------------------------------------

    @property
    def width(self) -> int:
        return self.tiles.width

    @property
    def height(self) -> int:
        return self.tiles.height

    def describe(self) -> str:
        scale = f"pixel {self.pixel_um:.3g} um" if self.pixel_um else "pixel inconnu"
        return (f"{self.image_path.name} · {self.width:,} × {self.height:,} px · {scale} · "
                f"tuiles de {self.zone_size} px").replace(",", " ")

    def close(self) -> None:
        self.automation.cancel()
        if self._layers is not None:
            self._layers.cancel.set()           # calcul des calques : s'arrete apres la tuile en cours
        stop = getattr(self._backup_thread, "stop", None)
        if stop is not None:
            stop.set()
        # Le -wal reporte dans la base : le fichier se suffit a lui-meme.
        try:
            with closing(sqlite3.connect(self.database_path, timeout=10)) as connection:
                connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error as error:
            self.log(f"Base pas entierement resynchronisee ({error}) : elle le sera a la prochaine ouverture.")

    # -- Tuiles ------------------------------------------------------------------

    def zones(self) -> list[dict]:
        return self.database.list_zones(self.image_id)

    def zone(self, zone_id: int) -> dict:
        return self.database.get_zone(self.image_id, zone_id)

    def zone_at(self, x: float, y: float) -> dict | None:
        size = self.zone_size
        row, col = int(y // size), int(x // size)
        for zone in self.zones():
            if zone["row_index"] == row and zone["col_index"] == col:
                return zone
        return None

    def set_zone_status(self, zone_id: int, status: str, annotator: str | None) -> dict:
        if status not in ZONE_STATUSES:
            raise SessionError(f"Statut inconnu : {status}")
        zone = self.zone(zone_id)
        return self.database.update_zone(self.image_id, zone_id, status, annotator, zone["revision"])

    def confirm_zone(self, zone_id: int, annotator: str | None) -> list[int]:
        """Valider une tuile confirme aussi ses propositions restantes, telles quelles."""
        return self.database.confirm_zone_cracks(self.image_id, zone_id, annotator)

    def change_zone_cracks(self, zone_id: int, mode: str, annotator: str | None) -> list[dict]:
        """Toutes les fissures d'une tuile d'un coup.

        « reject » : supprimees. « accept » : classement accepte et valide — la
        suggestion pour une fissure non classee, sa categorie sinon ; une fissure
        non classee sans suggestion reste telle quelle. Rend l'etat d'avant de
        chaque fissure changee, pour annuler (`restore_cracks`).
        """
        if mode not in ("reject", "accept"):
            raise SessionError(f"Action inconnue : {mode}")
        changes, previous = [], []
        for crack in self.cracks(zone_id):
            category, state = crack["category"], "rejected"
            if mode == "accept":
                if category == "unclassified" and crack.get("suggestion"):
                    category = crack["suggestion"]["category"]
                state = "validated"
                if category == "unclassified" or (category, state) == (crack["category"], crack["state"]):
                    continue
            changes.append((crack["id"], crack["revision"], category, state))
            previous.append({"id": crack["id"], "category": crack["category"], "state": crack["state"]})
        self.database.set_cracks(self.image_id, changes, annotator)
        return previous

    def restore_cracks(self, previous: list[dict], annotator: str | None) -> None:
        """Remet categorie et etat de plusieurs fissures, d'une seule ecriture (annulation)."""
        changes = []
        for item in previous:
            current = self.crack(item["id"])
            changes.append((item["id"], current["revision"], item["category"], item["state"]))
        self.database.set_cracks(self.image_id, changes, annotator)

    def unconfirm_zone(self, zone_id: int, crack_ids, annotator: str | None) -> list[int]:
        return self.database.confirm_zone_cracks(self.image_id, zone_id, annotator, "validated", "proposed",
                                                 set(crack_ids))

    def import_analysis(self, result: dict, log=None) -> dict:
        """Le resultat de l'analyse complete devient des propositions modifiables (voir analyse/importation.py)."""
        from analyse.importation import import_into
        return import_into(self.database_path, self.image_id, self.zone_size, self.pixel_um, result,
                           self.backup_dir, log or self.log)

    def prepare(self, zone_id: int) -> None:
        """Prepare le routage d'une tuile en avance (le premier trace est alors immediat)."""
        self.detector.prepare_route(self.zone(zone_id))

    # -- Traces ------------------------------------------------------------------

    def cracks(self, zone_id: int | None = None, include_rejected: bool = False) -> list[dict]:
        if zone_id is None:
            return self.database.list_image_cracks(self.image_id, include_rejected)
        return self.database.list_cracks(self.image_id, zone_id, include_rejected)

    def crack(self, crack_id: int) -> dict:
        return self.database.get_crack(self.image_id, crack_id)

    def _clean_anchors(self, zone: dict, anchors) -> list[list[float]]:
        result = []
        for x, y in anchors:
            if not (zone["x"] <= x < zone["x"] + zone["width"] and zone["y"] <= y < zone["y"] + zone["height"]):
                raise SessionError("Les points d'une fissure doivent rester dans la meme tuile.")
            result.append([round(float(x), 2), round(float(y), 2)])
        return result

    def create_crack(self, zone_id: int, anchors, annotator: str | None) -> dict:
        zone = self.zone(zone_id)
        anchors = self._clean_anchors(zone, anchors)
        geometry = self.detector.route(zone, anchors)
        return self._with_suggestion(
            self.database.create_crack(self.image_id, zone_id, anchors, geometry, annotator))

    def update_crack(self, crack_id: int, expected_revision: int, anchors=None,
                     category: str | None = None, state: str | None = None,
                     annotator: str | None = None) -> dict:
        if category is not None and category not in CRACK_CATEGORIES:
            raise SessionError(f"Categorie inconnue : {category}")
        if state is not None and state not in CRACK_STATES:
            raise SessionError(f"Etat inconnu : {state}")
        current = self.crack(crack_id)
        zone = self.zone(current["zone_id"])
        if anchors is None:
            anchors, geometry = current["anchors"], current["geometry"]
        else:
            anchors = self._clean_anchors(zone, anchors)
            geometry = self.detector.route(zone, anchors)
        return self._with_suggestion(self.database.update_crack(
            self.image_id, crack_id, expected_revision, anchors, geometry,
            category or current["category"], state or current["state"], annotator))

    def _with_suggestion(self, crack: dict) -> dict:
        """Le trace, avec une suggestion de categorie a jour s'il en manque une."""
        # Une fissure deja classee (par l'analyse ou par vous) garde sa categorie.
        if crack.get("state") != "proposed" or crack.get("suggestion") or not self.automation.has_model \
                or crack.get("category") not in (None, "unclassified"):
            return crack
        try:
            if self.automation.suggest(crack):
                return self.crack(crack["id"])
        except Exception as error:  # une suggestion ne doit jamais empecher d'enregistrer
            self.log(f"Suggestion impossible pour la fissure {crack['id']} : {error}")
        return crack

    def detect(self, zone_id: int, annotator: str | None) -> tuple[list[dict], bool]:
        """Propositions du detecteur sur une tuile SANS travail ; rien n'est ecrase."""
        zone = self.zone(zone_id)
        if self.database.list_cracks(self.image_id, zone_id, include_rejected=False):
            return self.cracks(zone_id), False
        saved, created = self.database.add_detected_cracks_if_empty(
            self.image_id, zone_id, self._proposals_for(zone), annotator)
        if created:
            try:
                # Detecteur puis colmatage : chaque proposition est coupee en
                # portions de categorie homogene, chacune avec sa suggestion.
                self.automation.suggest_zone(zone_id)
            except Exception as error:  # les propositions restent, sans suggestion
                self.log(f"Suggestions impossibles : {error}")
        return self.cracks(zone_id), created

    def split_crack(self, crack_id: int, expected_revision: int, x: float, y: float,
                    annotator: str | None) -> list[dict]:
        """Coupe une fissure en deux au point le plus proche de (x, y)."""
        from .polyligne import cumulative, pieces, project
        crack = self.crack(crack_id)
        along, _, _ = project(crack["geometry"], x, y)
        total = cumulative(crack["geometry"])[-1]
        if along < 3 or along > total - 3:
            raise SessionError("Trop près d'une extrémité.")
        return self.database.split_crack(self.image_id, crack_id, expected_revision,
                                         pieces(crack["anchors"], crack["geometry"], [along]), annotator)

    def restore_crack(self, crack_id: int, anchors, geometry, category: str, state: str,
                      annotator: str | None) -> dict:
        """Remet un trace exactement comme il etait (annulation), sans recalculer son chemin."""
        current = self.crack(crack_id)
        return self.database.update_crack(self.image_id, crack_id, current["revision"], anchors,
                                          geometry, category, state, annotator)

    def _proposals_for(self, zone: dict) -> list[dict]:
        """Propositions du detecteur, reprises du cache des couches s'il y en a."""
        cached = self.layers.cached_proposals(zone)
        return cached[0] if cached is not None else self.detector.analyze(zone).cracks

    # -- Calques de diagnostic ---------------------------------------------------

    @property
    def layers(self):
        """Cache des analyses par tuile : score, masque, squelette, axes, couts."""
        if self._layers is None:
            from .analysis_service import AnalysisLayers
            self._layers = AnalysisLayers(self.detector, self.cache_dir)
        return self._layers

    def layer_computed(self, zone: dict) -> bool:
        return self.layers.path(zone).exists()

    def layer_array(self, zone: dict, kind: str):
        """Le calque d'une tuile a la resolution d'analyse, ou None s'il n'est pas calcule.

        Ne lance jamais de calcul : c'est le role de `layers.start()`.
        """
        import numpy as np
        path = self.layers.path(zone)
        if not path.exists():
            return None
        with np.load(path) as data:
            return data[kind]

    def export_layers(self, destination: Path, zones: list[dict]) -> int:
        """Archive ZIP des calques calcules, a la resolution d'analyse. Rend le nombre de tuiles."""
        import json
        import zipfile
        computed = [z for z in zones if self.layer_computed(z)]
        with zipfile.ZipFile(destination, "w") as archive:
            for zone in computed:
                archive.write(self.layers.path(zone), f"tuile-{zone['id']}.npz")
            archive.writestr("metadata.json", json.dumps({
                "image": str(self.image_path), "pixel_um": self.pixel_um,
                "coordonnees": "pixels de la mosaique d'origine, x vers la droite, y vers le bas ; "
                               "chaque tableau couvre la tuile indiquee.",
                "resolution": "celle de l'analyse, pas une largeur ni une aire physique de fissure.",
                "calques": sorted(self.layers.KINDS),
                "tuiles": [{k: z[k] for k in ("id", "x", "y", "width", "height")} for z in computed]},
                indent=2, ensure_ascii=False))
        return len(computed)

    # -- Analyse complete (python -m analyse analyser) -------------------------------

    @property
    def analysis_dir(self) -> Path:
        from .paths import analysis_root
        return analysis_root() / self.image_id

    def analysis_result(self) -> dict | None:
        """Le dernier resultat de l'analyse complete, relu seulement s'il a change."""
        import json
        path = self.analysis_dir / "resultat.json"
        if not path.exists():
            return None
        stamp = path.stat().st_mtime_ns
        cached = getattr(self, "_analysis_cache", None)
        if cached is None or cached[0] != stamp:
            self._analysis_cache = (stamp, json.loads(path.read_text(encoding="utf-8")))
        return self._analysis_cache[1]

    # -- Mesures -----------------------------------------------------------------

    def report(self, zone_id: int | None = None, validated_only: bool = False,
               source: str = "annotations", tolerance_px: float = 2.0) -> dict:
        """Longueurs dedoublonnees, par categorie, orientation et tuile.

        `source` : « annotations » (traces actifs) ou « automatic » (axes du
        detecteur, sur les seules tuiles dont les calques sont calcules ; les
        autres sont signalees dans `unprocessed_zone_ids`, jamais comptees zero).
        """
        from .length_analysis import measure
        from .overlap_measure import distribute_lengths
        zones = self.zones()
        if zone_id is not None:
            zones = [z for z in zones if z["id"] == zone_id]
        if validated_only:
            zones = [z for z in zones if z["status"] == "done"]
        ids = {z["id"] for z in zones}
        missing = []
        if source == "automatic":
            missing = [z["id"] for z in zones if not self.layer_computed(z)]
            zones = [z for z in zones if z["id"] not in set(missing)]
            cracks = self.layers.automatic_cracks(zones)
        elif source == "annotations":
            cracks = [c for c in self.cracks() if c["zone_id"] in ids]
        elif source == "complete":
            result = self.analysis_result()
            if result is None:
                raise SessionError("Pas encore d'analyse complète : Automatique > Analyse complète de la mosaïque.")
            cracks = [{"id": k, "zone_id": s["zone_id"], "category": s["category"], "state": "validated",
                       "geometry": s["geometry"]} for k, s in enumerate(result["segments"]) if s["zone_id"] in ids]
        else:
            raise SessionError(f"Source inconnue : {source}")
        result = measure(cracks, tolerance_px)
        per_zone = distribute_lengths(result["segments"], zones)
        result.update(
            source=source, validated_only=validated_only, unprocessed_zone_ids=missing,
            pixel_um=self.pixel_um, image={"width": self.width, "height": self.height},
            zones=[{**{k: z[k] for k in ("id", "x", "y", "width", "height", "status")},
                    "length_px": per_zone[z["id"]]} for z in zones])
        return result

    def preview(self, x0: int, y0: int, x1: int, y1: int, max_side: int = 1800):
        """Apercu en niveaux de gris d'une region, au plus `max_side` px de cote.

        Rend (pixels, (x0, y0, x1, y1)) : l'etendue exacte couverte, en pixels
        de la mosaique. Lu dans la pyramide, sans jamais charger l'image.
        """
        import numpy as np
        tiles = self.tiles
        span = max(x1 - x0, y1 - y0, 1)
        scale = 1
        while span / scale > max_side:
            scale *= 2
        level = tiles.max_level - (scale.bit_length() - 1)
        size = tiles.tile_size
        left, top = x0 // scale, y0 // scale
        right, bottom = -(-x1 // scale), -(-y1 // scale)
        canvas = np.zeros((bottom - top, right - left), np.uint8)
        for ty in range(top // size, (bottom - 1) // size + 1):
            for tx in range(left // size, (right - 1) // size + 1):
                tile = tiles.tile_array(level, tx, ty)
                ox, oy = tx * size, ty * size
                ax, ay = max(left, ox), max(top, oy)
                bx, by = min(right, ox + tile.shape[1]), min(bottom, oy + tile.shape[0])
                if bx > ax and by > ay:
                    canvas[ay - top:by - top, ax - left:bx - left] = tile[ay - oy:by - oy, ax - ox:bx - ox]
        extent = (left * scale, top * scale, min(self.width, right * scale), min(self.height, bottom * scale))
        return canvas, extent
