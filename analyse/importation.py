"""Le resultat de l'analyse complete devient des fissures modifiables dans l'application.

Seulement dans les tuiles NON validees : une tuile validee garde vos fissures
telles quelles. Dans les autres :

- les anciennes propositions automatiques jamais verifiees sont remplacees ;
- un morceau qui double une fissure deja tracee (validee, ou tracee a la main)
  n'est pas importe ;
- une fissure qui traverse un bord de tuile y est coupee (une fissure
  appartient a une seule tuile).

Les fissures importees sont des propositions (pointilles) qui portent deja leur
etat ; valider la tuile les confirme telles qu'elles sont.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

NODE_SPACING_UM = 100.0       # un noeud tous les 100 um : de quoi corriger sans encombrer
DUPLICATE_UM = 10.0
MIN_PIECE_UM = 5.0
MIN_RUN_UM = 100.0            # un troncon d'etat plus court rejoint son voisin, a l'import


def consolidate(segments: list[dict], min_um: float = MIN_RUN_UM) -> list[dict]:
    """Moins de morceaux a corriger : le long de chaque fissure, un troncon d'etat
    plus court que `min_um` rejoint le plus long de ses voisins, puis les troncons
    voisins de meme etat sont fusionnes en une seule geometrie."""
    by_crack: dict = {}
    for segment in segments:
        by_crack.setdefault(segment.get("crack"), []).append(segment)
    result = []
    for crack, parts in by_crack.items():
        if crack is None:
            result.extend(parts)
            continue
        runs = [dict(p, geometry=list(p["geometry"])) for p in parts]   # deja dans l'ordre du trace
        while len(runs) > 1:
            lengths = [r.get("length_um", 0.0) for r in runs]
            short = [i for i, value in enumerate(lengths) if value < min_um]
            if not short:
                break
            i = min(short, key=lambda k: lengths[k])
            neighbours = [k for k in (i - 1, i + 1) if 0 <= k < len(runs)]
            runs[i]["category"] = runs[max(neighbours, key=lambda k: lengths[k])]["category"]
            merged = []
            for run in runs:
                if merged and merged[-1]["category"] == run["category"]:
                    merged[-1]["geometry"] = merged[-1]["geometry"] + run["geometry"][1:]
                    merged[-1]["length_um"] = merged[-1].get("length_um", 0.0) + run.get("length_um", 0.0)
                else:
                    merged.append(run)
            runs = merged
        result.extend(runs)
    return result


def import_result(database, image_id: str, zone_size: int, pixel_um: float, result: dict,
                  annotator: str | None = "analyse") -> dict:
    """Importe les troncons de `result` dans la base `database` (AnnotationDatabase)."""
    from scipy.spatial import cKDTree
    from skimage.measure import approximate_polygon

    from annotation_app.polyligne import cumulative, point_at

    from .evaluation import clip_to_rect
    from .masque import densify

    pixel_um = pixel_um or 1.0
    zones = database.list_zones(image_id)
    grid = {(z["row_index"], z["col_index"]): z for z in zones}
    targets = {z["id"]: z for z in zones if z["status"] != "done"}
    kept = {}
    for crack in database.list_image_cracks(image_id):
        if crack["state"] == "proposed" and crack.get("source") in ("auto", "analyse"):
            continue                                       # sera remplace
        kept.setdefault(crack["zone_id"], []).extend(densify(crack["geometry"], 4.0))
    trees = {zone_id: cKDTree(points) for zone_id, points in kept.items() if points}
    tolerance = DUPLICATE_UM / pixel_um
    spacing = NODE_SPACING_UM / pixel_um
    items, duplicates = [], 0
    for segment in consolidate(result["segments"]):
        points = np.asarray(segment["geometry"], dtype=float)
        (x0, y0), (x1, y1) = points.min(axis=0), points.max(axis=0)
        for row in range(int(y0 // zone_size), int(y1 // zone_size) + 1):
            for col in range(int(x0 // zone_size), int(x1 // zone_size) + 1):
                zone = grid.get((row, col))
                if zone is None or zone["id"] not in targets:
                    continue
                rect = (zone["x"], zone["y"], zone["width"], zone["height"])
                for piece in clip_to_rect(segment["geometry"], rect, spacing=4.0):
                    if cumulative(piece)[-1] * pixel_um < MIN_PIECE_UM:
                        continue
                    tree = trees.get(zone["id"])
                    if tree is not None and (tree.query(piece)[0] <= tolerance).mean() >= 0.5:
                        duplicates += 1
                        continue
                    simple = approximate_polygon(np.asarray(piece), tolerance=1.0)
                    geometry = [[round(float(x), 2), round(float(y), 2)] for x, y in simple]
                    arc = cumulative(geometry)
                    count = max(1, int(round(arc[-1] / spacing)))
                    anchors = [[round(float(x), 2), round(float(y), 2)]
                               for x, y in (point_at(geometry, arc[-1] * k / count, arc) for k in range(count + 1))]
                    items.append({"zone_id": zone["id"], "geometry": geometry, "anchors": anchors,
                                  "category": segment["category"]})
    removed, added = database.replace_proposals(image_id, list(targets), items, annotator)
    return {"importees": added, "remplacees": removed, "doublons_ignores": duplicates,
            "tuiles_validees": len(zones) - len(targets)}


def import_into(database_path: Path, image_id: str, zone_size: int, pixel_um: float, result: dict,
                backup_dir: Path | None = None, log=print) -> dict:
    """Sauvegarde la base, puis importe."""
    from annotation_app.backup import backup_database
    from annotation_app.bases import backup_dir_for
    from annotation_app.database import AnnotationDatabase

    backup_dir = Path(backup_dir or backup_dir_for(database_path))
    saved = backup_database(Path(database_path), backup_dir)
    if saved:
        log(f"Sauvegarde de la base avant import : {saved}")
    summary = import_result(AnnotationDatabase(database_path), image_id, zone_size, pixel_um, result)
    log(f"Import dans l'application : {summary['importees']} fissure(s) proposee(s) dans les tuiles non "
        f"validees ({summary['remplacees']} anciennes propositions remplacees, {summary['doublons_ignores']} "
        f"doublons de vos traces ignores) ; {summary['tuiles_validees']} tuiles validees gardees telles quelles.")
    return summary
