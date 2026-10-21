"""La mosaique entiere : detection, nettoyage, colmatage tous les 10 um, longueurs.

    python -m analyse analyser            (reprend la ou il s'etait arrete)

Rien n'est ecrit dans la base d'annotations. Dans `data/analyse/<image>/` :

    detections_<detecteur>.json   traces bruts du detecteur, tuile par tuile (reprise)
    resultat.json                 troncons classes, totaux, parts par tuile
    segments.csv                  un troncon par ligne (s'ouvre dans Excel)
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from . import donnees, reseau
from .classement import CATEGORIES, ColmatageModel

SEALED = ("filled", "reopened")


def _file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12] if Path(path).exists() else "aucun"


def detect_all(annotations, detector, mask, out_dir: Path, detector_path: Path, log=print,
               cancelled=lambda: False, only: set | None = None) -> dict[int, list]:
    """Traces du detecteur sur chaque tuile qui touche l'echantillon, enregistres au fur et a mesure."""
    path = out_dir / f"detections_{_file_hash(detector_path)}.json"
    done = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    zones = [z for z in annotations.zones if _touches(z, mask) and (only is None or z["id"] in only)]
    for index, zone in enumerate(zones, start=1):
        if cancelled():
            raise InterruptedError
        key = str(zone["id"])
        if key in done:
            continue
        analysis = detector.analyze(zone)
        done[key] = [crack["geometry"] for crack in analysis.cracks]
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(done), encoding="utf-8")
        temporary.replace(path)
        log(f"[detection {index}/{len(zones)}] tuile {zone['id']} : {len(done[key])} trace(s) "
            f"({analysis.diagnostics.get('elapsed_seconds', 0):.0f} s)")
    return {int(k): v for k, v in done.items() if only is None or int(k) in only}


def _touches(zone, mask) -> bool:
    if mask is None:
        return True
    rows = slice(int(zone["y"] / mask.step), int((zone["y"] + zone["height"]) / mask.step) + 1)
    cols = slice(int(zone["x"] / mask.step), int((zone["x"] + zone["width"]) / mask.step) + 1)
    return bool(mask.mask[rows, cols].any())


def zone_of(point, annotations) -> int | None:
    size = annotations.zone_size
    row, col = int(point[1] // size), int(point[0] // size)
    for zone in annotations.zones:
        if zone["row_index"] == row and zone["col_index"] == col:
            return zone["id"]
    return None


def classify_all(lines, model: ColmatageModel, source, pixel_um, annotations, log=print,
                 cancelled=lambda: False) -> list[dict]:
    """Chaque fissure -> troncons homogenes, avec leur geometrie et leur tuile."""
    from annotation_app.polyligne import cut

    segments = []
    for index, line in enumerate(lines):
        if cancelled():
            raise InterruptedError
        if index % 100 == 0:
            log(f"[colmatage] fissure {index + 1}/{len(lines)}")
        for part in model.classify_line(source.read, source.width, source.height, line, pixel_um):
            geometry = cut(line, part["start_um"] / pixel_um, part["end_um"] / pixel_um)
            if len(geometry) < 2:
                continue
            middle = geometry[len(geometry) // 2]
            segments.append({"crack": index, "category": part["category"],
                             "confidence": round(part["confidence"], 3),
                             "length_um": round(part["end_um"] - part["start_um"], 2),
                             "zone_id": zone_of(middle, annotations), "geometry": geometry})
    return segments


def summarize(segments, mask, annotations, rng_seed=287298) -> dict:
    lengths = {c: 0.0 for c in CATEGORIES}
    per_zone: dict[int, dict] = {}
    for s in segments:
        lengths[s["category"]] += s["length_um"]
        zone = per_zone.setdefault(s["zone_id"], {c: 0.0 for c in CATEGORIES})
        zone[s["category"]] += s["length_um"]
    total = sum(lengths.values())
    sealed = sum(lengths[c] for c in SEALED) / total if total else float("nan")
    # Variabilite d'une tuile a l'autre : on retire des tuiles au hasard (bootstrap).
    keys = [k for k in per_zone if k is not None]
    rng = np.random.default_rng(rng_seed)
    sealed_by_zone = np.array([sum(per_zone[k][c] for c in SEALED) for k in keys])
    total_by_zone = np.array([sum(per_zone[k].values()) for k in keys])
    draws = []
    for _ in range(2000):
        pick = rng.integers(0, len(keys), len(keys))
        if total_by_zone[pick].sum():
            draws.append(sealed_by_zone[pick].sum() / total_by_zone[pick].sum())
    area = mask.area_mm2() if mask is not None else None
    low = float(np.percentile(draws, 2.5)) if draws else None
    high = float(np.percentile(draws, 97.5)) if draws else None
    return {
        "longueur_mm": {c: round(v / 1000, 3) for c, v in lengths.items()},
        "longueur_totale_mm": round(total / 1000, 3),
        "surface_echantillon_mm2": round(area, 1) if area else None,
        "densite_mm_par_mm2": round(total / 1000 / area, 4) if area else None,
        "part_colmatee": sealed,
        "part_colmatee_ic95_tuiles": [low, high],
        "tuiles": {str(k): {c: round(v / 1000, 3) for c, v in d.items()} for k, d in per_zone.items()},
        "troncons": len(segments),
        "longueur_basse_confiance_mm": round(sum(s["length_um"] for s in segments
                                                 if s["confidence"] is not None and s["confidence"] < 0.6) / 1000, 3),
    }


def forget(out: Path, detector_path: Path, log=print) -> None:
    """Repartir de zero : oublie les detections deja faites et le dernier resultat."""
    for name in (f"detections_{_file_hash(detector_path)}.json", "resultat.json", "resultat_test.json",
                 "segments.csv", "rapport.pdf"):
        path = out / name
        if path.exists():
            path.unlink()
            log(f"  efface : {name}")


def run(database: Path, image_path: Path | None = None, log=print, cancelled=lambda: False,
        test_only: bool = False, detector_path: Path | None = None, redo: bool = False) -> dict:
    """`test_only` : seulement les tuiles de test (pour mesurer vite une modification).
    `detector_path` : un detecteur candidat, au lieu de celui en service.
    `redo` : repartir de zero, detection comprise."""
    from annotation_app.detector_model import default_model_path
    from annotation_app.echelle import make_detector, require_pixel_size_um
    from annotation_app.paths import MODELS_DIR
    from annotation_app.tile_service import PyramidTileService
    from crop_roi import LargeGrayImage

    from .apprentissage import MODEL_PATH_NAME
    from .masque import SampleMask

    annotations = donnees.load(database, image_path)
    out = annotations.output_dir()
    out.mkdir(parents=True, exist_ok=True)
    pixel_um = require_pixel_size_um(annotations.image_path)
    source = LargeGrayImage(annotations.image_path)
    mask = SampleMask.load_or_compute(annotations.image_path, source, pixel_um, out)
    model_path = MODELS_DIR / MODEL_PATH_NAME
    model = ColmatageModel.load(model_path)
    if model is None:
        raise RuntimeError("Pas de modele de colmatage : lancer d'abord « python -m analyse colmatage ».")
    detector_path = Path(detector_path) if detector_path else default_model_path()
    tiles = PyramidTileService(annotations.image_path)
    detector, _ = make_detector(tiles, zone_size=annotations.zone_size, model_path=detector_path)
    log(f"Detecteur {Path(detector_path).name} (analyse a {detector.analysis_um:.2f} um), "
        f"modele de colmatage {model_path.name}")
    if redo:
        log("Repartir de zero :")
        forget(out, detector_path, log)

    only = None
    if test_only:
        split = donnees.split_for(annotations)
        only = {z["id"] for z in donnees.zones_by_role(annotations, split)["test"]}
        log(f"Tuiles de test seulement ({len(only)})")
    detections = detect_all(annotations, detector, mask, out, detector_path, log, cancelled, only)
    proposals = [line for lines in detections.values() for line in lines]
    lines = reseau.clean(proposals, mask, pixel_um, log)
    segments = classify_all(lines, model, source, pixel_um, annotations, log, cancelled)
    summary = summarize(segments, mask, annotations)
    result = {"image": annotations.image_id, "image_path": str(annotations.image_path),
              "analyse_le": datetime.now().isoformat(timespec="seconds"), "pixel_um": pixel_um,
              "perimetre": "tuiles de test" if test_only else "mosaique entiere",
              "detecteur": {"fichier": Path(detector_path).name, "empreinte": _file_hash(detector_path),
                            "analyse_um": detector.analysis_um},
              "colmatage": {"fichier": model_path.name, "empreinte": _file_hash(model_path),
                            "cout_changement": model.change_cost, "test": model.report.get("test"),
                            "pixel_um": model.pixel_um},
              "resume": summary, "segments": segments}
    # Un essai sur les tuiles de test n'ecrase pas le resultat de la mosaique entiere.
    name = "resultat_test" if test_only else "resultat"
    temporary = out / f"{name}.tmp"
    temporary.write_text(json.dumps(result), encoding="utf-8")
    temporary.replace(out / f"{name}.json")
    if test_only:
        log(f"Resultat (tuiles de test) : {out / (name + '.json')}")
        return result
    with open(out / "segments.csv", "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream, delimiter=";")
        writer.writerow(["fissure", "categorie", "confiance", "longueur_um", "tuile", "x1", "y1", "x2", "y2"])
        for s in segments:
            (x1, y1), (x2, y2) = s["geometry"][0], s["geometry"][-1]
            writer.writerow([s["crack"], s["category"], s["confidence"], s["length_um"], s["zone_id"],
                             round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)])
    log(f"\nLongueur totale {summary['longueur_totale_mm']:.1f} mm "
        f"({summary['densite_mm_par_mm2']} mm/mm2) ; part colmatee {100 * summary['part_colmatee']:.1f} % "
        f"(IC 95 % entre tuiles {100 * summary['part_colmatee_ic95_tuiles'][0]:.1f}-"
        f"{100 * summary['part_colmatee_ic95_tuiles'][1]:.1f} %)")
    log("  " + ", ".join(f"{c} {v:.1f} mm" for c, v in summary["longueur_mm"].items()))
    log(f"Resultat : {out / 'resultat.json'}")
    return result


def load_result(annotations_or_dir, name: str = "resultat") -> dict | None:
    out = annotations_or_dir if isinstance(annotations_or_dir, Path) else annotations_or_dir.output_dir()
    path = Path(out) / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
