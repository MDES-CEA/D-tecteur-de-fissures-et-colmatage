"""Mesure la detection automatique contre les traces humains valides.

Sans ce banc, chaque seuil du detecteur est regle a l'aveugle. Les tuiles
marquees `done` et leurs polylignes non rejetees servent de verite terrain :
c'est le seul jeu etiquete du projet.

La comparaison se fait en pixels, a l'echelle d'analyse du detecteur, avec une
tolerance exprimee en pixels pleine resolution :

  precision = part des pixels proposes situes a moins de `tolerance` d'un trace
              humain (ce qui est propose est-il une vraie fissure ?)
  rappel    = part des pixels humains situes a moins de `tolerance` d'une
              proposition (les vraies fissures sont-elles proposees ?)

Deux agregats sont affiches. La moyenne par tuile donne le meme poids a chaque
tuile ; l'agregat global met tous les pixels dans le meme sac et reflete donc
mieux le travail reel de correction, une tuile chargee comptant davantage.

Usage :

    python outils/evaluate_crack_detection.py
    python outils/evaluate_crack_detection.py --limit 8 --json resultats.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.draw import line as draw_line

# Rangé dans outils/ : le dossier du projet doit rester importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from annotation_app.database import AnnotationDatabase  # noqa: E402
from annotation_app.detector_model import default_model_path  # noqa: E402
from annotation_app.echelle import make_detector  # noqa: E402
from annotation_app.bases import tool_database  # noqa: E402
from annotation_app.paths import benchmark_image  # noqa: E402
from annotation_app.tile_service import PyramidTileService  # noqa: E402


# La mosaique de reference si elle est la, sinon la derniere ouverte.
DEFAULT_IMAGE = benchmark_image()

# Etapes de `_analyze_uncached` reportees telles quelles : c'est la ou l'on voit
# quel filtre mange les branches.
STAGE_KEYS = (
    "skeleton_pixels",
    "raw_branches",
    "after_spurs",
    "after_score",
    "after_cycles",
    "after_parallel",
    "accepted_stitches",
    "proposed_lines",
)


def rasterize(polylines, shape, zone, scale) -> np.ndarray:
    """Trace des polylignes en coordonnees image dans le repere d'analyse."""
    mask = np.zeros(shape, dtype=bool)
    for points in polylines:
        points = np.asarray(points, dtype=float)
        if len(points) < 2:
            continue
        rows = (points[:, 1] - zone["y"]) / scale
        cols = (points[:, 0] - zone["x"]) / scale
        for first_row, first_col, second_row, second_col in zip(
            rows, cols, rows[1:], cols[1:]
        ):
            rr, cc = draw_line(
                int(round(first_row)), int(round(first_col)),
                int(round(second_row)), int(round(second_col)),
            )
            valid = (rr >= 0) & (cc >= 0) & (rr < shape[0]) & (cc < shape[1])
            mask[rr[valid], cc[valid]] = True
    return mask


def polyline_length(polylines) -> float:
    total = 0.0
    for points in polylines:
        points = np.asarray(points, dtype=float)
        if len(points) >= 2:
            total += float(np.hypot(*np.diff(points, axis=0).T).sum())
    return total


def evaluate_zone(detector, zone, truth_polylines, tolerance_px, ignore_polylines=None,
                  analysis=None) -> dict | None:
    """Mesure une tuile, y compris les fausses détections sur vérité vide.

    `ignore_polylines` : traces jamais verifies (pointilles) ; ce qui est
    propose a cote d'eux ne compte ni comme juste ni comme faux.
    `analysis` : une analyse deja faite (traces nettoyes, par exemple).
    """
    started = perf_counter()
    analysis = analysis or detector.analyze(zone)
    scale = analysis.scale
    shape = analysis.score.shape

    truth = rasterize(truth_polylines, shape, zone, scale)
    proposed = rasterize(
        [crack["geometry"] for crack in analysis.cracks], shape, zone, scale
    )

    # La tolerance est donnee en pixels pleine resolution : on la ramene a
    # l'echelle reduite ou la comparaison a lieu.
    tolerance = max(1.0, tolerance_px / scale)
    if ignore_polylines:
        ignored = rasterize(ignore_polylines, shape, zone, scale)
        if ignored.any():
            unknown = distance_transform_edt(~ignored) <= tolerance
            if truth.any():
                unknown &= distance_transform_edt(~truth) > tolerance
            proposed &= ~unknown
    if proposed.any() and truth.any():
        near_truth = int((distance_transform_edt(~truth)[proposed] <= tolerance).sum())
        near_proposed = int(
            (distance_transform_edt(~proposed)[truth] <= tolerance).sum()
        )
    else:
        near_truth = 0
        near_proposed = 0

    proposed_total = int(proposed.sum())
    truth_total = int(truth.sum())
    precision = near_truth / proposed_total if proposed_total else 0.0
    recall = near_proposed / truth_total if truth_total else 0.0
    return {
        "zone_id": int(zone["id"]),
        "x": int(zone["x"]),
        "y": int(zone["y"]),
        "scale": scale,
        "human_cracks": len(truth_polylines),
        "human_length_px": round(polyline_length(truth_polylines), 1),
        "proposed_lines": len(analysis.cracks),
        "proposed_pixels": proposed_total,
        "truth_pixels": truth_total,
        "empty_truth": truth_total == 0,
        "false_positive_length_px": polyline_length([c['geometry'] for c in analysis.cracks]) if truth_total == 0 else 0.0,
        "near_truth": near_truth,
        "near_proposed": near_proposed,
        "precision": precision,
        "recall": recall,
        "f1": harmonic(precision, recall),
        "seconds": round(perf_counter() - started, 2),
        "stages": {key: analysis.diagnostics.get(key) for key in STAGE_KEYS},
        "detection_mode": analysis.diagnostics.get("detection_mode"),
    }


def harmonic(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE,
                        required=DEFAULT_IMAGE is None)
    parser.add_argument("--database", type=Path, default=None,
                        help="base d'annotations (defaut : celle de --image, dans annotation_app/data/bases/)")
    parser.add_argument(
        "--tolerance", type=float, default=10.0,
        help="Distance en pixels pleine resolution acceptee entre un trace "
             "propose et le trace humain (defaut : 10)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="N'evaluer que les N premieres tuiles, pour un essai rapide",
    )
    parser.add_argument(
        "--zones", type=int, nargs="*", default=None,
        help="Identifiants de tuiles precis, au lieu de toutes les validees",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="seulement les tuiles de test de l'analyse automatique (python -m analyse preparer)",
    )
    parser.add_argument(
        "--json", type=Path, default=None,
        help="Ecrit le detail par tuile dans ce fichier, pour comparer deux "
             "versions du detecteur",
    )
    parser.add_argument(
        "--model", type=Path, default=None,
        help="Modele de reperage a evaluer (.joblib par pixel ou .pt reseau), "
             "pour comparer un candidat au modele en service sans avoir a "
             "l'ecraser. Par defaut, celui que l'application utiliserait",
    )
    args = parser.parse_args()
    args.database, args.image = tool_database(args.database, args.image)

    database = AnnotationDatabase(args.database)
    image_id = database.image_id_for_path(args.image)
    if image_id is None:
        raise SystemExit(f"Image inconnue de la base : {args.image}")
    connection = sqlite3.connect(f"file:{Path(args.database).resolve().as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    if args.test:
        # Les tuiles de test de l'analyse automatique : jamais apprises.
        from analyse import donnees
        split = donnees.split_for(donnees.load(args.database, args.image))
        rows = ",".join(str(r) for r in split["lignes_test"])
        zones = [dict(row) for row in connection.execute(
            f"SELECT * FROM zones WHERE status='done' AND image_id=? AND row_index IN ({rows}) ORDER BY id",
            (image_id,))]
    elif args.zones:
        placeholders = ",".join("?" * len(args.zones))
        query = f"SELECT * FROM zones WHERE image_id=? AND id IN ({placeholders}) ORDER BY id"
        zones = [dict(row) for row in connection.execute(query, (image_id, *args.zones))]
    else:
        # Seulement les tuiles validees DE CETTE mosaique.
        zones = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM zones WHERE status='done' AND image_id=? ORDER BY id", (image_id,)
            )
        ]
    if args.limit:
        zones = zones[: args.limit]
    if not zones:
        raise SystemExit("Aucune tuile validee a evaluer.")

    model_path = args.model if args.model is not None else default_model_path()
    # Le meme detecteur que l'application, a l'echelle de cette mosaique ; la
    # tolerance, donnee en pixels de la mosaique de reference, garde donc le
    # meme sens physique (10 px de reference = 16 um).
    detector, factor = make_detector(PyramidTileService(args.image),
                                     zone_size=database.image_zone_size(image_id), model_path=model_path)
    args.tolerance *= factor
    if detector.model_artifact is None:
        print(f"Aucun modele charge depuis {args.model} : mode classique.")
    else:
        print(
            f"Modele : {model_path.name}, analyse a "
            f"{detector.model_artifact.get('max_analysis_side')} px de cote, "
            f"seuil {detector.model_artifact.get('threshold')}"
        )
    results: list[dict] = []
    skipped = 0

    def polylines(zone_id, state):
        return [json.loads(row["geometry_json"]) for row in connection.execute(
            "SELECT geometry_json FROM cracks WHERE zone_id=? AND state=?", (zone_id, state))]

    for index, zone in enumerate(zones, start=1):
        # Verite : les fissures classees ; les pointilles jamais verifies ne
        # comptent ni pour ni contre.
        result = evaluate_zone(detector, zone, polylines(zone["id"], "validated"), args.tolerance,
                               ignore_polylines=polylines(zone["id"], "proposed"))
        if result is None:
            skipped += 1
            continue
        results.append(result)
        print(
            f"[{index}/{len(zones)}] tuile {result['zone_id']:>4} "
            f"x={result['x']:>6} y={result['y']:>6}  "
            f"{result['human_cracks']:>2} fissures humaines, "
            f"{result['proposed_lines']:>3} proposees  "
            f"precision {result['precision'] * 100:3.0f} %  "
            f"rappel {result['recall'] * 100:3.0f} %  "
            f"F1 {result['f1'] * 100:3.0f} %  "
            f"({result['seconds']:.1f} s)",
            flush=True,
        )

    if not results:
        raise SystemExit("Aucune tuile validee ne porte de trace exploitable.")

    # Keep historical macro scores on positive tiles; empty tiles contribute
    # false-positive pixels to micro precision and are reported separately.
    positive = [r for r in results if not r['empty_truth']]
    precisions = np.array([r["precision"] for r in positive] or [0.0])
    recalls = np.array([r["recall"] for r in positive] or [0.0])
    f1s = np.array([r["f1"] for r in positive] or [0.0])
    negative = [r for r in results if r['empty_truth']]
    print(f"Tuiles humaines vides : {len(negative)} ; longueur faussement proposee : "
          f"{sum(r['false_positive_length_px'] for r in negative):.1f} px")

    proposed_pixels = sum(r["proposed_pixels"] for r in results)
    truth_pixels = sum(r["truth_pixels"] for r in results)
    micro_precision = (
        sum(r["near_truth"] for r in results) / proposed_pixels
        if proposed_pixels else 0.0
    )
    micro_recall = (
        sum(r["near_proposed"] for r in results) / truth_pixels
        if truth_pixels else 0.0
    )

    empty = sum(1 for r in results if r["proposed_lines"] == 0)
    print()
    print("=" * 78)
    print(
        f"{len(results)} tuiles evaluees"
        + (f", {skipped} sans trace humain" if skipped else "")
        + f", tolerance {args.tolerance:g} px pleine resolution"
    )
    print(f"mode de detection      : {results[0]['detection_mode']}")
    print(
        f"moyenne tuiles positives: precision {precisions.mean() * 100:5.1f} %   "
        f"rappel {recalls.mean() * 100:5.1f} %   F1 {f1s.mean() * 100:5.1f} %"
    )
    print(
        f"agregat global         : precision {micro_precision * 100:5.1f} %   "
        f"rappel {micro_recall * 100:5.1f} %   "
        f"F1 {harmonic(micro_precision, micro_recall) * 100:5.1f} %"
    )
    print(
        f"tuiles sans proposition: {empty} / {len(results)}"
        f"   mediane F1 {np.median(f1s) * 100:.1f} %"
    )
    print(
        f"lignes proposees       : {sum(r['proposed_lines'] for r in results)} "
        f"pour {sum(r['human_cracks'] for r in results)} fissures humaines"
    )
    print()
    print("Moyenne des etapes de la chaine (ou passent les branches) :")
    for key in STAGE_KEYS:
        values = [r["stages"][key] for r in results if r["stages"][key] is not None]
        if values:
            print(f"  {key:20s} : {np.mean(values):8.1f}")

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "tolerance_px": args.tolerance,
                    "zones": results,
                    "summary": {
                        "zones": len(results),
                        "empty_truth_zones": len(negative),
                        "false_positive_length_px_on_empty": sum(r['false_positive_length_px'] for r in negative),
                        "macro_precision": float(precisions.mean()),
                        "macro_recall": float(recalls.mean()),
                        "macro_f1": float(f1s.mean()),
                        "micro_precision": micro_precision,
                        "micro_recall": micro_recall,
                        "micro_f1": harmonic(micro_precision, micro_recall),
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nDetail ecrit dans {args.json}")


if __name__ == "__main__":
    main()
