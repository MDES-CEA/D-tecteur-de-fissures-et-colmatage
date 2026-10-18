"""Entraîne le réseau de segmentation des fissures sur les tuiles validées.

Motivation mesurée : le modèle par pixel décide à partir de dix descripteurs
locaux et ne voit aucune forme, d'où un masque qui allume 6,4 % de l'image pour
un couloir de fissures qui en représente 1 à 2 %. Toute la chaîne aval passe
ensuite son temps à jeter ce bruit, et y perd un tiers du rappel.

Usage, sur la mosaïque à apprendre (toutes ses fissures tracées sur une
dizaine de tuiles, puis « Valider la tuile ») :

    python outils/train_crack_unet.py --image "D:\\MEB\\mosaique.tif" --installer

Par défaut le réseau repart du détecteur en service (`--de-zero` pour s'en
passer) : avec peu de tuiles, c'est nettement meilleur, et 800 pas suffisent
(un quart d'heure sur le processeur). Un cinquième des tuiles validées est
mis de côté, jamais appris, et sert à comparer l'ancien et le nouveau réseau ;
`--installer` ne remplace le détecteur en service que si le nouveau fait mieux
sur ces tuiles, en gardant une copie datée de l'ancien. Relancer ensuite le
serveur pour qu'il le charge.

L'image est lue exactement comme par l'application, à la résolution
d'analyse choisie (`--resolution-analyse`) : 1,6 µm par pixel par défaut, qui
voit les fissures de 2 µm, ou 3,2 µm comme le détecteur d'origine. Le modèle
enregistre cette résolution, et l'application détecte ensuite à la même
(`echelle.py`). La comparaison avec l'ancien détecteur porte sur les tracés
proposés, chacun à sa propre résolution.

Comme pour `train_crack_detector.py`, seules la présence et la position des
tracés servent de cible ; les catégories ne sont jamais utilisées.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.ndimage import binary_dilation
from skimage.draw import line as draw_line

# Rangé dans outils/ : le dossier du projet doit rester importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import crack_length_analysis_multi as base  # noqa: E402
from annotation_app.database import AnnotationDatabase  # noqa: E402
from annotation_app.echelle import REFERENCE_PIXEL_UM, make_detector  # noqa: E402
from annotation_app.bases import tool_database  # noqa: E402
from annotation_app.paths import benchmark_image  # noqa: E402
from annotation_app.tile_service import PyramidTileService  # noqa: E402
from annotation_app.unet_model import DEFAULT_UNET_PATH, build_network, prepare_inputs  # noqa: E402

CANDIDATE_PATH = DEFAULT_UNET_PATH.with_name("crack_detector_unet_candidat.pt")


# La mosaique de reference si elle est la, sinon la derniere ouverte.
DEFAULT_IMAGE = benchmark_image()


def target_mask(polylines, shape, zone, scale) -> np.ndarray:
    """Rasterise les tracés humains, épaissis d'un pixel d'analyse.

    À l'échelle 2 la cible fait donc environ 6 px pleine résolution, soit
    l'ordre de grandeur d'une fissure plus la marge de pointage humaine.
    """
    mask = np.zeros(shape, dtype=bool)
    for points in polylines:
        points = np.asarray(points, dtype=float)
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
    return binary_dilation(mask, structure=np.ones((3, 3)))


def read_reduced(detector, zone):
    """La tuile telle que l'application la donne au reseau, et son echelle.

    Pour une mosaique plus fine que la reference, le detecteur de
    l'application (`ScaledDetector`) la ramene d'abord a la resolution de
    reference : l'apprentissage doit voir exactement la meme image.
    """
    if hasattr(detector, "inner"):
        gray, scale = detector.inner._read_reduced(detector._reference_zone(zone))
        return gray, scale * detector.factor
    return detector._read_reduced(zone)


def line_scores(model_path, tiles_service, zone_size, connection, zones, factor) -> dict:
    """Precision, rappel et F1 des TRACES proposes par un detecteur, sur des tuiles.

    Meme mesure que le banc (`evaluate_crack_detection.py`) : chaque detecteur
    travaille a sa propre resolution d'analyse, la tolerance est physique
    (10 px de reference, 16 um). Comparer des masques pixel a pixel n'aurait
    pas de sens entre deux resolutions differentes.
    """
    from outils.evaluate_crack_detection import evaluate_zone

    detector, _ = make_detector(tiles_service, zone_size=zone_size, model_path=model_path)
    totals = {"near_truth": 0, "proposed_pixels": 0, "near_proposed": 0, "truth_pixels": 0}
    for zone in zones:
        result = evaluate_zone(detector, zone, polylines_of(connection, zone["id"], "validated"), 10.0 * factor,
                               ignore_polylines=polylines_of(connection, zone["id"], "proposed"))
        for key in totals:
            totals[key] += result[key]
    precision = totals["near_truth"] / totals["proposed_pixels"] if totals["proposed_pixels"] else 0.0
    recall = totals["near_proposed"] / totals["truth_pixels"] if totals["truth_pixels"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "analysis_um": detector.analysis_um}


def describe(score: dict) -> str:
    return (f"precision {score['precision'] * 100:5.1f} %  rappel {score['recall'] * 100:5.1f} %  "
            f"F1 {score['f1'] * 100:5.1f} %  (analyse a {score['analysis_um']:.1f} um)")


def polylines_of(connection, zone_id, state) -> list:
    return [json.loads(row["geometry_json"]) for row in connection.execute(
        "SELECT geometry_json FROM cracks WHERE zone_id=? AND state=?", (zone_id, state))]


def load_tiles(connection, detector, zones) -> list[dict]:
    """Tuiles validees : cible = fissures classees ; les propositions jamais
    verifiees (pointilles) ne comptent ni comme fissure ni comme fond."""
    tiles = []
    for index, zone in enumerate(zones, start=1):
        gray, scale = read_reduced(detector, zone)
        ridge, _ = base.build_cost_image(gray.astype(np.float32) / 255.0)
        target = target_mask(polylines_of(connection, zone["id"], "validated"), gray.shape, zone, scale)
        pending = target_mask(polylines_of(connection, zone["id"], "proposed"), gray.shape, zone, scale)
        ignore = binary_dilation(pending, iterations=2) & ~target
        tiles.append({
            "id": int(zone["id"]),
            "inputs": prepare_inputs(gray, ridge).astype(np.float16),
            "target": target,
            "weight": ~ignore,
            "centers": np.argwhere(target).astype(np.int32),
        })
        print(f"[{index}/{len(zones)}] tuile {zone['id']} : "
              f"{int(target.sum()):>7} px de fissure, {int(ignore.sum()):>6} px ignores", flush=True)
    return tiles


def load_negatives(connection, detector, zones, validated_tree, tolerance, margin=160) -> list[dict]:
    """Traits du detecteur rejetes par l'annotateur, hors des tuiles validees.

    Seuls les vrais faux positifs servent : un trait rejete puis retrace a la
    main (a moins de `tolerance` d'une fissure validee) etait une vraie fissure.
    Autour de chacun, un couloir de quelques pixels est appris comme « pas une
    fissure » ; le reste de la tuile, jamais annote, ne compte pas.
    """
    negatives = []
    for zone in zones:
        lines = []
        for row in connection.execute(
                "SELECT geometry_json FROM cracks WHERE zone_id=? AND state='rejected' AND source='auto'",
                (zone["id"],)):
            points = np.asarray(json.loads(row["geometry_json"]), dtype=float)
            if len(points) < 2:
                continue
            dense = np.vstack([np.linspace(a, b, max(2, int(np.hypot(*(b - a)) / 3)))
                               for a, b in zip(points, points[1:])])
            near = (validated_tree.query(dense, distance_upper_bound=tolerance)[0] < tolerance).mean() \
                if validated_tree is not None else 0.0
            if near <= 0.2:
                lines.append(points.tolist())
        if not lines:
            continue
        gray, scale = read_reduced(detector, zone)
        corridor = binary_dilation(target_mask(lines, gray.shape, zone, scale), iterations=3)
        rows, cols = np.nonzero(corridor)
        top, bottom = max(0, rows.min() - margin), min(gray.shape[0], rows.max() + margin)
        left, right = max(0, cols.min() - margin), min(gray.shape[1], cols.max() + margin)
        gray_crop = gray[top:bottom, left:right]
        ridge, _ = base.build_cost_image(gray_crop.astype(np.float32) / 255.0)
        weight = corridor[top:bottom, left:right]
        negatives.append({
            "id": int(zone["id"]),
            "inputs": prepare_inputs(gray_crop, ridge).astype(np.float16),
            "target": np.zeros(weight.shape, dtype=bool),
            "weight": weight,
            "centers": np.argwhere(weight).astype(np.int32),
        })
        print(f"  negatifs : tuile {zone['id']}, {len(lines)} trait(s) rejete(s)", flush=True)
    return negatives


def pad_to(tile, size):
    """Agrandit une tuile plus petite que l'imagette (bord reflete, cible vide).

    Le reseau descend trois fois : une imagette doit faire exactement `size`,
    multiple de 8. Une petite tuile (bord de mosaique, grille fine) la
    tronquerait.
    """
    height, width = tile["target"].shape
    extra_rows, extra_cols = max(0, size - height), max(0, size - width)
    if not extra_rows and not extra_cols:
        return tile
    mode = "reflect" if extra_rows < height and extra_cols < width else "edge"
    inputs = np.pad(tile["inputs"], ((0, 0), (0, extra_rows), (0, extra_cols)), mode=mode)
    target = np.pad(tile["target"], ((0, extra_rows), (0, extra_cols)))
    weight = np.pad(tile["weight"], ((0, extra_rows), (0, extra_cols)))     # le bord ajoute ne compte pas
    return {**tile, "inputs": inputs, "target": target, "weight": weight}


def sample_patch(tile, size, rng):
    """Une imagette, centrée 4 fois sur 5 sur un pixel d'intérêt (fissure, ou trait rejeté)."""
    tile = pad_to(tile, size)
    height, width = tile["target"].shape
    points = tile["centers"]
    if len(points) and rng.random() < 0.8:
        row, col = points[rng.integers(len(points))]
        row = int(row) - size // 2 + int(rng.integers(-size // 4, size // 4 + 1))
        col = int(col) - size // 2 + int(rng.integers(-size // 4, size // 4 + 1))
    else:
        row = int(rng.integers(0, max(1, height - size)))
        col = int(rng.integers(0, max(1, width - size)))
    row = max(0, min(row, height - size))
    col = max(0, min(col, width - size))
    inputs = tile["inputs"][:, row:row + size, col:col + size].astype(np.float32)
    target = tile["target"][row:row + size, col:col + size].astype(np.float32)
    weight = tile["weight"][row:row + size, col:col + size].astype(np.float32)

    turns = int(rng.integers(4))
    if turns:
        inputs = np.rot90(inputs, turns, axes=(1, 2))
        target = np.rot90(target, turns)
        weight = np.rot90(weight, turns)
    if rng.random() < 0.5:
        inputs, target, weight = inputs[:, ::-1], target[::-1], weight[::-1]
    if rng.random() < 0.5:
        inputs, target, weight = inputs[:, :, ::-1], target[:, ::-1], weight[:, ::-1]
    return np.ascontiguousarray(inputs), np.ascontiguousarray(target), np.ascontiguousarray(weight)


def batch(tiles, size, count, rng, negatives=(), negative_share=0.0):
    """Imagettes des tuiles validees, et une part prise sur les traits rejetes."""
    triples = []
    for _ in range(count):
        if negatives and rng.random() < negative_share:
            source = negatives[rng.integers(len(negatives))]
        else:
            source = tiles[rng.integers(len(tiles))]
        triples.append(sample_patch(source, size, rng))
    inputs = np.stack([t[0] for t in triples])
    targets = np.stack([t[1] for t in triples])[:, None]
    weights = np.stack([t[2] for t in triples])[:, None]
    return inputs, targets, weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE,
                        required=DEFAULT_IMAGE is None)
    parser.add_argument("--database", type=Path, default=None,
                        help="base d'annotations (défaut : celle de --image, dans annotation_app/data/bases/)")
    parser.add_argument("--output", type=Path, default=CANDIDATE_PATH,
                        help="réseau appris (défaut : un candidat, à côté du détecteur en service)")
    parser.add_argument("--exclude-zones", type=int, nargs="*", default=None,
                        help="Tuiles réservées à la validation et jamais apprises "
                             "(défaut : un cinquième des tuiles validées, tiré au sort)")
    parser.add_argument("--de-zero", action="store_true",
                        help="partir d'un réseau vierge plutôt que du détecteur en service")
    parser.add_argument("--installer", action="store_true",
                        help="remplacer le détecteur en service si le nouveau fait mieux")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--resolution-analyse", type=float, default=REFERENCE_PIXEL_UM,
                        help="taille d'un pixel d'analyse en um : 1,6 (defaut, voit les fissures fines) "
                             "ou 3,2 (le detecteur d'origine, quatre fois plus rapide)")
    parser.add_argument("--decoupage", action="store_true",
                        help="decoupage de l'analyse automatique (python -m analyse preparer) : les lignes "
                             "de test ne sont jamais lues, celles de validation servent aux reglages")
    parser.add_argument("--negatifs", action="store_true",
                        help="apprendre aussi les traits rejetes hors des tuiles validees (pas une fissure)")
    parser.add_argument("--part-negatifs", type=float, default=0.2,
                        help="part des imagettes prises sur les traits rejetes")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patch", type=int, default=256)
    parser.add_argument("--base", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=287298)
    args = parser.parse_args()
    args.database, args.image = tool_database(args.database, args.image)

    import torch
    from torch.nn import functional

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    print(f"torch {torch.__version__}, {torch.get_num_threads()} threads")

    database = AnnotationDatabase(args.database)
    image_id = database.image_id_for_path(args.image)
    if image_id is None:
        raise SystemExit(f"Image inconnue de la base : {args.image}")
    # Lecture seule : l'apprentissage ne doit rien pouvoir ecrire dans les annotations.
    connection = sqlite3.connect(f"file:{Path(args.database).resolve().as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    # Seulement les tuiles validees DE CETTE mosaique.
    zones = [dict(row) for row in connection.execute(
        "SELECT * FROM zones WHERE status='done' AND image_id=? ORDER BY id", (image_id,))]
    roles = {}
    if args.decoupage:
        from analyse import donnees
        split = donnees.split_for(donnees.load(args.database, args.image))
        roles = {zone["id"]: donnees.role(zone, split) for zone in connection.execute(
            "SELECT id, row_index FROM zones WHERE image_id=?", (image_id,))}
        test_count = sum(roles[zone["id"]] == "test" for zone in zones)
        zones = [zone for zone in zones if roles[zone["id"]] != "test"]
        print(f"Decoupage : lignes de test {split['lignes_test']} ({test_count} tuiles, jamais lues), "
              f"lignes de validation {split['lignes_validation']}")
    if len(zones) < 3:
        raise SystemExit(f"{len(zones)} tuile(s) validee(s) sur {image_id} : il en faut au moins 3, "
                         "idealement une dizaine, avec TOUTES leurs fissures tracees.")
    # Le meme detecteur que l'application, a l'echelle de cette mosaique et a
    # la resolution d'analyse choisie : c'est elle que le modele declarera.
    zone_size = database.image_zone_size(image_id)
    tiles_service = PyramidTileService(args.image)
    detector, factor = make_detector(tiles_service, zone_size=zone_size,
                                     analysis_um=args.resolution_analyse, model_path=None)
    if factor != 1.0:
        print(f"Mosaique {factor:.2f} fois plus fine que la reference.")
    print(f"Resolution d'analyse : {detector.analysis_um:.2f} um par pixel")

    if args.decoupage:
        excluded = {zone["id"] for zone in zones if roles[zone["id"]] == "validation"}
    elif args.exclude_zones is not None:
        excluded = set(args.exclude_zones)
    else:
        # Un cinquieme des tuiles, jamais apprises : la seule mesure honnete.
        ids = [zone["id"] for zone in zones]
        excluded = set(np.random.default_rng(args.seed).choice(ids, size=max(1, len(ids) // 5),
                                                               replace=False).tolist())
    print(f"Tuiles mises de cote pour la validation : {sorted(excluded)}")
    print(f"Préparation de {len(zones)} tuiles...")
    tiles = load_tiles(connection, detector, zones)
    training = [tile for tile in tiles
                if tile["id"] not in excluded and len(tile["centers"])]
    validation = [tile for tile in tiles
                  if tile["id"] in excluded and len(tile["centers"])]
    negatives = []
    if args.negatifs:
        from scipy.spatial import cKDTree

        from annotation_app.echelle import pixel_size_um
        points = []
        for row in connection.execute(
                "SELECT geometry_json FROM cracks WHERE image_id=? AND state='validated'", (image_id,)):
            line = np.asarray(json.loads(row["geometry_json"]), dtype=float)
            for a, b in zip(line, line[1:]):
                points.append(np.linspace(a, b, max(2, int(np.hypot(*(b - a)) / 3))))
        tree = cKDTree(np.vstack(points)) if points else None
        tolerance = 5.0 / (pixel_size_um(args.image) or REFERENCE_PIXEL_UM)
        others = [dict(row) for row in connection.execute(
            "SELECT * FROM zones WHERE status!='done' AND image_id=? ORDER BY id", (image_id,))]
        if roles:
            others = [zone for zone in others if roles[zone["id"]] == "apprentissage"]
        print(f"Traits rejetes hors tuiles validees ({len(others)} tuiles candidates)...")
        negatives = load_negatives(connection, detector, others, tree, tolerance)
        print(f"{len(negatives)} tuile(s) de traits rejetes, {args.part_negatifs:.0%} des imagettes")
    if not validation:
        # Sans exclusion demandee, on garde quand meme de quoi surveiller la
        # convergence, mais les chiffres seront alors optimistes.
        validation = training[::6]
        print("Aucune tuile exclue : la validation recoupe l'apprentissage.")
    print(f"{len(training)} tuiles apprises, {len(validation)} en validation")

    positive = sum(int(tile["target"].sum()) for tile in training)
    total = sum(tile["target"].size for tile in training)
    pos_weight = float(min(20.0, (total - positive) / max(positive, 1)))
    print(f"Fissures : {100 * positive / total:.2f} % des pixels, "
          f"pos_weight {pos_weight:.1f}")

    network = build_network(in_channels=2, base=args.base)
    if not args.de_zero and DEFAULT_UNET_PATH.exists():
        previous = torch.load(DEFAULT_UNET_PATH, map_location="cpu", weights_only=False)
        if previous.get("base", args.base) == args.base:
            network.load_state_dict(previous["state_dict"])
            print(f"Départ : détecteur en service ({DEFAULT_UNET_PATH.name})")
    parameters = sum(p.numel() for p in network.parameters())
    print(f"Réseau : {parameters:,} paramètres")
    optimiser = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.steps)
    weight = torch.tensor(pos_weight)

    def loss_of(logits, targets, weights):
        """Entropie croisee + Dice, sur les seuls pixels de poids 1 (ni ignores, ni hors couloir)."""
        entropy = functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=weight, reduction="none")
        entropy = (entropy * weights).sum() / weights.sum().clamp(min=1.0)
        probability = torch.sigmoid(logits) * weights
        overlap = (probability * targets).sum()
        dice = 1.0 - (2.0 * overlap + 1.0) / (
            probability.sum() + (targets * weights).sum() + 1.0)
        return entropy + dice

    fixed_validation = [batch(validation, args.patch, args.batch, np.random.default_rng(7))
                        for _ in range(8)]
    best = float("inf")
    started = perf_counter()
    network.train()
    for step in range(1, args.steps + 1):
        inputs, targets, weights = batch(training, args.patch, args.batch, rng, negatives, args.part_negatifs)
        logits = network(torch.from_numpy(inputs))
        loss = loss_of(logits, torch.from_numpy(targets), torch.from_numpy(weights))
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        schedule.step()

        if step % 100 == 0 or step == 20:
            network.eval()
            with torch.no_grad():
                scores = [
                    float(loss_of(network(torch.from_numpy(a)), torch.from_numpy(b), torch.from_numpy(c)))
                    for a, b, c in fixed_validation
                ]
            network.train()
            mean = float(np.mean(scores))
            marker = ""
            if mean < best:
                best = mean
                marker = "  <- meilleur"
                torch.save(
                    {
                        "state_dict": network.state_dict(),
                        "in_channels": 2,
                        "base": args.base,
                        "threshold": 0.5,
                        "max_analysis_side": 1000,
                        "analysis_um": float(detector.analysis_um),
                        "trained_zone_ids": [tile["id"] for tile in training],
                        "negative_zone_ids": [tile["id"] for tile in negatives],
                        "split": "analyse" if args.decoupage else "aleatoire",
                        "features_version": "unet-1",
                    },
                    args.output,
                )
            rate = step / (perf_counter() - started)
            print(f"  pas {step:>5}/{args.steps}  apprentissage {loss.item():.4f}  "
                  f"validation {mean:.4f}  {rate:.2f} pas/s{marker}", flush=True)

    # Calibration du seuil sur les tuiles de validation, en pleine tuile.
    from annotation_app.unet_model import load_unet, unet_score

    artifact = load_unet(args.output)
    best_threshold, best_f1 = 0.5, -1.0
    probabilities, targets = [], []
    for tile in validation:
        inputs = tile["inputs"].astype(np.float32)
        with torch.no_grad():
            padded = np.pad(inputs, ((0, 0), (0, (-inputs.shape[1]) % 8),
                                     (0, (-inputs.shape[2]) % 8)), mode="reflect")
            logits = artifact["model"](torch.from_numpy(padded)[None])
            probability = torch.sigmoid(logits)[0, 0].numpy()
        counted = tile["weight"].ravel()                  # sans les pointilles jamais verifies
        probabilities.append(probability[:inputs.shape[1], :inputs.shape[2]].ravel()[counted])
        targets.append(tile["target"].ravel()[counted])
    probability = np.concatenate(probabilities)
    truth = np.concatenate(targets)
    sweep = np.concatenate([np.arange(0.10, 0.90, 0.05), np.arange(0.90, 0.996, 0.01)])
    for threshold in sweep:
        predicted = probability >= threshold
        hit = float((predicted & truth).sum())
        if not hit:
            continue
        precision = hit / predicted.sum()
        recall = hit / truth.sum()
        f1 = 2 * precision * recall / (precision + recall)
        print(f"  seuil {threshold:.2f} : precision {precision * 100:5.1f} %  "
              f"rappel {recall * 100:5.1f} %  F1 {f1 * 100:5.1f} %")
        if f1 > best_f1:
            best_threshold, best_f1 = float(threshold), f1

    checkpoint = torch.load(args.output, map_location="cpu", weights_only=False)
    checkpoint["threshold"] = best_threshold
    checkpoint["image"] = image_id
    torch.save(checkpoint, args.output)
    print(f"Seuil retenu : {best_threshold:.2f} (F1 pixel {best_f1 * 100:.1f} %)")
    print(f"Réseau enregistré : {args.output}")

    # Ancien contre nouveau, sur les memes tuiles jamais apprises, et sur ce
    # qui compte : les traces proposes, chacun a sa propre resolution
    # d'analyse, tolerance de 16 um (10 px de reference) comme le banc.
    validation_zones = [zone for zone in zones if zone["id"] in {tile["id"] for tile in validation}]
    print(f"\nComparaison sur les traces proposes ({len(validation_zones)} tuile(s) jamais apprises) :")
    new_score = line_scores(args.output, tiles_service, zone_size, connection, validation_zones, factor)
    print(f"  nouveau : {describe(new_score)}")
    if DEFAULT_UNET_PATH.exists():
        old_score = line_scores(DEFAULT_UNET_PATH, tiles_service, zone_size, connection, validation_zones, factor)
        print(f"  ancien  : {describe(old_score)}")
    else:
        old_score = {"f1": 0.0}
    new_f1, old_f1 = new_score["f1"], old_score["f1"]
    if not args.installer or args.output == DEFAULT_UNET_PATH:
        return
    if new_f1 <= old_f1:
        print("Le nouveau ne fait pas mieux : le détecteur en service est conservé.")
        return
    import shutil
    from datetime import datetime
    if DEFAULT_UNET_PATH.exists():
        # Les versions remplacees, a part : `modeles/` ne garde que ce qui sert.
        archive = DEFAULT_UNET_PATH.parent / "anciens" / f"crack_detector_unet_{datetime.now():%Y%m%d-%H%M}.pt"
        archive.parent.mkdir(exist_ok=True)
        shutil.copy2(DEFAULT_UNET_PATH, archive)
        print(f"Ancien détecteur gardé sous anciens/{archive.name}")
    shutil.copy2(args.output, DEFAULT_UNET_PATH)
    print("Nouveau détecteur installé. Relancer le serveur pour qu'il le charge.")


if __name__ == "__main__":
    main()
