"""Analyse automatique, en ligne de commande : `python -m analyse <etape>`.

    preparer    decoupage apprentissage / test, masque de l'echantillon, image de controle
    colmatage   apprend le classement du colmatage le long des fissures, le mesure sur le test
    analyser    toute la mosaique : detection, nettoyage, colmatage, longueurs
    evaluer     le dernier resultat contre l'annotateur, sur les tuiles de test
    rapport     refait le rapport PDF du dernier resultat (avec le sondage s'il y en a un)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def prepare(args) -> None:
    import numpy as np
    from PIL import Image, ImageDraw

    from annotation_app.echelle import require_pixel_size_um
    from crop_roi import LargeGrayImage

    from . import donnees
    from .masque import SampleMask

    annotations = donnees.load(args.database, args.image)
    out = annotations.output_dir()
    split = donnees.split_for(annotations)
    roles = donnees.zones_by_role(annotations, split)
    print(f"Mosaique {annotations.image_id} : {len(annotations.zones)} tuiles, "
          f"{len(annotations.done_ids)} validees")
    print(f"Lignes de test {split['lignes_test']}, de validation {split['lignes_validation']}")
    pixel_um = require_pixel_size_um(annotations.image_path)
    for name, zones in roles.items():
        ids = {z["id"] for z in zones}
        length = sum(donnees.polyline_length(c["geometry"]) for c in annotations.validated
                     if c["zone_id"] in ids) * pixel_um / 1000
        pending = [c for c in annotations.unverified if c["zone_id"] in ids]
        print(f"  {name:13} {len(zones):3} tuiles validees, {length:6.1f} mm de fissures classees, "
              f"{len(pending)} morceaux jamais verifies (ignores)")

    source = LargeGrayImage(annotations.image_path)
    mask = SampleMask.load_or_compute(annotations.image_path, source, pixel_um, out)
    print(f"Echantillon : {mask.area_mm2():.0f} mm2")
    for name, cracks in (("fissures validees", annotations.validated),
                         ("traits rejetes", annotations.rejected)):
        total = kept = 0.0
        for crack in cracks:
            total += donnees.polyline_length(crack["geometry"])
            kept += sum(donnees.polyline_length(p) for p in mask.keep_pieces(crack["geometry"]))
        print(f"  masque : {name:18} retires {(total - kept) * pixel_um / 1000:5.1f} mm "
              f"sur {total * pixel_um / 1000:6.1f} mm ({(total - kept) / max(total, 1):.1%})")

    # Image de controle : masque, tuiles de test et de validation, fissures validees
    step = mask.step
    small = np.asarray(source._memmap[::step, ::step]) if source.direct else None
    rgb = np.stack([small] * 3, axis=-1).astype(np.float32)
    outside = ~mask.mask[:rgb.shape[0], :rgb.shape[1]]
    rgb[outside] = rgb[outside] * 0.35 + np.array([0, 60, 160]) * 0.65
    edge = mask.mask & (mask.inside_um <= 100)
    rgb[edge[:rgb.shape[0], :rgb.shape[1]]] = (255, 200, 0)
    image = Image.fromarray(rgb.astype(np.uint8))
    draw = ImageDraw.Draw(image)
    colors = {"test": (230, 40, 200), "validation": (40, 200, 230), "apprentissage": (60, 170, 60)}
    for name, zones in roles.items():
        for z in zones:
            draw.rectangle([z["x"] / step, z["y"] / step, (z["x"] + z["width"]) / step,
                            (z["y"] + z["height"]) / step], outline=colors[name], width=3)
    for crack in annotations.validated:
        draw.line([(x / step, y / step) for x, y in crack["geometry"]], fill=(220, 40, 40), width=2)
    half = image.height // 2 + 1
    sheet = Image.new("RGB", (image.width * 2 + 20, half), "white")
    sheet.paste(image.crop((0, 0, image.width, half)), (0, 0))
    sheet.paste(image.crop((0, half, image.width, image.height)), (image.width + 20, 0))
    sheet.thumbnail((2400, 2400))
    sheet.save(out / "controle_preparation.png")
    print(f"Image de controle : {out / 'controle_preparation.png'}")
    print("  magenta = tuiles de test, cyan = validation, vert = apprentissage ; bleu = hors echantillon,"
          " jaune = bande de bord")


def colmatage(args) -> None:
    from . import apprentissage
    apprentissage.run(args.database, args.image, epochs=args.epoques, retrain=not args.reglage_seulement)


def analyse_mosaic(args) -> None:
    """Toute la mosaique, puis la mesure sur les tuiles de test et le rapport PDF."""
    from . import chaine, donnees, rapport

    result = chaine.run(args.database, args.image, test_only=args.test_seulement, detector_path=args.detecteur,
                        redo=args.refaire)
    if args.importer and not args.test_seulement:
        from .importation import import_into
        annotations = donnees.load(args.database, args.image)
        import_into(args.database, annotations.image_id, annotations.zone_size, result["pixel_um"], result)
    print("\nMesure sur les tuiles de test...", flush=True)
    evaluate(args)
    if not args.test_seulement:
        path = rapport.build_for(donnees.load(args.database, args.image), result)
        print(f"Rapport : {path}", flush=True)


def import_again(args) -> None:
    """Reimporte le dernier resultat en propositions, sans rien recalculer."""
    from . import chaine, donnees, rapport
    from .importation import import_into

    annotations = donnees.load(args.database, args.image)
    result = chaine.load_result(annotations)
    if result is None:
        raise SystemExit("Pas encore de resultat : lancer « python -m analyse analyser ».")
    import_into(args.database, annotations.image_id, annotations.zone_size, result["pixel_um"], result)
    print(f"Rapport : {rapport.build_for(donnees.load(args.database, args.image), result)}")


def report(args) -> None:
    from . import donnees, rapport
    print(f"Rapport : {rapport.build_for(donnees.load(args.database, args.image))}")


def evaluate(args) -> None:
    """Le dernier resultat complet, contre l'annotateur, sur les tuiles de test."""
    from annotation_app.echelle import require_pixel_size_um

    from . import chaine, donnees
    from .evaluation import end_to_end

    annotations = donnees.load(args.database, args.image)
    test_only = getattr(args, "test_seulement", False)
    result = chaine.load_result(annotations, "resultat_test" if test_only else "resultat")
    if result is None:
        raise SystemExit("Pas encore de resultat : lancer « python -m analyse analyser ».")
    split = donnees.split_for(annotations)
    s = end_to_end(annotations, split, result, require_pixel_size_um(annotations.image_path))
    out = annotations.output_dir() / "evaluation_test.json"
    if not s["tuiles"]:
        # Une mesure d'avant (annotations supprimees depuis) ne doit plus passer pour celle-ci.
        if out.exists():
            out.write_text(json.dumps({"tuiles": 0}), encoding="utf-8")
        print("Pas de tuile de test annotée sur cette mosaïque : la mesure de la méthode et la correction "
              "viennent des tuiles de test de la mosaïque d'apprentissage des modèles.")
        return
    print(f"{s['tuiles']} tuiles de test (jamais apprises) :")
    print(f"  detection : precision {s['detection_precision']:.1%}, rappel {s['detection_rappel']:.1%} "
          f"(tolerance 16 um, pointilles ignores)")
    print("  rappel par etat : " + ", ".join(f"{c} {v:.0%}" for c, v in s["detection_rappel_par_categorie"].items()
                                              if v is not None))
    print(f"  longueur  : automatique {s['longueur_auto_mm']:.1f} mm, annotateur {s['longueur_humaine_mm']:.1f} mm")
    print(f"  part colmatee : automatique {s['part_colmatee_auto']:.1%}, annotateur {s['part_colmatee_humaine']:.1%} "
          f"(biais {100 * (s['part_colmatee_auto'] - s['part_colmatee_humaine']):+.1f} points)")
    if s["erreur_part_par_tuile_points"] is not None:
        print(f"  erreur moyenne sur la part colmatee, tuile par tuile : {s['erreur_part_par_tuile_points']:.1f} points")
    for t in s["detail"]:
        h, a = t["longueurs_um"]["humain"], t["longueurs_um"]["auto"]
        print(f"    tuile {t['zone_id']:>4} (l{t['ligne']} c{t['colonne']}) : annotateur "
              f"{sum(h.values()) / 1000:5.1f} mm, auto {sum(a.values()) / 1000:5.1f} mm")
    print("  correction par etat (longueur annotee / automatique) : "
          + ", ".join(f"{c} x{r:.2f}" for c, r in s["ratios"].items() if r))
    if not test_only:
        # La correction, mesuree sur le test, est controlee sur d'autres tuiles : la validation.
        from .evaluation import corrected
        v = end_to_end(annotations, split, result, require_pixel_size_um(annotations.image_path), role_name="validation")
        fixed = corrected(v["longueur_auto_par_etat_mm"], s["ratios"])
        s["controle_validation"] = {"tuiles": v["tuiles"], "part_colmatee_humaine": v["part_colmatee_humaine"],
                                    "part_colmatee_auto": v["part_colmatee_auto"],
                                    "part_colmatee_corrigee": fixed["part_colmatee"],
                                    "longueur_humaine_mm": v["longueur_humaine_mm"],
                                    "longueur_auto_mm": v["longueur_auto_mm"],
                                    "longueur_corrigee_mm": fixed["longueur_totale_mm"]}
        print(f"  controle sur les lignes de validation ({v['tuiles']} tuiles) : part colmatee annotateur "
              f"{v['part_colmatee_humaine']:.1%}, automatique {v['part_colmatee_auto']:.1%}, "
              f"corrigee {fixed['part_colmatee']:.1%} ; longueur annotateur {v['longueur_humaine_mm']:.1f} mm, "
              f"automatique {v['longueur_auto_mm']:.1f} mm, corrigee {fixed['longueur_totale_mm']:.1f} mm")
    out.write_text(json.dumps(s, indent=2), encoding="utf-8")
    print(f"Detail : {out}")
    # Sur la mosaique d'apprentissage des modeles, la mesure accompagne aussi les
    # modeles : une mosaique sans annotation s'en servira (correction, rapport).
    from annotation_app.paths import MODELS_DIR
    from .apprentissage import MODEL_PATH_NAME
    from .classement import ColmatageModel
    from .evaluation import MODEL_VALIDATION_NAME
    model = ColmatageModel.load(MODELS_DIR / MODEL_PATH_NAME)
    if model is not None and model.report.get("image") == annotations.image_id and not test_only:
        summary = {k: v for k, v in s.items() if k != "detail"}
        summary.update(mosaique=annotations.image_id, mesure_le=result.get("analyse_le"),
                       detecteur=result.get("detecteur"), colmatage=result.get("colmatage", {}).get("fichier"))
        (MODELS_DIR / MODEL_VALIDATION_NAME).write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                        encoding="utf-8")
        print(f"Mesure jointe aux modeles : {MODELS_DIR / MODEL_VALIDATION_NAME}")


def main(argv=None) -> None:
    from annotation_app.bases import tool_database

    parser = argparse.ArgumentParser(prog="python -m analyse", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database", type=Path, default=None,
                        help="base d'annotations (defaut : celle de --image, dans annotation_app/data/bases/)")
    parser.add_argument("--image", type=Path, default=None,
                        help="mosaique (defaut : la mosaique de reference, sinon la derniere ouverte)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preparer", help="decoupage, masque, image de controle")
    command = commands.add_parser("colmatage", help="apprend le classement du colmatage, le mesure sur le test")
    command.add_argument("--epoques", type=int, default=18)
    command.add_argument("--reglage-seulement", action="store_true",
                         help="reprend le reseau deja appris : refait seulement le reglage du lissage et la mesure")
    command = commands.add_parser("analyser", help="analyse toute la mosaique (reprend la ou elle s'etait arretee)")
    command.add_argument("--test-seulement", action="store_true",
                         help="seulement les tuiles de test, pour mesurer vite une modification")
    command.add_argument("--detecteur", type=Path, default=None,
                         help="un detecteur candidat (.pt) au lieu de celui en service")
    command.add_argument("--refaire", action="store_true",
                         help="repartir de zero : refaire aussi la detection de toutes les tuiles")
    command.add_argument("--importer", action="store_true",
                         help="importer le resultat dans la base, en propositions modifiables (tuiles non validees)")
    command = commands.add_parser("evaluer", help="le dernier resultat contre l'annotateur, sur les tuiles de test")
    command.add_argument("--test-seulement", action="store_true", help="le resultat des tuiles de test seulement")
    commands.add_parser("rapport", help="refait le rapport PDF du dernier resultat (avec le sondage)")
    commands.add_parser("importer", help="reimporte le dernier resultat en propositions, sans recalcul")
    args = parser.parse_args(argv)
    args.database, args.image = tool_database(args.database, args.image)
    from annotation_app.echelle import UnknownPixelSize
    try:
        {"preparer": prepare, "colmatage": colmatage, "analyser": analyse_mosaic,
         "evaluer": evaluate, "rapport": report, "importer": import_again}[args.command](args)
    except UnknownPixelSize as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
    # Sortie immediate une fois le travail fini et ecrit : sinon l'arret de
    # l'interpreteur (fils de torch, lecture par memmap) peut rester suspendu,
    # et l'application attendrait la fin d'un calcul deja termine.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
