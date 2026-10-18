"""Analyse complete d'une mosaique SANS annotation : le cas d'un nouvel echantillon."""

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
import tifffile

from annotation_app.paths import MODELS_DIR

MODELS_READY = (MODELS_DIR / "colmatage_reseau.pt").exists() and (MODELS_DIR / "crack_detector_unet.pt").exists()


def synthetic_mosaic(path: Path) -> None:
    """Un echantillon clair sur de la resine sombre, avec deux fissures ondulees."""
    rng = np.random.default_rng(3)
    pixels = np.full((3000, 4000), 18, dtype=np.float32)              # resine
    pixels[250:2750, 250:3750] = 150                                  # echantillon
    pixels += rng.normal(0, 12, pixels.shape)
    for y0, level in ((900, 35), (2000, 45)):
        xs = np.arange(500, 3500)
        ys = (y0 + 80 * np.sin(xs / 350)).astype(int)
        for dy in range(-3, 4):
            pixels[ys + dy, xs] = level + rng.normal(0, 5, len(xs))
    tifffile.imwrite(path, np.clip(pixels, 0, 255).astype(np.uint8), tile=(512, 512))


@unittest.skipUnless(MODELS_READY, "modeles de l'analyse absents")
class NewMosaicTests(unittest.TestCase):
    def test_full_analysis_report_and_no_fake_validation(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            root = Path(folder)
            env = {"CRACK_ANALYSE_DIR": str(root / "analyse"), "CRACK_PIXEL_UM": "1.595",
                   "CRACK_BACKUP_DIR": str(root / "sauvegardes"), "CRACK_CACHE_DIR": str(root / "cache")}
            with mock.patch.dict(os.environ, env):
                image, database = root / "nouvel_echantillon.tif", root / "base.sqlite3"
                synthetic_mosaic(image)
                from annotation_app.session import Session
                Session(image, database, log=lambda text: None).close()      # grille des tuiles, aucune annotation

                from analyse import chaine, donnees, rapport
                from analyse.__main__ import main
                result = chaine.run(database, image, log=lambda *a: None)
                summary = result["resume"]
                self.assertGreater(summary["longueur_totale_mm"], 3.0)          # les deux fissures (~2 x 5 mm)
                self.assertAlmostEqual(sum(summary["longueur_mm"].values()), summary["longueur_totale_mm"], places=2)
                annotations = donnees.load(database, image)
                out = annotations.output_dir()
                self.assertTrue((out / "resultat.json").exists())
                self.assertFalse((out / "decoupage.json").exists())           # rien a figer sans tuile validee

                main(["--database", str(database), "--image", str(image), "evaluer"])
                self.assertFalse((out / "evaluation_test.json").exists())     # pas de fausse validation
                pdf = rapport.build_for(annotations, result)
                self.assertGreater(pdf.stat().st_size, 20000)

                # Le bouton de l'application : analyse + import en propositions modifiables + rapport.
                main(["--database", str(database), "--image", str(image), "analyser", "--importer"])
                from annotation_app.database import AnnotationDatabase
                cracks = AnnotationDatabase(database).list_image_cracks(annotations.image_id)
                self.assertTrue(cracks)
                self.assertTrue(all(c["source"] == "analyse" and c["state"] == "proposed" for c in cracks))
                self.assertTrue(all(c["category"] in ("open", "filled", "reopened") for c in cracks))
                current = rapport.current_state(donnees.load(database, image), None, result["pixel_um"])
                self.assertAlmostEqual(current["resume"]["longueur_totale_mm"], summary["longueur_totale_mm"],
                                       delta=0.1 * summary["longueur_totale_mm"])

                # Repartir de zero efface detections et resultat avant de tout refaire.
                detections = list(out.glob("detections_*.json"))
                self.assertEqual(len(detections), 1)
                chaine.forget(out, MODELS_DIR / "crack_detector_unet.pt", log=lambda *a: None)
                self.assertFalse((out / "resultat.json").exists())
                self.assertFalse(detections[0].exists())


if __name__ == "__main__":
    unittest.main()
