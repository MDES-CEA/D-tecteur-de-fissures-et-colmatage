"""La boucle entrainer -> traiter -> verifier, sur une image synthetique."""

import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from annotation_app.automatisation import Automation
from annotation_app.database import AnnotationDatabase

PIXEL_UM = 0.3712
KINDS = ("open", "filled", "reopened")


def draw(image, kind, x0, x1, y, width, rng, matrix=110.0):
    """Fissure horizontale de x0 a x1 en y, de l'aspect donne."""
    rows = np.arange(image.shape[0])[:, None]
    distance = np.abs(rows - y) - width / 2
    band = np.zeros(image.shape, dtype=bool)
    band[:, x0 - 40:x1 + 40] = True
    inside = band & (distance < 0)
    if kind == "filled":
        image[inside] = rng.normal(0.85 * matrix, 9.0, size=inside.sum())
    else:
        image[inside] = rng.normal(0.1 * matrix, 3.0, size=inside.sum())
    if kind == "reopened":
        rim = band & (distance >= 0) & (distance < 25)
        image[rim] = rng.normal(0.88 * matrix, 5.0, size=rim.sum())
        speckles = rim & (rng.random(image.shape) < 0.3)
        image[speckles] = 0.6 * matrix


class Source:
    def __init__(self, array):
        self.array = array
        self.height, self.width = array.shape

    def read(self, x, y, width, height):
        return self.array[y:y + height, x:x + width]


class AutomationTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        folder = Path(self.folder.name)
        rng = np.random.default_rng(11)
        image = rng.normal(110.0, 5.0, size=(3000, 4000))
        self.database = AnnotationDatabase(folder / "base.sqlite3")
        (folder / "synthetique.tif").write_bytes(b"x")
        self.database.initialize_image("synth", folder / "synthetique.tif", 4000, 3000, 2000)
        zones = {(z["row_index"], z["col_index"]): z for z in self.database.list_zones("synth")}
        self.zones = zones

        # Etiquettes : 24 fissures validees en haut de l'image, 8 de chaque sorte.
        columns = [(100, 900, 0), (1000, 1900, 0), (2100, 2900, 1), (3000, 3900, 1)]
        index = 0
        for y in (200, 500, 800, 1100, 1400, 1700):
            for x0, x1, col in columns:
                kind = KINDS[index % 3]
                width = (6, 14, 30)[(index // 3) % 3]
                draw(image, kind, x0, x1, y, width, rng)
                crack = self.database.create_crack("synth", zones[(0, col)]["id"], [[x0, y], [x1, y]],
                                                   [[x0, y], [x1, y]], "test")
                self.database.update_crack("synth", crack["id"], 0, crack["anchors"], crack["geometry"],
                                           kind, "validated", "test")
                index += 1

        # A traiter : trois fissures d'une tuile vierge, que le « detecteur » trouve.
        self.expected = {}
        self.proposals = []
        for kind, y in zip(KINDS, (2250, 2550, 2800)):
            draw(image, kind, 200, 1800, y, 16, rng)
            self.proposals.append({"anchors": [[200, y], [1800, y]], "geometry": [[200, y], [1800, y]]})
            self.expected[y] = kind
        # ... et une tuile ou l'annotateur a deja rejete une proposition.
        rejected = self.database.create_crack("synth", zones[(1, 1)]["id"], [[2100, 2500], [2900, 2500]],
                                              [[2100, 2500], [2900, 2500]], "test")
        self.database.update_crack("synth", rejected["id"], 0, rejected["anchors"], rejected["geometry"],
                                   "unclassified", "rejected", "test")

        self.source = Source(np.clip(image, 0, 255).astype(np.uint8))
        self.detected_zones = []

        def detect(zone):
            self.detected_zones.append(zone["id"])
            return self.proposals if zone["id"] == zones[(1, 0)]["id"] else []

        # Chemin en texte, comme il arrive de la variable d'environnement.
        self.automation = Automation(self.database, "synth", self.source, PIXEL_UM, detect,
                                     model_path=str(folder / "colmatage.joblib"))

    def wait(self):
        deadline = time.time() + 120
        while self.automation.describe()["running"]:
            self.assertLess(time.time(), deadline, "tache trop longue")
            time.sleep(0.1)
        status = self.automation.describe()
        self.assertIsNone(status["error"], status)
        return status

    def test_train_then_process_then_review(self):
        self.automation.start_training()
        status = self.wait()
        self.assertTrue(Path(self.folder.name, "colmatage.joblib").exists())
        self.assertGreaterEqual(status["model"]["correct"], 0.9)

        zones = [self.zones[(1, 0)], self.zones[(1, 1)]]
        self.automation.start_processing(zones)
        self.wait()
        # La tuile deja travaillee (meme seulement rejetee) n'est pas re-detectee.
        self.assertEqual(self.detected_zones, [self.zones[(1, 0)]["id"]])

        proposed = self.database.list_cracks("synth", self.zones[(1, 0)]["id"])
        self.assertEqual(len(proposed), 3)
        for crack in proposed:
            self.assertEqual(crack["state"], "proposed")
            self.assertEqual(crack["category"], "unclassified")      # une suggestion n'est pas une categorie
            self.assertEqual(crack["suggestion"]["category"], self.expected[crack["geometry"][0][1]])

        # Verification : la file suit la grille, et retrouve la suivante meme
        # quand la precedente vient de la quitter.
        first, remaining = self.automation.next_review()
        self.assertEqual(remaining, 3)
        ids = sorted(c["id"] for c in proposed)
        self.assertEqual(first["id"], ids[0])
        self.database.update_crack("synth", ids[0], first["revision"], first["anchors"],
                                   first["geometry"], first["suggestion"]["category"], "validated", "test")
        crack, remaining = self.automation.next_review(after=ids[0], zone=first["zone_id"])
        self.assertEqual(crack["id"], ids[1])
        self.assertEqual(remaining, 2)

        # Modifier un trace rend sa suggestion caduque.
        moved = self.database.update_crack("synth", crack["id"], crack["revision"], crack["anchors"],
                                           [[200, 2552], [1800, 2552]], "unclassified", "proposed", "test")
        self.assertIsNone(moved["suggestion"])

    def test_detector_proposal_is_split_into_portions(self):
        """Colmatee a gauche, ouverte a droite : deux portions, chacune sa suggestion."""
        self.automation.start_training()
        self.wait()
        rng = np.random.default_rng(5)
        image = self.source.array.astype(float)
        draw(image, "filled", 200, 1000, 2100, 16, rng)
        draw(image, "open", 1000, 1800, 2100, 16, rng)
        self.source.array[:] = np.clip(image, 0, 255).astype(np.uint8)
        zone = self.zones[(1, 0)]
        proposal = {"anchors": [[200, 2100], [1800, 2100]], "geometry": [[200, 2100], [1800, 2100]]}
        self.database.add_detected_cracks_if_empty("synth", zone["id"], [proposal], "automatique")

        self.automation.suggest_zone(zone["id"])
        portions = sorted(self.database.list_cracks("synth", zone["id"]), key=lambda c: c["geometry"][0][0])
        self.assertGreaterEqual(len(portions), 2)
        self.assertAlmostEqual(sum(c["length_px"] for c in portions), 1600, delta=1)
        for crack in portions:
            middle = (crack["geometry"][0][0] + crack["geometry"][-1][0]) / 2
            self.assertEqual(crack["state"], "proposed")
            if middle < 900:
                self.assertEqual(crack["suggestion"]["category"], "filled", crack["geometry"])
            elif middle > 1100:
                self.assertEqual(crack["suggestion"]["category"], "open", crack["geometry"])
        # La coupe tombe a la jonction des deux aspects, a 20 um pres.
        cut = portions[0]["geometry"][-1][0]
        self.assertAlmostEqual(cut, 1000, delta=20 / PIXEL_UM)

    def test_training_refuses_too_few_examples(self):
        for crack in self.database.labelled_cracks("synth", ("reopened",))[2:]:
            self.database.update_crack("synth", crack["id"], crack["revision"], crack["anchors"],
                                       crack["geometry"], "uncertain", "validated", "test")
        with self.assertRaises(RuntimeError) as raised:
            self.automation.start_training()
        self.assertIn("rouverte", str(raised.exception))

    def test_processing_stops_on_request(self):
        zones = list(self.zones.values())
        self.automation.start_processing(zones)
        self.automation.cancel()
        status = self.wait()
        self.assertLess(status["done"], len(zones) + 1)


if __name__ == "__main__":
    unittest.main()
