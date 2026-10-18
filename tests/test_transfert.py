"""Passer une mosaique d'un poste a l'autre : exporter sa base, copier l'image, l'ouvrir, importer.

Les deux postes sont deux dossiers `data/` temporaires ; l'image change de
chemin, comme sur un autre ordinateur. Meme enchainement que le menu Fichier.
"""

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from annotation_app import bases


@unittest.skipIf(os.environ.get("CRACK_SKIP_SESSION_TESTS"), "sessions sautees")
class TransferTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import tifffile
        self.folder = TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.saved = dict(os.environ)
        os.environ.update({"CRACK_CACHE_DIR": str(self.root / "cache"), "CRACK_PIXEL_UM": "1.595",
                           "CRACK_COLMATAGE_MODEL": str(self.root / "colmatage.joblib"),
                           "CRACK_ANALYSE_DIR": str(self.root / "analyse")})
        for name in ("CRACK_DATABASE_PATH", "CRACK_IMAGE_ID", "CRACK_BACKUP_DIR"):
            os.environ.pop(name, None)
        self.pixels = (150 + np.random.default_rng(0).normal(0, 12, (1200, 1500))).clip(0, 255).astype(np.uint8)
        self.tifffile = tifffile
        self.logs = []

    def tearDown(self):
        import gc
        os.environ.clear()
        os.environ.update(self.saved)
        gc.collect()
        self.folder.cleanup()

    def image(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tifffile.imwrite(path, self.pixels, tile=(512, 512))
        return path

    def open(self, image, data):
        from annotation_app.session import Session
        return Session(image, log=self.logs.append, data_dir=data)

    def work_on_first_post(self) -> Path:
        """Poste A : deux fissures tracees, puis la base exportee ; l'image n'y est plus visible ensuite."""
        image = self.image(self.root / "posteA" / "MEB" / "Echantillon.tif")
        session = self.open(image, self.root / "posteA" / "data")
        zone = session.zones()[0]
        for k in range(2):
            session.database.create_crack(session.image_id, zone["id"], [[10, 10 + k], [90, 90]],
                                          [[10, 10 + k], [90, 90]], "test")
        exported = bases.export_database(session.database_path, self.root / "cle" / "echantillon.sqlite3",
                                         session.image_id)
        session.close()
        del session                         # la lecture par memmap tient le fichier ouvert
        import gc
        gc.collect()
        image.unlink()                      # sur le poste B, ce chemin n'existe pas
        return exported

    def receive(self, exported: Path, name: str) -> list:
        """Poste B : l'image copiee, ouverte (base vide creee), puis la base importee et la mosaique rouverte."""
        data = self.root / "posteB" / "data"
        image = self.image(self.root / "posteB" / "images" / name)
        session = self.open(image, data)
        self.assertEqual(session.cracks(), [])
        plan = bases.plan_import(exported, data)
        replaced = [p.resolve() for p in plan.replaced]
        reopen = session.database_path in replaced            # comme la fenetre
        session.close()
        bases.apply_import(plan, data, self.logs.append)
        self.assertTrue(reopen, f"la base vide de la mosaique ouverte n'est pas remplacee ({self.logs})")
        session = self.open(image, data)
        cracks = session.cracks()
        session.close()
        return cracks

    def test_same_file_name_on_the_new_post(self):
        self.assertEqual(len(self.receive(self.work_on_first_post(), "Echantillon.tif")), 2)

    def test_renamed_file_on_the_new_post(self):
        self.assertEqual(len(self.receive(self.work_on_first_post(), "mosaique copiee.tif")), 2)


if __name__ == "__main__":
    unittest.main()
