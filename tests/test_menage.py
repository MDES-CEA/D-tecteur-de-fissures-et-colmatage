import json
import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from annotation_app import menage


class MenageTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.root = Path(self.folder.name)

    def tearDown(self):
        self.folder.cleanup()

    def _cache(self, name, image, tiles):
        folder = self.root / "cache" / name
        folder.mkdir(parents=True)
        (folder / "source.json").write_text(json.dumps({"image": str(image)}), encoding="utf-8")
        for k, size in enumerate(tiles):
            path = folder / f"tuile-{k}.npz"
            path.write_bytes(b"x" * size)
            os.utime(path, (time.time() - 100 + k, time.time() - 100 + k))
        return folder

    def test_cache_of_a_vanished_mosaic_goes_then_the_oldest_tiles(self):
        image = self.root / "mosaique.tif"
        image.write_bytes(b"x")
        kept = self._cache("vivante", image, [100, 100, 100])
        gone = self._cache("disparue", self.root / "absente.tif", [100])
        menage.tidy_cache(self.root / "cache", limit_bytes=250)
        self.assertFalse(gone.exists())
        self.assertEqual(sorted(p.name for p in kept.glob("*.npz")), ["tuile-1.npz", "tuile-2.npz"])

    def test_only_the_current_detector_detections_stay(self):
        detector = self.root / "detecteur.pt"
        detector.write_bytes(b"reseau")
        folder = self.root / "analyse" / "mosaique"
        folder.mkdir(parents=True)
        current = folder / f"detections_{menage._hash(detector)}.json"
        old = folder / "detections_000000000000.json"
        result = folder / "resultat.json"
        for path in (current, old, result):
            path.write_text("{}", encoding="utf-8")
        menage.tidy_detections(self.root / "analyse", detector)
        self.assertTrue(current.exists())
        self.assertTrue(result.exists())
        self.assertFalse(old.exists())

    def test_empty_wal_next_to_a_backup_goes_never_one_with_data(self):
        backups = self.root / "sauvegardes" / "ancienne"
        backups.mkdir(parents=True)
        for name, content in (("a.sqlite3", b"base"), ("a.sqlite3-wal", b""), ("a.sqlite3-shm", b"s"),
                              ("b.sqlite3", b"base"), ("b.sqlite3-wal", b"donnees"), ("b.sqlite3-shm", b"s")):
            (backups / name).write_bytes(content)
        menage.tidy_backups(self.root / "sauvegardes")
        self.assertEqual(sorted(p.name for p in backups.iterdir()),
                         ["a.sqlite3", "b.sqlite3", "b.sqlite3-shm", "b.sqlite3-wal"])


if __name__ == "__main__":
    unittest.main()
