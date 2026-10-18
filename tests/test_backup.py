import sqlite3
import unittest
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from annotation_app.backup import MINIMUM_KEPT, backup_database, prune


class BackupTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.database = Path(self.folder.name) / "mosaique-a.sqlite3"     # une base de bases/
        self.directory = Path(self.folder.name) / "sauvegardes"
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("CREATE TABLE cracks (id INTEGER PRIMARY KEY, note TEXT)")
            connection.execute("INSERT INTO cracks(note) VALUES ('premiere')")

    def _write(self, note):
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute("INSERT INTO cracks(note) VALUES (?)", (note,))

    def test_the_copy_holds_the_committed_work(self):
        copy = backup_database(self.database, self.directory, datetime(2026, 10, 1, 9))
        with closing(sqlite3.connect(copy)) as connection:
            self.assertEqual(
                connection.execute("SELECT note FROM cracks").fetchall(), [("premiere",)])
            self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")

    def test_an_unchanged_database_is_not_copied_again(self):
        first = backup_database(self.database, self.directory, datetime(2026, 10, 1, 9))
        again = backup_database(self.database, self.directory, datetime(2026, 10, 1, 13))
        self.assertEqual(again, first)
        self.assertEqual(len(list(self.directory.iterdir())), 1)
        self._write("seconde")
        changed = backup_database(self.database, self.directory, datetime(2026, 10, 1, 17))
        self.assertNotEqual(changed, first)
        self.assertEqual(len(list(self.directory.iterdir())), 2)

    def test_old_copies_go_but_the_most_recent_always_stay(self):
        start = datetime(2026, 1, 1)
        for day in range(MINIMUM_KEPT + 5):
            self._write(f"jour {day}")
            backup_database(self.database, self.directory, start + timedelta(days=day))
        # Un long arret : tout a plus de 14 jours, les plus recentes restent.
        prune(self.directory, "mosaique-a", start + timedelta(days=200))
        kept = sorted(self.directory.iterdir())
        self.assertEqual(len(kept), MINIMUM_KEPT)
        self.assertIn(f"{start + timedelta(days=MINIMUM_KEPT + 4):%Y%m%d}", kept[-1].name)

    def test_unrelated_files_are_never_pruned(self):
        self.directory.mkdir()
        manual = self.directory / "mosaique-a-avant-phase0-20200101-000000.sqlite3"
        manual.write_bytes(b"x")
        other = self.directory / "mosaique-a-2-20200101-000000.sqlite3"     # copie d'une autre mosaique
        other.write_bytes(b"x")
        prune(self.directory, "mosaique-a", datetime(2026, 10, 1))
        self.assertTrue(manual.exists())
        self.assertTrue(other.exists())

    def test_copies_of_the_old_single_base_are_never_pruned(self):
        """Les copies `annotations-AAAAMMJJ-HHMMSS` gardent un travail peut-etre a restaurer."""
        self.directory.mkdir()
        old = [self.directory / f"annotations-202001{day:02d}-000000.sqlite3" for day in range(1, MINIMUM_KEPT + 6)]
        for path in old:
            path.write_bytes(b"x")
        set_aside = self.directory / "annotations-avant-separation-20261017-090000.sqlite3"
        set_aside.write_bytes(b"x")
        for stem in ("annotations", "mosaique-a", "annotations-2"):
            prune(self.directory, stem, datetime(2026, 10, 1))
        self.assertTrue(all(path.exists() for path in old))
        self.assertTrue(set_aside.exists())

    def test_a_missing_database_is_not_an_error(self):
        self.assertIsNone(backup_database(Path(self.folder.name) / "absente.sqlite3", self.directory))


if __name__ == "__main__":
    unittest.main()
