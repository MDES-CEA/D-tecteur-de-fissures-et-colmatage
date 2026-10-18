"""Une base par mosaique : separation de l'ancienne base, recherche, sessions, import et export.

Tout se passe dans des dossiers temporaires : jamais dans annotation_app/data/.
"""

import os
import shutil
import sqlite3
import unittest
from contextlib import closing
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from annotation_app import bases
from annotation_app.database import AnnotationDatabase

NOW = datetime(2026, 10, 17, 9, 30, 0)


def _image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def _fill(database: AnnotationDatabase, image_id: str, image: Path, width: int, height: int, cracks: int) -> None:
    """Une image avec tuiles, fissures, revisions et suggestions."""
    database.initialize_image(image_id, image, width, height, 2000)
    zones = database.list_zones(image_id)
    for k in range(cracks):
        zone = zones[k % len(zones)]
        x, y = zone["x"] + 10 + k, zone["y"] + 10
        crack = database.create_crack(image_id, zone["id"], [[x, y], [x + 50, y]], [[x, y], [x + 50, y]], "test")
        crack = database.update_crack(image_id, crack["id"], crack["revision"], crack["anchors"],
                                      crack["geometry"], "filled", "validated", "test")
        database.set_suggestion(crack["id"], crack["revision"], "filled", 0.9)
    database.update_zone(image_id, zones[0]["id"], "done", "test", zones[0]["revision"])


def _snapshot(path: Path) -> dict:
    """Identifiants de toutes les lignes, image par image."""
    with closing(sqlite3.connect(path)) as connection:
        result = {}
        for (image_id,) in connection.execute("SELECT id FROM images ORDER BY id"):
            result[image_id] = {
                "zones": connection.execute(
                    "SELECT id, status, revision FROM zones WHERE image_id=? ORDER BY id", (image_id,)).fetchall(),
                "cracks": connection.execute(
                    "SELECT id, zone_id, revision FROM cracks WHERE image_id=? ORDER BY id", (image_id,)).fetchall(),
                "revisions": connection.execute(
                    "SELECT r.crack_id, r.revision FROM crack_revisions r JOIN cracks c ON c.id=r.crack_id "
                    "WHERE c.image_id=? ORDER BY 1, 2", (image_id,)).fetchall(),
                "suggestions": connection.execute(
                    "SELECT s.crack_id, s.revision FROM suggestions s JOIN cracks c ON c.id=s.crack_id "
                    "WHERE c.image_id=? ORDER BY 1", (image_id,)).fetchall(),
            }
    return result


class _Folder(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.data = self.root / "data"
        self.data.mkdir()
        # Les variables du poste ne doivent pas s'en meler.
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in ("CRACK_BACKUP_DIR", "CRACK_DATABASE_PATH", "CRACK_IMAGE_ID", "CRACK_IMAGE_PATH"):
            os.environ.pop(name, None)
        self.logs = []

    def old_base(self, data: Path | None = None) -> Path:
        """L'ancienne base unique, avec deux mosaiques."""
        path = (data or self.data) / "annotations.sqlite3"
        database = AnnotationDatabase(path)
        self.image_a = _image(self.root / "images" / "Mosaique A.tif")
        self.image_b = _image(self.root / "images" / "autre" / "mosaique-b.tif")
        _fill(database, "mosaique-a", self.image_a, 4500, 2100, 5)
        _fill(database, "mosaique-b", self.image_b, 6000, 4000, 3)
        return path

    def bases_files(self, data: Path | None = None) -> list[str]:
        folder = (data or self.data) / "bases"
        return sorted(p.name for p in folder.glob("*.sqlite3")) if folder.exists() else []


class MigrationTests(_Folder):
    def test_each_mosaic_gets_its_file_with_the_same_identifiers(self):
        old = self.old_base()
        before = _snapshot(old)
        created = bases.migrate(self.data, self.logs.append, NOW)

        self.assertEqual(self.bases_files(), ["mosaique-a.sqlite3", "mosaique-b.sqlite3"])
        self.assertEqual(sorted(p.name for p in created), self.bases_files())
        for image_id in ("mosaique-a", "mosaique-b"):
            path = self.data / "bases" / f"{image_id}.sqlite3"
            self.assertEqual(_snapshot(path), {image_id: before[image_id]})
            with closing(sqlite3.connect(path)) as connection:
                self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            self.assertFalse(Path(f"{path}-wal").exists())           # se suffit a lui-meme
        self.assertFalse(old.exists())
        kept = self.data / "sauvegardes" / "annotations-avant-separation-20261017-093000.sqlite3"
        self.assertEqual(_snapshot(kept), before)                    # l'original, intact
        self.assertEqual(sum(line.startswith("Base separee") for line in self.logs), 2)

        # Les identifiants continuent apres ceux de l'ancienne base.
        database = AnnotationDatabase(self.data / "bases" / "mosaique-b.sqlite3")
        zone = database.list_zones("mosaique-b")[0]
        crack = database.create_crack("mosaique-b", zone["id"], [[1, 1], [9, 9]], [[1, 1], [9, 9]], "test")
        self.assertGreater(crack["id"], max(c[0] for i in before.values() for c in i["cracks"]))

        # Un deuxieme passage est sans effet.
        listing = sorted(p.name for p in self.data.rglob("*"))
        self.assertEqual(bases.migrate(self.data, self.logs.append, NOW), [])
        self.assertEqual(sorted(p.name for p in self.data.rglob("*")), listing)

    def test_a_failure_changes_nothing(self):
        old = self.old_base()
        before = _snapshot(old)
        with mock.patch("annotation_app.bases.os.rename", side_effect=PermissionError("verrouille")):
            with self.assertRaises(bases.MigrationError) as raised:
                bases.migrate(self.data, self.logs.append, NOW)
        self.assertIn("Rien n'a change", str(raised.exception))
        self.assertTrue(old.exists())
        self.assertEqual(_snapshot(old), before)
        self.assertEqual(sorted(p.name for p in self.data.iterdir()), ["annotations.sqlite3"])

        # Une copie qui ne correspond pas a l'original : rien non plus.
        with mock.patch("annotation_app.bases._verify", side_effect=bases.MigrationError("copie differente")):
            with self.assertRaises(bases.MigrationError):
                bases.migrate(self.data, self.logs.append, NOW)
        self.assertEqual(_snapshot(old), before)
        self.assertEqual(sorted(p.name for p in self.data.iterdir()), ["annotations.sqlite3"])

        bases.migrate(self.data, self.logs.append, NOW)                # puis elle reussit
        self.assertEqual(len(self.bases_files()), 2)

    def test_a_base_open_elsewhere_is_left_alone(self):
        old = self.old_base()
        other = sqlite3.connect(old)                                   # une autre fenetre
        other.execute("SELECT count(*) FROM cracks").fetchone()
        try:
            with self.assertRaises(bases.MigrationError) as raised:
                bases.migrate(self.data, self.logs.append, NOW)
            self.assertIn("ouverte ailleurs", str(raised.exception))
            self.assertTrue(old.exists())
            self.assertEqual(self.bases_files(), [])
        finally:
            other.close()
        bases.migrate(self.data, self.logs.append, NOW)
        self.assertEqual(len(self.bases_files()), 2)

    def test_work_still_in_the_wal_is_kept(self):
        source = self.root / "source"
        source.mkdir()
        old = self.old_base(source)
        keeper = sqlite3.connect(old)               # empeche le report du -wal a la fermeture
        keeper.execute("SELECT 1 FROM images").fetchone()
        database = AnnotationDatabase(old)
        zone = database.list_zones("mosaique-a")[0]
        late = database.create_crack("mosaique-a", zone["id"], [[5, 5], [80, 5]], [[5, 5], [80, 5]], "test")
        self.assertGreater(Path(f"{old}-wal").stat().st_size, 0)
        for suffix in ("", "-wal"):                 # comme apres un arret brutal
            shutil.copyfile(f"{old}{suffix}", self.data / f"annotations.sqlite3{suffix}")
        keeper.close()

        bases.migrate(self.data, self.logs.append, NOW)
        ids = [c[0] for c in _snapshot(self.data / "bases" / "mosaique-a.sqlite3")["mosaique-a"]["cracks"]]
        self.assertIn(late["id"], ids)

    def test_an_existing_different_base_stops_everything(self):
        self.old_base()
        folder = self.data / "bases"
        folder.mkdir()
        intruder = folder / "mosaique-b.sqlite3"
        _fill(AnnotationDatabase(intruder), "mosaique-b", _image(self.root / "x" / "b.tif"), 10, 10, 0)
        with self.assertRaises(bases.MigrationError):
            bases.migrate(self.data, self.logs.append, NOW)
        self.assertEqual(self.bases_files(), ["mosaique-b.sqlite3"])
        self.assertTrue((self.data / "annotations.sqlite3").exists())


class ResolutionTests(_Folder):
    def setUp(self):
        super().setUp()
        self.old_base()
        bases.migrate(self.data, self.logs.append, NOW)

    def find(self, image, width, height, image_id=None):
        return bases.database_for(image, width, height, image_id, self.data, self.logs.append)

    def test_a_known_path_finds_its_base(self):
        path, image_id = self.find(self.image_b, 6000, 4000)
        self.assertEqual((path.name, image_id), ("mosaique-b.sqlite3", "mosaique-b"))
        self.assertEqual(bases.find_database(self.image_a, self.data), self.data / "bases" / "mosaique-a.sqlite3")

    def test_a_moved_file_keeps_its_base(self):
        moved = self.root / "ailleurs" / "renommee.tif"
        moved.parent.mkdir()
        self.image_a.replace(moved)
        path, image_id = self.find(moved, 4500, 2100)
        self.assertEqual((path.name, image_id), ("mosaique-a.sqlite3", "mosaique-a"))
        self.assertTrue(any("deplacee" in line for line in self.logs))
        self.assertIsNone(bases.find_database(moved, self.data))     # lecture seule : chemin exact

    def test_a_new_image_gets_a_new_file_and_a_free_identifier(self):
        new = _image(self.root / "nouvelle" / "Nouvelle.tif")
        path, image_id = self.find(new, 1000, 1000)
        self.assertEqual((path.name, image_id), ("nouvelle.sqlite3", "nouvelle"))
        self.assertFalse(path.exists())                              # rien n'est cree ici
        # Meme nom qu'une mosaique d'une autre base, autres dimensions.
        homonym = _image(self.root / "homonyme" / "mosaique-b.tif")
        path, image_id = self.find(homonym, 1000, 1000)
        self.assertEqual((path.name, image_id), ("mosaique-b-2.sqlite3", "mosaique-b-2"))
        # Le nom de l'ancienne base unique n'est jamais donne.
        named = _image(self.root / "x" / "annotations.tif")
        self.assertEqual(self.find(named, 1000, 1000)[1], "annotations-2")

    def test_two_vanished_images_with_these_dimensions_are_not_taken(self):
        AnnotationDatabase(self.data / "bases" / "jumelle.sqlite3").initialize_image(
            "jumelle", self.root / "disparue.tif", 4500, 2100, 2000)
        self.image_a.unlink()
        new = _image(self.root / "nouvelle.tif")
        path, image_id = self.find(new, 4500, 2100)
        self.assertEqual(image_id, "nouvelle")
        self.assertTrue(any("aucune n'est" in line for line in self.logs))

    def test_an_unreadable_file_is_reported_ignored_and_untouched(self):
        broken = self.data / "bases" / "abimee.sqlite3"
        broken.write_bytes(b"pas une base SQLite" * 100)
        content = broken.read_bytes()
        same_name = _image(self.root / "abimee.tif")
        path, image_id = self.find(same_name, 100, 100)
        self.assertEqual(image_id, "abimee-2")                       # ne l'ecrase jamais
        self.assertTrue(any("illisible" in line for line in self.logs))
        self.assertEqual(broken.read_bytes(), content)

    def test_a_file_with_several_images_is_searched_too(self):
        common = self.data / "bases" / "commun.sqlite3"
        database = AnnotationDatabase(common)
        c1 = _image(self.root / "c" / "c1.tif")
        c2 = _image(self.root / "c" / "c2.tif")
        database.initialize_image("c1", c1, 100, 100, 2000)
        database.initialize_image("c2", c2, 200, 200, 2000)
        self.assertEqual(self.find(c2, 200, 200), (common, "c2"))
        self.assertEqual(bases.find_database(c1, self.data), common)

    def test_a_forced_identifier(self):
        new = _image(self.root / "force.tif")
        self.assertEqual(self.find(new, 10, 10, "mon-id"), (self.data / "bases" / "mon-id.sqlite3", "mon-id"))
        with self.assertRaises(ValueError):                          # deja enregistree sous un autre
            self.find(self.image_a, 4500, 2100, "autre-id")
        with self.assertRaises(ValueError):
            self.find(new, 10, 10, "..\\dehors")

    def test_the_read_only_search_creates_nothing(self):
        empty = self.root / "vide"
        self.assertIsNone(bases.find_database(self.image_a, empty))
        self.assertFalse(empty.exists())

    def test_tools_find_the_base_of_their_image(self):
        self.assertEqual(bases.tool_database(Path("x.sqlite3"), None, self.data), (Path("x.sqlite3"), None))
        self.assertEqual(bases.tool_database(None, self.image_b, self.data)[0],
                         self.data / "bases" / "mosaique-b.sqlite3")
        with self.assertRaises(SystemExit) as raised:
            bases.tool_database(None, _image(self.root / "jamais.tif"), self.data)
        self.assertIn("jamais ouverte", str(raised.exception))

    def test_backup_folder(self):
        base = self.data / "bases" / "mosaique-a.sqlite3"
        self.assertEqual(bases.backup_dir_for(base, self.data), (self.data / "sauvegardes").resolve())
        other = self.root / "ailleurs" / "base.sqlite3"
        self.assertEqual(bases.backup_dir_for(other, self.data), other.resolve().parent / "sauvegardes")
        with mock.patch.dict(os.environ, {"CRACK_BACKUP_DIR": str(self.root / "reseau")}):
            self.assertEqual(bases.backup_dir_for(base, self.data), (self.root / "reseau").resolve())


class ImportExportTests(_Folder):
    def setUp(self):
        super().setUp()
        self.old_base()
        self.before = _snapshot(self.data / "annotations.sqlite3")
        bases.migrate(self.data, self.logs.append, NOW)

    def test_export_then_import_on_another_post(self):
        exported = bases.export_database(self.data / "bases" / "mosaique-a.sqlite3",
                                         self.root / "cle" / "mosaique-a.sqlite3", "mosaique-a")
        self.assertEqual(_snapshot(exported), {"mosaique-a": self.before["mosaique-a"]})

        elsewhere = self.root / "collegue"
        plan = bases.plan_import(exported, elsewhere)
        self.assertEqual(([p.name for p in plan.targets.values()], plan.replaced), (["mosaique-a.sqlite3"], []))
        [target] = bases.apply_import(plan, elsewhere, self.logs.append, NOW)
        self.assertEqual(_snapshot(target), {"mosaique-a": self.before["mosaique-a"]})

    def test_export_of_a_forced_base_keeps_only_this_mosaic(self):
        kept = self.data / "sauvegardes" / "annotations-avant-separation-20261017-093000.sqlite3"
        exported = bases.export_database(kept, self.root / "b.sqlite3", "mosaique-b")
        self.assertEqual(_snapshot(exported), {"mosaique-b": self.before["mosaique-b"]})

    def test_a_copy_of_the_old_single_base_is_imported_one_base_per_mosaic(self):
        """Restaurer depuis une ancienne copie `annotations-AAAAMMJJ-HHMMSS.sqlite3`."""
        kept = self.data / "sauvegardes" / "annotations-avant-separation-20261017-093000.sqlite3"
        copy = self.root / "annotations-20261012-102947.sqlite3"
        shutil.copyfile(kept, copy)
        plan = bases.plan_import(copy, self.data)
        self.assertEqual(len(plan.replaced), 2)
        bases.apply_import(plan, self.data, self.logs.append, NOW)
        self.assertEqual(self.bases_files(), ["mosaique-a.sqlite3", "mosaique-b.sqlite3"])
        for image_id in ("mosaique-a", "mosaique-b"):
            self.assertEqual(_snapshot(self.data / "bases" / f"{image_id}.sqlite3"),
                             {image_id: self.before[image_id]})
        self.assertEqual(len(list((self.data / "sauvegardes").glob("*-avant-import-*.sqlite3"))), 2)

    def test_import_over_an_existing_base_keeps_the_old_one(self):
        received = self.root / "recue.sqlite3"
        bases.export_database(self.data / "bases" / "mosaique-b.sqlite3", received, "mosaique-b")
        database = AnnotationDatabase(received)                      # travail du collegue
        zone = database.list_zones("mosaique-b")[1]
        database.create_crack("mosaique-b", zone["id"], [[1, 1], [9, 9]], [[1, 1], [9, 9]], "collegue")
        expected = _snapshot(received)

        plan = bases.plan_import(received, self.data)
        self.assertEqual(plan.replaced, [self.data / "bases" / "mosaique-b.sqlite3"])
        bases.apply_import(plan, self.data, self.logs.append, NOW)
        self.assertEqual(_snapshot(self.data / "bases" / "mosaique-b.sqlite3"), expected)
        saved = self.data / "sauvegardes" / "mosaique-b-avant-import-20261017-093000.sqlite3"
        self.assertEqual(_snapshot(saved), {"mosaique-b": self.before["mosaique-b"]})
        self.assertEqual(self.bases_files(), ["mosaique-a.sqlite3", "mosaique-b.sqlite3"])

    def test_what_is_not_an_annotation_base_is_refused(self):
        text = self.root / "texte.sqlite3"
        text.write_bytes(b"bonjour" * 200)
        other = self.root / "autre.sqlite3"
        with closing(sqlite3.connect(other)) as connection:
            connection.execute("CREATE TABLE t(x)")
        empty = self.root / "vide.sqlite3"
        AnnotationDatabase(empty).initialize_image("x", self.root / "x.tif", 10, 10, 2000)
        with closing(sqlite3.connect(empty)) as connection, connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("DELETE FROM images")
        for path in (text, other, empty, self.data / "bases" / "mosaique-a.sqlite3"):
            with self.subTest(path=path.name), self.assertRaises(ValueError):
                bases.plan_import(path, self.data)
        self.assertEqual(self.bases_files(), ["mosaique-a.sqlite3", "mosaique-b.sqlite3"])


@unittest.skipIf(os.environ.get("CRACK_SKIP_SESSION_TESTS"), "sessions sautees")
class SessionTests(_Folder):
    """Deux mosaiques ouvertes : deux fichiers ; l'ancienne base est separee a l'ouverture."""

    def setUp(self):
        super().setUp()
        import numpy as np
        import tifffile
        os.environ.update({"CRACK_CACHE_DIR": str(self.root / "cache"), "CRACK_PIXEL_UM": "1.595",
                           "CRACK_COLMATAGE_MODEL": str(self.root / "colmatage.joblib"),
                           "CRACK_ANALYSE_DIR": str(self.root / "analyse")})
        pixels = (150 + np.random.default_rng(0).normal(0, 12, (1200, 1500))).clip(0, 255).astype(np.uint8)
        self.first = self.root / "images" / "Premiere.tif"
        self.second = self.root / "images" / "seconde.tif"
        self.first.parent.mkdir()
        tifffile.imwrite(self.first, pixels, tile=(512, 512))
        tifffile.imwrite(self.second, pixels[:1000], tile=(512, 512))

    def open(self, image, **options):
        from annotation_app.session import Session
        session = Session(image, log=self.logs.append, data_dir=self.data, **options)
        self.addCleanup(session.close)
        return session

    def test_two_mosaics_two_files(self):
        old = AnnotationDatabase(self.data / "annotations.sqlite3")   # travail d'avant la separation
        _fill(old, "premiere", self.first, 1500, 1200, 2)

        first = self.open(self.first)
        self.assertFalse(first.database_forced)
        self.assertEqual(first.database_path, (self.data / "bases" / "premiere.sqlite3").resolve())
        self.assertEqual(len(first.cracks()), 2)                       # repris de l'ancienne base
        self.assertFalse((self.data / "annotations.sqlite3").exists())
        self.assertEqual(len(list((self.data / "sauvegardes").glob("annotations-avant-separation-*.sqlite3"))), 1)
        second = self.open(self.second)
        self.assertEqual(second.database_path, (self.data / "bases" / "seconde.sqlite3").resolve())
        self.assertEqual(second.cracks(), [])
        self.assertEqual(self.bases_files(), ["premiere.sqlite3", "seconde.sqlite3"])
        self.assertEqual(second.backup_dir, (self.data / "sauvegardes").resolve())
        self.assertEqual((self.data / "derniere_image.txt").read_text(encoding="utf-8"), str(self.second.resolve()))

        zone = second.zones()[0]
        second.database.create_crack(second.image_id, zone["id"], [[1, 1], [9, 9]], [[1, 1], [9, 9]], "test")
        second.close()
        self.assertFalse(Path(f"{second.database_path}-wal").exists()
                         and Path(f"{second.database_path}-wal").stat().st_size)   # -wal vide ou absent
        with closing(sqlite3.connect(f"file:{second.database_path.as_posix()}?mode=ro&immutable=1",
                                     uri=True)) as connection:            # le fichier seul suffit
            self.assertEqual(connection.execute("SELECT count(*) FROM cracks").fetchone()[0], 1)

    def test_a_forced_base_behaves_as_before(self):
        forced = self.root / "forcee.sqlite3"
        first = self.open(self.first, database_path=forced)
        second = self.open(self.second, database_path=forced)
        self.assertTrue(first.database_forced and second.database_forced)
        self.assertEqual(first.database_path, forced.resolve())
        self.assertEqual(second.database_path, forced.resolve())
        self.assertEqual(self.bases_files(), [])
        self.assertEqual(first.backup_dir, (self.root / "sauvegardes").resolve())
        self.assertFalse((self.data / "derniere_image.txt").exists())
