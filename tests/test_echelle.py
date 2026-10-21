import json
import math
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import tifffile

from annotation_app.detection_service import CrackDetector
from annotation_app.echelle import (
    REFERENCE_ANALYSIS_UM,
    REFERENCE_PIXEL_UM,
    ScaledDetector,
    UnknownPixelSize,
    analysis_side,
    make_detector,
    parse_pixel_size,
    pixel_record_file,
    pixel_size_info,
    pixel_size_um,
    record_pixel_size_um,
    require_pixel_size_um,
    resolution_factor,
    zone_size,
)

FACTOR = 4.0
SIZE = 4000


def crack_distance(x, y):
    """Distance a la fissure synthetique y = 0.5 x + 500 (pleine resolution)."""
    return abs(0.5 * x - y + 500) / np.hypot(0.5, 1.0)


class _ArraySource:
    def __init__(self, array):
        self.array = array
        self.height, self.width = array.shape

    def read(self, x, y, width, height):
        return self.array[y:y + height, x:x + width].copy()


class ScaleTests(unittest.TestCase):
    def test_reference_images_are_left_alone(self):
        self.assertEqual(resolution_factor(None), 1.0)
        self.assertEqual(resolution_factor(REFERENCE_PIXEL_UM), 1.0)
        self.assertAlmostEqual(resolution_factor(0.3712), REFERENCE_PIXEL_UM / 0.3712)
        self.assertEqual(zone_size(1.0), 2000)

    def test_the_analysis_resolution_is_physical_whatever_the_zone_size(self):
        factor = REFERENCE_PIXEL_UM / 0.3712
        # La reduction obtenue (comme la calcule le detecteur) doit etre 2 a
        # 3,2 um et 1 a 1,6 um, pour une zone de 3 mm comme pour l'ancienne
        # grille de 2000 px, et meme si l'arrondi des bords ajoute un pixel.
        for zone_size in (8594, 2000):
            reference = round(zone_size / factor)
            for analysis_um, reduction in ((REFERENCE_ANALYSIS_UM, 2), (REFERENCE_PIXEL_UM, 1)):
                side = analysis_side(zone_size, factor, analysis_um)
                for width in (reference - 1, reference, reference + 1):
                    self.assertEqual(math.ceil(width / side), reduction, (zone_size, analysis_um, width))

    def test_a_finer_mosaic_is_analysed_at_the_requested_resolution(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "fine.tif"
            tifffile.imwrite(path, np.zeros((8, 8), np.uint8), description=json.dumps({"pixel_nm": 371.2}))
            source = SimpleNamespace(read=None, width=40000, height=40000)
            tiles = SimpleNamespace(image_path=path, source=source, width=40000, height=40000)
            # Sans modele ni demande : la resolution du detecteur d'origine.
            detector, _ = make_detector(tiles, zone_size=8594, model_path=None)
            self.assertAlmostEqual(detector.analysis_um, REFERENCE_ANALYSIS_UM)
            self.assertEqual(math.ceil(2000 / detector.inner.max_analysis_side), 2)
            # Demandee (entrainement) : 1,6 um.
            detector, _ = make_detector(tiles, zone_size=8594, analysis_um=REFERENCE_PIXEL_UM, model_path=None)
            self.assertEqual(math.ceil(2000 / detector.inner.max_analysis_side), 1)

    def test_the_pixel_size_written_by_the_assembler_is_read_back(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "mosaique.tif"
            tifffile.imwrite(path, np.zeros((8, 8), np.uint8),
                             description=json.dumps({"pixel_nm": 371.2}))
            self.assertAlmostEqual(pixel_size_um(path), 0.3712)
            plain = Path(folder) / "sans_echelle.tif"
            tifffile.imwrite(plain, np.zeros((8, 8), np.uint8))
            self.assertIsNone(pixel_size_um(plain))
            detector, factor = make_detector(SimpleNamespace(image_path=plain, source=None))
            self.assertIsInstance(detector, CrackDetector)
            self.assertEqual(factor, 1.0)


class PixelSizeEntryTests(unittest.TestCase):
    """Image sans metadonnees : la taille du pixel saisie par l'annotateur."""

    def setUp(self):
        self.folder = TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.data = self.root / "data"
        self.forced = os.environ.pop("CRACK_PIXEL_UM", None)

    def tearDown(self):
        import gc
        if self.forced is not None:
            os.environ["CRACK_PIXEL_UM"] = self.forced
        gc.collect()
        self.folder.cleanup()

    def plain(self, name="sans_echelle.tif", side=8):
        path = self.root / "MEB" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(path, np.zeros((side, side), np.uint8))
        return path

    def test_a_typed_pixel_size_is_kept_for_this_image(self):
        image = self.plain()
        self.assertEqual(pixel_size_info(image, self.data), (None, None))
        with self.assertRaises(UnknownPixelSize):
            require_pixel_size_um(image, self.data)
        record_pixel_size_um(image, 0.75, self.data)
        self.assertEqual(pixel_size_info(image, self.data), (0.75, "saisie"))
        self.assertEqual(require_pixel_size_um(image, self.data), 0.75)
        record_pixel_size_um(image, 0.8, self.data)              # correction : remplace
        self.assertEqual(pixel_size_um(image, self.data), 0.8)
        self.assertEqual(len(json.loads(pixel_record_file(self.data).read_text("utf-8"))["images"]), 1)

    def test_the_metadata_win_over_a_typed_value(self):
        image = self.root / "mosaique.tif"
        tifffile.imwrite(image, np.zeros((8, 8), np.uint8), description=json.dumps({"pixel_nm": 371.2}))
        record_pixel_size_um(image, 5.0, self.data)
        value, origin = pixel_size_info(image, self.data)
        self.assertAlmostEqual(value, 0.3712)
        self.assertEqual(origin, "metadonnees")

    def test_a_moved_image_keeps_its_value_but_another_image_does_not(self):
        image = self.plain()
        record_pixel_size_um(image, 0.75, self.data)
        moved = self.root / "ailleurs" / image.name
        moved.parent.mkdir()
        image.rename(moved)
        self.assertEqual(pixel_size_um(moved, self.data), 0.75)
        replaced = self.plain(side=16)                           # meme chemin, autre image
        self.assertIsNone(pixel_size_um(replaced, self.data))

    def test_typed_values(self):
        self.assertEqual(parse_pixel_size(" 1,595 "), 1.595)
        self.assertEqual(parse_pixel_size("0.3712"), 0.3712)
        for text in ("", "abc", "0", "-1", "5000"):
            with self.assertRaises(ValueError, msg=text):
                parse_pixel_size(text)

    def test_the_session_and_the_analysis_use_the_typed_value(self):
        from annotation_app.session import Session
        from analyse.__main__ import main

        image = self.plain(side=1024)
        session = Session(image, log=lambda text: None, data_dir=self.data)
        self.assertIsNone(session.pixel_um)
        database = session.database_path
        session.close()
        with self.assertRaises(SystemExit) as raised:
            main(["--database", str(database), "--image", str(image), "preparer"])
        self.assertIn("Taille du pixel inconnue", str(raised.exception))

        record_pixel_size_um(image, 0.3712, self.data)
        session = Session(image, log=lambda text: None, data_dir=self.data)
        self.assertEqual((session.pixel_um, session.pixel_source), (0.3712, "saisie"))
        self.assertAlmostEqual(session.factor, REFERENCE_PIXEL_UM / 0.3712)
        self.assertIsInstance(session.detector, ScaledDetector)
        self.assertIn("(saisi)", session.describe())
        session.close()


class ScaledDetectorTests(unittest.TestCase):
    """Une image 4 fois plus fine que la reference, avec une fissure diagonale."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(5)
        image = rng.normal(150, 6, size=(SIZE, SIZE))
        rows, cols = np.mgrid[0:SIZE, 0:SIZE]
        image[crack_distance(cols, rows) < 7] = 45
        source = _ArraySource(np.clip(image, 0, 255).astype(np.uint8))
        tiles = SimpleNamespace(source=source, image_path=Path("synthetique.tif"),
                                width=SIZE, height=SIZE)
        cls.detector = ScaledDetector(tiles, FACTOR, model_path=None)
        cls.zone = {"id": 1, "x": 0, "y": 0, "width": SIZE, "height": SIZE}

    def test_a_manual_trace_follows_the_crack_and_keeps_the_clicks(self):
        anchors = [[400.0, 701.5], [1803.25, 1401.0], [3200.0, 2100.0]]
        geometry = self.detector.route(self.zone, anchors)
        self.assertEqual(geometry[0], anchors[0])
        self.assertEqual(geometry[-1], anchors[-1])
        self.assertIn(anchors[1], geometry)
        # Trace a la resolution de reference, puis remis en pleine resolution :
        # il reste dans la fissure (7 px de demi-largeur), a un pixel de
        # reference pres. L'axe exact en pleine resolution est l'affaire de
        # l'analyse du colmatage, qui relit l'image fine le long du trace.
        worst = max(crack_distance(x, y) for x, y in geometry)
        self.assertLess(worst, 7 + FACTOR)

    def test_detection_finds_the_crack_in_full_resolution_coordinates(self):
        analysis = self.detector.analyze(self.zone)
        self.assertEqual(analysis.diagnostics["resolution_factor"], FACTOR)
        self.assertTrue(analysis.cracks)
        points = [point for crack in analysis.cracks for point in crack["geometry"]]
        self.assertGreater(max(x for x, _ in points), SIZE / 2)   # pas reste en coordonnees reduites
        self.assertLess(np.median([crack_distance(x, y) for x, y in points]), 2 * FACTOR)


if __name__ == "__main__":
    unittest.main()
