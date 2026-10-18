import gc
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from outils import assembler_mosaique as assembler  # noqa: E402
from crop_roi import LargeGrayImage  # noqa: E402


class AssemblyTests(unittest.TestCase):
    """Une image decoupee en tuiles qui se recouvrent doit etre reconstituee
    au pixel pres, malgre des positions de platine imprecises."""

    def test_tiles_go_back_exactly_where_they_were_cut(self):
        rng = np.random.default_rng(3)
        source = gaussian_filter(rng.normal(size=(720, 940)), 2.5)
        source = ((source - source.min()) / np.ptp(source) * 255).astype(np.uint8)
        rows, cols, th, tw = 4, 4, 160, 200
        step_x, step_y = 150, 112                  # 25 et 30 % de recouvrement
        truth = {}
        with TemporaryDirectory() as folder:
            folder = Path(folder)
            for r in range(rows):
                for c in range(cols):
                    # Platine imprecise, et axes legerement tournes.
                    x = 30 + c * step_x + 2 * r + int(rng.integers(-4, 5))
                    y = 30 + r * step_y + 3 * c + int(rng.integers(-4, 5))
                    truth[(r, c)] = (x, y)
                    tifffile.imwrite(folder / f"essai_{r + 1}_{c + 1}.tif", source[y:y + th, x:x + tw])

            grid = assembler.scan(folder, 25.0)
            self.assertEqual((grid["rows"], grid["cols"]), (rows, cols))
            tiles = {(r, c): assembler.read_tile(grid["files"][f"{r},{c}"])
                     for r in range(rows) for c in range(cols)}
            pairs = []
            for r in range(rows):
                for c in range(cols):
                    for kind, other, expected in (("R", (r, c + 1), (step_x, 0)), ("D", (r + 1, c), (0, step_y))):
                        if other in tiles:
                            dx, dy, peak, second = assembler.register_pair(
                                tiles[(r, c)], tiles[other], expected, radius=12)
                            pairs.append({"kind": kind, "row": r, "col": c, "dx": dx, "dy": dy,
                                          "peak": peak, "second": second})
            positions, _, _ = assembler.solve_positions(pairs, rows, cols)

            x_min = min(x for x, _ in truth.values())
            y_min = min(y for _, y in truth.values())
            for key, (x, y) in truth.items():
                self.assertEqual(positions[key], (x - x_min, y - y_min), key)

            output = folder / "mosaique.tif"
            width, height = assembler.compose(grid, positions, 1, output, folder / "apercu.png")
            with tifffile.TiffFile(output) as tif:
                self.assertTrue(tif.is_bigtiff)
                mosaic = tif.asarray()
            self.assertEqual(mosaic.shape, (height, width))
            covered = np.zeros_like(mosaic, dtype=bool)
            for x, y in positions.values():
                covered[y:y + th, x:x + tw] = True
            expected = source[y_min:y_min + height, x_min:x_min + width]
            np.testing.assert_array_equal(mosaic[covered], expected[covered])
            self.assertTrue((mosaic[~covered] == 0).all())
            self.assertTrue((folder / "apercu.png").exists())

            # L'application doit pouvoir la lire directement, sans tout charger.
            image = LargeGrayImage(output)
            try:
                self.assertTrue(image.direct)
                np.testing.assert_array_equal(image.read(10, 20, 30, 5), mosaic[20:25, 10:40])
            finally:
                # Windows refuse d'effacer un fichier encore ouvert.
                image.image.close()
                del image
                gc.collect()

    def test_a_featureless_overlap_is_not_trusted(self):
        rng = np.random.default_rng(1)
        flat_a = rng.normal(100, 8, size=(160, 200)).astype(np.uint8)
        flat_b = rng.normal(100, 8, size=(160, 200)).astype(np.uint8)
        _, _, peak, second = assembler.register_pair(flat_a, flat_b, (160, 0), radius=12)
        self.assertFalse(assembler.accepted(peak, second))


if __name__ == "__main__":
    unittest.main()
