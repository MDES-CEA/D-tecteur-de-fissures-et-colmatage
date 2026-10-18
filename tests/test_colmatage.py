"""Les descripteurs du gel distinguent les trois aspects sur des fissures
synthetiques, quelle que soit leur largeur et l'eclairage."""

import unittest

import numpy as np

from colmatage.profils import FEATURE_NAMES, trace_windows

PIXEL_UM = 0.3712
F = {name: index for index, name in enumerate(FEATURE_NAMES)}


def synthetic(kind: str, width_px: int, matrix: float = 110.0, seed: int = 0):
    """Fissure horizontale en y = 300, sur une image 600 x 1200."""
    rng = np.random.default_rng(seed)
    image = rng.normal(matrix, 5.0, size=(600, 1200))
    rows = np.broadcast_to(np.arange(600)[:, None], image.shape)
    distance = np.abs(rows - 300) - width_px / 2       # < 0 dans la fissure
    inside = distance < 0
    if kind == "filled":
        image[inside] = rng.normal(0.85 * matrix, 9.0, size=image.shape)[inside]
    else:
        image[inside] = rng.normal(0.1 * matrix, 3.0, size=image.shape)[inside]
    if kind == "reopened":
        # Comme mesure sur les traces etiquetes : une bande plus sombre que la
        # matrice (~0,89), parsemee de grains sombres.
        rim = (distance >= 0) & (distance < 25)
        speckles = rng.random(image.shape) < 0.3
        image[rim] = rng.normal(0.88 * matrix, 5.0, size=image.shape)[rim]
        image[rim & speckles] = 0.6 * matrix
    image = np.clip(image, 0, 255).astype(np.uint8)
    read = lambda x, y, w, h: image[y:y + h, x:x + w]  # noqa: E731
    windows = trace_windows(read, 1200, 600, [[100, 300], [1100, 300]], PIXEL_UM)
    return np.nanmedian(np.array([w.features for w in windows]), axis=0), len(windows)


class GelFeatureTests(unittest.TestCase):
    def test_windows_are_cut_every_ten_micrometres(self):
        _, count = synthetic("open", 12)
        expected = int((1000 * PIXEL_UM - 20) // 10) + 1
        self.assertEqual(count, expected)

    def test_the_void_width_is_measured_in_micrometres(self):
        for width in (6, 20, 60):
            features, _ = synthetic("open", width)
            self.assertAlmostEqual(features[F["void_width_um"]], width * PIXEL_UM, delta=3 * PIXEL_UM)

    def test_a_filled_crack_has_gel_and_no_void(self):
        features, _ = synthetic("filled", 40)
        self.assertEqual(features[F["void_width_um"]], 0.0)
        self.assertGreater(features[F["centre_gel"]], 0.5)
        open_features, _ = synthetic("open", 40)
        self.assertLess(open_features[F["centre_gel"]], 0.2)

    def test_a_gel_rim_is_seen_on_open_cracks_of_any_width(self):
        for width in (8, 40):
            rim, _ = synthetic("reopened", width)
            clean, _ = synthetic("open", width)
            self.assertLess(rim[F["near_level_min"]], clean[F["near_level_min"]] - 0.05, width)
            self.assertGreater(rim[F["near_dark_max"]], clean[F["near_dark_max"]] + 0.2, width)

    def test_illumination_does_not_change_the_description(self):
        dim, _ = synthetic("reopened", 20, matrix=70.0)
        bright, _ = synthetic("reopened", 20, matrix=150.0)
        for name in ("void_width_um", "near_level_min", "near_dark_max", "centre_gel"):
            self.assertAlmostEqual(dim[F[name]], bright[F[name]], delta=0.08 if name != "void_width_um" else 1.0)


if __name__ == "__main__":
    unittest.main()
