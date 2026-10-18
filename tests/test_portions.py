"""Coupe des traces et portions de categorie homogene le long d'une fissure."""

import unittest

import numpy as np

from annotation_app.polyligne import cumulative, pieces, project
from colmatage.modele import CLASSES, STEP_UM, portions_from_windows


class PolylineTests(unittest.TestCase):
    def test_cut_keeps_geometry_and_shares_anchors(self):
        geometry = [[0, 0], [100, 0], [100, 50]]
        anchors = [[0, 0], [60, 0], [100, 50]]
        first, second = pieces(anchors, geometry, [120])
        self.assertEqual(first[1], [[0, 0], [100, 0], [100, 20]])
        self.assertEqual(second[1], [[100, 20], [100, 50]])
        self.assertEqual(first[0], [[0, 0], [60, 0], [100, 20]])       # le noeud interieur reste
        self.assertEqual(second[0], [[100, 20], [100, 50]])
        total = sum(cumulative(g)[-1] for _, g in (first, second))
        self.assertAlmostEqual(total, 150)

    def test_projection(self):
        along, point, distance = project([[0, 0], [100, 0]], 30, 4)
        self.assertEqual((along, point, distance), (30, (30, 0), 4))


def probabilities(labels):
    """Probabilites franches (0,8) pour une suite d'etiquettes de fenetres."""
    values = np.full((len(labels), len(CLASSES)), 0.1)
    for row, label in enumerate(labels):
        values[row, CLASSES.index(label)] = 0.8
    return values


class PortionTests(unittest.TestCase):
    def portions(self, labels, min_um=10.0):
        distances = [10.0 + STEP_UM * i for i in range(len(labels))]   # centres des fenetres de 20 um
        return portions_from_windows(distances, probabilities(labels), distances[-1] + 10.0, min_um)

    def test_one_category_gives_one_portion(self):
        portions = self.portions(["filled"] * 6)
        self.assertEqual([p["category"] for p in portions], ["filled"])
        self.assertEqual((portions[0]["start_um"], portions[0]["end_um"]), (0.0, 70.0))

    def test_change_of_category_splits_halfway_between_windows(self):
        portions = self.portions(["filled"] * 4 + ["open"] * 3)
        self.assertEqual([p["category"] for p in portions], ["filled", "open"])
        self.assertEqual(portions[0]["end_um"], 45.0)        # entre 40 et 50 um
        self.assertEqual(portions[1]["start_um"], 45.0)
        self.assertAlmostEqual(portions[0]["confidence"], 0.8)

    def test_ten_micrometres_keeps_a_single_window(self):
        labels = ["filled"] * 3 + ["open"] + ["filled"] * 3
        self.assertEqual(len(self.portions(labels, min_um=10.0)), 3)
        merged = self.portions(labels, min_um=20.0)          # une fenetre seule : 10 um < 20 um
        self.assertEqual([p["category"] for p in merged], ["filled"])


if __name__ == "__main__":
    unittest.main()
