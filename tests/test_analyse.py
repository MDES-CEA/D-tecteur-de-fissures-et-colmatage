"""Analyse automatique : rechainage, lissage, mesures, masque, bandes."""

import unittest

import numpy as np

from analyse import bandes, classement, donnees
from analyse.evaluation import Score, match_changes
from analyse.masque import SampleMask


class ChainTests(unittest.TestCase):
    def test_pieces_sharing_ends_become_one_crack(self):
        cracks = [
            {"id": 1, "zone_id": 7, "category": "open", "geometry": [[0, 0], [100, 0]]},
            {"id": 2, "zone_id": 7, "category": "filled", "geometry": [[150, 0], [100, 0]]},   # a l'envers
            {"id": 3, "zone_id": 7, "category": "open", "geometry": [[150, 0], [300, 0]]},
            {"id": 4, "zone_id": 7, "category": "open", "geometry": [[0, 50], [80, 50]]},      # seule
        ]
        chains = sorted(donnees.chains(cracks), key=lambda c: len(c["segments"]))
        self.assertEqual(len(chains), 2)
        long = chains[1]
        self.assertEqual([s[2] for s in long["segments"]], ["open", "filled", "open"])
        self.assertEqual([round(s[1]) for s in long["segments"]], [100, 150, 300])
        self.assertEqual(long["geometry"][0], [0, 0])
        self.assertEqual(long["geometry"][-1], [300, 0])

    def test_junction_stops_the_chain(self):
        cracks = [{"id": i, "zone_id": 1, "category": "open", "geometry": g} for i, g in enumerate(
            [[[0, 0], [50, 0]], [[50, 0], [100, 0]], [[50, 0], [50, 60]]])]
        self.assertEqual(len(donnees.chains(cracks)), 3)


class SmoothingTests(unittest.TestCase):
    def test_strong_short_change_survives_weak_flips_do_not(self):
        p = np.tile([0.8, 0.1, 0.1], (30, 1))
        p[10] = [0.4, 0.55, 0.05]                      # indice faible, une fenetre
        p[20:23] = [0.02, 0.96, 0.02]                  # indice net, trois fenetres
        states = classement.viterbi(p, change_cost=3.0)
        self.assertEqual(states[10], 0)
        self.assertTrue((states[20:23] == 1).all())
        self.assertEqual(list(classement.viterbi(p, 0.0)[[10, 21]]), [1, 1])   # sans lissage : tout bouge

    def test_segments_bounds_are_halfway_between_windows(self):
        states = np.array([0, 0, 1, 1, 1])
        centers = np.array([12, 22, 32, 42, 52.0])
        segments = classement.segments(states, centers, 64.0)
        self.assertEqual([(s["category"], s["start_um"], s["end_um"]) for s in segments],
                         [("open", 0.0, 27.0), ("filled", 27.0, 64.0)])

    def test_heads_combine_into_the_rule(self):
        p = classement.categories_from_heads(np.array([0.1, 0.9, 0.9]), np.array([0.9, 0.1, 0.9]))
        self.assertEqual(list(p.argmax(axis=1)), [0, 1, 2])   # pas de gel -> non colmatee meme vide
        np.testing.assert_allclose(p.sum(axis=1), 1.0)


class ScoreTests(unittest.TestCase):
    def test_length_and_changes(self):
        truth = [{"category": "open", "start_um": 0, "end_um": 100},
                 {"category": "filled", "start_um": 100, "end_um": 200}]
        predicted = [{"category": "open", "start_um": 0, "end_um": 110},
                     {"category": "filled", "start_um": 110, "end_um": 150},
                     {"category": "reopened", "start_um": 150, "end_um": 200}]
        score = Score(tolerance_um=20)
        score.add(truth, predicted, tile=1)
        s = score.summary()
        self.assertAlmostEqual(s["longueur_bien_classee"], 140 / 200)
        self.assertAlmostEqual(s["colmatee_oui_non"], 190 / 200)
        self.assertAlmostEqual(s["part_colmatee_vraie"], 0.5)
        self.assertAlmostEqual(s["part_colmatee_predite"], 90 / 200)
        self.assertEqual((s["changements_vrais"], s["changements_predits"], s["changements_justes"]), (1, 2, 1))

    def test_each_true_change_is_matched_once(self):
        self.assertEqual(match_changes([100.0], [95.0, 105.0], 20), 1)


class MaskTests(unittest.TestCase):
    def setUp(self):
        mask = np.zeros((40, 40), dtype=bool)
        mask[5:35, 5:35] = True                      # un carre, cellules de 25 px
        self.mask = SampleMask(mask, step=25, pixel_um=0.5)

    def test_lines_along_the_edge_go_crossing_lines_stay(self):
        y = 6.5 * 25                                  # a ~19 um du bord haut, parallele
        along = self.mask.keep_pieces([[300, y], [700, y]], band_um=100)
        self.assertEqual(along, [])
        crossing = self.mask.keep_pieces([[500, 130], [500, 600]], band_um=100)
        self.assertEqual(len(crossing), 1)
        self.assertLess(crossing[0][0][1], 200)       # garde aussi le bout pres du bord
        outside = self.mask.keep_pieces([[10, 10], [60, 10]], band_um=100)
        self.assertEqual(outside, [])


class BandTests(unittest.TestCase):
    def test_windows_are_normalized_and_centered(self):
        pixel_um = 0.375
        image = np.full((600, 1200), 120, dtype=np.uint8)
        image[297:303, 100:1100] = 10                 # fissure horizontale, vide noir
        read = lambda x, y, w, h: image[y:y + h, x:x + w]   # noqa: E731
        windows, centers, positions = bandes.windows(read, 1200, 600, [[150, 300], [1050, 300]], pixel_um)
        self.assertEqual(windows.shape[1:], (bandes.ACROSS, 32))
        self.assertAlmostEqual(np.diff(centers).mean(), bandes.STEP_UM, delta=0.5)
        profile = windows.astype(float).mean(axis=(0, 2))
        self.assertLess(profile[bandes.ACROSS // 2], 0.3)   # le vide au centre
        self.assertAlmostEqual(profile[5], 1.0, delta=0.05)  # la matrice au bord, normalisee a 1
        self.assertTrue(np.allclose(positions[:, 1], 300, atol=1))


class NetworkCleaningTests(unittest.TestCase):
    def test_ends_facing_across_a_tile_border_are_joined(self):
        from analyse.reseau import join_across_gaps
        pixel_um = 0.5
        left = [[0, 100], [490, 100]]                   # s'arrete avant le bord de tuile (x = 500)
        right = [[515, 101], [900, 103]]                # reprend apres
        crossing = [[495, 0], [495, 40]]                # perpendiculaire, loin : ne doit pas etre relie
        joined = join_across_gaps([left, right, crossing], pixel_um, join_um=16)
        lengths = sorted(len(line) for line in joined)
        self.assertEqual(len(joined), 2)
        self.assertEqual(lengths, [2, 4])

    def test_side_by_side_ends_are_not_joined(self):
        from analyse.reseau import join_across_gaps
        a = [[0, 0], [100, 0]]
        b = [[0, 10], [100, 10]]                         # paralleles : les bouts ne se font pas face
        self.assertEqual(len(join_across_gaps([a, b], 1.0, join_um=16)), 2)


class AuditTests(unittest.TestCase):
    def result(self):
        segments = [{"category": "open", "length_um": 300.0, "geometry": [[0, 0], [600, 0]]},
                    {"category": "filled", "length_um": 100.0, "geometry": [[0, 50], [200, 50]]}]
        return {"segments": segments, "resume": {"part_colmatee": 0.25}, "analyse_le": "x"}

    def test_points_follow_the_length(self):
        from analyse.audit import draw_points
        points = draw_points(self.result(), count=400)
        share = np.mean([p["predicted"] == "filled" for p in points])
        self.assertAlmostEqual(share, 0.25, delta=0.06)
        for p in points:
            self.assertIn(p["y"], (0.0, 50.0))

    def test_correction_removes_the_machine_bias(self):
        from analyse.audit import estimate
        # La machine dit « open » partout ; l'annotateur voit la moitie colmatee.
        points = [{"predicted": "open"} for _ in range(100)]
        answers = {str(i): ("filled" if i % 2 else "open") for i in range(100)}
        e = estimate(points, answers, machine_sealed=0.0)
        self.assertAlmostEqual(e["part_colmatee_corrigee"], 0.5)
        low, high = e["part_colmatee_corrigee_ic95"]
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)
        answers["0"] = "not_crack"
        self.assertAlmostEqual(estimate(points, answers, 0.0)["pas_une_fissure"], 0.01)


class CorrectionTests(unittest.TestCase):
    def test_ratios_measured_on_test_fix_the_lengths(self):
        from analyse.evaluation import corrected
        # Sur le test : trop de non colmatee detectee (x0,7), des colmatees manquees (x1,2).
        ratios = {"open": 0.7, "filled": 1.2, "reopened": 1.0}
        fixed = corrected({"open": 100.0, "filled": 50.0, "reopened": 10.0}, ratios)
        self.assertAlmostEqual(fixed["longueur_mm"]["open"], 70.0)
        self.assertAlmostEqual(fixed["longueur_totale_mm"], 140.0)
        self.assertAlmostEqual(fixed["part_colmatee"], 70.0 / 140.0)


class SummaryTests(unittest.TestCase):
    def test_totals_and_sealed_fraction(self):
        from analyse.chaine import summarize
        segments = [{"category": "open", "length_um": 600.0, "zone_id": 1, "confidence": 0.9},
                    {"category": "filled", "length_um": 300.0, "zone_id": 2, "confidence": 0.5},
                    {"category": "reopened", "length_um": 100.0, "zone_id": 2, "confidence": 0.8}]
        s = summarize(segments, None, None)
        self.assertAlmostEqual(s["longueur_totale_mm"], 1.0)
        self.assertAlmostEqual(s["part_colmatee"], 0.4)
        self.assertAlmostEqual(s["longueur_basse_confiance_mm"], 0.3)
        low, high = s["part_colmatee_ic95_tuiles"]
        self.assertLessEqual(low, 0.4)
        self.assertGreaterEqual(high, 0.4)


if __name__ == "__main__":
    unittest.main()
