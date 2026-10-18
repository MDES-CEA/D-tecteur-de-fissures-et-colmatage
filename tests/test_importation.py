"""Le resultat de l'analyse devient des propositions modifiables, sans toucher a votre travail."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from analyse.importation import import_result
from annotation_app.database import AnnotationDatabase


class ConsolidateTests(unittest.TestCase):
    def test_short_runs_join_their_longest_neighbour(self):
        from analyse.importation import consolidate
        segments = [
            {"crack": 0, "category": "open", "length_um": 300.0, "geometry": [[0, 0], [300, 0]]},
            {"crack": 0, "category": "filled", "length_um": 40.0, "geometry": [[300, 0], [340, 0]]},
            {"crack": 0, "category": "open", "length_um": 200.0, "geometry": [[340, 0], [540, 0]]},
            {"crack": 0, "category": "filled", "length_um": 400.0, "geometry": [[540, 0], [940, 0]]},
            {"crack": 1, "category": "reopened", "length_um": 30.0, "geometry": [[0, 50], [30, 50]]},
        ]
        merged = consolidate(segments, min_um=100)
        self.assertEqual([(s["crack"], s["category"]) for s in merged],
                         [(0, "open"), (0, "filled"), (1, "reopened")])   # une fissure seule reste entiere
        self.assertEqual(merged[0]["geometry"], [[0, 0], [300, 0], [340, 0], [540, 0]])
        self.assertAlmostEqual(merged[0]["length_um"], 540.0)


class ImportTests(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        root = Path(folder.name)
        (root / "m.tif").write_bytes(b"x")
        self.db = AnnotationDatabase(root / "base.sqlite3")
        self.db.initialize_image("m", root / "m.tif", 4000, 2000, 2000)       # 2 tuiles : A (x<2000), B
        zones = {z["col_index"]: z for z in self.db.list_zones("m")}
        self.a, self.b = zones[0], zones[1]
        # A : une ancienne proposition automatique jamais verifiee ; B : une fissure a vous, validee.
        self.db.add_detected_cracks_if_empty("m", self.a["id"], [
            {"anchors": [[100, 1500], [900, 1500]], "geometry": [[100, 1500], [900, 1500]]}], "auto")
        mine = self.db.create_crack("m", self.b["id"], [[2200, 300], [3800, 300]], [[2200, 300], [3800, 300]], "moi")
        self.mine = self.db.update_crack("m", mine["id"], 0, mine["anchors"], mine["geometry"], "open", "validated", "moi")

    def result(self, *segments):
        return {"segments": [{"category": c, "geometry": g} for c, g in segments]}

    def test_import_replaces_old_proposals_cuts_at_borders_and_skips_duplicates(self):
        summary = import_result(self.db, "m", 2000, 0.5, self.result(
            ("filled", [[500, 1000], [3500, 1000]]),        # traverse le bord A | B
            ("open", [[2200, 302], [3800, 302]]),           # double votre fissure de B
        ))
        self.assertEqual(summary["remplacees"], 1)
        self.assertEqual(summary["doublons_ignores"], 1)
        cracks = self.db.list_image_cracks("m")
        imported = [c for c in cracks if c["source"] == "analyse"]
        self.assertEqual(sorted(c["zone_id"] for c in imported), sorted([self.a["id"], self.b["id"]]))
        for crack in imported:
            self.assertEqual((crack["state"], crack["category"]), ("proposed", "filled"))
            zone = self.a if crack["zone_id"] == self.a["id"] else self.b
            self.assertTrue(all(zone["x"] <= x < zone["x"] + zone["width"] for x, _ in crack["geometry"]))
            self.assertGreaterEqual(len(crack["anchors"]), 2)
        self.assertIn(self.mine["id"], {c["id"] for c in cracks})            # votre fissure est intacte
        self.assertFalse(any(c["source"] == "auto" for c in cracks))         # l'ancienne proposition a disparu

    def test_validated_tiles_are_left_alone(self):
        self.db.update_zone("m", self.b["id"], "done", "moi", self.b["revision"])
        summary = import_result(self.db, "m", 2000, 0.5, self.result(("filled", [[2100, 1500], [3900, 1500]])))
        self.assertEqual(summary["importees"], 0)
        self.assertEqual(summary["tuiles_validees"], 1)

    def test_validating_a_tile_confirms_its_proposals_and_can_be_undone(self):
        import_result(self.db, "m", 2000, 0.5, self.result(("reopened", [[300, 800], [1700, 800]])))
        changed = self.db.confirm_zone_cracks("m", self.a["id"], "moi")
        self.assertEqual(len(changed), 1)
        crack = self.db.get_crack("m", changed[0])
        self.assertEqual((crack["state"], crack["category"]), ("validated", "reopened"))
        back = self.db.confirm_zone_cracks("m", self.a["id"], "moi", "validated", "proposed", set(changed))
        self.assertEqual(back, changed)
        self.assertEqual(self.db.get_crack("m", changed[0])["state"], "proposed")


if __name__ == "__main__":
    unittest.main()
