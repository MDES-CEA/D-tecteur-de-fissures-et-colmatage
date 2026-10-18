from __future__ import annotations

import importlib.util
import math
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from PIL import Image

import numpy as np

from annotation_app.database import AnnotationDatabase, CrackConflict, ZoneConflict
from annotation_app.detection_service import (
    CrackDetector,
    _drop_parallel_branches,
    _looks_like_tile_seam,
    _maximum_score_forest,
    _skeletonize_mask,
    _stitch_fragments,
    _supervised_thresholds,
    _threshold_mask,
)
from annotation_app.paths import last_image, reference_image, remember_image
from annotation_app.tile_service import PyramidTileService, TileNotFound


PROJECT_DIR = Path(__file__).resolve().parent.parent
# La mosaïque de référence est désignée sur chaque poste, hors du dépôt.
MOSAIC = reference_image()


class TileServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if MOSAIC is None:
            raise unittest.SkipTest("aucune mosaique de reference sur ce poste")
        cls.tiles = PyramidTileService(MOSAIC)

    def test_real_mosaic_metadata(self):
        self.assertGreater(min(self.tiles.width, self.tiles.height), 0)
        self.assertTrue(self.tiles.source.direct)
        self.assertEqual(
            self.tiles.max_level,
            math.ceil(math.log2(max(self.tiles.width, self.tiles.height))))

    def test_low_and_full_resolution_tiles_are_valid_png(self):
        for level, x, y in ((0, 0, 0), (16, 0, 0), (16, 27, 100)):
            with Image.open(BytesIO(self.tiles.tile_png(level, x, y))) as tile:
                self.assertEqual(tile.format, "PNG")
                self.assertLessEqual(tile.width, self.tiles.tile_size)
                self.assertLessEqual(tile.height, self.tiles.tile_size)

    def test_rejects_tile_outside_image(self):
        with self.assertRaises(TileNotFound):
            self.tiles.tile_png(16, 999, 999)


class LastImageTests(unittest.TestCase):
    def test_the_last_image_is_remembered_while_it_exists(self):
        with TemporaryDirectory() as folder:
            record = Path(folder) / "data" / "derniere_image.txt"
            self.assertIsNone(last_image(record))
            image = Path(folder) / "dossier accentué" / "mosaïque.tif"
            image.parent.mkdir()
            image.write_bytes(b"x")
            remember_image(image, record)
            self.assertEqual(last_image(record), image.resolve())
            image.unlink()
            self.assertIsNone(last_image(record))

    def test_the_reference_image_is_designated_outside_the_code(self):
        with TemporaryDirectory() as folder:
            record = Path(folder) / "image_reference.txt"
            self.assertIsNone(reference_image(record))
            image = Path(folder) / "reference.tif"
            image.write_bytes(b"x")
            record.write_text(f"  {image}\n", encoding="utf-8")
            self.assertEqual(reference_image(record), image)


class ZoneDatabaseTests(unittest.TestCase):
    def setUp(self):
        # Jamais dans annotation_app/data/ : c'est l'etat reel du poste.
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.database = AnnotationDatabase(Path(folder.name) / "test_annotations.sqlite3")
        self.database.initialize_image(
            "test-image", PROJECT_DIR / "fake.tif", 4500, 2100, 2000
        )

    def test_an_image_is_recognised_by_its_path(self):
        """Ouvrir une mosaique connue doit retrouver son travail.

        C'est ce qui permet de la designer par son fichier, sans avoir a se
        souvenir d'un identifiant — et sans risquer de rattacher ses traces a
        une autre image.
        """
        self.assertEqual(
            self.database.image_id_for_path(PROJECT_DIR / "fake.tif"), "test-image")
        self.assertIsNone(
            self.database.image_id_for_path(PROJECT_DIR / "jamais-vue.tif"))

    def test_the_same_file_cannot_be_registered_twice(self):
        """Sinon la contrainte d'unicite du chemin remonte telle quelle."""
        with self.assertRaises(ValueError) as raised:
            self.database.initialize_image(
                "autre-nom", PROJECT_DIR / "fake.tif", 4500, 2100, 2000)
        self.assertIn("test-image", str(raised.exception))

    def _register(self, path: Path, width: int = 4000, height: int = 4000) -> str:
        image_id, _ = self.database.resolve_image_id(path, width, height)
        self.database.initialize_image(image_id, path, width, height, 2000)
        return image_id

    def test_two_images_with_the_same_name_never_share_traces(self):
        """Deux « RECALAGE.tif » de deux echantillons gardent chacun ses traces.

        Avant, l'identifiant ne dependait que du nom : la seconde image
        reprenait les traces de la premiere, qui disparaissait de la base.
        """
        with TemporaryDirectory() as folder:
            first = Path(folder) / "A" / "RECALAGE.tif"
            second = Path(folder) / "B" / "RECALAGE.tif"
            for path in (first, second):
                path.parent.mkdir()
                path.write_bytes(b"x")
            first_id = self._register(first)
            zone = self.database.list_zones(first_id)[0]
            self.database.create_crack(
                first_id, zone["id"], [[1, 1], [9, 9]], [[1, 1], [9, 9]], "test")

            second_id = self._register(second)

            self.assertNotEqual(first_id, second_id)
            self.assertEqual(self.database.list_image_cracks(second_id), [])
            self.assertEqual(len(self.database.list_image_cracks(first_id)), 1)
            self.assertEqual(self.database.image_id_for_path(first), first_id)

    def test_a_moved_file_keeps_its_annotations(self):
        with TemporaryDirectory() as folder:
            before = Path(folder) / "avant" / "mosaique.tif"
            after = Path(folder) / "apres" / "mosaique renommee.tif"
            before.parent.mkdir()
            before.write_bytes(b"x")
            image_id = self._register(before, 6000, 2000)
            after.parent.mkdir()
            before.replace(after)

            moved_id, note = self.database.resolve_image_id(after, 6000, 2000)
            self.assertEqual(moved_id, image_id)
            self.assertIn("deplacee", note)
            self.database.initialize_image(moved_id, after, 6000, 2000, 2000)
            self.assertEqual(self.database.image_id_for_path(after), image_id)

    def test_a_vanished_file_is_not_taken_over_by_other_dimensions(self):
        with TemporaryDirectory() as folder:
            other = Path(folder) / "autre.tif"
            other.write_bytes(b"x")
            # « test-image » a disparu (fake.tif n'existe pas) mais ne mesure
            # pas 4000 x 4000 : ce n'est pas elle.
            image_id, _ = self.database.resolve_image_id(other, 4000, 4000)
            self.assertNotEqual(image_id, "test-image")
            with self.assertRaises(ValueError):
                self.database.initialize_image("test-image", other, 4000, 4000, 2000)

    def test_a_forced_identifier_cannot_take_another_images_traces(self):
        """`CRACK_IMAGE_ID` ne permet pas non plus de rattacher deux images."""
        with TemporaryDirectory() as folder:
            first = Path(folder) / "premiere.tif"
            second = Path(folder) / "seconde.tif"
            for path in (first, second):
                path.write_bytes(b"x")
            first_id = self._register(first)
            with self.assertRaises(ValueError) as raised:
                self.database.initialize_image(first_id, second, 4000, 4000, 2000)
            self.assertIn(str(first), str(raised.exception))
            self.assertEqual(self.database.image_id_for_path(first), first_id)

    def test_grid_and_optimistic_revision(self):
        zones = self.database.list_zones("test-image")
        self.assertEqual(len(zones), 6)
        first = zones[0]
        updated = self.database.update_zone(
            "test-image", first["id"], "in_progress", "Alice", first["revision"]
        )
        self.assertEqual(updated["revision"], 1)
        self.assertEqual(updated["assigned_to"], "Alice")
        with self.assertRaises(ZoneConflict):
            self.database.update_zone(
                "test-image", first["id"], "done", "Bob", first["revision"]
            )

    def test_cracks_are_versioned_and_detection_never_overwrites_work(self):
        zone = self.database.list_zones("test-image")[0]
        proposal = [{
            "network_index": 2,
            "anchors": [[10, 20], [30, 40]],
            "geometry": [[10, 20], [20, 30], [30, 40]],
        }]
        cracks, created = self.database.add_detected_cracks_if_empty(
            "test-image", zone["id"], proposal, "Alice"
        )
        self.assertTrue(created)
        self.assertEqual(len(cracks), 1)
        self.assertEqual(cracks[0]["state"], "proposed")

        ignored = [{
            "network_index": 9,
            "anchors": [[1, 1], [2, 2]],
            "geometry": [[1, 1], [2, 2]],
        }]
        existing, created_again = self.database.add_detected_cracks_if_empty(
            "test-image", zone["id"], ignored, "Bob"
        )
        self.assertFalse(created_again)
        self.assertEqual(existing[0]["network_index"], 2)

        updated = self.database.update_crack(
            "test-image",
            cracks[0]["id"],
            cracks[0]["revision"],
            cracks[0]["anchors"],
            cracks[0]["geometry"],
            "filled",
            "validated",
            "Alice",
        )
        self.assertEqual(updated["category"], "filled")
        self.assertEqual(updated["revision"], 1)
        with self.assertRaises(CrackConflict):
            self.database.update_crack(
                "test-image",
                cracks[0]["id"],
                0,
                cracks[0]["anchors"],
                cracks[0]["geometry"],
                "open",
                "validated",
                "Bob",
            )

    def test_manual_crack_stays_in_progress_until_it_is_classified(self):
        zone = self.database.list_zones("test-image")[0]
        crack = self.database.create_crack(
            "test-image",
            zone["id"],
            [[10, 20], [30, 40]],
            [[10, 20], [20, 30], [30, 40]],
            "Alice",
        )
        self.assertEqual(crack["category"], "unclassified")
        self.assertEqual(crack["state"], "proposed")

    def test_length_statistics_group_active_cracks_by_category(self):
        zone = self.database.list_zones("test-image")[0]
        open_crack = self.database.create_crack(
            "test-image", zone["id"], [[0, 0], [3, 4]], [[0, 0], [3, 4]], "Alice"
        )
        open_crack = self.database.update_crack(
            "test-image",
            open_crack["id"],
            open_crack["revision"],
            open_crack["anchors"],
            open_crack["geometry"],
            "open",
            "validated",
            "Alice",
        )
        rejected = self.database.create_crack(
            "test-image", zone["id"], [[0, 0], [0, 100]], [[0, 0], [0, 100]], "Alice"
        )
        self.database.update_crack(
            "test-image",
            rejected["id"],
            rejected["revision"],
            rejected["anchors"],
            rejected["geometry"],
            "filled",
            "rejected",
            "Alice",
        )
        unclassified = self.database.create_crack(
            "test-image", zone["id"], [[0, 0], [0, 2]], [[0, 0], [0, 2]], "Alice"
        )

        stats = self.database.length_statistics("test-image", zone["id"])
        self.assertEqual(stats["length_px"]["open"], 5.0)
        self.assertEqual(stats["length_px"]["filled"], 0.0)
        self.assertEqual(stats["length_px"]["unclassified"], 2.0)
        self.assertEqual(stats["length_px"]["total"], 7.0)
        self.assertEqual(stats["crack_count"]["total"], 2)
        visible_ids = {
            crack["id"] for crack in self.database.list_image_cracks("test-image")
        }
        self.assertEqual(visible_ids, {open_crack["id"], unclassified["id"]})

    def test_detection_can_be_retried_after_all_proposals_are_rejected(self):
        zone = self.database.list_zones("test-image")[0]
        first = [{
            "network_index": 1,
            "anchors": [[10, 20], [30, 40]],
            "geometry": [[10, 20], [30, 40]],
        }]
        cracks, created = self.database.add_detected_cracks_if_empty(
            "test-image", zone["id"], first, "Alice"
        )
        self.assertTrue(created)
        rejected = self.database.update_crack(
            "test-image",
            cracks[0]["id"],
            cracks[0]["revision"],
            cracks[0]["anchors"],
            cracks[0]["geometry"],
            cracks[0]["category"],
            "rejected",
            "Alice",
        )
        self.assertEqual(rejected["state"], "rejected")

        second = [{
            "network_index": 9,
            "anchors": [[50, 60], [70, 80]],
            "geometry": [[50, 60], [70, 80]],
        }]
        retried, retried_created = self.database.add_detected_cracks_if_empty(
            "test-image", zone["id"], second, "Bob"
        )
        self.assertTrue(retried_created)
        self.assertEqual(len(retried), 1)
        self.assertEqual(retried[0]["network_index"], 9)


class CenterlineTests(unittest.TestCase):
    def test_analysis_maps_are_the_exact_pipeline_arrays(self):
        detector = CrackDetector.__new__(CrackDetector)
        detector.analyze = lambda _zone: SimpleNamespace(
            cost=np.asarray([[1.0, 0.25], [0.5, 0.0]], dtype=np.float32),
            score=np.asarray([[0.0, 0.25], [0.5, 1.0]], dtype=np.float32),
            mask=np.asarray([[False, True], [True, False]], dtype=bool),
            skeleton=np.asarray([[False, False], [True, False]], dtype=bool),
        )

        expected = {
            "cost": [[255, 64], [128, 0]],
            "score": [[0, 64], [128, 255]],
            "mask": [[0, 255], [255, 0]],
            "skeleton": [[0, 0], [255, 0]],
        }
        for kind, pixels in expected.items():
            with self.subTest(kind=kind):
                content = detector.analysis_map_png({"id": 1}, kind)
                with Image.open(BytesIO(content)) as image:
                    self.assertEqual(image.format, "PNG")
                    self.assertEqual(image.mode, "L")
                    self.assertEqual(image.size, (2, 2))
                    np.testing.assert_array_equal(np.asarray(image), pixels)

    def test_analysis_map_rejects_unknown_kind(self):
        detector = CrackDetector.__new__(CrackDetector)
        detector.analyze = lambda _zone: SimpleNamespace(
            cost=np.zeros((1, 1)),
            score=np.zeros((1, 1)),
            mask=np.zeros((1, 1), dtype=bool),
            skeleton=np.zeros((1, 1), dtype=bool),
        )
        with self.assertRaises(ValueError):
            detector.analysis_map_png({"id": 1}, "unknown")

    def test_routed_geometry_keeps_every_user_anchor(self):
        detector = CrackDetector.__new__(CrackDetector)
        route_map = SimpleNamespace(scale=1, cost=np.ones((100, 100), dtype=float))
        detector.prepare_route = lambda _zone: route_map
        zone = {"id": 1, "x": 0, "y": 0, "width": 100, "height": 100}
        anchors = [[10.25, 10.5], [50.75, 20.25], [80.5, 80.75]]
        geometry = detector.route(zone, anchors)
        for anchor in anchors:
            self.assertIn(anchor, geometry)

    def test_parallel_branch_is_removed(self):
        score = np.ones((40, 80), dtype=float)
        first = np.asarray([(10, col) for col in range(5, 70)], dtype=int)
        second = np.asarray([(14, col) for col in range(5, 70)], dtype=int)
        branches = [
            {"pixels": first, "start": tuple(first[0]), "end": tuple(first[-1])},
            {"pixels": second, "start": tuple(second[0]), "end": tuple(second[-1])},
        ]
        kept = _drop_parallel_branches(branches, score, radius=6)
        self.assertEqual(len(kept), 1)

    def test_cycle_is_reduced_to_a_forest(self):
        score = np.ones((30, 30), dtype=float)
        branches = []
        for start, end in (
            ((5, 5), (5, 20)),
            ((5, 20), (20, 20)),
            ((20, 20), (20, 5)),
            ((20, 5), (5, 5)),
        ):
            pixels = np.asarray([start, end], dtype=int)
            branches.append({"pixels": pixels, "start": start, "end": end})
        self.assertEqual(len(_maximum_score_forest(branches, score)), 3)

    def test_low_cost_fragments_are_stitched_once(self):
        cost = np.ones((80, 160), dtype=np.float32)
        cost[39:42, 5:155] = 0.01
        left = np.asarray([(40, col) for col in range(5, 50)], dtype=int)
        right = np.asarray([(40, col) for col in range(90, 150)], dtype=int)
        branches = [
            {"pixels": left, "start": tuple(left[0]), "end": tuple(left[-1])},
            {"pixels": right, "start": tuple(right[0]), "end": tuple(right[-1])},
        ]
        stitched, accepted = _stitch_fragments(branches, cost, scale=2)
        self.assertEqual(accepted, 1)
        self.assertEqual(len(stitched), 1)
        self.assertLessEqual(np.ptp(stitched[0]["pixels"][:, 0]), 2)

    def test_a_real_branching_is_not_swallowed_by_the_main_path(self):
        """Le chemin principal ne traverse un reseau ramifie qu'une seule fois.

        Les branches qui le longent sont des doublons et doivent disparaitre ;
        celles qui s'en ecartent sont de vraies ramifications. Les jeter toutes
        coutait 1,9 point de rappel.
        """
        cost = np.ones((160, 160), dtype=np.float32)
        cost[19:22, 5:156] = 0.01
        cost[20:121, 79:82] = 0.01
        pieces = [
            np.asarray([(20, col) for col in range(5, 61)], dtype=int),
            np.asarray([(20, col) for col in range(100, 156)], dtype=int),
            np.asarray([(row, 80) for row in range(20, 121)], dtype=int),
        ]
        branches = [
            {"pixels": piece, "start": tuple(piece[0]), "end": tuple(piece[-1])}
            for piece in pieces
        ]
        stitched, _ = _stitch_fragments(branches, cost, scale=2)
        self.assertEqual(len(stitched), 2)
        spans = sorted(int(np.ptp(branch["pixels"][:, 0])) for branch in stitched)
        self.assertLessEqual(spans[0], 2)      # le chemin principal, horizontal
        self.assertGreaterEqual(spans[1], 90)  # la ramification, verticale


class MaskThresholdTests(unittest.TestCase):
    """Le masque doit garder le fin et le long, pas seulement le gros."""

    def test_thin_oblique_fragment_survives_the_area_filter(self):
        # 12 px obliques : sous le seuil d'aire, au-dessus du seuil d'etendue.
        score = np.zeros((80, 80), dtype=np.float32)
        steps = np.arange(20, 32)
        score[steps, steps] = 0.9
        mask = _threshold_mask(score, scale=4, low=0.455, high=0.7)
        self.assertTrue(mask[steps, steps].all())

    def test_compact_speck_is_still_removed(self):
        score = np.zeros((80, 80), dtype=np.float32)
        score[40:43, 40:43] = 0.9
        mask = _threshold_mask(score, scale=4, low=0.455, high=0.7)
        self.assertFalse(mask.any())

    def test_confident_tile_keeps_the_model_threshold(self):
        score = np.zeros((100, 100), dtype=np.float32)
        score[:, :20] = 0.95
        low, high = _supervised_thresholds(score, 0.70)
        self.assertAlmostEqual(high, 0.70)
        self.assertAlmostEqual(low, 0.70 * 0.65)

    def test_faint_tile_lowers_the_threshold_instead_of_yielding_nothing(self):
        score = np.full((100, 100), 0.05, dtype=np.float32)
        score[50, 10:90] = 0.55
        low, high = _supervised_thresholds(score, 0.70)
        self.assertLess(high, 0.70)
        self.assertLess(low, high)
        self.assertTrue((score >= high).any())

    def test_object_thresholds_follow_the_analysis_scale(self):
        """Un objet de taille physique donnee doit survivre a toute echelle.

        Les seuils sont en pixels pleine resolution. Exprimes en pixels
        d'analyse, ils changeraient de sens physique d'une echelle a l'autre :
        c'est ce qui faisait passer la detection a l'echelle 2 pour mauvaise
        alors qu'elle etait seulement quatre fois plus permissive.
        """
        for scale in (2, 4):
            with self.subTest(scale=scale):
                extent = int(round(48 / scale))
                score = np.zeros((120, 120), dtype=np.float32)
                steps = np.arange(20, 20 + extent)
                score[steps, steps] = 0.9
                mask = _threshold_mask(score, scale=scale, low=0.455, high=0.7)
                self.assertTrue(mask[steps, steps].all())

    def test_threshold_never_falls_below_the_floor(self):
        low, high = _supervised_thresholds(np.zeros((50, 50), dtype=np.float32), 0.70)
        self.assertAlmostEqual(high, 0.30)


class SkeletonCleanupTests(unittest.TestCase):
    """Le nettoyage du squelette doit traiter obliques et axes a egalite.

    Une regression ici est invisible : la chaine continue de tourner et rend
    des propositions, mais uniquement sur les structures alignees sur les axes,
    donc sur les artefacts d'acquisition plutot que sur les fissures.
    """

    @staticmethod
    def _ribbon(points) -> np.ndarray:
        mask = np.zeros((160, 160), dtype=bool)
        for row, col in points:
            mask[row - 1:row + 2, col - 1:col + 2] = True
        return mask

    def test_diagonal_crack_survives_the_cleanup(self):
        steps = np.arange(20, 140)
        skeleton = _skeletonize_mask(self._ribbon(zip(steps, steps)), scale=4)
        self.assertGreater(skeleton.sum(), 80)

    def test_curved_crack_survives_the_cleanup(self):
        columns = np.arange(20, 140)
        rows = np.rint(80 + 40 * np.sin(columns / 30)).astype(int)
        skeleton = _skeletonize_mask(self._ribbon(zip(rows, columns)), scale=4)
        self.assertGreater(skeleton.sum(), 80)

    def test_oblique_and_horizontal_cracks_are_kept_alike(self):
        columns = np.arange(20, 140)
        horizontal = _skeletonize_mask(
            self._ribbon(zip(np.full_like(columns, 80), columns)), scale=4
        ).sum()
        diagonal = _skeletonize_mask(
            self._ribbon(zip(np.arange(20, 140), columns)), scale=4
        ).sum()
        self.assertGreater(diagonal, 0.5 * horizontal)

    def test_isolated_speck_is_still_removed(self):
        mask = np.zeros((160, 160), dtype=bool)
        mask[80:83, 80:83] = True
        self.assertEqual(_skeletonize_mask(mask, scale=4).sum(), 0)

    def test_skeleton_threshold_is_the_same_length_at_every_scale(self):
        for scale in (2, 4):
            with self.subTest(scale=scale):
                long_enough = np.arange(20, 20 + int(round(160 / scale)))
                too_short = np.arange(20, 20 + int(round(40 / scale)))
                self.assertGreater(
                    _skeletonize_mask(
                        self._ribbon(zip(long_enough, long_enough)), scale
                    ).sum(),
                    0,
                )
                self.assertEqual(
                    _skeletonize_mask(
                        self._ribbon(zip(too_short, too_short)), scale
                    ).sum(),
                    0,
                )


class UNetTests(unittest.TestCase):
    """Garde-fous sur le branchement du reseau, pas sur sa qualite."""

    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("torch") is None:
            raise unittest.SkipTest("torch absent")

    def test_inputs_are_scaled_to_the_unit_range(self):
        from annotation_app.unet_model import prepare_inputs

        rng = np.random.default_rng(0)
        gray = (rng.integers(40, 90, size=(64, 48))).astype(np.uint8)
        ridge = rng.random((64, 48)).astype(np.float32)
        inputs = prepare_inputs(gray, ridge)
        self.assertEqual(inputs.shape, (2, 64, 48))
        self.assertGreaterEqual(inputs.min(), 0.0)
        self.assertLessEqual(inputs[0].max(), 1.0)

    def test_a_tile_of_any_size_comes_back_at_its_own_size(self):
        """Le reseau descend trois fois : les cotes non multiples de 8 doivent
        etre rembourres puis recoupes, sans decaler la carte."""
        from annotation_app.unet_model import build_network, unet_score

        network = build_network()
        network.eval()
        for shape in ((32, 32), (30, 45), (17, 8)):
            with self.subTest(shape=shape):
                gray = np.full(shape, 60, dtype=np.uint8)
                gray[0, 0] = 10
                ridge = np.zeros(shape, dtype=np.float32)
                score = unet_score(network, gray, ridge)
                self.assertEqual(score.shape, shape)
                self.assertTrue(np.all((score >= 0.0) & (score <= 1.0)))

    def test_the_learned_analysis_resolution_is_read_back(self):
        """Sinon un reseau appris a 1,6 um serait utilise a 3,2 um."""
        import torch

        from annotation_app.unet_model import build_network, load_unet
        with TemporaryDirectory() as folder:
            path = Path(folder) / "reseau.pt"
            torch.save({"state_dict": build_network(in_channels=2, base=16).state_dict(),
                        "in_channels": 2, "base": 16, "threshold": 0.9, "analysis_um": 1.595}, path)
            self.assertEqual(load_unet(path)["analysis_um"], 1.595)

    def test_a_checkpoint_is_recognised_as_a_unet_artifact(self):
        import torch

        from annotation_app.detector_model import load_detector_model
        from annotation_app.unet_model import build_network

        with TemporaryDirectory() as folder:
            path = Path(folder) / "network.pt"
            torch.save(
                {
                    "state_dict": build_network().state_dict(),
                    "in_channels": 2,
                    "base": 16,
                    "threshold": 0.42,
                    "max_analysis_side": 1000,
                },
                path,
            )
            artifact = load_detector_model(path)
        self.assertEqual(artifact["kind"], "unet")
        self.assertAlmostEqual(artifact["threshold"], 0.42)


class SeamFilterTests(unittest.TestCase):
    def test_rejects_slightly_warped_tile_seam(self):
        columns = np.arange(100)
        rows = np.rint(40 + 2.5 * np.sin(columns / 18)).astype(int)
        pixels = np.column_stack([rows, columns])
        self.assertTrue(_looks_like_tile_seam({"pixels": pixels}))

    def test_keeps_diagonal_crack(self):
        pixels = np.column_stack([np.arange(100), np.arange(100)])
        self.assertFalse(_looks_like_tile_seam({"pixels": pixels}))


if __name__ == "__main__":
    unittest.main()

