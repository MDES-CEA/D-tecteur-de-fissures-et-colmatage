"""Fenetre de bureau, hors ecran : tracer, classer, annuler, valider, mesurer."""

import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
import tifffile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from PySide6.QtCore import QPointF, QSettings, Qt
    from PySide6.QtWidgets import QApplication
except ImportError:              # environnement sans Qt : rien a tester ici
    QApplication = None


@unittest.skipIf(QApplication is None, "PySide6 absent")
class DesktopWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        folder = TemporaryDirectory(ignore_cleanup_errors=True)    # Windows : image encore ouverte
        self.addCleanup(folder.cleanup)
        root = Path(folder.name)
        QSettings.setDefaultFormat(QSettings.Format.IniFormat)
        QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(root / "reglages"))
        patcher = mock.patch.dict(os.environ, {
            "CRACK_CACHE_DIR": str(root / "cache"), "CRACK_BACKUP_DIR": str(root / "sauvegardes"),
            "CRACK_COLMATAGE_MODEL": str(root / "colmatage.joblib"), "CRACK_PIXEL_UM": "1.595"})
        patcher.start()
        self.addCleanup(patcher.stop)

        # Du bruit, sinon le reseau ne voit rien ; une fissure droite pour le
        # trace manuel, et une ondulee pour la detection, qui ecarte les traits
        # parfaitement droits (rayures de polissage).
        pixels = (150 + np.random.default_rng(0).normal(0, 12, (2500, 3000))).clip(0, 255).astype(np.uint8)
        pixels[1000:1004, 100:1900] = 40
        xs = np.arange(200, 1800)
        ys = (500 + 60 * np.sin(xs / 300)).astype(int)
        for dy in range(-2, 3):
            pixels[ys + dy, xs] = 40
        image = root / "mosaique.tif"
        tifffile.imwrite(image, pixels, tile=(512, 512))

        from annotation_app.bureau.fenetre import MainWindow
        from annotation_app.session import Session
        self.session = Session(image, root / "base.sqlite3", log=lambda text: None)
        self.window = MainWindow(self.session)
        self.window.resize(1200, 800)
        self.window.show()
        self.addCleanup(self.window.close)

    def wait(self, condition, seconds=60):
        end = time.time() + seconds
        while time.time() < end:
            self.app.processEvents()
            if condition():
                return
            time.sleep(0.01)
        self.fail(f"delai depasse : {self.window.message.text()}")

    def idle(self):
        self.wait(lambda: not self.window.busy and not self.window._tasks)

    def click(self, x, y, modifiers=Qt.KeyboardModifier.NoModifier):
        self.window.on_click(QPointF(x, y), Qt.MouseButton.LeftButton, modifiers)

    def test_trace_classify_undo_and_validate(self):
        window = self.window
        self.assertEqual(len(window.zones), 4)             # tuiles de 2000 px a cette echelle
        window.set_mode("draw")
        self.click(200, 1002)
        self.assertIsNotNone(window.pending)
        self.click(900, 1002)
        self.idle()
        crack = window.selected_crack()
        self.assertEqual(crack["id"], window.draft_id)
        self.click(1700, 1002)                             # le trace se prolonge
        self.idle()
        crack = window.selected_crack()
        self.assertEqual(len(crack["anchors"]), 3)
        self.assertGreater(crack["length_px"], 1450)
        # Un point dans une autre tuile est refuse, sans rien casser.
        self.click(2500, 1002)
        self.assertIn("même tuile", window.message.text())

        window.categorize("reopened")
        self.idle()
        crack = window.selected_crack()
        self.assertEqual((crack["category"], crack["state"]), ("reopened", "validated"))
        self.assertIsNone(window.draft_id)
        window.undo()
        self.idle()
        self.assertEqual(window.selected_crack()["category"], "unclassified")

        window.categorize("filled")
        self.idle()
        window.set_zone_status("done")
        self.idle()
        self.assertEqual(window.zones_by_id[crack["zone_id"]]["status"], "done")
        result = self.session.report(validated_only=True)
        self.assertAlmostEqual(result["length_px"]["filled"], crack["length_px"], delta=2)

    def test_cut_a_crack_and_undo(self):
        window = self.window
        window.set_mode("draw")
        self.click(200, 1002)
        self.click(1700, 1002)
        self.idle()
        window.finish_draft()
        original = window.selected_crack()
        window.set_mode("cut")
        self.click(900, 1010)                     # a cote du trace : coupe au point le plus proche
        self.idle()
        first, second = sorted((c for c in window.cracks), key=lambda c: c["geometry"][0][0])
        self.assertEqual(first["id"], original["id"])
        self.assertAlmostEqual(first["geometry"][-1][0], 900, delta=2)
        self.assertEqual(first["geometry"][-1], second["geometry"][0])
        self.assertAlmostEqual(first["length_px"] + second["length_px"], original["length_px"], delta=1)
        window.set_mode("select")
        window.select_crack(second["id"])
        window.categorize("open")                 # chaque morceau se classe a part
        self.idle()
        self.assertEqual(window.cracks_by_id[first["id"]]["category"], "unclassified")

        window.undo()                             # annule le classement
        self.idle()
        window.undo()                             # annule la coupe : le trace d'origine, a l'identique
        self.idle()
        self.assertEqual([c["id"] for c in window.cracks], [original["id"]])
        self.assertEqual(window.cracks[0]["geometry"], original["geometry"])

    def test_audit_dialog_records_answers_and_corrects(self):
        from PySide6.QtCore import QEvent
        from PySide6.QtGui import QKeyEvent

        from analyse.audit import Audit
        from annotation_app.bureau.dialogues import AuditDialog

        root = Path(self.session.cache_dir).parent
        result = {"analyse_le": "essai", "resume": {"part_colmatee": 0.0},
                  "segments": [{"category": "open", "length_um": 500.0, "geometry": [[300, 1002], [1600, 1002]]}]}
        audit = Audit(root / "audit.json", result, count=6)
        dialog = AuditDialog(audit, self.session.tiles.source, 0.3712)

        def press(key):
            dialog.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier))

        press(Qt.Key.Key_Eacute)                    # colmatee (AZERTY)
        press(Qt.Key.Key_1)                         # non colmatee
        press(Qt.Key.Key_Backspace)                 # on revient sur la derniere
        press(Qt.Key.Key_Agrave)                    # pas une fissure
        self.assertEqual(sorted(audit.data["reponses"].values()), ["filled", "not_crack"])
        reloaded = Audit(root / "audit.json", result)          # garde sur disque
        estimate = reloaded.estimate()
        self.assertEqual(estimate["points"], 2)
        self.assertAlmostEqual(estimate["part_colmatee_corrigee"], 1.0)   # la machine disait 0 %
        self.assertIn("Point 3 / 6", dialog.progress.text())
        dialog.close()

    def test_layers_figures_and_exports(self):
        window, session = self.window, self.session
        zone = window.zones[0]
        window.select_zone(zone["id"])
        self.assertFalse(session.layer_computed(zone))
        window.set_layer("score")                     # le calque de la tuile se calcule tout seul
        self.wait(lambda: not window.poll_layers() and session.layer_computed(zone), 300)
        self.wait(lambda: len(window.view.overlay_items) == 1, 60)
        for kind in ("mask", "skeleton", "centerline", "cost", "route_cost"):
            window.set_layer(kind)
            self.idle()
            self.assertEqual(len(window.view.overlay_items), 1, kind)
        self.assertTrue(window.layer_actions["route_cost"].isChecked())
        window.act_layer_all.setChecked(True)
        window.refresh_layers()
        self.idle()
        self.assertEqual(len(window.view.overlay_items), 1)              # une seule tuile calculee
        window.set_layer(None)
        self.idle()
        self.assertEqual(window.view.overlay_items, [])

        # Axes automatiques : mesures sur les seules tuiles calculees, les autres signalees.
        automatic = session.report(source="automatic")
        self.assertEqual(len(automatic["unprocessed_zone_ids"]), 3)
        self.assertGreater(automatic["total_px"], 1000)                   # la fissure synthetique

        root = Path(session.cache_dir).parent
        self.assertEqual(session.export_layers(root / "calques.zip", window.zones), 1)

        window.open_measures("figures")
        dialog = window.measures
        dialog.source.setCurrentIndex(dialog.source.findData("automatic"))
        self.wait(lambda: dialog.result is not None and dialog.result["source"] == "automatic")
        for index in range(dialog.kind.count()):
            dialog.kind.setCurrentIndex(index)
            for extension in ("png", "svg", "pdf"):
                path = root / f"figure-{index}.{extension}"
                dialog.figure.savefig(path, dpi=60)
                self.assertGreater(path.stat().st_size, 1000)
        self.assertIn("3 tuile(s) non calculée(s)", dialog.summary.text())
        dialog.close()

    def test_delete_and_tile_navigation(self):
        window = self.window
        window.set_mode("draw")
        self.click(200, 1002)
        self.click(900, 1002)
        self.idle()
        window.finish_draft()
        self.assertEqual(window.mode, "select")
        crack_id = window.selected_crack_id
        window.delete_selected()
        self.idle()
        self.assertNotIn(crack_id, window.cracks_by_id)
        window.undo()                                      # annuler une suppression la retablit
        self.idle()
        self.assertIn(crack_id, window.cracks_by_id)

        window.selected_zone_id = None
        window.navigate(1)
        self.assertEqual(window.selected_zone_id, window.zones_order[0])
        window.navigate(1)
        self.assertEqual(window.selected_zone_id, window.zones_order[1])


if __name__ == "__main__":
    unittest.main()
