"""Une longue tache dans un autre processus annonce sa fin (hors ecran).

Regression : la methode branchee sur `QProcess.finished` s'appelait
`finished`, comme le signal de QDialog ; PySide6 ne l'appelait jamais, et
l'analyse complete restait « En cours… » pour toujours.
"""

import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from PySide6.QtWidgets import QApplication
except ImportError:              # environnement sans Qt : rien a tester ici
    QApplication = None


@unittest.skipIf(QApplication is None, "PySide6 absent")
class ProcessDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        folder = TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(folder.cleanup)
        self.logs = Path(folder.name)          # jamais dans annotation_app/data/
        self.calls = []

    def run_dialog(self, code: int):
        from annotation_app.bureau.dialogues import ProcessDialog
        script = f"import sys; print('[detection 3/7] une ligne', flush=True); sys.exit({code})"
        dialog = ProcessDialog("Essai", ["-c", script], "essai", "Essai.",
                               on_success=lambda: self.calls.append("fin"), log_dir=self.logs)
        self.addCleanup(dialog.close)
        dialog.start()
        self.assertIsNotNone(dialog.process)
        end = time.time() + 60
        while dialog.process is not None and time.time() < end:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertIsNone(dialog.process, "la fin du processus n'a jamais ete vue")
        return dialog

    def test_success_calls_back_and_says_so(self):
        dialog = self.run_dialog(0)
        self.assertEqual(self.calls, ["fin"])
        self.assertIn("Terminé", dialog.state.text())
        self.assertEqual(dialog.progress.value(), dialog.progress.maximum())   # 3/7 -> au bout
        self.assertTrue(dialog.start_button.isEnabled())
        self.assertFalse(dialog.stop_button.isEnabled())
        self.assertIsNone(dialog.log_file)                   # journal ferme
        journal = next(self.logs.glob("essai-*.log")).read_text(encoding="utf-8")
        self.assertIn("une ligne", journal)

    def test_failure_does_not_call_back(self):
        dialog = self.run_dialog(1)
        self.assertEqual(self.calls, [])
        self.assertIn("Erreur", dialog.state.text())
        self.assertTrue(dialog.start_button.isEnabled())
        self.assertIsNone(dialog.log_file)


if __name__ == "__main__":
    unittest.main()
