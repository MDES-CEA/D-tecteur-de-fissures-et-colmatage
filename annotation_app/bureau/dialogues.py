"""Fenetres secondaires : bilan, affichage, reentrainement du detecteur, aide."""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..paths import DATA_DIR, PROJECT_DIR

# Ne jamais nommer une methode comme un signal de la classe (`finished` pour
# un QDialog) : PySide6 n'appellerait jamais la methode branchee sous ce nom.


def _succeeded(code: int, status) -> bool:
    """Fin normale, code 0 ; un plantage (CrashExit) ou un arret est une erreur."""
    return status == QProcess.ExitStatus.NormalExit and code == 0


def _fill(progress: QProgressBar) -> None:
    """La barre au bout : la tache ne signale pas toujours sa derniere etape."""
    if progress.maximum() <= 0:
        progress.setRange(0, 1)
    progress.setValue(progress.maximum())
    progress.setFormat("Terminé")
    progress.setVisible(True)

LABELS = {
    "open": "Non colmatée",
    "filled": "Colmatée",
    "reopened": "Rouverte (gel, remplissage arraché)",
    "mixed": "Mixte",
    "uncertain": "Incertaine",
    "unclassified": "Non classée",
    "conflict": "Conflit (deux catégories superposées)",
}

SHORTCUTS = """\
<h3>Naviguer</h3>
<table cellpadding=3>
<tr><td><b>Molette</b></td><td>zoomer</td></tr>
<tr><td><b>Glisser</b></td><td>déplacer l'image</td></tr>
<tr><td><b>Ctrl+0</b></td><td>voir toute la mosaïque</td></tr>
<tr><td><b>Page préc. / Page suiv.</b></td><td>tuile précédente / suivante</td></tr>
<tr><td><b>Espace</b> (maintenu)</td><td>masquer les fissures</td></tr>
</table>
<h3>Tracer et corriger</h3>
<table cellpadding=3>
<tr><td><b>C</b></td><td>mode Tracer</td></tr>
<tr><td><b>V</b></td><td>mode Sélection</td></tr>
<tr><td><b>Échap</b></td><td>terminer le tracé</td></tr>
<tr><td><b>Z</b></td><td>retirer le dernier point</td></tr>
<tr><td><b>Maj+clic</b></td><td>ajouter un nœud sur la fissure sélectionnée</td></tr>
<tr><td><b>Glisser un nœud</b></td><td>le déplacer ; <b>clic droit</b> : le retirer</td></tr>
<tr><td><b>X</b></td><td>mode Couper : un clic coupe la fissure en deux</td></tr>
<tr><td><b>Suppr</b></td><td>supprimer la fissure sélectionnée</td></tr>
<tr><td><b>Ctrl+Z</b></td><td>annuler la dernière action</td></tr>
</table>
<h3>Classer</h3>
<table cellpadding=3>
<tr><td><b>1</b> ou <b>&amp;</b></td><td>Non colmatée — pas de gel</td></tr>
<tr><td><b>2</b> ou <b>é</b></td><td>Colmatée — gel et remplissage</td></tr>
<tr><td><b>4</b> ou <b>'</b></td><td>Rouverte — gel sur les parois, remplissage arraché</td></tr>
<tr><td><b>3</b> ou <b>"</b></td><td>Incertaine</td></tr>
<tr><td><b>5</b> ou <b>(</b></td><td>Mixte</td></tr>
<tr><td><b>Entrée</b></td><td>accepter la catégorie suggérée</td></tr>
<tr><td><b>Ctrl+Entrée</b></td><td>valider la tuile (confirme les propositions restantes)</td></tr>
</table>
<h3>Afficher et mesurer</h3>
<table cellpadding=3>
<tr><td><b>F</b> / <b>G</b></td><td>fissures / grille</td></tr>
<tr><td><b>Ctrl+B</b></td><td>bilan des longueurs</td></tr>
<tr><td><b>Ctrl+Maj+B</b></td><td>figures</td></tr>
</table>
"""


class ShortcutsDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Raccourcis clavier")
        layout = QVBoxLayout(self)
        text = QLabel(SHORTCUTS)
        text.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(text)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class LevelsDialog(QDialog):
    """Luminosite et contraste de l'affichage (l'image et les mesures ne changent pas)."""

    def __init__(self, brightness: int, contrast: int, apply, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Luminosité et contraste")
        self.apply = apply
        form = QFormLayout(self)
        self.brightness = self._slider(50, 200, brightness)
        self.contrast = self._slider(50, 300, contrast)
        form.addRow("Luminosité (%)", self.brightness)
        form.addRow("Contraste (%)", self.contrast)
        buttons = QDialogButtonBox()
        reset = buttons.addButton("Revenir à 100 %", QDialogButtonBox.ButtonRole.ResetRole)
        buttons.addButton(QDialogButtonBox.StandardButton.Close)
        reset.clicked.connect(self.reset)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _slider(self, low: int, high: int, value: int) -> QSlider:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(low, high)
        slider.setValue(value)
        slider.setMinimumWidth(260)
        slider.valueChanged.connect(lambda _: self.apply(self.brightness.value(), self.contrast.value()))
        return slider

    def reset(self) -> None:
        self.brightness.setValue(100)
        self.contrast.setValue(100)


class MeasuresDialog(QDialog):
    """Mesures : bilan des longueurs et figures, avec les memes options.

    Non modale : elle reste ouverte pendant qu'on annote, « Actualiser »
    relit le travail.
    """

    def __init__(self, window, tab: str = "bilan", parent=None) -> None:
        super().__init__(parent or window)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        from ..figures import KINDS

        self.main = window
        self.result = None
        self.preview = None
        self.extent = None
        self.generation = 0
        self.setWindowTitle("Mesures")
        self.resize(900, 760)
        layout = QVBoxLayout(self)

        options = QGridLayout()
        self.scope = QComboBox()
        self.scope.addItem("Toute la mosaïque", "image")
        self.scope.addItem("La tuile sélectionnée", "zone")
        self.source = QComboBox()
        self.source.addItem("Fissures de la mosaïque", "annotations")
        self.source.addItem("Axes bruts du détecteur (diagnostic)", "automatic")
        self.source.setToolTip("Tuiles aux calques calculés uniquement")
        self.validated = QCheckBox("Seulement les tuiles validées")
        self.tolerance = QDoubleSpinBox()
        self.tolerance.setRange(0, 5)
        self.tolerance.setSingleStep(0.5)
        self.tolerance.setValue(2)
        self.tolerance.setSuffix(" px")
        self.tolerance.setToolTip("Tracés parallèles plus proches : comptés une fois")
        options.addWidget(QLabel("Périmètre"), 0, 0)
        options.addWidget(self.scope, 0, 1)
        options.addWidget(QLabel("Mesurer"), 0, 2)
        options.addWidget(self.source, 0, 3)
        options.addWidget(self.validated, 1, 1)
        options.addWidget(QLabel("Superposition"), 1, 2)
        options.addWidget(self.tolerance, 1, 3)
        options.setColumnStretch(3, 1)
        layout.addLayout(options)

        self.tabs = QTabWidget()
        bilan = QWidget()
        inner = QVBoxLayout(bilan)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Catégorie", "Longueur", "Part"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        inner.addWidget(self.table)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        inner.addWidget(self.summary)
        export = QPushButton("Exporter les segments (CSV)…")
        export.clicked.connect(self.export_csv)
        row = QHBoxLayout()
        row.addWidget(export)
        row.addStretch()
        inner.addLayout(row)
        self.tabs.addTab(bilan, "Bilan")

        figures = QWidget()
        inner = QVBoxLayout(figures)
        row = QHBoxLayout()
        self.kind = QComboBox()
        for key, label in KINDS.items():
            self.kind.addItem(label, key)
        self.background = QCheckBox("Image en fond")
        self.background.setChecked(True)
        save = QPushButton("Enregistrer la figure…")
        save.clicked.connect(self.save_figure)
        row.addWidget(QLabel("Figure"))
        row.addWidget(self.kind)
        row.addWidget(self.background)
        row.addStretch()
        row.addWidget(save)
        inner.addLayout(row)
        self.figure = Figure(figsize=(8, 6), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        inner.addWidget(self.canvas, 1)
        self.tabs.addTab(figures, "Figures")
        layout.addWidget(self.tabs, 1)

        buttons = QHBoxLayout()
        self.state = QLabel("")
        refresh = QPushButton("Actualiser")
        close = QPushButton("Fermer")
        refresh.clicked.connect(self.refresh)
        close.clicked.connect(self.close)
        buttons.addWidget(self.state, 1)
        buttons.addWidget(refresh)
        buttons.addWidget(close)
        layout.addLayout(buttons)

        self.tabs.setCurrentIndex(1 if tab == "figures" else 0)
        for widget in (self.scope, self.source):
            widget.currentIndexChanged.connect(self.refresh)
        self.validated.toggled.connect(self.refresh)
        self.tolerance.valueChanged.connect(self.refresh)
        self.kind.currentIndexChanged.connect(self.refresh_figure)
        self.background.toggled.connect(self.refresh_figure)
        self.refresh()

    def show_tab(self, tab: str) -> None:
        self.tabs.setCurrentIndex(1 if tab == "figures" else 0)
        self.show()
        self.raise_()
        self.activateWindow()

    # -- Calcul -------------------------------------------------------------------

    def _extent(self, zone_id):
        session = self.main.session
        if zone_id is None:
            return (0, 0, session.width, session.height)
        zone = session.zone(zone_id)
        return (zone["x"], zone["y"], zone["x"] + zone["width"], zone["y"] + zone["height"])

    def refresh(self) -> None:
        zone = self.main.selected_zone_id if self.scope.currentData() == "zone" else None
        if self.scope.currentData() == "zone" and zone is None:
            self.state.setText("Aucune tuile sélectionnée.")
            return
        self.generation += 1
        generation = self.generation
        session = self.main.session
        validated, tolerance, source = self.validated.isChecked(), self.tolerance.value(), self.source.currentData()
        extent = self._extent(zone)
        self.state.setText("Calcul…")

        def compute():
            result = session.report(zone, validated, source, tolerance)
            preview = session.preview(*extent)
            return result, preview

        def done(values):
            if generation != self.generation:
                return                          # options changees entre-temps
            self.result, self.preview = values
            self.extent = extent
            self.state.setText("")
            self.show_table()
            self.refresh_figure()

        self.main.run(compute, done, "Calcul des mesures…", blocking=False)

    def _length(self, value: float) -> str:
        pixel = self.result.get("pixel_um")
        return f"{value * pixel / 1000:,.2f} mm".replace(",", " ") if pixel else f"{value:,.0f} px".replace(",", " ")

    def show_table(self) -> None:
        result = self.result
        total = result["total_px"] or 1.0
        rows = [(key, value) for key, value in result["length_px"].items() if value > 0]
        self.table.setRowCount(len(rows))
        for row, (key, value) in enumerate(rows):
            for column, text in enumerate([LABELS.get(key, key), self._length(value), f"{100 * value / total:.1f} %"]):
                self.table.setItem(row, column, QTableWidgetItem(text))
        self.table.resizeColumnsToContents()
        fraction = result.get("filled_fraction")
        parts = [f"Longueur totale : {self._length(result['total_px'])} sur {len(result['zones'])} tuile(s)."]
        if result["source"] in ("annotations", "complete"):   # les axes bruts n'ont pas de categorie
            parts.append(f"Part colmatée (colmatée + rouverte) : {100 * fraction:.1f} %."
                         if fraction is not None else "Aucune fissure classée.")
        if result["unprocessed_zone_ids"]:
            parts.append(f"{len(result['unprocessed_zone_ids'])} tuile(s) non calculée(s), exclue(s).")
        if result.get("overlap_removed_px"):
            parts.append(f"Superpositions retirées : {self._length(result['overlap_removed_px'])}.")
        if not result.get("pixel_um"):
            parts.append("Pixel inconnu : longueurs en px.")
        self.summary.setText(" ".join(parts))

    def refresh_figure(self) -> None:
        if self.result is None:
            return
        from ..figures import SPATIAL, draw
        kind = self.kind.currentData()
        preview = self.preview if kind in SPATIAL and self.background.isChecked() else None
        draw(self.figure, self.result, kind, self.extent, preview, self.tolerance.value())
        self.canvas.draw_idle()

    # -- Exports ------------------------------------------------------------------

    def save_figure(self) -> None:
        if self.result is None:
            return
        name = f"figure-{self.kind.currentData()}.png"
        path, _ = QFileDialog.getSaveFileName(self, "Enregistrer la figure", name,
                                              "Image PNG (*.png);;Vectoriel SVG (*.svg);;PDF (*.pdf)")
        if path:
            self.figure.savefig(path, dpi=200)
            self.main.status(f"Figure enregistrée : {path}")

    def export_csv(self) -> None:
        if not self.result:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Exporter les segments", "segments.csv", "CSV (*.csv)")
        if not path:
            return
        pixel = self.result.get("pixel_um")
        with open(path, "w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream, delimiter=";")
            writer.writerow(["categorie", "longueur_px", "longueur_um", "x1", "y1", "x2", "y2", "source"])
            for segment in self.result["segments"]:
                (x1, y1), (x2, y2) = segment["geometry"]
                length = segment["length_px"]
                writer.writerow([segment["category"], f"{length:.2f}",
                                 f"{length * pixel:.2f}" if pixel else "", f"{x1:.1f}", f"{y1:.1f}",
                                 f"{x2:.1f}", f"{y2:.1f}", self.result["source"]])
        self.main.status(f"Segments exportés : {path}")


class TrainingDialog(QDialog):
    """Reentrainement du detecteur de fissures, suivi en direct.

    Lance `outils/train_crack_unet.py` tel quel ; le detecteur en service n'est
    remplace qu'a la toute fin, et seulement si le nouveau fait mieux.

    `database` : une base forcee, sinon None, et la base se deduit de l'image
    choisie dans la fenetre (data/bases/).
    """

    STEPS_PER_MINUTE = 40

    def __init__(self, image: Path | None, database: Path | None = None, parent=None,
                 log_dir: Path | None = None) -> None:
        super().__init__(parent)
        self.database = Path(database) if database else None
        self.log_dir = Path(log_dir) if log_dir else DATA_DIR / "entrainements"
        self.process = None
        self.stopping = False
        self.log_file = None
        self.setWindowTitle("Réentraîner le détecteur de fissures")
        self.resize(820, 600)
        layout = QVBoxLayout(self)

        form = QFormLayout()
        row = QHBoxLayout()
        self.image = QLineEdit(str(image) if image else "")
        browse = QPushButton("Parcourir…")
        browse.clicked.connect(self.browse)
        row.addWidget(self.image)
        row.addWidget(browse)
        form.addRow("Mosaïque", row)
        self.tiles = QLabel("")
        form.addRow("", self.tiles)
        self.image.textChanged.connect(self.update_tiles)

        steps_row = QHBoxLayout()
        self.steps = QSpinBox()
        self.steps.setRange(100, 10000)
        self.steps.setSingleStep(100)
        self.steps.setValue(800)
        self.duration = QLabel("")
        self.steps.valueChanged.connect(self.update_duration)
        steps_row.addWidget(self.steps)
        steps_row.addWidget(self.duration)
        steps_row.addStretch()
        form.addRow("Nombre de pas", steps_row)

        self.fine = QRadioButton("1,6 µm/pixel — fissures fines, détection 4× plus lente")
        self.coarse = QRadioButton("3,2 µm/pixel — plus rapide")
        self.fine.setChecked(True)
        resolution = QVBoxLayout()
        resolution.addWidget(self.fine)
        resolution.addWidget(self.coarse)
        form.addRow("Résolution d'analyse", resolution)
        self.from_current = QCheckBox("Partir du détecteur actuel")
        self.from_current.setChecked(True)
        self.install = QCheckBox("Installer s'il fait mieux (l'ancien est archivé)")
        self.install.setChecked(True)
        form.addRow("", self.from_current)
        form.addRow("", self.install)
        layout.addLayout(form)

        buttons = QHBoxLayout()
        self.start_button = QPushButton("Lancer")
        self.stop_button = QPushButton("Arrêter")
        self.stop_button.setEnabled(False)
        close = QPushButton("Fermer")
        self.state = QLabel("")
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        close.clicked.connect(self.close)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.state, 1)
        buttons.addWidget(close)
        layout.addLayout(buttons)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Consolas", 9))
        layout.addWidget(self.log, 1)
        self.update_tiles()
        self.update_duration()

    def browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choisir la mosaïque", self.image.text(),
                                              "Images TIFF (*.tif *.tiff);;Tous les fichiers (*.*)")
        if path:
            self.image.setText(str(Path(path)))

    def count_tiles(self) -> tuple[int | None, str]:
        image = Path(self.image.text().strip())
        if not image.is_file():
            return None, "Fichier introuvable."
        try:
            from ..bases import find_database
            from ..database import AnnotationDatabase
            path = self.database or find_database(image, log=lambda _: None)
            if path is None:
                return 0, "Mosaïque jamais ouverte dans l'application."
            database = AnnotationDatabase(path)
            image_id = database.image_id_for_path(image)
            if image_id is None:
                return 0, "Aucune tuile validée."
            count = sum(zone["status"] == "done" for zone in database.list_zones(image_id))
        except Exception as error:
            return None, f"Base illisible : {error}"
        if count < 3:
            return count, f"{count} tuile(s) validée(s) — minimum 3."
        return count, f"{count} tuiles validées" + ("." if count >= 10 else " — 10 conseillées.")

    def update_tiles(self) -> None:
        self.tiles.setText(self.count_tiles()[1])

    def update_duration(self) -> None:
        self.duration.setText(f"environ {max(1, round(self.steps.value() / self.STEPS_PER_MINUTE))} min")

    def write(self, text: str) -> None:
        self.log.moveCursor(self.log.textCursor().MoveOperation.End)
        self.log.insertPlainText(text)
        self.log.ensureCursorVisible()
        if self.log_file:
            self.log_file.write(text)

    def start(self) -> None:
        count, message = self.count_tiles()
        if count is None or count < 3:
            QMessageBox.warning(self, "Entraînement impossible", message)
            return
        from ..echelle import REFERENCE_ANALYSIS_UM, REFERENCE_PIXEL_UM
        resolution = REFERENCE_PIXEL_UM if self.fine.isChecked() else REFERENCE_ANALYSIS_UM
        arguments = ["-u", str(PROJECT_DIR / "outils" / "train_crack_unet.py"), "--image", self.image.text().strip(),
                     "--steps", str(self.steps.value()), "--resolution-analyse", f"{resolution:.4f}"]
        if self.database is not None:          # sinon l'outil trouve la base de l'image lui-meme
            arguments += ["--database", str(self.database)]
        # Lignes de test de l'analyse jamais apprises ; traits rejetes appris comme « pas une fissure ».
        arguments += ["--decoupage", "--negatifs"]
        if self.install.isChecked():
            arguments.append("--installer")
        if not self.from_current.isChecked():
            arguments.append("--de-zero")
        folder = self.log_dir
        folder.mkdir(parents=True, exist_ok=True)
        log_path = folder / f"detecteur-{datetime.now():%Y%m%d-%H%M}.log"
        self.log_file = log_path.open("w", encoding="utf-8")
        self.log.clear()
        self.write(f"Journal : {log_path}\n\n")

        self.process = QProcess(self)
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONIOENCODING", "utf-8")
        self.process.setProcessEnvironment(environment)
        self.process.setWorkingDirectory(str(PROJECT_DIR))
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._process_finished)
        self.stopping = False
        self.process.start(sys.executable, arguments)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.state.setText("Entraînement en cours…")

    def _read_output(self) -> None:
        if self.process is not None:
            self.write(bytes(self.process.readAllStandardOutput()).decode("utf-8", "replace"))

    def _process_finished(self, code: int, status) -> None:
        self._read_output()                   # les dernieres lignes, s'il en reste
        if self.log_file:
            self.log_file.close()
            self.log_file = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.process = None
        if _succeeded(code, status):
            self.state.setText("Terminé. Si le détecteur a été remplacé (voir le journal), rouvrir la mosaïque.")
        elif self.stopping:
            self.state.setText("Arrêté. Le détecteur en service n'a pas changé.")
        else:
            self.state.setText(f"Erreur (code {code}) : voir le journal.")
        QApplication.alert(self)

    def stop(self) -> None:
        if self.process is not None:
            # Le detecteur en service n'est remplace qu'a la fin : rien a craindre.
            self.stopping = True
            self.process.kill()
            self.state.setText("Arrêt demandé…")

    def closeEvent(self, event) -> None:
        if self.process is not None:
            if QMessageBox.question(self, "Entraînement en cours", "Arrêter l'entraînement et fermer ?") \
                    != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.process.kill()
            self.process.waitForFinished(3000)
        super().closeEvent(event)

class ProcessDialog(QDialog):
    """Une longue tache dans un autre processus (analyse complete, apprentissage), suivie en direct."""

    def __init__(self, title: str, arguments: list[str], log_name: str, intro: str,
                 progress_patterns=(r"\[detection (\d+)/(\d+)\]", r"epoque\s+(\d+)/(\d+)",
                                    r"\[colmatage\] fissure (\d+)/(\d+)"), on_success=None, parent=None,
                 options=(), log_dir: Path | None = None) -> None:
        """`options` : [(libelle, argument ajoute si coche, coche par defaut)].

        `on_success` n'est appele que si la tache se termine normalement, en code 0.
        """
        super().__init__(parent)
        import re
        self.patterns = [re.compile(p) for p in progress_patterns]
        self.arguments = arguments
        self.log_name = log_name
        self.on_success = on_success
        self.log_dir = Path(log_dir) if log_dir else DATA_DIR / "entrainements"
        self.process = None
        self.stopping = False
        self.log_file = None
        self.setWindowTitle(title)
        self.resize(820, 560)
        layout = QVBoxLayout(self)
        text = QLabel(intro)
        text.setWordWrap(True)
        layout.addWidget(text)
        self.options = []
        for label, flag, checked in options:
            box = QCheckBox(label)
            box.setChecked(checked)
            layout.addWidget(box)
            self.options.append((box, flag))
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        buttons = QHBoxLayout()
        self.start_button = QPushButton("Lancer")
        self.stop_button = QPushButton("Arrêter")
        self.stop_button.setEnabled(False)
        close = QPushButton("Fermer")
        self.state = QLabel("")
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        close.clicked.connect(self.close)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.state, 1)
        buttons.addWidget(close)
        layout.addLayout(buttons)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Consolas", 9))
        layout.addWidget(self.log, 1)

    def write(self, text: str) -> None:
        self.log.moveCursor(self.log.textCursor().MoveOperation.End)
        self.log.insertPlainText(text)
        self.log.ensureCursorVisible()
        if self.log_file:
            self.log_file.write(text)
            self.log_file.flush()
        for line in text.splitlines():
            for pattern in self.patterns:
                found = pattern.search(line)
                if found:
                    done, total = int(found.group(1)), int(found.group(2))
                    self.progress.setVisible(True)
                    self.progress.setMaximum(total)
                    self.progress.setValue(done)
                    self.progress.setFormat(f"{line.split(']')[0].strip('[ ')} — %v/%m" if "]" in line
                                            else f"%v/%m")

    def start(self) -> None:
        folder = self.log_dir
        folder.mkdir(parents=True, exist_ok=True)
        log_path = folder / f"{self.log_name}-{datetime.now():%Y%m%d-%H%M}.log"
        self.log_file = log_path.open("w", encoding="utf-8")
        self.log.clear()
        self.progress.setVisible(False)
        self.progress.setRange(0, 0)
        self.progress.setFormat("%v/%m")
        self.write(f"Journal : {log_path}\n\n")
        self.stopping = False
        self.process = QProcess(self)
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONIOENCODING", "utf-8")
        self.process.setProcessEnvironment(environment)
        self.process.setWorkingDirectory(str(PROJECT_DIR))
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._process_finished)
        extra = [flag for box, flag in self.options if box.isChecked()]
        self.process.start(sys.executable, self.arguments + extra)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.state.setText("En cours…")

    def _read_output(self) -> None:
        if self.process is not None:
            self.write(bytes(self.process.readAllStandardOutput()).decode("utf-8", "replace"))

    def _process_finished(self, code: int, status) -> None:
        self._read_output()                   # les dernieres lignes, s'il en reste
        if self.log_file:
            self.log_file.close()
            self.log_file = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.process = None
        if _succeeded(code, status):
            _fill(self.progress)
            self.state.setText(f"Terminé à {datetime.now():%H:%M}.")
            if self.on_success is not None:
                self.on_success()
        elif self.stopping:
            self.state.setText("Arrêté. Relancer reprend où ça s'était arrêté.")
        else:
            self.state.setText(f"Erreur (code {code}) : voir le journal. Relancer reprend où ça s'était arrêté.")
        QApplication.alert(self)              # la tache dure longtemps : signaler sa fin

    def stop(self) -> None:
        if self.process is not None:
            self.stopping = True
            self.process.kill()
            self.state.setText("Arrêt demandé…")

    def closeEvent(self, event) -> None:
        if self.process is not None:
            if QMessageBox.question(self, "Tâche en cours", "Arrêter et fermer ?") \
                    != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.process.kill()
            self.process.waitForFinished(3000)
        super().closeEvent(event)


class AuditDialog(QDialog):
    """Sondage : un point tiré au hasard sur les fissures détectées, une touche par point.

    La réponse de la machine n'est pas montrée : elle influencerait la vôtre.
    """

    KEYS = {Qt.Key.Key_1: "open", Qt.Key.Key_Ampersand: "open",
            Qt.Key.Key_2: "filled", Qt.Key.Key_Eacute: "filled",
            Qt.Key.Key_4: "reopened", Qt.Key.Key_Apostrophe: "reopened",
            Qt.Key.Key_0: "not_crack", Qt.Key.Key_Agrave: "not_crack"}
    FIELD_UM = 100.0

    def __init__(self, audit, source, pixel_um: float, parent=None) -> None:
        super().__init__(parent)
        self.audit = audit
        self.source = source
        self.pixel_um = pixel_um
        self.history = []
        self.index = audit.next_index()
        self.setWindowTitle("Contrôle de l'analyse par sondage")
        layout = QVBoxLayout(self)
        self.image = QLabel()
        self.image.setFixedSize(480, 480)
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image, alignment=Qt.AlignmentFlag.AlignCenter)
        keys = QLabel("<b>1</b> / <b>&amp;</b> non colmatée · <b>2</b> / <b>é</b> colmatée · "
                      "<b>4</b> / <b>'</b> rouverte · <b>0</b> / <b>à</b> pas une fissure · "
                      "<b>Retour arrière</b> point précédent")
        layout.addWidget(keys)
        self.progress = QLabel("")
        self.estimate = QLabel("")
        self.estimate.setWordWrap(True)
        layout.addWidget(self.progress)
        layout.addWidget(self.estimate)
        close = QPushButton("Fermer (les réponses sont gardées)")
        close.clicked.connect(self.close)
        close.setAutoDefault(False)
        layout.addWidget(close)
        self.show_point()

    def show_point(self) -> None:
        from PySide6.QtGui import QImage, QPainter, QPen, QPixmap, QColor
        import numpy as np

        answered = len(self.audit.data["reponses"])
        total = len(self.audit.points)
        self.progress.setText(f"Point {min(answered + 1, total)} / {total} — {answered} réponse(s)")
        e = self.audit.estimate()
        if e.get("fissures_controlees"):
            low, high = e["part_colmatee_corrigee_ic95"]
            self.estimate.setText(
                f"Part colmatée : machine {100 * e['part_colmatee_machine']:.1f} %, corrigée "
                f"<b>{100 * e['part_colmatee_corrigee']:.1f} %</b> (IC 95 % {100 * low:.1f}–{100 * high:.1f} %) · "
                f"accord oui/non {100 * e['accord_colmatee_oui_non']:.0f} % · pas une fissure "
                f"{100 * e['pas_une_fissure']:.0f} %")
        if self.index is None:
            self.image.setText("Sondage terminé. Le rapport sera mis à jour au prochain\n"
                               "« Automatique > Ouvrir le rapport ».")
            return
        point = self.audit.points[self.index]
        half = int(round(self.FIELD_UM / 2 / self.pixel_um))
        x0 = int(max(0, min(self.source.width - 2 * half, point["x"] - half)))
        y0 = int(max(0, min(self.source.height - 2 * half, point["y"] - half)))
        crop = np.ascontiguousarray(np.asarray(self.source.read(x0, y0, 2 * half, 2 * half), dtype=np.uint8))
        image = QImage(crop.data, crop.shape[1], crop.shape[0], crop.shape[1],
                       QImage.Format.Format_Grayscale8).copy()
        pixmap = QPixmap.fromImage(image).scaled(480, 480, Qt.AspectRatioMode.KeepAspectRatio,
                                                 Qt.TransformationMode.SmoothTransformation)
        painter = QPainter(pixmap)
        painter.setPen(QPen(QColor("#ffd400"), 2))
        scale = pixmap.width() / (2 * half)
        cx, cy = (point["x"] - x0) * scale, (point["y"] - y0) * scale
        painter.drawEllipse(int(cx - 14), int(cy - 14), 28, 28)
        painter.end()
        self.image.setPixmap(pixmap)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Backspace and self.history:
            self.index = self.history.pop()
            self.audit.answer(self.index, None)
            self.show_point()
            return
        answer = self.KEYS.get(event.key())
        if answer is None or self.index is None:
            super().keyPressEvent(event)
            return
        self.audit.answer(self.index, answer)
        self.history.append(self.index)
        self.index = self.audit.next_index()
        self.show_point()
