"""Fenetre principale de l'application d'annotation des fissures (PySide6)."""

from __future__ import annotations

import math
import traceback
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, QRunnable, QSettings, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..session import Session, SessionError
from .calques import BINARY, HELP, LAYERS, colorize, reduce
from .dialogues import (LABELS, AuditDialog, LevelsDialog, MeasuresDialog, ProcessDialog, ShortcutsDialog,
                        TrainingDialog)
from .vue import MosaicView

OVERVIEW_SIDE = 256        # cote maximal d'un calque de tuile vu en ensemble
LAYER_OPACITY = 0.7

CATEGORY_KEYS = [  # (categorie, libelle, touche chiffre, touche AZERTY, rappel ; « && » affiche « & »)
    ("open", "Non colmatée", Qt.Key.Key_1, Qt.Key.Key_Ampersand, "1 / &&"),
    ("filled", "Colmatée", Qt.Key.Key_2, Qt.Key.Key_Eacute, "2 / é"),
    ("reopened", "Rouverte", Qt.Key.Key_4, Qt.Key.Key_Apostrophe, "4 / '"),
    ("uncertain", "Incertaine", Qt.Key.Key_3, Qt.Key.Key_QuoteDbl, '3 / "'),
    ("mixed", "Mixte", Qt.Key.Key_5, Qt.Key.Key_ParenLeft, "5 / ("),
]
STATUS_LABELS = {"todo": "non validée", "in_progress": "en cours", "review": "à relire", "done": "validée"}


class _TaskSignals(QObject):
    done = Signal(object)
    failed = Signal(str)


class _Task(QRunnable):
    def __init__(self, function) -> None:
        super().__init__()
        self.function = function
        self.signals = _TaskSignals()

    def run(self) -> None:
        try:
            result = self.function()
        except SessionError as error:
            self.signals.failed.emit(str(error))
        except Exception as error:  # remonte a l'annotateur, sans fermer l'application
            traceback.print_exc()
            self.signals.failed.emit(f"{type(error).__name__} : {error}")
        else:
            self.signals.done.emit(result)


def _distance_to_polyline(x: float, y: float, points) -> float:
    best = math.inf
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        dx, dy = x2 - x1, y2 - y1
        length = dx * dx + dy * dy
        t = 0.0 if not length else max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / length))
        best = min(best, math.hypot(x - x1 - t * dx, y - y1 - t * dy))
    return best


class MainWindow(QMainWindow):
    def __init__(self, session: Session) -> None:
        super().__init__()
        self.settings = QSettings()        # organisation et application : voir __main__
        self.edits = QThreadPool(self)
        self.edits.setMaxThreadCount(1)          # les modifications, dans l'ordre
        self.background = QThreadPool(self)
        self.background.setMaxThreadCount(2)     # preparation des tuiles, calques
        # L'affichage (calques, mesures) a part : derriere des preparations de
        # tuiles (une quinzaine de secondes chacune), il attendait sans fin.
        self.display_pool = QThreadPool(self)
        self.display_pool.setMaxThreadCount(2)
        self._tasks: set = set()
        self.busy = False
        self.session = None
        self.view = MosaicView(self)
        self.view.clicked.connect(self.on_click)
        self.view.hovered.connect(self.on_hover)
        self.view.zoomChanged.connect(lambda zoom: self.zoom_label.setText(f"zoom {zoom:.3g}×"))
        self.view.installEventFilter(self)
        self.setCentralWidget(self.view)
        self._build_actions()
        self._build_menus()
        self._build_toolbar()
        self._build_panel()
        self._build_statusbar()
        self.auto_timer = QTimer(self, interval=1000)
        self.auto_timer.timeout.connect(self.poll)
        self.layers_running = False
        self.layers_completed = 0
        self.layer_cache: dict = {}
        self.layer_generation = 0
        self.measures = None
        geometry = self.settings.value("geometrie")
        if geometry is not None:
            self.restoreGeometry(geometry)
        else:
            self.resize(1500, 950)
        self.open_session(session)

    # =================================================================== Construction

    def _action(self, text: str, slot, shortcut=None, tip: str = "", checkable: bool = False) -> QAction:
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        if tip:
            action.setStatusTip(tip)
            action.setToolTip(tip)
        action.setCheckable(checkable)
        action.triggered.connect(slot)
        self.addAction(action)
        return action

    def _build_actions(self) -> None:
        a = self._action
        self.act_open = a("Ouvrir une mosaïque…", self.choose_mosaic, "Ctrl+O")
        self.act_annotator = a("Nom de l'annotateur…", self.ask_annotator)
        self.act_export_base = a("Exporter la base de cette mosaïque…", self.export_base,
                                 tip="Un fichier .sqlite3 à donner tel quel")
        self.act_import_base = a("Importer une base…", self.import_base)
        self.act_quit = a("Quitter", self.close, "Ctrl+Q")

        self.act_undo = a("Annuler", self.undo, "Ctrl+Z", "Annuler la dernière modification")
        self.act_finish = a("Terminer le tracé", self.finish_draft, "Esc")
        self.act_remove_point = a("Retirer le dernier point", self.remove_last_point, "Z")
        self.act_delete = a("Supprimer la fissure", self.delete_selected, "Del")

        self.modes = QActionGroup(self)
        self.act_select = a("Sélection", lambda: self.set_mode("select"), "V", checkable=True)
        self.act_draw = a("Tracer", lambda: self.set_mode("draw"), "C", checkable=True)
        self.act_insert = a("Ajouter des nœuds", lambda: self.set_mode("insert"), "I", checkable=True)
        self.act_cut = a("Couper une fissure", lambda: self.set_mode("cut"), "X", checkable=True,
                         tip="Un clic sur une fissure la coupe en deux, à classer séparément")
        for action in (self.act_select, self.act_draw, self.act_insert, self.act_cut):
            self.modes.addAction(action)
        self.act_select.setChecked(True)

        self.category_actions = {}
        for category, label, _, _, keys in CATEGORY_KEYS:
            action = QAction(f"{label}\t{keys}", self)
            action.triggered.connect(lambda _=False, c=category: self.categorize(c))
            self.category_actions[category] = action
        self.act_accept = a("Accepter la catégorie suggérée\tEntrée", self.accept_suggestion)

        self.act_previous = a("Tuile précédente", lambda: self.navigate(-1), "PgUp")
        self.act_next = a("Tuile suivante", lambda: self.navigate(1), "PgDown")
        self.act_center = a("Centrer sur la tuile", self.center_zone, "T")
        self.act_validate = a("Valider la tuile", lambda: self.set_zone_status("done"), "Ctrl+Return")
        self.act_unvalidate = a("Remettre la tuile « non validée »", lambda: self.set_zone_status("todo"))
        self.act_accept_zone = a("Accepter tous les classements de la tuile", lambda: self.change_zone_cracks("accept"),
                                 "Shift+Return", "Chaque fissure garde sa catégorie (ou prend celle suggérée) et "
                                 "devient validée ; la tuile, elle, n'est pas validée")
        self.act_delete_zone = a("Supprimer toutes les fissures de la tuile", lambda: self.change_zone_cracks("reject"),
                                 "Ctrl+Del", "Ctrl+Z pour annuler")
        self.act_grid = a("Afficher la grille des tuiles", self.refresh_zones, "G", checkable=True)
        self.act_grid.setChecked(self.settings.value("grille", "false") == "true")

        self.act_full_analysis = a("Analyser la mosaïque…", self.full_analysis,
                                   tip="Calcule toute la mosaïque ; le résultat arrive en propositions à corriger")
        self.act_report = a("Ouvrir le rapport", self.open_report)
        self.act_audit = a("Contrôler l'analyse (sondage)…", self.audit,
                           tip="200 points au hasard, une touche par point : corrige la part colmatée")
        self.act_train = a("Réentraîner le classement du colmatage…", self.train_colmatage)
        self.act_train_detector = a("Réentraîner le détecteur de fissures…", self.train_detector)

        self.act_fit = a("Tout voir", self.view.fit_all, "Ctrl+0")
        self.act_zoom_in = a("Zoom avant", lambda: self.view.zoom_by(1.5), "Ctrl++")
        self.act_zoom_out = a("Zoom arrière", lambda: self.view.zoom_by(1 / 1.5), "Ctrl+-")
        self.act_show_cracks = a("Afficher les fissures", self.toggle_annotations, "F", checkable=True)
        self.act_show_cracks.setChecked(self.settings.value("afficher_annotations", "true") == "true")
        self.act_levels = a("Luminosité et contraste…", self.edit_levels)

        # Calques du detecteur : un diagnostic, calcule tout seul pour la tuile affichee.
        self.current_layer = None
        self.layer_group = QActionGroup(self)
        self.layer_actions = {}
        for kind, label in [(None, "Aucun calque"), *LAYERS.items()]:
            action = a(label, lambda _=False, k=kind: self.set_layer(k), tip=HELP.get(kind, ""), checkable=True)
            self.layer_group.addAction(action)
            self.layer_actions[kind] = action
        self.layer_actions[None].setChecked(True)
        self.act_layer_all = a("Sur toute la mosaïque (tuiles déjà calculées)", lambda: self.refresh_layers(),
                               checkable=True)
        self.act_export_layers = a("Exporter les calques calculés (ZIP)…", self.export_layers)

        self.act_lengths = a("Bilan des longueurs…", lambda: self.open_measures("bilan"), "Ctrl+B")
        self.act_figures = a("Figures…", lambda: self.open_measures("figures"), "Ctrl+Shift+B")
        self.act_shortcuts = a("Raccourcis clavier", lambda: ShortcutsDialog(self).exec(), "F1")

    def _build_menus(self) -> None:
        bar = self.menuBar()
        menu = bar.addMenu("&Fichier")
        menu.addActions([self.act_open, self.act_annotator])
        menu.addSeparator()
        menu.addActions([self.act_export_base, self.act_import_base])
        menu.addSeparator()
        menu.addAction(self.act_quit)

        menu = bar.addMenu("&Édition")
        menu.addAction(self.act_undo)
        menu.addSeparator()
        menu.addActions([self.act_finish, self.act_remove_point, self.act_delete])

        menu = bar.addMenu("&Outils")
        menu.addActions([self.act_select, self.act_draw, self.act_insert, self.act_cut])

        menu = bar.addMenu("&Classer")
        menu.addActions(list(self.category_actions.values()))
        menu.addSeparator()
        menu.addAction(self.act_accept)
        menu.addAction(self.act_delete)

        menu = bar.addMenu("&Tuile")
        menu.addActions([self.act_previous, self.act_next, self.act_center])
        menu.addSeparator()
        menu.addActions([self.act_validate, self.act_unvalidate])
        menu.addSeparator()
        menu.addActions([self.act_accept_zone, self.act_delete_zone])

        menu = bar.addMenu("&Automatique")
        menu.addActions([self.act_full_analysis, self.act_report, self.act_audit])
        menu.addSeparator()
        menu.addActions([self.act_train, self.act_train_detector])

        menu = bar.addMenu("&Affichage")
        menu.addActions([self.act_fit, self.act_zoom_in, self.act_zoom_out])
        menu.addSeparator()
        menu.addActions([self.act_show_cracks, self.act_grid, self.act_levels])
        layers = menu.addMenu("Calques du détecteur (diagnostic)")
        layers.addActions(list(self.layer_actions.values()))
        layers.addSeparator()
        layers.addActions([self.act_layer_all, self.act_export_layers])
        self.view_menu = menu

        menu = bar.addMenu("&Mesures")
        menu.addActions([self.act_lengths, self.act_figures])

        menu = bar.addMenu("&Aide")
        menu.addAction(self.act_shortcuts)

    def _build_toolbar(self) -> None:
        bar = self.addToolBar("Outils principaux")
        bar.setObjectName("outils")
        bar.setMovable(False)
        bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.act_select.setIconText("Sélection (V)")
        self.act_draw.setIconText("Tracer (C)")
        self.act_cut.setIconText("Couper (X)")
        self.act_previous.setIconText("◀ Tuile")
        self.act_next.setIconText("Tuile ▶")
        self.act_validate.setIconText("Valider la tuile")
        self.act_fit.setIconText("Tout voir")
        self.act_undo.setIconText("Annuler")
        bar.addActions([self.act_select, self.act_draw, self.act_cut])
        bar.addSeparator()
        bar.addActions([self.act_previous, self.act_next, self.act_validate])
        bar.addSeparator()
        bar.addActions([self.act_undo, self.act_fit])

    def _build_panel(self) -> None:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        box = QGroupBox("Tuile")
        inner = QVBoxLayout(box)
        self.zone_label = QLabel("Aucune tuile sélectionnée")
        self.zone_label.setWordWrap(True)
        inner.addWidget(self.zone_label)
        row = QHBoxLayout()
        for action in (self.act_validate, self.act_unvalidate):
            button = QPushButton(action.text().split("\t")[0])
            button.clicked.connect(action.trigger)
            row.addWidget(button)
        inner.addLayout(row)
        layout.addWidget(box)

        box = QGroupBox("Fissure sélectionnée")
        inner = QVBoxLayout(box)
        self.crack_label = QLabel("Aucune fissure sélectionnée")
        self.crack_label.setWordWrap(True)
        self.suggestion_label = QLabel("")
        self.suggestion_label.setWordWrap(True)
        inner.addWidget(self.crack_label)
        inner.addWidget(self.suggestion_label)
        grid = QGridLayout()
        self.category_buttons = {}
        for index, (category, label, _, _, keys) in enumerate(CATEGORY_KEYS):
            button = QPushButton(f"{label}  ({keys})")
            button.setToolTip(LABELS.get(category, label))
            button.clicked.connect(lambda _=False, c=category: self.categorize(c))
            grid.addWidget(button, index // 2, index % 2)
            self.category_buttons[category] = button
        inner.addLayout(grid)
        row = QHBoxLayout()
        cut = QPushButton("Couper (X)")
        cut.clicked.connect(self.act_cut.trigger)
        delete = QPushButton("Supprimer (Suppr)")
        delete.clicked.connect(self.delete_selected)
        row.addWidget(cut)
        row.addWidget(delete)
        inner.addLayout(row)
        layout.addWidget(box)

        box = QGroupBox("Analyse automatique")
        inner = QVBoxLayout(box)
        self.analysis_label = QLabel("")
        self.analysis_label.setWordWrap(True)
        inner.addWidget(self.analysis_label)
        row = QHBoxLayout()
        analyse = QPushButton("Analyser la mosaïque…")
        analyse.clicked.connect(self.full_analysis)
        report = QPushButton("Rapport")
        report.clicked.connect(self.open_report)
        row.addWidget(analyse, 2)
        row.addWidget(report, 1)
        inner.addLayout(row)
        layout.addWidget(box)

        box = QGroupBox("Annotateur")
        inner = QVBoxLayout(box)
        self.annotator_edit = QLineEdit(self.settings.value("annotateur", ""))
        self.annotator_edit.setPlaceholderText("Votre nom")
        self.annotator_edit.editingFinished.connect(
            lambda: self.settings.setValue("annotateur", self.annotator_edit.text().strip()))
        inner.addWidget(self.annotator_edit)
        layout.addWidget(box)
        layout.addStretch()

        self.panel_layout = layout
        self.dock = dock = QDockWidget("Panneau", self)
        dock.setObjectName("panneau")
        self.panel_scroll = scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        dock.setWidget(scroll)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable |
                         QDockWidget.DockWidgetFeature.DockWidgetClosable)
        dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea |
                             Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea)
        dock.dockLocationChanged.connect(self._panel_moved)
        bottom = self.settings.value("panneau_bas", "false") == "true"
        self.act_panel_bottom = self._action("Panneau en bas (compact)", self._toggle_panel_bottom,
                                             checkable=True)
        self.act_panel_bottom.setChecked(bottom)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea if bottom
                           else Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self._panel_moved(self.dockWidgetArea(dock))
        self.view_menu.addActions([self.act_panel_bottom, dock.toggleViewAction()])

    def _toggle_panel_bottom(self) -> None:
        area = (Qt.DockWidgetArea.BottomDockWidgetArea if self.act_panel_bottom.isChecked()
                else Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(area, self.dock)
        self._panel_moved(area)

    def _panel_moved(self, area) -> None:
        """En haut ou en bas, les blocs du panneau se rangent cote a cote, en une bande compacte."""
        from PySide6.QtWidgets import QBoxLayout
        horizontal = area in (Qt.DockWidgetArea.BottomDockWidgetArea, Qt.DockWidgetArea.TopDockWidgetArea)
        self.panel_layout.setDirection(QBoxLayout.Direction.LeftToRight if horizontal
                                       else QBoxLayout.Direction.TopToBottom)
        self.panel_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded if horizontal
                                                       else Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.panel_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff if horizontal
                                                     else Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        for box in self.panel_scroll.widget().findChildren(QGroupBox):
            box.setMaximumWidth(380 if horizontal else 16777215)
        self.panel_scroll.widget().adjustSize()
        content = self.panel_scroll.widget().sizeHint()
        if horizontal:
            self.panel_scroll.setMinimumWidth(0)
            self.panel_scroll.setMinimumHeight(0)
            height = content.height() + 24             # + barre de defilement horizontale
            self.resizeDocks([self.dock], [height], Qt.Orientation.Vertical)
        else:
            self.panel_scroll.setMinimumHeight(0)
            self.panel_scroll.setMinimumWidth(content.width() + 20)
        self.act_panel_bottom.setChecked(area == Qt.DockWidgetArea.BottomDockWidgetArea)
        self.settings.setValue("panneau_bas", "true" if area == Qt.DockWidgetArea.BottomDockWidgetArea else "false")

    def _build_statusbar(self) -> None:
        bar = self.statusBar()
        self.message = QLabel("")
        self.busy_label = QLabel("")
        self.coordinates = QLabel("")
        self.zoom_label = QLabel("")
        bar.addWidget(self.message, 1)
        bar.addPermanentWidget(self.busy_label)
        bar.addPermanentWidget(self.coordinates)
        bar.addPermanentWidget(self.zoom_label)

    # =================================================================== Session

    def open_session(self, session: Session) -> None:
        if self.session is not None:
            self.session.close()
        self.session = session
        self.selected_zone_id = None
        self.selected_crack_id = None
        self.draft_id = None
        self.pending = None
        self.undo_stack = []
        self.preparing = set()
        self.mode = "select"
        self.act_select.setChecked(True)
        self.view.set_image(session.tiles)
        self.view.set_levels(int(self.settings.value("luminosite", 100)), int(self.settings.value("contraste", 100)))
        self.setWindowTitle(f"Fissures — {session.image_path.name}")
        self.reload()
        last = self.settings.value(f"tuile/{session.image_id}")
        if last is not None and int(last) in self.zones_by_id:
            self.select_zone(int(last), focus=True)
        self.status(session.describe())
        self.layer_cache.clear()
        self.layers_completed = 0
        if self.measures is not None:
            self.measures.close()
            self.measures = None
        self.refresh_layers()
        self.refresh_analysis()
        self.poll()

    def choose_mosaic(self) -> None:
        start = str(self.session.image_path.parent) if self.session else ""
        path, _ = QFileDialog.getOpenFileName(self, "Ouvrir une mosaïque", start,
                                              "Images TIFF (*.tif *.tiff);;Tous les fichiers (*.*)")
        if not path:
            return
        # Une base forcee suit la fenetre ; sinon chaque mosaique a la sienne.
        forced = self.session.database_path if self.session and self.session.database_forced else None
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            session = Session(Path(path), forced, log=self.status)
        except SessionError as error:
            QMessageBox.critical(self, "Ouverture impossible", str(error))
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.open_session(session)

    def export_base(self) -> None:
        """Copie coherente de la base de cette mosaique, a donner telle quelle."""
        from .. import bases
        session = self.session
        path, _ = QFileDialog.getSaveFileName(self, "Exporter la base de cette mosaïque",
                                              f"{session.image_id}.sqlite3", "Base SQLite (*.sqlite3)")
        if not path:
            return
        self.run(lambda: bases.export_database(session.database_path, Path(path), session.image_id),
                 lambda target: self.status(f"Base exportée : {target}"), "Export de la base…")

    def import_base(self) -> None:
        """Une base recue d'un collegue rejoint data/bases/ ; l'ancienne part dans les sauvegardes."""
        from .. import bases
        from ..database import _same_path
        path, _ = QFileDialog.getOpenFileName(self, "Importer une base", "",
                                              "Base SQLite (*.sqlite3 *.sqlite *.db);;Tous les fichiers (*.*)")
        if not path:
            return
        try:
            plan = bases.plan_import(Path(path))
        except ValueError as error:
            QMessageBox.warning(self, "Import impossible", str(error))
            return
        names = ", ".join(image["id"] for image in plan.images)
        if plan.replaced:
            answer = QMessageBox.question(
                self, "Remplacer la base ?",
                f"Une base existe déjà pour cette mosaïque : "
                f"{', '.join(p.name for p in plan.replaced)}.\n"
                f"La remplacer par celle reçue ? L'ancienne est gardée dans les sauvegardes.")
            if answer != QMessageBox.StandardButton.Yes:
                return
        session = self.session
        reopen = (session is not None and not session.database_forced
                  and any(_same_path(session.database_path, replaced) for replaced in plan.replaced))
        if reopen:
            self.edits.waitForDone(10000)
            session.close()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            bases.apply_import(plan, log=print)
            self.status(f"Base importée : {names}")
        except Exception as error:                 # rien n'est perdu : l'ancienne base reste en place
            QMessageBox.critical(self, "Import impossible", f"{type(error).__name__} : {error}")
        finally:
            QApplication.restoreOverrideCursor()
        if reopen:
            try:
                self.open_session(Session(session.image_path, log=self.status))
            except SessionError as error:
                QMessageBox.critical(self, "Ouverture impossible", str(error))

    def annotator(self) -> str | None:
        return self.annotator_edit.text().strip() or None

    def ask_annotator(self) -> None:
        name, ok = QInputDialog.getText(self, "Annotateur", "Votre nom :",
                                        text=self.annotator_edit.text())
        if ok:
            self.annotator_edit.setText(name.strip())
            self.settings.setValue("annotateur", name.strip())

    def closeEvent(self, event) -> None:
        self.settings.setValue("geometrie", self.saveGeometry())
        self.settings.setValue("grille", "true" if self.act_grid.isChecked() else "false")
        self.auto_timer.stop()
        if self.measures is not None:
            self.measures.close()
        self.background.clear()          # preparations de tuiles pas encore commencees
        self.display_pool.clear()
        self.view.pool.clear()
        self.edits.waitForDone(10000)    # les modifications, elles, vont au bout
        if self.session is not None:
            self.session.close()         # apres les modifications : le -wal est alors vide
        super().closeEvent(event)

    # =================================================================== Taches de fond

    def run(self, function, on_done=None, busy_text: str = "Calcul…", blocking: bool = True,
            on_failed=None, pool: QThreadPool | None = None) -> None:
        """Execute `function` hors de l'interface ; `on_done(resultat)` ensuite, `on_failed(message)` en cas d'erreur."""
        if blocking and self.busy:
            self.status("Calcul en cours…", error=True)
            return
        task = _Task(function)
        self._tasks.add(task)
        if blocking:
            self.set_busy(True, busy_text)

        def finish():
            self._tasks.discard(task)
            if blocking:
                self.set_busy(False)

        def done(result):
            finish()
            if on_done is not None:
                on_done(result)

        def failed(message):
            finish()
            self.status(message, error=True)
            if on_failed is not None:
                on_failed(message)
            if blocking:
                self.reload()

        task.signals.done.connect(done)
        task.signals.failed.connect(failed)
        (pool or (self.edits if blocking else self.background)).start(task)

    def set_busy(self, busy: bool, text: str = "") -> None:
        self.busy = busy
        self.busy_label.setText(f"⏳ {text}" if busy else "")
        if busy:
            QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
        else:
            QApplication.restoreOverrideCursor()

    def status(self, text: str, error: bool = False) -> None:
        self.message.setText(text)
        self.message.setStyleSheet("color: #b3261e;" if error else "")

    # =================================================================== Affichage

    def reload(self) -> None:
        self.zones = self.session.zones()
        self.zones_by_id = {zone["id"]: zone for zone in self.zones}
        self.zones_order = [zone["id"] for zone in self.zones]
        self.cracks = self.session.cracks()
        self.cracks_by_id = {crack["id"]: crack for crack in self.cracks}
        if self.selected_crack_id not in self.cracks_by_id:
            self.selected_crack_id = None
        if self.draft_id not in self.cracks_by_id:
            self.draft_id = None
        self.refresh_zones()
        self.refresh_cracks()
        self.refresh_analysis()

    def refresh_zones(self) -> None:
        self.view.set_zones(self.zones, self.selected_zone_id, self.act_grid.isChecked())
        self.update_panel()

    def refresh_cracks(self) -> None:
        self.view.set_cracks(self.cracks, self.selected_crack_id, self.act_show_cracks.isChecked())
        crack = self.selected_crack()
        self.view.set_handles(crack["anchors"] if crack else None, self.handle_moved, self.handle_removed)
        self.view.set_marker(self.pending[1:] if self.pending else None)
        self.update_panel()

    def update_panel(self) -> None:
        zone = self.zones_by_id.get(self.selected_zone_id) if self.selected_zone_id else None
        if zone:
            count = sum(c["zone_id"] == zone["id"] for c in self.cracks)
            self.zone_label.setText(f"Ligne {zone['row_index'] + 1}, colonne {zone['col_index'] + 1} — "
                                    f"<b>{STATUS_LABELS.get(zone['status'], zone['status'])}</b><br>"
                                    f"{count} fissure(s) tracée(s)")
        else:
            self.zone_label.setText("Aucune tuile sélectionnée.")
        crack = self.selected_crack()
        for action in (*self.category_actions.values(), self.act_delete):
            action.setEnabled(crack is not None)
        for button in self.category_buttons.values():
            button.setEnabled(crack is not None)
            button.setStyleSheet("")
        if crack is None:
            self.crack_label.setText("Aucune fissure sélectionnée.")
            self.suggestion_label.setText("")
            return
        length = crack["length_px"]
        pixel = self.session.pixel_um
        length_text = f"{length * pixel / 1000:.2f} mm" if pixel else f"{length:.0f} px"
        state = "en cours de tracé" if crack["id"] == self.draft_id else (
            "proposée, à vérifier" if crack["state"] == "proposed" else "validée")
        self.crack_label.setText(f"Fissure {crack['id']} — <b>{LABELS.get(crack['category'], crack['category'])}"
                                 f"</b><br>{len(crack['anchors'])} nœuds · {length_text} · {state}")
        suggestion = crack.get("suggestion") if crack["state"] == "proposed" else None
        if suggestion:
            self.suggestion_label.setText(
                f"Suggestion : <b>{LABELS.get(suggestion['category'])}</b> "
                f"({100 * suggestion['confidence']:.0f} %)")
            self.category_buttons[suggestion["category"]].setStyleSheet("border: 2px solid #e0a800;")
        else:
            self.suggestion_label.setText("")

    def on_hover(self, point: QPointF) -> None:
        if self.session and 0 <= point.x() < self.session.width and 0 <= point.y() < self.session.height:
            self.coordinates.setText(f"x {point.x():,.0f} · y {point.y():,.0f}".replace(",", " "))
        else:
            self.coordinates.setText("")

    def edit_levels(self) -> None:
        def apply(brightness, contrast):
            self.view.set_levels(brightness, contrast)
            self.settings.setValue("luminosite", brightness)
            self.settings.setValue("contraste", contrast)
        LevelsDialog(int(self.settings.value("luminosite", 100)), int(self.settings.value("contraste", 100)),
                     apply, self).exec()

    # =================================================================== Selection

    def selected_crack(self) -> dict | None:
        return self.cracks_by_id.get(self.selected_crack_id) if self.selected_crack_id else None

    def select_zone(self, zone_id: int, focus: bool = False) -> None:
        changed = zone_id != self.selected_zone_id
        self.selected_zone_id = zone_id
        self.settings.setValue(f"tuile/{self.session.image_id}", zone_id)
        self.refresh_zones()
        if changed:
            self.refresh_layers()
        if focus:
            self.center_zone()
        self.prepare_ahead(zone_id)

    def prepare_ahead(self, zone_id: int) -> None:
        """Prepare le routage de la tuile choisie, puis de la suivante, en fond.

        Le premier trace dans une tuile attend cette preparation (une quinzaine
        de secondes a 0,37 um), les suivants non. La tuile suivante est donc
        prete quand on y arrive par « Tuile suivante ». Le detecteur garde les
        trois dernieres en memoire ; une tuile deja prete ne coute rien.
        """
        targets = [zone_id]
        # Pas d'avance quand un calque est affiche : son calcul attendrait derriere.
        if zone_id in self.zones_order and self.layer_kind() is None:
            index = self.zones_order.index(zone_id)
            targets += self.zones_order[index + 1:index + 2]
        targets = [z for z in targets if z not in self.preparing]
        if not targets:
            return
        self.preparing.update(targets)
        session = self.session

        def prepare():
            for target in targets:               # dans l'ordre : la tuile choisie d'abord
                session.prepare(target)

        def finished(_=None):
            self.preparing.difference_update(targets)

        self.run(prepare, finished, blocking=False, on_failed=finished)

    def select_crack(self, crack_id: int) -> None:
        crack = self.cracks_by_id.get(crack_id)
        if crack is None:
            return
        self.selected_crack_id = crack_id
        if crack["zone_id"] != self.selected_zone_id:
            self.select_zone(crack["zone_id"])
        self.refresh_cracks()

    def center_zone(self) -> None:
        zone = self.zones_by_id.get(self.selected_zone_id)
        if zone:
            self.view.fit_rect(QRectF(zone["x"], zone["y"], zone["width"], zone["height"]), 0.03)

    def navigate(self, offset: int) -> None:
        if not self.zones_order:
            return
        if self.selected_zone_id in self.zones_order:
            index = self.zones_order.index(self.selected_zone_id) + offset
        else:
            index = 0
        index = max(0, min(len(self.zones_order) - 1, index))
        self.selected_crack_id = None
        self.select_zone(self.zones_order[index], focus=True)
        self.refresh_cracks()

    def crack_at(self, x: float, y: float) -> dict | None:
        tolerance = self.view.tolerance(9)
        best, best_distance = None, tolerance
        for crack in self.cracks:
            xs = [p[0] for p in crack["geometry"]]
            ys = [p[1] for p in crack["geometry"]]
            if not (min(xs) - tolerance <= x <= max(xs) + tolerance and min(ys) - tolerance <= y <= max(ys) + tolerance):
                continue
            distance = _distance_to_polyline(x, y, crack["geometry"])
            if distance <= best_distance:
                best, best_distance = crack, distance
        return best

    # =================================================================== Souris et clavier

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        {"select": self.act_select, "draw": self.act_draw, "insert": self.act_insert,
         "cut": self.act_cut}[mode].setChecked(True)
        self.status({"select": "Mode Sélection.", "draw": "Mode Tracer — Échap pour terminer.",
                     "insert": "Mode Ajout de nœuds — Échap pour quitter.",
                     "cut": "Mode Couper — Échap pour quitter."}[mode])

    def on_click(self, point: QPointF, button, modifiers) -> None:
        if button != Qt.MouseButton.LeftButton or self.session is None:
            return
        x, y = point.x(), point.y()
        if not (0 <= x < self.session.width and 0 <= y < self.session.height):
            return
        zone = self.session.zone_at(x, y)
        if zone is None:
            return
        if self.mode == "draw":
            self.draw_click(x, y, zone)
        elif self.mode == "cut":
            self.cut_at(x, y)
        elif self.mode == "insert" or (modifiers & Qt.KeyboardModifier.ShiftModifier and self.selected_crack()):
            self.insert_node(x, y)
        else:
            hit = self.crack_at(x, y)
            if hit:
                self.select_crack(hit["id"])
                self.status(f"Fissure {hit['id']}.")
            else:
                self.selected_crack_id = None
                self.select_zone(zone["id"])
                self.refresh_cracks()

    def eventFilter(self, watched, event) -> bool:
        """Touches de la vue : Espace maintenu, Entree, categories (chiffres ou AZERTY)."""
        if watched is not self.view or event.type() not in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            return super().eventFilter(watched, event)
        key = event.key()
        if key == Qt.Key.Key_Space:
            if not event.isAutoRepeat():
                self.view.set_cracks_visible(event.type() == QEvent.Type.KeyRelease
                                             and self.act_show_cracks.isChecked())
            return True
        if event.type() != QEvent.Type.KeyPress:
            return False
        modifiers = event.modifiers() & ~(Qt.KeyboardModifier.KeypadModifier | Qt.KeyboardModifier.ShiftModifier)
        if modifiers:
            return False
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.change_zone_cracks("accept")          # Maj+Entree : toute la tuile
            else:
                self.accept_suggestion()
            return True
        for category, _, digit, azerty, _ in CATEGORY_KEYS:
            if key in (digit, azerty):
                self.categorize(category)
                return True
        return False

    # =================================================================== Tracer et corriger

    def draft(self) -> dict | None:
        return self.cracks_by_id.get(self.draft_id) if self.draft_id else None

    def draw_click(self, x: float, y: float, zone: dict) -> None:
        if self.busy:
            self.status("Calcul en cours…", error=True)
            return
        draft = self.draft()
        if draft:
            if zone["id"] != draft["zone_id"]:
                self.status("Les points d'une fissure doivent rester dans la même tuile.", error=True)
                return
            last = draft["anchors"][-1]
            if math.hypot(x - last[0], y - last[1]) < 3:
                return
            anchors = [list(p) for p in draft["anchors"]] + [[x, y]]
            self.edit_crack(draft, anchors=anchors, state="proposed",
                            message=f"Point {len(anchors)} ajouté.")
            return
        if self.pending is None:
            self.pending = (zone["id"], x, y)
            self.selected_crack_id = None
            self.select_zone(zone["id"])
            self.refresh_cracks()
            self.status("Premier point posé.")
            return
        zone_id, x0, y0 = self.pending
        if zone["id"] != zone_id:
            self.status("Les points d'une fissure doivent rester dans la même tuile.", error=True)
            return
        if math.hypot(x - x0, y - y0) < 3:
            return
        annotator = self.annotator()

        def created(crack):
            self.pending = None
            self.draft_id = crack["id"]
            self.selected_crack_id = crack["id"]
            self.undo_stack.append({"type": "create", "id": crack["id"]})
            self.reload()
            self.status(f"Fissure {crack['id']} créée.")

        self.run(lambda: self.session.create_crack(zone_id, [[x0, y0], [x, y]], annotator), created,
                 "Calcul du chemin…")

    def edit_crack(self, crack: dict, message: str = "", record: bool = True, after=None, **changes) -> None:
        previous = {"anchors": [list(p) for p in crack["anchors"]], "category": crack["category"],
                    "state": crack["state"]}
        annotator = self.annotator()

        def saved(updated):
            if record:
                self.undo_stack.append({"type": "crack", "id": updated["id"], "previous": previous})
            self.reload()
            if message:
                self.status(message)
            if after is not None:
                after(updated)

        self.run(lambda: self.session.update_crack(crack["id"], crack["revision"], annotator=annotator,
                                                   **changes),
                 saved, "Recalcul de la fissure…" if "anchors" in changes else "Enregistrement…")

    def finish_draft(self) -> None:
        had_draft = self.draft_id is not None
        self.draft_id = None
        self.pending = None
        self.set_mode("select")
        self.refresh_cracks()
        if had_draft:
            self.status("Tracé terminé.")

    def remove_last_point(self) -> None:
        if self.pending is not None:
            self.pending = None
            self.refresh_cracks()
            self.status("Premier point retiré.")
            return
        draft = self.draft()
        if draft is None:
            return
        if len(draft["anchors"]) > 2:
            self.edit_crack(draft, anchors=[list(p) for p in draft["anchors"][:-1]], state="proposed",
                            message="Point retiré.")
            return
        first = draft["anchors"][0]

        def back_to_first(_):
            self.draft_id = None
            self.pending = (draft["zone_id"], first[0], first[1])
            self.refresh_cracks()
            self.status("Point retiré.")

        self.edit_crack(draft, state="rejected", record=False, after=back_to_first)

    def insert_node(self, x: float, y: float) -> None:
        crack = self.selected_crack()
        if crack is None:
            self.status("Aucune fissure sélectionnée.", error=True)
            return
        geometry = crack["geometry"]
        best, along, total = None, 0.0, 0.0
        for (x1, y1), (x2, y2) in zip(geometry, geometry[1:]):
            dx, dy = x2 - x1, y2 - y1
            size = math.hypot(dx, dy)
            if size:
                t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / (size * size)))
                px, py = x1 + t * dx, y1 + t * dy
                distance = math.hypot(px - x, py - y)
                if best is None or distance < best[0]:
                    best, along = (distance, px, py), total + t * size
            total += size
        if best is None or best[0] > self.view.tolerance(12):
            self.status("Cliquer sur la fissure sélectionnée.", error=True)
            return

        def position(point):
            return _position_along(geometry, point)

        positions = [position(p) for p in crack["anchors"]]
        index = next((i for i, p in enumerate(positions) if p > along + 1e-6), None)
        if index is None or index < 1 or any(math.hypot(px - best[1], py - best[2]) < 3 for px, py in crack["anchors"]):
            self.status("Cliquer entre deux nœuds.", error=True)
            return
        anchors = [list(p) for p in crack["anchors"]]
        anchors.insert(index, [best[1], best[2]])
        self.edit_crack(crack, anchors=anchors, message="Nœud ajouté.")

    def cut_at(self, x: float, y: float) -> None:
        """Coupe la fissure cliquee en deux, a l'endroit du clic."""
        if self.busy:
            self.status("Calcul en cours…", error=True)
            return
        crack = self.crack_at(x, y)
        if crack is None:
            self.status("Cliquer sur une fissure.", error=True)
            return
        if crack["id"] == self.draft_id:
            self.draft_id = None
        previous = {"anchors": [list(p) for p in crack["anchors"]], "geometry": crack["geometry"],
                    "category": crack["category"], "state": crack["state"]}
        annotator = self.annotator()

        def done(pieces):
            self.undo_stack.append({"type": "split", "id": pieces[0]["id"], "previous": previous,
                                    "new_ids": [p["id"] for p in pieces[1:]]})
            self.selected_crack_id = pieces[0]["id"]
            self.reload()
            self.status(f"Fissure {crack['id']} coupée en deux.")

        self.run(lambda: self.session.split_crack(crack["id"], crack["revision"], x, y, annotator), done,
                 "Coupe…")

    def handle_moved(self, index: int, point: QPointF) -> None:
        crack = self.selected_crack()
        if crack is None or self.busy:
            self.refresh_cracks()
            return
        zone = self.zones_by_id[crack["zone_id"]]
        x = max(zone["x"], min(zone["x"] + zone["width"] - 1, point.x()))
        y = max(zone["y"], min(zone["y"] + zone["height"] - 1, point.y()))
        anchors = [list(p) for p in crack["anchors"]]
        anchors[index] = [x, y]
        self.edit_crack(crack, anchors=anchors, message="Nœud déplacé")

    def handle_removed(self, index: int) -> None:
        crack = self.selected_crack()
        if crack is None or self.busy:
            return
        if len(crack["anchors"]) <= 2:
            self.status("Deux nœuds minimum.", error=True)
            return
        anchors = [list(p) for i, p in enumerate(crack["anchors"]) if i != index]
        self.edit_crack(crack, anchors=anchors, message=f"Nœud retiré · {len(anchors)} nœuds")

    def categorize(self, category: str) -> None:
        crack = self.selected_crack()
        if crack is None:
            self.status("Aucune fissure sélectionnée.", error=True)
            return
        if crack["id"] == self.draft_id:
            self.draft_id = None
        self.edit_crack(crack, category=category, state="validated",
                        message=f"Fissure {crack['id']} classée « {LABELS[category]} »")

    def accept_suggestion(self) -> None:
        crack = self.selected_crack()
        if crack and crack["state"] == "proposed" and crack.get("suggestion"):
            self.categorize(crack["suggestion"]["category"])

    def delete_selected(self) -> None:
        if self.pending is not None and self.selected_crack() is None:
            self.pending = None
            self.refresh_cracks()
            return
        crack = self.selected_crack()
        if crack is None:
            return
        if crack["id"] == self.draft_id:
            self.draft_id = None
        self.edit_crack(crack, state="rejected", message=f"Fissure {crack['id']} supprimée")

    def undo(self) -> None:
        if self.pending is not None:
            self.pending = None
            self.refresh_cracks()
            return
        if not self.undo_stack:
            self.status("Rien à annuler.")
            return
        action = self.undo_stack.pop()
        if action["type"] == "bulk":
            annotator = self.annotator()
            self.run(lambda: self.session.restore_cracks(action["previous"], annotator),
                     lambda _: (self.reload(), self.status(f"{action['label']} annulée.")))
            return
        if action["type"] == "zone":
            annotator = self.annotator()

            def restore_zone():
                self.session.set_zone_status(action["id"], action["previous"], annotator)
                if action.get("confirmed"):
                    self.session.unconfirm_zone(action["id"], action["confirmed"], annotator)

            self.run(restore_zone, lambda _: (self.reload(), self.status("Statut de la tuile rétabli.")))
            return
        if action["type"] == "split":
            def restore():
                previous = action["previous"]
                self.session.restore_crack(action["id"], previous["anchors"], previous["geometry"],
                                           previous["category"], previous["state"], self.annotator())
                for crack_id in action["new_ids"]:
                    crack = self.session.crack(crack_id)
                    if crack["state"] != "rejected":
                        self.session.update_crack(crack_id, crack["revision"], state="rejected",
                                                  annotator=self.annotator())
            self.run(restore, lambda _: (self.reload(), self.status("Coupe annulée.")))
            return
        crack = self.session.crack(action["id"])
        if action["type"] == "create":
            if crack["id"] == self.draft_id:
                self.draft_id = None
            self.edit_crack(crack, state="rejected", record=False, message="Création de la fissure annulée")
            return
        previous = action["previous"]
        self.edit_crack(crack, record=False, message="Dernière modification annulée",
                        anchors=previous["anchors"], category=previous["category"], state=previous["state"])

    # =================================================================== Tuiles

    def set_zone_status(self, status: str) -> None:
        zone = self.zones_by_id.get(self.selected_zone_id)
        if zone is None:
            self.status("Sélectionner d'abord une tuile.", error=True)
            return
        previous = zone["status"]
        annotator = self.annotator()
        session = self.session

        def change():
            updated = session.set_zone_status(zone["id"], status, annotator)
            # Valider une tuile confirme ses propositions restantes, telles quelles.
            confirmed = session.confirm_zone(zone["id"], annotator) if status == "done" else []
            return updated, confirmed

        def done(values):
            updated, confirmed = values
            self.undo_stack.append({"type": "zone", "id": updated["id"], "previous": previous,
                                    "confirmed": confirmed})
            self.reload()
            self.status(f"Tuile {STATUS_LABELS[status]}."
                        + (f" {len(confirmed)} proposition(s) confirmée(s)." if confirmed else ""))

        self.run(change, done)

    def change_zone_cracks(self, mode: str) -> None:
        """Toutes les fissures de la tuile : classements acceptes (« accept ») ou supprimees (« reject »)."""
        zone = self.zones_by_id.get(self.selected_zone_id)
        if zone is None:
            self.status("Sélectionner d'abord une tuile.", error=True)
            return
        label = "Acceptation des classements" if mode == "accept" else "Suppression des fissures de la tuile"
        annotator = self.annotator()
        session = self.session

        def done(previous):
            if not previous:
                self.status("Rien à changer dans cette tuile.")
                return
            self.undo_stack.append({"type": "bulk", "previous": previous, "label": label})
            changed = {item["id"] for item in previous}
            if mode == "reject":
                if self.selected_crack_id in changed:
                    self.selected_crack_id = None
                if self.draft_id in changed:
                    self.draft_id = None
            self.reload()
            what = "classement(s) accepté(s)" if mode == "accept" else "fissure(s) supprimée(s)"
            self.status(f"{len(previous)} {what}. Ctrl+Z pour annuler.")

        self.run(lambda: session.change_zone_cracks(zone["id"], mode, annotator), done)

    # =================================================================== Automatique

    def poll(self) -> None:
        """Suivi du calcul des calques en fond."""
        running = self.poll_layers()
        if running and not self.auto_timer.isActive():
            self.auto_timer.start()
        elif not running:
            self.auto_timer.stop()

    def train_colmatage(self) -> None:
        """Le reseau de colmatage, reappris sur vos fissures validees (autre processus, ~1 h)."""
        def installed():
            self.session.automation.reload_network()
            self.status("Apprentissage terminé : le nouveau modèle n'est installé que s'il fait mieux (voir le journal).")

        ProcessDialog(
            "Réentraîner le classement du colmatage",
            ["-u", "-m", "analyse", "--database", str(self.session.database_path),
             "--image", str(self.session.image_path), "colmatage"],
            "colmatage",
            "Apprend sur les fissures validées des lignes d'apprentissage, se règle sur les lignes de "
            "validation, puis se mesure sur les lignes de test, jamais vues. Il ne remplace le modèle en "
            "service que s'il fait mieux que lui sur ces tuiles de test ; l'ancien est gardé sous un nom daté. "
            "Compter environ une heure.",
            on_success=installed, parent=self).exec()

    # =================================================================== Analyse complete

    def toggle_annotations(self) -> None:
        self.settings.setValue("afficher_annotations", "true" if self.act_show_cracks.isChecked() else "false")
        self.refresh_cracks()

    def refresh_analysis(self) -> None:
        """Le bloc « Analyse automatique » du panneau : ou en est la mosaique."""
        result = self.session.analysis_result() if self.session else None
        if result is None:
            self.analysis_label.setText("Mosaïque pas encore analysée.")
            return
        done = sum(z["status"] == "done" for z in self.zones)
        imported = sum(c.get("source") == "analyse" for c in self.cracks)
        lines = [f"Analysée le {result['analyse_le'][:16].replace('T', ' à ')}.",
                 f"Tuiles validées : {done} sur {len(self.zones)}."]
        if imported:
            lines.append(f"{imported} proposition(s) de l'analyse à vérifier.")
        self.analysis_label.setText("<br>".join(lines))

    def full_analysis(self) -> None:
        """Le seul bouton : toute la mosaique, puis le resultat en propositions a corriger."""
        def done():
            self.reload()
            self.refresh_analysis()
            self.status("Analyse terminée : les fissures sont proposées en pointillés, tuile par tuile.")

        ProcessDialog(
            "Analyser la mosaïque",
            ["-u", "-m", "analyse", "--database", str(self.session.database_path),
             "--image", str(self.session.image_path), "analyser", "--importer"],
            "analyse",
            "Calcule toute la mosaïque (environ 45 minutes la première fois, arrêtable) : fissures, bords "
            "exclus, état du colmatage tous les 10 µm, puis le rapport. Les fissures trouvées arrivent en "
            "propositions (pointillés) dans les tuiles non validées, à la place des anciennes propositions ; "
            "vos tuiles validées ne sont pas touchées. Ensuite : de tuile en tuile (Page suiv.), corriger "
            "avec les outils habituels, puis « Valider la tuile », qui confirme ce qui reste.",
            on_success=done, parent=self).exec()

    def audit(self) -> None:
        result = self.session.analysis_result()
        if result is None:
            QMessageBox.information(self, "Sondage", "Lancer d'abord l'analyse complète de la mosaïque.")
            return
        from analyse.audit import Audit
        audit = Audit(self.session.analysis_dir / "audit.json", result)
        AuditDialog(audit, self.session.tiles.source, self.session.pixel_um, self).exec()
        self.refresh_analysis()

    def open_report(self) -> None:
        import os
        path = self.session.analysis_dir / "rapport.pdf"
        result = self.session.analysis_result()
        if result is None:
            QMessageBox.information(self, "Rapport", "Lancer d'abord l'analyse complète de la mosaïque.")
            return

        def open_it(_=None):
            os.startfile(str(path))                  # le lecteur PDF du poste

        audit_path = self.session.analysis_dir / "audit.json"
        stale = not path.exists() or (audit_path.exists() and audit_path.stat().st_mtime > path.stat().st_mtime)
        if not stale:
            open_it()
            return
        session = self.session

        def rebuild():
            from analyse import donnees, rapport
            return rapport.build_for(donnees.load(session.database_path, session.image_path), result)

        self.run(rebuild, open_it, "Mise à jour du rapport…")

    # =================================================================== Calques (diagnostic)

    def layer_kind(self) -> str | None:
        return self.current_layer

    def set_layer(self, kind: str | None) -> None:
        self.current_layer = kind
        self.layer_actions[kind].setChecked(True)
        self.refresh_layers()

    def _layer_zones(self) -> list[dict]:
        if self.act_layer_all.isChecked():
            return self.zones
        return [z for z in self.zones if z["id"] == self.selected_zone_id]

    def _ensure_layer(self) -> None:
        """Le calque de la tuile affichee se calcule tout seul s'il manque (jusqu'a une demi-minute)."""
        zone = self.zones_by_id.get(self.selected_zone_id)
        if zone is None:
            computed = self.act_layer_all.isChecked() and any(self.session.layer_computed(z) for z in self.zones)
            if not computed:
                self.status("Cliquer sur une tuile : son calque se calcule (jusqu'à une demi-minute).")
            return
        if self.layers_running or self.session.layer_computed(zone):
            return
        self.session.layers.start([zone])
        self.layers_completed = 0
        self.status(f"Calcul du calque de la tuile {zone['id']} (jusqu'à une demi-minute)…")
        self.poll()

    def refresh_layers(self) -> None:
        """Recharge le calque affiche, en fond ; les tuiles deja colorees sont reprises du cache."""
        self.layer_generation += 1
        generation = self.layer_generation
        kind = self.layer_kind()
        if kind is None or self.session is None:
            self.view.set_overlays([], 1.0)
            self.layer_cache.clear()
            return
        self._ensure_layer()
        session, selected = self.session, self.selected_zone_id
        zones = self._layer_zones()
        known = set(self.layer_cache)

        def build():
            entries = []
            for zone in zones:
                path = session.layers.path(zone)
                if not path.exists():
                    continue
                full = zone["id"] == selected
                key = (kind, zone["id"], path.stat().st_mtime_ns, full)
                if key in known:
                    entries.append((key, zone, None))
                    continue
                values = session.layer_array(zone, kind)
                if not full:
                    values = reduce(values, kind, OVERVIEW_SIDE)
                entries.append((key, zone, colorize(values, kind)))
            return entries

        def done(entries):
            if generation != self.layer_generation:
                return                                   # un autre choix a ete fait entre-temps
            cache, overlays = {}, []
            for key, zone, image in entries:
                image = image if image is not None else self.layer_cache.get(key)
                if image is None:
                    continue
                cache[key] = image
                overlays.append((QRectF(zone["x"], zone["y"], zone["width"], zone["height"]), image,
                                 kind in BINARY))
            self.layer_cache = cache
            self.view.set_overlays(overlays, LAYER_OPACITY)

        self.run(build, done, blocking=False, pool=self.display_pool)

    def poll_layers(self) -> bool:
        if self.session is None or self.session._layers is None:
            running = False
            status = {}
        else:
            with self.session.layers.job_lock:
                status = dict(self.session.layers.status)
            running = status.get("running", False)
        if running and status["completed"] != self.layers_completed:
            self.layers_completed = status["completed"]
            self.refresh_layers()
        elif not running and self.layers_running:     # vient de finir
            errors = status.get("errors", [])
            self.layers_running = False
            if errors:
                self.status(f"Calques impossibles pour la tuile {errors[0]['zone_id']} : {errors[0]['message']}",
                            error=True)
            else:
                self.status("Calques calculés.")
            self.refresh_layers()
        self.layers_running = running
        return running

    def export_layers(self) -> None:
        zones = self._layer_zones() if not self.act_layer_all.isChecked() else self.zones
        if not any(self.session.layer_computed(z) for z in zones):
            self.status("Aucun calque calculé à exporter.", error=True)
            return
        path, _ = QFileDialog.getSaveFileName(self, "Exporter les calques", "calques.zip", "Archive ZIP (*.zip)")
        if not path:
            return
        self.run(lambda: self.session.export_layers(Path(path), zones),
                 lambda count: self.status(f"{count} tuile(s) exportée(s) : {path}"), "Export des calques…")

    # =================================================================== Mesures

    def open_measures(self, tab: str) -> None:
        if self.measures is None:
            self.measures = MeasuresDialog(self, tab)
            self.measures.finished.connect(lambda _: setattr(self, "measures", None))
            self.measures.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        else:
            self.measures.refresh()
        self.measures.show_tab(tab)

    def train_detector(self) -> None:
        session = self.session
        # Sans base forcee, la base suit l'image choisie dans la fenetre.
        TrainingDialog(session.image_path, session.database_path if session.database_forced else None,
                       self).exec()


def _position_along(geometry, point) -> float:
    """Abscisse curviligne du projete de `point` sur la polyligne."""
    x, y = point
    best, along, total = None, 0.0, 0.0
    for (x1, y1), (x2, y2) in zip(geometry, geometry[1:]):
        dx, dy = x2 - x1, y2 - y1
        size = math.hypot(dx, dy)
        if size:
            t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / (size * size)))
            distance = math.hypot(x1 + t * dx - x, y1 + t * dy - y)
            if best is None or distance < best:
                best, along = distance, total + t * size
        total += size
    return along
