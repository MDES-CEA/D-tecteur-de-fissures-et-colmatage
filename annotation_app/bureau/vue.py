"""Vue de la mosaique : image par tuiles, fissures, grille, poignees.

L'image fait des gigaoctets : on n'en charge que les tuiles visibles, au niveau
de detail adapte au zoom, dans des fils de fond. Un apercu de toute la mosaique
reste toujours charge dessous, pour qu'aucune zone ne soit jamais vide pendant
un deplacement.

Les coordonnees de la scene sont les pixels de la mosaique en pleine
resolution : un trace s'y dessine tel qu'il est stocke dans la base.
"""

from __future__ import annotations

from math import ceil, floor, log2

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)

# Couleurs des categories, les memes que dans le bilan et les figures.
CATEGORY_COLORS = {
    "unclassified": "#d28b00",
    "open": "#d43d2f",
    "filled": "#087db6",
    "reopened": "#14a38b",
    "mixed": "#7650a8",
    "uncertain": "#777777",
}
STATUS_COLORS = {"todo": "#9a9a9a", "in_progress": "#b87800", "review": "#7650a8", "done": "#2f9e55"}
MAX_ZOOM = 8.0                 # pixels d'ecran par pixel d'image, au plus


class _TileSignals(QObject):
    loaded = Signal(object, object)          # (generation, cle), QImage


class _TileJob(QRunnable):
    def __init__(self, tiles, key, lut, generation, signals) -> None:
        super().__init__()
        self.tiles, self.key, self.lut, self.generation, self.signals = tiles, key, lut, generation, signals

    def run(self) -> None:
        try:
            pixels = self.tiles.tile_array(*self.key)
        except Exception:
            return
        if self.lut is not None:
            pixels = self.lut[pixels]
        pixels = np.ascontiguousarray(pixels, dtype=np.uint8)
        height, width = pixels.shape
        image = QImage(pixels.data, width, height, width, QImage.Format.Format_Grayscale8).copy()
        self.signals.loaded.emit((self.generation, self.key), image)


class HandleItem(QGraphicsEllipseItem):
    """Noeud d'un trace : on le deplace a la souris, clic droit pour le retirer."""

    def __init__(self, index: int, x: float, y: float, moved, removed) -> None:
        super().__init__(-6, -6, 12, 12)
        self.index, self.moved, self.removed = index, moved, removed
        self.setPos(x, y)
        self.setBrush(QBrush(QColor("white")))
        self.setPen(QPen(QColor("#202020"), 1.5))
        self.setZValue(60)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.setToolTip("Glisser pour deplacer ce noeud ; clic droit pour le retirer")

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            self.removed(self.index)
            event.accept()
            return
        self._start = self.pos()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)
        start = getattr(self, "_start", None)
        # Un simple clic sur un noeud ne recalcule pas la fissure.
        if event.button() == Qt.MouseButton.LeftButton and start is not None and self.pos() != start:
            self.moved(self.index, self.pos())


class MosaicView(QGraphicsView):
    clicked = Signal(QPointF, object, object)     # position, bouton, modificateurs
    hovered = Signal(QPointF)
    zoomChanged = Signal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self.setBackgroundBrush(QColor(45, 45, 45))
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(4)
        self.signals = _TileSignals()
        self.signals.loaded.connect(self._tile_ready)
        self.refresh_timer = QTimer(self, singleShot=True, interval=40)
        self.refresh_timer.timeout.connect(self._update_tiles)

        self.tiles = None
        self.tile_items: dict[tuple, QGraphicsPixmapItem] = {}
        self.pending: set[tuple] = set()
        self.generation = 0
        self.lut = None
        self.min_scale = 1e-4
        self.zone_items: dict[int, QGraphicsRectItem] = {}
        self.crack_items: list[QGraphicsPathItem] = []
        self.handle_items: list[HandleItem] = []
        self.overlay_items: list[QGraphicsPixmapItem] = []
        self.analysis_items: list[QGraphicsPathItem] = []
        self.marker = None
        self._press = None
        self._pending_fit = False           # False : rien ; None : tout voir ; (rect, marge)

    # -- Image ---------------------------------------------------------------

    def set_image(self, tiles) -> None:
        self.generation += 1
        self.pool.clear()
        self.scene().clear()
        self.tile_items.clear()
        self.pending.clear()
        self.zone_items.clear()
        self.crack_items.clear()
        self.handle_items.clear()
        self.overlay_items.clear()
        self.analysis_items.clear()
        self.marker = None
        self.tiles = tiles
        self.scene().setSceneRect(QRectF(0, 0, tiles.width, tiles.height))
        k = max(0, ceil(log2(max(tiles.width, tiles.height) / 1024)))
        self.base_level = max(0, tiles.max_level - k)
        self.fit_all()

    def set_levels(self, brightness: int, contrast: int) -> None:
        """Luminosite et contraste en %, appliques aux tuiles (100/100 : l'image brute)."""
        if brightness == 100 and contrast == 100:
            self.lut = None
        else:
            values = np.arange(256, dtype=np.float32)
            values = (values - 128) * contrast / 100 + 128
            values = values * brightness / 100
            self.lut = np.clip(values, 0, 255).astype(np.uint8)
        self.generation += 1
        for item in self.tile_items.values():
            self.scene().removeItem(item)
        self.tile_items.clear()
        self.pending.clear()
        self.schedule()

    def _level_scale(self, level: int) -> int:
        return 1 << (self.tiles.max_level - level)

    def _level_for_zoom(self, zoom: float) -> int:
        ratio = 1.0 / max(zoom, 1e-9)        # pixels d'image par pixel d'ecran
        if ratio <= 1:
            return self.tiles.max_level
        return max(0, self.tiles.max_level - int(floor(log2(ratio))))

    def schedule(self) -> None:
        self.refresh_timer.start()

    def _visible_rect(self) -> QRectF:
        return self.mapToScene(self.viewport().rect()).boundingRect().intersected(self.sceneRect())

    def _tile_keys(self, level: int, rect: QRectF) -> list[tuple]:
        span = self.tiles.tile_size * self._level_scale(level)
        x0, x1 = int(rect.left() // span), int(rect.right() // span)
        y0, y1 = int(rect.top() // span), int(rect.bottom() // span)
        width, height = self.tiles.level_dimensions(level)
        x1 = min(x1, ceil(width / self.tiles.tile_size) - 1)
        y1 = min(y1, ceil(height / self.tiles.tile_size) - 1)
        return [(level, x, y) for y in range(max(0, y0), y1 + 1) for x in range(max(0, x0), x1 + 1)]

    def _update_tiles(self) -> None:
        if self.tiles is None:
            return
        rect = self._visible_rect()
        level = self._level_for_zoom(self.transform().m11())
        wanted = self._tile_keys(self.base_level, self.sceneRect())
        if level != self.base_level:
            wanted += self._tile_keys(level, rect)
        # Le centre de la vue d'abord : c'est la que l'oeil regarde.
        center = rect.center()

        def distance(key):
            span = self.tiles.tile_size * self._level_scale(key[0])
            return (key[0] != self.base_level, abs((key[1] + .5) * span - center.x()) +
                    abs((key[2] + .5) * span - center.y()))

        for key in sorted(wanted, key=distance):
            if key not in self.tile_items and key not in self.pending:
                self.pending.add(key)
                self.pool.start(_TileJob(self.tiles, key, self.lut, self.generation, self.signals))
        visible = set(wanted)
        current_done = all(key in self.tile_items for key in visible if key[0] == level)
        for key in list(self.tile_items):
            stale_level = key[0] not in (self.base_level, level)
            if (stale_level and current_done) or (key[0] == level and key not in visible and len(self.tile_items) > 400):
                self.scene().removeItem(self.tile_items.pop(key))

    def _tile_ready(self, tagged, image: QImage) -> None:
        generation, key = tagged
        if generation != self.generation:
            return                      # tuile d'avant un changement de contraste ou d'image
        self.pending.discard(key)
        if key in self.tile_items:
            return
        level, tx, ty = key
        scale = self._level_scale(level)
        span = self.tiles.tile_size * scale
        item = QGraphicsPixmapItem(QPixmap.fromImage(image))
        item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        item.setPos(tx * span, ty * span)
        item.setScale(scale)
        item.setZValue(-1000 + level)
        self.scene().addItem(item)
        self.tile_items[key] = item
        if not self.pending:
            self._update_tiles()       # nettoie les niveaux devenus inutiles

    # -- Zoom et deplacement -----------------------------------------------------

    def fit_all(self) -> None:
        if not self.isVisible():
            # Avant l'affichage, la vue n'a pas encore sa taille : cadrer plus tard.
            self._pending_fit = None
            return
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.min_scale = self.transform().m11() * 0.5
        self._zoom_changed()

    def fit_rect(self, rect: QRectF, margin: float = 0.08) -> None:
        if not self.isVisible():
            self._pending_fit = (rect, margin)
            return
        if self.min_scale <= 1e-4:
            self.fit_all()
        grow = max(rect.width(), rect.height()) * margin
        self.fitInView(rect.adjusted(-grow, -grow, grow, grow), Qt.AspectRatioMode.KeepAspectRatio)
        if self.transform().m11() > MAX_ZOOM:
            factor = MAX_ZOOM / self.transform().m11()
            self.scale(factor, factor)
        self._zoom_changed()

    def zoom_by(self, factor: float) -> None:
        current = self.transform().m11()
        target = min(MAX_ZOOM, max(self.min_scale, current * factor))
        if target != current:
            self.scale(target / current, target / current)
            self._zoom_changed()

    def _zoom_changed(self) -> None:
        self.zoomChanged.emit(self.transform().m11())
        self.schedule()

    def wheelEvent(self, event) -> None:
        self.zoom_by(1.25 if event.angleDelta().y() > 0 else 0.8)

    def scrollContentsBy(self, dx: int, dy: int) -> None:
        super().scrollContentsBy(dx, dy)
        self.schedule()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self.tiles is not None and self._pending_fit is not False:
            pending, self._pending_fit = self._pending_fit, False
            QTimer.singleShot(0, lambda: self.fit_all() if pending is None else self.fit_rect(*pending))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.schedule()

    def mousePressEvent(self, event) -> None:
        if isinstance(self.itemAt(event.position().toPoint()), HandleItem):
            self._press = None
            super().mousePressEvent(event)
            return
        self._press = {"start": event.position(), "last": event.position(),
                       "dragging": False, "button": event.button()}

    def mouseMoveEvent(self, event) -> None:
        self.hovered.emit(self.mapToScene(event.position().toPoint()))
        press = self._press
        if press is None:
            super().mouseMoveEvent(event)
            return
        delta = event.position() - press["last"]
        if not press["dragging"] and (event.position() - press["start"]).manhattanLength() > 4:
            press["dragging"] = True
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
        if press["dragging"]:
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
            press["last"] = event.position()

    def mouseReleaseEvent(self, event) -> None:
        press, self._press = self._press, None
        if press is None:
            super().mouseReleaseEvent(event)
            return
        self.viewport().unsetCursor()
        if not press["dragging"]:
            self.clicked.emit(self.mapToScene(event.position().toPoint()), press["button"], event.modifiers())

    def tolerance(self, pixels: float = 9.0) -> float:
        """Distance en pixels d'image correspondant a `pixels` a l'ecran."""
        return pixels / max(self.transform().m11(), 1e-9)

    # -- Calques -------------------------------------------------------------------

    def set_zones(self, zones: list[dict], selected: int | None, grid_visible: bool) -> None:
        for zone in zones:
            item = self.zone_items.get(zone["id"])
            if item is None:
                item = QGraphicsRectItem(zone["x"], zone["y"], zone["width"], zone["height"])
                item.setZValue(5)
                self.scene().addItem(item)
                self.zone_items[zone["id"]] = item
            chosen = zone["id"] == selected
            pen = QPen(QColor("#2f7de1") if chosen else QColor(STATUS_COLORS.get(zone["status"], "#999")),
                       3 if chosen else 1)
            pen.setCosmetic(True)
            item.setPen(pen)
            if zone["status"] == "done" and not chosen:
                item.setBrush(QColor(47, 158, 85, 28))
            else:
                item.setBrush(Qt.BrushStyle.NoBrush)
            item.setVisible(grid_visible or chosen)

    def set_cracks(self, cracks: list[dict], selected: int | None, visible: bool = True) -> None:
        for item in self.crack_items:
            self.scene().removeItem(item)
        self.crack_items.clear()
        for crack in cracks:
            if crack["state"] == "rejected" or len(crack["geometry"]) < 2:
                continue
            suggestion = crack.get("suggestion") if crack["state"] == "proposed" else None
            category = suggestion["category"] if suggestion else crack["category"]
            path = QPainterPath(QPointF(*crack["geometry"][0]))
            for x, y in crack["geometry"][1:]:
                path.lineTo(x, y)
            pen = QPen(QColor(CATEGORY_COLORS.get(category, "#d28b00")), 4 if crack["id"] == selected else 2)
            pen.setCosmetic(True)
            if suggestion or crack["state"] == "proposed":
                pen.setStyle(Qt.PenStyle.DashLine)
            item = QGraphicsPathItem(path)
            item.setPen(pen)
            item.setZValue(30 if crack["id"] == selected else 20)
            item.setVisible(visible)
            item.setData(0, crack["id"])
            self.scene().addItem(item)
            self.crack_items.append(item)

    def set_overlays(self, overlays: list, opacity: float) -> None:
        """Calques de diagnostic : [(rectangle de la tuile, QImage, net)], sous les fissures.

        `net` : agrandir sans lisser, pour que les traits fins d'un squelette
        ou d'un masque restent des traits.
        """
        for item in self.overlay_items:
            self.scene().removeItem(item)
        self.overlay_items.clear()
        for rect, image, sharp in overlays:
            item = QGraphicsPixmapItem(QPixmap.fromImage(image))
            item.setTransformationMode(Qt.TransformationMode.FastTransformation if sharp
                                       else Qt.TransformationMode.SmoothTransformation)
            item.setPos(rect.x(), rect.y())
            item.setScale(rect.width() / max(1, image.width()))
            item.setOpacity(opacity)
            item.setZValue(2 if sharp else 1)
            self.scene().addItem(item)
            self.overlay_items.append(item)

    def set_analysis(self, segments: list | None, visible: bool) -> None:
        """Resultat de l'analyse complete : un trait fin par etat, sous les fissures annotees."""
        for item in self.analysis_items:
            self.scene().removeItem(item)
        self.analysis_items.clear()
        if not segments or not visible:
            return
        paths = {}
        for segment in segments:
            geometry = segment["geometry"]
            if len(geometry) < 2:
                continue
            path = paths.setdefault(segment["category"], QPainterPath())
            path.moveTo(QPointF(*geometry[0]))
            for x, y in geometry[1:]:
                path.lineTo(x, y)
        for category, path in paths.items():
            item = QGraphicsPathItem(path)
            pen = QPen(QColor(CATEGORY_COLORS.get(category, "#d28b00")), 2)
            pen.setCosmetic(True)
            item.setPen(pen)
            item.setZValue(15)
            item.setOpacity(0.85)
            self.scene().addItem(item)
            self.analysis_items.append(item)

    def set_overlay_opacity(self, opacity: float) -> None:
        for item in self.overlay_items:
            item.setOpacity(opacity)

    def set_cracks_visible(self, visible: bool) -> None:
        for item in self.crack_items:
            item.setVisible(visible)

    def set_handles(self, anchors: list | None, moved=None, removed=None) -> None:
        for item in self.handle_items:
            self.scene().removeItem(item)
        self.handle_items.clear()
        for index, (x, y) in enumerate(anchors or []):
            item = HandleItem(index, x, y, moved, removed)
            self.scene().addItem(item)
            self.handle_items.append(item)

    def set_marker(self, point: tuple[float, float] | None) -> None:
        if self.marker is not None:
            self.scene().removeItem(self.marker)
            self.marker = None
        if point is not None:
            self.marker = QGraphicsEllipseItem(-6, -6, 12, 12)
            self.marker.setPos(*point)
            self.marker.setBrush(QBrush(QColor("#ffd400")))
            self.marker.setPen(QPen(QColor("#202020"), 1.5))
            self.marker.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
            self.marker.setZValue(60)
            self.scene().addItem(self.marker)
