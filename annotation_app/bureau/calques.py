"""Calques de diagnostic du detecteur : mise en couleur pour l'affichage.

Les tableaux viennent du cache d'analyse (`Session.layer_array`), a la
resolution d'analyse. Les calques continus (score, couts) passent par une
palette ; les calques binaires (masque, squelette, axes) sont une couleur vive
sur fond transparent, pour voir l'image dessous.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtGui import QImage

# Dans l'ordre du menu.
LAYERS = {
    "score": "Score du détecteur",
    "mask": "Masque (score seuillé)",
    "skeleton": "Squelette du masque",
    "centerline": "Axes automatiques",
    "cost": "Coût de la détection",
    "route_cost": "Coût du tracé manuel",
}
BINARY = {"mask": (0, 229, 255), "skeleton": (255, 234, 0), "centerline": (255, 0, 200)}
PALETTES = {"score": "inferno", "cost": "viridis", "route_cost": "viridis"}
# Infobulles des menus.
HELP = {
    "score": "Probabilité de fissure par pixel (jaune = forte)",
    "mask": "Pixels classés fissure",
    "skeleton": "Ligne centrale du masque",
    "centerline": "Fissures que « Détecter » proposerait",
    "cost": "Sombre = passage facile pour la détection",
    "route_cost": "Sombre = passage facile pour le tracé manuel",
}

_LUTS: dict[str, np.ndarray] = {}


def _lut(name: str) -> np.ndarray:
    if name not in _LUTS:
        from matplotlib import colormaps
        _LUTS[name] = (colormaps[name](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    return _LUTS[name]


def reduce(values: np.ndarray, kind: str, max_side: int) -> np.ndarray:
    """Reduit un calque a `max_side` px de cote au plus.

    Un calque binaire est reduit par maximum : un trait d'un pixel reste visible.
    """
    height, width = values.shape
    step = int(np.ceil(max(height, width) / max_side))
    if step <= 1:
        return values
    out_h, out_w = -(-height // step), -(-width // step)
    padded = np.zeros((out_h * step, out_w * step), dtype=np.float32)
    padded[:height, :width] = values
    blocks = padded.reshape(out_h, step, out_w, step)
    return blocks.max(axis=(1, 3)) if kind in BINARY or kind == "score" else blocks.mean(axis=(1, 3))


def colorize(values: np.ndarray, kind: str) -> QImage:
    """QImage RGBA d'un calque."""
    height, width = values.shape
    rgba = np.zeros((height, width, 4), np.uint8)
    if kind in BINARY:
        on = values > 0
        rgba[on, :3] = BINARY[kind]
        rgba[on, 3] = 255
    else:
        levels = (np.clip(values, 0, 1) * 255).astype(np.uint8)
        rgba[..., :3] = _lut(PALETTES[kind])[levels]
        if kind == "score":
            # Un score faible est transparent : seules les fissures probables ressortent.
            rgba[..., 3] = np.clip(levels.astype(np.int16) * 2, 0, 255).astype(np.uint8)
        else:
            rgba[..., 3] = 255
    rgba = np.ascontiguousarray(rgba)
    return QImage(rgba.data, width, height, width * 4, QImage.Format.Format_RGBA8888).copy()
