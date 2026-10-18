"""Bandes redressees le long des fissures, decoupees en fenetres pour le classement.

Comme `colmatage.profils.straighten`, l'image est echantillonnee
perpendiculairement au trace, tout du long : la fissure devient une bande
horizontale, ses deux parois de part et d'autre. Deux differences :

- la lecture se fait par morceaux de trace : une fissure de 10 mm en diagonale
  ne charge pas un carre de 10 mm de cote ;
- la bande est normalisee par le niveau de la matrice voisine (1 = matrice,
  0 = vide noir), puis reduite a ~0,75 um par pixel : assez pour voir un lisere
  de gel de quelques um, quatre fois moins de calcul qu'en pleine resolution.

Une fenetre fait 24 um le long du trace et +/- 44 um en travers, une tous les
10 um : la meme grille que le modele de colmatage precedent.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates

STEP_UM = 10.0              # une fenetre tous les 10 um
WINDOW_UM = 24.0            # longueur le long du trace
HALF_UM = 44.5              # demi-largeur en travers
MATRIX_UM = 30.0            # au-dela, la matrice de reference
TARGET_UM = 0.75            # resolution des fenetres
ACROSS = 120                # pixels en travers (fenetre reduite)
CHUNK_PX = 2048             # longueur de trace lue d'un coup


def reduction_for(pixel_um: float) -> int:
    return max(1, int(round(TARGET_UM / pixel_um)))


def sample_axis(points) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Axe du trace echantillonne tous les pixels, et ses normales lissees."""
    pts = np.asarray(points, dtype=float)
    steps = np.hypot(*np.diff(pts, axis=0).T)
    pts = pts[np.r_[True, steps > 0]]
    if len(pts) < 2:
        return None, None
    arc = np.r_[0.0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    t = np.arange(0.0, arc[-1], 1.0)
    if len(t) < 3:
        return None, None
    x = np.interp(t, arc, pts[:, 0])
    y = np.interp(t, arc, pts[:, 1])
    return np.stack([x, y], axis=1), arc[-1]


def straighten(read, width: int, height: int, points, pixel_um: float):
    """Bande redressee (longueur x largeur), pleine resolution, lue par morceaux."""
    axis, _ = sample_axis(points)
    if axis is None:
        return None, None
    half = int(round(HALF_UM / pixel_um)) + reduction_for(pixel_um)
    smooth = 6.0 / pixel_um
    tx = np.gradient(gaussian_filter1d(axis[:, 0], smooth, mode="nearest"))
    ty = np.gradient(gaussian_filter1d(axis[:, 1], smooth, mode="nearest"))
    norm = np.hypot(tx, ty) + 1e-9
    nx, ny = -ty / norm, tx / norm
    offsets = np.arange(-half, half + 1)
    strip = np.empty((len(axis), len(offsets)), dtype=np.float32)
    for start in range(0, len(axis), CHUNK_PX):
        stop = min(len(axis), start + CHUNK_PX)
        X = axis[start:stop, 0, None] + nx[start:stop, None] * offsets[None, :]
        Y = axis[start:stop, 1, None] + ny[start:stop, None] * offsets[None, :]
        x0, y0 = int(max(0, np.floor(X.min()) - 1)), int(max(0, np.floor(Y.min()) - 1))
        x1, y1 = int(min(width, np.ceil(X.max()) + 2)), int(min(height, np.ceil(Y.max()) + 2))
        if x1 <= x0 or y1 <= y0:
            strip[start:stop] = 0
            continue
        block = np.asarray(read(x0, y0, x1 - x0, y1 - y0), dtype=np.float32)
        strip[start:stop] = map_coordinates(block, [Y - y0, X - x0], order=1, mode="nearest")
    return strip, axis


def windows(read, width: int, height: int, points, pixel_um: float):
    """Fenetres normalisees d'un trace : (n, ACROSS, along) float16, centres en um, positions (x, y).

    Rend (None, [], []) si le trace est trop court pour une fenetre.
    """
    strip, axis = straighten(read, width, height, points, pixel_um)
    if strip is None:
        return None, np.array([]), np.zeros((0, 2))
    r = reduction_for(pixel_um)
    length = int(round(WINDOW_UM / pixel_um))
    step = max(1, int(round(STEP_UM / pixel_um)))
    if len(strip) < length // 2:
        return None, np.array([]), np.zeros((0, 2))
    if len(strip) < length:
        strip = np.pad(strip, ((0, length - len(strip)), (0, 0)), mode="edge")
        starts = [0]
    else:
        starts = list(range(0, len(strip) - length + 1, step))
    half = strip.shape[1] // 2
    far = int(round(MATRIX_UM / pixel_um))
    along = length // r
    out = np.empty((len(starts), ACROSS, along), dtype=np.float16)
    centers, positions = [], []
    for k, start in enumerate(starts):
        part = strip[start:start + length]
        left = np.median(part[:, :max(1, half - far)])
        right = np.median(part[:, half + far:]) if half + far < part.shape[1] else left
        matrix = max(left, right, 1.0)
        part = part / matrix
        # Reduction par moyenne de blocs r x r, puis recentrage a ACROSS pixels.
        rows = (part.shape[0] // r) * r
        cols = (part.shape[1] // r) * r
        reduced = part[:rows, :cols].reshape(rows // r, r, cols // r, r).mean(axis=(1, 3))
        mid = reduced.shape[1] // 2
        lo = mid - ACROSS // 2
        across = reduced[:, max(0, lo):lo + ACROSS]
        if across.shape[1] < ACROSS:
            across = np.pad(across, ((0, 0), (0, ACROSS - across.shape[1])), mode="edge")
        out[k] = np.clip(across[:along].T, 0, 3)
        center = min(len(axis) - 1, start + length // 2)
        centers.append((start + length / 2) * pixel_um)
        positions.append(axis[center])
    return out, np.array(centers), np.array(positions)
