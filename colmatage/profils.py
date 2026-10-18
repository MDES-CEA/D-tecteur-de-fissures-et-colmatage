"""Descripteurs du gel d'alteration le long d'une fissure tracee.

La fissure est d'abord « redressee » : on echantillonne l'image en pleine
resolution perpendiculairement au trace, tout du long. Elle devient une bande
horizontale, de largeur variable, ses deux parois de part et d'autre. On la
decoupe ensuite en fenetres de `WINDOW_UM` de long, une tous les `STEP_UM`, et
chaque fenetre recoit des descripteurs de trois familles, toutes mesurees
RELATIVEMENT A LA MATRICE VOISINE (jamais en niveau de gris absolu : c'est ce
qui avait fait classer l'eclairage du MEB a la premiere tentative) :

- le coeur : un vide noir, et sa largeur ;
- les parois : une bande d'epaisseur PHYSIQUE fixe le long de chaque paroi, ou
  l'on mesure niveau, texture granuleuse et taches sombres. Les bandes partent
  des parois, pas du centre : une fissure de 2 um et une de 20 um sont decrites
  de la meme facon, et le lisere de gel a une epaisseur qui ne depend pas de la
  largeur de la fissure. Chaque cote est mesure a part : le gel peut n'etre que
  d'un cote ;
- le voisinage du trace sans vide : pour une fissure pleine, ou le trace longe
  souvent un bord du gel.

Mesure sur les premiers traces etiquetes (profil median, niveau / matrice) :

    au centre      a 10 px de l'axe
    non colmatee   0,08     0,97     parois franches
    colmatee       0,90     0,94     gel gris, craquele, marbre
    rouverte       0,12     0,89     vide, mais lisere granuleux
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates

FEATURES_VERSION = 1
STEP_UM = 10.0            # une fenetre tous les 10 um
WINDOW_UM = 20.0          # longueur d'une fenetre le long de la fissure
HALF_UM = 45.0            # demi-largeur du profil, de part et d'autre du trace
SEARCH_UM = 9.0           # le trace peut etre decale du vide de quelques um
NEAR_UM = 6.0             # bande de paroi interne
FAR_WALL_UM = 13.0        # ... et externe
MATRIX_UM = 30.0          # au-dela, la matrice de reference
CENTRE_UM = 7.5           # voisinage du trace, pour une fissure sans vide
VOID_LEVEL = 0.45         # coeur noir : sous 45 % de la matrice
EDGE_LEVEL = 0.6          # paroi : on sort du vide au-dessus de 60 %
DARK_LEVEL = 0.8          # tache sombre, apres leger lissage
BRIGHT_LEVEL = 1.25       # particule claire

FEATURE_NAMES = (
    "void_width_um", "void_level", "void_fraction",
    "near_level_min", "near_level_max", "near_texture_min", "near_texture_max",
    "near_dark_min", "near_dark_max", "near_bright_max",
    "far_level_min", "far_texture_max", "far_dark_max",
    "wall_roughness_um", "edge_contrast",
    "centre_level", "centre_texture", "centre_dark", "centre_gel",
    "matrix_texture", "asymmetry",
)


@dataclass
class Window:
    distance_um: float        # abscisse curviligne du centre de la fenetre
    x: float                  # position du centre, pixels de l'image
    y: float
    features: np.ndarray      # dans l'ordre de FEATURE_NAMES


def straighten(read, width: int, height: int, points, half_px: int, smooth_px: float):
    """Image redressee (longueur x 2*half_px+1) et l'axe echantillonne (x, y)."""
    pts = np.asarray(points, dtype=float)
    steps = np.hypot(*np.diff(pts, axis=0).T)
    keep = np.r_[True, steps > 0]
    pts = pts[keep]
    if len(pts) < 2:
        return None, None
    arc = np.r_[0.0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    t = np.arange(0.0, arc[-1], 1.0)
    if len(t) < 3:
        return None, None
    x = np.interp(t, arc, pts[:, 0])
    y = np.interp(t, arc, pts[:, 1])
    # Direction lissee : le trace route a la resolution de reference fait des
    # marches de quelques pixels, qui feraient tourner les normales.
    tx = np.gradient(gaussian_filter1d(x, smooth_px, mode="nearest"))
    ty = np.gradient(gaussian_filter1d(y, smooth_px, mode="nearest"))
    norm = np.hypot(tx, ty) + 1e-9
    nx, ny = -ty / norm, tx / norm
    offsets = np.arange(-half_px, half_px + 1)
    X = x[:, None] + nx[:, None] * offsets[None, :]
    Y = y[:, None] + ny[:, None] * offsets[None, :]
    x0, y0 = int(max(0, np.floor(X.min()) - 1)), int(max(0, np.floor(Y.min()) - 1))
    x1, y1 = int(min(width, np.ceil(X.max()) + 2)), int(min(height, np.ceil(Y.max()) + 2))
    window = read(x0, y0, x1 - x0, y1 - y0).astype(np.float32)
    strip = map_coordinates(window, [Y - y0, X - x0], order=1, mode="nearest")
    return strip, np.stack([x, y], axis=1)


def _band(normalized, side: int, edge: int, start: int, stop: int):
    """Pixels d'une bande parallele a la paroi, de start a stop px de celle-ci."""
    width = normalized.shape[1]
    if side < 0:
        lo, hi = max(0, edge - stop), max(0, edge - start)
    else:
        lo, hi = min(width, edge + start + 1), min(width, edge + stop + 1)
    return normalized[:, lo:hi]


def _stats(band, smoothed_band, highpass_band, noise):
    if band.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    return (float(np.median(band)),
            float(highpass_band.std() / noise) if noise > 0 else np.nan,
            float((smoothed_band < DARK_LEVEL).mean()),
            float((smoothed_band > BRIGHT_LEVEL).mean()))


def window_features(strip: np.ndarray, pixel_um: float) -> np.ndarray:
    """Descripteurs d'une fenetre de fissure redressee (longueur x largeur)."""
    half = strip.shape[1] // 2
    px = lambda um: max(1, int(round(um / pixel_um)))  # noqa: E731
    far = px(MATRIX_UM)
    # Matrice : le cote le plus clair. Un gel large peut deborder d'un cote
    # jusque dans la bande de reference ; il est plus sombre que la matrice.
    left_ref = float(np.median(strip[:, :half - far + 1])) if half - far + 1 > 0 else np.nan
    right_ref = float(np.median(strip[:, half + far:])) if half + far < strip.shape[1] else np.nan
    matrix = np.nanmax([left_ref, right_ref])
    if not np.isfinite(matrix) or matrix <= 1:
        return np.full(len(FEATURE_NAMES), np.nan)
    normalized = strip / matrix
    smoothed = gaussian_filter(normalized, 1.0)
    highpass = normalized - gaussian_filter(normalized, 3.0)
    reference = np.r_[highpass[:, :max(1, half - far + 1)].ravel(), highpass[:, half + far:].ravel()]
    noise = float(reference.std()) or 1e-6

    profile = gaussian_filter1d(np.median(normalized, axis=0), 1.0)
    search = px(SEARCH_UM)
    centre = half - search + int(np.argmin(profile[half - search:half + search + 1]))
    if profile[centre] < VOID_LEVEL:
        left = centre
        while left > 0 and profile[left - 1] < EDGE_LEVEL:
            left -= 1
        right = centre
        while right < len(profile) - 1 and profile[right + 1] < EDGE_LEVEL:
            right += 1
        void_width = (right - left + 1) * pixel_um
        core = normalized[:, left:right + 1]
        void_level = float(np.median(core))
        void_fraction = float((smoothed[:, left:right + 1] < VOID_LEVEL).mean())
        # Rugosite des parois : ecart-type, le long de la fenetre, de la
        # position ou chaque ligne sort du vide.
        rows = smoothed < EDGE_LEVEL
        edges = []
        for row in rows:
            l = centre
            while l > 0 and row[l - 1]:
                l -= 1
            r = centre
            while r < len(row) - 1 and row[r + 1]:
                r += 1
            edges.append((l, r))
        edges = np.array(edges, dtype=float)
        roughness = float(np.mean(edges.std(axis=0))) * pixel_um
        edge_contrast = float(np.mean([
            profile[max(0, left - px(3.0))], profile[min(len(profile) - 1, right + px(3.0))]
        ]) - void_level)
    else:
        left = right = centre
        void_width, void_level, void_fraction = 0.0, np.nan, 0.0
        roughness, edge_contrast = np.nan, np.nan

    near, outer = px(NEAR_UM), px(FAR_WALL_UM)
    sides = []
    for side, edge in ((-1, left), (1, right)):
        args = (side, edge)
        near_stats = _stats(_band(normalized, *args, 1, near), _band(smoothed, *args, 1, near),
                            _band(highpass, *args, 1, near), noise)
        far_stats = _stats(_band(normalized, *args, near, outer), _band(smoothed, *args, near, outer),
                           _band(highpass, *args, near, outer), noise)
        sides.append((near_stats, far_stats))
    near_levels = [s[0][0] for s in sides]
    near_textures = [s[0][1] for s in sides]
    near_darks = [s[0][2] for s in sides]
    near_brights = [s[0][3] for s in sides]
    far_levels = [s[1][0] for s in sides]
    far_textures = [s[1][1] for s in sides]
    far_darks = [s[1][2] for s in sides]

    centre_px = px(CENTRE_UM)
    lo, hi = max(0, centre - centre_px), min(strip.shape[1], centre + centre_px + 1)
    centre_band, centre_smooth = normalized[:, lo:hi], smoothed[:, lo:hi]
    centre_level = float(np.median(centre_band))
    centre_texture = float(highpass[:, lo:hi].std() / noise)
    centre_dark = float((centre_smooth < DARK_LEVEL).mean())
    # Gel : ni vide ni matrice — gris intermediaire.
    centre_gel = float(((centre_smooth > VOID_LEVEL) & (centre_smooth < 0.93)).mean())
    asymmetry = float(abs(near_levels[0] - near_levels[1])) if all(np.isfinite(near_levels)) else np.nan

    return np.array([
        void_width, void_level, void_fraction,
        np.nanmin(near_levels), np.nanmax(near_levels), np.nanmin(near_textures), np.nanmax(near_textures),
        np.nanmin(near_darks), np.nanmax(near_darks), np.nanmax(near_brights),
        np.nanmin(far_levels), np.nanmax(far_textures), np.nanmax(far_darks),
        roughness, edge_contrast,
        centre_level, centre_texture, centre_dark, centre_gel,
        noise, asymmetry,
    ], dtype=float)


def trace_windows(read, width: int, height: int, points, pixel_um: float) -> list[Window]:
    """Fenetres successives d'un trace, avec leurs descripteurs."""
    half = int(round(HALF_UM / pixel_um))
    strip, axis = straighten(read, width, height, points, half, smooth_px=6.0 / pixel_um)
    if strip is None:
        return []
    length = int(round(WINDOW_UM / pixel_um))
    step = int(round(STEP_UM / pixel_um))
    windows = []
    if len(strip) < length:
        starts = [0] if len(strip) >= length // 2 else []
    else:
        starts = range(0, len(strip) - length + 1, step)
    for start in starts:
        part = strip[start:start + length]
        middle = min(len(axis) - 1, start + len(part) // 2)
        with np.errstate(all="ignore"):
            features = window_features(part, pixel_um)
        windows.append(Window((start + len(part) / 2) * pixel_um,
                              float(axis[middle, 0]), float(axis[middle, 1]), features))
    return windows
