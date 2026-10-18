"""Nettoyage des traces detectes, sur toute la mosaique.

Le detecteur travaille tuile par tuile : une fissure qui traverse un bord de
tuile en sort coupee en deux, et ses bouts s'arretent a quelques pixels du
bord. Ici :

1. les bords de l'echantillon sont retires (`masque.py`) ;
2. deux bouts qui se font face a travers un bord de tuile (ou un petit trou)
   sont relies, si leurs directions se prolongent ;
3. les petits morceaux isoles qui restent (un bout de fissure sorti de la bande
   de bord, par exemple) sont retires sous `MIN_ISOLATED_UM`.
"""

from __future__ import annotations

from math import hypot

import numpy as np

from .donnees import polyline_length

JOIN_UM = 16.0                  # ecart maximal entre deux bouts relies
JOIN_COSINE = -0.5              # les deux bouts se font face (plus de 120 degres)
MIN_ISOLATED_UM = 50.0


def _direction(points, at_start: bool, reach: float):
    """Direction sortante d'une extremite, mesuree sur `reach` pixels."""
    pts = points if at_start else points[::-1]
    origin = np.asarray(pts[0], dtype=float)
    for p in pts[1:]:
        v = origin - np.asarray(p, dtype=float)
        if hypot(*v) >= reach:
            return v / hypot(*v)
    v = origin - np.asarray(pts[-1], dtype=float)
    n = hypot(*v)
    return v / n if n else np.zeros(2)


def join_across_gaps(lines: list[list], pixel_um: float, join_um: float = JOIN_UM) -> list[list]:
    """Relie les bouts libres qui se font face a moins de `join_um`."""
    tolerance = join_um / pixel_um
    reach = 20.0 / pixel_um
    lines = [list(map(list, line)) for line in lines if len(line) >= 2]
    while True:
        ends = []
        for index, line in enumerate(lines):
            for at_start in (True, False):
                point = np.asarray(line[0] if at_start else line[-1], dtype=float)
                ends.append((index, at_start, point, _direction(line, at_start, reach)))
        if len(ends) < 2:
            return lines
        points = np.array([e[2] for e in ends])
        from scipy.spatial import cKDTree
        pairs = cKDTree(points).query_pairs(tolerance)
        candidates = []
        for i, j in pairs:
            (a, sa, pa, da), (b, sb, pb, db) = ends[i], ends[j]
            if a == b:
                continue
            gap = pb - pa
            distance = hypot(*gap)
            facing = float(np.dot(da, db)) <= JOIN_COSINE
            if distance > 1e-6:
                # chaque bout doit pointer vers l'autre
                facing &= float(np.dot(da, gap / distance)) > 0.3 and float(np.dot(db, -gap / distance)) > 0.3
            if facing:
                candidates.append((distance, a, sa, b, sb))
        if not candidates:
            return lines
        distance, a, sa, b, sb = min(candidates)
        first = lines[a] if not sa else lines[a][::-1]          # finit au bout relie
        second = lines[b] if sb else lines[b][::-1]             # commence au bout relie
        merged = first + second
        lines = [line for k, line in enumerate(lines) if k not in (a, b)] + [merged]


def clean(proposals: list[list], mask, pixel_um: float, log=None) -> list[list]:
    """Traces detectes (toutes tuiles) -> fissures nettoyees, en pixels de la mosaique."""
    kept = []
    removed_edge = 0.0
    for line in proposals:
        pieces = mask.keep_pieces(line) if mask is not None else [line]
        removed_edge += polyline_length(line) - sum(polyline_length(p) for p in pieces)
        kept.extend(pieces)
    joined = join_across_gaps(kept, pixel_um)
    minimum = MIN_ISOLATED_UM / pixel_um
    final = [line for line in joined if polyline_length(line) >= minimum]
    if log:
        log(f"  nettoyage : {len(proposals)} traces -> {len(final)} fissures ; bords retires "
            f"{removed_edge * pixel_um / 1000:.1f} mm ; {len(joined) - len(final)} morceaux isoles < "
            f"{MIN_ISOLATED_UM:g} um retires")
    return final
