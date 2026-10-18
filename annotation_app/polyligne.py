"""Geometrie des traces : abscisse le long d'une polyligne, projection, decoupe.

Une fissure est une polyligne (sa geometrie routee) passant par ses noeuds.
La couper, c'est couper la geometrie a une abscisse donnee, sans la recalculer,
et repartir les noeuds entre les morceaux.
"""

from __future__ import annotations

from math import hypot


def cumulative(points) -> list[float]:
    """Abscisse curviligne de chaque sommet, en pixels."""
    result = [0.0]
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        result.append(result[-1] + hypot(x2 - x1, y2 - y1))
    return result


def project(points, x: float, y: float) -> tuple[float, tuple[float, float], float]:
    """(abscisse, point projete, distance) du point le plus proche sur la polyligne."""
    best = None
    along = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        dx, dy = x2 - x1, y2 - y1
        size = hypot(dx, dy)
        t = 0.0 if not size else max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / (size * size)))
        px, py = x1 + t * dx, y1 + t * dy
        distance = hypot(px - x, py - y)
        if best is None or distance < best[2]:
            best = (along + t * size, (px, py), distance)
        along += size
    return best


def point_at(points, position: float, arc=None) -> tuple[float, float]:
    arc = arc or cumulative(points)
    position = max(0.0, min(arc[-1], position))
    for index in range(1, len(points)):
        if arc[index] >= position:
            span = arc[index] - arc[index - 1]
            t = 0.0 if not span else (position - arc[index - 1]) / span
            (x1, y1), (x2, y2) = points[index - 1], points[index]
            return x1 + t * (x2 - x1), y1 + t * (y2 - y1)
    return tuple(points[-1])


def cut(points, start: float, end: float) -> list[list[float]]:
    """Morceau de la polyligne entre les abscisses `start` et `end`."""
    arc = cumulative(points)
    piece = [list(point_at(points, start, arc))]
    piece += [list(p) for p, a in zip(points, arc) if start < a < end]
    piece.append(list(point_at(points, end, arc)))
    return [[round(x, 2), round(y, 2)] for x, y in piece]


def pieces(anchors, geometry, cuts: list[float]) -> list[tuple[list, list]]:
    """Coupe un trace aux abscisses `cuts` (pixels). Rend [(noeuds, geometrie)].

    Chaque morceau garde les noeuds qui tombent chez lui, plus ses deux bouts :
    deplacer ensuite un de ses noeuds ne recalcule que ce morceau.
    """
    total = cumulative(geometry)[-1]
    bounds = [0.0, *sorted(cuts), total]
    positions = [project(geometry, x, y)[0] for x, y in anchors]
    result = []
    for start, end in zip(bounds, bounds[1:]):
        shape = cut(geometry, start, end)
        inner = [[round(x, 2), round(y, 2)] for (x, y), p in zip(anchors, positions)
                 if start + 1.0 < p < end - 1.0]
        result.append(([shape[0], *inner, shape[-1]], shape))
    return result
