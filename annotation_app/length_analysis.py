"""Euclidean centerline measurements; union of coincident collinear intervals.

No dilation or skeleton pixel counting is used for length. Nearby, distinct
lines are deliberately not merged. Conflicting labels form their own class.
"""
from collections import defaultdict
from math import atan2, degrees, hypot

CATEGORIES = ('open', 'filled', 'reopened', 'mixed', 'uncertain', 'unclassified', 'conflict')
LABELS = ('Non colmatée', 'Colmatée', 'Rouverte', 'Mixte', 'Incertaine', 'Non classée', 'Conflit')
COLORS = ('#d43d2f', '#087db6', '#14a38b', '#7650a8', '#666666', '#d28b00', '#ff00aa')
# Ont ete colmatees a un moment : le gel le prouve, meme si la fissure est vide.
SEALED = ('filled', 'reopened')


def nearby_overlaps(cracks, radius=2.0):
    """Review candidates only: near-parallel samples from different traces.

    The sampled extent is a diagnostic, never subtracted from the length.
    Real parallel cracks can trigger this test and must be judged on the image.
    """
    import numpy as np
    from scipy.spatial import cKDTree
    points, directions, owners = [], [], []
    for crack in cracks:
        if crack.get('state') == 'rejected':
            continue
        for a, b in zip(crack['geometry'], crack['geometry'][1:]):
            a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
            length = float(np.linalg.norm(b-a))
            if length <= 0:
                continue
            n = max(1, int(np.ceil(length/2)))
            points.extend(a+(b-a)*((np.arange(n)+.5)/n)[:,None])
            directions.extend([(b-a)/length]*n)
            owners.extend([crack['id']]*n)
    if not points:
        return []
    points, directions, owners = np.asarray(points), np.asarray(directions), np.asarray(owners)
    tree = cKDTree(points)
    # Bounded neighbour count avoids quadratic allocation on repeated traces.
    distances, indices = tree.query(points, k=min(8,len(points)), distance_upper_bound=radius)
    if indices.ndim == 1:
        return []
    hits = defaultdict(set)
    bounds = {}
    for i in range(len(points)):
        for distance, j in zip(distances[i], indices[i]):
            if not np.isfinite(distance) or j <= i or owners[i] == owners[j] or distance < .05:
                continue
            if abs(float(directions[i] @ directions[j])) < .966:
                continue
            delta = points[j]-points[i]
            if abs(float(delta[0]*directions[i,1]-delta[1]*directions[i,0])) < .05:
                continue
            pair = tuple(sorted((int(owners[i]),int(owners[j]))))
            hits[pair].add(i)
            bounds.setdefault(pair, points[i].tolist())
    return [{'trace_ids': list(pair), 'review_x': bounds[pair][0], 'review_y': bounds[pair][1],
             'matched_samples': len(samples)} for pair,samples in sorted(hits.items()) if len(samples)>=10]


def measure(cracks, tolerance_px=0):
    from .overlap_measure import measure_union
    return measure_union(cracks, tolerance_px)
