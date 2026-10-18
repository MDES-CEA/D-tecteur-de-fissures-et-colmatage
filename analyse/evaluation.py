"""Mesures d'un classement le long des fissures, contre les etiquettes de l'annotateur.

Tout se compte en LONGUEUR (la mesure finale est une longueur), plus les
changements d'etat : un changement propose est juste s'il tombe a moins de
`tolerance_um` d'un changement de l'annotateur, chacun ne servant qu'une fois.
"""

from __future__ import annotations

import numpy as np

CATEGORIES = ("open", "filled", "reopened")
SEALED = {"filled", "reopened"}


def overlap_matrix(true_segments, predicted_segments) -> np.ndarray:
    """Longueur (meme unite que les bornes) de chaque couple (vraie categorie, predite)."""
    matrix = np.zeros((3, 3))
    for t in true_segments:
        if t["category"] not in CATEGORIES:
            continue
        for p in predicted_segments:
            length = min(t["end_um"], p["end_um"]) - max(t["start_um"], p["start_um"])
            if length > 0:
                matrix[CATEGORIES.index(t["category"]), CATEGORIES.index(p["category"])] += length
    return matrix


def changes(segments_list) -> list[float]:
    """Positions des changements d'etat entre deux categories connues et differentes."""
    return [b["start_um"] for a, b in zip(segments_list, segments_list[1:])
            if a["category"] != b["category"] and a["category"] in CATEGORIES and b["category"] in CATEGORIES]


def match_changes(true_positions, predicted_positions, tolerance_um: float = 20.0) -> int:
    """Nombre de changements predits qui retrouvent un changement vrai (un pour un)."""
    pairs = sorted((abs(t - p), i, j) for i, t in enumerate(true_positions)
                   for j, p in enumerate(predicted_positions) if abs(t - p) <= tolerance_um)
    used_t, used_p, hits = set(), set(), 0
    for _, i, j in pairs:
        if i not in used_t and j not in used_p:
            used_t.add(i)
            used_p.add(j)
            hits += 1
    return hits


class Score:
    """Cumule les fissures evaluees ; `summary()` rend les chiffres."""

    def __init__(self, tolerance_um: float = 20.0) -> None:
        self.tolerance = tolerance_um
        self.matrix = np.zeros((3, 3))
        self.true_changes = self.predicted_changes = self.matched = 0
        self.per_tile: dict = {}

    def add(self, true_segments, predicted_segments, tile=None) -> None:
        matrix = overlap_matrix(true_segments, predicted_segments)
        self.matrix += matrix
        t, p = changes(true_segments), changes(predicted_segments)
        self.true_changes += len(t)
        self.predicted_changes += len(p)
        self.matched += match_changes(t, p, self.tolerance)
        if tile is not None:
            self.per_tile[tile] = self.per_tile.get(tile, np.zeros((3, 3))) + matrix

    @staticmethod
    def sealed_fraction(matrix, predicted: bool) -> float:
        total = matrix.sum()
        if not total:
            return float("nan")
        axis = 0 if predicted else 1
        per_category = matrix.sum(axis=axis)
        return float(per_category[1:].sum() / total)

    def summary(self) -> dict:
        m = self.matrix
        total = m.sum() or 1.0
        sealed_true = self.sealed_fraction(m, predicted=False)
        sealed_pred = self.sealed_fraction(m, predicted=True)
        binary = (m[0, 0] + m[1:, 1:].sum()) / total
        precision = self.matched / self.predicted_changes if self.predicted_changes else 0.0
        recall = self.matched / self.true_changes if self.true_changes else 0.0
        tiles = [(self.sealed_fraction(v, False), self.sealed_fraction(v, True))
                 for v in self.per_tile.values() if v.sum() >= 500.0]
        errors = [abs(a - b) for a, b in tiles]
        return {
            "longueur_mm": float(m.sum() / 1000),
            "longueur_bien_classee": float(np.trace(m) / total),
            "colmatee_oui_non": float(binary),
            "part_colmatee_vraie": sealed_true,
            "part_colmatee_predite": sealed_pred,
            "biais_points": 100 * (sealed_pred - sealed_true),
            "rappel_par_categorie": {c: float(m[i, i] / m[i].sum()) if m[i].sum() else None
                                     for i, c in enumerate(CATEGORIES)},
            "changements_vrais": self.true_changes,
            "changements_predits": self.predicted_changes,
            "changements_justes": self.matched,
            "changements_precision": precision,
            "changements_rappel": recall,
            "changements_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "tuiles": len(tiles),
            "erreur_part_par_tuile_points": 100 * float(np.mean(errors)) if errors else None,
            "confusion_mm": (m / 1000).round(2).tolist(),
        }


def clip_to_rect(points, rect, spacing: float = 8.0) -> list[list]:
    """Morceaux d'une polyligne a l'interieur d'un rectangle (x, y, largeur, hauteur)."""
    from .masque import densify

    x0, y0, w, h = rect
    pieces, current = [], []
    for p in densify(points, spacing):
        if x0 <= p[0] < x0 + w and y0 <= p[1] < y0 + h:
            current.append(p)
        elif current:
            pieces.append(current)
            current = []
    if current:
        pieces.append(current)
    return [p for p in pieces if len(p) >= 2]


def _outside(points, unknown, zone, scale) -> list[list]:
    """Morceaux d'une polyligne hors de la zone inconnue (raster a l'echelle `scale`)."""
    from .masque import densify

    if not unknown.any():
        return [points]
    pieces, current = [], []
    for p in densify(points, scale):
        r = min(unknown.shape[0] - 1, max(0, int((p[1] - zone["y"]) / scale)))
        c = min(unknown.shape[1] - 1, max(0, int((p[0] - zone["x"]) / scale)))
        if not unknown[r, c]:
            current.append(p)
        elif current:
            pieces.append(current)
            current = []
    if current:
        pieces.append(current)
    return [p for p in pieces if len(p) >= 2]


def end_to_end(annotations, split, result, pixel_um: float, tolerance_um: float = 16.0, scale: int = 4,
               role_name: str = "test") -> dict:
    """Le resultat automatique complet contre l'annotateur, sur les tuiles d'un role (test par defaut).

    Detection : precision et rappel en pixels (tolerance `tolerance_um`, comme
    le banc), les pointilles jamais verifies ne comptant ni pour ni contre.
    Mesures : longueur par categorie et part colmatee, tuile par tuile, et le
    rapport longueur annotee / longueur automatique de chaque etat (`ratios`),
    qui sert a corriger les erreurs systematiques de la detection.
    """
    from scipy.ndimage import distance_transform_edt
    from skimage.draw import line as draw_line

    from .donnees import polyline_length, role

    def raster(lines, zone):
        mask = np.zeros((zone["height"] // scale + 1, zone["width"] // scale + 1), dtype=bool)
        for line in lines:
            pts = (np.asarray(line, dtype=float) - [zone["x"], zone["y"]]) / scale
            for (c0, r0), (c1, r1) in zip(pts, pts[1:]):
                rr, cc = draw_line(int(round(r0)), int(round(c0)), int(round(r1)), int(round(c1)))
                ok = (rr >= 0) & (cc >= 0) & (rr < mask.shape[0]) & (cc < mask.shape[1])
                mask[rr[ok], cc[ok]] = True
        return mask

    tolerance = tolerance_um / pixel_um / scale
    tiles = []
    for zone in annotations.zones:
        if zone["status"] != "done" or role(zone, split) != role_name:
            continue
        rect = (zone["x"], zone["y"], zone["width"], zone["height"])
        human = [c for c in annotations.validated if c["zone_id"] == zone["id"]]
        pending = [c["geometry"] for c in annotations.unverified if c["zone_id"] == zone["id"]]
        auto = [(s["category"], piece) for s in result["segments"] for piece in clip_to_rect(s["geometry"], rect)]
        truth = raster([c["geometry"] for c in human], zone)
        proposed = raster([piece for _, piece in auto], zone)
        unknown = np.zeros(truth.shape, dtype=bool)
        if pending:
            # Pres des pointilles jamais verifies (et loin de toute fissure classee),
            # on ne sait pas ce qui est vrai : rien n'y compte, ni detection ni longueur.
            unknown = distance_transform_edt(~raster(pending, zone)) <= tolerance
            if truth.any():
                unknown &= distance_transform_edt(~truth) > tolerance
            proposed &= ~unknown
        lengths = {"humain": {c: 0.0 for c in CATEGORIES}, "auto": {c: 0.0 for c in CATEGORIES}}
        for c in human:
            if c["category"] in CATEGORIES:
                lengths["humain"][c["category"]] += polyline_length(c["geometry"]) * pixel_um
        for category, piece in auto:
            for part in _outside(piece, unknown, zone, scale):
                lengths["auto"][category] += polyline_length(part) * pixel_um
        near_truth = int((distance_transform_edt(~truth)[proposed] <= tolerance).sum()) if truth.any() else 0
        near_auto_map = distance_transform_edt(~proposed) <= tolerance if proposed.any() else np.zeros_like(truth)
        found = {}
        for category in CATEGORIES:
            mask = raster([c["geometry"] for c in human if c["category"] == category], zone)
            found[category] = (int(near_auto_map[mask].sum()), int(mask.sum()))
        tiles.append({"zone_id": zone["id"], "ligne": zone["row_index"], "colonne": zone["col_index"],
                      "longueurs_um": lengths, "proposes": int(proposed.sum()), "proposes_justes": near_truth,
                      "vrais": int(truth.sum()), "vrais_trouves": int(near_auto_map[truth].sum()),
                      "trouves_par_categorie": found})

    def sealed(d):
        total = sum(d.values())
        return (d["filled"] + d["reopened"]) / total if total else float("nan")

    human_total = {c: sum(t["longueurs_um"]["humain"][c] for t in tiles) for c in CATEGORIES}
    auto_total = {c: sum(t["longueurs_um"]["auto"][c] for t in tiles) for c in CATEGORIES}
    precision = sum(t["proposes_justes"] for t in tiles) / max(1, sum(t["proposes"] for t in tiles))
    recall = sum(t["vrais_trouves"] for t in tiles) / max(1, sum(t["vrais"] for t in tiles))
    per_tile = [(sealed(t["longueurs_um"]["humain"]), sealed(t["longueurs_um"]["auto"])) for t in tiles
                if sum(t["longueurs_um"]["humain"].values()) >= 500 and sum(t["longueurs_um"]["auto"].values()) >= 500]
    recall_by_category = {}
    for category in CATEGORIES:
        hit = sum(t["trouves_par_categorie"][category][0] for t in tiles)
        total = sum(t["trouves_par_categorie"][category][1] for t in tiles)
        recall_by_category[category] = hit / total if total else None
    return {
        "tuiles": len(tiles),
        "detection_precision": precision,
        "detection_rappel": recall,
        "detection_rappel_par_categorie": recall_by_category,
        "longueur_humaine_mm": sum(human_total.values()) / 1000,
        "longueur_auto_mm": sum(auto_total.values()) / 1000,
        "part_colmatee_humaine": sealed(human_total),
        "part_colmatee_auto": sealed(auto_total),
        "longueur_humaine_par_etat_mm": {c: v / 1000 for c, v in human_total.items()},
        "longueur_auto_par_etat_mm": {c: v / 1000 for c, v in auto_total.items()},
        "ratios": {c: human_total[c] / auto_total[c] if auto_total[c] else None for c in CATEGORIES},
        "erreur_part_par_tuile_points": 100 * float(np.mean([abs(a - b) for a, b in per_tile])) if per_tile else None,
        "detail": tiles,
    }


MODEL_VALIDATION_NAME = "analyse_validation.json"


def validation_for(out_dir) -> dict | None:
    """La mesure de la methode pour une mosaique : ses propres tuiles de test si elle en a,
    sinon celle faite sur la mosaique d'apprentissage des modeles (fichier joint aux modeles)."""
    import json
    from pathlib import Path

    from annotation_app.paths import MODELS_DIR

    path = Path(out_dir) / "evaluation_test.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("tuiles"):
            return {**data, "source": "tuiles de test de cette mosaïque"}
    path = MODELS_DIR / MODEL_VALIDATION_NAME
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return {**data, "source": f"tuiles de test de la mosaïque d'apprentissage ({data.get('mosaique')})"}
    return None


def corrected(lengths_mm: dict, ratios: dict) -> dict:
    """Longueurs par etat corrigees des erreurs systematiques mesurees sur les tuiles de test,
    et la part colmatee qui en resulte."""
    fixed = {c: lengths_mm.get(c, 0.0) * (ratios.get(c) or 1.0) for c in CATEGORIES}
    total = sum(fixed.values())
    return {"longueur_mm": fixed, "longueur_totale_mm": total,
            "part_colmatee": (fixed["filled"] + fixed["reopened"]) / total if total else float("nan")}


def describe(name: str, s: dict) -> str:
    return (f"{name:34} longueur bien classee {s['longueur_bien_classee']:5.1%} | colmatee oui/non "
            f"{s['colmatee_oui_non']:5.1%} | part colmatee {s['part_colmatee_predite']:5.1%} "
            f"(vraie {s['part_colmatee_vraie']:5.1%}, biais {s['biais_points']:+5.1f} pts) | changements "
            f"{s['changements_predits']:4} proposes, {s['changements_justes']:3}/{s['changements_vrais']} retrouves "
            f"(F1 {s['changements_f1']:4.0%})"
            + (f" | erreur par tuile {s['erreur_part_par_tuile_points']:.1f} pts"
               if s.get("erreur_part_par_tuile_points") is not None else ""))
