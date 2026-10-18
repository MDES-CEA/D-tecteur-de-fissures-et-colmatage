"""Controle par sondage : quelques centaines de points tires au hasard sur les fissures.

Les points sont tires uniformement le long de la longueur detectee : chacun
represente la meme longueur de fissure. L'annotateur dit, pour chacun, ce
qu'il voit (non colmatee, colmatee, rouverte, ou pas une fissure), sans voir
la reponse de la machine.

On en tire :
- la part de longueur detectee qui n'est pas une fissure ;
- la part colmatee CORRIGEE : celle de la machine, plus l'ecart moyen entre
  l'annotateur et la machine sur les points controles. C'est une estimation
  sans biais meme si la machine se trompe, avec son intervalle de confiance
  (estimation « assistee par prediction »).
"""

from __future__ import annotations

import json
from datetime import datetime
from math import sqrt
from pathlib import Path

import numpy as np

ANSWERS = ("open", "filled", "reopened", "not_crack")
SEALED = {"filled", "reopened"}


def draw_points(result: dict, count: int = 200, seed: int = 287298) -> list[dict]:
    """Points uniformes le long des troncons du resultat."""
    segments = result["segments"]
    lengths = np.array([s["length_um"] for s in segments], dtype=float)
    if not len(lengths) or lengths.sum() <= 0:
        return []
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.uniform(0, lengths.sum(), count))
    bounds = np.cumsum(lengths)
    points = []
    for position in positions:
        k = int(np.searchsorted(bounds, position))
        k = min(k, len(segments) - 1)
        offset = position - (bounds[k] - lengths[k])
        segment = segments[k]
        geometry = np.asarray(segment["geometry"], dtype=float)
        steps = np.hypot(*np.diff(geometry, axis=0).T)
        arc = np.r_[0, np.cumsum(steps)]
        along = offset / max(segment["length_um"], 1e-9) * arc[-1]
        x = float(np.interp(along, arc, geometry[:, 0]))
        y = float(np.interp(along, arc, geometry[:, 1]))
        points.append({"x": round(x, 1), "y": round(y, 1), "predicted": segment["category"],
                       "segment": k})
    order = rng.permutation(len(points))              # ordre de presentation au hasard
    return [points[i] for i in order]


class Audit:
    """Points a controler et reponses, enregistres a chaque reponse."""

    def __init__(self, path: Path, result: dict, count: int = 200) -> None:
        self.path = Path(path)
        stamp = result.get("analyse_le")
        data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else None
        if data is None or data.get("resultat") != stamp:
            data = {"resultat": stamp, "cree_le": datetime.now().isoformat(timespec="seconds"),
                    "points": draw_points(result, count), "reponses": {}}
        self.data = data
        self.result = result

    @property
    def points(self) -> list[dict]:
        return self.data["points"]

    def answer(self, index: int, value: str | None) -> None:
        if value is None:
            self.data["reponses"].pop(str(index), None)
        else:
            if value not in ANSWERS:
                raise ValueError(value)
            self.data["reponses"][str(index)] = value
        self.save()

    def next_index(self) -> int | None:
        for index in range(len(self.points)):
            if str(index) not in self.data["reponses"]:
                return index
        return None

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
        temporary.replace(self.path)

    def estimate(self) -> dict:
        return estimate(self.points, self.data["reponses"], self.result["resume"]["part_colmatee"])


def estimate(points, answers: dict, machine_sealed: float) -> dict:
    """Part colmatee corrigee par le sondage, et part de « pas une fissure »."""
    answered = [(points[int(i)], a) for i, a in answers.items()]
    n = len(answered)
    if not n:
        return {"points": 0}
    not_crack = sum(a == "not_crack" for _, a in answered)
    cracks = [(p, a) for p, a in answered if a != "not_crack"]
    out = {"points": n, "pas_une_fissure": not_crack / n,
           "pas_une_fissure_ic95": _wilson(not_crack, n)}
    if cracks:
        human = np.array([a in SEALED for _, a in cracks], dtype=float)
        machine = np.array([p["predicted"] in SEALED for p, _ in cracks], dtype=float)
        difference = human - machine
        correction = float(difference.mean())
        spread = float(difference.std(ddof=1)) if len(cracks) > 1 else 1.0
        corrected = machine_sealed + correction
        half = 1.96 * spread / sqrt(len(cracks))
        agreement = float(np.mean([(p["predicted"] == a) for p, a in cracks]))
        confusion = {f"{p}->{a}": 0 for p in ("open", "filled", "reopened") for a in ("open", "filled", "reopened")}
        for p, a in cracks:
            confusion[f"{p['predicted']}->{a}"] += 1
        out.update({"fissures_controlees": len(cracks), "accord_3_classes": agreement,
                    "accord_colmatee_oui_non": float((human == machine).mean()),
                    "part_colmatee_machine": machine_sealed, "correction_points": 100 * correction,
                    "part_colmatee_corrigee": corrected,
                    "part_colmatee_corrigee_ic95": [max(0.0, corrected - half), min(1.0, corrected + half)],
                    "confusion": confusion})
    return out


def _wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if not n:
        return [0.0, 1.0]
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, center - half), min(1.0, center + half)]
