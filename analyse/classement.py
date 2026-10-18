"""Etat du colmatage le long des fissures : un reseau tous les 10 um, puis un lissage.

Le reseau regarde une fenetre redressee de la fissure (`bandes.py`) et repond
a deux questions, celles de la regle de l'annotateur :

    gel visible ?   non -> non colmatee (open)
                    oui -> fissure vide ?  non -> colmatee (filled)
                                           oui -> rouverte (reopened)

Chaque question a beaucoup d'exemples (le gel : colmatees + rouvertes ; le
vide : non colmatees + rouvertes) ; la classe rare, rouverte, nait de leur
combinaison au lieu d'etre apprise seule.

Le lissage (Viterbi) suit la fissure fenetre apres fenetre : un changement
d'etat coute `change_cost`. Un troncon court mais net survit, un va-et-vient
sur des indices faibles disparait. Aucune longueur minimale : le cout est
regle sur les fissures de validation, pour retrouver les changements de
l'annotateur sans en inventer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .bandes import STEP_UM

CATEGORIES = ("open", "filled", "reopened")


# -- Reseau -----------------------------------------------------------------------------

def build_network():
    import torch
    from torch import nn

    def block(c_in, c_out):
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

    class ColmatageNet(nn.Module):
        """Entree (n, 1, travers=120, long=32) ; sortie : logits (gel, vide)."""

        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(block(1, 16), block(16, 32), block(32, 64))
            # On moyenne le long de la fissure, pas en travers : la position par
            # rapport aux parois (gel sur la paroi, ou loin dans la matrice) compte.
            self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64 * 15, 64),
                                      nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(64, 2))

        def forward(self, x):
            x = self.features(x)                  # (n, 64, 15, 4)
            return self.head(x.mean(dim=3))

    return ColmatageNet()


def heads_from_categories(labels: np.ndarray) -> np.ndarray:
    """Categories (indices) -> cibles (gel, vide)."""
    gel = np.isin(labels, (1, 2)).astype(np.float32)
    void = np.isin(labels, (0, 2)).astype(np.float32)
    return np.stack([gel, void], axis=1)


def categories_from_heads(p_gel: np.ndarray, p_void: np.ndarray) -> np.ndarray:
    """Probabilites (open, filled, reopened) : pas de gel -> non colmatee, quoi qu'il en soit du vide."""
    return np.stack([1 - p_gel, p_gel * (1 - p_void), p_gel * p_void], axis=1)


def augment(batch: np.ndarray, rng) -> np.ndarray:
    out = batch.copy()
    flip_across = rng.random(len(out)) < 0.5
    out[flip_across] = out[flip_across, :, ::-1, :]
    flip_along = rng.random(len(out)) < 0.5
    out[flip_along] = out[flip_along, :, :, ::-1]
    out *= rng.uniform(0.92, 1.08, size=(len(out), 1, 1, 1)).astype(np.float32)
    out += rng.normal(0, 0.02, size=out.shape).astype(np.float32)
    return out


def train(train_x, train_y, val_x, val_y, epochs: int = 18, batch_size: int = 256, seed: int = 287298,
          log=print):
    """Apprend le reseau ; garde l'etat qui classe le mieux la longueur de validation."""
    import torch
    from torch.nn import functional

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    network = build_network()
    optimiser = torch.optim.AdamW(network.parameters(), lr=1e-3, weight_decay=1e-3)
    steps = epochs * int(np.ceil(len(train_x) / batch_size))
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=2e-3, total_steps=steps)
    targets = heads_from_categories(train_y)
    best, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        network.train()
        order = rng.permutation(len(train_x))
        losses = []
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            x = torch.from_numpy(augment(train_x[index].astype(np.float32)[:, None], rng))
            y = torch.from_numpy(targets[index])
            loss = functional.binary_cross_entropy_with_logits(network(x), y)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            schedule.step()
            losses.append(loss.item())
        probabilities = predict(network, val_x)
        accuracy = float((probabilities.argmax(axis=1) == val_y).mean())
        sealed = float(((probabilities[:, 0] < 0.5) == (val_y != 0)).mean())
        marker = ""
        if accuracy > best:
            best, best_state = accuracy, {k: v.clone() for k, v in network.state_dict().items()}
            marker = "  <- meilleur"
        log(f"  epoque {epoch:2}/{epochs}  perte {np.mean(losses):.3f}  validation : "
            f"3 classes {accuracy:.1%}, colmatee oui/non {sealed:.1%}{marker}")
    network.load_state_dict(best_state)
    return network


def predict(network, windows: np.ndarray, temperature: tuple[float, float] = (1.0, 1.0),
            batch_size: int = 1024) -> np.ndarray:
    """Probabilites (open, filled, reopened) de chaque fenetre."""
    import torch

    if len(windows) == 0:
        return np.zeros((0, 3))
    network.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            x = torch.from_numpy(windows[start:start + batch_size].astype(np.float32)[:, None])
            logits.append(network(x).numpy())
    logits = np.concatenate(logits) / np.asarray(temperature)[None, :]
    p = 1 / (1 + np.exp(-logits))
    return categories_from_heads(p[:, 0], p[:, 1])


def calibrate(network, windows, labels) -> tuple[float, float]:
    """Temperature de chaque question, pour des probabilites honnetes (lissage)."""
    import torch

    network.eval()
    with torch.no_grad():
        logits = network(torch.from_numpy(windows.astype(np.float32)[:, None])).numpy()
    targets = heads_from_categories(labels)
    result = []
    for k in range(2):
        best_t, best_loss = 1.0, np.inf
        for t in np.exp(np.linspace(np.log(0.5), np.log(4), 40)):
            p = np.clip(1 / (1 + np.exp(-logits[:, k] / t)), 1e-6, 1 - 1e-6)
            loss = -np.mean(targets[:, k] * np.log(p) + (1 - targets[:, k]) * np.log(1 - p))
            if loss < best_loss:
                best_t, best_loss = float(t), loss
        result.append(best_t)
    return tuple(result)


# -- Lissage le long de la fissure ---------------------------------------------------------

def viterbi(probabilities: np.ndarray, change_cost: float, priors=None) -> np.ndarray:
    """Etats les plus probables le long de la fissure ; chaque changement coute `change_cost`."""
    if len(probabilities) == 0:
        return np.zeros(0, dtype=int)
    log_p = np.log(np.clip(probabilities, 1e-6, 1.0))
    if priors is not None:
        log_p = log_p - np.log(np.clip(np.asarray(priors), 1e-6, 1.0))[None, :]
    n, k = log_p.shape
    score = log_p[0].copy()
    back = np.zeros((n, k), dtype=int)
    for i in range(1, n):
        stay = score
        move = score.max() - change_cost
        best_other = int(np.argmax(score))
        choose_move = move > stay
        back[i] = np.where(choose_move, best_other, np.arange(k))
        score = np.where(choose_move, move, stay) + log_p[i]
    states = np.zeros(n, dtype=int)
    states[-1] = int(np.argmax(score))
    for i in range(n - 1, 0, -1):
        states[i - 1] = back[i, states[i]]
    return states


def segments(states: np.ndarray, centers_um: np.ndarray, total_um: float, probabilities=None) -> list[dict]:
    """Troncons homogenes : bornes a mi-chemin entre les centres de deux fenetres voisines."""
    if len(states) == 0:
        return []
    result, start = [], 0
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[start]:
            begin = 0.0 if start == 0 else (centers_um[start - 1] + centers_um[start]) / 2
            end = float(total_um) if i == len(states) else (centers_um[i - 1] + centers_um[i]) / 2
            confidence = float(probabilities[start:i, states[start]].mean()) if probabilities is not None else None
            result.append({"category": CATEGORIES[int(states[start])], "start_um": float(begin),
                           "end_um": float(end), "confidence": confidence})
            start = i
    return result


# -- Modele complet ---------------------------------------------------------------------------

def feature_probabilities(features_model, read, width, height, line, pixel_um, centers_um) -> np.ndarray:
    """Probabilites du modele par descripteurs (colmatage.profils), aux centres donnes.

    Lu par morceaux de 2 mm : la lecture reste bornee meme sur une longue fissure.
    """
    from annotation_app.polyligne import cumulative, cut
    from colmatage.profils import trace_windows

    total = cumulative(line)[-1]
    chunk = 2000.0 / pixel_um
    distances, features = [], []
    for start in np.arange(0.0, total, chunk):
        part = cut(line, start, min(total, start + chunk))
        for w in trace_windows(read, width, height, part, pixel_um):
            distances.append(w.distance_um + start * pixel_um)
            features.append(w.features)
    if not features:
        return np.full((len(centers_um), 3), 1 / 3)
    p = features_model["model"].predict_proba(np.array(features))
    nearest = np.abs(np.asarray(distances)[None, :] - np.asarray(centers_um)[:, None]).argmin(axis=1)
    return p[nearest]


class ColmatageModel:
    """Reseau (+ eventuellement le modele par descripteurs), temperatures, cout de changement.

    `blend` : poids du reseau dans le melange des probabilites (1 = reseau seul,
    0 = descripteurs seuls). Il est choisi sur les lignes de validation.
    """

    VERSION = 1

    def __init__(self, network, temperature=(1.0, 1.0), change_cost: float = 4.0, priors=None,
                 report: dict | None = None, features_model=None, blend: float = 1.0) -> None:
        self.network = network
        self.temperature = tuple(temperature)
        self.change_cost = float(change_cost)
        self.priors = list(priors) if priors is not None else None
        self.report = report or {}
        self.features_model = features_model
        self.blend = float(blend) if features_model is not None else 1.0

    @property
    def pixel_um(self) -> float | None:
        """Pixel des images d'apprentissage, s'il est connu."""
        return self.report.get("pixel_um") or (self.features_model or {}).get("pixel_um")

    def probabilities(self, windows) -> np.ndarray:
        return predict(self.network, windows, self.temperature)

    def classify(self, windows, centers_um, total_um) -> list[dict]:
        """Reseau seul, sur des fenetres deja extraites."""
        p = self.probabilities(windows)
        return segments(viterbi(p, self.change_cost, self.priors), centers_um, total_um, p)

    def classify_line(self, read, width: int, height: int, line, pixel_um: float) -> list[dict]:
        """Une fissure entiere -> troncons homogenes (en um le long du trace)."""
        from .bandes import windows as extract
        from .donnees import polyline_length

        found, centers, _ = extract(read, width, height, line, pixel_um)
        if found is None:
            return []
        p = self.probabilities(found)
        if self.blend < 1.0:
            p_features = feature_probabilities(self.features_model, read, width, height, line, pixel_um, centers)
            p = self.blend * p + (1 - self.blend) * p_features
        total_um = polyline_length(line) * pixel_um
        return segments(viterbi(p, self.change_cost, self.priors), centers, total_um, p)

    def save(self, path: Path) -> None:
        import torch
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save({"version": self.VERSION, "state_dict": self.network.state_dict(),
                    "temperature": self.temperature, "change_cost": self.change_cost,
                    "priors": self.priors, "report": self.report, "step_um": STEP_UM,
                    "features_model": self.features_model, "blend": self.blend}, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> "ColmatageModel | None":
        import torch
        path = Path(path)
        if not path.exists():
            return None
        data = torch.load(path, map_location="cpu", weights_only=False)
        if data.get("version") != cls.VERSION:
            return None
        network = build_network()
        network.load_state_dict(data["state_dict"])
        network.eval()
        return cls(network, data["temperature"], data["change_cost"], data.get("priors"), data.get("report"),
                   data.get("features_model"), data.get("blend", 1.0))


def report_text(report: dict) -> str:
    return json.dumps(report, indent=2, ensure_ascii=False)
