"""Le modele de colmatage : apprendre, mesurer, enregistrer, predire.

Partage par l'application (bouton « Entrainer », suggestions) et par l'analyse
(`analyse/apprentissage.py`, `analyse/classement.py`).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from annotation_app.paths import MODELS_DIR

from .profils import FEATURE_NAMES, FEATURES_VERSION, STEP_UM, trace_windows

CLASSES = ("open", "filled", "reopened")
LABELS = {"open": "non colmatee", "filled": "colmatee", "reopened": "rouverte"}
MODEL_PATH = MODELS_DIR / "colmatage.joblib"
MIN_TRACES = 5                 # par categorie, pour apprendre et valider
# Portion la plus courte qu'on distingue le long d'une fissure. Une fenetre
# fait 20 um et il y en a une tous les 10 um : 10 um, c'est une fenetre seule.
MIN_PORTION_UM = 50.0


def features_for_traces(read, width, height, traces, pixel_um, progress=None, cancelled=None):
    """Descripteurs de toutes les fenetres des traces etiquetes."""
    features, labels, groups = [], [], []
    for index, trace in enumerate(traces, start=1):
        if cancelled and cancelled():
            raise InterruptedError
        for window in trace_windows(read, width, height, trace["geometry"], pixel_um):
            features.append(window.features)
            labels.append(CLASSES.index(trace["category"]))
            groups.append(trace["id"])
        if progress:
            progress(index, len(traces))
    return np.array(features), np.array(labels), np.array(groups)


def make_model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        learning_rate=0.08, max_iter=200, max_leaf_nodes=15, min_samples_leaf=20,
        l2_regularization=1.0, class_weight="balanced", random_state=287298)


def make_rule():
    """Regle de reference a deux seuils, que le classifieur doit battre."""
    from sklearn.tree import DecisionTreeClassifier
    return DecisionTreeClassifier(max_depth=2, class_weight="balanced", random_state=287298)


def impute(values, reference):
    """L'arbre de la regle simple n'accepte pas les valeurs manquantes."""
    filled = values.copy()
    medians = np.nanmedian(reference, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    rows, cols = np.where(~np.isfinite(filled))
    filled[rows, cols] = medians[cols]
    return filled


def cross_validate(X, y, groups, factory, impute_missing: bool = False, folds: int = 5):
    """Probabilites hors echantillon, chaque trace ENTIERE dans un seul pli.

    Melanger les fenetres d'une meme fissure entre apprentissage et test
    surestime le resultat : deux fenetres voisines se ressemblent.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    folds = max(2, min(folds, min(len(set(groups[y == k])) for k in range(len(CLASSES)))))
    probabilities = np.zeros((len(y), len(CLASSES)))
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=287298)
    for train, test in splitter.split(X, y, groups):
        train_x = impute(X[train], X[train]) if impute_missing else X[train]
        test_x = impute(X[test], X[train]) if impute_missing else X[test]
        probabilities[test] = factory().fit(train_x, y[train]).predict_proba(test_x)
    return probabilities


def per_trace(probabilities, y, groups):
    traces = {}
    for trace in np.unique(groups):
        mask = groups == trace
        mean = probabilities[mask].mean(axis=0)
        traces[int(trace)] = {"true": int(y[mask][0]), "pred": int(np.argmax(mean)),
                              "confidence": float(mean.max()), "windows": int(mask.sum())}
    return traces


def metrics(probabilities, y, groups) -> dict:
    traces = per_trace(probabilities, y, groups)
    true = np.array([t["true"] for t in traces.values()])
    pred = np.array([t["pred"] for t in traces.values()])
    length = np.array([t["windows"] * STEP_UM for t in traces.values()])
    confusion = np.zeros((3, 3), dtype=int)
    for a, b in zip(true, pred):
        confusion[a, b] += 1
    recall = [confusion[k, k] / max(1, confusion[k].sum()) for k in range(3)]
    sealed = (true != 0) == (pred != 0)
    return {
        "fenetres_justes": float((probabilities.argmax(axis=1) == y).mean()),
        "traces_justes": float((pred == true).mean()),
        "traces_justes_equilibre": float(np.mean(recall)),
        "longueur_juste": float(length[pred == true].sum() / length.sum()),
        "colmate_oui_non_traces": float(sealed.mean()),
        "colmate_oui_non_longueur": float(length[sealed].sum() / length.sum()),
        "confusion": confusion.tolist(),
        "mal_classees": {t: info for t, info in traces.items() if info["true"] != info["pred"]},
    }


def check_counts(traces) -> dict:
    counts = {c: sum(t["category"] == c for t in traces) for c in CLASSES}
    missing = [LABELS[c] for c, n in counts.items() if n < MIN_TRACES]
    if missing:
        raise ValueError(f"{MIN_TRACES} fissures validées minimum par catégorie ; "
                         f"pas assez en : {', '.join(missing)}.")
    return counts


def train(read, width, height, traces, pixel_um, progress=None, cancelled=None):
    """Mesure par validation croisee, puis apprend sur tout. Rend (artefact, rapport)."""
    counts = check_counts(traces)
    started = time.time()
    X, y, groups = features_for_traces(read, width, height, traces, pixel_um, progress, cancelled)
    rule = metrics(cross_validate(X, y, groups, make_rule, impute_missing=True), y, groups)
    model = metrics(cross_validate(X, y, groups, make_model), y, groups)
    report = {
        "entraine_le": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "traces": counts, "fenetres": int(len(y)), "pixel_um": pixel_um,
        "regle_simple": rule, "classifieur": model,
        "duree_s": round(time.time() - started, 1),
    }
    artifact = {"model": make_model().fit(X, y), "features_version": FEATURES_VERSION,
                "feature_names": FEATURE_NAMES, "classes": CLASSES, "pixel_um": pixel_um,
                "trace_ids": sorted(map(int, set(groups))), "report": report}
    return artifact, report


def save(artifact, path: Path = MODEL_PATH) -> None:
    import joblib
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    joblib.dump(artifact, temporary, compress=3)
    temporary.replace(path)


def load(path: Path = MODEL_PATH):
    """Le modele enregistre, ou None s'il manque ou date d'autres descripteurs."""
    if not Path(path).exists():
        return None
    import joblib
    artifact = joblib.load(path)
    if artifact.get("features_version") != FEATURES_VERSION:
        return None
    return artifact


def portions_from_windows(distances_um, probabilities, total_um: float,
                          min_um: float = MIN_PORTION_UM) -> list[dict]:
    """Decoupe un trace en portions de categorie homogene, fenetre par fenetre.

    Une portion plus courte que `min_um` rejoint celle de ses voisines dont la
    categorie y est la plus probable. Les bornes tombent a mi-chemin entre les
    centres des dernieres fenetres de deux portions.
    """
    probabilities = np.asarray(probabilities)
    labels = probabilities.argmax(axis=1)
    runs = []                                       # [categorie, premiere fenetre, fin exclue]
    for index, label in enumerate(labels):
        if runs and runs[-1][0] == label:
            runs[-1][2] = index + 1
        else:
            runs.append([int(label), index, index + 1])
    while len(runs) > 1:
        short = [i for i, (_, a, b) in enumerate(runs) if (b - a) * STEP_UM < min_um]
        if not short:
            break
        i = min(short, key=lambda k: runs[k][2] - runs[k][1])
        _, a, b = runs[i]
        neighbours = [k for k in (i - 1, i + 1) if 0 <= k < len(runs)]
        target = max(neighbours, key=lambda k: probabilities[a:b, runs[k][0]].mean())
        runs[i][0] = runs[target][0]
        merged = []
        for run in runs:                            # fusionne les voisines devenues identiques
            if merged and merged[-1][0] == run[0]:
                merged[-1][2] = run[2]
            else:
                merged.append(run)
        runs = merged
    bounds = [0.0] + [(distances_um[runs[k][2] - 1] + distances_um[runs[k + 1][1]]) / 2
                      for k in range(len(runs) - 1)] + [float(total_um)]
    portions = []
    for (label, a, b), start, end in zip(runs, bounds, bounds[1:]):
        mean = probabilities[a:b].mean(axis=0)
        portions.append({"category": CLASSES[label], "confidence": float(mean[label]),
                         "probabilities": {c: round(float(p), 3) for c, p in zip(CLASSES, mean)},
                         "start_um": float(start), "end_um": float(end)})
    return portions


def predict_portions(artifact, read, width, height, geometry, pixel_um,
                     min_um: float = MIN_PORTION_UM) -> list[dict]:
    """Portions de categorie homogene d'un trace ; [] s'il est trop court pour etre lu."""
    windows = trace_windows(read, width, height, geometry, pixel_um)
    if not windows:
        return []
    probabilities = artifact["model"].predict_proba(np.array([w.features for w in windows]))
    points = np.asarray(geometry, dtype=float)
    total_um = float(np.hypot(*np.diff(points, axis=0).T).sum()) * pixel_um
    return portions_from_windows([w.distance_um for w in windows], probabilities, total_um, min_um)


def predict(artifact, read, width, height, geometry, pixel_um) -> dict | None:
    """Categorie suggeree pour un trace, avec sa confiance."""
    windows = trace_windows(read, width, height, geometry, pixel_um)
    if not windows:
        return None
    probabilities = artifact["model"].predict_proba(np.array([w.features for w in windows]))
    mean = probabilities.mean(axis=0)
    best = int(np.argmax(mean))
    return {"category": CLASSES[best], "confidence": float(mean[best]),
            "probabilities": {c: round(float(p), 3) for c, p in zip(CLASSES, mean)}}
