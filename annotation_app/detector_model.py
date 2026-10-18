"""Modèle supervisé de repérage des fissures, sans classification métier."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude

import crack_length_analysis_multi as base

from .paths import MODELS_DIR


FEATURES_VERSION = 1
DEFAULT_MODEL_PATH = MODELS_DIR / "crack_detector.joblib"


def pixel_features(gray: np.ndarray, ridge_score: np.ndarray | None = None) -> np.ndarray:
    """Calcule des descripteurs locaux insensibles au contraste absolu d'une tuile."""
    normalized = gray.astype(np.float32) / 255.0
    if ridge_score is None:
        ridge_score, _ = base.build_cost_image(normalized)
    blur2 = gaussian_filter(normalized, 2)
    blur6 = gaussian_filter(normalized, 6)
    blur15 = gaussian_filter(normalized, 15)
    blur35 = gaussian_filter(normalized, 35)
    local_variance = np.maximum(
        gaussian_filter(normalized * normalized, 8) - blur6 * blur6,
        0,
    )
    return np.stack(
        [
            normalized,
            ridge_score,
            blur2 - normalized,
            blur6 - normalized,
            blur15 - normalized,
            blur35 - normalized,
            blur15 - blur2,
            gaussian_gradient_magnitude(normalized, 1),
            gaussian_gradient_magnitude(normalized, 3),
            np.sqrt(local_variance),
        ],
        axis=-1,
    ).astype(np.float32)


def default_model_path() -> Path:
    """Le réseau s'il est présent et utilisable, sinon le modèle par pixel.

    L'application démarre donc sans `torch` : elle reprend alors le modèle
    `.joblib`, nettement moins bon mais sans dépendance supplémentaire.
    """
    import importlib.util

    from .unet_model import DEFAULT_UNET_PATH

    if DEFAULT_UNET_PATH.exists() and importlib.util.find_spec("torch") is not None:
        return DEFAULT_UNET_PATH
    return DEFAULT_MODEL_PATH


def load_detector_model(path: Path = DEFAULT_MODEL_PATH) -> dict | None:
    """Charge le modèle de repérage, par pixel (.joblib) ou U-Net (.pt).

    Les deux formats produisent le même contrat : un artefact portant une carte
    de probabilité par pixel et son seuil. `detection_service` distingue les
    deux sur la clé `kind`, le reste de la chaîne est identique.
    """
    path = Path(path)
    if not path.exists():
        return None
    if path.suffix == ".pt":
        from .unet_model import load_unet

        return load_unet(path)
    artifact = joblib.load(path)
    if artifact.get("features_version") != FEATURES_VERSION:
        return None
    return artifact


def train_model(features: np.ndarray, labels: np.ndarray):
    """Entraîne le classifieur avec un seul thread, fiable sous Windows."""
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=160,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=287298,
    ).fit(features, labels)
