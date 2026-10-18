"""Reseau de segmentation des fissures, alternative au modele par pixel.

Le modele par pixel (`detector_model.py`) decide a partir de dix descripteurs
locaux : flous gaussiens, gradients, variance. Il ne voit aucune forme, donc il
ne peut pas distinguer une structure sombre allongee d'une structure sombre
ronde. C'est la cause mesuree du masque trop gras : 6,4 % de l'image allumee
pour un couloir de fissures qui en represente 1 a 2 %.

Un reseau a convolutions voit un voisinage, donc la forme. Il recoit deux
canaux : le niveau de gris normalise par percentiles, insensible a l'eclairage
absolu du MEB, et le score de cretes de `build_cost_image`, deja calcule par
`prepare_route` et donc gratuit a l'inference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .paths import MODELS_DIR

DEFAULT_UNET_PATH = MODELS_DIR / "crack_detector_unet.pt"

# Trois sous-echantillonnages : les cotes doivent etre des multiples de 8.
DOWNSAMPLING = 8


def prepare_inputs(gray: np.ndarray, ridge: np.ndarray) -> np.ndarray:
    """Construit les deux canaux, a l'identique a l'apprentissage et a l'usage.

    La normalisation par percentiles est faite sur la tuile entiere, pas sur
    l'imagette : une imagette uniforme aurait des statistiques absurdes, et le
    reseau doit voir le meme contraste qu'a l'inference.
    """
    values = gray.astype(np.float32)
    low, high = np.percentile(values, [1.0, 99.0])
    values = np.clip((values - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    return np.stack([values, ridge.astype(np.float32)]).astype(np.float32)


def build_network(in_channels: int = 2, base: int = 16):
    """U-Net compact. Import de torch differe : le reste de l'app s'en passe."""
    import torch
    from torch import nn

    class Block(nn.Module):
        def __init__(self, inputs: int, outputs: int) -> None:
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
                nn.BatchNorm2d(outputs),
                nn.ReLU(inplace=True),
                nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
                nn.BatchNorm2d(outputs),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.body(x)

    class CrackUNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encode1 = Block(in_channels, base)
            self.encode2 = Block(base, base * 2)
            self.encode3 = Block(base * 2, base * 4)
            self.bottom = Block(base * 4, base * 8)
            self.pool = nn.MaxPool2d(2)
            self.raise3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
            self.decode3 = Block(base * 8, base * 4)
            self.raise2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
            self.decode2 = Block(base * 4, base * 2)
            self.raise1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
            self.decode1 = Block(base * 2, base)
            self.head = nn.Conv2d(base, 1, 1)

        def forward(self, x):
            first = self.encode1(x)
            second = self.encode2(self.pool(first))
            third = self.encode3(self.pool(second))
            deep = self.bottom(self.pool(third))
            up = self.decode3(torch.cat([self.raise3(deep), third], dim=1))
            up = self.decode2(torch.cat([self.raise2(up), second], dim=1))
            up = self.decode1(torch.cat([self.raise1(up), first], dim=1))
            return self.head(up)

    return CrackUNet()


def load_unet(path: Path = DEFAULT_UNET_PATH) -> dict | None:
    """Charge un reseau entraine, ou None si le fichier ou torch manquent."""
    if not Path(path).exists():
        return None
    try:
        import torch
    except ImportError:
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    network = build_network(
        in_channels=checkpoint.get("in_channels", 2),
        base=checkpoint.get("base", 16),
    )
    network.load_state_dict(checkpoint["state_dict"])
    network.eval()
    return {
        "kind": "unet",
        "model": network,
        "threshold": float(checkpoint.get("threshold", 0.5)),
        "max_analysis_side": int(checkpoint.get("max_analysis_side", 1000)),
        # Resolution d'analyse apprise (um par pixel) ; absente des reseaux
        # d'origine, qui travaillent a 3,2 um (voir echelle.py).
        "analysis_um": checkpoint.get("analysis_um"),
        "trained_zone_ids": checkpoint.get("trained_zone_ids", []),
        "features_version": checkpoint.get("features_version"),
    }


def unet_score(network, gray: np.ndarray, ridge: np.ndarray) -> np.ndarray:
    """Probabilite de fissure pour chaque pixel de la tuile entiere."""
    import torch

    inputs = prepare_inputs(gray, ridge)
    height, width = inputs.shape[1:]
    pad_rows = (-height) % DOWNSAMPLING
    pad_cols = (-width) % DOWNSAMPLING
    if pad_rows or pad_cols:
        inputs = np.pad(inputs, ((0, 0), (0, pad_rows), (0, pad_cols)), mode="reflect")
    with torch.no_grad():
        logits = network(torch.from_numpy(inputs)[None])
        probability = torch.sigmoid(logits)[0, 0].numpy()
    return probability[:height, :width].astype(np.float32)
