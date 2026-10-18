"""Echelle physique des mosaiques : une image plus fine vue a la resolution
de reference.

Tout le reperage (seuils geometriques, filtre de cretes, reseau) a ete regle et
appris sur la premiere mosaique, dont le pixel mesure 1,595 um. Une acquisition
quatre fois plus fine montre une fissure de 2 px sur 8, et chaque seuil y
change de sens physique.

Plutot que de retoucher `detection_service`, valide sur le banc, on lui presente
l'image a la resolution de reference : `ScaledDetector` reduit chaque zone avant
de la lui confier et ramene les traces en pleine resolution. Les mosaiques a la
resolution de reference ne passent pas par lui : leur comportement, et la cle
de leur cache, restent exactement ceux d'avant.

Le detail fin reste disponible en pleine resolution pour ce qui en a besoin,
par exemple la reconnaissance du gel d'alteration le long des parois.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

# Pixel de la mosaique d'apprentissage, lu dans ses metadonnees Zeiss
# (`ap_image_pixel_size`). Toutes les mosaiques de cette acquisition le portent.
REFERENCE_PIXEL_UM = 1.595
REFERENCE_ZONE_SIZE = 2000      # cote d'une zone d'annotation, pixels de reference
# Resolution d'analyse du detecteur d'origine : zones de 2000 px de reference
# reduites de moitie. Un detecteur reentraine peut en declarer une autre
# (`analysis_um` dans son fichier) ; l'application detecte alors a celle-la.
REFERENCE_ANALYSIS_UM = 2 * REFERENCE_PIXEL_UM
ANALYSIS_CHOICES_UM = (REFERENCE_ANALYSIS_UM, REFERENCE_PIXEL_UM)   # 3,2 et 1,6 um
# En deca de cet ecart, on considere l'image a la resolution de reference.
TOLERANCE = 0.15


def pixel_size_um(path: Path) -> float | None:
    """Taille du pixel d'une image, en micrometres, si le fichier la donne.

    Dans l'ordre : `CRACK_PIXEL_UM` pour forcer une valeur, la description
    ecrite par `assembler_mosaique.py`, les metadonnees Zeiss d'une tuile
    brute, puis la resolution TIFF si elle a une unite.
    """
    forced = os.environ.get("CRACK_PIXEL_UM")
    if forced:
        return float(forced)
    try:
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            tags = page.tags
            description = tags["ImageDescription"].value if "ImageDescription" in tags else ""
            if isinstance(description, str) and description.startswith("{"):
                try:
                    value = json.loads(description).get("pixel_nm")
                    if value:
                        return float(value) / 1000
                except ValueError:
                    pass
            if "CZ_SEM" in tags:
                meta = tags["CZ_SEM"].value
                if isinstance(meta, dict) and "ap_image_pixel_size" in meta:
                    _, value, unit = meta["ap_image_pixel_size"]
                    return float(value) * {"nm": 1e-3, "µm": 1.0, "um": 1.0, "mm": 1e3}.get(unit, 1e-3)
            if "XResolution" in tags and "ResolutionUnit" in tags:
                numerator, denominator = tags["XResolution"].value
                unit = int(tags["ResolutionUnit"].value)
                per_unit = numerator / denominator if denominator else 0
                if per_unit > 1 and unit in (2, 3):          # pouce, centimetre
                    return (25400.0 if unit == 2 else 10000.0) / per_unit
    except (OSError, ValueError, KeyError, tifffile.TiffFileError):
        pass
    return None


def resolution_factor(pixel_um: float | None) -> float:
    """Combien de pixels de l'image dans un pixel de reference (1 si inconnu)."""
    if not pixel_um:
        return 1.0
    factor = REFERENCE_PIXEL_UM / pixel_um
    return 1.0 if abs(factor - 1.0) < TOLERANCE else factor


def zone_size(factor: float) -> int:
    """Cote des zones d'annotation : la meme surface physique qu'a la reference."""
    return int(round(REFERENCE_ZONE_SIZE * factor))


class _ReducedSource:
    """L'image vue a la resolution de reference, lue a la demande."""

    def __init__(self, source, factor: float) -> None:
        self.source = source
        self.factor = factor
        self.width = int(source.width / factor)
        self.height = int(source.height / factor)

    def read(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        f = self.factor
        left, top = int(round(x * f)), int(round(y * f))
        right = min(self.source.width, int(round((x + width) * f)))
        bottom = min(self.source.height, int(round((y + height) * f)))
        block = self.source.read(left, top, right - left, bottom - top)
        return np.asarray(
            Image.fromarray(block).resize((width, height), resample=Image.Resampling.BOX))


class _ReducedTiles:
    def __init__(self, tiles, factor: float) -> None:
        self.source = _ReducedSource(tiles.source, factor)
        self.image_path = tiles.image_path
        self.width = self.source.width
        self.height = self.source.height


class ScaledDetector:
    """Meme interface que `CrackDetector`, pour une image plus fine que la reference."""

    def __init__(self, tiles, factor: float, **options) -> None:
        from .detection_service import CrackDetector

        self.tiles = tiles
        self.factor = factor
        self.inner = CrackDetector(_ReducedTiles(tiles, factor), **options)
        self.max_analysis_side = self.inner.max_analysis_side
        self.model_artifact = self.inner.model_artifact

    def _reference_zone(self, zone: dict) -> dict:
        f = self.factor
        x, y = int(round(zone["x"] / f)), int(round(zone["y"] / f))
        return {**zone, "x": x, "y": y,
                "width": max(1, int(round((zone["x"] + zone["width"]) / f)) - x),
                "height": max(1, int(round((zone["y"] + zone["height"]) / f)) - y)}

    def _to_image(self, zone: dict, points) -> list[list[float]]:
        """Points de reference -> pixels de l'image, bornes a la zone."""
        f = self.factor
        right, bottom = zone["x"] + zone["width"] - 1, zone["y"] + zone["height"] - 1
        return [[round(min(right, max(zone["x"], x * f)), 2), round(min(bottom, max(zone["y"], y * f)), 2)]
                for x, y in points]

    def prepare_route(self, zone: dict):
        prepared = self.inner.prepare_route(self._reference_zone(zone))
        return replace(prepared, scale=prepared.scale * self.factor,
                       diagnostics={**prepared.diagnostics, "resolution_factor": round(self.factor, 3)})

    def analyze(self, zone: dict):
        analysis = self.inner.analyze(self._reference_zone(zone))
        cracks = [{**crack, "anchors": self._to_image(zone, crack["anchors"]),
                   "geometry": self._to_image(zone, crack["geometry"])} for crack in analysis.cracks]
        return replace(analysis, scale=analysis.scale * self.factor, cracks=cracks,
                       diagnostics={**analysis.diagnostics, "resolution_factor": round(self.factor, 3)})

    def analysis_map_png(self, zone: dict, kind: str) -> bytes:
        return self.inner.analysis_map_png(self._reference_zone(zone), kind)

    def route(self, zone: dict, anchors: list[list[float]]) -> list[list[float]]:
        f = self.factor
        reference = [[round(x / f, 2), round(y / f, 2)] for x, y in anchors]
        geometry = self._to_image(zone, self.inner.route(self._reference_zone(zone), reference))
        # Les noeuds poses par l'annotateur doivent rester exactement a leur
        # place : on remplace leur image arrondie par le clic d'origine.
        exact = {tuple(self._to_image(zone, [point])[0]): [round(float(a), 2), round(float(b), 2)]
                 for point, (a, b) in zip(reference, anchors)}
        return [exact.get(tuple(point), point) for point in geometry]


def analysis_side(zone_size: int, factor: float, analysis_um: float) -> int:
    """Cote maximal d'analyse d'une zone pour detecter a `analysis_um` par pixel.

    La detection travaille en pixels de reference (1,595 um), reduits d'un
    facteur entier : 2 donne 3,2 um, 1 donne 1,6 um. Le cote d'analyse en
    decoule, quelle que soit la taille des zones de l'image.
    """
    reference_zone = zone_size / factor
    reduction = max(1, int(round(analysis_um / REFERENCE_PIXEL_UM)))
    # Marge de 2 px : les bords d'une zone, arrondis, peuvent lui donner un
    # pixel de reference de plus, ce qui ferait passer la reduction de 2 a 3.
    return max(64, int(np.ceil((reference_zone + 2) / reduction)))


def make_detector(tiles, zone_size: int | None = None, analysis_um: float | None = None, **options):
    """Le detecteur adapte a l'echelle de l'image, et son facteur.

    `zone_size` : cote des zones de l'image, en pixels de l'image.
    `analysis_um` : resolution d'analyse voulue ; par defaut celle que declare
    le modele, sinon celle du detecteur d'origine (3,2 um) pour une image plus
    fine que la reference. Une mosaique a la resolution de reference, avec un
    modele qui ne declare rien, garde exactement le comportement d'origine.
    """
    from .detection_service import CrackDetector

    factor = resolution_factor(pixel_size_um(tiles.image_path))
    detector = CrackDetector(tiles, **options) if factor == 1.0 else ScaledDetector(tiles, factor, **options)
    declared = (detector.model_artifact or {}).get("analysis_um")
    chosen = analysis_um or declared
    if chosen is None and factor != 1.0:
        chosen = REFERENCE_ANALYSIS_UM
    if chosen and zone_size and "max_analysis_side" not in options:
        side = analysis_side(zone_size, factor, chosen)
        if factor != 1.0:
            detector.inner.max_analysis_side = side
        detector.max_analysis_side = side
    detector.analysis_um = chosen or REFERENCE_ANALYSIS_UM
    return detector, factor
