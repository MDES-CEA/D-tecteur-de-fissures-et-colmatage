"""Pyramide d'image virtuelle pour la grande mosaique MEB.

Le TIFF de la mosaique est non compresse et stocke en une seule bande. ``LargeGrayImage``
l'expose par memmap : chaque tuile demandee par l'affichage est donc lue et
sous-echantillonnee directement, sans charger la mosaique complete.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
from io import BytesIO
from math import ceil, log2
from pathlib import Path

import numpy as np
from PIL import Image

from crop_roi import LargeGrayImage


class TileNotFound(ValueError):
    """La tuile demandee est hors de la pyramide."""


class PyramidTileService:
    def __init__(self, image_path: Path, tile_size: int = 512) -> None:
        self.image_path = image_path.resolve()
        self.source = LargeGrayImage(self.image_path)
        self.width = self.source.width
        self.height = self.source.height
        self.tile_size = tile_size
        self.max_level = int(ceil(log2(max(self.width, self.height))))
        stat = self.image_path.stat()
        identity = f'pyramid-1|{self.image_path}|{stat.st_size}|{stat.st_mtime_ns}|{self.width}|{self.height}|{tile_size}'
        self.cache_key = hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]

    def level_scale(self, level: int) -> int:
        if level < 0 or level > self.max_level:
            raise TileNotFound(f"Niveau hors limites : {level}")
        return 1 << (self.max_level - level)

    def level_dimensions(self, level: int) -> tuple[int, int]:
        scale = self.level_scale(level)
        return ceil(self.width / scale), ceil(self.height / scale)

    @lru_cache(maxsize=512)
    def tile_png(self, level: int, tile_x: int, tile_y: int) -> bytes:
        pixels = self.tile_array(level, tile_x, tile_y)
        stream = BytesIO()
        Image.fromarray(pixels, mode="L").save(
            stream, format="PNG", compress_level=1, optimize=False
        )
        return stream.getvalue()

    def tile_array(self, level: int, tile_x: int, tile_y: int) -> np.ndarray:
        """Niveaux de gris d'une tuile de la pyramide (au plus tile_size de cote)."""
        if tile_x < 0 or tile_y < 0:
            raise TileNotFound("Coordonnees de tuile negatives.")

        scale = self.level_scale(level)
        level_width, level_height = self.level_dimensions(level)
        left = tile_x * self.tile_size
        top = tile_y * self.tile_size
        if left >= level_width or top >= level_height:
            raise TileNotFound("Tuile hors de l'image.")

        output_width = min(self.tile_size, level_width - left)
        output_height = min(self.tile_size, level_height - top)
        source_left = left * scale
        source_top = top * scale
        source_right = min(self.width, (left + output_width) * scale)
        source_bottom = min(self.height, (top + output_height) * scale)

        if self.source.direct:
            # La tranche striee ne materialise que <= tile_size**2 octets.
            pixels = np.asarray(
                self.source._memmap[
                    source_top:source_bottom:scale,
                    source_left:source_right:scale,
                ],
                dtype=np.uint8,
            )
            if self.source._lut is not None:
                pixels = self.source._lut[pixels]
            pixels = np.ascontiguousarray(pixels)
        else:
            crop = self.source.image.crop(
                (source_left, source_top, source_right, source_bottom)
            ).convert("L")
            if crop.size != (output_width, output_height):
                crop = crop.resize(
                    (output_width, output_height), resample=Image.Resampling.BOX
                )
            pixels = np.asarray(crop, dtype=np.uint8)
        return pixels

    def metadata(self) -> dict[str, int | str | bool]:
        return {
            "name": self.image_path.name,
            "width": self.width,
            "height": self.height,
            "tile_size": self.tile_size,
            "min_level": 0,
            "max_level": self.max_level,
            "direct_read": self.source.direct,
            "tile_cache_key": self.cache_key,
        }
