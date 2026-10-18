"""Contour de l'echantillon : ce qui est dehors, ou longe le bord, n'est pas une fissure.

L'echantillon est reconnu par sa FORME, sur un apercu de la mosaique : tout ce
qui n'est pas de la resine (sombre), puis le grand corps arrondi du monolithe.
Une ouverture morphologique de 300 um coupe ce qui ne lui est rattache que par
une attache fine, comme la bande striee du haut de la mosaique. Le niveau de
gris seul ne suffit pas : la mosaique s'eclaircit vers le haut.

Pres du contour, seuls les traits qui le LONGENT sont ecartes (decollements,
interfaces de couches alterees) ; une fissure qui le traverse reste comptee.
Mesure sur les annotations : cela retire les bords rejetes et 0,4 % de la
longueur des fissures validees.
"""

from __future__ import annotations

from math import hypot
from pathlib import Path

import numpy as np

STEP = 25                       # un pixel d'apercu = 25 pixels de la mosaique
OPENING_UM = 300.0
EDGE_BAND_UM = 100.0
ALONG_COSINE = 0.5              # plus parallele que perpendiculaire au contour


def densify(points, spacing: float) -> list[list[float]]:
    """Ajoute des points pour qu'aucun segment ne depasse `spacing` pixels.

    Un trace simplifie a des segments de plusieurs centaines de pixels : juge
    sur son seul milieu, un segment qui sort du contour serait garde entier.
    """
    result = [list(points[0])]
    for a, b in zip(points, points[1:]):
        steps = max(1, int(np.ceil(hypot(b[0] - a[0], b[1] - a[1]) / spacing)))
        for k in range(1, steps + 1):
            t = k / steps
            result.append([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])])
    return result


class SampleMask:
    def __init__(self, mask: np.ndarray, step: int, pixel_um: float) -> None:
        from scipy import ndimage as ndi

        self.mask = mask.astype(bool)
        self.step = step
        self.pixel_um = pixel_um
        self.um_per_cell = step * pixel_um
        self.inside_um = ndi.distance_transform_edt(self.mask) * self.um_per_cell
        self.gy, self.gx = np.gradient(ndi.gaussian_filter(self.inside_um, 2))

    # -- Construction ---------------------------------------------------------------

    @classmethod
    def compute(cls, source, pixel_um: float, step: int = STEP) -> "SampleMask":
        """`source` : l'image (LargeGrayImage), lue par memmap avec un pas de `step`."""
        from scipy import ndimage as ndi
        from skimage.filters import threshold_otsu
        from skimage.morphology import disk

        if getattr(source, "direct", False):
            small = np.asarray(source._memmap[::step, ::step]).astype(np.float32)
        else:
            small = np.asarray(source.read(0, 0, source.width, source.height))[::step, ::step].astype(np.float32)
        smooth = ndi.gaussian_filter(small, 1.5)
        solid = smooth > threshold_otsu(smooth)
        radius = max(1, int(round(OPENING_UM / (step * pixel_um))))
        core = ndi.binary_opening(solid, structure=disk(radius))
        labels, count = ndi.label(core)
        if count == 0:
            return cls(solid, step, pixel_um)
        sizes = ndi.sum(core, labels, range(1, count + 1))
        core = labels == (1 + int(np.argmax(sizes)))
        sample = ndi.binary_dilation(core, structure=disk(radius)) & solid
        return cls(ndi.binary_fill_holes(sample), step, pixel_um)

    @classmethod
    def load_or_compute(cls, image_path: Path, source, pixel_um: float, cache_dir: Path) -> "SampleMask":
        """Le masque, calcule une fois par mosaique puis relu (quelques secondes)."""
        stat = Path(image_path).stat()
        identity = f"{Path(image_path).resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{STEP}|{OPENING_UM}"
        path = Path(cache_dir) / "masque_echantillon.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                if str(data["identity"]) == identity:
                    return cls(np.unpackbits(data["mask"])[:int(np.prod(data["shape"]))]
                               .reshape(data["shape"]).astype(bool), int(data["step"]), float(data["pixel_um"]))
        mask = cls.compute(source, pixel_um)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, mask=np.packbits(mask.mask), shape=np.array(mask.mask.shape),
                            step=mask.step, pixel_um=pixel_um, identity=np.array(identity))
        return mask

    # -- Questions --------------------------------------------------------------------

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        row = min(self.mask.shape[0] - 1, max(0, int(y / self.step)))
        col = min(self.mask.shape[1] - 1, max(0, int(x / self.step)))
        return row, col

    def inside(self, x: float, y: float) -> bool:
        return bool(self.mask[self._cell(x, y)])

    def distance_um(self, x: float, y: float) -> float:
        """Distance au contour, vers l'interieur (0 dehors)."""
        return float(self.inside_um[self._cell(x, y)])

    def segment_kind(self, a, b, band_um: float = EDGE_BAND_UM) -> str:
        """« dehors », « bord » (longe le contour) ou « garde »."""
        x, y = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        cell = self._cell(x, y)
        if not self.mask[cell]:
            return "dehors"
        if self.inside_um[cell] > band_um:
            return "garde"
        nx, ny = self.gx[cell], self.gy[cell]
        dx, dy = b[0] - a[0], b[1] - a[1]
        norm = hypot(nx, ny) * hypot(dx, dy)
        if norm == 0 or abs(nx * dx + ny * dy) / norm < ALONG_COSINE:
            return "bord"
        return "garde"

    def keep_pieces(self, points, band_um: float = EDGE_BAND_UM) -> list[list]:
        """Les morceaux d'une polyligne qui ne sont ni dehors ni le long du bord."""
        points = densify(points, 10.0 / self.pixel_um)
        pieces, current = [], []
        for a, b in zip(points, points[1:]):
            if self.segment_kind(a, b, band_um) == "garde":
                if not current:
                    current = [list(a)]
                current.append(list(b))
            elif current:
                pieces.append(current)
                current = []
        if current:
            pieces.append(current)
        return [p for p in pieces if len(p) >= 2]

    def area_mm2(self) -> float:
        return float(self.mask.sum()) * (self.um_per_cell / 1000) ** 2
