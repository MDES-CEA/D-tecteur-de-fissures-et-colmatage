#!/usr/bin/env python3
"""Extraction d'une région d'intérêt dans une très grande image (mosaïque MEB).

Les mosaïques assemblées (typiquement 14 000 x 52 000 px) ne peuvent pas passer
telles quelles dans crack_length_analysis_multi : la carte de coût et le filtre
de Sato demanderaient plusieurs dizaines de Go de RAM. On découpe donc la zone
qui contient la fissure à mesurer, et on analyse ce morceau.

Le découpage ne change RIEN à la mesure : les longueurs restent exprimées en
pixels de l'image d'origine, à la résolution d'origine. C'est un simple cadrage,
pas un sous-échantillonnage.

Quand le TIFF est non compressé et stocké en une seule bande (le cas des
mosaïques produites par les logiciels d'assemblage MEB), les pixels sont lus par
``numpy.memmap`` : seules les lignes demandées touchent la RAM, l'ouverture est
instantanée quelle que soit la taille du fichier. Sinon on retombe sur Pillow,
avec le garde-fou anti-« decompression bomb » relevé explicitement.

Utilisation ::

    # 1. vue d'ensemble réduite, pour repérer la zone
    python crop_roi.py "mosaique.tif" --overview

    # 2a. découpe par coordonnées (en pixels de l'image complète)
    python crop_roi.py "mosaique.tif" --crop 5200 31000 2500 2000

    # 2b. ou découpe en traçant un rectangle à la souris
    python crop_roi.py "mosaique.tif" --pick

    # 3. analyse du morceau, inchangée
    python crack_length_analysis_multi.py mosaique_x5200_y31000_2500x2000.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

# Le garde-fou de Pillow (~179 Mpx) vise les images piégées reçues d'un tiers.
# Ici l'image vient de notre propre microscope : on le relève en connaissance
# de cause. La vraie limite n'est pas là, elle est dans la RAM (voir plus bas).
Image.MAX_IMAGE_PIXELS = None

# Au-delà, l'analyse complète n'est pas raisonnable sur une machine de bureau :
# gray en float64 = 8 octets/px, et le filtre de Sato en empile plusieurs fois
# plus. 40 Mpx ~ 0,3 Go en float64, déjà confortable.
ANALYSIS_WARN_PIXELS = 40_000_000

OVERVIEW_MAX_SIDE = 2000  # côté le plus long de la vue d'ensemble


class LargeGrayImage:
    """Accès paresseux aux niveaux de gris d'une grande image.

    ``read(x, y, width, height)`` ne charge que le rectangle demandé.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.image = Image.open(path)
        self.width, self.height = self.image.size
        self._memmap: np.memmap | None = None
        self._lut: np.ndarray | None = None
        self._setup_memmap()

    # -- ouverture ---------------------------------------------------

    def _setup_memmap(self) -> None:
        """Prépare la lecture directe si le TIFF s'y prête."""
        if self.image.format != "TIFF":
            return
        tags = self.image.tag_v2

        compression = tags.get(259, 1)
        compression = compression[0] if isinstance(compression, tuple) else compression
        offsets = tags.get(273)
        counts = tags.get(279)
        bits = tags.get(258, (8,))
        bits = bits[0] if isinstance(bits, tuple) else bits
        samples = tags.get(277, 1)
        samples = samples[0] if isinstance(samples, tuple) else samples

        expected = self.width * self.height
        usable = (
            compression == 1            # non compressé
            and bits == 8               # 8 bits par pixel
            and samples == 1            # un seul canal
            and 322 not in tags         # bandes, pas tuiles
            and offsets is not None
            and counts is not None
            and len(offsets) == 1       # une seule bande couvrant tout
            and counts[0] == expected
        )
        if not usable:
            return

        self._memmap = np.memmap(
            self.path,
            dtype=np.uint8,
            mode="r",
            offset=int(offsets[0]),
            shape=(self.height, self.width),
        )

        # Mode palette : les octets sont des index, pas des niveaux de gris.
        # Sur les mosaïques MEB la palette est un dégradé de gris, mais l'index 0
        # sert souvent de « hors champ » et pointe vers le blanc : il faut donc
        # l'appliquer, pas la supposer identité.
        if self.image.mode == "P" and 320 in tags:
            colormap = tags[320]
            size = len(colormap) // 3
            red = np.array(colormap[0:size], dtype=np.uint32)
            green = np.array(colormap[size:2 * size], dtype=np.uint32)
            blue = np.array(colormap[2 * size:3 * size], dtype=np.uint32)
            # Les valeurs TIFF sont sur 16 bits ; certains logiciels écrivent
            # index*256 au lieu de index*257, d'où le décalage plutôt qu'une
            # division par 257.
            grey = ((red + green + blue) // 3 >> 8).astype(np.uint8)
            if not np.array_equal(grey, np.arange(size, dtype=np.uint8)):
                self._lut = grey

    @property
    def direct(self) -> bool:
        return self._memmap is not None

    # -- lecture -----------------------------------------------------

    def read(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        """Rectangle en niveaux de gris (uint8), coordonnées image complète."""
        x = max(0, min(x, self.width - 1))
        y = max(0, min(y, self.height - 1))
        width = max(1, min(width, self.width - x))
        height = max(1, min(height, self.height - y))

        if self._memmap is not None:
            block = np.array(self._memmap[y:y + height, x:x + width])
            return self._lut[block] if self._lut is not None else block

        crop = self.image.crop((x, y, x + width, y + height)).convert("L")
        return np.asarray(crop)

    def overview(self, max_side: int = OVERVIEW_MAX_SIDE) -> tuple[np.ndarray, int]:
        """Vue d'ensemble sous-échantillonnée, et le pas utilisé."""
        step = max(1, int(np.ceil(max(self.width, self.height) / max_side)))
        if self._memmap is not None:
            block = np.array(self._memmap[::step, ::step])
            if self._lut is not None:
                block = self._lut[block]
        else:
            reduced = self.image.reduce(step).convert("L")
            block = np.asarray(reduced)
        return block, step


# ====================================================================
# SORTIES
# ====================================================================


def save_overview(image: LargeGrayImage, output_path: Path, max_side: int) -> None:
    """Vue d'ensemble avec une grille de coordonnées de l'image complète."""
    import matplotlib.pyplot as plt

    block, step = image.overview(max_side)
    height, width = block.shape

    fig, ax = plt.subplots(figsize=(min(16, width / 90 + 4), min(16, height / 90 + 2)))
    ax.imshow(block, cmap="gray", interpolation="nearest")

    # Graduations exprimées en pixels de l'image complète.
    n_ticks = 8
    xs = np.linspace(0, width, n_ticks, dtype=int)
    ys = np.linspace(0, height, n_ticks, dtype=int)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(v * step) for v in xs], fontsize=7, rotation=45)
    ax.set_yticks(ys)
    ax.set_yticklabels([str(v * step) for v in ys], fontsize=7)
    ax.grid(color="#00A0A0", alpha=0.35, linewidth=0.6)
    ax.set_xlabel("x (pixels de l'image complete)", fontsize=8)
    ax.set_ylabel("y (pixels de l'image complete)", fontsize=8)
    ax.set_title(
        f"{image.path.name}\n"
        f"{image.width} x {image.height} px  |  vue reduite d'un facteur {step}",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Vue d'ensemble : {output_path}")
    print(f"  facteur de reduction : {step}")
    print(f"  pour convertir une coordonnee lue ici -> image complete : x {step}")


def save_crop(
    image: LargeGrayImage,
    x: int,
    y: int,
    width: int,
    height: int,
    output_path: Path | None,
) -> Path:
    block = image.read(x, y, width, height)
    actual_h, actual_w = block.shape

    if output_path is None:
        output_path = image.path.with_name(
            f"{image.path.stem}_x{x}_y{y}_{actual_w}x{actual_h}.png"
        )
    Image.fromarray(block).save(output_path)

    pixels = actual_w * actual_h
    print(f"Decoupe        : {output_path}")
    print(f"  origine dans l'image complete : x={x}, y={y}")
    print(f"  taille : {actual_w} x {actual_h} = {pixels:,} px")
    if pixels > ANALYSIS_WARN_PIXELS:
        print(
            f"  ATTENTION : {pixels/1e6:.0f} Mpx, c'est encore beaucoup.\n"
            f"  L'analyse convertit en float64 ({pixels*8/1024**3:.1f} Go) et le\n"
            f"  filtre de Sato en empile plusieurs fois plus. Vise plutot\n"
            f"  moins de {ANALYSIS_WARN_PIXELS/1e6:.0f} Mpx par decoupe."
        )
    return output_path


def pick_interactively(image: LargeGrayImage, max_side: int) -> tuple[int, int, int, int]:
    """Trace un rectangle à la souris sur la vue d'ensemble."""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector

    backend = matplotlib.get_backend().lower()
    if backend in {"agg", "pdf", "ps", "svg", "template"} or "inline" in backend:
        raise SystemExit(
            f"Backend matplotlib non interactif : {matplotlib.get_backend()}\n"
            "Lance depuis un terminal, ou utilise --crop X Y L H."
        )

    block, step = image.overview(max_side)
    selection: dict[str, tuple[int, int, int, int]] = {}

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(block, cmap="gray", interpolation="nearest")
    ax.set_title(
        "Trace un rectangle autour de la fissure, puis ferme la fenetre.\n"
        f"{image.width} x {image.height} px  |  vue reduite d'un facteur {step}"
    )

    def on_select(press, release) -> None:
        x0, x1 = sorted((press.xdata, release.xdata))
        y0, y1 = sorted((press.ydata, release.ydata))
        x = int(round(x0 * step))
        y = int(round(y0 * step))
        width = max(1, int(round((x1 - x0) * step)))
        height = max(1, int(round((y1 - y0) * step)))
        selection["box"] = (x, y, width, height)
        ax.set_title(
            f"Selection : x={x}, y={y}, {width} x {height} px "
            f"({width*height/1e6:.1f} Mpx)\nFerme la fenetre pour valider."
        )
        fig.canvas.draw_idle()

    selector = RectangleSelector(
        ax,
        on_select,
        useblit=True,
        button=[1],
        minspanx=3,
        minspany=3,
        interactive=True,
    )
    plt.show(block=True)
    del selector

    if "box" not in selection:
        raise SystemExit("Aucune zone selectionnee.")
    return selection["box"]


# ====================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extraire une region d'interet d'une tres grande image MEB.",
    )
    parser.add_argument("image", type=Path, help="Image source (mosaique)")
    parser.add_argument(
        "--overview",
        action="store_true",
        help="Ecrire une vue d'ensemble reduite avec grille de coordonnees",
    )
    parser.add_argument(
        "--crop",
        nargs=4,
        type=int,
        metavar=("X", "Y", "LARGEUR", "HAUTEUR"),
        help="Decouper ce rectangle (coordonnees de l'image complete)",
    )
    parser.add_argument(
        "--pick",
        action="store_true",
        help="Choisir la zone a la souris sur la vue d'ensemble",
    )
    parser.add_argument("-o", "--output", type=Path, help="Fichier de sortie")
    parser.add_argument(
        "--max-side",
        type=int,
        default=OVERVIEW_MAX_SIDE,
        help=f"Cote le plus long de la vue d'ensemble (defaut {OVERVIEW_MAX_SIDE})",
    )
    args = parser.parse_args()

    path = args.image.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    image = LargeGrayImage(path)
    pixels = image.width * image.height
    print(f"Image          : {path.name}")
    print(f"  {image.width} x {image.height} = {pixels:,} px, mode {image.image.mode}")
    print(f"  lecture directe par memmap : {'oui' if image.direct else 'non (via Pillow)'}")
    if not image.direct:
        print(
            "  (TIFF compresse ou multi-bandes : chaque lecture decode l'image\n"
            "   entiere, prevois la RAM correspondante)"
        )
    print()

    if not (args.overview or args.crop or args.pick):
        # Rien de demandé : la vue d'ensemble est le point de départ naturel.
        args.overview = True

    if args.overview:
        output = args.output if (args.output and not (args.crop or args.pick)) else None
        save_overview(
            image,
            output or path.with_name(f"{path.stem}_overview.png"),
            args.max_side,
        )

    box = None
    if args.pick:
        box = pick_interactively(image, args.max_side)
    elif args.crop:
        box = tuple(args.crop)

    if box is not None:
        x, y, width, height = box
        crop_path = save_crop(image, x, y, width, height, args.output)
        print()
        print("Analyse de la decoupe :")
        print(f'  python crack_length_analysis_multi.py "{crop_path.name}"')


if __name__ == "__main__":
    main()
