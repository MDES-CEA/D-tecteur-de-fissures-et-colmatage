"""Assemble une mosaique MEB a partir de ses tuiles qui se recouvrent.

Les tuiles s'appellent `<prefixe>_<ligne>_<colonne>.tif`, lignes et colonnes
numerotees a partir de 1, et se recouvrent d'un pourcentage fixe (lu dans les
metadonnees Zeiss `CZ_SEM` s'il y est, 9 % sinon).

    python outils\\assembler_mosaique.py "D:\\MEB\\echantillon" --apercu
    python outils\\assembler_mosaique.py "D:\\MEB\\echantillon"
    python outils\\assembler_mosaique.py "D:\\MEB\\echantillon" --reduction 4
    python outils\\assembler_mosaique.py "D:\\MEB\\echantillon" --zone 28000 110000 6000 4000

Pourquoi ne pas simplement poser les tuiles sur la grille theorique : les axes
de la platine ne sont pas exactement ceux du balayage (environ 0,25 degre sur
la premiere acquisition, soit 12 px de derive verticale a chaque colonne), et
la platine ne revient pas au pixel pres. Cumule sur la grille, l'erreur atteint
des centaines de pixels. Trois etapes donc :

1. recalage de chaque paire de voisines sur l'image, dans leur recouvrement.
   Les images sont tres bruitees : on correle apres un filtre passe-bande, qui
   retire le bruit et les variations lentes d'eclairage, et on ne retient que
   les recalages nets. Les zones sans texture donnent des pics ambigus, que
   l'on ecarte plutot que de les croire ;
2. positions globales par moindres carres. Les paires ecartees prennent le pas
   moyen mesure sur les autres, avec un poids faible ; les mesures aberrantes
   sont retirees une a une ;
3. composition par bandes, ecrite directement dans un BigTIFF non compresse a
   une seule bande — le format que l'application lit par memmap. Chaque pixel
   vient de la tuile dont il est le plus pres du centre : pas de fondu, qui
   dedoublerait les fissures fines au moindre ecart, et les niveaux de gris ne
   sont pas modifies.

Les positions sont enregistrees a cote de la mosaique (`_positions.csv`) avec
un rapport (`_rapport.json`) ; un second lancement les reprend sans recaler.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy import sparse
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import spsolve
from skimage.feature import match_template


NAME = re.compile(r"^(?P<prefix>.+)_(?P<row>\d+)_(?P<col>\d+)\.tiff?$", re.IGNORECASE)
DEFAULT_OVERLAP = 9.0          # %
SEARCH_RADIUS = 60             # px : ecart toléré a la position attendue
FINE_SIGMA, COARSE_SIGMA = 1.5, 10.0
MIN_PEAK = 0.12                # correlation minimale d'un recalage retenu
MIN_MARGIN = 0.04              # avance minimale sur le second pic
STRONG_PEAK = 0.5              # au-dela, le pic suffit, meme sur une crete
MODEL_WEIGHT = 0.01            # poids d'une paire ecartee (pas moyen)
PRIOR_WEIGHT = 1e-4            # rappel tres faible vers la grille moyenne
OUTLIER_PX = 3.0               # residu au-dela duquel une mesure est ecartee
TREE_TOLERANCE_PX = 6.0        # premiere selection, contre l'arbre de confiance
BAND_HEIGHT = 1024
OVERVIEW_SIDE = 4000


# --- Inventaire ---------------------------------------------------------------

def scan(folder: Path, overlap: float | None) -> dict:
    groups: dict[str, dict[tuple[int, int], Path]] = {}
    for path in folder.iterdir():
        match = NAME.match(path.name)
        if match:
            key = (int(match["row"]), int(match["col"]))
            groups.setdefault(match["prefix"], {})[key] = path
    if not groups:
        raise SystemExit(f"Aucune tuile <prefixe>_<ligne>_<colonne>.tif dans {folder}")
    prefix = max(groups, key=lambda name: len(groups[name]))
    if len(groups) > 1:
        print(f"Plusieurs series de tuiles : « {prefix} » retenue ({len(groups[prefix])} tuiles).")
    found = groups[prefix]
    row0 = min(r for r, _ in found)
    col0 = min(c for _, c in found)
    files = {(r - row0, c - col0): path for (r, c), path in found.items()}
    rows = max(r for r, _ in files) + 1
    cols = max(c for _, c in files) + 1
    missing = [(r + row0, c + col0) for r in range(rows) for c in range(cols) if (r, c) not in files]
    if missing:
        print(f"ATTENTION : {len(missing)} tuile(s) manquante(s), laissees en noir : {missing[:10]}")

    with tifffile.TiffFile(files[min(files)]) as tif:
        page = tif.pages[0]
        height, width = page.shape[:2]
        meta = page.tags["CZ_SEM"].value if "CZ_SEM" in page.tags else {}
    pixel_nm = None
    if isinstance(meta, dict) and "ap_image_pixel_size" in meta:
        _, value, unit = meta["ap_image_pixel_size"]
        pixel_nm = float(value) * {"nm": 1, "µm": 1e3, "um": 1e3, "mm": 1e6}.get(unit, 1)
    if overlap is None:
        overlap = DEFAULT_OVERLAP
        if isinstance(meta, dict) and "ap_field_overlap" in meta:
            overlap = float(meta["ap_field_overlap"][1])
    return {
        "folder": str(folder), "prefix": prefix, "rows": rows, "cols": cols,
        "row0": row0, "col0": col0, "height": height, "width": width,
        "overlap": overlap, "pixel_nm": pixel_nm,
        "files": {f"{r},{c}": str(path) for (r, c), path in files.items()},
    }


def read_tile(path: str) -> np.ndarray:
    """Niveaux de gris d'une tuile, palette appliquee si elle n'est pas l'identite."""
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        data = page.asarray()
        colormap = page.colormap
    if data.ndim == 3:
        data = data[..., 0]
    if colormap is not None:
        grey = (np.asarray(colormap, dtype=np.uint32).mean(axis=0) // 256).astype(np.uint8)
        if not np.array_equal(grey, np.arange(grey.size, dtype=np.uint8)):
            data = grey[data]
    return np.ascontiguousarray(data, dtype=np.uint8)


# --- Recalage d'une paire -----------------------------------------------------

def _filtered(tile: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    """Passe-bande sur un rectangle de la tuile, avec la marge que le flou exige."""
    pad = int(3 * COARSE_SIGMA)
    h, w = tile.shape
    py0, py1, px0, px1 = max(0, y0 - pad), min(h, y1 + pad), max(0, x0 - pad), min(w, x1 + pad)
    block = tile[py0:py1, px0:px1].astype(np.float32)
    band = gaussian_filter(block, FINE_SIGMA) - gaussian_filter(block, COARSE_SIGMA)
    return band[y0 - py0:y1 - py0, x0 - px0:x1 - px0]


def register_pair(a: np.ndarray, b: np.ndarray, expected: tuple[int, int],
                  radius: int = SEARCH_RADIUS, binning: int = 2):
    """Position de B dans le repere de A, cherchee a +-radius de `expected`.

    Rend (dx, dy, pic, second pic) ; None si le recouvrement est trop etroit.
    Le second pic, pris hors du voisinage du premier, mesure l'ambiguite : un
    recouvrement sans texture donne deux pics de meme hauteur.
    """
    ex, ey = expected
    h, w = a.shape
    x0, x1 = max(0, ex), min(w, ex + w)
    y0, y1 = max(0, ey), min(h, ey + h)
    tx0, tx1, ty0, ty1 = x0 + radius, x1 - radius, y0 + radius, y1 - radius
    if tx1 - tx0 < 16 or ty1 - ty0 < 16:
        return None
    template = _filtered(b, ty0 - ey, ty1 - ey, tx0 - ex, tx1 - ex)
    search = _filtered(a, y0, y1, x0, x1)
    if template.std() == 0 or search.std() == 0:
        return (ex, ey, 0.0, 0.0)
    score = match_template(search[::binning, ::binning], template[::binning, ::binning])
    iy, ix = np.unravel_index(int(np.argmax(score)), score.shape)
    masked = score.copy()
    masked[max(0, iy - 3):iy + 4, max(0, ix - 3):ix + 4] = -1.0
    second = float(masked.max())

    th, tw = template.shape
    q = template - template.mean()
    qn = float(np.sqrt((q * q).sum()))
    best = (-2.0, iy * binning, ix * binning)
    for oy in range(iy * binning - binning, iy * binning + binning + 1):
        for ox in range(ix * binning - binning, ix * binning + binning + 1):
            if 0 <= oy <= search.shape[0] - th and 0 <= ox <= search.shape[1] - tw:
                p = search[oy:oy + th, ox:ox + tw]
                p = p - p.mean()
                denominator = float(np.sqrt((p * p).sum())) * qn
                value = float((p * q).sum() / denominator) if denominator else 0.0
                if value > best[0]:
                    best = (value, oy, ox)
    peak, oy, ox = best
    # Decalage nul (oy = ox = radius) <=> B exactement a la position attendue.
    return (ex + ox - radius, ey + oy - radius, peak, second)


def _register_row(task):
    """Paires d'une ligne : vers la droite, et vers la ligne suivante."""
    row, cols, files, right, down = task
    current = {c: read_tile(files[f"{row},{c}"]) for c in range(cols) if f"{row},{c}" in files}
    below = {c: read_tile(files[f"{row + 1},{c}"]) for c in range(cols) if f"{row + 1},{c}" in files}
    results = []
    for c in range(cols):
        if c in current and c + 1 in current:
            found = register_pair(current[c], current[c + 1], right)
            if found:
                results.append(("R", row, c, *found))
        if c in current and c in below:
            found = register_pair(current[c], below[c], down)
            if found:
                results.append(("D", row, c, *found))
    return results


def accepted(peak: float, second: float) -> bool:
    """Recalage digne de confiance : pic net, ou simplement fort.

    Le long d'une grosse fissure, la correlation forme une crete : le second
    pic, pris juste a cote, est presque aussi haut que le premier sans que la
    mesure soit douteuse. Un pic fort suffit donc ; un pic faible doit en plus
    se detacher nettement.
    """
    return peak >= MIN_PEAK and (peak - second >= MIN_MARGIN or peak >= STRONG_PEAK)


# --- Positions globales -------------------------------------------------------

def _orientation(grid: dict) -> tuple[int, int]:
    """Colonnes vers la droite ou la gauche, lignes vers le bas ou le haut.

    Teste les deux sens sur des paires du centre et garde le meilleur. Sur la
    premiere acquisition : colonne suivante a droite, ligne suivante dessous.
    """
    files, rows, cols = grid["files"], grid["rows"], grid["cols"]
    step_x = round(grid["width"] * (1 - grid["overlap"] / 100))
    step_y = round(grid["height"] * (1 - grid["overlap"] / 100))
    samples = [(r, c) for r in np.linspace(1, rows - 2, 4).astype(int)
               for c in np.linspace(0, cols - 2, 3).astype(int)]
    totals = {"R+": 0.0, "R-": 0.0, "D+": 0.0, "D-": 0.0}
    for r, c in samples:
        a = read_tile(files[f"{r},{c}"]) if f"{r},{c}" in files else None
        if a is None:
            continue
        if f"{r},{c + 1}" in files and cols > 1:
            b = read_tile(files[f"{r},{c + 1}"])
            for sign in (1, -1):
                found = register_pair(a, b, (sign * step_x, 0))
                totals["R+" if sign > 0 else "R-"] += max(0.0, found[2] - found[3]) if found else 0.0
        if f"{r + 1},{c}" in files and rows > 1:
            b = read_tile(files[f"{r + 1},{c}"])
            for sign in (1, -1):
                found = register_pair(a, b, (0, sign * step_y))
                totals["D+" if sign > 0 else "D-"] += max(0.0, found[2] - found[3]) if found else 0.0
    sx = 1 if totals["R+"] >= totals["R-"] else -1
    sy = 1 if totals["D+"] >= totals["D-"] else -1
    return sx * step_x, sy * step_y


LOCAL_ROWS = 7                 # demi-largeur de la fenetre du pas local


def solve_positions(pairs: list, rows: int, cols: int):
    """Positions des tuiles, robustes aux recalages faux.

    1. Arbre de confiance : on relie les tuiles en prenant les recalages par
       pic decroissant, et le pas local seulement pour ce qui resterait isole.
       Un recalage faux a rarement le meilleur pic de son voisinage : il
       n'entre pas dans l'arbre ;
    2. on garde les recalages coherents avec ces positions, puis moindres
       carres sur eux ; trois tours, chacun jugeant sur les positions du
       precedent.

    Retirer « le pire residu d'abord » ne marchait pas : une seule mesure
    fausse repartit son erreur sur ses voisines, et c'etait parfois une bonne
    mesure qui partait — y compris de vrais ecarts de platine de 48 px.

    Une paire sans recalage fiable prend le pas median des paires recalees des
    lignes voisines, et non de toute la mosaique : sur la premiere acquisition,
    le pas a derive de 8 px en x et 6 px en y au fil des neuf heures.
    """
    good = [p for p in pairs if accepted(p["peak"], p["second"])]
    for kind in ("R", "D"):
        if not any(p["kind"] == kind for p in good) and any(p["kind"] == kind for p in pairs):
            raise SystemExit(f"Aucun recalage fiable pour les paires {kind} : verifier le recouvrement.")

    def local_step(kind: str, row: int) -> tuple[float, float]:
        same = [p for p in good if p["kind"] == kind]
        for half in (LOCAL_ROWS, 2 * LOCAL_ROWS, 4 * LOCAL_ROWS, rows):
            chosen = [p for p in same if abs(p["row"] - row) <= half and
                      (kind == "D" or p["row"] % 2 == row % 2)]   # serpentin : sens alterne
            if len(chosen) >= 10 or half == rows:
                break
        chosen = chosen or same
        return (float(np.median([p["dx"] for p in chosen])), float(np.median([p["dy"] for p in chosen])))

    model = {(kind, row): local_step(kind, row) for kind in ("R", "D") for row in range(rows)
             if any(p["kind"] == kind for p in good)}

    n = rows * cols
    trusted = [accepted(p["peak"], p["second"]) for p in pairs]
    active = list(trusted)
    steps_right = [model[("R", r)] for r in range(rows) if ("R", r) in model] or [(0.0, 0.0)]
    steps_down = [model[("D", r)] for r in range(rows) if ("D", r) in model] or [(0.0, 0.0)]
    mean_right = np.mean(steps_right, axis=0)
    mean_down = np.mean(steps_down, axis=0)

    def solve():
        rows_i, rows_j, weights, targets = [], [], [], []
        for p, use in zip(pairs, active):
            i = p["row"] * cols + p["col"]
            j = i + 1 if p["kind"] == "R" else i + cols
            rows_i.append(i)
            rows_j.append(j)
            if use:
                weights.append(1.0)
                targets.append((p["dx"], p["dy"]))
            else:
                weights.append(MODEL_WEIGHT)
                targets.append(model[(p["kind"], p["row"])])
        m = len(pairs)
        index = np.arange(m)
        difference = sparse.csr_matrix(
            (np.r_[-np.ones(m), np.ones(m)], (np.r_[index, index], np.r_[rows_i, rows_j])),
            shape=(m, n))
        prior = sparse.identity(n, format="csr")
        nominal = np.array([[c * mean_right[0] + r * mean_down[0], c * mean_right[1] + r * mean_down[1]]
                            for r in range(rows) for c in range(cols)])
        matrix = sparse.vstack([difference, prior]).tocsr()
        w = sparse.diags(np.r_[weights, np.full(n, PRIOR_WEIGHT)])
        normal = (matrix.T @ w @ matrix).tocsc()
        solution = np.zeros((n, 2))
        for axis in range(2):
            rhs = np.r_[np.array(targets)[:, axis], nominal[:, axis]]
            solution[:, axis] = spsolve(normal, matrix.T @ (w @ rhs))
        return solution

    def edge(p):
        i = p["row"] * cols + p["col"]
        return i, (i + 1 if p["kind"] == "R" else i + cols)

    # 1. Arbre de confiance (Kruskal), puis parcours.
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    links = [[] for _ in range(n)]
    measured = sorted((p for p, ok in zip(pairs, trusted) if ok), key=lambda p: -p["peak"])
    for p, step in [(p, (p["dx"], p["dy"])) for p in measured] + \
                   [(p, model[(p["kind"], p["row"])]) for p in pairs]:
        i, j = edge(p)
        if root(i) != root(j):
            parent[root(i)] = root(j)
            links[i].append((j, step))
            links[j].append((i, (-step[0], -step[1])))
    positions = np.full((n, 2), np.nan)
    for start in range(n):
        if not np.isnan(positions[start, 0]):
            continue
        positions[start] = (0.0, 0.0)
        stack = [start]
        while stack:
            i = stack.pop()
            for j, (dx, dy) in links[i]:
                if np.isnan(positions[j, 0]):
                    positions[j] = positions[i] + (dx, dy)
                    stack.append(j)

    def residual(p, where):
        i, j = edge(p)
        return float(np.max(np.abs(where[j] - where[i] - (p["dx"], p["dy"]))))

    # 2. Mesures coherentes, puis moindres carres ; trois tours.
    for tolerance in (TREE_TOLERANCE_PX, OUTLIER_PX, OUTLIER_PX):
        active = [ok and residual(p, positions) <= tolerance for p, ok in zip(pairs, trusted)]
        positions = solve()
    for p, ok, use in zip(pairs, trusted, active):
        p["used"] = bool(use)
        if ok:
            p["residual"] = residual(p, positions)
    removed = sum(ok and not use for ok, use in zip(trusted, active))
    positions -= positions.min(axis=0)
    return {(r, c): (int(round(positions[r * cols + c][0])), int(round(positions[r * cols + c][1])))
            for r in range(rows) for c in range(cols)}, model, removed


# --- Composition --------------------------------------------------------------

def _reduce(tile: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return tile
    h, w = (tile.shape[0] // factor) * factor, (tile.shape[1] // factor) * factor
    blocks = tile[:h, :w].reshape(h // factor, factor, w // factor, factor)
    return np.round(blocks.mean(axis=(1, 3))).astype(np.uint8)


def compose(grid: dict, positions: dict, reduction: int, output: Path | None,
            overview: Path | None, zone: tuple[int, int, int, int] | None = None) -> tuple[int, int]:
    """Ecrit la mosaique bande par bande ; rend sa taille (largeur, hauteur).

    `zone` (x, y, largeur, hauteur, en pixels pleine resolution) ne compose
    qu'un rectangle : pratique pour verifier un raccord sans tout ecrire.
    """
    th, tw = grid["height"] // reduction, grid["width"] // reduction
    placed = {key: (round(x / reduction), round(y / reduction)) for key, (x, y) in positions.items()
              if f"{key[0]},{key[1]}" in grid["files"]}
    full_width = max(x for x, _ in placed.values()) + tw
    full_height = max(y for _, y in placed.values()) + th
    if zone:
        ox, oy = zone[0] // reduction, zone[1] // reduction
        width = min(zone[2] // reduction, full_width - ox)
        height = min(zone[3] // reduction, full_height - oy)
    else:
        ox, oy, width, height = 0, 0, full_width, full_height

    factor = max(1, int(np.ceil(max(width, height) / OVERVIEW_SIDE)))
    band = max(factor, (BAND_HEIGHT // factor) * factor)
    temporary = None
    target = None
    if output is not None:
        temporary = output.with_name(output.stem + ".partiel.tif")
        extra = {}
        if grid["pixel_nm"]:
            per_cm = 1e7 / (grid["pixel_nm"] * reduction)
            extra = {"resolution": (per_cm, per_cm), "resolutionunit": "CENTIMETER"}
        description = json.dumps({"pixel_nm": grid["pixel_nm"] and grid["pixel_nm"] * reduction,
                                  "source": grid["prefix"], "reduction": reduction})
        target = tifffile.memmap(temporary, shape=(height, width), dtype=np.uint8, bigtiff=True,
                                 photometric="minisblack", description=description, **extra)

    order = sorted(placed, key=lambda key: placed[key][1])
    cache: dict = {}
    small_rows = []
    columns = np.arange(tw)
    distance_x = np.abs(columns + 0.5 - tw / 2) / (tw / 2)
    started = time.time()
    for y0 in range(0, height, band):
        y1 = min(height, y0 + band)
        best = np.full((y1 - y0, width), np.inf, dtype=np.float32)
        data = np.zeros((y1 - y0, width), dtype=np.uint8)
        for key in order:
            x, y = placed[key]
            x -= ox
            y -= oy
            if y >= y1:
                break
            if y + th <= y0 or x + tw <= 0 or x >= width:
                continue
            if key not in cache:
                cache[key] = _reduce(read_tile(grid["files"][f"{key[0]},{key[1]}"]), reduction)
            tile = cache[key]
            ly0, ly1 = max(y0, y) - y, min(y1, y + th) - y
            lx0, lx1 = max(0, -x), min(tw, width - x)
            distance_y = np.abs(np.arange(ly0, ly1) + 0.5 - th / 2) / (th / 2)
            distance = np.maximum(distance_y[:, None], distance_x[None, lx0:lx1])
            region = (slice(y + ly0 - y0, y + ly1 - y0), slice(x + lx0, x + lx1))
            closer = distance < best[region]
            best[region][closer] = distance[closer]
            data[region][closer] = tile[ly0:ly1, lx0:lx1][closer]
        for key in [k for k in cache if placed[k][1] - oy + th <= y1]:
            del cache[key]
        if target is not None:
            target[y0:y1] = data
        rows_small = (y1 - y0) // factor or 1
        cut = data[:rows_small * factor, :(width // factor) * factor]
        if cut.size:
            small_rows.append(cut.reshape(rows_small, factor, -1, factor).mean(axis=(1, 3)).astype(np.uint8))
        done = y1 / height
        elapsed = time.time() - started
        print(f"\r  composition {done:6.1%}  ({elapsed / 60:.1f} min, reste ~{elapsed / done * (1 - done) / 60:.1f} min)",
              end="", flush=True)
    print()
    if target is not None:
        target.flush()
        del target
        temporary.replace(output)
    if overview is not None and small_rows:
        Image.fromarray(np.vstack(small_rows)).save(overview)
    return width, height


# --- Programme ----------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dossier", type=Path, help="dossier des tuiles <prefixe>_<ligne>_<colonne>.tif")
    parser.add_argument("--sortie", type=Path, help="mosaique a ecrire (defaut : a cote du dossier)")
    parser.add_argument("--reduction", type=int, default=1,
                        help="facteur de reduction (moyenne de blocs) ; 1 = pleine resolution")
    parser.add_argument("--apercu", action="store_true",
                        help="seulement une image reduite PNG, pour verifier avant d'ecrire la mosaique")
    parser.add_argument("--zone", type=int, nargs=4, metavar=("X", "Y", "LARGEUR", "HAUTEUR"),
                        help="ne composer qu'un rectangle (pixels pleine resolution), en PNG")
    parser.add_argument("--recaler", action="store_true", help="refaire le recalage meme si les positions existent")
    parser.add_argument("--recouvrement", type=float, help="recouvrement en %% (defaut : metadonnees, sinon 9)")
    parser.add_argument("--processus", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    args = parser.parse_args()

    folder = args.dossier.resolve()
    grid = scan(folder, args.recouvrement)
    base = args.sortie.with_suffix("") if args.sortie else folder.parent / f"{grid['prefix']} ENTIER"
    positions_file = base.with_name(base.name + "_positions.csv")
    report_file = base.with_name(base.name + "_rapport.json")
    print(f"{grid['rows']} lignes x {grid['cols']} colonnes de {grid['width']} x {grid['height']} px, "
          f"recouvrement {grid['overlap']:g} %, pixel {grid['pixel_nm'] or '?'} nm")

    if positions_file.exists() and not args.recaler:
        positions = {}
        with positions_file.open(encoding="utf-8") as stream:
            for line in csv.DictReader(stream):
                positions[(int(line["ligne"]) - grid["row0"], int(line["colonne"]) - grid["col0"])] = (
                    int(line["x"]), int(line["y"]))
        print(f"Positions reprises de {positions_file.name} (--recaler pour les recalculer)")
    else:
        started = time.time()
        pairs = None
        # Le recalage est le long ; s'il est deja dans le rapport, seules les
        # positions sont recalculees (apres une retouche de leur calcul).
        if report_file.exists() and not args.recaler:
            saved = json.loads(report_file.read_text(encoding="utf-8"))
            same = all(saved.get("grille", {}).get(k) == grid[k] for k in ("rows", "cols", "width", "height"))
            if same and saved.get("detail"):
                pairs = [{k: p[k] for k in ("kind", "row", "col", "dx", "dy", "peak", "second")}
                         for p in saved["detail"]]
                print(f"Recalages repris de {report_file.name} ; positions recalculees")
        if pairs is None:
            right_x, down_y = _orientation(grid)
            right, down = (right_x, 0), (0, down_y)
            print(f"Sens : colonne suivante {'a droite' if right_x > 0 else 'a gauche'}, "
                  f"ligne suivante {'dessous' if down_y > 0 else 'dessus'}")
            tasks = [(r, grid["cols"], grid["files"], right, down) for r in range(grid["rows"])]
            pairs = []
            with ProcessPoolExecutor(max_workers=args.processus) as pool:
                for done, results in enumerate(pool.map(_register_row, tasks), start=1):
                    pairs.extend({"kind": k, "row": r, "col": c, "dx": dx, "dy": dy, "peak": pk, "second": sc}
                                 for k, r, c, dx, dy, pk, sc in results)
                    print(f"\r  recalage : ligne {done}/{grid['rows']}  ({(time.time() - started) / 60:.1f} min)",
                          end="", flush=True)
            print()
        positions, model, removed = solve_positions(pairs, grid["rows"], grid["cols"])
        good = sum(p["used"] for p in pairs)
        residuals = [p["residual"] for p in pairs if p["used"]]
        print(f"  {good}/{len(pairs)} paires recalees retenues, {removed} ecartees comme aberrantes ; "
              f"residu median {np.median(residuals):.2f} px, max {max(residuals):.2f} px")
        positions_file.parent.mkdir(parents=True, exist_ok=True)
        with positions_file.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["ligne", "colonne", "fichier", "x", "y"])
            for (r, c), (x, y) in sorted(positions.items()):
                name = Path(grid["files"].get(f"{r},{c}", "")).name
                writer.writerow([r + grid["row0"], c + grid["col0"], name, x, y])
        report_file.write_text(json.dumps({
            "grille": {k: grid[k] for k in ("prefix", "rows", "cols", "width", "height", "overlap", "pixel_nm")},
            "pas_locaux": {f"{k}{r + grid['row0']}": v for (k, r), v in model.items()},
            "paires": len(pairs), "retenues": good, "aberrantes_ecartees": removed,
            "duree_min": round((time.time() - started) / 60, 1),
            "detail": pairs,
        }, indent=1, ensure_ascii=False), encoding="utf-8")

    if args.zone:
        output = base.with_name(f"{base.name}_zone_{'_'.join(map(str, args.zone))}.png")
        sizes = compose(grid, positions, args.reduction, None, output, tuple(args.zone))
        print(f"Zone : {output}  ({sizes[0]} x {sizes[1]} px)")
        return
    if args.apercu:
        output = base.with_name(base.name + "_apercu.png")
        reduction = max(args.reduction, 8)
        sizes = compose(grid, positions, reduction, None, output)
        print(f"Apercu : {output}  (mosaique reduite {reduction} fois : {sizes[0]} x {sizes[1]} px)")
        return

    suffix = f" reduit {args.reduction}" if args.reduction > 1 else ""
    output = args.sortie or base.with_name(base.name + suffix + ".tif")
    overview = output.with_name(output.stem + "_apercu.png")
    print(f"Ecriture de {output}")
    width, height = compose(grid, positions, args.reduction, output, overview)
    size_gb = width * height / 1e9
    print(f"Termine : {width} x {height} px ({size_gb:.1f} Go), apercu {overview.name}")


if __name__ == "__main__":
    sys.exit(main())
