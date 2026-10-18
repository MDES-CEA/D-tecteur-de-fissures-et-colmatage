"""Detection conservative de lignes centrales et reroutage des corrections.

Le detecteur travaille sur une copie reduite de la zone. Le filtre de cretes
et l'obscurite locale compensent les contrastes sans renforcer les raccords de
tuiles. Le squelette est transforme en foret (aucun cycle), puis les branches
quasi superposees sont eliminees. Cette convention privilegie explicitement
une seule ligne centrale par fissure plutot que les deux bords d'une fissure
large. Les resultats restent des propositions que l'annotateur peut corriger.

Deux regles gouvernent les seuils de cette chaine, toutes deux etablies par la
mesure sur les tuiles validees (`evaluate_crack_detection.py`) :

1. Toute recherche de composante connexe est en 8-connexite. La 4-connexite,
   valeur par defaut de scipy et de skimage, coupe les structures obliques en
   pixels isoles et supprime donc silencieusement les fissures.
2. Tout seuil geometrique est exprime en pixels pleine resolution puis ramene a
   l'echelle d'analyse. Une constante en pixels d'analyse change de sens
   physique des que l'echelle change.

Le dernier filtre, une longueur minimale sur la ligne finale, est le plus
rentable de la chaine : la precision passe de 48 a 67 % pour 1,4 point de
rappel.
"""

from __future__ import annotations

from collections import OrderedDict, Counter
from dataclasses import dataclass
from io import BytesIO
from math import ceil
from threading import RLock
from time import perf_counter

import numpy as np
from PIL import Image
from scipy.ndimage import (
    binary_dilation,
    convolve,
    find_objects,
    label as connected_components,
)
from skimage.filters import apply_hysteresis_threshold, threshold_otsu
from skimage.graph import route_through_array
from skimage.measure import approximate_polygon
from skimage.morphology import (
    disk,
    remove_small_holes,
    remove_small_objects,
    skeletonize,
)

import crack_length_analysis_multi as base

from .tile_service import PyramidTileService
from .detector_model import (
    DEFAULT_MODEL_PATH,
    default_model_path,
    load_detector_model,
    pixel_features,
)
from .unet_model import unet_score

# `model_path=None` signifie explicitement « aucun modele » : c'est ce que
# demandent les scripts d'apprentissage. Il faut donc un autre marqueur pour
# « choisis toi-meme le meilleur modele disponible ».
AUTOMATIC_MODEL = object()


# Tous les seuils geometriques sont exprimes en PIXELS PLEINE RESOLUTION et
# ramenes a l'echelle d'analyse au moment de l'emploi. Une constante en pixels
# d'analyse change de sens physique des que l'echelle change : c'est ce qui
# faisait passer la detection a l'echelle 2 pour mauvaise alors qu'elle etait
# seulement quatre fois plus permissive.
MIN_OBJECT_AREA = 288.0      # aire minimale d'un objet du masque
MIN_OBJECT_EXTENT = 40.0     # ... sauf s'il est au moins aussi etendu
MIN_SKELETON_PIECE = 80.0    # longueur minimale d'un morceau de squelette
MIN_SPUR_LENGTH = 32.0       # en deca, une branche terminale est un ergot
MIN_NETWORK_LENGTH = 40.0    # longueur cumulee minimale d'un reseau
BRANCH_KEEP_RADIUS = 48.0    # au-dela, une branche n'est pas un doublon

# Longueur minimale d'une proposition finale. Mesure sur les 56 tuiles
# validees : ce seul seuil fait passer la precision de 48 a 67 % en ne coutant
# que 1,4 point de rappel. Les lignes courtes sont presque toutes du bruit et
# ne portent quasiment aucune longueur de fissure. C'est le filtre longiligne
# le plus rentable de toute la chaine ; la rectitude, testee, degrade.
MIN_LINE_LENGTH = 200.0

NEIGHBOURS = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
OFFSETS = tuple(
    (row, col)
    for row in (-1, 0, 1)
    for col in (-1, 0, 1)
    if row or col
)


@dataclass
class ZoneAnalysis:
    scale: int
    score: np.ndarray
    cost: np.ndarray
    mask: np.ndarray
    skeleton: np.ndarray
    cracks: list[dict]
    diagnostics: dict


@dataclass
class ZoneRouteMap:
    """Donnees minimales necessaires au routage manuel entre deux clics."""

    scale: int
    gray: np.ndarray
    score: np.ndarray
    cost: np.ndarray
    diagnostics: dict


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(self, item: tuple[int, int]) -> tuple[int, int]:
        self.parent.setdefault(item, item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first: tuple[int, int], second: tuple[int, int]) -> bool:
        root_first, root_second = self.find(first), self.find(second)
        if root_first == root_second:
            return False
        self.parent[root_second] = root_first
        return True


def _local_normalize(gray: np.ndarray) -> np.ndarray:
    """Convertit en gris flottant sans renforcer les raccords de tuiles.

    Le filtre de cretes et sa composante d'obscurite sont deja locaux. Une
    egalisation locale explicite amplifie au contraire les sauts de contraste
    aux raccords et les transforme en fausses fissures.
    """
    return gray.astype(np.float32) / 255.0


def _remove_small_and_short(
    mask: np.ndarray, min_area: int, min_extent: int
) -> np.ndarray:
    """Garde un objet s'il est assez gros OU assez etendu.

    Le tri par aire seul penalise structurellement les objets fins, qui sont
    justement ceux que l'on cherche : a 2 px de large, une fissure doit courir
    sur 9 px pour atteindre l'aire d'une tache ronde de 5 px de diametre. Le
    critere d'etendue rattrape les fragments de fissure sans reprendre les
    agglomerats compacts, qui restent petits dans les deux directions.

    Les composantes sont recherchees en 8-connexite, sans quoi une fissure
    oblique est vue comme une suite de pixels isoles.
    """
    labels, count = connected_components(mask, structure=np.ones((3, 3)))
    keep = np.zeros(count + 1, dtype=bool)
    for index, slices in enumerate(find_objects(labels), start=1):
        if slices is None:
            continue
        height = slices[0].stop - slices[0].start
        width = slices[1].stop - slices[1].start
        area = int((labels[slices] == index).sum())
        keep[index] = area >= min_area or max(height, width) >= min_extent
    return keep[labels]


def _supervised_thresholds(
    score: np.ndarray, model_threshold: float
) -> tuple[float, float]:
    """Seuils d'hysteresis du mode supervise, avec un filet par tuile.

    Le seuil du modele est absolu. Sur une tuile ou la probabilite reste
    partout en dessous, l'hysteresis ne trouve aucune amorce et rend un masque
    vide alors que la carte de score montre nettement la fissure. Le percentile
    ne sert que de filet : il peut abaisser le seuil, jamais le relever. Une
    tuile deja bien detectee garde donc exactement le comportement du modele,
    et le resultat ne depend pas de l'ordre dans lequel les tuiles sont vues.
    """
    high = float(min(model_threshold, max(0.30, float(np.percentile(score, 99.0)))))
    return high * 0.65, high


def _threshold_mask(
    mask_score: np.ndarray, scale: int, low: float, high: float
) -> np.ndarray:
    """Seuille le score par hysteresis, puis retire ce qui n'est ni gros ni long."""
    min_area = max(6, int(round(MIN_OBJECT_AREA / (scale * scale))))
    min_extent = max(8, int(round(MIN_OBJECT_EXTENT / scale)))
    mask = apply_hysteresis_threshold(mask_score, low, high)
    mask = _remove_small_and_short(mask, min_area, min_extent)
    mask = remove_small_holes(mask, max_size=max(24, min_area * 2) - 1)
    return binary_dilation(mask, structure=disk(1))


def _skeletonize_mask(mask: np.ndarray, scale: int) -> np.ndarray:
    """Reduit le masque a son axe, puis retire les eclats isoles.

    `connectivity=2` est imperatif : `skeletonize()` produit des chaines
    8-connexes, alors que la 4-connexite utilisee par defaut coupe toute
    diagonale en pixels isoles, tous plus petits que le seuil. Avec la valeur
    par defaut, 88 a 96 % du squelette disparaissait sur les tuiles reelles et
    seul survivait ce qui est aligne sur les axes, c'est-a-dire les croix de
    calibration et les raccords de tuiles. Les fissures, obliques ou courbes,
    etaient integralement supprimees.
    """
    skeleton = skeletonize(mask)
    skeleton[:3, :] = False
    skeleton[-3:, :] = False
    skeleton[:, :3] = False
    skeleton[:, -3:] = False
    return remove_small_objects(
        skeleton,
        max_size=max(8, int(round(MIN_SKELETON_PIECE / scale))) - 1,
        connectivity=2,
    )


def _representatives(node_labels: np.ndarray, count: int) -> dict[int, tuple[int, int]]:
    result: dict[int, tuple[int, int]] = {}
    for index, slices in enumerate(find_objects(node_labels), start=1):
        if slices is None:
            continue
        block = node_labels[slices]
        points = np.argwhere(block == index)
        origin = np.array([slices[0].start, slices[1].start])
        points = points + origin
        centre = points.mean(axis=0)
        point = points[np.argmin(np.square(points - centre).sum(axis=1))]
        result[index] = (int(point[0]), int(point[1]))
    return result


def _trace_branches(skeleton: np.ndarray) -> list[dict]:
    """Decompose rapidement un squelette en chemins entre ses noeuds."""
    degree = convolve(skeleton.astype(np.uint8), NEIGHBOURS, mode="constant")
    degree *= skeleton
    node_mask = skeleton & (degree != 2)
    node_labels, node_count = connected_components(
        node_mask, structure=np.ones((3, 3), dtype=np.uint8)
    )
    representatives = _representatives(node_labels, node_count)
    height, width = skeleton.shape
    seen: set[tuple[int, int]] = set()
    branches: list[dict] = []

    def neighbours(point: tuple[int, int]) -> list[tuple[int, int]]:
        row, col = point
        return [
            (row + d_row, col + d_col)
            for d_row, d_col in OFFSETS
            if 0 <= row + d_row < height
            and 0 <= col + d_col < width
            and skeleton[row + d_row, col + d_col]
        ]

    def edge(first: tuple[int, int], second: tuple[int, int]) -> tuple[int, int]:
        first_id = first[0] * width + first[1]
        second_id = second[0] * width + second[1]
        return (first_id, second_id) if first_id < second_id else (second_id, first_id)

    for start_array in np.argwhere(node_mask):
        start = (int(start_array[0]), int(start_array[1]))
        start_label = int(node_labels[start])
        for first in neighbours(start):
            if int(node_labels[first]) == start_label or edge(start, first) in seen:
                continue
            pixels = [representatives[start_label], start, first]
            seen.add(edge(start, first))
            previous, current = start, first
            while node_labels[current] == 0:
                following = [
                    point
                    for point in neighbours(current)
                    if point != previous and edge(current, point) not in seen
                ]
                if not following:
                    break
                next_point = following[0]
                seen.add(edge(current, next_point))
                previous, current = current, next_point
                pixels.append(current)
            end_label = int(node_labels[current])
            end = representatives.get(end_label, current)
            pixels.append(end)
            cleaned = [pixels[0]]
            cleaned.extend(point for point in pixels[1:] if point != cleaned[-1])
            if len(cleaned) >= 2:
                branches.append(
                    {
                        "pixels": np.asarray(cleaned, dtype=np.int32),
                        "start": representatives[start_label],
                        "end": end,
                    }
                )

    # Les composantes sans extremite sont des boucles fermees. Elles sont
    # volontairement omises : une proposition manquante est corrigeable, deux
    # contours automatiques superposes seraient difficiles a fiabiliser.
    return branches


def _branch_quality(branch: dict, score: np.ndarray) -> float:
    pixels = branch["pixels"]
    return len(pixels) * float(score[pixels[:, 0], pixels[:, 1]].mean())


def _prune_spurs(branches: list[dict], min_length: int = 8) -> list[dict]:
    current = branches
    for _ in range(4):
        degrees = Counter(
            point for branch in current for point in (branch["start"], branch["end"])
        )
        kept = [
            branch
            for branch in current
            if len(branch["pixels"]) >= min_length
            or not (
                degrees[branch["start"]] == 1 or degrees[branch["end"]] == 1
            )
        ]
        if len(kept) == len(current):
            break
        current = kept
    return current


def _maximum_score_forest(branches: list[dict], score: np.ndarray) -> list[dict]:
    """Supprime tous les cycles, notamment les echelles des fissures larges."""
    union = _UnionFind()
    result = []
    for branch in sorted(branches, key=lambda item: _branch_quality(item, score), reverse=True):
        if branch["start"] != branch["end"] and union.union(
            branch["start"], branch["end"]
        ):
            result.append(branch)
    return result


def _mark_neighbourhood(mask: np.ndarray, pixels: np.ndarray, radius: int) -> None:
    height, width = mask.shape
    for d_row in range(-radius, radius + 1):
        max_col = int(np.sqrt(max(0, radius * radius - d_row * d_row)))
        for d_col in range(-max_col, max_col + 1):
            rows = pixels[:, 0] + d_row
            cols = pixels[:, 1] + d_col
            valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
            mask[rows[valid], cols[valid]] = True


def _drop_parallel_branches(
    branches: list[dict], score: np.ndarray, radius: int
) -> list[dict]:
    """Elimine une branche si l'essentiel est deja couvert par une meilleure."""
    covered = np.zeros(score.shape, dtype=bool)
    result = []
    for branch in sorted(branches, key=lambda item: _branch_quality(item, score), reverse=True):
        pixels = branch["pixels"]
        overlap = float(covered[pixels[:, 0], pixels[:, 1]].mean())
        if overlap >= 0.55:
            continue
        result.append(branch)
        _mark_neighbourhood(covered, pixels, radius)
    return result


def _looks_like_tile_seam(branch: dict) -> bool:
    """Reconnait un raccord long, presque horizontal ou vertical."""
    pixels = branch["pixels"]
    # Les raccords peuvent onduler de quelques pixels apres le sous-echantillonnage
    # et la skeletonisation. Une egalite exacte des coordonnees laissait donc
    # passer les memes raccords sous forme de fragments.
    if len(pixels) < 80:
        return False
    row_span = int(np.ptp(pixels[:, 0]))
    col_span = int(np.ptp(pixels[:, 1]))
    long_span = max(row_span, col_span)
    short_span = min(row_span, col_span)
    allowed_drift = max(2, min(8, int(round(long_span * 0.04))))
    return long_span >= 80 and short_span <= allowed_drift


def _network_indices(branches: list[dict], shape: tuple[int, int]) -> list[int]:
    mask = np.zeros(shape, dtype=bool)
    for branch in branches:
        pixels = branch["pixels"]
        mask[pixels[:, 0], pixels[:, 1]] = True
    labels, _ = connected_components(
        binary_dilation(mask, structure=np.ones((3, 3)), iterations=2),
        structure=np.ones((3, 3)),
    )
    roots = [int(labels[branch["pixels"][0, 0], branch["pixels"][0, 1]]) for branch in branches]
    renumber = {root: index for index, root in enumerate(dict.fromkeys(roots))}
    return [renumber[root] for root in roots]


def _route_between(
    cost: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    padding: int,
) -> tuple[np.ndarray, float]:
    top = max(0, min(start[0], end[0]) - padding)
    bottom = min(cost.shape[0], max(start[0], end[0]) + padding + 1)
    left = max(0, min(start[1], end[1]) - padding)
    right = min(cost.shape[1], max(start[1], end[1]) + padding + 1)
    path, _ = route_through_array(
        cost[top:bottom, left:right],
        (start[0] - top, start[1] - left),
        (end[0] - top, end[1] - left),
        fully_connected=True,
        geometric=True,
    )
    pixels = np.asarray([(row + top, col + left) for row, col in path], dtype=np.int32)
    mean_cost = float(cost[pixels[:, 0], pixels[:, 1]].mean())
    return pixels, mean_cost


def _stitch_fragments(
    branches: list[dict],
    cost: np.ndarray,
    scale: int,
    max_gap_original: int = 600,
    max_mean_cost: float = 0.075,
) -> tuple[list[dict], int]:
    """Relie les fragments si le vide entre eux suit clairement une fissure.

    Le seuil porte sur tout le chemin de moindre cout, pas seulement sur la
    distance. Cela relie les morceaux separes par une ombre, tout en refusant
    les raccords de tuile et les objets voisins. Chaque groupe devient son
    chemin principal unique : une ramification automatique ne peut donc pas
    recreer deux bords paralleles.
    """
    if len(branches) < 2:
        return branches, 0
    max_gap = max(20, int(round(max_gap_original / scale)))
    candidates = []
    for first_index, first in enumerate(branches):
        first_ends = (first["start"], first["end"])
        for second_index in range(first_index + 1, len(branches)):
            second = branches[second_index]
            second_ends = (second["start"], second["end"])
            choices = [
                (
                    float(np.hypot(a[0] - b[0], a[1] - b[1])),
                    a,
                    b,
                )
                for a in first_ends
                for b in second_ends
            ]
            distance, start, end = min(choices, key=lambda item: item[0])
            if distance <= max_gap:
                candidates.append((distance, first_index, second_index, start, end))

    union = _UnionFind()
    for index in range(len(branches)):
        union.find((index, -1))
    accepted = 0
    # La limite garde un temps borne meme dans une zone tres texturee.
    for _, first_index, second_index, start, end in sorted(candidates)[:500]:
        if union.find((first_index, -1)) == union.find((second_index, -1)):
            continue
        try:
            _, mean_cost = _route_between(cost, start, end, padding=28)
        except (ValueError, IndexError):
            continue
        if mean_cost <= max_mean_cost:
            union.union((first_index, -1), (second_index, -1))
            accepted += 1

    groups: dict[tuple[int, int], list[dict]] = {}
    for index, branch in enumerate(branches):
        groups.setdefault(union.find((index, -1)), []).append(branch)

    stitched = []
    for group in groups.values():
        if len(group) == 1:
            stitched.append(group[0])
            continue
        endpoints = [point for branch in group for point in (branch["start"], branch["end"])]
        start, end = max(
            (
                (first, second)
                for index, first in enumerate(endpoints)
                for second in endpoints[index + 1:]
            ),
            key=lambda pair: float(
                np.hypot(pair[0][0] - pair[1][0], pair[0][1] - pair[1][1])
            ),
        )
        try:
            pixels, _ = _route_between(cost, start, end, padding=65)
        except (ValueError, IndexError):
            pixels = max(group, key=lambda item: len(item["pixels"]))["pixels"]
            start = (int(pixels[0, 0]), int(pixels[0, 1]))
            end = (int(pixels[-1, 0]), int(pixels[-1, 1]))
        stitched.append({"pixels": pixels, "start": start, "end": end})
        # Le chemin principal ne traverse qu'une seule fois un reseau ramifie.
        # Les branches du groupe qui ne le longent pas sont de vraies
        # ramifications, pas les doublons que cette fonction cherche a eviter :
        # les jeter coutait 1,9 point de rappel sans rien rapporter en
        # precision. Celles qui longent le chemin restent ecartees.
        covered = np.zeros(cost.shape, dtype=bool)
        _mark_neighbourhood(
            covered, pixels, max(3, int(round(BRANCH_KEEP_RADIUS / scale)))
        )
        for branch in group:
            aside = branch["pixels"]
            if float(covered[aside[:, 0], aside[:, 1]].mean()) < 0.5:
                stitched.append(branch)
    return stitched, accepted


def _sample_anchors(points: list[list[int]], spacing: float = 110.0) -> list[list[int]]:
    if len(points) <= 2:
        return [points[0], points[-1]]
    anchors = [points[0]]
    distance = 0.0
    previous = points[0]
    for point in points[1:-1]:
        distance += float(np.hypot(point[0] - previous[0], point[1] - previous[1]))
        if distance >= spacing:
            anchors.append(point)
            distance = 0.0
        previous = point
    if points[-1] != anchors[-1]:
        anchors.append(points[-1])
    return anchors


class CrackDetector:
    """Detecteur partage par l'API, avec un petit cache pour les corrections."""

    def __init__(
        self,
        tiles: PyramidTileService,
        max_analysis_side: int = 1000,
        cache_size: int = 3,
        model_path=AUTOMATIC_MODEL,
    ) -> None:
        self.tiles = tiles
        self.max_analysis_side = max_analysis_side
        self.cache_size = cache_size
        self._cache: OrderedDict[int, ZoneAnalysis] = OrderedDict()
        self._route_cache: OrderedDict[int, ZoneRouteMap] = OrderedDict()
        self._cache_lock = RLock()
        self._analysis_lock = RLock()
        if model_path is AUTOMATIC_MODEL:
            model_path = default_model_path()
        self.model_artifact = load_detector_model(model_path) if model_path else None

    def _read_reduced(self, zone: dict) -> tuple[np.ndarray, int]:
        gray = self.tiles.source.read(
            zone["x"], zone["y"], zone["width"], zone["height"]
        )
        scale = max(1, int(ceil(max(gray.shape) / self.max_analysis_side)))
        if scale > 1:
            size = (ceil(gray.shape[1] / scale), ceil(gray.shape[0] / scale))
            gray = np.asarray(
                Image.fromarray(gray).resize(size, resample=Image.Resampling.BOX)
            )
        return gray, scale

    def analyze(self, zone: dict) -> ZoneAnalysis:
        zone_id = int(zone["id"])
        with self._cache_lock:
            cached = self._cache.get(zone_id)
            if cached is not None:
                self._cache.move_to_end(zone_id)
                return cached

        with self._analysis_lock:
            with self._cache_lock:
                cached = self._cache.get(zone_id)
                if cached is not None:
                    self._cache.move_to_end(zone_id)
                    return cached
            analysis = self._analyze_uncached(zone)
            with self._cache_lock:
                self._cache[zone_id] = analysis
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
            return analysis

    def prepare_route(self, zone: dict) -> ZoneRouteMap:
        """Construit seulement la carte de cout utile au geste manuel.

        La segmentation automatique, nettement plus couteuse, reste reservee
        a l'outil experimental et n'est plus sur le chemin critique des deux
        clics de l'annotateur.
        """
        zone_id = int(zone["id"])
        with self._cache_lock:
            cached = self._route_cache.get(zone_id)
            if cached is not None:
                self._route_cache.move_to_end(zone_id)
                return cached

        with self._analysis_lock:
            with self._cache_lock:
                cached = self._route_cache.get(zone_id)
                if cached is not None:
                    self._route_cache.move_to_end(zone_id)
                    return cached
            started = perf_counter()
            gray, scale = self._read_reduced(zone)
            normalized = _local_normalize(gray)
            score, cost = base.build_cost_image(normalized)
            prepared = ZoneRouteMap(
                scale=scale,
                gray=gray,
                score=score.astype(np.float32),
                cost=cost.astype(np.float32),
                diagnostics={
                    "elapsed_seconds": round(perf_counter() - started, 2),
                    "analysis_scale": scale,
                },
            )
            with self._cache_lock:
                self._route_cache[zone_id] = prepared
                while len(self._route_cache) > self.cache_size:
                    self._route_cache.popitem(last=False)
            return prepared

    def _analyze_uncached(self, zone: dict) -> ZoneAnalysis:
        started = perf_counter()
        prepared = self.prepare_route(zone)
        scale = prepared.scale
        if self.model_artifact is not None and self.model_artifact.get("kind") == "unet":
            # Le reseau voit la tuile entiere : pas de reduction supplementaire,
            # et le score de cretes deja calcule par prepare_route lui sert de
            # second canal, donc gratuitement.
            detection_gray = prepared.gray
            score = unet_score(
                self.model_artifact["model"], detection_gray, prepared.score
            )
            detection_mode = "unet"
            dark_columns = detection_gray.mean(axis=0) < 20.4
            dark_rows = detection_gray.mean(axis=1) < 20.4
            score[:, dark_columns] = 0.0
            score[dark_rows, :] = 0.0
            cost = np.exp(-base.PATH_ATTRACTION * score).astype(np.float32)
            cost[:, dark_columns] = 10.0
            cost[dark_rows, :] = 10.0
            low, high = _supervised_thresholds(
                score, float(self.model_artifact["threshold"])
            )
        elif self.model_artifact is not None:
            detection_gray = prepared.gray
            detection_score = prepared.score
            model_side = int(self.model_artifact.get("max_analysis_side", 500))
            extra_scale = max(1, int(ceil(max(detection_gray.shape) / model_side)))
            if extra_scale > 1:
                size = (
                    ceil(detection_gray.shape[1] / extra_scale),
                    ceil(detection_gray.shape[0] / extra_scale),
                )
                detection_gray = np.asarray(
                    Image.fromarray(detection_gray).resize(
                        size, resample=Image.Resampling.BOX
                    )
                )
                # Le filtre de crêtes dépend de son échelle. Il doit être
                # recalculé à la résolution apprise, pas redimensionné depuis
                # la carte utilisée par le routage manuel.
                detection_score = None
                scale *= extra_scale
            values = pixel_features(detection_gray, detection_score)
            score = self.model_artifact["model"].predict_proba(
                values.reshape(-1, values.shape[-1])
            )[:, 1].reshape(detection_gray.shape).astype(np.float32)
            detection_mode = "supervised"
            dark_columns = detection_gray.mean(axis=0) < 20.4
            dark_rows = detection_gray.mean(axis=1) < 20.4
            score[:, dark_columns] = 0.0
            score[dark_rows, :] = 0.0
            cost = np.exp(-base.PATH_ATTRACTION * score).astype(np.float32)
            cost[:, dark_columns] = 10.0
            cost[dark_rows, :] = 10.0
            low, high = _supervised_thresholds(
                score, float(self.model_artifact["threshold"])
            )
        else:
            score = prepared.score
            cost = prepared.cost
            high = float(threshold_otsu(score))
            low = float(min(np.percentile(score, 90.0), high * 0.75))
            detection_mode = "classical"
        mask = _threshold_mask(score, scale, low, high)
        skeleton = _skeletonize_mask(mask, scale)

        raw_branches = _trace_branches(skeleton)
        branches = _prune_spurs(
            raw_branches, min_length=max(4, int(round(MIN_SPUR_LENGTH / scale)))
        )
        after_spurs = len(branches)
        branches = [
            branch
            for branch in branches
            if float(score[branch["pixels"][:, 0], branch["pixels"][:, 1]].mean())
            >= high * (0.75 if detection_mode == "supervised" else 0.88)
        ]
        after_score = len(branches)
        branches = _maximum_score_forest(branches, score)
        after_cycles = len(branches)
        branches = _drop_parallel_branches(
            branches, score, radius=max(4, int(round(12 / scale)))
        )
        after_parallel = len(branches)

        # Filtrage au niveau du reseau : garde les petites ramifications d'une
        # grande fissure, mais retire les petits objets isoles.
        networks = _network_indices(branches, score.shape) if branches else []
        network_sizes: Counter[int] = Counter()
        for branch, network in zip(branches, networks):
            network_sizes[network] += len(branch["pixels"])
        selected = [
            (branch, network)
            for branch, network in zip(branches, networks)
            if network_sizes[network]
            >= max(6, int(round(MIN_NETWORK_LENGTH / scale)))
            and not _looks_like_tile_seam(branch)
        ]
        selected.sort(key=lambda item: _branch_quality(item[0], score), reverse=True)
        selected = selected[:120]
        before_stitch = len(selected)
        stitched, accepted_stitches = _stitch_fragments(
            [branch for branch, _ in selected],
            cost,
            scale,
            max_gap_original=240 if detection_mode == "supervised" else 600,
            max_mean_cost=0.025 if detection_mode == "supervised" else 0.075,
        )
        stitched = [
            branch for branch in stitched if not _looks_like_tile_seam(branch)
        ]
        # Le filtre de longueur porte sur la ligne FINALE, apres raboutage : un
        # fragment court peut appartenir a une longue fissure, et ne devient du
        # bruit avere que si rien ne l'a prolonge.
        before_length = len(stitched)
        minimum_line = max(2, int(round(MIN_LINE_LENGTH / scale)))
        stitched = [
            branch for branch in stitched if len(branch["pixels"]) >= minimum_line
        ]
        selected = list(zip(stitched, _network_indices(stitched, score.shape)))

        cracks = []
        for branch, network in selected:
            # RDP limite fortement le JSON/SVG sans transformer la ligne en
            # deux contours. Le chemin original reste l'axe du squelette et
            # recoit le meme lissage que les chemins traces manuellement.
            smoothing_sigma = max(
                0.5, float(base.PATH_SMOOTHING_SIGMA) / scale
            )
            smoothed = base.smooth_path(branch["pixels"], smoothing_sigma)
            rc = approximate_polygon(smoothed, tolerance=0.75)
            points = [
                [
                    min(zone["x"] + zone["width"] - 1, zone["x"] + int(round(col * scale))),
                    min(zone["y"] + zone["height"] - 1, zone["y"] + int(round(row * scale))),
                ]
                for row, col in rc
            ]
            cleaned = [points[0]]
            cleaned.extend(point for point in points[1:] if point != cleaned[-1])
            if len(cleaned) < 2:
                continue
            cracks.append(
                {
                    "network_index": network,
                    "anchors": _sample_anchors(cleaned),
                    "geometry": cleaned,
                }
            )

        diagnostics = {
            "elapsed_seconds": round(perf_counter() - started, 2),
            "analysis_scale": scale,
            "low_threshold": round(low, 5),
            "high_threshold": round(high, 5),
            "skeleton_pixels": int(skeleton.sum()),
            "raw_branches": len(raw_branches),
            "after_spurs": after_spurs,
            "after_score": after_score,
            "after_cycles": after_cycles,
            "after_parallel": after_parallel,
            "before_stitch": before_stitch,
            "accepted_stitches": accepted_stitches,
            "after_stitch": before_length,
            "min_line_pixels": minimum_line,
            "after_length": len(selected),
            "proposed_lines": len(cracks),
            "single_centerline": True,
            "detection_mode": detection_mode,
            "trained_zones": (
                len(self.model_artifact.get("trained_zone_ids", []))
                if self.model_artifact is not None else 0
            ),
        }
        return ZoneAnalysis(
            scale=scale,
            score=score.astype(np.float32),
            cost=cost.astype(np.float32),
            mask=mask.astype(bool),
            skeleton=skeleton.astype(bool),
            cracks=cracks,
            diagnostics=diagnostics,
        )

    def analysis_map_png(self, zone: dict, kind: str) -> bytes:
        """Rend visibles les étapes exactes utilisées avant les propositions."""
        analysis = self.analyze(zone)
        if kind == "cost":
            # Coût faible = chemin favorable. Il reste donc sombre, comme dans
            # la matrice réellement transmise au pathfinding.
            values = np.clip(analysis.cost, 0.0, 1.0)
        elif kind == "score":
            values = np.clip(analysis.score, 0.0, 1.0)
        elif kind == "mask":
            values = analysis.mask.astype(np.float32)
        elif kind == "skeleton":
            values = analysis.skeleton.astype(np.float32)
        else:
            raise ValueError(kind)
        image = Image.fromarray(np.round(values * 255).astype(np.uint8), mode="L")
        output = BytesIO()
        image.save(output, format="PNG", optimize=True)
        return output.getvalue()

    def route(self, zone: dict, anchors: list[list[float]]) -> list[list[float]]:
        """Reroute et lisse chaque segment entre deux poignees.

        Le pathfinding et le lissage sont ceux du module historique. Chaque
        segment est lisse separement afin que tous les noeuds poses par
        l'utilisateur restent exactement a leur place.
        """
        analysis = self.prepare_route(zone)
        scale = analysis.scale
        reduced = []
        for x, y in anchors:
            col = int(round((x - zone["x"]) / scale))
            row = int(round((y - zone["y"]) / scale))
            reduced.append(
                (
                    max(0, min(row, analysis.cost.shape[0] - 1)),
                    max(0, min(col, analysis.cost.shape[1] - 1)),
                )
            )

        complete: list[list[float]] = []
        smoothing_sigma = max(0.5, float(base.PATH_SMOOTHING_SIGMA) / scale)
        for segment_index, (start, end) in enumerate(zip(reduced, reduced[1:])):
            padding = 55
            top = max(0, min(start[0], end[0]) - padding)
            bottom = min(analysis.cost.shape[0], max(start[0], end[0]) + padding + 1)
            left = max(0, min(start[1], end[1]) - padding)
            right = min(analysis.cost.shape[1], max(start[1], end[1]) + padding + 1)
            local_start = (start[0] - top, start[1] - left)
            local_end = (end[0] - top, end[1] - left)
            if start == end:
                segment = np.asarray([start, end], dtype=float)
            else:
                try:
                    path, _ = route_through_array(
                        analysis.cost[top:bottom, left:right],
                        local_start,
                        local_end,
                        fully_connected=True,
                        geometric=True,
                    )
                    segment = np.asarray(
                        [(row + top, col + left) for row, col in path], dtype=float
                    )
                except (ValueError, IndexError):
                    segment = np.asarray([start, end], dtype=float)

            smoothed = base.smooth_path(segment, smoothing_sigma)
            simplified = approximate_polygon(smoothed, tolerance=0.5)
            points = [
                [
                    round(
                        max(
                            zone["x"],
                            min(zone["x"] + zone["width"] - 1, zone["x"] + col * scale),
                        ),
                        2,
                    ),
                    round(
                        max(
                            zone["y"],
                            min(zone["y"] + zone["height"] - 1, zone["y"] + row * scale),
                        ),
                        2,
                    ),
                ]
                for row, col in simplified
            ]
            # Le lissage historique preserve les extremites du chemin pixel.
            # On restaure en plus les clics sub-pixel exacts de l'interface.
            points[0] = [round(float(value), 2) for value in anchors[segment_index]]
            points[-1] = [round(float(value), 2) for value in anchors[segment_index + 1]]
            if complete and points and complete[-1] == points[0]:
                points = points[1:]
            complete.extend(points)

        if len(complete) < 2:
            return [[round(float(x), 2), round(float(y), 2)] for x, y in anchors]
        return complete
