#!/usr/bin/env python3
"""Semi-automatic crack-length analysis for grayscale SEM images (multi-crack).

Version generalisee de crack_length_analysis.py. Au lieu des 4 points fixes
imposant la topologie "extremite - jonction - extremite + branche", l'utilisateur
place un nombre quelconque de points d'ancrage, groupes en un nombre quelconque
de fissures::

    crack_groups = [
        [point_1, point_2, point_3],   # premiere fissure (2 points intermediaires)
        [point_2, point_4],            # branche partant de la jonction point_2
        [point_5, point_6],            # fissure independante
    ]

Chaque paire de points successifs d'un groupe produit un chemin de cout minimal.
Sont donc geres : fissure isolee, fissure courbe guidee, plusieurs fissures
independantes, une ou plusieurs jonctions, et les boucles fermees
([p1, p2, p3, p1]).

La carte de cout, la classification ouvert/colmate et la mesure des aretes
uniques sont conservees a l'identique.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_filter1d, median_filter
from skimage import filters
from skimage.graph import route_through_array
from sklearn.mixture import GaussianMixture


# Paramètre a ajuster pour l'analyse des fissures dans les images SEM
RIDGE_SIGMAS = range(1, 8)
PATH_ATTRACTION = 6.0
LOCAL_GRAY_SIGMA = 4.0
CLASS_SMOOTHING_PIXELS = 15  # Impair

# Distance (px) en dessous de laquelle un nouveau clic est accroché à un point
# ou à un chemin déjà tracé. Garantit que les jonctions sont réellement
# communes au niveau du graphe même si les clics ne sont pas identiques.
SNAP_DISTANCE = 8

# Lissage des coordonnées du chemin AVANT la mesure de longueur (voir
# smooth_path). Corrige deux surestimations : l'escalier 8-connexe (jusqu'à
# +8,2 % sur une droite oblique, maximum théorique atteint à 22°) et le
# vagabondage du chemin dans la largeur de la fissure.
# 0 = mesure brute, pixel par pixel.
#
# Erreur mesurée sur des formes de longueur vraie connue (largeur 6 px) :
#
#   morphologie          sigma=0   sigma=1   sigma=2
#   droite horizontale    +0.28%    +0.09%    +0.05%
#   oblique 22 deg        +8.24%    +1.48%    +0.79%
#   diagonale 45 deg      +0.00%    +0.00%    +0.00%   (poids sqrt(2) exacts)
#   arc R=150             +5.49%    +1.18%    +0.64%
#   arc R=60              +0.59%    -1.34%    -1.63%
#   tortueuse (sinus)     +1.20%    -1.39%    -2.01%
#
# Au-delà de sigma=1 la vraie tortuosité commence à être rabotée : monter
# sigma améliore les fissures droites et pénalise les fissures sinueuses.
PATH_SMOOTHING_SIGMA = 10

# Contrôles de la classification ouvert/colmaté (voir classify_paths).
GMM_MIN_BIC_GAIN = 10.0          # gain de BIC minimal pour accepter 2 populations
GMM_MIN_COMPONENT_WEIGHT = 0.05  # une population trop rare = artefact
GMM_MIN_GRAY_GAP = 0.03          # écart minimal entre les deux moyennes de gris

CLASS_COLORS = {
    "open": "#0072E6",
    "filled": "#E62337",
    "transition": "#7A3DB8",
}


def normalize_percentile(array: np.ndarray) -> np.ndarray:
    """Rescale une gamme dans l'intervalle 0,1"""
    low, high = np.percentile(array, [1.0, 99.5])
    return np.clip((array - low) / (high - low + 1e-12), 0.0, 1.0)


def choose_image() -> Path:
    """Si pas de chemin d'accès dans le code, ouvre une fenetre pour choisir l'image SEM."""
    try:
        from tkinter import Tk, filedialog
    except ImportError as exc:
        raise RuntimeError("Provide the image path on the command line.") from exc

    root = Tk()
    root.withdraw()
    filename = filedialog.askopenfilename(
        title="Select the SEM image",
        filetypes=[
            ("Images", "*.png *.PNG *.tif *.tiff *.jpg *.jpeg"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    if not filename:
        raise SystemExit("No image selected.")
    return Path(filename)


def build_cost_image(gray: np.ndarray, show: bool = False,) -> tuple[np.ndarray, np.ndarray]:
    """Attribut un cout à chaque pixel, plus il est élevé, plus la probabilité de faire partie d'une fissure est grande. """
    denoised = gaussian_filter(gray, sigma=0.8) #Lisse un peu l'image en réduisant le bruit (nécessaire ?)

    ridge = filters.sato( #filtre de sato, c'est un algo qui détecte les crêtes (ridge) dans une image. C'est lui qui détexte les fissures.
        denoised,
        sigmas=RIDGE_SIGMAS,#largeur de fissure, le filtre cherche des largeurs entre 1 et 7 pixels
        black_ridges=True, #Indique que les crêtes sont plus sombres que l'environnement (a ajuster ? automatiquement ?)
    )
    local_darkness = np.maximum( #Detection des zones plsu sombres localement, pour aider à détecter les fissures. Compare chaque pixel à son voisinage (moyenne lcoale)
        gaussian_filter(denoised, sigma=7.0) - denoised,
        0.0,
    )

    score = (
        0.75 * normalize_percentile(ridge)
        + 0.25 * normalize_percentile(local_darkness) #On normalise les deux mesure entre 0 et 1, puis on les combine pour avoir un score final de probabilité d'être une fissure avec un poids spécifique pour chacune
    )
    cost = np.exp(-PATH_ATTRACTION * score)

    # Dégage les pixels entièrement noirs (bordures de l'image) pour éviter que le chemin ne passe par là.
    dark_columns = gray.mean(axis=0) < 0.08
    dark_rows = gray.mean(axis=1) < 0.08
    cost[:, dark_columns] = 10.0
    cost[dark_rows, :] = 10.0
    # ================================================================
    # VISUALISATION
    # ================================================================
    if show:
        fig, axes = plt.subplots(
            1,
            4,
            figsize=(18, 5),
            constrained_layout=True,
        )

        axes[0].imshow(gray, cmap="gray")
        axes[0].set_title("Image originale")

        im1 = axes[1].imshow(ridge, cmap="magma")
        axes[1].set_title("Réponse du filtre de Sato")
        fig.colorbar(im1, ax=axes[1], shrink=0.8)

        im2 = axes[2].imshow(score, cmap="magma")
        axes[2].set_title("Score de fissure")
        fig.colorbar(im2, ax=axes[2], shrink=0.8)

        # L'échelle logarithmique permet de mieux voir les faibles coûts.
        im3 = axes[3].imshow(
            np.log10(cost + 1e-12),
            cmap="viridis",
        )
        axes[3].set_title("log10 de la carte de coût")
        fig.colorbar(im3, ax=axes[3], shrink=0.8)

        for axis in axes:
            axis.axis("off")

        plt.show()
    return score, cost

# ====================================================================
# SELECTION INTERACTIVE MULTI-FISSURES
# ====================================================================


class CrackSelector:
    """Sélection de N fissures définies chacune par N points d'ancrage.

    Commandes :
      * clic gauche   : ajouter un point (accroché aux points/chemins voisins) ;
      * clic droit    : terminer la fissure courante (idem Entrée) ;
      * Entrée        : terminer la fissure courante ;
      * Retour arrière: supprimer le dernier point ;
      * Échap         : terminer toute la sélection.

    Le chemin de coût minimal est tracé en direct après chaque clic : on voit
    donc immédiatement si un point intermédiaire supplémentaire est nécessaire
    pour empêcher le chemin de suivre une interface voisine.
    """

    def __init__(
        self,
        rgb: np.ndarray,
        cost: np.ndarray,
        snap_distance: int = SNAP_DISTANCE,
    ) -> None:
        self.cost = cost
        self.height, self.width = cost.shape
        self.snap_distance = snap_distance

        self.groups: list[list[tuple[int, int]]] = []  # fissures terminées
        self.current: list[tuple[int, int]] = []       # fissure en cours
        self.paths: list[list[np.ndarray]] = []        # chemins des fissures terminées
        self.current_paths: list[np.ndarray] = []      # chemins de la fissure en cours

        self.fig, self.ax = plt.subplots(figsize=(11, 8))
        self.ax.imshow(rgb)
        self.ax.axis("off")
        self._artists: list = []

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    # -- utilitaires -------------------------------------------------

    def _all_groups(self) -> list[list[tuple[int, int]]]:
        groups = list(self.groups)
        if self.current:
            groups.append(self.current)
        return groups

    def _all_paths(self) -> list[np.ndarray]:
        paths = [path for group in self.paths for path in group]
        paths.extend(self.current_paths)
        return paths

    def _snap(self, row: int, col: int) -> tuple[int, int]:
        """Accroche un clic à un point d'ancrage ou à un pixel de chemin proche."""
        target = np.array([row, col], dtype=float)

        anchors = [point for group in self._all_groups() for point in group]
        if anchors:
            candidates = np.asarray(anchors, dtype=float)
            distances = np.linalg.norm(candidates - target, axis=1)
            index = int(np.argmin(distances))
            if distances[index] <= self.snap_distance:
                # Un point d'ancrage existant est prioritaire : c'est celui que
                # l'utilisateur a explicitement placé (jonction voulue).
                return anchors[index]

        paths = self._all_paths()
        if paths:
            pixels = np.concatenate(paths, axis=0).astype(float)
            distances = np.linalg.norm(pixels - target, axis=1)
            index = int(np.argmin(distances))
            if distances[index] <= self.snap_distance:
                pixel = pixels[index]
                return (int(pixel[0]), int(pixel[1]))

        return (row, col)

    def _route(self, start: tuple[int, int], end: tuple[int, int]) -> np.ndarray | None:
        if start == end:
            return None
        path, _ = route_through_array(
            self.cost,
            start,
            end,
            fully_connected=True,
            geometric=True,
        )
        return np.asarray(path, dtype=int)

    # -- évènements --------------------------------------------------

    def _on_click(self, event) -> None:
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", ""):
            return  # zoom/pan actif : on ne compte pas le clic comme un point
        if event.inaxes is not self.ax or event.xdata is None:
            return

        if event.button == 3:  # clic droit = fin de la fissure courante
            self._finish_current()
            self._redraw()
            return
        if event.button != 1:
            return

        row = int(np.clip(round(event.ydata), 0, self.height - 1))
        col = int(np.clip(round(event.xdata), 0, self.width - 1))
        point = self._snap(row, col)

        if self.current and point == self.current[-1]:
            return  # doublon, rien à router

        if self.current:
            path = self._route(self.current[-1], point)
            if path is not None:
                self.current_paths.append(path)
        self.current.append(point)
        self._redraw()

    def _on_key(self, event) -> None:
        if event.key == "enter":
            self._finish_current()
            self._redraw()
        elif event.key == "backspace":
            self._undo()
            self._redraw()
        elif event.key == "escape":
            self._finish_current()
            plt.close(self.fig)

    def _finish_current(self) -> None:
        if len(self.current) >= 2:
            self.groups.append(self.current)
            self.paths.append(self.current_paths)
        elif self.current:
            print("Fissure ignorée : au moins 2 points sont nécessaires.")
        self.current = []
        self.current_paths = []

    def _undo(self) -> None:
        if not self.current and self.groups:
            # Rien à défaire dans la fissure en cours : on rouvre la précédente.
            self.current = self.groups.pop()
            self.current_paths = self.paths.pop()
        if not self.current:
            return
        self.current.pop()
        if self.current_paths:
            self.current_paths.pop()

    # -- affichage ---------------------------------------------------

    def _redraw(self) -> None:
        for artist in self._artists:
            artist.remove()
        self._artists = []

        groups = self._all_groups()
        path_sets = list(self.paths)
        if self.current:
            path_sets.append(self.current_paths)

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for index, (group, paths) in enumerate(zip(groups, path_sets)):
            color = colors[index % len(colors)]
            for path in paths:
                line, = self.ax.plot(
                    path[:, 1],
                    path[:, 0],
                    color=color,
                    linewidth=1.8,
                )
                self._artists.append(line)
            scatter = self.ax.scatter(
                [point[1] for point in group],
                [point[0] for point in group],
                s=30,
                c="yellow",
                edgecolors="black",
                linewidths=0.5,
                zorder=5,
            )
            self._artists.append(scatter)
            for number, (row, col) in enumerate(group, start=1):
                text = self.ax.text(
                    col + 5,
                    row,
                    f"{index + 1}.{number}",
                    fontsize=7,
                    color="black",
                    bbox={"facecolor": "white", "alpha": 0.7, "pad": 1},
                    zorder=6,
                )
                self._artists.append(text)

        self.ax.set_title(
            "Clic gauche = point | Entrée ou clic droit = fissure suivante | "
            "Retour arrière = annuler | Échap = terminer\n"
            f"Fissures terminées : {len(self.groups)} | "
            f"points dans la fissure en cours : {len(self.current)}"
        )
        self.fig.canvas.draw_idle()

    def run(self) -> list[list[tuple[int, int]]]:
        self._redraw()
        plt.show(block=True)
        self._finish_current()  # au cas où la fenêtre est fermée à la croix
        return self.groups


NON_INTERACTIVE_BACKENDS = {"agg", "pdf", "ps", "svg", "cairo", "template"}


def check_interactive_backend() -> None:
    """Vérifie qu'une fenêtre cliquable est possible.

    Sur un backend non interactif (Agg, ou "inline" dans un notebook / la
    fenêtre interactive de VS Code), plt.show() rend la main immédiatement :
    la sélection reviendrait vide sans aucune explication.
    """
    backend = matplotlib.get_backend()
    name = backend.lower()
    if name in NON_INTERACTIVE_BACKENDS or "inline" in name or "nbagg" in name:
        raise SystemExit(
            f"Backend matplotlib non interactif : {backend}\n"
            "La sélection par clics a besoin d'une vraie fenêtre.\n"
            "  - lance le script depuis un terminal :\n"
            "      python crack_length_analysis_multi.py mon_image.png\n"
            "    (et non dans une fenêtre interactive / notebook)\n"
            "  - ou force un backend GUI avant de lancer :\n"
            "      set MPLBACKEND=TkAgg\n"
            "  - ou passe les points sans interface : --groups / --annotations"
        )


def select_crack_groups(
    rgb: np.ndarray,
    cost: np.ndarray,
) -> list[list[tuple[int, int]]]:
    """Sélection interactive d'un nombre quelconque de fissures."""
    check_interactive_backend()
    groups = CrackSelector(rgb, cost).run()
    if not groups:
        raise SystemExit(
            "Aucune fissure sélectionnée : la fenêtre a été fermée sans valider.\n"
            "Rappel : clic gauche = point, Entrée = fissure suivante, "
            "Échap = terminer (une fissure a besoin d'au moins 2 points)."
        )
    return groups


# ====================================================================
# EXTRACTION DES CHEMINS
# ====================================================================


def extract_paths(
    cost: np.ndarray,
    crack_groups: list[list[tuple[int, int]]],
) -> list[np.ndarray]:
    """Extract an arbitrary number of guided crack paths.

    Chaque paire de points successifs d'un groupe produit un chemin de coût
    minimal. Aucune jonction ni aucun nombre de branches n'est imposé.
    """
    paths = []
    for group in crack_groups:
        if len(group) < 2:
            continue
        for start, end in zip(group[:-1], group[1:]):
            if tuple(start) == tuple(end):
                continue
            path, _ = route_through_array(#cherche le chemin avec un cout cumulé minimal entre les points
                cost,
                start,
                end,
                fully_connected=True, #8 degrès de libertés
                geometric=True, #un chemin plus direct est favorisé, même si le cout est un peu plus élevé. Cout horizontal/vertical = 1 et cout diagonal = sqrt(2) (distance euclidienne)
            )
            paths.append(np.asarray(path, dtype=int))#liste x, y des pixels du chemin
    return paths


def path_owners(crack_groups: list[list[tuple[int, int]]]) -> list[int]:
    """Index de la fissure d'origine pour chaque chemin produit par extract_paths.

    Doit suivre exactement la même logique de filtrage que extract_paths.
    """
    owners = []
    for index, group in enumerate(crack_groups):
        if len(group) < 2:
            continue
        for start, end in zip(group[:-1], group[1:]):
            if tuple(start) == tuple(end):
                continue
            owners.append(index)
    return owners


# ====================================================================
# CLASSIFICATION OUVERT / COLMATE
# ====================================================================


def classify_paths( #classe chaque pixel du chemin en fissure ouverte ou remplie, en fonction de la valeur de gris locale
    gray: np.ndarray,
    paths: list[np.ndarray],
) -> tuple[list[np.ndarray], np.ndarray, float, float, bool]:
    """Classify path pixels into open (False) and filled (True).

    Deux contrôles évitent de forcer une classe colmatée sur une fissure
    entièrement ouverte :
      1. le modèle à deux populations doit battre le modèle à une population
         au sens du BIC ;
      2. les deux populations doivent être suffisamment peuplées et séparées.
    Si l'un des contrôles échoue, tout est classé "ouvert".
    """
    local_gray = gaussian_filter(gray, sigma=LOCAL_GRAY_SIGMA) #niveau de gris local, voisinnage de 4 pixel (sigma = 4.0) correspond à un lissage au final
    # Le modèle est ajusté sur les pixels UNIQUES : avec des groupes arbitraires
    # une portion partagée (jonction, branche retracée) serait sinon comptée
    # plusieurs fois et biaiserait les poids du mélange.
    unique_pixels = np.unique(np.concatenate(paths, axis=0), axis=0)
    fit_values = local_gray[unique_pixels[:, 0], unique_pixels[:, 1]].reshape(-1, 1)#niveau de gris sous les pixels des chemins

    single = GaussianMixture(n_components=1, random_state=42).fit(fit_values)
    mixture = GaussianMixture(n_components=2, random_state=42).fit(fit_values)
    means = mixture.means_.ravel()
    weights = mixture.weights_.ravel()

    two_populations = bool(
        len(fit_values) >= 2 * CLASS_SMOOTHING_PIXELS
        and mixture.bic(fit_values) < single.bic(fit_values) - GMM_MIN_BIC_GAIN
        and weights.min() >= GMM_MIN_COMPONENT_WEIGHT
        and abs(means[1] - means[0]) >= GMM_MIN_GRAY_GAP
    )

    if not two_populations:
        # Une seule population de gris : la fissure est considérée ouverte sur
        # toute sa longueur plutôt que coupée arbitrairement en deux classes.
        classifications = [np.zeros(len(path), dtype=bool) for path in paths]
        open_mean = float(single.means_.ravel()[0])
        return classifications, local_gray, open_mean, float("nan"), False

    # Fissures colmatées = plus sombres que les fissures ouvertes, donc la classe avec la moyenne la plus faible est considérée comme remplie.
    filled_class = int(np.argmax(means))
    filled_mean = float(means[filled_class])
    open_mean = float(means[1 - filled_class])

    classifications = []
    for path in paths:
        values = local_gray[path[:, 0], path[:, 1]]
        labels = mixture.predict(values.reshape(-1, 1))
        filled = labels == filled_class
        filled = median_filter(
            filled.astype(np.uint8),
            size=CLASS_SMOOTHING_PIXELS,
            mode="nearest",
        ).astype(bool)
        classifications.append(filled)

    return classifications, local_gray, open_mean, filled_mean, True


# ====================================================================
# MESURE DES LONGUEURS
# ====================================================================


def smooth_path(
    path: np.ndarray,
    sigma: float = PATH_SMOOTHING_SIGMA,
) -> np.ndarray:
    """Lisse les coordonnées du chemin le long de son index.

    Le chemin brut est une chaîne de pixels 8-connexes : il monte en escalier
    sur les portions obliques et zigzague dans la largeur de la fissure. Les
    deux effets ne font qu'ajouter de la longueur. On lisse donc les
    coordonnées avant de mesurer.

    Les deux extrémités sont remises à leur position exacte : ce sont les
    points d'ancrage de l'utilisateur (et les jonctions partagées entre
    fissures), ils ne doivent pas bouger.

    Attention : sur une fissure large et fortement courbée, le chemin de coût
    minimal longe le bord intérieur de la courbe plutôt que l'axe. Le lissage
    ne crée pas ce raccourci mais il le démasque, l'escalier le compensait
    partiellement. C'est un biais de routage, pas un biais de mesure.
    """
    if sigma <= 0 or len(path) < 3:
        return path.astype(float)

    rows = gaussian_filter1d(path[:, 0].astype(float), sigma, mode="nearest")
    cols = gaussian_filter1d(path[:, 1].astype(float), sigma, mode="nearest")
    smoothed = np.column_stack([rows, cols])
    smoothed[0] = path[0]
    smoothed[-1] = path[-1]
    return smoothed


def path_length(points: np.ndarray) -> float:
    """Longueur d'une polyligne."""
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


SMOOTHING_CHECK_SIGMAS = (0.5, 1.0, 2.0)


def show_smoothing_diagnostic(
    rgb: np.ndarray,
    paths: list[np.ndarray],
    score: np.ndarray | None = None,
    output_path: Path | None = None,
    sigmas: tuple[float, ...] = SMOOTHING_CHECK_SIGMAS,
    n_zooms: int = 3,
    window: int = 25,
    half_size: int = 22,
) -> None:
    """Compare visuellement le chemin brut et le chemin lissé.

    Ligne 1 : vues d'ensemble, image et carte de score de fissure.
    Ligne 2 : zooms au pixel sur l'image, là où la correction est la plus forte.
    Ligne 3 : les mêmes zooms sur la carte de score.

    La carte de score est celle qui guide le routage : elle montre si le chemin
    vagabonde parce que la crête est large ou floue, ou s'il fait simplement de
    l'escalier sur une crête nette.
    """
    reference = sigmas[len(sigmas) // 2]
    colors = ["#00A0A0", "#0072E6", "#B84DD8"]

    # Repère les fenêtres où le brut est le plus long que le lissé.
    candidates: list[tuple[float, int, int]] = []  # (exces, index chemin, index pixel)
    for path_index, path in enumerate(paths):
        smoothed = smooth_path(path, reference)
        for centre in range(window, len(path) - window, max(1, window // 2)):
            piece = slice(centre - window, centre + window)
            raw_len = path_length(path[piece].astype(float))
            smooth_len = path_length(smoothed[piece])
            if smooth_len > 0:
                candidates.append((raw_len / smooth_len, path_index, centre))

    candidates.sort(reverse=True)
    # Un zoom par chemin au maximum, pour ne pas montrer trois fois le même endroit.
    chosen: list[tuple[float, int, int]] = []
    for candidate in candidates:
        if any(
            other[1] == candidate[1] and abs(other[2] - candidate[2]) < 2 * window
            for other in chosen
        ):
            continue
        chosen.append(candidate)
        if len(chosen) == n_zooms:
            break
    if not chosen:  # chemins trop courts pour une fenêtre
        chosen = [(1.0, 0, len(paths[0]) // 2)]

    n_zooms = len(chosen)
    n_rows = 2 if score is None else 3
    fig = plt.figure(figsize=(4.6 * n_zooms, 4.6 * n_rows), dpi=120)
    grid = fig.add_gridspec(n_rows, n_zooms)

    total_raw = sum(path_length(p.astype(float)) for p in paths)
    total_smooth = sum(path_length(smooth_path(p, reference)) for p in paths)
    correction = 100 * (total_smooth - total_raw) / total_raw if total_raw else 0.0

    def contiguous_runs(mask: np.ndarray) -> list[np.ndarray]:
        """Tronçons continus d'un masque, pour ne pas relier deux entrées
        séparées d'un même chemin dans la fenêtre par un trait fantôme."""
        indices = np.flatnonzero(mask)
        if len(indices) == 0:
            return []
        cuts = np.flatnonzero(np.diff(indices) > 1) + 1
        return [run for run in np.split(indices, cuts) if len(run) > 1]

    def draw_paths(axis, r0=0, c0=0, zoomed=False) -> dict[object, float]:
        """Trace brut puis lissés en coordonnées locales, renvoie les longueurs.

        Les longueurs sont cumulées sur tous les chemins traversant la fenêtre :
        une seule entrée de légende par courbe, quel que soit le nombre de
        branches visibles.
        """
        totals: dict[object, float] = {"raw": 0.0}
        totals.update({sigma: 0.0 for sigma in sigmas})

        for path in paths:
            if zoomed:
                inside = (
                    (path[:, 0] >= r0) & (path[:, 0] < r0 + 2 * half_size)
                    & (path[:, 1] >= c0) & (path[:, 1] < c0 + 2 * half_size)
                )
            else:
                inside = np.ones(len(path), dtype=bool)
            runs = contiguous_runs(inside)
            if not runs:
                continue

            smoothed_by_sigma = {
                sigma: smooth_path(path, sigma)
                for sigma in sigmas
                if zoomed or sigma == reference
            }
            for run in runs:
                axis.plot(
                    path[run, 1] - c0,
                    path[run, 0] - r0,
                    color="#FF8C00",
                    linewidth=1.6 if zoomed else 1.4,
                    marker="s" if zoomed else None,
                    markersize=3,
                )
                totals["raw"] += path_length(path[run].astype(float))
                for color, sigma in zip(colors, sigmas):
                    smoothed = smoothed_by_sigma.get(sigma)
                    if smoothed is None:
                        continue
                    axis.plot(
                        smoothed[run, 1] - c0,
                        smoothed[run, 0] - r0,
                        color=color,
                        linewidth=1.8 if zoomed else 1.0,
                    )
                    totals[sigma] += path_length(smoothed[run])
        return totals

    def legend_handles(totals: dict[object, float]) -> list[Line2D]:
        handles = [
            Line2D([0], [0], color="#FF8C00", lw=2,
                   label=f"brut ({totals['raw']:.1f} px)")
        ]
        for color, sigma in zip(colors, sigmas):
            handles.append(
                Line2D([0], [0], color=color, lw=2,
                       label=f"sigma={sigma:g} ({totals[sigma]:.1f} px)")
            )
        return handles

    # ---- ligne 1 : vues d'ensemble ----
    top = grid[0, :].subgridspec(1, 1 if score is None else 2)
    overview = fig.add_subplot(top[0, 0])
    overview.imshow(rgb)
    overview.axis("off")
    draw_paths(overview)
    overview.set_title(
        f"Brut {total_raw:.1f} px  ->  lissé sigma={reference:g} : "
        f"{total_smooth:.1f} px  ({correction:+.2f} %)\n"
        "somme des chemins, avant déduplication des portions partagées",
        fontsize=10,
    )

    if score is not None:
        score_view = fig.add_subplot(top[0, 1])
        image = score_view.imshow(score, cmap="magma")
        score_view.axis("off")
        draw_paths(score_view)
        score_view.set_title(
            "Score de fissure (ce qui guide le routage)", fontsize=10
        )
        fig.colorbar(image, ax=score_view, shrink=0.75)

    height, width = rgb.shape[:2]
    for rank, (_, path_index, centre) in enumerate(chosen, start=1):
        row, col = paths[path_index][centre]
        for view in (overview, score_view) if score is not None else (overview,):
            view.add_patch(
                plt.Rectangle(
                    (col - half_size, row - half_size),
                    2 * half_size,
                    2 * half_size,
                    fill=False,
                    edgecolor="yellow",
                    linewidth=1.5,
                )
            )
        overview.text(
            col + half_size + 4,
            row - half_size,
            str(rank),
            color="black",
            fontsize=10,
            bbox={"facecolor": "yellow", "alpha": 0.85, "pad": 1},
        )

    # ---- lignes 2 et 3 : zooms au pixel ----
    for rank, (_, path_index, centre) in enumerate(chosen):
        row, col = paths[path_index][centre]
        r0 = int(np.clip(row - half_size, 0, max(0, height - 2 * half_size)))
        c0 = int(np.clip(col - half_size, 0, max(0, width - 2 * half_size)))
        r1 = min(r0 + 2 * half_size, height)
        c1 = min(c0 + 2 * half_size, width)

        axis = fig.add_subplot(grid[1, rank])
        axis.imshow(rgb[r0:r1, c0:c1], interpolation="nearest")
        totals = draw_paths(axis, r0, c0, zoomed=True)
        excess = (
            100 * (totals["raw"] - totals[reference]) / totals[reference]
            if totals[reference] else 0.0
        )
        axis.set_title(
            f"zoom {rank + 1} — image  ({excess:+.1f} % de brut en trop)",
            fontsize=10,
        )
        axis.legend(
            handles=legend_handles(totals),
            fontsize=7,
            loc="upper left",
            framealpha=0.9,
        )
        axis.set_xticks([])
        axis.set_yticks([])

        if score is not None:
            axis = fig.add_subplot(grid[2, rank])
            axis.imshow(
                score[r0:r1, c0:c1],
                cmap="magma",
                interpolation="nearest",
            )
            draw_paths(axis, r0, c0, zoomed=True)
            axis.set_title(f"zoom {rank + 1} — score", fontsize=10)
            axis.set_xticks([])
            axis.set_yticks([])

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Diagnostic     : {output_path}")
    plt.show()
    plt.close(fig)


def measure_unique_edges(
    paths: list[np.ndarray],
    classifications: list[np.ndarray],
    owners: list[int] | None = None,
    smoothing_sigma: float = PATH_SMOOTHING_SIGMA,
) -> tuple[
    float,
    float,
    list[tuple[np.ndarray, str]],
    dict[int, tuple[float, float, float]],
    float,
]:
    """Measure unique graph edges and return drawable segments by class.

    Chaque liaison est stockée sous une clé indépendante de son orientation,
    donc une portion partagée par deux chemins (jonction, branches partiellement
    superposées) n'est comptée qu'une seule fois. Cette portion est attribuée à
    la première fissure qui la parcourt.

    La déduplication reste faite sur les pixels ENTIERS du chemin brut : c'est
    ce qui garantit qu'une portion commune est reconnue comme telle. Seule la
    longueur attribuée à chaque arête est mesurée sur le chemin lissé. La
    longueur brute est renvoyée en plus, pour pouvoir juger de la correction.
    """
    if owners is None:
        owners = list(range(len(paths)))

    unique_edges: dict[
        tuple[tuple[int, int], tuple[int, int]],
        tuple[float, float, str, np.ndarray, int],
    ] = {}

    for path, filled, owner in zip(paths, classifications, owners):
        smoothed = smooth_path(path, smoothing_sigma)
        for index in range(len(path) - 1):
            point_a = tuple(int(v) for v in path[index])
            point_b = tuple(int(v) for v in path[index + 1])
            key = tuple(sorted((point_a, point_b)))

            step = float(np.linalg.norm(smoothed[index + 1] - smoothed[index]))
            raw_step = float(np.linalg.norm(path[index + 1] - path[index]))
            if filled[index] and filled[index + 1]:
                edge_class = "filled"
            elif not filled[index] and not filled[index + 1]:
                edge_class = "open"
            else:
                edge_class = "transition"

            if key not in unique_edges:
                segment_xy = np.array(
                    [smoothed[index, ::-1], smoothed[index + 1, ::-1]],
                    dtype=float,
                )
                unique_edges[key] = (step, raw_step, edge_class, segment_xy, owner)

    length_open = 0.0
    length_filled = 0.0
    length_raw = 0.0
    per_crack: dict[int, list[float]] = {owner: [0.0, 0.0, 0.0] for owner in owners}
    segments_by_class = {
        "open": [],
        "filled": [],
        "transition": [],
    }

    for step, raw_step, edge_class, segment, owner in unique_edges.values():
        segments_by_class[edge_class].append(segment)
        if edge_class == "open":
            open_step, filled_step = step, 0.0
        elif edge_class == "filled":
            open_step, filled_step = 0.0, step
        else:
            open_step, filled_step = step / 2.0, step / 2.0
        length_open += open_step
        length_filled += filled_step
        length_raw += raw_step
        per_crack[owner][0] += open_step
        per_crack[owner][1] += filled_step
        per_crack[owner][2] += raw_step

    drawable = [
        (
            np.asarray(segments_by_class[name]),
            CLASS_COLORS[name],
        )
        for name in ("open", "filled", "transition")
        if segments_by_class[name]
    ]

    per_crack_tuples = {
        owner: (values[0], values[1], values[2])
        for owner, values in per_crack.items()
    }
    return length_open, length_filled, drawable, per_crack_tuples, length_raw


# ====================================================================
# SORTIES
# ====================================================================


def save_result_figure(
    rgb: np.ndarray,
    crack_groups: list[list[tuple[int, int]]],
    drawable: list[tuple[np.ndarray, str]],
    length_open: float,
    length_filled: float,
    output_path: Path,
    two_populations: bool = True,
) -> None:
    """Save the classified crack overlay."""
    total = length_open + length_filled

    fig, ax = plt.subplots(figsize=(11, 8), dpi=200)
    ax.imshow(rgb)

    for segments, color in drawable:
        ax.add_collection(
            LineCollection(
                segments,
                colors=color,
                linewidths=2.2,
            )
        )

    for crack_number, group in enumerate(crack_groups, start=1):
        for number, (row, col) in enumerate(group, start=1):
            ax.scatter(col, row, s=24, c="yellow", edgecolors="black", linewidths=0.5)
            ax.text(
                col + 4,
                row,
                f"{crack_number}.{number}",
                fontsize=7,
                bbox={"facecolor": "white", "alpha": 0.75, "pad": 1},
            )

    title = (
        f"Open: {length_open:.1f} px | "
        f"Filled: {length_filled:.1f} px | "
        f"Total: {total:.1f} px | "
        f"Cracks: {len(crack_groups)}"
    )
    if not two_populations:
        title += "\n(une seule population de gris : fissure considérée ouverte)"
    ax.set_title(title)
    ax.axis("off")
    ax.legend(
        handles=[
            Line2D([0], [0], color=CLASS_COLORS["open"], lw=3, label="Open"),
            Line2D([0], [0], color=CLASS_COLORS["filled"], lw=3, label="Filled"),
            Line2D([0], [0], color=CLASS_COLORS["transition"], lw=3, label="Transition"),
        ],
        loc="lower left",
        framealpha=0.9,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_summary(
    output_path: Path,
    per_crack: dict[int, tuple[float, float, float]],
    length_open: float,
    length_filled: float,
    length_raw: float,
    open_mean: float,
    filled_mean: float,
    two_populations: bool,
) -> None:
    """Save one CSV row per crack plus a total row."""

    def row(
        name: str,
        open_px: float,
        filled_px: float,
        raw_px: float,
    ) -> dict[str, str]:
        total = open_px + filled_px
        fraction = 100 * filled_px / total if total > 0 else 0.0
        return {
            "crack_id": name,
            "length_total_px": f"{total:.6f}",
            "length_open_px": f"{open_px:.6f}",
            "length_filled_px": f"{filled_px:.6f}",
            "filled_fraction_percent": f"{fraction:.6f}",
            "length_total_raw_px": f"{raw_px:.6f}",
            "smoothing_sigma": f"{PATH_SMOOTHING_SIGMA:g}",
            "open_gray_mean": f"{open_mean:.6f}",
            "filled_gray_mean": f"{filled_mean:.6f}",
            "two_populations": str(two_populations),
        }

    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "crack_id",
                "length_total_px",
                "length_open_px",
                "length_filled_px",
                "filled_fraction_percent",
                "length_total_raw_px",
                "smoothing_sigma",
                "open_gray_mean",
                "filled_gray_mean",
                "two_populations",
            ],
        )
        writer.writeheader()
        for owner in sorted(per_crack):
            open_px, filled_px, raw_px = per_crack[owner]
            writer.writerow(row(str(owner + 1), open_px, filled_px, raw_px))
        writer.writerow(row("all", length_open, length_filled, length_raw))


# ====================================================================
# ANNOTATIONS (passerelle vers le traitement par lots)
# ====================================================================


def save_annotations(
    output_path: Path,
    image_path: Path,
    shape: tuple[int, int],
    crack_groups: list[list[tuple[int, int]]],
) -> None:
    """Enregistre les points d'ancrage pour rejouer l'analyse sans clics."""
    payload = {
        "image": image_path.name,
        "shape": [int(shape[0]), int(shape[1])],
        "crack_groups": [
            [[int(row), int(col)] for row, col in group] for group in crack_groups
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_annotations(path: Path) -> list[list[tuple[int, int]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        [(int(row), int(col)) for row, col in group]
        for group in payload["crack_groups"]
    ]


def parse_groups(values: list[str] | None) -> list[list[tuple[int, int]]] | None:
    """Parse ``--groups "x1,y1;x2,y2" "x3,y3;x4,y4"`` and return row,column groups."""
    if values is None:
        return None
    groups = []
    for value in values:
        group = []
        for token in value.split(";"):
            token = token.strip()
            if not token:
                continue
            x_text, y_text = token.split(",")
            group.append((int(round(float(y_text))), int(round(float(x_text)))))
        groups.append(group)
    return groups


def validate_groups(
    crack_groups: list[list[tuple[int, int]]],
    shape: tuple[int, int],
) -> None:
    height, width = shape
    for index, group in enumerate(crack_groups, start=1):
        if len(group) < 2:
            raise ValueError(f"Fissure {index}: au moins 2 points sont nécessaires")
        for row, col in group:
            if not (0 <= row < height and 0 <= col < width):
                raise ValueError(f"Point outside image: row={row}, column={col}")


def analyse_image(
    image_path: Path,
    crack_groups: list[list[tuple[int, int]]] | None,
    output_dir: Path | None,
    show_cost: bool = False,
    show_smoothing: bool = False,
) -> tuple[Path, Path, Path, float, float]:
    """Analyse complète d'une image. crack_groups=None => sélection interactive."""
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    gray = np.asarray(Image.open(image_path).convert("L"), dtype=float) / 255.0

    # La carte de coût est nécessaire dès la sélection : les chemins sont
    # tracés en direct pendant les clics.
    score, cost = build_cost_image(gray, show=show_cost)

    if crack_groups is None:
        crack_groups = select_crack_groups(rgb, cost)
    validate_groups(crack_groups, gray.shape)

    paths = extract_paths(cost, crack_groups)
    if not paths:
        raise SystemExit("Aucun chemin extrait.")
    owners = path_owners(crack_groups)
    classes, _, open_mean, filled_mean, two_populations = classify_paths(gray, paths)
    length_open, length_filled, drawable, per_crack, length_raw = measure_unique_edges(
        paths,
        classes,
        owners,
    )

    output_dir = output_dir if output_dir is not None else image_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_path = output_dir / f"{image_path.stem}_crack_analysis.png"
    csv_path = output_dir / f"{image_path.stem}_crack_lengths.csv"
    annotations_path = output_dir / f"{image_path.stem}_crack_points.json"

    if show_smoothing:
        show_smoothing_diagnostic(
            rgb,
            paths,
            score,
            output_dir / f"{image_path.stem}_smoothing_check.png",
        )

    save_result_figure(
        rgb,
        crack_groups,
        drawable,
        length_open,
        length_filled,
        figure_path,
        two_populations,
    )
    save_summary(
        csv_path,
        per_crack,
        length_open,
        length_filled,
        length_raw,
        open_mean,
        filled_mean,
        two_populations,
    )
    save_annotations(annotations_path, image_path, gray.shape, crack_groups)

    total = length_open + length_filled
    print(f"Image          : {image_path.name}")
    print(f"Cracks         : {len(crack_groups)} ({len(paths)} paths)")
    print(f"Total length   : {total:.2f} px")
    if PATH_SMOOTHING_SIGMA > 0 and length_raw > 0:
        print(
            f"  (brut {length_raw:.2f} px, "
            f"correction du lissage {100 * (total - length_raw) / length_raw:+.2f} %)"
        )
    print(f"Open length    : {length_open:.2f} px")
    print(f"Filled length  : {length_filled:.2f} px")
    if total > 0:
        print(f"Filled fraction: {100 * length_filled / total:.2f} %")
    if not two_populations:
        print("Classification : une seule population de gris, tout est classé ouvert")
    for owner in sorted(per_crack):
        open_px, filled_px, raw_px = per_crack[owner]
        print(
            f"  crack {owner + 1}: total {open_px + filled_px:.2f} px "
            f"(brut {raw_px:.2f}), "
            f"open {open_px:.2f} px, filled {filled_px:.2f} px"
        )
    print(f"Figure         : {figure_path}")
    print(f"CSV            : {csv_path}")
    print(f"Annotations    : {annotations_path}")

    return figure_path, csv_path, annotations_path, length_open, length_filled


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure open and filled crack lengths in an SEM image."
    )
    parser.add_argument("image", nargs="?", type=Path, help="Input image")
    parser.add_argument(
        "--groups",
        nargs="+",
        metavar="X1,Y1;X2,Y2;...",
        help="Skip interactive selection: one argument per crack, points separated by ';'",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        help="Reuse the anchor points saved by a previous run (JSON)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory; defaults to the image directory",
    )
    parser.add_argument(
        "--show-cost",
        action="store_true",
        help="Display the intermediate cost-map figure",
    )
    parser.add_argument(
        "--show-smoothing",
        action="store_true",
        help="Compare raw and smoothed paths, zoomed, over image and score map",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the final result window",
    )
    args = parser.parse_args()

    image_path = args.image if args.image is not None else choose_image()
    image_path = image_path.expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    if args.annotations is not None:
        crack_groups = load_annotations(args.annotations.expanduser().resolve())
    else:
        crack_groups = parse_groups(args.groups)

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else None
    )

    figure_path, _, _, _, _ = analyse_image(
        image_path,
        crack_groups,
        output_dir,
        show_cost=args.show_cost,
        show_smoothing=args.show_smoothing,
    )

    if not args.no_show:
        result = Image.open(figure_path)
        plt.figure(figsize=(11, 8))
        plt.imshow(result)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
