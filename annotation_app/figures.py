"""Figures du bilan : longueurs, orientation, carte des axes, longueur par tuile.

Dessinees sur une `matplotlib.figure.Figure` fournie, a partir du resultat de
`Session.report` : l'interface les affiche, et les enregistre en PNG, SVG ou
PDF telles quelles. Longueurs et coordonnees en mm quand la taille de pixel
est connue, en pixels sinon.
"""

from __future__ import annotations

from .length_analysis import CATEGORIES, COLORS, LABELS

KINDS = {
    "lengths": "Longueurs par catégorie",
    "orientation": "Orientation des fissures",
    "map": "Carte des axes",
    "spatial": "Longueur par tuile",
}
SPATIAL = {"map", "spatial"}


def draw(figure, result: dict, kind: str, extent: tuple, preview=None, tolerance_px: float = 2.0) -> None:
    """Dessine la figure `kind` ; `extent` = (x0, y0, x1, y1) en pixels de la mosaique.

    `preview` : (pixels, etendue) de l'image de fond, pour les cartes.
    """
    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch, Rectangle

    figure.clear()
    pixel_um = result.get("pixel_um")
    scale = pixel_um / 1000 if pixel_um else 1.0          # pixels -> mm
    unit = "mm" if pixel_um else "px"
    ax = figure.subplots()

    if kind == "lengths":
        values = [result["length_px"][c] * scale for c in CATEGORIES]
        ax.bar(LABELS, values, color=COLORS)
        ax.set_ylabel(f"Longueur ({unit})")
        ax.tick_params(axis="x", rotation=20)
    elif kind == "orientation":
        values = [v * scale for v in result["orientation_length_px"]]
        ax.bar(range(0, 180, 10), values, width=10, align="edge", color="#4a6fa5", edgecolor="white")
        ax.set_xlim(0, 180)
        ax.set_xticks(range(0, 181, 30))
        ax.set_xlabel("Orientation (°) — 0 : horizontale, 90 : verticale")
        ax.set_ylabel(f"Longueur ({unit})")
    elif kind in SPATIAL:
        x0, y0, x1, y1 = extent
        if preview is not None:
            pixels, (px0, py0, px1, py1) = preview
            ax.imshow(pixels, cmap="gray", vmin=0, vmax=255,
                      extent=(px0 * scale, px1 * scale, py1 * scale, py0 * scale))
        if kind == "map":
            for category, color in zip(CATEGORIES, COLORS):
                lines = [[(x * scale, y * scale) for x, y in s["geometry"]]
                         for s in result["segments"] if s["category"] == category]
                if lines:
                    ax.add_collection(LineCollection(lines, colors=color, linewidths=1))
            present = {s["category"] for s in result["segments"]}
            handles = [Patch(color=c, label=l) for cat, c, l in zip(CATEGORIES, COLORS, LABELS) if cat in present]
            if handles:
                ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
        else:
            zones = result["zones"]
            norm = Normalize(0, max([z["length_px"] * scale for z in zones] + [1e-9]))
            cmap = colormaps["viridis"]
            for z in zones:
                ax.add_patch(Rectangle((z["x"] * scale, z["y"] * scale), z["width"] * scale, z["height"] * scale,
                                       facecolor=cmap(norm(z["length_px"] * scale)), alpha=.8,
                                       edgecolor="white", linewidth=.3))
            figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=f"Longueur par tuile ({unit})")
        ax.set(xlim=(x0 * scale, x1 * scale), ylim=(y1 * scale, y0 * scale), aspect="equal",
               xlabel=f"x ({unit})", ylabel=f"y ({unit})")
    else:
        raise ValueError(f"Figure inconnue : {kind}")

    source = "Axes automatiques (non vérifiés)" if result.get("source") == "automatic" else "Fissures tracées"
    scope = "tuiles validées" if result.get("validated_only") else "toutes les tuiles"
    title = f"{KINDS[kind]}\n{source} · {scope}"
    if result.get("unprocessed_zone_ids"):
        title += f" · {len(result['unprocessed_zone_ids'])} tuile(s) non calculée(s)"
    ax.set_title(title, fontsize=10)
    total = result["total_px"] * scale
    figure.supxlabel(f"Total {total:.2f} {unit} · superpositions comptées une fois "
                     f"(tolérance {tolerance_px:g} px)", fontsize=7)
