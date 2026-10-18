"""Rapport PDF d'une analyse automatique : le resultat, la carte, les tuiles, des exemples.

Ecrit pour etre compris d'un coup d'oeil, sans connaitre la methode. Les
details techniques restent dans `resultat.json` et le journal.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

# Memes teintes que l'application ; le bleu et le vert d'eau sont ecartes en
# clarte pour rester distincts, daltonismes compris (ecart OKLab >= 15).
COLORS = {"open": "#d43d2f", "filled": "#0a5a8f", "reopened": "#2fb5a0"}
LABELS = {"open": "Non colmatée", "filled": "Colmatée", "reopened": "Rouverte"}
INK, MUTED = "#1f1f1f", "#6b6b6b"
WARN = "#b26a00"
# Au-dela de cet ecart entre la resolution des fenetres de l'image et celle de
# l'apprentissage, le classement du colmatage n'a pas ete valide.
RESOLUTION_TOLERANCE = 1.3


def _overview(source, step: int):
    if getattr(source, "direct", False):
        return np.asarray(source._memmap[::step, ::step])
    return np.asarray(source.read(0, 0, source.width, source.height))[::step, ::step]


def _pct(value) -> str:
    return "—" if value is None or value != value else f"{100 * value:.0f} %"   # insecable


def _num(value: float, digits: int = 0) -> str:
    """Nombre a la francaise : 1 551 ; 0,37."""
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


def _mm(value: float) -> str:
    return f"{_num(value, 1 if value < 10 else 0)} mm"


def _date(text: str) -> str:
    try:
        return datetime.fromisoformat(text).strftime("%d/%m/%Y à %H:%M")
    except (TypeError, ValueError):
        return str(text)


def current_state(annotations, mask, pixel_um: float) -> dict | None:
    """Les fissures de la mosaique telles qu'elles sont dans l'application : l'analyse importee,
    plus vos corrections (None si l'analyse n'a pas ete importee)."""
    from .chaine import summarize
    from .donnees import polyline_length

    cracks = annotations.validated + annotations.unverified
    if not any(c.get("source") == "analyse" for c in cracks):
        return None
    segments = [{"crack": c["id"], "category": c["category"], "confidence": None, "zone_id": c["zone_id"],
                 "length_um": polyline_length(c["geometry"]) * pixel_um, "geometry": c["geometry"]}
                for c in cracks if c["category"] in COLORS]
    done = annotations.done_ids
    return {"segments": segments, "resume": summarize(segments, mask, annotations),
            "tuiles_validees": len(done), "tuiles": len(annotations.zones),
            "longueur_verifiee_mm": sum(s["length_um"] for s in segments if s["zone_id"] in done) / 1000,
            "par_etat_verifie_mm": {c: sum(s["length_um"] for s in segments if s["zone_id"] in done
                                           and s["category"] == c) / 1000 for c in COLORS}}


def resolution_problem(image_pixel_um, model_pixel_um) -> str | None:
    """Avertissement court si l'image est trop grossiere pour le modele de colmatage."""
    from .bandes import reduction_for

    if not image_pixel_um or not model_pixel_um:
        return None
    seen = image_pixel_um * reduction_for(image_pixel_um)          # pixel des fenetres classees
    learnt = model_pixel_um * reduction_for(model_pixel_um)
    if seen <= learnt * RESOLUTION_TOLERANCE:
        return None
    return (f"⚠ Non fiable : image à {_num(image_pixel_um, 2)} µm par pixel, "
            f"modèle appris à {_num(model_pixel_um, 2)} µm")


def _summary_page(figure, result, summary, raw, current, warning: str | None) -> None:
    """Page 1 : le resultat en grand et le detail par etat."""
    name = Path(result.get("image_path") or result["image"]).name
    figure.text(0.08, 0.955, name, fontsize=17, weight="bold", color=INK)
    figure.text(0.08, 0.932, f"Analyse automatique du {_date(result.get('analyse_le'))}", fontsize=10,
                color=MUTED)

    share = summary["part_colmatee"]
    figure.text(0.08, 0.845, _pct(share), fontsize=46, weight="bold", color=MUTED if warning else INK)
    figure.text(0.36, 0.875, "de la longueur des fissures\nest colmatée (colmatée ou rouverte)",
                fontsize=12.5, color=INK, va="center", linespacing=1.4)
    if warning:
        figure.text(0.36, 0.835, warning, fontsize=10.5, color=WARN, weight="bold", va="center")
    low, high = summary["part_colmatee_ic95_tuiles"]
    note = f"Selon les zones de l'échantillon : entre {_pct(low)} et {_pct(high)}." if low is not None else ""
    if current and current["tuiles_validees"]:
        note += (f"\nAvec vos corrections ({current['tuiles_validees']} tuiles validées sur {current['tuiles']}) ; "
                 f"avant elles, la machine donnait {_pct(raw['part_colmatee'])}.")
    figure.text(0.08, 0.80, note, fontsize=10, color=MUTED, va="top", linespacing=1.5)

    # Barre : la longueur totale, partagee par etat (un blanc entre les morceaux).
    total = summary["longueur_totale_mm"]
    lengths = summary["longueur_mm"]
    bar = figure.add_axes((0.08, 0.705, 0.84, 0.035))
    left = 0.0
    for category in COLORS:
        width = lengths[category] / total if total else 0
        if width > 0:
            bar.barh(0, width, left=left, color=COLORS[category], edgecolor="white", linewidth=2, height=1)
        left += width
    bar.set_xlim(0, 1)
    bar.set_ylim(-0.5, 0.5)
    bar.axis("off")

    # Tableau : un carre de couleur, l'etat, la longueur, la part.
    y = 0.665
    for category, label in LABELS.items():
        figure.add_artist(_square(figure, 0.08, y - 0.006, COLORS[category]))
        figure.text(0.11, y, label, fontsize=11, color=INK, va="center")
        figure.text(0.62, y, _mm(lengths[category]), fontsize=11, color=INK, va="center", ha="right")
        figure.text(0.76, y, _pct(lengths[category] / total if total else None), fontsize=11, color=MUTED,
                    va="center", ha="right")
        y -= 0.03
    figure.text(0.11, y, "Total", fontsize=11, color=INK, va="center", weight="bold")
    figure.text(0.62, y, _mm(total), fontsize=11, color=INK, va="center", ha="right", weight="bold")
    area = summary.get("surface_echantillon_mm2")
    if area:
        figure.text(0.08, y - 0.035, f"Échantillon de {_num(area)} mm², soit "
                                     f"{_num(summary['densite_mm_par_mm2'], 2)} mm de fissure par mm².",
                    fontsize=10, color=MUTED, va="center")

    figure.text(0.08, 0.04, "Bords de l'échantillon et petits traits isolés (moins de 50 µm) exclus. L'état est "
                            "relevé tous les 10 µm le long de chaque fissure.", fontsize=8, color=MUTED)


def _square(figure, x, y, color):
    from matplotlib.patches import Rectangle
    return Rectangle((x, y), 0.016, 0.012, transform=figure.transFigure, color=color, figure=figure)


def build(result: dict, annotations, source, mask, out_path: Path, current: dict | None = None,
          model_pixel_um: float | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch

    raw = result["resume"]
    summary = current["resume"] if current else raw
    pixel_um = result["pixel_um"]
    segments = current["segments"] if current else result["segments"]
    model_pixel_um = model_pixel_um or (result.get("colmatage") or {}).get("pixel_um")
    page = (8.27, 11.69)
    with PdfPages(out_path) as pdf:
        # 1. Le resultat
        figure = Figure(figsize=page)
        _summary_page(figure, result, summary, raw, current, resolution_problem(pixel_um, model_pixel_um))
        pdf.savefig(figure)

        # 2. La carte
        step = 60
        small = _overview(source, step)
        figure = Figure(figsize=page, layout="constrained")
        halves = figure.subplots(1, 2)
        split_row = small.shape[0] // 2
        mm = pixel_um / 1000
        width_mm = small.shape[1] * step * mm
        for k, ax in enumerate(halves):
            rows = slice(0, split_row) if k == 0 else slice(split_row, small.shape[0])
            top, bottom = rows.start * step * mm, rows.stop * step * mm
            ax.imshow(small[rows], cmap="gray", vmin=0, vmax=255, extent=(0, width_mm, bottom, top))
            for category, color in COLORS.items():
                pieces = [[(x * mm, y * mm) for x, y in s["geometry"]] for s in segments
                          if s["category"] == category
                          and any(top <= y * mm <= bottom for _, y in s["geometry"][::max(1, len(s["geometry"]) // 4)])]
                ax.add_collection(LineCollection(pieces, colors=color, linewidths=0.6))
            ax.set_xlim(0, width_mm)
            ax.set_ylim(bottom, top)
            ax.set_xlabel("mm", color=MUTED)
            ax.tick_params(colors=MUTED, labelsize=7)
        halves[0].set_title("haut de l'échantillon", fontsize=9, color=MUTED)
        halves[1].set_title("bas de l'échantillon", fontsize=9, color=MUTED)
        halves[0].legend(handles=[Patch(color=c, label=f"{LABELS[k]} ({_mm(summary['longueur_mm'][k])})")
                                  for k, c in COLORS.items()], loc="lower left", fontsize=7)
        figure.suptitle("Où sont les fissures, et dans quel état", fontsize=13, color=INK)
        pdf.savefig(figure)

        # 3. Tuile par tuile
        grid = np.full((max(z["row_index"] for z in annotations.zones) + 1,
                        max(z["col_index"] for z in annotations.zones) + 1), np.nan)
        totals = np.zeros_like(grid)
        zones = {z["id"]: z for z in annotations.zones}
        for key, lengths in summary["tuiles"].items():
            if key in ("None", None) or int(key) not in zones:
                continue
            zone = zones[int(key)]
            total = sum(lengths.values())
            totals[zone["row_index"], zone["col_index"]] = total
            if total >= 0.3:
                grid[zone["row_index"], zone["col_index"]] = (lengths["filled"] + lengths["reopened"]) / total
        figure = Figure(figsize=page, layout="constrained")
        left, right = figure.subplots(1, 2)
        shares = matplotlib.colormaps["Blues"].with_extremes(bad="#e4e4e4")
        image = left.imshow(100 * grid, cmap=shares, vmin=0, vmax=100)
        figure.colorbar(image, ax=left, label="% colmaté", shrink=0.6)
        left.set_title("Part colmatée\n(gris : trop peu de fissures)", fontsize=10, color=INK)
        image = right.imshow(totals, cmap="Oranges")
        figure.colorbar(image, ax=right, label="mm de fissures", shrink=0.6)
        right.set_title("Longueur de fissures", fontsize=10, color=INK)
        for ax in (left, right):
            ax.set_xlabel("colonne", color=MUTED)
            ax.set_ylabel("ligne", color=MUTED)
            ax.tick_params(colors=MUTED, labelsize=7)
        figure.suptitle("Tuile par tuile (une case = une tuile de 3 mm)", fontsize=13, color=INK)
        pdf.savefig(figure)

        # 4. Exemples au hasard, pour un controle visuel
        rng = np.random.default_rng(1)
        figure = Figure(figsize=page, layout="constrained")
        axes = figure.subplots(6, 4)
        half = int(round(40 / pixel_um))
        for row_pair, category in enumerate(COLORS):
            candidates = [s for s in segments if s["category"] == category and s["length_um"] >= 20]
            picks = [candidates[i] for i in rng.permutation(len(candidates))[:8]]
            for k in range(8):
                ax = axes[2 * row_pair + k // 4, k % 4]
                ax.set_xticks([])
                ax.set_yticks([])
                if k >= len(picks):
                    ax.axis("off")
                    continue
                s = picks[k]
                cx, cy = s["geometry"][len(s["geometry"]) // 2]
                x0 = int(max(0, min(source.width - 2 * half, cx - half)))
                y0 = int(max(0, min(source.height - 2 * half, cy - half)))
                crop = np.asarray(source.read(x0, y0, 2 * half, 2 * half))
                ax.imshow(crop, cmap="gray", vmin=0, vmax=255)
                g = np.asarray(s["geometry"]) - [x0, y0]
                ax.plot(g[:, 0], g[:, 1], color=COLORS[category], linewidth=0.8, alpha=0.7)
                ax.set_xlim(0, 2 * half)
                ax.set_ylim(2 * half, 0)
                doubt = " ?" if s.get("confidence") is not None and s["confidence"] < 0.6 else ""
                ax.set_title(f"{LABELS[category]}{doubt}", fontsize=7, color=INK)
        figure.suptitle("Exemples tirés au hasard, pour vérifier à l'œil\n"
                        "carrés de 80 µm ; le trait de couleur montre l'état donné par la machine ; "
                        "« ? » : elle hésite", fontsize=10, color=INK)
        pdf.savefig(figure)
    return out_path


def build_for(annotations, result=None) -> Path:
    """Rapport du dernier resultat, avec vos corrections s'il a ete importe."""
    from annotation_app.echelle import pixel_size_um
    from annotation_app.paths import MODELS_DIR
    from crop_roi import LargeGrayImage

    from . import chaine
    from .apprentissage import MODEL_PATH_NAME
    from .classement import ColmatageModel
    from .masque import SampleMask

    result = result or chaine.load_result(annotations)
    out = annotations.output_dir()
    source = LargeGrayImage(annotations.image_path)
    mask = SampleMask.load_or_compute(annotations.image_path, source, pixel_size_um(annotations.image_path), out)
    current = current_state(annotations, mask, result["pixel_um"])
    model_pixel_um = (result.get("colmatage") or {}).get("pixel_um")
    if model_pixel_um is None:               # resultat d'avant l'enregistrement de ce pixel
        model = ColmatageModel.load(MODELS_DIR / MODEL_PATH_NAME)
        model_pixel_um = model.pixel_um if model is not None else None
    return build(result, annotations, source, mask, out / "rapport.pdf", current, model_pixel_um)
