"""Apprentissage du classement du colmatage, et sa mesure sur les tuiles de test.

1. Les fissures validees sont rechainees (une fissure coupee en morceaux de
   categories differentes redevient une fissure, avec ses changements d'etat) ;
2. chaque fissure donne une fenetre tous les 10 um, etiquetee par la categorie
   du morceau ou tombe son centre (les fenetres a cheval sur un changement ne
   servent pas a apprendre) ;
3. le reseau apprend sur les lignes d'apprentissage ; temperatures et cout de
   changement se reglent sur les lignes de validation ;
4. tout est mesure sur les lignes de test, jamais vues, contre le modele
   precedent reappris sur les memes lignes (comparaison equitable).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from . import bandes, classement, donnees
from .evaluation import Score, describe

CATEGORIES = classement.CATEGORIES
MODEL_PATH_NAME = "colmatage_reseau.pt"


def _identity(annotations, chains) -> str:
    digest = hashlib.sha256()
    for chain in chains:
        digest.update(json.dumps([chain["segments"], chain["geometry"][:2]]).encode())
    digest.update(json.dumps([bandes.STEP_UM, bandes.WINDOW_UM, bandes.HALF_UM, bandes.TARGET_UM,
                              bandes.ACROSS, 2]).encode())
    return digest.hexdigest()[:16]


def build_windows(annotations, split, source, pixel_um, log=print) -> dict:
    """Fenetres etiquetees de toutes les fissures validees, mises en cache sur disque."""
    zones = {z["id"]: z for z in annotations.zones}
    chains = donnees.chains(annotations.validated)
    identity = _identity(annotations, chains)
    cache = annotations.output_dir() / "fenetres_colmatage.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            if str(data["identity"]) == identity:
                meta = json.loads(str(data["meta"]))
                return {"windows": data["windows"], "labels": data["labels"], "chain": data["chain"],
                        "centers": data["centers"], "chains": meta, "identity": identity}
    half_window = bandes.WINDOW_UM / 2 / pixel_um
    all_windows, labels, chain_ids, centers, meta = [], [], [], [], []
    for index, chain in enumerate(chains):
        if index % 50 == 0:
            log(f"  fenetres : fissure {index + 1}/{len(chains)}")
        w, c, _ = bandes.windows(source.read, source.width, source.height, chain["geometry"], pixel_um)
        total_um = donnees.polyline_length(chain["geometry"]) * pixel_um
        meta.append({"role": donnees.role(zones[chain["zone_id"]], split), "zone_id": chain["zone_id"],
                     "total_um": total_um,
                     "segments": [{"start_um": a * pixel_um, "end_um": b * pixel_um, "category": cat}
                                  for a, b, cat, _ in chain["segments"]]})
        if w is None:
            continue
        for k, center_um in enumerate(c):
            s = center_um / pixel_um
            label = -1
            for a, b, cat, _ in chain["segments"]:
                if a <= s <= b:
                    label = CATEGORIES.index(cat) if cat in CATEGORIES else -1
                    break
            # A cheval sur un changement : on ne sait pas quoi apprendre.
            cats = {cat for a, b, cat, _ in chain["segments"] if b > s - half_window and a < s + half_window}
            if len(cats) > 1:
                label = -1
            labels.append(label)
            chain_ids.append(index)
            centers.append(center_um)
        all_windows.append(w)
    data = {"windows": np.concatenate(all_windows), "labels": np.array(labels, dtype=np.int8),
            "chain": np.array(chain_ids, dtype=np.int32), "centers": np.array(centers, dtype=np.float32),
            "chains": meta, "identity": identity}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, windows=data["windows"], labels=data["labels"], chain=data["chain"],
             centers=data["centers"], meta=np.array(json.dumps(meta)), identity=np.array(identity))
    return data


def _by_chain(data, role):
    """[(indice de fissure, indices de ses fenetres)] pour un role."""
    result = []
    for index, chain in enumerate(data["chains"]):
        if chain["role"] == role:
            rows = np.nonzero(data["chain"] == index)[0]
            if len(rows):
                result.append((index, rows))
    return result


def evaluate(data, probabilities, role, change_cost, tolerance_um=20.0, splitter=None) -> dict:
    """Score d'un jeu de probabilites par fenetre (meme ordre que data["windows"])."""
    score = Score(tolerance_um)
    for index, rows in _by_chain(data, role):
        chain = data["chains"][index]
        centers = data["centers"][rows]
        p = probabilities[rows]
        if splitter is not None:
            predicted = splitter(centers, p, chain["total_um"])
        else:
            predicted = classement.segments(classement.viterbi(p, change_cost), centers, chain["total_um"], p)
        score.add(chain["segments"], predicted, tile=chain["zone_id"])
    return score.summary()


def tune_change_cost(data, probabilities, log=print) -> float:
    """Cout de changement, regle sur la validation.

    Regle : ne jamais proposer MOINS de changements d'etat que l'annotateur
    n'en a fait (l'etat change vraiment le long des fissures : un lissage trop
    fort les effacerait), et parmi ces couts, le plus juste en longueur.
    """
    trials = []
    for cost in (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 13.0, 16.0, 20.0, 25.0):
        s = evaluate(data, probabilities, "validation", cost)
        trials.append((cost, s))
        log(f"    cout {cost:5.1f} : longueur {s['longueur_bien_classee']:.1%}, changements "
            f"{s['changements_predits']} proposes / {s['changements_vrais']} vrais, F1 {s['changements_f1']:.0%}")
    allowed = [(c, s) for c, s in trials if s["changements_predits"] >= s["changements_vrais"]] or trials[:1]
    return max(allowed, key=lambda item: (item[1]["longueur_bien_classee"], -item[0]))[0]


def baseline(annotations, split, data, source, pixel_um, log=print):
    """(probabilites, modele) du modele par descripteurs, reappris sur les lignes
    d'apprentissage ; garde sur disque tant que les annotations ne changent pas."""
    import joblib

    out = annotations.output_dir()
    cache, saved = out / "precedent_probabilites.npz", out / "precedent_modele.joblib"
    if cache.exists() and saved.exists():
        with np.load(cache, allow_pickle=False) as stored:
            if str(stored["identity"]) == data["identity"]:
                return stored["probabilities"], joblib.load(saved)
    found = baseline_probabilities(annotations, split, data, source, pixel_um, log)
    if found is None:
        return None, None
    probabilities, artifact = found
    np.savez(cache, probabilities=probabilities, identity=np.array(data["identity"]))
    joblib.dump(artifact, saved, compress=3)
    return probabilities, artifact


def baseline_probabilities(annotations, split, data, source, pixel_um, log=print):
    """Le modele precedent (descripteurs + gradient boosting), reappris sur les seules lignes
    d'apprentissage, evalue aux memes positions que le reseau. Rend (probabilites, modele)."""
    from colmatage import modele
    from colmatage.profils import trace_windows

    zones = {z["id"]: z for z in annotations.zones}
    traces = [c for c in annotations.validated if c["category"] in CATEGORIES
              and donnees.role(zones[c["zone_id"]], split) == "apprentissage"]
    try:
        artifact, report = modele.train(source.read, source.width, source.height, traces, pixel_um)
    except ValueError as error:
        log(f"  modele precedent impossible : {error}")
        return None
    log(f"  modele precedent reappris sur {len(traces)} morceaux")
    from annotation_app.polyligne import cumulative, cut

    chains = donnees.chains(annotations.validated)
    probabilities = np.full((len(data["labels"]), 3), 1 / 3)
    chunk = 2000.0 / pixel_um                    # par morceaux de 2 mm : la lecture reste bornee
    for role in ("validation", "test"):
        for index, rows in _by_chain(data, role):
            geometry = chains[index]["geometry"]
            total = cumulative(geometry)[-1]
            distances, features = [], []
            for start in np.arange(0.0, total, chunk):
                part = cut(geometry, start, min(total, start + chunk))
                for w in trace_windows(source.read, source.width, source.height, part, pixel_um):
                    distances.append(w.distance_um + start * pixel_um)
                    features.append(w.features)
            if not features:
                continue
            distances = np.array(distances)
            p = artifact["model"].predict_proba(np.array(features))
            # Chaque fenetre du reseau prend la fenetre du modele precedent la plus proche.
            nearest = np.abs(distances[None, :] - data["centers"][rows][:, None]).argmin(axis=1)
            probabilities[rows] = p[nearest]
    return probabilities, artifact


def run(database: Path, image_path: Path | None = None, epochs: int = 18, log=print,
        retrain: bool = True) -> dict:
    """`retrain=False` : reprend le reseau enregistre ; ne refait que le reglage et la mesure."""
    from annotation_app.echelle import require_pixel_size_um
    from annotation_app.paths import MODELS_DIR
    from colmatage.modele import portions_from_windows
    from crop_roi import LargeGrayImage

    annotations = donnees.load(database, image_path)
    split = donnees.split_for(annotations)
    pixel_um = require_pixel_size_um(annotations.image_path)
    source = LargeGrayImage(annotations.image_path)
    log("Fenetres le long des fissures validees...")
    data = build_windows(annotations, split, source, pixel_um, log)
    roles = np.array([data["chains"][i]["role"] for i in data["chain"]])
    known = data["labels"] >= 0
    train_rows = np.nonzero(known & (roles == "apprentissage"))[0]
    val_rows = np.nonzero(known & (roles == "validation"))[0]
    log(f"{len(train_rows)} fenetres d'apprentissage, {len(val_rows)} de validation, "
        f"{int((known & (roles == 'test')).sum())} de test")
    counts = np.bincount(data["labels"][train_rows], minlength=3)
    log("  apprentissage : " + ", ".join(f"{c} {n}" for c, n in zip(CATEGORIES, counts)))

    path = MODELS_DIR / MODEL_PATH_NAME
    saved = None if retrain else classement.ColmatageModel.load(path)
    if saved is None:
        log("Apprentissage du reseau...")
        network = classement.train(data["windows"][train_rows], data["labels"][train_rows],
                                   data["windows"][val_rows], data["labels"][val_rows], epochs=epochs, log=log)
        temperature = classement.calibrate(network, data["windows"][val_rows], data["labels"][val_rows])
    else:
        log(f"Reseau deja appris ({path.name}) : reglage et mesure seulement.")
        network, temperature = saved.network, saved.temperature
    log(f"Temperatures (gel, vide) : {temperature[0]:.2f}, {temperature[1]:.2f}")
    p_network = classement.predict(network, data["windows"], temperature)

    log("Modele par descripteurs (le precedent), reappris sur les memes lignes...")
    p_features, features_model = baseline(annotations, split, data, source, pixel_um, log)

    # Candidats : reseau seul, descripteurs seuls, melanges. Le choix et le cout de
    # changement se font sur la VALIDATION ; le test n'est que mesure.
    candidates = {1.0: p_network}
    if p_features is not None:
        candidates.update({0.0: p_features, 0.3: 0.3 * p_network + 0.7 * p_features,
                           0.5: 0.5 * p_network + 0.5 * p_features, 0.7: 0.7 * p_network + 0.3 * p_features})
    choices = {}
    for blend, p in candidates.items():
        cost = tune_change_cost(data, p, lambda *a: None)
        v = evaluate(data, p, "validation", cost)
        # Une erreur d'un point sur la part colmatee pese comme un point de longueur mal classee.
        choices[blend] = (v["longueur_bien_classee"] - abs(v["biais_points"]) / 100, cost, v)
        log(f"  reseau {blend:.0%} : cout {cost:g}, validation " + describe("", v))
    blend = max(choices, key=lambda b: choices[b][0])
    cost = choices[blend][1]
    log(f"Retenu : reseau {blend:.0%} / descripteurs {1 - blend:.0%}, cout de changement {cost:g}")

    results = {"retenu": evaluate(data, candidates[blend], "test", cost),
               "reseau + lissage": evaluate(data, p_network, "test", choices[1.0][1])}
    if p_features is not None:
        def ten_microns(centers, p, total):
            return portions_from_windows(list(centers), p, total, 10.0)
        results["precedent, portions 10 um"] = evaluate(data, p_features, "test", 0.0, splitter=ten_microns)
        results["descripteurs + lissage"] = evaluate(data, p_features, "test", choices[0.0][1])

    log("\nTUILES DE TEST (jamais vues) :")
    for name, summary in results.items():
        log("  " + describe(name, summary))

    report = {"entraine_le": datetime.now().isoformat(timespec="seconds"), "image": annotations.image_id,
              "pixel_um": pixel_um,
              "fenetres": {"apprentissage": int(len(train_rows)), "validation": int(len(val_rows))},
              "temperature": temperature, "cout_changement": cost, "melange_reseau": blend, "test": results,
              "validation": {f"reseau {b:.0%}": c[2] for b, c in choices.items()},
              "decoupage": {k: split[k] for k in ("lignes_test", "lignes_validation")}}
    model = classement.ColmatageModel(network, temperature, cost, report=report,
                                      features_model=features_model if blend < 1.0 else None, blend=blend)

    # Garde-fou : le nouveau modele ne remplace celui en service que s'il fait
    # mieux que lui sur les memes tuiles de test ; sinon il reste a cote.
    installed = classement.ColmatageModel.load(path)
    new_score = _score(results["retenu"])
    if installed is None:
        decision, reason = True, "aucun modele en service"
    elif not any(chain["role"] == "test" for chain in data["chains"]) or not results["retenu"]["longueur_mm"]:
        decision, reason = False, "pas de tuile de test validee : impossible de comparer"
    else:
        old = evaluate(data, _probabilities_of(installed, data, annotations, source, pixel_um), "test",
                       installed.change_cost)
        results["modele en service"] = old
        log("  " + describe("modele en service", old))
        old_score = _score(old)
        decision = new_score > old_score
        reason = (f"score test {100 * new_score:.1f} contre {100 * old_score:.1f} pour le modele en service "
                  "(longueur bien classee moins l'ecart sur la part colmatee)")
    report["installe"] = decision
    report["decision"] = reason
    if decision:
        if installed is not None:
            # Les versions remplacees, a part : `modeles/` ne garde que ce qui sert.
            archive = path.parent / "anciens" / f"{path.stem}_{datetime.now():%Y%m%d-%H%M}{path.suffix}"
            archive.parent.mkdir(exist_ok=True)
            import shutil
            shutil.copy2(path, archive)
            log(f"Ancien modele garde sous anciens/{archive.name}")
        model.save(path)
        log(f"Nouveau modele INSTALLE ({reason}) : {path}")
    else:
        candidate = path.with_name(f"{path.stem}_candidat{path.suffix}")
        model.save(candidate)
        log(f"Modele en service CONSERVE ({reason}). Le nouveau est garde a cote : {candidate.name}")
    out = annotations.output_dir() / "colmatage_rapport.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Rapport : {out}")
    return report


def _score(summary: dict) -> float:
    """Longueur bien classee, moins l'ecart (en fraction) sur la part colmatee."""
    return summary["longueur_bien_classee"] - abs(summary["biais_points"]) / 100


def _probabilities_of(model, data, annotations, source, pixel_um) -> np.ndarray:
    """Probabilites d'un modele enregistre (reseau, et descripteurs s'il en a) aux fenetres de test."""
    probabilities = model.probabilities(data["windows"])
    if model.blend < 1.0 and model.features_model is not None:
        chains = donnees.chains(annotations.validated)
        for index, rows in _by_chain(data, "test"):
            features = classement.feature_probabilities(model.features_model, source.read, source.width,
                                                        source.height, chains[index]["geometry"], pixel_um,
                                                        data["centers"][rows])
            probabilities[rows] = model.blend * probabilities[rows] + (1 - model.blend) * features
    return probabilities
