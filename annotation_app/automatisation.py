"""Boucle d'annotation assistee : entrainer, traiter, verifier.

1. **Entrainer** : le modele de colmatage apprend sur les fissures validees de
   l'image (categories non colmatee, colmatee, rouverte), apres une mesure de
   sa justesse par validation croisee, affichee a l'annotateur ;
2. **Traiter** des tuiles : celles qui n'ont aucun travail recoivent les
   propositions du detecteur, et chaque trace non classe recoit une
   SUGGESTION de categorie. Rien n'est jamais ecrase : une suggestion n'est
   pas une categorie, le trace reste « propose » ;
3. **Verifier** : l'annotateur passe d'une proposition a l'autre et valide la
   suggestion, la corrige ou rejette le trace. Ce qu'il valide devient une
   etiquette pour le prochain entrainement.

Les taches longues tournent en arriere-plan, une a la fois, et s'arretent sur
demande entre deux tuiles.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from threading import Event, Lock, Thread


class Automation:
    def __init__(self, database, image_id: str, source, pixel_um: float | None, detect,
                 model_path=None, network_path=None) -> None:
        from colmatage import modele

        self.modele = modele
        self.database = database
        self.image_id = image_id
        self.source = source            # .read(x, y, w, h), .width, .height
        self.pixel_um = pixel_um
        self.detect = detect            # zone -> propositions du detecteur
        self.model_path = Path(model_path) if model_path else modele.MODEL_PATH
        self.model = modele.load(self.model_path) if pixel_um else None
        # Le reseau de l'analyse automatique (python -m analyse colmatage), s'il
        # existe, remplace le modele par descripteurs. Un modele donne
        # explicitement (tests, base de travail separee) garde la main.
        if network_path is None and model_path is None:
            from .paths import MODELS_DIR
            network_path = os.environ.get("CRACK_COLMATAGE_RESEAU") or MODELS_DIR / "colmatage_reseau.pt"
        self.network_path = Path(network_path) if network_path else None
        self.network = None
        self.reload_network()
        self.lock = Lock()
        self.stop = Event()
        self.status = {"task": None, "running": False, "done": 0, "total": 0,
                       "started": None, "message": "", "error": None, "result": None}

    def reload_network(self) -> None:
        self.network = None
        if self.network_path is None or not self.pixel_um or not self.network_path.exists():
            return
        try:
            from analyse.classement import ColmatageModel
            self.network = ColmatageModel.load(self.network_path)
        except Exception as error:  # un reseau illisible ne doit pas empecher d'annoter
            print(f"Reseau de colmatage illisible ({self.network_path.name}) : {error}", flush=True)

    @property
    def has_model(self) -> bool:
        return self.network is not None or self.model is not None

    # -- Etat ---------------------------------------------------------------

    def describe(self) -> dict:
        report = (self.model or {}).get("report") or {}
        model = None
        if self.network is not None:
            tests = self.network.report.get("test") or {}
            test = tests.get("retenu") or tests.get("reseau + lissage") or {}
            model = {"kind": "reseau", "trained_at": self.network.report.get("entraine_le"),
                     "length_correct": test.get("longueur_bien_classee"),
                     "sealed_bias": test.get("biais_points")}
        elif self.model:
            classifier = report.get("classifieur", {})
            model = {"trained_at": report.get("entraine_le"), "traces": report.get("traces"),
                     "correct": classifier.get("traces_justes"),
                     # par fenetre de 10 um : la part de longueur bien classee
                     "length_correct": classifier.get("fenetres_justes"),
                     "sealed_length": classifier.get("colmate_oui_non_longueur"),
                     "rule_correct": report.get("regle_simple", {}).get("traces_justes")}
        with self.lock:
            status = dict(self.status)
        return {**status, "model": model, "pixel_um": self.pixel_um,
                "to_review": len(self.database.review_queue(self.image_id))}

    def _begin(self, task: str, total: int, message: str) -> None:
        with self.lock:
            if self.status["running"]:
                raise RuntimeError("Un calcul automatique est déjà en cours.")
            self.stop.clear()
            self.status = {"task": task, "running": True, "done": 0, "total": total,
                           "started": time.time(), "message": message, "error": None, "result": None}

    def _progress(self, done: int, total: int | None = None, message: str | None = None) -> None:
        with self.lock:
            self.status["done"] = done
            if total is not None:
                self.status["total"] = total
            if message is not None:
                self.status["message"] = message

    def _end(self, message: str, result=None, error: str | None = None) -> None:
        with self.lock:
            self.status.update(running=False, message=message, result=result, error=error)

    def cancel(self) -> None:
        self.stop.set()

    def next_review(self, after: int | None = None, zone: int | None = None) -> tuple[dict | None, int]:
        """Prochaine proposition a verifier, apres le trace `after` de la tuile `zone`.

        L'ordre est celui de la grille. Le trace precedent a souvent quitte la
        file (il vient d'etre valide) : on se repere donc a sa place dans la
        grille, pas a sa position dans la file. Rend (trace, nombre restant).
        """
        queue = self.database.review_queue(self.image_id)
        if not queue:
            return None, 0
        order = {z["id"]: (z["row_index"], z["col_index"]) for z in self.database.list_zones(self.image_id)}
        key = lambda zone_id, crack_id: (*order.get(zone_id, (0, 0)), crack_id)  # noqa: E731
        chosen = queue[0]
        if after is not None and zone is not None:
            last = key(zone, after)
            chosen = next((c for c in queue if key(c["zone_id"], c["id"]) > last), queue[0])
        return chosen, len(queue)

    # -- Suggestions ----------------------------------------------------------

    def _network_portions(self, geometry) -> list[dict]:
        """Troncons du reseau (etat tous les 10 um, lisse le long de la fissure)."""
        portions = self.network.classify_line(self.source.read, self.source.width, self.source.height,
                                              geometry, self.pixel_um)
        for portion in portions:
            portion["probabilities"] = {portion["category"]: round(portion["confidence"], 3)}
        return portions

    def suggest(self, crack: dict) -> dict | None:
        """Calcule et enregistre la suggestion d'un trace ; None sans modele."""
        if not self.has_model or not self.pixel_um or crack.get("state") != "proposed":
            return None
        if self.network is not None:
            portions = self._network_portions(crack["geometry"])
            if not portions:
                return None
            # Un trace a la main n'est jamais coupe : la categorie la plus longue.
            lengths = {}
            for p in portions:
                lengths[p["category"]] = lengths.get(p["category"], 0.0) + p["end_um"] - p["start_um"]
            category = max(lengths, key=lengths.get)
            total = sum(lengths.values()) or 1.0
            prediction = {"category": category, "confidence": lengths[category] / total,
                          "probabilities": {c: round(v / total, 3) for c, v in lengths.items()}}
        else:
            prediction = self.modele.predict(self.model, self.source.read, self.source.width,
                                             self.source.height, crack["geometry"], self.pixel_um)
        if prediction is None:
            return None
        self.database.set_suggestion(crack["id"], crack["revision"], prediction["category"],
                                     prediction["confidence"], prediction["probabilities"])
        return prediction

    def suggest_portions(self, crack: dict) -> int:
        """Proposition du detecteur : coupee en portions de categorie homogene.

        Chaque portion devient un trace a part, avec sa suggestion : une
        fissure colmatee sur une partie et ouverte sur l'autre n'est plus
        resumee par une seule categorie. Rend le nombre de suggestions posees.
        """
        if not self.has_model or not self.pixel_um or crack.get("state") != "proposed":
            return 0
        from .polyligne import pieces
        if self.network is not None:
            portions = self._network_portions(crack["geometry"])
        else:
            portions = self.modele.predict_portions(self.model, self.source.read, self.source.width,
                                                    self.source.height, crack["geometry"], self.pixel_um)
        if not portions:
            return 0
        saved = [crack]
        if len(portions) > 1:
            cuts = [p["start_um"] / self.pixel_um for p in portions[1:]]
            saved = self.database.split_crack(self.image_id, crack["id"], crack["revision"],
                                              pieces(crack["anchors"], crack["geometry"], cuts), "automatique")
        for piece, portion in zip(saved, portions):
            self.database.set_suggestion(piece["id"], piece["revision"], portion["category"],
                                         portion["confidence"], portion["probabilities"])
        return len(saved)

    def suggest_auto(self, crack: dict) -> int:
        """Suggestion(s) d'un trace : par portions pour le detecteur, entiere pour un trace a la main."""
        if crack.get("source") == "auto":
            return self.suggest_portions(crack)
        return int(self.suggest(crack) is not None)

    def suggest_zone(self, zone_id: int, refresh: bool = False) -> int:
        """Suggestions des propositions d'une tuile ; `refresh` refait aussi celles qui existent."""
        count = 0
        for crack in self.database.list_cracks(self.image_id, zone_id):
            if crack["state"] == "proposed" and (refresh or crack["suggestion"] is None):
                count += self.suggest_auto(crack)
        return count

    # -- Taches ---------------------------------------------------------------

    def start_training(self) -> None:
        if not self.pixel_um:
            raise RuntimeError("Taille de pixel inconnue.")
        traces = self.database.labelled_cracks(self.image_id, self.modele.CLASSES)
        try:
            self.modele.check_counts(traces)
        except ValueError as error:
            raise RuntimeError(str(error)) from error
        self._begin("train", len(traces), "Lecture des fissures validées…")
        Thread(target=self._train, args=(traces,), daemon=True).start()

    def _train(self, traces) -> None:
        try:
            artifact, report = self.modele.train(
                self.source.read, self.source.width, self.source.height, traces, self.pixel_um,
                progress=lambda done, total: self._progress(done, total),
                cancelled=self.stop.is_set)
            self.modele.save(artifact, self.model_path)
            self.model = artifact
            # Les suggestions deja faites venaient de l'ancien modele.
            zones = {crack["zone_id"] for crack in self.database.list_image_cracks(self.image_id)
                     if crack["state"] == "proposed"}
            self._progress(0, len(zones), "Mise à jour des suggestions…")
            for done, zone_id in enumerate(sorted(zones), start=1):
                self.suggest_zone(zone_id, refresh=True)
                self._progress(done)
            classifier = report["classifieur"]
            self._end(f"Modèle entraîné sur {sum(report['traces'].values())} fissures : "
                      f"{classifier['fenetres_justes']:.0%} de la longueur bien classée.", result=report)
        except InterruptedError:
            self._end("Entraînement arrêté, ancien modèle conservé.")
        except Exception as error:  # l'erreur doit remonter a l'interface
            self._end("Entraînement impossible.", error=str(error))

    def start_processing(self, zones: list[dict]) -> None:
        self._begin("process", len(zones), "Traitement des tuiles…")
        Thread(target=self._process, args=(zones,), daemon=True).start()

    def _process(self, zones) -> None:
        detected = suggested = 0
        try:
            for done, zone in enumerate(zones, start=1):
                if self.stop.is_set():
                    self._end(f"Arrêté après {done - 1} tuile(s) : {detected} fissure(s), {suggested} suggestion(s).")
                    return
                # Une tuile ou l'annotateur a deja travaille, meme seulement
                # pour rejeter des propositions, n'est pas re-detectee : ses
                # faux positifs reviendraient.
                touched = self.database.list_cracks(self.image_id, zone["id"], include_rejected=True)
                if not touched:
                    proposals = self.detect(zone)
                    saved, created = self.database.add_detected_cracks_if_empty(
                        self.image_id, zone["id"], proposals, "automatique")
                    if created:
                        detected += len(saved)
                suggested += self.suggest_zone(zone["id"])
                self._progress(done)
            note = "" if self.has_model else " Pas de modèle de colmatage : aucune suggestion."
            self._end(f"{len(zones)} tuile(s) traitée(s) : {detected} fissure(s), "
                      f"{suggested} suggestion(s).{note}")
        except Exception as error:
            self._end("Traitement interrompu (erreur).", error=str(error))
