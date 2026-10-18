# Mode d'emploi de l'application

Lancer : double-clic sur `../Lancer.cmd` (voir le README du dossier pour le
premier lancement sur un poste). *Aide > Raccourcis clavier* (F1) rappelle
toutes les touches.

## Le travail, en deux temps

1. **Automatique > Analyser la mosaïque…** (ou le bouton du panneau) : la
   machine calcule toute la mosaïque — fissures, bords de l'échantillon exclus,
   état du colmatage tous les 10 µm — puis le rapport. Environ 45 minutes la
   première fois, arrêtable ; relancer reprend où ça s'était arrêté. Les
   fissures trouvées arrivent en **propositions** (pointillés, déjà colorées
   par état) dans les tuiles non validées, à la place des anciennes
   propositions. Les tuiles validées ne sont jamais touchées.
2. **De tuile en tuile** (Page suiv. / Page préc.), corriger ce qui ne va pas
   avec les outils ci-dessous, puis **Valider la tuile** (Ctrl+Entrée) : cela
   confirme les propositions restantes telles quelles (elles passent en trait
   plein). Ctrl+Z annule la validation.

Le rapport (*Automatique > Ouvrir le rapport*) décrit l'état actuel : l'analyse
plus vos corrections, le nombre de tuiles vérifiées, et une estimation où les
tuiles non vérifiées sont corrigées des erreurs de détection mesurées sur les
tuiles de test.

## La fenêtre

- **au centre**, la mosaïque. Molette : zoom ; glisser : déplacer l'image
  (dans tous les modes) ; **Ctrl+0** : tout voir ;
- **à droite**, le panneau : la **tuile** sélectionnée (statut, valider), la
  **fissure** sélectionnée (catégorie, longueur, boutons de classement),
  l'**analyse automatique** (le bouton, le rapport, où en est la mosaïque) et
  votre **nom d'annotateur** ;
- **en bas**, la barre d'état : ce qui vient de se passer, le calcul en cours,
  la position du curseur, le zoom.

## Les menus

| Menu | Contenu |
| --- | --- |
| **Fichier** | ouvrir une autre mosaïque, nom de l'annotateur, exporter la base de cette mosaïque, importer une base, quitter |
| **Édition** | annuler (Ctrl+Z), terminer le tracé (Échap), retirer le dernier point (Z), supprimer la fissure (Suppr) |
| **Outils** | Sélection (V), Tracer (C), Ajouter des nœuds (I), Couper une fissure (X) |
| **Classer** | les catégories, accepter la catégorie suggérée (Entrée) |
| **Tuile** | précédente / suivante (Page préc. / suiv.), centrer (T), valider (Ctrl+Entrée), remettre « non validée », accepter tous les classements (Maj+Entrée), supprimer toutes les fissures (Ctrl+Suppr) |
| **Automatique** | analyser la mosaïque, ouvrir le rapport, sondage ; réentraîner le colmatage ou le détecteur |
| **Affichage** | tout voir, zoom, afficher les fissures (F), la grille (G), luminosité et contraste, calques du détecteur (diagnostic), le panneau |
| **Mesures** | bilan des longueurs (Ctrl+B) et figures (Ctrl+Maj+B), exports CSV, PNG, SVG, PDF |
| **Aide** | raccourcis clavier (F1) |

## Corriger et tracer

1. Cliquer dans une tuile pour la sélectionner (contour bleu).
2. **V — Sélection** : cliquer sur une fissure pour la sélectionner. Ses nœuds
   apparaissent : les **glisser** pour corriger, **clic droit** pour en retirer
   un. **Maj+clic** sur la fissure (ou l'outil *Ajouter des nœuds*, I) ajoute
   un nœud. **Suppr** supprime la fissure.
3. **Changer l'état** avec les touches (chiffres ou AZERTY sans Maj) ou les
   boutons :

   | Touche | Catégorie |
   | --- | --- |
   | 1 ou & | Non colmatée — pas de gel |
   | 2 ou é | Colmatée — gel et remplissage |
   | 4 ou ' | Rouverte — gel sur les parois, remplissage arraché |
   | 3 ou " | Incertaine |
   | 5 ou ( | Mixte |

   Le critère est détaillé dans `../colmatage/README.md`. Une fissure qui
   change d'état le long de son tracé : la **couper** (X, puis clic à l'endroit
   du changement) et classer chaque morceau.
4. **C — Tracer** une fissure oubliée : cliquer le long de la fissure, point
   après point ; le chemin suit l'image entre deux points ; **Échap** termine,
   **Z** retire le dernier point. Tous les points d'une fissure restent dans
   la même tuile. La machine propose son état (Entrée pour l'accepter).
5. **Valider la tuile** (Ctrl+Entrée) quand tout y est juste.

**Ctrl+Z** annule la dernière action (tracé, déplacement, catégorie,
suppression, coupe, validation). **Espace** maintenu masque les fissures pour
voir l'image dessous ; **F** les masque jusqu'à nouvel ordre.

Couleurs : rouge non colmatée, bleu colmatée, vert rouverte, violet mixte,
gris incertaine, orange non classée. **Pointillés** : proposition pas encore
vérifiée ; **trait plein** : vérifiée.

## Sondage, réentraînement

**Automatique > Contrôler l'analyse (sondage)…** : 200 points au hasard sur
les fissures de l'analyse, une touche par point (1 non colmatée, 2 colmatée,
4 rouverte, 0 pas une fissure ; Retour arrière pour revenir), sans voir la
réponse de la machine. Une estimation rapide de sa justesse sur cette
mosaïque, sans tout corriger. Détails : `../analyse/README.md`.

Les tuiles que vous validez servent aussi au prochain réentraînement
(*Automatique > Réentraîner…*) : le **détecteur de fissures** (trouver les
fissures) et le **classement du colmatage** (leur état) sont deux modèles
distincts.

## Calques du détecteur (diagnostic)

Pour comprendre ce que le détecteur voit, et pourquoi il rate ou invente une
fissure : *Affichage > Calques du détecteur*. Le calque de la tuile
sélectionnée se calcule tout seul (quelques secondes).

| Calque | Ce qu'il montre |
| --- | --- |
| Score du détecteur | probabilité de fissure par pixel (jaune = forte) |
| Masque | pixels classés fissure |
| Squelette du masque | ligne centrale du masque |
| Axes automatiques | les axes bruts du détecteur, avant le nettoyage de l'analyse |
| Coût de la détection / du tracé manuel | sombre = passage facile |

*Sur toute la mosaïque* affiche aussi, réduites, les tuiles déjà calculées ;
*Exporter les calques (ZIP)* donne les tableaux bruts (`.npz`).

## Mesures

*Mesures > Bilan des longueurs* (Ctrl+B) ou *Figures* (Ctrl+Maj+B) ouvrent la
même fenêtre, qui peut rester ouverte pendant qu'on annote (**Actualiser**
relit le travail). En haut, les options communes :

- **Périmètre** : toute la mosaïque ou la tuile sélectionnée ;
- **Mesurer** : les fissures de la mosaïque (l'analyse + vos corrections), ou,
  pour un diagnostic, les axes bruts du détecteur ;
- **Seulement les tuiles validées** ; **Superposition** : la tolérance.

Onglet **Bilan** : longueur par catégorie, et la **part colmatée** (colmatée +
rouverte, parmi les fissures classées non colmatée / colmatée / rouverte).
Export CSV des segments (séparateur `;`, s'ouvre dans Excel).

Onglet **Figures** : longueurs par catégorie, orientation (par tranches de
10°), carte des axes colorés par catégorie, longueur par tuile. Les cartes ont
l'image en fond (décochable). Axes et longueurs en mm. *Enregistrer la
figure* : PNG (200 dpi), SVG ou PDF, tels qu'affichés.

La longueur est euclidienne, le long de l'axe tracé, convertie en mm avec la
taille de pixel de la mosaïque. Ce n'est ni une aire de masque ni un nombre de
pixels de squelette. Les tracés rejetés sont exclus. Les portions
superposées, parallèles à cinq degrés près et distantes de moins de la
**tolérance** (2 px par défaut), ne comptent qu'une fois ; une portion portant
deux catégories différentes devient **Conflit**.

### Limite du dédoublonnage

**Le dédoublonnage ne résiste pas à un redessin à main levée.** Deux tracés qui
suivent la même fissure en oscillant autour l'un de l'autre ne sont fusionnés
que là où ils restent dans la tolérance. Mesuré en redessinant une fissure
réelle de 3 590 px :

| écart latéral du second tracé | longueur totale | erreur |
| --- | --- | --- |
| 0 px, copie exacte | 3 590 | 0 % |
| σ = 1 px | 4 213 | +17 % |
| σ = 1,5 px | 5 014 | **+40 %** |
| décalage uniforme de 2,1 px | 4 321 | +20 % |

Monter la tolérance ne sauve pas le cas (à 5 px, encore +30 %, et l'on
commence à fusionner des fissures parallèles distinctes).
**Ne pas retracer une fissure déjà tracée** : corriger ses nœuds, ou la
supprimer avant d'en refaire une.

Il n'y a volontairement **pas de compteur de fissures** : compter suppose de
décider ce qui relie deux tracés (jonction, croisement, proximité), ce qui
n'est pas vérifiable à l'œil. Un chiffre invérifiable est pire que pas de
chiffre.

## Conservation

Chaque mosaïque a sa base, `data/bases/<identifiant>.sqlite3`. Chaque
modification y est enregistrée aussitôt, avec son historique ; les révisions
empêchent d'écraser sans le savoir une modification faite ailleurs. La base
est sauvegardée à l'ouverture puis toutes les 4 heures (voir le README du
dossier). Les fichiers de l'application :

| | |
| --- | --- |
| `bureau/` | l'interface : `fenetre.py` (menus, panneau, tracé), `vue.py` (affichage par tuiles), `calques.py` (couleurs des calques), `dialogues.py` (mesures, réglages, réentraînement) |
| `session.py` | une mosaïque ouverte : base, image, détecteur, automatisation — toute la logique, sans interface |
| `database.py` | la base SQLite |
| `bases.py` | une base par mosaïque : la retrouver, séparer l'ancienne base unique, exporter, importer |
| `tile_service.py` | lecture de la mosaïque par tuiles et niveaux de zoom |
| `echelle.py` | taille de pixel, tuiles de 3 mm, détecteur à la bonne résolution |
| `detection_service.py`, `unet_model.py`, `detector_model.py` | le détecteur de fissures et le routage des tracés |
| `analysis_service.py` | cache des analyses déjà calculées (les calques) |
| `figures.py` | les figures du bilan (matplotlib, sans interface) |
| `automatisation.py` | entraîner, traiter, vérifier |
| `length_analysis.py`, `overlap_measure.py` | mesure des longueurs |
| `backup.py`, `paths.py` | sauvegardes, emplacements |
