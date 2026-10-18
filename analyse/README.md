# Analyse automatique

Mesure une mosaïque sans annotation : longueur de fissures, état du colmatage
tous les 10 µm le long de chaque fissure, part colmatée avec son incertitude.
Vos annotations servent une fois, à apprendre et à **mesurer** la méthode ;
elles ne sont jamais modifiées. Les résultats vont dans
`annotation_app/data/analyse/<image>/`.

Dans l'application, un seul bouton : **Analyser la mosaïque…**. Le résultat
arrive en propositions modifiables (pointillés) dans les tuiles non validées ;
on corrige ensuite de tuile en tuile avec les outils d'annotation, et
« Valider la tuile » confirme ce qui reste. Le rapport décrit l'état actuel
(analyse + corrections). En ligne de commande, l'import se fait avec
`python -m analyse analyser --importer`.

L'analyse part de l'image seule : une mosaïque **sans aucune annotation**
s'analyse de la même façon. Son rapport reprend alors la mesure de la
méthode (et la correction) faite sur les tuiles de test de la mosaïque
d'apprentissage, jointe aux modèles (`annotation_app/modeles/analyse_validation.json`).

## Les étapes

| Étape | Ce qui se passe |
| --- | --- |
| Masque | L'échantillon est reconnu par sa forme (le corps du monolithe) ; la résine et ce qui n'y est rattaché que par une attache fine sont dehors. Près du contour (100 µm), les traits qui le **longent** sont écartés ; une fissure qui le traverse reste. |
| Détection | Le réseau de segmentation, à 1,6 µm par pixel, tuile par tuile ; reprise automatique après une interruption. |
| Nettoyage | Deux bouts qui se font face à travers un bord de tuile sont reliés ; les morceaux isolés de moins de 50 µm sont retirés. |
| Colmatage | Une fenêtre redressée de la fissure tous les 10 µm (24 µm le long, ±44 µm en travers, à 0,75 µm par pixel). Un petit réseau répond à deux questions : *gel visible ?* et *fissure vide ?* — pas de gel : non colmatée ; gel et vide : rouverte ; gel sans vide : colmatée. |
| Lissage | Le long de chaque fissure, un changement d'état coûte un prix réglé sur les fissures de validation : un tronçon court mais net survit, un va-et-vient sur des indices faibles disparaît. Aucune longueur minimale. |
| Mesures | Longueurs par état, par tuile et au total ; part colmatée = (colmatée + rouverte) / total. |

## Ce qui garantit les chiffres

- **Tuiles de test** : des lignes entières de la grille (`decoupage.json`),
  jamais apprises. Chaque modèle y est mesuré contre vos annotations :
  détection (précision, rappel), longueur bien classée, biais sur la part
  colmatée, changements d'état retrouvés ou inventés.
- **Correction des erreurs de détection** : sur les tuiles de test, la
  détection rate davantage de fissures colmatées (moins contrastées qu'un
  vide) et ajoute des traits classés non colmatés. Pour chaque état, le
  rapport longueur annotée / longueur automatique y est mesuré, puis appliqué
  à toute la mosaïque (« part colmatée corrigée »). Cette correction est
  contrôlée sur les lignes de validation, qu'elle n'a pas servi à mesurer.
- **Sondage** : 200 points tirés au hasard le long des fissures détectées ;
  vous dites ce que vous voyez, sans voir la réponse de la machine. La part
  colmatée est corrigée de l'écart moyen machine / vous, avec un intervalle
  de confiance à 95 %, et la part de « pas une fissure » est estimée. Une
  dizaine de minutes ; à refaire pour chaque nouvelle mosaïque.

## En ligne de commande

Depuis le dossier du projet, avec le Python du projet :

```powershell
python -m analyse preparer              # découpage apprentissage / test, masque, image de contrôle
python -m analyse colmatage             # apprend le colmatage (~1 h), se mesure sur le test
python -m analyse analyser              # toute la mosaïque (1 à 2 h, reprend où il s'était arrêté)
python -m analyse analyser --test-seulement   # seulement les tuiles de test : mesurer vite une modification
python -m analyse evaluer               # le dernier résultat contre vos annotations, tuiles de test
python -m analyse rapport               # refait le rapport PDF (avec le sondage)
```

Le détecteur se réentraîne avec le même découpage :
`python outils/train_crack_unet.py --image "…" --decoupage --negatifs` — les
traits que vous avez rejetés hors des tuiles validées y sont appris comme
« pas une fissure », les pointillés jamais vérifiés ne comptent pas.

## Fichiers

| Fichier | Rôle |
| --- | --- |
| `donnees.py` | annotations en lecture seule, découpage, fissures rechaînées |
| `masque.py` | contour de l'échantillon, exclusion des bords |
| `reseau.py` | nettoyage des tracés détectés |
| `bandes.py` | fenêtres redressées le long des fissures |
| `classement.py` | réseau de colmatage, lissage le long de la fissure |
| `apprentissage.py` | apprentissage et mesure du colmatage |
| `evaluation.py` | mesures contre l'annotateur |
| `chaine.py` | la mosaïque entière |
| `audit.py` | sondage et correction |
| `rapport.py` | rapport PDF |
