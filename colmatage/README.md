# Reconnaissance du colmatage

Objectif : dire, le long de chaque fissure tracée, si elle **a été colmatée**,
d'après le **gel d'altération**, et non d'après son aspect actuel.

## Le critère

Une fissure vide n'est pas forcément une fissure jamais colmatée : la
préparation (résine, polissage) peut la rouvrir. Le gel, lui, reste. Il prend
deux formes :

| | ce qu'on voit | catégorie |
| --- | --- | --- |
| remplissage | matériau gris intermédiaire **dans** la fissure | **Colmatée** (`é` / 2) |
| liseré | bande granuleuse, plus sombre que la matrice, **le long des parois** d'une fissure vide | **Rouverte** (`'` / 4) |
| rien | parois nettes, matrice homogène jusqu'au bord | **Non colmatée** (`&` / 1) |

« Colmatée » et « Rouverte » comptent toutes deux dans la part colmatée.
« Incertaine » (`"` / 3) quand on ne peut pas trancher : ces tracés sont
écartés de l'apprentissage.

Le modèle n'a aucune règle écrite sur le gel : il apprend ce critère à partir
de vos fissures **validées**. Une fissure dont l'état change en chemin se coupe
en morceaux, classés chacun, plutôt que d'être classée « Mixte ».

## Ce que contient ce dossier

| | |
| --- | --- |
| `profils.py` | profils en travers de la fissure tous les 10 µm : parois, contenu, bandes le long des parois |
| `modele.py` | classement par ces descripteurs : suggestions de l'application, et 30 % du mélange utilisé par l'analyse (`analyse/classement.py`) |

Le réseau principal et son apprentissage sont dans [`analyse/`](../analyse/README.md)
(`python -m analyse colmatage`, ou *Automatique > Réentraîner le classement du colmatage* dans
l'application).

## Résolution

Le modèle actuel a appris sur des images à 0,37 µm par pixel. Sur une image
plus grossière (1,6 µm), le liseré de gel ne fait plus qu'un ou deux pixels :
le rapport d'analyse le signale et ses parts colmatées ne sont pas fiables.
Acquérir les mosaïques à 0,37 µm ou plus fin, avec les mêmes réglages du MEB.
