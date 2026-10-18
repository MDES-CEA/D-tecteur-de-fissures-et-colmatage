# Référence technique

Le [README](../README.md) suffit pour utiliser l'application. Ce document
rassemble les détails : assemblage des tuiles, résolutions, identification
des mosaïques, sauvegardes, commandes, dépendances.

## Fichiers à ne pas modifier

`crack_length_analysis_multi.py` (carte de coût, routage, mesure) et
`crop_roi.py` (lecture des mosaïques géantes) sont les fichiers de référence.
L'application les importe sous ce nom, d'où leur place à la racine.

## Lancement

- **VS Code** : *Fichier > Ouvrir le dossier…* sur `identification fissure`
  lui-même, puis **Ctrl+Maj+B**. Les autres actions sont dans
  *Terminal > Exécuter la tâche…* ;
- **terminal** : `.\lanceurs\lancer.ps1`, avec `-Choisir`,
  `-Image "D:\MEB\mosaique.tif"`, `-ImageId` ou `-Database`.

Premier lancement : l'environnement Python est créé dans `.venv\` avec
exactement les versions de `constraints_annotation_app.txt` (environ 1 Go).
Derrière un proxy, définir `$env:HTTPS_PROXY` avant de relancer.

## Assembler une mosaïque à partir des tuiles du MEB

Quand le logiciel du microscope ne peut pas rabouter lui-même (mosaïque trop
grosse), `outils/assembler_mosaique.py` le fait à partir du dossier des tuiles
`<préfixe>_<ligne>_<colonne>.tif` :

```powershell
& $py .\outils\assembler_mosaique.py "D:\MEB\echantillon" --apercu    # contrôle rapide (PNG réduit)
& $py .\outils\assembler_mosaique.py "D:\MEB\echantillon"             # mosaïque pleine résolution
& $py .\outils\assembler_mosaique.py "D:\MEB\echantillon" --zone 28000 110000 6000 4000   # vérifier un raccord
```

Chaque paire de tuiles voisines est recalée dans son recouvrement (9 % par
défaut, lu dans les métadonnées Zeiss), puis les positions sont ajustées
ensemble par moindres carrés. Pas de fondu : chaque pixel vient d'une seule
tuile, ce qui évite de dédoubler les fissures fines. Le résultat est un BigTIFF
non compressé à une bande, avec ses positions (`_positions.csv`) et un rapport
de recalage (`_rapport.json`). Il porte sa taille de pixel ; l'application
s'en sert.

La mosaïque doit être un TIFF **non compressé à une seule bande** : c'est ce
qui permet de la lire par morceaux sans jamais la charger entière.

## Images de résolutions différentes

Les tuiles d'annotation ont toujours la même surface : 3 mm de côté (8594 px à
0,371 µm, 2000 px à 1,595 µm). La détection voit l'image à une résolution
fixe, enregistrée avec le détecteur (1,6 µm aujourd'hui). Les tracés sont
toujours reportés en pleine résolution.

Le classement du colmatage regarde le bord des fissures à 0,75 µm par pixel.
Le modèle actuel a appris sur des images à 0,37 µm : sur une image plus
grossière, le rapport signale que les parts colmatées ne sont pas fiables.

## Identification des mosaïques

Chaque mosaïque a sa base : `annotation_app\data\bases\<identifiant>.sqlite3`.
L'identifiant est déduit de l'image, en cherchant dans toutes les bases :

1. le chemin est déjà enregistré : on reprend sa base ;
2. le fichier a été déplacé ou renommé : une seule image enregistrée a disparu
   de son ancien emplacement et a exactement les mêmes dimensions — on la
   reprend, et le terminal l'annonce ;
3. sinon, nouvelle base, identifiant tiré du nom de fichier, avec un suffixe
   (`-2`, `-3`…) s'il est déjà pris. Deux `RECALAGE.tif` de deux échantillons
   ne partagent donc jamais leurs tracés.

Une base illisible est signalée dans le terminal et laissée telle quelle.

À l'import d'une base reçue, une base locale est remplacée (après
confirmation, l'ancienne part dans les sauvegardes) si elle a le même
identifiant, le même chemin d'image, ou une image de mêmes dimensions présente
sur ce poste alors que celle de la base reçue n'y est pas : la même mosaïque
copiée, éventuellement renommée.

`-ImageId` (lanceur) force l'identifiant ; il est refusé s'il est déjà celui
d'une autre image. `-Database` force une base unique pour toutes les mosaïques
ouvertes : c'est la façon de repartir de zéro, ou d'ouvrir une ancienne copie
sans rien remplacer.

Les outils en ligne de commande (`python -m analyse`, `outils\…`) prennent la
base de `--image`, sans rien créer ; une mosaïque jamais ouverte dans
l'application est refusée. `--database` donne une autre base.

Variables d'environnement : `CRACK_IMAGE_PATH`, `CRACK_IMAGE_ID`,
`CRACK_DATABASE_PATH`, `CRACK_BACKUP_DIR`, `CRACK_CACHE_DIR`,
`CRACK_ANALYSE_DIR`, `CRACK_COLMATAGE_MODEL`, `CRACK_PIXEL_UM` (forcer la
taille de pixel), `CRACK_REFERENCE_IMAGE`.

## Mosaïque de référence

La mosaïque dont les tuiles validées servent de vérité terrain est désignée
sur chaque poste, hors du dépôt : son chemin, sur une ligne, dans
`annotation_app\data\image_reference.txt` (ou `CRACK_REFERENCE_IMAGE`). C'est
l'image par défaut du banc, des entraînements et de certains tests (sans elle,
ils sont sautés).

## Sauvegardes

Une copie de la base à chaque ouverture, avant toute écriture, puis toutes les
4 heures, dans `annotation_app\data\sauvegardes\`
(`<identifiant>-AAAAMMJJ-HHMMSS.sqlite3`). Une copie identique à la précédente
n'est pas gardée ; celles de plus de 14 jours sont retirées, sauf les 10 plus
récentes de chaque base. À la fermeture, la base se suffit à elle-même (pas de
`-wal` à joindre).

Restaurer à la main : fermer l'application, remplacer
`bases\<identifiant>.sqlite3` par la copie renommée (supprimer
`-wal`/`-shm` s'ils existent), relancer.

**Ancienne base unique.** Avant le 17/10/2026, toutes les mosaïques étaient
dans `data\annotations.sqlite3`. Au premier lancement, elle est séparée en une
base par mosaïque (mêmes identifiants, vérifiée table par table), tout ou
rien. L'original est rangé dans
`sauvegardes\ancienne-base-unique\` avec les anciennes copies
`annotations-AAAAMMJJ-HHMMSS.sqlite3`, jamais retirées. *Fichier > Importer
une base…* sur l'une d'elles la sépare de même.

Ne pas poser les bases sur un lecteur réseau partagé par plusieurs postes : le
verrouillage de SQLite n'y est pas fiable.

## Ce qui est écrit sur le disque

| `annotation_app\data\` | quoi | se recalcule ? |
| --- | --- | --- |
| `bases\` | le travail, une base par mosaïque | non |
| `sauvegardes\` | copies des bases | non |
| `analyse\<mosaïque>\` | résultat, rapport PDF, `segments.csv`, `decoupage.json` (lignes de test et de validation) | non |
| même dossier | détections brutes, masque, caches d'apprentissage | oui |
| `analysis_cache\` | calques de diagnostic, environ 35 Mo par tuile à 0,37 µm | oui |
| `entrainements\` | journaux des analyses et apprentissages | — |

À chaque ouverture d'une mosaïque, `annotation_app/menage.py` retire ce qui
se recalcule et ne sera plus relu : calques des mosaïques disparues puis les
plus anciens au-delà de 2 Go, détections d'un ancien détecteur, `-wal`/`-shm`
vides à côté des sauvegardes. `python -m annotation_app.menage` affiche la
place prise par dossier. Les modèles remplacés par un réentraînement vont
dans `annotation_app\modeles\anciens\`.

## Commandes utiles

```powershell
$py = ".\.venv\Scripts\python.exe"

# Analyse complète en ligne de commande (comme le menu Automatique)
& $py -m analyse --image "D:\MEB\mosaique.tif" analyser --importer

# Mesurer la détection sur les tuiles validées (précision, rappel)
& $py .\outils\evaluate_crack_detection.py
& $py .\outils\evaluate_crack_detection.py --limit 8      # essai rapide

# Réentraîner le détecteur (le plus simple : « Entrainer le detecteur.cmd »)
& $py .\outils\train_crack_unet.py --image "D:\MEB\mosaique.tif" --installer

# Tests
& $py -m unittest discover
```

Aucun réglage de la détection ne devrait être retenu sans une mesure avant et
après sur des tuiles qui n'ont pas servi à l'apprentissage :

```powershell
& $py .\outils\train_crack_unet.py --exclude-zones 6 11 14 19 22 --output candidat.pt
& $py .\outils\evaluate_crack_detection.py --model candidat.pt --zones 6 11 14 19 22
```

## Dépendances

Installées par le lanceur : `requirements_annotation_app.txt` dit quoi,
`constraints_annotation_app.txt` fixe les versions exactes (un autre numpy ou
torch peut déplacer une détection de quelques pixels). Modifier l'un des deux
déclenche une mise à jour au lancement suivant. Interface : PySide6 ; réseau
de segmentation : torch (version CPU).
