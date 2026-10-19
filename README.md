# Identification des fissures

Application pour repérer les fissures sur des mosaïques MEB, dire si elles sont
colmatées, et mesurer leurs longueurs.

## Installer et lancer

1. Avoir Python 3.10 à 3.13 (64 bits) sur le poste.
2. Double-cliquer sur **`Lancer.cmd`**, ou sur **`Choisir une image.cmd`** pour
   ouvrir une autre mosaïque.

Le premier lancement installe tout seul ce qu'il faut (quelques minutes, une
seule fois). Le terminal resté ouvert affiche les messages de l'app. 

## Utiliser

1. **Ouvrir la mosaïque** (*Fichier > Ouvrir une mosaïque…*).
2. **Analyser** : *Automatique > Analyser la mosaïque…*. Compter environ
   45 minutes. Les fissures trouvées s'affichent en pointillés.
3. **Corriger**, tuile par tuile (touche Page suivante) : retracer, reclasser,
   supprimer. Puis **Valider la tuile entière** (Ctrl+Entrée).
4. **Lire le rapport** : *Automatique > Ouvrir le rapport*. La première page
   donne la part colmatée. 
5. **Réentraîner** quand une centaine de tuiles sont validées :
   *Automatique > Réentraîner le classement du colmatage*. Le nouveau modèle
   n'est gardé que s'il fait mieux que l'ancien, pareil pour le modèle de reconnaissance de fissures. 

Raccourcis clavier : **F1** dans l'application. Mode d'emploi détaillé :
[`annotation_app/README.md`](annotation_app/README.md).

## Classer une fissure

On juge d'après le **gel d'altération** autour des fissures qu'elles soient ouvertes ou colmatées. 

| Visible | Catégorie |
| --- | --- |
| pas de gel | **non colmatée** |
| du gel le long des bords, mais la fissure est vide | **rouverte** |
| du gel et un remplissage | **colmatée** |

La part colmatée compte « colmatée » et « rouverte ».

Une fissure qui change d'état en chemin : la couper et classer chaque morceau.
Seules les fissures validées servent à l'apprentissage : vérifier les
catégories avant de valider une tuile.

## Pour un résultat fiable

- Acquérir les mosaïques à **0,37 µm par pixel ou plus fin**, avec les mêmes
  réglages du MEB. Sur une image plus grossière, le modèle de colmatage ne sera pas fiable. 
- Sur chaque nouvelle mosaïque, **valider entièrement quelques tuiles de
  test**. 

**Transférer les informations d'une mosaïque** :

1. ici : *Fichier > Exporter la base de cette mosaïque…* ;
2. sur l'autre poste : copier la mosaïque, l'ouvrir, puis
   *Fichier > Importer une base…* et accepter de remplacer.

## Où est mon travail

| `annotation_app\data\` | |
| --- | --- |
| `bases\` | **contient les infos de fissures et de colmatages**, un fichier par mosaïque |
| `sauvegardes\` | une copie à chaque ouverture et toutes les 4 heures |
| `analyse\` | résultats et rapports PDF des analyses |

Restaurer une copie : *Fichier > Importer une base…* sur le fichier choisi
dans `sauvegardes\`.

Pour sauvegarder sur un autre disque dans
PowerShell :

```powershell
[Environment]::SetEnvironmentVariable("CRACK_BACKUP_DIR", "H:\Sauvegardes fissures", "User")
```

## Contenu du dossier

| | |
| --- | --- |
| `Lancer.cmd`, `Choisir une image.cmd` | lancer l'application |
| `Entrainer le detecteur.cmd` | réentraîner le repérage des fissures |
| `annotation_app/` | l'application ; `modeles/` : les modèles en service |
| `analyse/` | l'analyse complète d'une mosaïque et son rapport |
| `colmatage/` | le critère du colmatage |
| `outils/` | assemblage des tuiles du MEB, entraînement et mesure du détecteur |
| `lanceurs/`, `tests/` | scripts de lancement, tests automatiques |
| `docs/` | [`TECHNIQUE.md`](docs/TECHNIQUE.md) : tous les détails |
