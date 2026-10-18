"""Emplacements du projet, quel que soit le dossier ou il a ete copie.

Tout se deduit de l'emplacement de ce fichier ; aucun chemin absolu. Deux
dossiers a ne pas confondre :

- `modeles/` : les reseaux de reperage. Ils font partie du code et voyagent
  avec lui d'un poste a l'autre ;
- `data/`    : l'etat propre a CE poste — bases d'annotations, cache d'analyse,
  sauvegardes, derniere image ouverte.

Une base SQLite par mosaique, `data/bases/<image_id>.sqlite3` (voir
`bases.py`) : un fichier se donne tel quel a un collegue. Les copies de
sauvegarde vont dans `data/sauvegardes/`. L'ancienne base unique,
`data/annotations.sqlite3`, est separee au premier lancement puis rangee dans
`data/sauvegardes/` sous un nom date ; elle n'est jamais effacee.
"""

from __future__ import annotations

import os
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
MODELS_DIR = APP_DIR / "modeles"
DATA_DIR = APP_DIR / "data"
BASES_DIR = DATA_DIR / "bases"            # une base par mosaique
BACKUP_DIR = DATA_DIR / "sauvegardes"
# L'ancienne base unique, separee au premier besoin (bases.migrate). Le nom
# DEFAULT_DATABASE ne reste que pour LEGACY/ : ce n'est plus la base de travail.
OLD_DATABASE = DATA_DIR / "annotations.sqlite3"
DEFAULT_DATABASE = OLD_DATABASE
LAST_IMAGE_FILE = DATA_DIR / "derniere_image.txt"
# La mosaique de reference est designee HORS du code, sur chaque poste : le
# depot ne contient ainsi aucun nom d'echantillon ni chemin personnel.
REFERENCE_IMAGE_FILE = DATA_DIR / "image_reference.txt"


def analysis_root() -> Path:
    """Dossier des resultats de l'analyse automatique (`CRACK_ANALYSE_DIR` pour un autre)."""
    return Path(os.environ.get("CRACK_ANALYSE_DIR") or DATA_DIR / "analyse")


def _recorded_image(record: Path) -> Path | None:
    try:
        text = record.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    path = Path(text) if text else None
    return path if path is not None and path.is_file() else None


def last_image(record: Path = LAST_IMAGE_FILE) -> Path | None:
    """Derniere mosaique ouverte sur ce poste, si elle existe encore."""
    return _recorded_image(record)


def remember_image(path: Path, record: Path = LAST_IMAGE_FILE) -> None:
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(str(Path(path).resolve()), encoding="utf-8")


def default_image() -> Path | None:
    """Image a ouvrir sans autre precision.

    `CRACK_IMAGE_PATH` si elle est definie, meme si le fichier manque : mieux
    vaut une erreur claire qu'une autre image ouverte en silence. Sinon la
    derniere image ouverte, puis la mosaique de reference.
    """
    chosen = os.environ.get("CRACK_IMAGE_PATH")
    if chosen:
        return Path(chosen)
    return last_image() or reference_image()


def reference_image(record: Path = REFERENCE_IMAGE_FILE) -> Path | None:
    """Mosaique de reference de ce poste, si elle est designee et presente.

    C'est celle dont les tuiles validees servent de verite terrain au banc et
    aux entrainements. On la designe par `CRACK_REFERENCE_IMAGE`, ou en
    ecrivant son chemin dans `data/image_reference.txt`.
    """
    chosen = os.environ.get("CRACK_REFERENCE_IMAGE")
    if chosen:
        return Path(chosen) if Path(chosen).is_file() else None
    return _recorded_image(record)


def benchmark_image() -> Path | None:
    """Image par defaut du banc et des entrainements : la reference, sinon la
    derniere ouverte."""
    return reference_image() or last_image()
