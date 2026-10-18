"""Sauvegardes automatiques des bases d'annotations.

Le travail sur une mosaique tient dans un fichier SQLite. L'application en
fait une copie a chaque ouverture, AVANT d'y ecrire quoi que ce soit, puis
toutes les quelques heures tant qu'elle tourne. Les copies portent le nom de la
base : `<base>-AAAAMMJJ-HHMMSS.sqlite3`.

Les copies de l'ancienne base unique (`annotations-AAAAMMJJ-HHMMSS`) ne sont
jamais retirees : aucune base ne porte plus ce nom, et elles gardent des
annotations qu'on voudra peut-etre restaurer.

La copie passe par l'API de sauvegarde de SQLite : elle reste coherente meme
pendant qu'un annotateur ecrit, ce qu'une simple copie de fichier ne garantit
pas avec le journal WAL.
"""

from __future__ import annotations

import filecmp
import re
import sqlite3
import threading
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path


INTERVAL_HOURS = 4
KEEP_DAYS = 14
# Garde-fou : meme apres un long arret, on ne descend jamais sous ce nombre.
MINIMUM_KEPT = 10
STAMP = "%Y%m%d-%H%M%S"
# Nom de l'ancienne base unique : ses copies ne sont jamais retirees.
PROTECTED_STEMS = frozenset({"annotations"})


def _backups(directory: Path, stem: str) -> list[Path]:
    pattern = re.compile(re.escape(stem) + r"-\d{8}-\d{6}\.sqlite3")
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if pattern.fullmatch(p.name))


def backup_database(
    database_path: Path,
    directory: Path,
    now: datetime | None = None,
    keep_days: int = KEEP_DAYS,
) -> Path | None:
    """Copie la base dans `directory` et rend le chemin de la copie.

    Une copie identique a la precedente n'est pas conservee : un serveur
    laisse ouvert sans annotation ne remplit pas le disque.
    """
    database_path = Path(database_path)
    if not database_path.exists():
        return None
    now = now or datetime.now()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = database_path.stem
    previous = _backups(directory, stem)
    target = directory / f"{stem}-{now.strftime(STAMP)}.sqlite3"
    temporary = target.with_suffix(".tmp")
    with closing(sqlite3.connect(database_path, timeout=30)) as source, \
            closing(sqlite3.connect(temporary)) as copy:
        source.backup(copy)
    temporary.replace(target)
    if previous and previous[-1] != target and filecmp.cmp(previous[-1], target, shallow=False):
        target.unlink()
        target = previous[-1]
    prune(directory, stem, now, keep_days)
    return target


def prune(directory: Path, stem: str, now: datetime, keep_days: int = KEEP_DAYS) -> None:
    """Retire les copies plus vieilles que `keep_days`, sauf les plus recentes.

    Jamais celles de l'ancienne base unique (`PROTECTED_STEMS`).
    """
    if stem in PROTECTED_STEMS:
        return
    limit = now - timedelta(days=keep_days)
    for path in _backups(directory, stem)[:-MINIMUM_KEPT]:
        taken = datetime.strptime(path.stem[len(stem) + 1:], STAMP)
        if taken < limit:
            path.unlink()


def start_periodic_backup(database_path: Path, directory: Path) -> threading.Thread:
    """Relance une sauvegarde toutes les `INTERVAL_HOURS` heures."""
    def run() -> None:
        while not stop.wait(INTERVAL_HOURS * 3600):
            try:
                backup_database(database_path, directory)
            except Exception as error:  # le serveur doit continuer a tourner
                print(f"Sauvegarde impossible dans {directory} : {error}", flush=True)

    stop = threading.Event()
    thread = threading.Thread(target=run, name="sauvegarde-base", daemon=True)
    thread.stop = stop
    thread.start()
    return thread
