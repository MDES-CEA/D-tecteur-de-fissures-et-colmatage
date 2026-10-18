"""Une base SQLite par mosaique : `data/bases/<image_id>.sqlite3`.

Chaque mosaique a son propre fichier, qu'on peut donner tel quel a un
collegue. Le schema est celui de `database.py`. Un fichier peut contenir
plusieurs images (base importee) : la recherche porte sur les images, pas sur
les noms de fichiers.

- `database_for`  : la base d'une mosaique, nouvelle si elle n'a jamais ete
  ouverte ;
- `find_database` : la meme recherche, en lecture seule, pour les outils ;
- `migrate`       : separe une fois pour toutes l'ancienne base unique
  `data/annotations.sqlite3`, tout ou rien ;
- `export_database`, `plan_import`, `apply_import` : le menu Fichier.

Les fichiers de `bases/` ne sont lus qu'en lecture seule ; un fichier
illisible est signale et ignore, jamais modifie. Toutes les fonctions prennent
le dossier `data/` en parametre (par defaut celui du projet) : les tests
travaillent ainsi dans un dossier temporaire.
"""

from __future__ import annotations

import os
import re
import shutil
import sqlite3
import tempfile
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from . import paths
from .database import _same_path, _slug, already_registered_message, resolve_among

SUFFIX = ".sqlite3"
STAMP = "%Y%m%d-%H%M%S"
TABLES = ("images", "zones", "cracks", "crack_revisions", "suggestions")
# Nom de l'ancienne base unique : jamais donne a une mosaique, pour que ses
# copies datees (annotations-AAAAMMJJ-HHMMSS) ne soient jamais confondues avec
# celles d'une base de mosaique (voir backup.prune).
OLD_STEM = "annotations"
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._ -]*")


class MigrationError(RuntimeError):
    """La separation n'a pas eu lieu ; rien n'a change."""


# -- Emplacements ------------------------------------------------------------------

def _data(data_dir) -> Path:
    return Path(data_dir) if data_dir is not None else paths.DATA_DIR


def bases_dir(data_dir=None) -> Path:
    return _data(data_dir) / paths.BASES_DIR.name


def old_database(data_dir=None) -> Path:
    return _data(data_dir) / paths.OLD_DATABASE.name


def default_backup_dir(data_dir=None) -> Path:
    return _data(data_dir) / paths.BACKUP_DIR.name


def backup_dir_for(database_path: Path, data_dir=None) -> Path:
    """Dossier des sauvegardes d'une base.

    `CRACK_BACKUP_DIR` s'il est defini ; sinon `data/sauvegardes/` pour les
    bases de `data/bases/` ; sinon `<dossier de la base>/sauvegardes`.
    """
    chosen = os.environ.get("CRACK_BACKUP_DIR")
    if chosen:
        return Path(chosen).resolve()
    database_path = Path(database_path).resolve()
    if _same_path(database_path.parent, bases_dir(data_dir).resolve()):
        return default_backup_dir(data_dir).resolve()
    return database_path.parent / "sauvegardes"


def check_file_name(image_id: str) -> None:
    """Un identifiant force doit pouvoir servir de nom de fichier."""
    if not _SAFE_NAME.fullmatch(image_id) or image_id.endswith((".", " ")):
        raise ValueError(
            f"Identifiant « {image_id} » inutilisable comme nom de base : "
            f"lettres, chiffres, tirets, points et espaces seulement.")


# -- Lecture -----------------------------------------------------------------------

def _connect_ro(path: Path) -> sqlite3.Connection:
    """Connexion qui ne peut rien ecrire, meme par erreur."""
    uri = "file:" + quote(Path(path).resolve().as_posix(), safe="/:") + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}


@dataclass
class Registry:
    """Les images de toutes les bases de `bases/`, chacune avec son fichier."""
    images: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    unreadable: list[Path] = field(default_factory=list)


def scan(data_dir=None) -> Registry:
    registry = Registry()
    folder = bases_dir(data_dir)
    if not folder.is_dir():
        return registry
    for path in sorted(folder.glob("*" + SUFFIX)):
        try:
            with closing(_connect_ro(path)) as connection:
                if not _tables(connection):
                    continue                     # base creee mais jamais remplie
                rows = connection.execute("SELECT id, path, width, height FROM images ORDER BY id").fetchall()
        except sqlite3.Error as error:
            registry.notes.append(f"Base illisible, ignoree : {path} ({error})")
            registry.unreadable.append(path)
            continue
        registry.images.extend({**dict(row), "file": path} for row in rows)
    return registry


def database_for(image_path: Path, width: int, height: int, image_id: str | None = None,
                 data_dir=None, log=print) -> tuple[Path, str]:
    """La base d'une mosaique et son identifiant ; rien n'est ecrit.

    Memes regles que `AnnotationDatabase.resolve_image_id`, sur les images de
    TOUTES les bases : chemin connu, sinon fichier deplace, sinon nouvelle base
    `bases/<identifiant>.sqlite3`, avec un identifiant libre dans toutes les
    bases. Un identifiant force (`CRACK_IMAGE_ID`) donne la base qui le
    contient deja, sinon `bases/<identifiant>.sqlite3`.
    """
    registry = scan(data_dir)
    for note in registry.notes:
        log(note)
    folder = bases_dir(data_dir)
    resolved = Path(image_path).resolve()
    if image_id:
        check_file_name(image_id)
        owner = next((row for row in registry.images
                      if _same_path(row["path"], resolved) and row["id"] != image_id), None)
        if owner is not None:
            raise ValueError(already_registered_message(owner["id"], image_id, resolved))
        holder = next((row for row in registry.images if row["id"] == image_id), None)
        return (holder["file"] if holder else folder / f"{image_id}{SUFFIX}"), image_id
    # Un fichier illisible garde son nom : une nouvelle base ne l'ecrase jamais.
    reserved = {OLD_STEM, *(path.stem for path in registry.unreadable)}
    image_id, row, note = resolve_among(registry.images, resolved, width, height, reserved)
    if note:
        log(note)
    return (row["file"] if row is not None else folder / f"{image_id}{SUFFIX}"), image_id


def find_database(image_path: Path, data_dir=None, log=print) -> Path | None:
    """La base ou ce fichier est enregistre, en lecture seule ; None s'il n'y est pas."""
    registry = scan(data_dir)
    for note in registry.notes:
        log(note)
    resolved = Path(image_path).resolve()
    row = next((row for row in registry.images if _same_path(row["path"], resolved)), None)
    return row["file"] if row is not None else None


def tool_database(database: Path | None, image: Path | None, data_dir=None) -> tuple[Path, Path | None]:
    """(base, image) d'un outil en ligne de commande.

    `--database` donne : rien ne change. Sinon la base de `--image` (par
    defaut la mosaique de reference, ou la derniere ouverte), trouvee en
    lecture seule ; l'ancienne base unique est d'abord separee si besoin.
    """
    if database is not None:
        return Path(database), image
    image = image or paths.benchmark_image()
    if image is None:
        raise SystemExit("Preciser --image (aucune mosaique de reference ni derniere image sur ce poste).")
    try:
        migrate(data_dir)
    except MigrationError as error:
        raise SystemExit(str(error)) from error
    found = find_database(image, data_dir)
    if found is None:
        raise SystemExit(
            f"Mosaique jamais ouverte dans l'application (ou deplacee depuis) :\n  {image}\n"
            f"L'ouvrir une fois dans l'application, qui lui cree sa base, ou preciser --database.")
    return found, image


# -- Comptes, extraction -----------------------------------------------------------

def _image_counts(connection: sqlite3.Connection, image_id: str) -> dict:
    """Nombre de lignes et sommes d'identifiants, table par table, pour une image."""
    queries = {
        "zones": "SELECT count(*), total(id), total(revision) FROM zones WHERE image_id = ?",
        "cracks": "SELECT count(*), total(id), total(revision) FROM cracks WHERE image_id = ?",
        "crack_revisions": "SELECT count(*), total(r.crack_id), total(r.revision) FROM crack_revisions r "
                           "JOIN cracks c ON c.id = r.crack_id WHERE c.image_id = ?",
        "suggestions": "SELECT count(*), total(s.crack_id), total(s.revision) FROM suggestions s "
                       "JOIN cracks c ON c.id = s.crack_id WHERE c.image_id = ?",
    }
    return {name: tuple(connection.execute(query, (image_id,)).fetchone()) for name, query in queries.items()}


def _table_totals(connection: sqlite3.Connection) -> dict:
    """Les memes comptes, sur les tables entieres."""
    columns = {"zones": "id", "cracks": "id", "crack_revisions": "crack_id", "suggestions": "crack_id"}
    return {name: tuple(connection.execute(
        f"SELECT count(*), total({column}), total(revision) FROM {name}").fetchone())
        for name, column in columns.items()}


def counts(path: Path, connect=_connect_ro) -> dict[str, dict]:
    """{image_id: comptes} d'une base ; leve MigrationError s'il reste des lignes sans image."""
    with closing(connect(path)) as connection:
        ids = [row[0] for row in connection.execute("SELECT id FROM images ORDER BY id")]
        result = {image_id: _image_counts(connection, image_id) for image_id in ids}
        totals = _table_totals(connection)
    for name, total in totals.items():
        summed = tuple(sum(values[name][k] for values in result.values()) for k in range(3))
        if summed != total:
            raise MigrationError(
                f"{path} : {total[0] - summed[0]} ligne(s) de la table {name} ne se rattachent a aucune "
                f"image. Rien n'a change.")
    return result


def _extract(source: Path, destination: Path, image_id: str | None, connect=_connect_ro) -> None:
    """Copie coherente de `source` (API de sauvegarde), reduite a une image si `image_id`.

    Les identifiants (tuiles, fissures, revisions, suggestions) restent ceux de
    l'original : les resultats de l'analyse y font reference. Le fichier rendu
    se suffit a lui-meme (journal classique, sans -wal).
    """
    with closing(connect(source)) as origin, closing(sqlite3.connect(destination)) as copy:
        origin.backup(copy)
    with closing(sqlite3.connect(destination)) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = DELETE")
        if image_id is not None:
            with connection:
                connection.execute("DELETE FROM images WHERE id != ?", (image_id,))   # cascades
        connection.execute("VACUUM")


def _verify(path: Path, expected: dict[str, dict]) -> None:
    with closing(_connect_ro(path)) as connection:
        check = connection.execute("PRAGMA integrity_check").fetchone()[0]
    if check != "ok":
        raise MigrationError(f"{path} : copie abimee ({check}).")
    found = counts(path)
    if found != expected:
        raise MigrationError(f"{path} : la copie ne correspond pas a l'original "
                             f"({_describe(found)} au lieu de {_describe(expected)}).")


def _summary(c: dict) -> str:
    return (f"{c['zones'][0]} tuiles, {c['cracks'][0]} fissures, {c['crack_revisions'][0]} revisions, "
            f"{c['suggestions'][0]} suggestions")


def _describe(values: dict[str, dict]) -> str:
    return "; ".join(f"{image_id} : {_summary(c)}" for image_id, c in values.items()) or "aucune image"


def _remove(path: Path) -> None:
    for suffix in ("-wal", "-shm", ""):
        candidate = Path(f"{path}{suffix}")
        if candidate.exists():
            candidate.unlink()


# -- Migration de l'ancienne base unique -------------------------------------------

def _checkpoint(path: Path) -> None:
    """Reporte le -wal dans le fichier ; echoue si la base est ouverte ailleurs."""
    with closing(sqlite3.connect(path, timeout=10)) as connection:
        busy, _, _ = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    # La derniere connexion qui se ferme retire -wal et -shm : s'ils restent,
    # une autre fenetre (ou un outil) tient encore la base.
    if busy or any(Path(f"{path}{suffix}").exists() for suffix in ("-wal", "-shm")):
        raise MigrationError(_busy_message(path))


def _busy_message(path: Path) -> str:
    return (f"Separation de la base impossible pour l'instant : {path} est ouverte ailleurs "
            f"(une autre fenetre de l'application, un outil ?). Les fermer puis relancer. Rien n'a change.")


def _connect_rw(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def migrate(data_dir=None, log=print, now: datetime | None = None) -> list[Path]:
    """Separe l'ancienne base `data/annotations.sqlite3` en une base par mosaique.

    Une seule fois, au premier besoin. Tout ou rien : chaque mosaique est
    copiee (API de sauvegarde), reduite a elle-meme, puis comparee a
    l'original table par table ; ce n'est qu'ensuite que les copies entrent
    dans `bases/` et que l'original est renomme
    `sauvegardes/annotations-avant-separation-AAAAMMJJ-HHMMSS.sqlite3`.
    L'original n'est jamais supprime. Au moindre echec, les copies sont
    retirees et MigrationError dit pourquoi. Rend les bases de `bases/`.
    """
    old = old_database(data_dir)
    if not old.exists():
        return []
    folder = bases_dir(data_dir)
    backups = default_backup_dir(data_dir)
    now = now or datetime.now()
    kept = backups / f"{old.stem}-avant-separation-{now.strftime(STAMP)}{SUFFIX}"
    placed: list[Path] = []
    new_folders = [path for path in (folder, backups) if not path.exists()]
    staging = None
    try:
        _checkpoint(old)
        with closing(_connect_rw(old)) as connection:
            empty = "images" not in _tables(connection)
        reference = {} if empty else counts(old, _connect_rw)     # base jamais remplie : rien a separer
        for image_id in reference:
            try:
                check_file_name(image_id)
            except ValueError as error:
                raise MigrationError(f"{error} Rien n'a change.") from error
        folder.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".separation-", dir=folder))
        planned = []
        for image_id, expected in reference.items():
            target = folder / f"{image_id}{SUFFIX}"
            if target.exists():
                # Separation deja faite (interrompue juste avant le renommage) : on
                # la reprend si la base est identique, jamais sinon.
                try:
                    _verify(target, {image_id: expected})
                except (MigrationError, sqlite3.Error) as error:
                    raise MigrationError(f"{target} existe deja et ne correspond pas a l'ancienne base. "
                                         f"Rien n'a change.") from error
                continue
            staged = staging / target.name
            _extract(old, staged, image_id, _connect_rw)      # rw : ne laisse ni -wal ni -shm
            _verify(staged, {image_id: expected})
            planned.append((staged, target))
        if not empty and counts(old, _connect_rw) != reference:
            raise MigrationError(f"{old} a change pendant la separation. Rien n'a change ; relancer.")
        _checkpoint(old)
        for staged, target in planned:
            if target.exists():
                raise MigrationError(f"{target} est apparu pendant la separation. Rien n'a change ; relancer.")
            os.replace(staged, target)
            placed.append(target)
        backups.mkdir(parents=True, exist_ok=True)
        if kept.exists():
            raise MigrationError(f"{kept} existe deja. Rien n'a change ; relancer.")
        try:
            os.rename(old, kept)                   # sous Windows, echoue si la base est ouverte
        except OSError as error:
            raise MigrationError(_busy_message(old)) from error
    except BaseException as error:
        for target in placed:
            try:
                _remove(target)
            except OSError:
                pass
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        for path in new_folders:
            try:
                path.rmdir()                       # seulement s'il est reste vide
            except OSError:
                pass
        if isinstance(error, (MigrationError, KeyboardInterrupt, SystemExit)):
            raise
        raise MigrationError(f"Separation de la base impossible : {error}. Rien n'a change.") from error
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
    for image_id, values in reference.items():
        log(f"Base separee : {folder / (image_id + SUFFIX)} ({_summary(values)})")
    log(f"Ancienne base gardee telle quelle : {kept}")
    return [folder / f"{image_id}{SUFFIX}" for image_id in reference]


# -- Export et import (menu Fichier) -----------------------------------------------

def export_database(database_path: Path, destination: Path, image_id: str) -> Path:
    """Copie coherente de la base de cette mosaique, reduite a elle seule."""
    database_path, destination = Path(database_path).resolve(), Path(destination).resolve()
    if _same_path(database_path, destination):
        raise ValueError("Choisir un autre fichier que la base elle-meme.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        if temporary.exists():
            temporary.unlink()
        _extract(database_path, temporary, image_id)
        with closing(_connect_ro(database_path)) as connection:
            expected = {image_id: _image_counts(connection, image_id)}
        _verify(temporary, expected)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


@dataclass
class ImportPlan:
    source: Path
    images: list[dict]                 # id, path, width, height
    targets: dict[str, Path]           # une base par mosaique, dans bases/
    replaced: list[Path]               # bases existantes des memes mosaiques


def plan_import(source: Path, data_dir=None) -> ImportPlan:
    """Verifie une base a importer et dit ce qu'elle remplacerait ; rien n'est ecrit."""
    source = Path(source).resolve()
    folder = bases_dir(data_dir)
    if _same_path(source.parent, folder.resolve()):
        raise ValueError("Cette base est déjà dans le dossier des bases.")
    try:
        with closing(_connect_ro(source)) as connection:
            missing = [name for name in TABLES if name not in _tables(connection)]
            if missing:
                raise ValueError(f"Ce n'est pas une base d'annotations (tables manquantes : {', '.join(missing)}).")
            images = [dict(row) for row in connection.execute(
                "SELECT id, path, width, height FROM images ORDER BY id")]
            check = connection.execute("PRAGMA quick_check").fetchone()[0]
    except sqlite3.Error as error:
        raise ValueError(f"Base illisible : {error}") from error
    if not images:
        raise ValueError("Base sans mosaïque (table images vide).")
    if check != "ok":
        raise ValueError(f"Base abîmée : {check}")

    def copied_here(row, image) -> bool:
        """La meme mosaique copiee sur ce poste, peut-etre sous un autre nom : memes dimensions,
        l'image recue n'existe pas ici, celle de la base locale si."""
        return ((row["width"], row["height"]) == (image["width"], image["height"])
                and not Path(image["path"]).exists() and Path(row["path"]).exists())

    def same_mosaic(row) -> bool:
        return any(row["id"] == image["id"] or _same_path(row["path"], image["path"]) or copied_here(row, image)
                   for image in images)

    registry = scan(data_dir)
    replaced = sorted({row["file"] for row in registry.images if same_mosaic(row)})
    for path in replaced:
        others = [row["id"] for row in registry.images if row["file"] == path and not same_mosaic(row)]
        if others:
            raise ValueError(f"{path.name} contient aussi {', '.join(others)} : import refusé, "
                             f"pour ne pas les perdre.")
    # Une base par mosaique, meme si le fichier recu en contient plusieurs
    # (une copie de l'ancienne base unique, par exemple).
    taken = {path.stem for path in folder.glob("*" + SUFFIX)} - {path.stem for path in replaced}
    targets = {}
    for image in images:
        name = image["id"]
        try:
            check_file_name(name)
        except ValueError:
            name = _slug(name)
        base, suffix = name, 2
        while name in taken or name == OLD_STEM:
            name, suffix = f"{base}-{suffix}", suffix + 1
        taken.add(name)
        targets[image["id"]] = folder / f"{name}{SUFFIX}"
    return ImportPlan(source, images, targets, replaced)


def apply_import(plan: ImportPlan, data_dir=None, log=print, now: datetime | None = None) -> list[Path]:
    """Copie la base recue dans `bases/`, une base par mosaique.

    Les bases remplacees partent d'abord dans les sauvegardes
    (`<base>-avant-import-AAAAMMJJ-HHMMSS.sqlite3`, jamais retirees) ; elles
    ne doivent plus etre ouvertes (fermer la session). Rien ne change tant
    que toutes les copies ne sont pas faites et verifiees.
    """
    now = now or datetime.now()
    stamp = now.strftime(STAMP)
    folder = bases_dir(data_dir)
    folder.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".import-", dir=folder))
    try:
        expected = counts(plan.source)
        staged = {}
        for image_id, target in plan.targets.items():
            staged[image_id] = staging / target.name
            _extract(plan.source, staged[image_id], image_id)
            _verify(staged[image_id], {image_id: expected[image_id]})
        for path in plan.replaced:
            directory = backup_dir_for(path, data_dir)
            directory.mkdir(parents=True, exist_ok=True)
            saved = directory / f"{path.stem}-avant-import-{stamp}{SUFFIX}"
            _extract(path, saved, None)
            _verify(saved, counts(path))
            log(f"Base remplacee, gardee dans les sauvegardes : {saved}")
        for path in plan.replaced:
            _remove(path)
        for image_id, target in plan.targets.items():
            os.replace(staged[image_id], target)
            log(f"Base importee : {target}")
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return list(plan.targets.values())
