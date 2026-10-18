"""Menage automatique de `data/`, a l'ouverture d'une mosaique.

Ne retire que ce qui se recalcule et ne sera plus relu ; jamais une base, une
sauvegarde, un resultat ou un rapport :

- calques de diagnostic (`analysis_cache/`) des mosaiques qui n'existent plus,
  puis les plus anciens au-dela de `CACHE_LIMIT_GB` (quelques secondes de
  calcul par tuile pour les refaire) ;
- detections brutes d'un ancien detecteur (`analyse/<image>/detections_*.json`) :
  l'analyse ne relit que celles du detecteur en service ;
- `-wal` vides et `-shm` laisses a cote des sauvegardes par une simple lecture.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

CACHE_LIMIT_GB = 2.0
MANIFEST = "source.json"          # meme nom que dans analysis_service


def _hash(path: Path) -> str:
    """Empreinte d'un detecteur, comme `analyse.chaine._file_hash`."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12] if Path(path).exists() else "aucun"


def _size(paths) -> int:
    return sum(p.stat().st_size for p in paths)


def tidy_cache(cache_dir: Path, limit_bytes: float = CACHE_LIMIT_GB * 1e9) -> list[Path]:
    """Calques des mosaiques disparues, puis les plus anciens au-dela de la limite."""
    removed = []
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        return removed
    for folder in cache_dir.iterdir():
        manifest = folder / MANIFEST
        if not folder.is_dir() or not manifest.exists():
            continue
        try:
            image = json.loads(manifest.read_text(encoding="utf-8")).get("image")
        except (OSError, ValueError):
            continue
        if image and not Path(image).exists():
            shutil.rmtree(folder, ignore_errors=True)
            removed.append(folder)
    files = sorted(cache_dir.glob("*/*.npz"), key=lambda p: p.stat().st_mtime)
    total = _size(files)
    for path in files:
        if total <= limit_bytes:
            break
        size = path.stat().st_size
        try:
            path.unlink()
        except OSError:
            continue                          # lu en ce moment : on le garde
        total -= size
        removed.append(path)
    return removed


def tidy_detections(analysis_root: Path, detector_path: Path) -> list[Path]:
    """Detections brutes d'un autre detecteur que celui en service."""
    current = f"detections_{_hash(detector_path)}.json"
    removed = []
    for path in Path(analysis_root).glob("*/detections_*.json"):
        if path.name != current:
            path.unlink()
            removed.append(path)
    return removed


def tidy_backups(backup_dir: Path) -> list[Path]:
    """`-wal` vides et leur `-shm`, laisses par une lecture des sauvegardes."""
    removed = []
    for wal in Path(backup_dir).rglob("*.sqlite3-wal"):
        if wal.stat().st_size:
            continue                          # contient des donnees : on n'y touche pas
        for path in (wal, Path(str(wal)[:-4] + "-shm")):
            if path.exists():
                path.unlink()
                removed.append(path)
    return removed


def tidy(data_dir: Path, cache_dir: Path, analysis_root: Path, detector_path: Path, log=print) -> None:
    """Tout le menage ; une erreur n'empeche jamais d'ouvrir la mosaique."""
    for name, step in (("calques", lambda: tidy_cache(cache_dir)),
                       ("detections", lambda: tidy_detections(analysis_root, detector_path)),
                       ("sauvegardes", lambda: tidy_backups(Path(data_dir) / "sauvegardes"))):
        try:
            removed = step()
        except OSError as error:
            log(f"Menage ({name}) incomplet : {error}")
            continue
        if removed:
            log(f"Menage : {len(removed)} fichier(s) recalculable(s) retire(s) ({name}).")


def disk_usage(data_dir: Path) -> dict[str, float]:
    """Mo occupes par dossier de `data/`, pour s'y retrouver."""
    return {folder.name: _size(p for p in folder.rglob("*") if p.is_file()) / 1e6
            for folder in sorted(Path(data_dir).iterdir()) if folder.is_dir()}


if __name__ == "__main__":
    from .paths import DATA_DIR
    for name, mb in disk_usage(DATA_DIR).items():
        print(f"{name:20} {mb:9.1f} Mo")
    print(f"(calques limites a {CACHE_LIMIT_GB:g} Go ; {os.environ.get('CRACK_CACHE_DIR') or 'analysis_cache'})")
