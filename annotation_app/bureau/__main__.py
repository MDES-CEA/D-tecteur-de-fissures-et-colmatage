"""Point d'entree : `python -m annotation_app.bureau`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QSplashScreen
from PySide6.QtGui import QColor, QPixmap


def choose_image(start: Path | None) -> Path | None:
    path, _ = QFileDialog.getOpenFileName(None, "Choisir la mosaïque à annoter", str(start or ""),
                                          "Images TIFF (*.tif *.tiff);;Tous les fichiers (*.*)")
    return Path(path) if path else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Annotation des fissures")
    parser.add_argument("--image", type=Path, help="mosaique a ouvrir")
    parser.add_argument("--database", type=Path,
                        help="base forcee (par defaut, chaque mosaique a la sienne dans data/bases/)")
    parser.add_argument("--choisir", action="store_true", help="choisir la mosaique dans une fenetre")
    parser.add_argument("--entrainer", action="store_true",
                        help="ouvrir seulement la fenetre de reentrainement du detecteur")
    args = parser.parse_args(argv)

    app = QApplication(sys.argv[:1])
    app.setOrganizationName("Fissures")
    app.setApplicationName("Annotation")

    import os

    from ..paths import benchmark_image, default_image, last_image

    if args.entrainer:
        from .dialogues import TrainingDialog
        image = args.image or benchmark_image()
        # Base forcee, sinon celle de l'image choisie dans la fenetre (data/bases/).
        database = args.database or os.environ.get("CRACK_DATABASE_PATH") or None
        if database is None:
            from ..bases import MigrationError, migrate
            try:
                migrate(log=lambda text: print(text, flush=True))
            except MigrationError as error:
                QMessageBox.critical(None, "Bases d'annotations", str(error))
                return 1
        dialog = TrainingDialog(image, database)
        dialog.show()
        return app.exec()

    image = args.image
    if image is None and not args.choisir:
        image = default_image()
    if image is None or not Path(image).is_file():
        image = choose_image(last_image())
        if image is None:
            return 0

    background = QPixmap(520, 90)
    background.fill(QColor(45, 45, 45))
    splash = QSplashScreen(background)
    splash.showMessage(f"Ouverture de {Path(image).name}…\n(la première fois, préparation de l'image)",
                       Qt.AlignmentFlag.AlignCenter, Qt.GlobalColor.white)
    splash.show()
    app.processEvents()

    from ..session import Session, SessionError
    from .fenetre import MainWindow

    def log(text: str) -> None:
        print(text, flush=True)
        splash.showMessage(text, Qt.AlignmentFlag.AlignCenter, Qt.GlobalColor.white)
        app.processEvents()

    try:
        session = Session(Path(image), args.database, log=log)
    except SessionError as error:
        splash.close()
        QMessageBox.critical(None, "Ouverture impossible", str(error))
        return 1
    window = MainWindow(session)
    window.show()
    splash.finish(window)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
