"""Persistent diagnostic tiles and cancellable mosaic preparation."""
import hashlib
import io
import json
import os
import shutil
from pathlib import Path
from threading import Lock, Thread, Event

import numpy as np
from PIL import Image, ImageDraw


class AnalysisLayers:
    KINDS = {'cost', 'route_cost', 'score', 'mask', 'skeleton', 'centerline'}

    def __init__(self, detector, directory):
        self.detector = detector
        source = detector.tiles.image_path
        key = cache_key(source, detector.max_analysis_side, getattr(detector, 'factor', 1.0))
        self.directory = Path(directory)/key
        self.directory.mkdir(parents=True, exist_ok=True)
        # Chaque dossier dit de quelle image il vient : c'est ce qui permet de
        # reconnaitre ensuite ceux qu'une nouvelle version a rendus inutiles.
        manifest = self.directory/MANIFEST
        if not manifest.exists():
            manifest.write_text(json.dumps({'image': str(source)}), encoding='utf-8')
        purge_obsolete_caches(Path(directory), self.directory, source)
        self.lock = Lock()
        self.cancel = Event()
        self.job_lock = Lock()
        self.status = {'running': False, 'completed': 0, 'total': 0, 'errors': []}
        self.overview_lock = Lock()
        self.overview_cache = {'signature': None, 'content': None}

    def path(self, zone):
        return self.directory / ('-'.join(str(zone[k]) for k in ('id', 'x', 'y', 'width', 'height'))+'.npz')

    def prepare(self, zone):
        path = self.path(zone)
        with self.lock:
            if path.exists():
                return
            analysis = self.detector.analyze(zone)
            centerline = Image.new('L', (analysis.score.shape[1], analysis.score.shape[0]))
            draw = ImageDraw.Draw(centerline)
            for crack in analysis.cracks:
                draw.line([((x-zone['x'])/analysis.scale, (y-zone['y'])/analysis.scale)
                           for x, y in crack['geometry']], fill=255, width=1)
            route = self.detector.prepare_route(zone)
            arrays = {kind: getattr(analysis, kind) for kind in ('cost', 'score', 'mask', 'skeleton')}
            arrays['route_cost'] = route.cost
            arrays['centerline'] = np.asarray(centerline) > 0
            arrays['geometry_json'] = np.array(json.dumps(analysis.cracks))
            arrays['diagnostics_json'] = np.array(json.dumps(analysis.diagnostics if hasattr(analysis, 'diagnostics') else {}))
            temporary = path.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, **arrays)
            temporary.replace(path)

    def cached_proposals(self, zone):
        """Propositions deja calculees pour cette tuile, ou None.

        `prepare()` enregistre exactement ce que rend `detector.analyze` :
        meme geometrie, memes diagnostics. Les relire coute quelques
        millisecondes la ou le reseau en demande plusieurs secondes. Sans
        cela, « Detecter » recalculait tout meme apres avoir prepare toute la
        mosaique.
        """
        path = self.path(zone)
        if not path.exists():
            return None
        try:
            with np.load(path) as data:
                cracks = json.loads(str(data['geometry_json']))
                diagnostics = json.loads(str(data['diagnostics_json']))
        except (KeyError, ValueError, OSError):
            return None
        return cracks, {**diagnostics, 'source': 'couches deja calculees'}

    def automatic_cracks(self, zones):
        cracks = []
        for zone in zones:
            if not self.path(zone).exists():
                continue
            with np.load(self.path(zone)) as data:
                proposals = json.loads(str(data['geometry_json']))
            for index, proposal in enumerate(proposals):
                cracks.append({'id': -(zone['id']*10000+index+1), 'zone_id':zone['id'],
                               'category':'unclassified', 'state':'proposed', 'geometry':proposal['geometry']})
        return cracks

    def raster(self, zone, kind):
        self.prepare(zone)
        with np.load(self.path(zone)) as data:
            values = data[kind]
        return Image.fromarray((np.clip(values, 0, 1)*255).astype('uint8'))

    def png(self, zone, kind):
        output = io.BytesIO()
        self.raster(zone, kind).save(output, format='PNG')
        return output.getvalue()

    def start(self, zones):
        with self.job_lock:
            if self.status['running']:
                return dict(self.status)
            self.cancel.clear()
            self.status = {'running': True, 'completed': 0, 'total': len(zones), 'errors': []}
        def run():
            try:
                for zone in zones:
                    if self.cancel.is_set():
                        break
                    try:
                        self.prepare(zone)
                    except Exception as error:
                        with self.job_lock:
                            self.status['errors'].append({'zone_id': zone['id'], 'message': str(error)})
                    else:
                        with self.job_lock:
                            self.status['completed'] += 1
            finally:
                with self.job_lock:
                    self.status['running'] = False
                    self.status['cancelled'] = self.cancel.is_set()
        Thread(target=run, daemon=True).start()
        return dict(self.status)

    def overview_signature(self, zones, kind):
        """Ce dont depend l'image d'ensemble, et rien d'autre.

        Elle ne change que si une tuile du perimetre vient d'etre calculee, ou
        si le perimetre lui-meme change. Deplacer la vue n'y change rien.
        """
        parts = [kind]
        for zone in sorted(zones, key=lambda z: z['id']):
            path = self.path(zone)
            if path.exists():
                parts.append(f"{zone['id']}:{path.stat().st_mtime_ns}")
        return hashlib.sha256('|'.join(parts).encode()).hexdigest()[:16]

    def overview(self, zones, kind):
        """Image d'ensemble des tuiles calculees, avec memorisation.

        Sans elle, chaque deplacement de la vue relisait les 208 fichiers du
        cache — 1,95 Go decompresses pour produire 30 Ko de PNG, une seconde a
        chaque geste. Le resultat ne depend que de `overview_signature`.
        """
        signature = self.overview_signature(zones, kind)
        with self.overview_lock:
            if self.overview_cache.get('signature') == signature:
                return self.overview_cache['content']
        content = self._render_overview(zones, kind)
        with self.overview_lock:
            self.overview_cache = {'signature': signature, 'content': content}
        return content

    def _render_overview(self, zones, kind):
        # Only completed tiles; transparent means unprocessed, never zero.
        scale = min(1, 1800/self.detector.tiles.height, 1800/self.detector.tiles.width)
        result = Image.new('RGBA', (round(self.detector.tiles.width*scale), round(self.detector.tiles.height*scale)))
        for zone in zones:
            if not self.path(zone).exists():
                continue
            with np.load(self.path(zone)) as data:
                values = data[kind]
            width = max(1, round((zone['x']+zone['width'])*scale)-round(zone['x']*scale))
            height = max(1, round((zone['y']+zone['height'])*scale)-round(zone['y']*scale))
            if kind in {'mask', 'skeleton', 'centerline'}:
                # Maximum binning preserves thin axes in overview only.
                reduced = np.zeros((height, width), dtype=np.uint8)
                yy, xx = np.nonzero(values)
                reduced[np.minimum(height-1, yy*height//values.shape[0]),
                        np.minimum(width-1, xx*width//values.shape[1])] = 255
                tile = Image.fromarray(reduced)
            else:
                tile = Image.fromarray((np.clip(values, 0, 1)*255).astype('uint8')).resize((width, height), Image.Resampling.BILINEAR)
            result.paste(tile.convert('RGBA'), (round(zone['x']*scale), round(zone['y']*scale)))
        output = io.BytesIO()
        result.save(output, format='PNG')
        return output.getvalue()


MANIFEST = 'source.json'


def cache_key(source, max_analysis_side, factor=1.0):
    """Nom du dossier de cache d'une mosaique.

    La cle doit dependre de ce qui PRODUIT les tableaux, et de rien d'autre.
    Deux pieges corriges ici, chacun ayant deja coute des heures de recalcul :
      - la date de modification : un simple `touch`, ou une modification
        annulee, changeait la cle. On hache le contenu des fichiers de code,
        qui sont petits ; la mosaique, elle, reste identifiee par sa taille et
        sa date, on ne va pas hacher 750 Mo.
      - ce fichier entier : y toucher pour l'affichage ou le cache invalidait
        tout. Seule la source de `prepare` compte, c'est elle qui ecrit les
        tableaux.
    """
    from .detector_model import default_model_path
    model = default_model_path()
    code = [Path(__file__).with_name('detection_service.py'),
            Path(__file__).with_name('unet_model.py'),
            Path(__file__).with_name('detector_model.py'),
            Path(__file__).parent.parent/'crack_length_analysis_multi.py',
            Path(__file__).parent.parent/'crop_roi.py']
    signature = [(str(source), source.stat().st_size, source.stat().st_mtime_ns)]
    for path in [model] + code:
        if path.exists():
            signature.append(
                (path.name, hashlib.sha256(path.read_bytes()).hexdigest()))
    import inspect
    signature.append(
        ('prepare', hashlib.sha256(
            inspect.getsource(AnalysisLayers.prepare).encode()).hexdigest()))
    signature.append(('side', max_analysis_side))
    # Seulement pour une image vue a une autre echelle : la cle des mosaiques
    # a la resolution de reference, et donc leur cache, ne changent pas.
    if factor != 1.0:
        signature.append(('factor', round(float(factor), 4)))
        signature.append(('echelle', hashlib.sha256(
            Path(__file__).with_name('echelle.py').read_bytes()).hexdigest()))
    from importlib.metadata import version, PackageNotFoundError
    for package in ('numpy','scipy','scikit-image','torch','pillow'):
        try:
            signature.append((package, version(package)))
        except PackageNotFoundError:
            signature.append((package, None))
    return hashlib.sha256(json.dumps(signature).encode()).hexdigest()[:20]


def purge_obsolete_caches(root, current, source):
    """Supprime les caches que la version en service ne relira jamais.

    Un reentrainement ou une retouche de la detection change la cle : l'ancien
    dossier, souvent plus d'un gigaoctet, restait sur le disque pour toujours.
    Ne sont retires que les dossiers vides et ceux de LA MEME image sous une
    ancienne cle ; les caches des autres mosaiques ne sont jamais touches, pas
    plus que les dossiers sans etiquette, dont on ignore l'origine.
    """
    for folder in Path(root).iterdir():
        if not folder.is_dir() or folder == current:
            continue
        try:
            if not any(folder.iterdir()):
                folder.rmdir()
                continue
            manifest = folder/MANIFEST
            if not manifest.exists():
                continue
            image = json.loads(manifest.read_text(encoding='utf-8')).get('image')
            if image and os.path.normcase(image) == os.path.normcase(str(source)):
                shutil.rmtree(folder)
        except (OSError, ValueError) as error:
            print(f'Cache {folder.name} non purge : {error}', flush=True)
