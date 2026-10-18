import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
from annotation_app.analysis_service import AnalysisLayers


class FakeDatabase:
    def list_zones(self, image_id):
        return [dict(id=1,x=0,y=0,width=20,height=20,status='done'),
                dict(id=2,x=20,y=0,width=20,height=20,status='todo')]


class AnalysisIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.folder=TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        source=Path(self.folder.name)/'image.tif'
        source.write_bytes(b'source fingerprint')
        values=np.eye(20,dtype=np.float32)
        tiles=SimpleNamespace(image_path=source,width=40,height=20,source=SimpleNamespace(direct=False))
        self.calls=0
        def analyze(zone):
            self.calls+=1
            return SimpleNamespace(score=values,cost=values,mask=values>0,skeleton=values>0,
                                   scale=1,cracks=[{'geometry':[[zone['x'],0],[zone['x']+19,19]]}])
        self.detector=SimpleNamespace(tiles=tiles,max_analysis_side=1000,analyze=analyze,
                                     prepare_route=lambda zone:SimpleNamespace(cost=values))

    def test_persistent_arrays_and_transparent_unprocessed_area(self):
        layers=AnalysisLayers(self.detector,Path(self.folder.name)/'layers')
        zones=FakeDatabase().list_zones('test')
        layers.prepare(zones[0]); layers.prepare(zones[0])
        self.assertEqual(self.calls,1)
        another=AnalysisLayers(self.detector,Path(self.folder.name)/'layers')
        another.prepare(zones[0])
        self.assertEqual(self.calls,1)
        image=Image.open(io.BytesIO(layers.overview(zones,'skeleton')))
        self.assertEqual(image.getpixel((25,5))[3],0)
        self.assertEqual(image.getpixel((5,5))[3],255)
        for kind in layers.KINDS:
            self.assertEqual(Image.open(io.BytesIO(layers.png(zones[0],kind))).size,(20,20))

    def test_cached_proposals_come_back_without_new_analysis(self):
        layers=AnalysisLayers(self.detector,Path(self.folder.name)/'layers')
        zone=FakeDatabase().list_zones('test')[0]
        self.assertIsNone(layers.cached_proposals(zone))
        layers.prepare(zone)
        cached=layers.cached_proposals(zone)
        self.assertIsNotNone(cached)
        self.assertEqual(cached[0][0]['geometry'],[[0,0],[19,19]])
        self.assertEqual(self.calls,1)

    def test_obsolete_caches_of_this_image_only_are_purged(self):
        root=Path(self.folder.name)/'purge'
        def folder(name, image=None, files=1):
            path=root/name; path.mkdir(parents=True)
            for index in range(files): (path/f'{index}.npz').write_bytes(b'x')
            if image: (path/'source.json').write_text(json.dumps({'image':str(image)}))
            return path
        source=self.detector.tiles.image_path
        old=folder('ancienne-version', source)
        empty=folder('vide', files=0)
        other=folder('autre-mosaique', Path(self.folder.name)/'autre.tif')
        unlabelled=folder('sans-etiquette')
        layers=AnalysisLayers(self.detector,root)
        self.assertFalse(old.exists())
        self.assertFalse(empty.exists())
        self.assertTrue(other.exists())
        self.assertTrue(unlabelled.exists())
        self.assertTrue((layers.directory/'source.json').exists())
        AnalysisLayers(self.detector,root)
        self.assertTrue(layers.directory.exists())


if __name__=='__main__':
    unittest.main()
