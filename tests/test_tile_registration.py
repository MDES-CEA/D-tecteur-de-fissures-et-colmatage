"""Coordinate and cache-identity contracts for the image pyramid."""
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from PIL import Image
from annotation_app.tile_service import PyramidTileService


def release(service):
    """Libere le fichier avant que le dossier temporaire soit efface.

    Sous Windows un `np.memmap` vivant verrouille le fichier : fermer la seule
    image PIL ne suffit pas, et `TemporaryDirectory.cleanup()` echoue.
    """
    service.source.image.close()
    memmap=getattr(service.source,'_memmap',None)
    if memmap is not None: memmap._mmap.close()


class TileRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.folder=TemporaryDirectory();self.addCleanup(self.folder.cleanup)
        y,x=np.indices((2053,1539))
        self.pixels=((x//17*13+y//23*19)%256).astype('uint8')
        self.path=Path(self.folder.name)/'coordinates.tif'
        Image.fromarray(self.pixels).save(self.path,compression='raw')
        self.tiles=PyramidTileService(self.path)
        self.addCleanup(release,self.tiles)

    def test_png_pixels_correspond_to_source_coordinates_at_each_level(self):
        for level in range(self.tiles.max_level+1):
            scale=self.tiles.level_scale(level)
            width,height=self.tiles.level_dimensions(level)
            # Test edges as well as complete tiles: odd dimensions must not
            # change the source origin or shift the final partial tile.
            for tx,ty in {(0,0),((width-1)//512,(height-1)//512)}:
                actual=np.asarray(Image.open(io.BytesIO(self.tiles.tile_png(level,tx,ty))))
                x0,y0=tx*512*scale,ty*512*scale
                expected=self.pixels[y0:min(y0+512*scale,self.pixels.shape[0]):scale,
                                     x0:min(x0+512*scale,self.pixels.shape[1]):scale]
                np.testing.assert_array_equal(actual,expected)

    def test_cache_identity_is_stable_but_distinguishes_images_and_tile_layout(self):
        same=PyramidTileService(self.path);self.addCleanup(release,same)
        other_path=Path(self.folder.name)/'other.tif'
        Image.fromarray(self.pixels).save(other_path,compression='raw')
        other=PyramidTileService(other_path);self.addCleanup(release,other)
        different_size=PyramidTileService(self.path,tile_size=256);self.addCleanup(release,different_size)
        self.assertEqual(self.tiles.cache_key,same.cache_key)
        self.assertNotEqual(self.tiles.cache_key,other.cache_key)
        self.assertNotEqual(self.tiles.cache_key,different_size.cache_key)
        self.assertEqual(self.tiles.metadata()['tile_cache_key'],self.tiles.cache_key)


if __name__=='__main__':unittest.main()
