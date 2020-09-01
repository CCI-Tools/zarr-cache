import os
import os.path
import shutil
import unittest

import xarray as xr

from tests.helpers import make_test_cube

STORE_NAME = 'store.zarr'


@unittest.skip('Run this to check how xarray opens a zarr that comprises groups only')
class ZarrTest(unittest.TestCase):

    def setUp(self) -> None:
        self.clearFiles()

    def tearDown(self) -> None:
        self.clearFiles()

    def clearFiles(self) -> None:
        if os.path.exists(STORE_NAME):
            shutil.rmtree(STORE_NAME)

    def test_it(self):
        """Test opening a multi-group Zarr using xarray.open_zarr()"""

        cube = make_test_cube()

        cube = cube.chunk(dict(time=5, lat=60, lon=120))
        os.mkdir(STORE_NAME)
        with open(f'{STORE_NAME}/.zgroup', 'w') as fp:
            fp.write('{"zarr_format": 2}')
        with open(f'{STORE_NAME}/.zattrs', 'w') as fp:
            fp.write('{}')
        cube.to_zarr(f'{STORE_NAME}/cube-1.zarr')
        cube.to_zarr(f'{STORE_NAME}/cube-2.zarr')
        cube.to_zarr(f'{STORE_NAME}/cube-3.zarr')

        ds = xr.open_zarr(STORE_NAME)

        self.assertEqual((), tuple(ds.dims))
        self.assertEqual([], list(ds.data_vars))
