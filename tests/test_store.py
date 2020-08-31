import unittest

import xarray as xr
import zarr.storage

from tests.helpers import make_test_cube
from zarr_cache import CachedStore
from zarr_cache.indexes import MemoryStoreIndex
from zarr_cache.openers import new_memory_store_opener


class StoreCacheStoreTest(unittest.TestCase):

    def test_it(self):
        cube = make_test_cube()
        cube = cube.chunk(dict(time=5, lat=60, lon=120))

        original_store = dict()
        cube.to_zarr(original_store)

        cube_key_count = len(original_store)
        cube_size = 0
        for k in original_store.keys():
            cube_size += zarr.storage.getsize(original_store, path=k)

        cached_stores = dict()

        store_opener = new_memory_store_opener(store_collection=cached_stores)

        store_index = MemoryStoreIndex(max_size=cube_size // 2)
        cache_store = CachedStore(original_store, 'my_store', store_index, store_opener)

        self.assertEqual(0, cache_store.misses)
        self.assertEqual(0, cache_store.hits)

        cached_cube = xr.open_zarr(cache_store)

        self.assertTrue(cache_store.misses < cube_key_count // 2)
        self.assertTrue(cache_store.hits < cube_key_count // 2)

        output_store = dict()
        cached_cube.to_zarr(output_store)

        self.assertTrue(cache_store.misses > cube_key_count // 2)
        self.assertTrue(cache_store.hits > cube_key_count // 2)

        self.assertEqual(set(original_store.keys()), set(output_store.keys()))

        cache_store.close()
