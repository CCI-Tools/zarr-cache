import unittest

import xarray as xr

from tests.helpers import make_test_cube
from zarr_cache.cache import StoreCache
from zarr_cache.indexes import MemoryStoreIndex
from zarr_cache.store import StoreCacheStore


class StoreCacheStoreTest(unittest.TestCase):

    def test_it(self):
        cube = make_test_cube()
        cube = cube.chunk(dict(time=5, lat=60, lon=120))

        original_store = dict()
        cube.to_zarr(original_store)

        cached_stores = dict()

        def store_opener(store_id: str):
            cached_store = dict()
            cached_stores[store_id] = cached_store
            return cached_store

        store_index = MemoryStoreIndex()
        cache_store = StoreCacheStore(original_store, 'my_store', StoreCache(store_index,
                                                                             store_opener))

        cached_cube = xr.open_zarr(cache_store)

        output_store = dict()
        cached_cube.to_zarr(output_store)

        self.assertEqual(set(original_store.keys()), set(output_store.keys()))

        cache_store.close()
