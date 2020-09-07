import math
import pickle
import unittest

import xarray as xr
import zarr.storage

from tests import helpers
from tests.helpers import make_test_cube
from zarr_cache import CachedStore
from zarr_cache import IndexedCacheStorage
from zarr_cache import MemoryCacheStorage
from zarr_cache import MemoryStoreIndex
from zarr_cache import MemoryStoreOpener


class CachedStoreTest(unittest.TestCase):

    def setUp(self) -> None:
        self.cache = dict()
        self.store = CachedStore(helpers.make_test_store(), 'test', MemoryCacheStorage(self.cache))

    def test_len(self):
        self.assertEqual(8, len(self.store))

    def test_iter(self):
        self.assertEqual(['.zgroup',
                          '.zattrs',
                          'var1/.zarray',
                          'var1/.zattrs',
                          'var1/0',
                          'var2/.zarray',
                          'var2/.zattrs',
                          'var2/0'], list(iter(self.store)))

    def test_contains(self):
        self.assertTrue('.zgroup' in self.store)
        self.assertFalse('bibo' in self.store)

    def test_keys(self):
        self.assertEqual(['.zgroup',
                          '.zattrs',
                          'var1/.zarray',
                          'var1/.zattrs',
                          'var1/0',
                          'var2/.zarray',
                          'var2/.zattrs',
                          'var2/0'], list(self.store.keys()))

    def test_clear(self):
        self.store.clear()
        self.assertEqual([], list(self.store.keys()))

    def test_close(self):
        self.store.close()

    def test_get_item(self):
        value = self.store['var1/0']
        self.assertIsNotNone(value)
        self.assertIn('test', self.cache)
        self.assertIn('var1/0', self.cache['test'])

    def test_get_item_with_filter(self):
        # noinspection PyUnusedLocal
        def filter(*args, **kargs) -> bool:
            return False

        # noinspection PyTypeChecker
        store = CachedStore(helpers.make_test_store(), 'test', MemoryCacheStorage(self.cache),
                            store_item_filter=filter)
        value = store['var1/0']
        self.assertIsNotNone(value)
        self.assertNotIn('test', self.cache)

    def test_get_item_invalid_key(self):
        with self.assertRaises(KeyError):
            # noinspection PyStatementEffect
            self.store['x']

    def test_set_item(self):
        value = bytes('{"zarr_format": 2}', encoding='utf8')
        self.store['var3/.zarray'] = value
        self.assertNotIn('test', self.cache)
        retrieved_value = self.store['var3/.zarray']
        self.assertIs(value, retrieved_value)
        self.assertIn('test', self.cache)
        self.assertIn('var3/.zarray', self.cache['test'])
        self.assertIs(retrieved_value, self.cache['test']['var3/.zarray'])

        value = bytes('{\n  "zarr_format": 2\n}', encoding='utf8')
        self.store['var3/.zarray'] = value
        retrieved_value = self.store['var3/.zarray']
        self.assertIs(value, retrieved_value)
        self.assertIn('test', self.cache)
        self.assertIn('var3/.zarray', self.cache['test'])
        self.assertIs(retrieved_value, self.cache['test']['var3/.zarray'])

    def test_del_item(self):
        value = bytes('{"zarr_format": 2}', encoding='utf8')
        del self.store['var1/.zarray']
        # Force 'var1/.zattrs' to be in cache
        value = self.store['var1/.zattrs']
        self.assertIn('test', self.cache)
        self.assertIn('var1/.zattrs', self.cache['test'])
        del self.store['var1/.zattrs']
        self.assertIn('test', self.cache)
        self.assertNotIn('var1/.zattrs', self.cache['test'])

    def test_getsize(self):
        self.assertEqual(12, self.store.getsize('var1/0'))

    def test_serialization(self):
        data = pickle.dumps(self.store)
        loaded_cache_store = pickle.loads(data)
        self.assertIsInstance(loaded_cache_store, CachedStore)

    def test_practical_use_with_xarray(self):
        cube = make_test_cube()
        cube = cube.chunk(dict(time=5, lat=60, lon=120))

        original_store = dict()
        cube.to_zarr(original_store)

        cube_key_count = len(original_store)
        cube_size = 0
        for k in original_store.keys():
            cube_size += zarr.storage.getsize(original_store, path=k)

        cached_stores = dict()

        store_opener = MemoryStoreOpener(stores=cached_stores)

        store_index = MemoryStoreIndex(max_size=cube_size // 2)
        cache_store = CachedStore(original_store, 'my_store', IndexedCacheStorage(store_index, store_opener))

        self.assertEqual(0, cache_store.hit_count)
        self.assertEqual(0, cache_store.miss_count)
        self.assertTrue(math.isnan(cache_store.hit_latency))
        self.assertTrue(math.isnan(cache_store.miss_latency))

        cached_cube = xr.open_zarr(cache_store)

        self.assertTrue(cache_store.miss_count < cube_key_count // 2)
        self.assertTrue(cache_store.hit_count < cube_key_count // 2)
        self.assertTrue(cache_store.hit_latency > 0.0)
        self.assertTrue(cache_store.miss_latency > 0.0)

        output_store = dict()
        cached_cube.to_zarr(output_store)

        self.assertTrue(cache_store.miss_count > cube_key_count // 2)
        self.assertTrue(cache_store.hit_count > cube_key_count // 2)
        self.assertTrue(cache_store.hit_latency > 0.0)
        self.assertTrue(cache_store.miss_latency > 0.0)

        self.assertEqual(set(original_store.keys()), set(output_store.keys()))

        cache_store.close()
