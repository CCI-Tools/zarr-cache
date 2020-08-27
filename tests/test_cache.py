import unittest

import numpy as np

from zarr_cache.cache import StoreCache
from zarr_cache.indexes import MemoryStoreIndex


class StoreCacheTest(unittest.TestCase):

    def test_it(self):
        cached_stores = dict()

        def store_opener(store_id: str):
            cached_store = dict()
            cached_stores[store_id] = cached_store
            return cached_store

        store_index = MemoryStoreIndex()
        store_cache = StoreCache(store_index, store_opener)

        default_store = dict(k1=np.array([1, 2, 3]))

        r = store_cache.get_value('s1', 'k1', default_store)
        self.assertEqual([1, 2, 3], list(r))
        self.assertEqual((0, 1), (store_cache.hits, store_cache.misses))

        r = store_cache.get_value('s1', 'k1', default_store)
        self.assertEqual([1, 2, 3], list(r))
        self.assertEqual((1, 1), (store_cache.hits, store_cache.misses))

        store_cache.put_value('s1', 'k2', np.array([4, 5, 6, 7]))
        self.assertEqual((1, 1), (store_cache.hits, store_cache.misses))

        r = store_cache.get_value('s1', 'k2', default_store)
        self.assertEqual([4, 5, 6, 7], list(r))
        self.assertEqual((2, 1), (store_cache.hits, store_cache.misses))
