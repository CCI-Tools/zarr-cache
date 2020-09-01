import unittest

import numpy as np

from zarr_cache._cache import DefaultStoreCache
from zarr_cache.indexes import MemoryStoreIndex


class DefaultStoreCacheTest(unittest.TestCase):

    def test_get_value(self):
        cached_stores = dict()

        def store_opener(store_id: str):
            cached_store = dict()
            if store_id == 's2':
                cached_store['k1'] = np.array([1, 2, 3])
            cached_stores[store_id] = cached_store
            return cached_store

        store_index = MemoryStoreIndex()
        store_cache = DefaultStoreCache(store_index, store_opener)

        with self.assertRaises(KeyError):
            store_cache.get_value('s1', 'k1')

        self.assertEqual([1, 2, 3], list(store_cache.get_value('s2', 'k1')))

        self.assertEqual(2, len(cached_stores))
        self.assertIn('s1', cached_stores)
        self.assertIn('s2', cached_stores)

    # noinspection PyTypeChecker
    def test_put_value(self):
        # noinspection PyUnusedLocal
        def store_opener(store_id: str):
            return dict()

        store_index = MemoryStoreIndex()
        store_cache = DefaultStoreCache(store_index, store_opener)

        store_cache.put_value('s1', 'k1', np.array([1, 2, 3]))
        store_cache.put_value('s1', 'k2', np.array([4, 5, 6]))

        self.assertEqual([1, 2, 3], list(store_cache.get_value('s1', 'k1')))
        self.assertEqual([4, 5, 6], list(store_cache.get_value('s1', 'k2')))

        store_cache.put_value('s2', 'k1', np.array([10, 20, 30]))
        store_cache.put_value('s2', 'k2', np.array([40, 50, 60]))

        self.assertEqual([1, 2, 3], list(store_cache.get_value('s1', 'k1')))
        self.assertEqual([4, 5, 6], list(store_cache.get_value('s1', 'k2')))
        self.assertEqual([10, 20, 30], list(store_cache.get_value('s2', 'k1')))
        self.assertEqual([40, 50, 60], list(store_cache.get_value('s2', 'k2')))

    # noinspection PyTypeChecker
    def test_has_value(self):
        # noinspection PyUnusedLocal
        def store_opener(store_id: str):
            return dict()

        store_index = MemoryStoreIndex()
        store_cache = DefaultStoreCache(store_index, store_opener)

        store_cache.put_value('s1', 'k1', np.array([1, 2, 3]))
        store_cache.put_value('s1', 'k2', np.array([4, 5, 6]))

        self.assertEqual(True, store_cache.has_value('s1', 'k1'))
        self.assertEqual(True, store_cache.has_value('s1', 'k2'))
        self.assertEqual(False, store_cache.has_value('s1', 'k3'))

    # noinspection PyTypeChecker
    def test_delete_value(self):
        s1, s2 = dict(), dict()

        # noinspection PyUnusedLocal
        def store_opener(store_id: str):
            return s1 if store_id == 's1' else s2

        store_index = MemoryStoreIndex()
        store_cache = DefaultStoreCache(store_index, store_opener)

        store_cache.put_value('s1', 'k1', np.array([1, 2, 3]))
        store_cache.put_value('s2', 'k2', np.array([4, 5, 6]))

        self.assertIn('k1', s1)
        self.assertIn('k2', s2)

        self.assertEqual(True, store_cache.delete_value('s1', 'k1'))
        self.assertEqual(False, store_cache.delete_value('s1', 'k2'))
        self.assertEqual(False, store_cache.delete_value('s2', 'k1'))
        self.assertEqual(True, store_cache.delete_value('s2', 'k2'))

        self.assertEqual({}, s1)
        self.assertEqual({}, s2)

    # noinspection PyTypeChecker
    def test_clear_store(self):
        class Store(dict):
            def __init__(self):
                super().__init__()
                self.closed = False

            def close(self):
                self.closed = True

        s1, s2 = Store(), Store()

        # noinspection PyUnusedLocal
        def store_opener(store_id: str):
            return s1 if store_id == 's1' else s2

        store_index = MemoryStoreIndex()
        store_cache = DefaultStoreCache(store_index, store_opener)

        store_cache.put_value('s1', 'k1', np.array([1, 2, 3]))
        store_cache.put_value('s2', 'k2', np.array([4, 5, 6]))

        self.assertEqual(1, len(s1))
        self.assertEqual(1, len(s2))

        self.assertEqual(False, s1.closed)
        self.assertEqual(False, s2.closed)
        store_cache.clear_store('s1')
        self.assertEqual({}, s1)
        self.assertEqual(True, s1.closed)
        self.assertEqual(False, s2.closed)
        store_cache.clear_store('s2')
        self.assertEqual({}, s2)
        self.assertEqual(True, s1.closed)
        self.assertEqual(True, s2.closed)

    # noinspection PyTypeChecker
    def test_close_store(self):
        class Store(dict):
            def __init__(self):
                super().__init__()
                self.closed = False

            def close(self):
                self.closed = True

        s1, s2 = Store(), Store()

        # noinspection PyUnusedLocal
        def store_opener(store_id: str):
            return s1 if store_id == 's1' else s2

        store_index = MemoryStoreIndex()
        store_cache = DefaultStoreCache(store_index, store_opener)

        store_cache.put_value('s1', 'k1', np.array([1, 2, 3]))
        store_cache.put_value('s2', 'k2', np.array([4, 5, 6]))

        self.assertEqual(False, s1.closed)
        self.assertEqual(False, s2.closed)
        store_cache.close_store('s1')
        self.assertEqual(True, s1.closed)
        self.assertEqual(False, s2.closed)
        store_cache.close_store('s2')
        self.assertEqual(True, s1.closed)
        self.assertEqual(True, s2.closed)
