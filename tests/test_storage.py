import abc
import pickle
import unittest

import numpy as np

from zarr_cache import IndexedCacheStorage, CacheStorage, StoreOpener
from zarr_cache import MemoryCacheStorage
from zarr_cache import MemoryStoreIndex
from zarr_cache import MemoryStoreOpener


class ClosableStore(dict):
    def __init__(self):
        super().__init__()
        self.closed = False

    def close(self):
        self.closed = True


# noinspection PyUnresolvedReferences
class AbstractCacheStorageTest(abc.ABC):
    @abc.abstractmethod
    def new_cache_storage(self, store_opener: StoreOpener = None) -> CacheStorage:
        pass

    def test_serialization(self):
        storage = self.new_cache_storage()
        data = pickle.dumps(storage)
        obj = pickle.loads(data)
        self.assertIsInstance(obj, type(self.new_cache_storage()))

    def test_has_get_put_value(self):
        storage = self.new_cache_storage()

        self.assertEqual(False, storage.has_value('store_1', 'key_1'))
        with self.assertRaises(KeyError):
            storage.get_value('store_1', 'key_1')

        # put
        storage.put_value('store_1', 'key_1', np.array([1, 2, 3]))
        self.assertEqual(True, storage.has_value('store_1', 'key_1'))
        self.assertEqual([1, 2, 3], list(storage.get_value('store_1', 'key_1')))

        # overwrite
        storage.put_value('store_1', 'key_1', np.array([4, 5, 6]))
        self.assertEqual(True, storage.has_value('store_1', 'key_1'))
        self.assertEqual([4, 5, 6], list(storage.get_value('store_1', 'key_1')))

        # put another one
        storage.put_value('store_1', 'key_2', np.array([1, 2, 3]))
        self.assertEqual(True, storage.has_value('store_1', 'key_2'))
        self.assertEqual([1, 2, 3], list(storage.get_value('store_1', 'key_2')))

    # noinspection PyTypeChecker
    def test_delete_value(self):
        storage = self.new_cache_storage()

        storage.put_value('store_1', 'key_1', np.array([1, 2, 3]))
        storage.put_value('store_2', 'key_2', np.array([4, 5, 6]))

        self.assertEqual(True, storage.delete_value('store_1', 'key_1'))
        self.assertEqual(False, storage.delete_value('store_1', 'key_2'))
        self.assertEqual(False, storage.delete_value('store_2', 'key_1'))
        self.assertEqual(True, storage.delete_value('store_2', 'key_2'))

    # noinspection PyTypeChecker
    def test_clear_store(self):
        store_1, store_2 = ClosableStore(), ClosableStore()

        def store_opener(store_id: str):
            return store_1 if store_id == 'store_1' else store_2

        storage = self.new_cache_storage(store_opener=store_opener)

        storage.put_value('store_1', 'key_1', np.array([1, 2, 3]))
        storage.put_value('store_2', 'key_2', np.array([4, 5, 6]))

        self.assertEqual(True, storage.has_value('store_1', 'key_1'))
        self.assertEqual(False, store_1.closed)
        storage.delete_store('store_1')
        self.assertEqual(True, store_1.closed)
        self.assertEqual(False, storage.has_value('store_1', 'key_1'))

        self.assertEqual(True, storage.has_value('store_2', 'key_2'))
        self.assertEqual(False, store_2.closed)
        storage.delete_store('store_2')
        self.assertEqual(True, store_2.closed)
        self.assertEqual(False, storage.has_value('store_2', 'key_2'))

        # Assert it does not raise
        storage.delete_store('store_3')

    # noinspection PyTypeChecker
    def test_close_store(self):
        store_1, store_2 = ClosableStore(), ClosableStore()

        def store_opener(store_id: str):
            return store_1 if store_id == 'store_1' else store_2

        storage = self.new_cache_storage(store_opener=store_opener)

        storage.put_value('store_1', 'key_1', np.array([1, 2, 3]))
        storage.put_value('store_2', 'key_2', np.array([4, 5, 6]))

        self.assertEqual(False, store_1.closed)
        self.assertEqual(False, store_2.closed)
        storage.close_store('store_1')
        self.assertEqual(True, store_1.closed)
        self.assertEqual(False, store_2.closed)
        storage.close_store('store_2')
        self.assertEqual(True, store_1.closed)
        self.assertEqual(True, store_2.closed)


class IndexedCacheStorageTest(AbstractCacheStorageTest, unittest.TestCase):
    def new_cache_storage(self, store_opener=None) -> CacheStorage:
        return IndexedCacheStorage(MemoryStoreIndex(),
                                   store_opener or MemoryStoreOpener())

    # noinspection PyUnresolvedReferences
    def test_specific_properties(self):
        store_opener = MemoryStoreOpener()
        storage = self.new_cache_storage(store_opener=store_opener)
        self.assertIsNotNone(storage.store_index)
        self.assertIs(store_opener, storage.store_opener)


class MemoryCacheStorageTest(AbstractCacheStorageTest, unittest.TestCase):
    def new_cache_storage(self, store_opener=None) -> CacheStorage:
        return MemoryCacheStorage(store_opener=store_opener)

    # noinspection PyUnresolvedReferences
    def test_specific_properties(self):
        store_opener = MemoryStoreOpener()
        storage = self.new_cache_storage(store_opener=store_opener)
        self.assertEqual({}, storage.stores)
        self.assertIs(store_opener, storage.store_opener)
