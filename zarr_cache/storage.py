# The MIT License (MIT)
# Copyright (c) 2020 by the ESA CCI Toolbox development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import abc
import collections
import threading
import time
import warnings
from builtins import property
from typing import Dict, Optional, Tuple

import zarr.storage

from .index import StoreIndex
from .opener import MemoryStoreOpener
from .opener import StoreOpener
from .util import close_store

_NAN = float('nan')


class CacheStorage(abc.ABC):
    """
    Represents the storage for a cache for multiple stores.
    """

    @abc.abstractmethod
    def has_value(self, store_id: str, key: str) -> bool:
        """
        Tests whether a value exists in a store in the cache storage.

        :param store_id: The store identifier.
        :param key: The key.
        :return: True, if the is a value for the given key.
        """

    @abc.abstractmethod
    def get_value(self, store_id: str, key: str) -> bytes:
        """
        Get a value from a store in the cache storage.

        :param store_id: The store identifier.
        :param key: The key.
        :return: The value.
        :raise KeyError: If the key does not exists.
        """

    @abc.abstractmethod
    def put_value(self, store_id: str, key: str, value: bytes, value_size: int = None):
        """
        Put a value into a store in the cache storage.

        :param store_id: The store identifier.
        :param key: The key.
        :param value: The value.
        :param value_size: Optional value size, may be passed for optimization, if known.
        """

    @abc.abstractmethod
    def delete_value(self, store_id: str, key: str) -> bool:
        """
        Delete a value from a store in the cache storage.

        :param store_id: The store identifier.
        :param key: The key.
        :return: True, if the value existed and could be deleted, False otherwise.
        """

    @abc.abstractmethod
    def delete_store(self, store_id: str):
        """
        Clear the store given by *store_id* in the cache storage.
        Removes the given store entirely.

        :param store_id: The store identifier.
        """

    @abc.abstractmethod
    def close_store(self, store_id: str):
        """
        Clear the store given by *store_id*.

        :param store_id: The store identifier.
        """


class CacheStorageDecorator(CacheStorage, abc.ABC):
    """
    A CacheStorage decorator wraps another (decorated) cache store to add some extra behaviour.

    :param decorated_storage: The decorated cache storage.
    """

    def __init__(self, decorated_storage: CacheStorage):
        self._decorated_storage = decorated_storage

    @property
    def decorated_storage(self):
        return self._decorated_storage

    def has_value(self, store_id: str, key: str) -> bool:
        return self.decorated_storage.has_value(store_id, key)

    def get_value(self, store_id: str, key: str) -> bytes:
        return self.decorated_storage.get_value(store_id, key)

    def put_value(self, store_id: str, key: str, value: bytes, value_size: int = None):
        self.decorated_storage.put_value(store_id, key, value, value_size=value_size)

    def delete_value(self, store_id: str, key: str) -> bool:
        return self.decorated_storage.delete_value(store_id, key)

    def delete_store(self, store_id: str):
        self.decorated_storage.delete_store(store_id)

    def close_store(self, store_id: str):
        self.decorated_storage.close_store(store_id)


class MemoryCacheStorage(CacheStorage):
    """
    A CacheStorage implementation that uses plain dictionary instances as storage.
    Should be used for testing only.

    :param stores: optional dictionary that will hold the stores.
    :param store_opener: Optional store opener. Must be a callable that receives a *store_id* and
        returns a ``collections.MutableMapping``. If not given, new dictionary instances will be created.
    """

    def __init__(self,
                 stores: Dict[str, Dict[str, bytes]] = None,
                 store_opener: StoreOpener = None):
        self._stores: Dict[str, Dict[str, bytes]] = stores if stores is not None else dict()
        self._store_opener = store_opener or MemoryStoreOpener()

    @property
    def stores(self) -> Optional[collections.MutableMapping]:
        return self._stores

    @property
    def store_opener(self) -> Optional[StoreOpener]:
        return self._store_opener

    def has_value(self, store_id: str, key: str) -> bool:
        return store_id in self._stores and key in self._stores[store_id]

    def get_value(self, store_id: str, key: str) -> bytes:
        return self._stores[store_id][key]

    def put_value(self, store_id: str, key: str, value: bytes, value_size: int = None):
        if store_id not in self._stores:
            self._stores[store_id] = self._store_opener(store_id) if self._store_opener is not None else dict()
        self._stores[store_id][key] = value

    def delete_value(self, store_id: str, key: str) -> bool:
        try:
            del self._stores[store_id][key]
            return True
        except KeyError:
            return False

    def delete_store(self, store_id: str):
        self.close_store(store_id)
        if store_id in self._stores:
            del self._stores[store_id]

    def close_store(self, store_id: str):
        if store_id in self._stores:
            close_store(self._stores[store_id])


class IndexedCacheStorage(CacheStorage):
    """
    A CacheStorage implementation that uses
    a store index to manage a cache's keys and
    a store opener to open stores from the cache storage.

    It is expected that the given *store_index* provides efficient means to
    iterate and lookup the used key in the cache. The opened stores
    from the *store_opener* are used to retrieve and store the actual values.

    Note that because the store opener opens partly cached stores,
    they may not represent valid Zarr groups, e.g. the ".zgroup" or ".zarray"
    keys may not (yet) be present.

    Manipulations of the opened stores and the index are synchronized via a mutex.
    The keys in the index and the key-values in the external store may still run
    out of sync if manipulated by multiple concurrent processes.

    :param store_index: The store index.
    :param store_opener: The store opener. Must be a callable that receives a *store_id* and
        returns a ``collections.MutableMapping``.
    """

    def __init__(self,
                 store_index: StoreIndex,
                 store_opener: StoreOpener):
        self._store_index = store_index
        self._store_opener = store_opener
        self._open_stores = {}
        # Fetch max_size as this will be frequently required
        self._max_size = self._store_index.max_size
        self._mutex = threading.Lock()

    @property
    def store_index(self) -> StoreIndex:
        return self._store_index

    @property
    def store_opener(self) -> StoreOpener:
        return self._store_opener

    def __getstate__(self):
        return (
            self._store_index,
            self._store_opener,
            self._open_stores,
            self._max_size,
        )

    def __setstate__(self, state):
        (
            self._store_index,
            self._store_opener,
            self._open_stores,
            self._max_size,
        ) = state
        self._mutex = threading.Lock()

    def has_value(self, store_id: str, key: str) -> bool:
        with self._mutex:
            return key in self._get_or_open_store(store_id)

    def get_value(self, store_id: str, key: str) -> bytes:
        # first try to obtain the value from the cache
        with self._mutex:
            store = self._get_or_open_store(store_id)
            value = store[key]
            # treat the end as most recently used
            self._store_index.mark_key(store_id, key)
            return value

    def put_value(self, store_id: str, key: str, value: bytes, value_size: int = None):
        # print(f'put_value: key={key}, typeof value={type(value)}')
        value_size = zarr.storage.buffer_size(value) if value_size is None else value_size
        # check size of the value against max size, as if the value itself exceeds max
        # size then we are never going to cache it
        if self._max_size is None or value_size <= self._max_size:
            with self._mutex:
                self._accommodate_space(value_size)
                store = self._get_or_open_store(store_id)
                store[key] = value
                self._store_index.push_key(store_id, key, value_size)

    def delete_value(self, store_id: str, key: str) -> bool:
        with self._mutex:
            res = False
            store = self._get_or_open_store(store_id)
            if key in store:
                del store[key]
                res = True
            self._store_index.delete_key(store_id, key)
            return res

    def delete_store(self, store_id: str):
        with self._mutex:
            store = self._get_or_open_store(store_id)
            try:
                store.clear()
                self._store_index.delete_store(store_id)
            finally:
                close_store(store)

    def close_store(self, store_id: str):
        with self._mutex:
            if store_id in self._open_stores:
                store = self._open_stores[store_id]
                del self._open_stores[store_id]
                close_store(store)

    def _get_or_open_store(self, store_id: str) -> collections.MutableMapping:
        if store_id not in self._open_stores:
            store = self._store_opener(store_id)
            self._open_stores[store_id] = store
        return self._open_stores[store_id]

    def _accommodate_space(self, new_size: int):
        if self._max_size is None:
            return
        current_size = self._store_index.current_size
        # ensure there is enough space in the cache for a new value
        while current_size + new_size > self._max_size:
            store_id, key, size = self._store_index.pop_key()
            store = self._get_or_open_store(store_id)
            try:
                del store[key]
            except KeyError:
                warnings.warn(f'Failed to delete key {key} from store {store_id}')
                continue
            current_size -= size


class TimingCacheStorage(CacheStorageDecorator):
    """
    A CacheStorage decorator that performs a timing on its get_value, put_value, and delete_value methods.

    :param decorated_storage: The decorated cache storage.
    """

    def __init__(self, decorated_storage: CacheStorage):
        super().__init__(decorated_storage)
        self._stats = dict(get_value=(0, 0, 0), put_value=(0, 0, 0), delete_value=(0, 0, 0))

    @property
    def get_value_stats(self) -> Tuple[float, float, float, int]:
        return self._get_stats('get_value')

    @property
    def put_value_stats(self) -> Tuple[float, float, float, int]:
        return self._get_stats('put_value')

    @property
    def delete_value_stats(self) -> Tuple[float, float, float, int]:
        return self._get_stats('delete_value')

    def _get_stats(self, method_name) -> Tuple[float, float, float, int]:
        time_sum, size_sum, count = self._stats[method_name]
        return size_sum / count if count > 0 else _NAN, \
               time_sum / count if count > 0 else _NAN, \
               size_sum / time_sum if time_sum > 0 else _NAN, \
               count

    def _update_stats(self, method_name, size_delta, time_delta):
        size_sum, time_sum, count = self._stats[method_name]
        self._stats[method_name] = size_sum + size_delta, time_sum + time_delta, count + 1

    def get_value(self, store_id: str, key: str) -> bytes:
        value_size = 0
        t0 = time.perf_counter()
        try:
            value = super().get_value(store_id, key)
            value_size = zarr.storage.buffer_size(value)
            return value
        finally:
            self._update_stats('get_value', value_size, time.perf_counter() - t0)

    def put_value(self, store_id: str, key: str, value: bytes, value_size: int = None):
        value_size = zarr.storage.buffer_size(value) if value_size is None else value_size
        t0 = time.perf_counter()
        try:
            super().put_value(store_id, key, value, value_size=value_size)
        finally:
            self._update_stats('put_value', value_size, time.perf_counter() - t0)

    def delete_value(self, store_id: str, key: str) -> bool:
        t0 = time.perf_counter()
        try:
            return super().delete_value(store_id, key)
        finally:
            self._update_stats('delete_value', 0, time.perf_counter() - t0)
