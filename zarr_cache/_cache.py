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
import warnings

import zarr.storage

from ._index import StoreIndex
from ._opener import StoreOpener
from .util import close_store


class StoreCache(abc.ABC):
    """
    A cache for a collection of stores.
    """

    @abc.abstractmethod
    def has_value(self, store_id: str, key: str) -> bool:
        """
        Tests whether a value exists.

        :param store_id: The store identifier.
        :param key: The key.
        :return: True, if the is a value for the given key.
        """

    @abc.abstractmethod
    def get_value(self, store_id: str, key: str) -> bytes:
        """
        Get a value from the cache.

        :param store_id: The store identifier.
        :param key: The key.
        :return: The value.
        :raise ValueError: If the key does not exists.
        """

    @abc.abstractmethod
    def put_value(self, store_id: str, key: str, value: bytes):
        """
        Put a value into the cache.

        :param store_id: The store identifier.
        :param key: The key.
        :param value: The value.
        """

    @abc.abstractmethod
    def delete_value(self, store_id: str, key: str) -> bool:
        """
        Delete a value from the cache.

        :param store_id: The store identifier.
        :param key: The key.
        :return: True, if the value existed and could be deleted, False otherwise.
        """

    @abc.abstractmethod
    def clear_store(self, store_id: str):
        """
        Clear the store given by *store_id*. Removes the given store entirely.

        :param store_id: The store identifier.
        """

    @abc.abstractmethod
    def close_store(self, store_id: str):
        """
        Clear the store given by *store_id*.

        :param store_id: The store identifier.
        """


class DefaultStoreCache(StoreCache):
    """
    A StoreCache implementation that uses
    a store index to manage a cache's keys and
    a store opener to open cached stores from their external representation.

    Manipulations of the opened stores and the index are synchronized via a mutex.
    Note that the keys in the index and the key-values in the external store may still run out of sync
    if manipulated by multiple concurrent processes.

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

    def put_value(self, store_id: str, key: str, value: bytes):
        # print(f'put_value: key={key}, typeof value={type(value)}')
        value_size = zarr.storage.buffer_size(value)
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

    def clear_store(self, store_id: str):
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
