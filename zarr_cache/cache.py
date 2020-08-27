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

import collections
import threading
import warnings
from typing import Callable, Any

import zarr.storage

from zarr_cache.index import StoreIndex

Store = collections.MutableMapping
StoreOpener = Callable[[str], Store]


# Note: Implementation borrowed partly from zarr.storage.LRUStoreCache

class StoreCache:
    """
    A cache for a collection of stores.

    :param store_index: The store index.
    :param store_opener: A callable that receives a *store_id* and returns a *store*.
    """

    def __init__(self,
                 store_index: StoreIndex,
                 store_opener: StoreOpener):
        self._store_index = store_index
        self._store_opener = store_opener
        self._open_stores = {}
        self._hits = 0
        self._misses = 0
        self._lock = threading.RLock()

    def __getstate__(self):
        return (
            self._store_index,
            self._store_opener,
            self._open_stores,
            self._hits,
            self._misses
        )

    def __setstate__(self, state):
        (
            self._store_index,
            self._store_opener,
            self._open_stores,
            self._hits,
            self._misses
        ) = state
        self._lock = threading.RLock()

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    def get_value(self, store_id: str, key: str, default_store: Store) -> bytes:
        try:
            # first try to obtain the value from the cache
            with self._lock:
                store = self._get_store(store_id)
                value = store[key]
                # cache hit if no KeyError is raised
                self._hits += 1
                # treat the end as most recently used
                self._store_index.mark_key(store_id, key)
        except KeyError:
            # cache miss, retrieve value from the store
            value = default_store[key]
            with self._lock:
                self._misses += 1
                # need to check if key is not in the cache, as it may have been cached
                # while we were retrieving the value from the store
                store = self._get_store(store_id)
                if key not in store:
                    self.put_value(store_id, key, value)
        return value

    def put_value(self, store_id: str, key: str, value: bytes):
        # print(f'put_value: key={key}, typeof value={type(value)}')
        value_size = zarr.storage.buffer_size(value)
        # check size of the value against max size, as if the value itself exceeds max
        # size then we are never going to cache it
        max_size = self._store_index.max_size
        if max_size is None or value_size <= max_size:
            with self._lock:
                self._accommodate_space(value_size)
                store = self._get_store(store_id)
                store[key] = value
                self._store_index.push_key(store_id, key, value_size)

    def delete_value(self, store_id: str, key: str):
        with self._lock:
            store = self._get_store(store_id)
            if key in store:
                del store[key]
            self._store_index.delete_key(store_id, key)

    def clear_store(self, store_id: str):
        with self._lock:
            store = self._get_store(store_id)
            try:
                store.clear()
                self._store_index.delete_keys(store_id)
            finally:
                close_store(store)

    def close_store(self, store_id: str):
        with self._lock:
            if store_id in self._open_stores:
                store = self._open_stores[store_id]
                del self._open_stores[store_id]
                close_store(store)

    def _get_store(self, store_id: str) -> Store:
        if store_id not in self._open_stores:
            store = self._store_opener(store_id)
            self._open_stores[store_id] = store
        return self._open_stores[store_id]

    def _accommodate_space(self, new_size: int):
        max_size = self._store_index.max_size
        if max_size is None:
            return
        current_size = self._store_index.current_size
        # ensure there is enough space in the cache for a new value
        while current_size + new_size > max_size:
            store_id, key, size = self._store_index.pop_key()
            store = self._get_store(store_id)
            try:
                del store[key]
            except KeyError:
                warnings.warn(f'Failed to delete key {key} from store {store_id}')
                continue
            current_size -= size


def close_store(store):
    if hasattr(store, 'close') and callable(getattr(store, 'close')):
        store.close()
