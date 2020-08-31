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
import time
from typing import List, Optional, AbstractSet, Callable, Union

import numpy as np
import zarr.storage

from .util import close_store

StoreItemFilter = Callable[[str, Union[bytes, np.ndarray], float], bool]


class CachedStore(collections.MutableMapping):
    """
    A Zarr key-value store that wraps (decorates) an original store so it can be cached in a
    multi-store cache whose keys are managed through the given *store_index*.

    :param original_store: The original store to be cached.
    :param cache_store: The original store to be cached.
    :param store_index: The store index.
    :param store_opener: Factory for writable cache stores.
    :param store_item_filter: Filter function that, if given, is called to decide whether a value should be cached or
        not. Its signature is ``store_item_filter(store_id, key, value, duration) -> bool``. If it returns True, a
        value will be cached. If *store_item_filter* is not given, values will always be cached.
    """

    def __init__(self,
                 original_store: collections.MutableMapping,
                 cache_store: collections.MutableMapping,
                 store_item_filter: StoreItemFilter = None):
        self._original_store = original_store
        self._cache_store = cache_store
        self._store_item_filter = store_item_filter
        self._cached_keys: Optional[AbstractSet[str]] = None
        self._lock = threading.RLock()

    def __getstate__(self):
        return (
            self._original_store,
            self._cache_store,
            self._store_item_filter,
            self._cached_keys,
        )

    def __setstate__(self, state):
        (
            self._original_store,
            self._cache_store,
            self._store_item_filter,
            self._cached_keys,
        ) = state
        self._lock = threading.RLock()

    def __len__(self):
        """Gets the number of keys in the original store."""
        return len(self._keys())

    def __iter__(self):
        """Iterates keys of the original store."""
        return iter(self._keys())

    def __contains__(self, key: str) -> bool:
        """Test whether *key* is in the original store."""
        return key in self._keys()

    def keys(self) -> AbstractSet[str]:
        """Get keys from the original store."""
        return self._keys()

    def clear(self):
        """Clear the original and cached store."""
        self._original_store.clear()
        with self._lock:
            self._cache_store.clear()
            self._invalidate_keys()

    def close(self):
        """Closes the original store and the cached store."""
        close_store(self._original_store)
        with self._lock:
            close_store(self._original_store)
            self._invalidate_keys()

    def __getitem__(self, key: str) -> bytes:
        """
        Get value for *key* either from the cached store or from the original.
        In the latter case, the value will be put into the cached store.
        :param key: The key.
        :return: An original or cached value.
        """
        try:
            return self._cache_store[key]
        except KeyError:
            if self._store_item_filter is not None:
                now = time.perf_counter()
                value = self._original_store[key]
                should_cache = self._store_item_filter(key, value, time.perf_counter() - now)
            else:
                value = self._original_store[key]
                should_cache = True
        if should_cache:
            with self._lock:
                self._cache_store[key] = value
                self._invalidate_keys()

    def __setitem__(self, key: str, value: bytes):
        """
        Set *key* to *value*.
        Sets the value in the original store.
        Sets the value in the cache store, only if the value was already cached.
        :param key: The key.
        :param value: The value
        """
        self._original_store[key] = value
        with self._lock:
            if key in self._cache_store:
                self._cache_store[key] = value
                self._invalidate_keys()

    def __delitem__(self, key: str):
        """
        Delete the value for *key*.
        Deletes the value from the original store and the cache store, if key exists.
        :param key: The key.
        """
        del self._original_store[key]
        with self._lock:
            try:
                del self._cache_store[key]
            except KeyError:
                # Ok, key wasn't cached so far.
                pass
            self._invalidate_keys()

    def listdir(self, path: str = None) -> List[str]:
        """List the keys under *path*, if any, of the original store."""
        return zarr.storage.listdir(self._original_store, path)

    def getsize(self, path: str = None) -> int:
        """Get the size in bytes of all values under *path*, if any."""
        return zarr.storage.getsize(self._original_store, path=path)

    def _keys(self) -> AbstractSet[str]:
        """Get keys from the original store."""
        if self._cached_keys is None:
            self._cached_keys = self._original_store.keys()
        return self._cached_keys

    def _invalidate_keys(self):
        self._cached_keys = None
