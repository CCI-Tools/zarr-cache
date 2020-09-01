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

from ._cache import StoreCache
from .util import close_store

StoreItemFilter = Callable[[str, Union[bytes, np.ndarray], float], bool]

_nan = float('nan')


class CachedStore(collections.MutableMapping):
    """
    A Zarr key-value store that wraps (decorates) an original store so it can be cached together with other
    stores in in a multi-store cache.

    :param original_store: The original store to be cached.
    :param store_id: The store identifier. Must be unique within the stores managed by *store_cache*.
    :param store_cache: The multi-store cache. Must implement the :class:StoreCache protocol.
    :param store_item_filter: Filter function that, if given, is called to decide whether a value should be cached or
        not. Its signature is ``store_item_filter(store_id, key, value, duration) -> bool``. If it returns True, a
        value will be cached. If *store_item_filter* is not given, values will always be cached.
    """

    def __init__(self,
                 original_store: collections.MutableMapping,
                 store_id: str,
                 store_cache: StoreCache,
                 store_item_filter: StoreItemFilter = None):
        self._store = original_store
        self._store_id = store_id
        self._store_cache = store_cache
        self._store_item_filter = store_item_filter
        self._cached_keys: Optional[AbstractSet[str]] = None
        self._hit_count = 0
        self._hit_latency_sum = 0.0
        self._miss_count = 0
        self._miss_latency_sum = 0.0
        self._lock = threading.RLock()

    def __getstate__(self):
        return (
            self._store,
            self._store_id,
            self._store_cache,
            self._store_item_filter,
            self._cached_keys,
            self._hit_count,
            self._hit_latency_sum,
            self._miss_count,
            self._miss_latency_sum,
        )

    def __setstate__(self, state):
        (
            self._store,
            self._store_id,
            self._store_cache,
            self._store_item_filter,
            self._cached_keys,
            self._hit_count,
            self._hit_latency_sum,
            self._miss_count,
            self._miss_latency_sum,
        ) = state
        self._lock = threading.RLock()

    @property
    def hit_count(self) -> int:
        """Gets the number of cache hits, that is, value retrievals from the cached store."""
        return self._hit_count

    @property
    def miss_count(self) -> int:
        """Gets the number of cache misses, that is, value retrievals from the original store."""
        return self._miss_count

    @property
    def hit_latency(self) -> float:
        """Gets the average latency of value retrievals from the cached store."""
        return self._hit_latency_sum / self._hit_count if self._hit_count > 0 else _nan

    @property
    def miss_latency(self) -> float:
        """Gets the average latency of value retrievals from the original store."""
        return self._miss_latency_sum / self._miss_count if self._miss_count > 0 else _nan

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
        self._store.clear()
        with self._lock:
            self._store_cache.clear_store(self._store_id)
            self._invalidate_keys()

    def close(self):
        """Closes the original store and the cached store."""
        close_store(self._store)
        with self._lock:
            self._store_cache.close_store(self._store_id)
            self._invalidate_keys()

    def __getitem__(self, key: str) -> bytes:
        """
        Get value for *key* either from the cached store or from the original.
        In the latter case, the value will be put into the cached store.
        :param key: The key.
        :return: An original or cached value.
        """
        try:
            now = time.perf_counter()
            value = self._store_cache.get_value(self._store_id, key)
            latency = time.perf_counter() - now
            self._hit_latency_sum += latency
            self._hit_count += 1
            return value
        except KeyError:
            now = time.perf_counter()
            value = self._store[key]
            latency = time.perf_counter() - now
            self._miss_latency_sum += latency
            self._miss_count += 1
            if self._store_item_filter is not None:
                should_cache = self._store_item_filter(key, value, time.perf_counter() - now)
            else:
                value = self._store[key]
                should_cache = True
            if should_cache:
                with self._lock:
                    self._store_cache.put_value(self._store_id, key, value)
                    self._invalidate_keys()
            return value

    def __setitem__(self, key: str, value: bytes):
        """
        Set *key* to *value*.
        Sets the value in the original store.
        Sets the value in the cache store, only if the value was already cached.
        :param key: The key.
        :param value: The value
        """
        self._store[key] = value
        with self._lock:
            if self._store_cache.has_value(self._store_id, key):
                self._store_cache.put_value(self._store_id, key, value)
                self._invalidate_keys()

    def __delitem__(self, key: str):
        """
        Delete the value for *key*.
        Deletes the value from the original store and the cache store, if key exists.
        :param key: The key.
        """
        del self._store[key]
        with self._lock:
            self._store_cache.delete_value(self._store_id, key)
            self._invalidate_keys()

    def listdir(self, path: str = None) -> List[str]:
        """List the keys under *path*, if any, of the original store."""
        return zarr.storage.listdir(self._store, path)

    def getsize(self, path: str = None) -> int:
        """Get the size in bytes of all values under *path*, if any."""
        return zarr.storage.getsize(self._store, path=path)

    def _keys(self) -> AbstractSet[str]:
        """Get keys from the original store."""
        if self._cached_keys is None:
            self._cached_keys = self._store.keys()
        return self._cached_keys

    def _invalidate_keys(self):
        self._cached_keys = None
