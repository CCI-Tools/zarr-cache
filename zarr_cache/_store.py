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
from typing import List

import zarr.storage

from ._cache import StoreCache
from ._cache import StoreItemFilter
from ._index import StoreIndex
from ._opener import StoreOpener
from .util import close_store


class CacheStore(collections.MutableMapping):
    """
    A Zarr key-value store that wraps (decorates) another store so it can be cached in a
    multi-store cache whose keys are managed through the given *store_index*.

    :param store_id: An identifier that uniquely identifies this store in the *store_cache*.
    :param store: The original store to be cached.
    :param store_index: The store index.
    :param store_opener: Factory for writable cache stores.
    :param store_item_filter: Filter function that, if given, is called to decide whether a value should be cached or
        not. Its signature is ``store_item_filter(store_id, key, value, duration) -> bool``. If it returns True, a
        value will be cached. If *store_item_filter* is not given, values will always be cached.
    """

    def __init__(self,
                 store: collections.MutableMapping,
                 store_id: str,
                 store_index: StoreIndex,
                 store_opener: StoreOpener,
                 store_item_filter: StoreItemFilter = None):
        self._store = store
        self._store_id = store_id
        self._store_cache = StoreCache(store_index, store_opener, store_item_filter)

    @property
    def hits(self) -> int:
        return self._store_cache.hits

    @property
    def misses(self) -> int:
        return self._store_cache.misses

    def __len__(self):
        """Gets the number of keys in the original store."""
        return len(self._store)

    def __iter__(self):
        """Iterates keys of the original store."""
        return self.keys()

    def __contains__(self, key: str) -> bool:
        """Test whether *key* is in the original store."""
        return key in self._store

    def keys(self):
        """Get keys from the original store."""
        return self._store.keys()

    def clear(self):
        """Clear the original and cached store."""
        self._store.clear()
        self._store_cache.clear_store(self._store_id)

    def close(self):
        """Closes the original store and the cached store."""
        close_store(self._store)
        self._store_cache.close_store(self._store_id)

    def __getitem__(self, key: str) -> bytes:
        """
        Get value for *key* either from the cached store or from the original.
        In the latter case, the value will be put into the cached store.
        :param key: The key.
        :return: An original or cached value.
        """
        return self._store_cache.get_value(self._store_id, key, self._store)

    def __setitem__(self, key: str, value: bytes):
        """
        Set *key* to *value*. Sets the value in the original store and the cached store.
        :param key: The key.
        :param value: The value
        """
        self._store[key] = value
        self._store_cache.put_value(self._store_id, key, value)

    def __delitem__(self, key: str):
        """
        Delete the value for *key*. Deletes the value from the original and cached store.
        :param key: The key.
        """
        del self._store[key]
        self._store_cache.delete_value(self._store_id, key)

    def listdir(self, path: str = None) -> List[str]:
        """List the keys under *path*, if any, of the original store."""
        return zarr.storage.listdir(self._store, path)

    def getsize(self, path: str = None) -> int:
        """Get the size in bytes of all values under *path*, if any."""
        return zarr.storage.getsize(self._store, path=path)
