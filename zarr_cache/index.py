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
from collections import OrderedDict
from typing import Tuple, Optional


class StoreIndex(abc.ABC):
    """
    An index of stores and their keys.
    This interface is meant to be backed by databases such as Redis.
    """

    @property
    @abc.abstractmethod
    def max_size(self) -> Optional[int]:
        """Maximum sum of sizes associated with the keys in this index. Optional value."""

    @property
    @abc.abstractmethod
    def current_size(self) -> int:
        """Current sum of sizes associated with the keys in this index."""

    @abc.abstractmethod
    def push_key(self, store_id: str, key: str, size: int):
        """
        Place a most recently used key into index.
        :param store_id: The store identifier.
        :param key: The key within the store.
        :param size: The size associated within the key.
        """

    @abc.abstractmethod
    def pop_key(self) -> Tuple[str, str, int]:
        """
        Pop a key from the index.
        E.g. an implementation may remove the least frequently used key.
        :return: A 3-tuple (*store_id*, *key*, *size*)
        """

    @abc.abstractmethod
    def mark_key(self, store_id: str, key: str):
        """
        Mark a key to be in use now.
        E.g. an implementation may increase usage count or make most recent in some list.
        :param store_id: The store identifier.
        :param key: The key within the store.
        """

    @abc.abstractmethod
    def delete_key(self, store_id: str, key: str) -> int:
        """
        Delete given key.
        :param store_id: The store identifier.
        :param key: The key within the store.
        :return: The size associated with the key
        """

    @abc.abstractmethod
    def delete_store(self, store_id: str) -> int:
        """
        Delete all keys belonging to given store.
        :param store_id: The store identifier.
        :return: The sum of sizes associated with the deleted keys
        """


class RedisStoreIndex(StoreIndex):
    """
    A store index implementation with Redis as backend.
    """

    @classmethod
    def create_index(cls, max_size: int):
        raise NotImplementedError()

    @property
    def max_size(self) -> int:
        raise NotImplementedError()

    @property
    def current_size(self) -> int:
        raise NotImplementedError()

    def push_key(self, store_id: str, key: str, size: int):
        raise NotImplementedError()

    def pop_key(self) -> Tuple[str, str, int]:
        pass

    def mark_key(self, store_id: str, key: str):
        pass

    def delete_key(self, store_id: str, key: str) -> int:
        raise NotImplementedError()

    def delete_store(self, store_id: str) -> int:
        raise NotImplementedError()


class MemoryStoreIndex(StoreIndex):
    """
    An in-memory store index implementation.
    Used for testing and as implementation reference only.
    Production indexes should be persistent and backed by a database such as Redis, see :class:RedisStoreIndex.

    :param is_lifo: Whether push/pop operate as LIFO or FIFO stack.
    :param max_size: optional maximum size.
    """

    def __init__(self, is_lifo: bool = True, max_size: int = None):
        self._is_lifo = is_lifo
        self._max_size = max_size
        self._current_size = 0
        self._entries = OrderedDict()

    @property
    def max_size(self) -> Optional[int]:
        return self._max_size

    @property
    def current_size(self) -> int:
        return self._current_size

    def push_key(self, store_id: str, key: str, size: int):
        # Push key on top
        self._entries[(store_id, key)] = size
        self._current_size += size

    def pop_key(self) -> Tuple[str, str, int]:
        # Pop key from bottom
        (store_id, key), size = self._entries.popitem(last=not self._is_lifo)
        self._current_size -= size
        return store_id, key, size

    def mark_key(self, store_id: str, key: str):
        # Move key to top
        k = store_id, key
        if k in self._entries:
            self._entries.move_to_end((store_id, key), last=self._is_lifo)

    def delete_key(self, store_id: str, key: str) -> int:
        k = store_id, key
        if k in self._entries:
            size = self._entries[k]
            del self._entries[k]
            self._current_size -= size
            return size
        return 0

    def delete_store(self, store_id: str):
        k_list = [k for k in self._entries.keys() if k[0] == store_id]
        size_sum = 0
        for k in k_list:
            size_sum += self.delete_key(k[0], k[1])
        return size_sum
