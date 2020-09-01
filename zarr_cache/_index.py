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
