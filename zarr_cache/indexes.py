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

from collections import OrderedDict
from typing import Tuple, Optional

from ._index import StoreIndex


class MemoryStoreIndex(StoreIndex):
    """
    An in-memory store index implementation.
    Used for testing and as implementation reference only.
    Production indexes should be persistent and backed by a database such as Redis.

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
