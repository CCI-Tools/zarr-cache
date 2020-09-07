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
from typing import Dict, Callable

# Store openers receive a store ID and return a MutableMapping.
# The returned stores must be writable.

StoreOpener = Callable[[str], collections.MutableMapping]


class MemoryStoreOpener:
    """
    A store opener that creates in-memory stores.

    :param stores: Optional dictionary used to hold the opened stores.
    """

    def __init__(self, stores: Dict[str, collections.MutableMapping] = None):
        self._stores = stores

    def __call__(self, store_id: str) -> collections.MutableMapping:
        """Open a store for store identifier *store_id*."""
        if self._stores is not None:
            if store_id not in self._stores:
                self._stores[store_id] = dict()
            return self._stores[store_id]
        return dict()


class S3StoreOpener:
    """
    A store opener that opens stores from S3-compatible object storage.
    Requires the packages ``s3fs`` and ``boto3`` to be installed.

    :param root_pattern: A root path pattern that may contain the placeholder "{store_id}".
        Defaults to "{store_id}.zarr". If this forms a relative path, then *s3_kwargs* must
        provide a keyword-argument "client_kwargs" that must provide a valid "endpoint_url" value.
    :param s3: Optional ``s3fs.S3FileSystem`` instance. If given, *s3_kwargs* are not allowed.
    :param s3_kwargs: Keyword-arguments passed to ``s3fs.S3FileSystem`` constructor
        in case *s3* is not provided.
    """

    # noinspection PyUnresolvedReferences
    def __init__(self,
                 root_pattern: str = '{store_id}.zarr',
                 s3: 's3fs.S3FileSystem' = None,
                 **s3_kwargs):
        import s3fs

        self._root_pattern = root_pattern
        if s3 is None:
            self._s3 = s3fs.S3FileSystem(**s3_kwargs)
        elif not s3_kwargs:
            self._s3 = s3
        else:
            raise ValueError(f'Unexpected keyword arguments: {", ".join(s3_kwargs.keys())}')

    def __call__(self, store_id: str) -> collections.MutableMapping:
        """Open a store for store identifier *store_id*."""
        import s3fs
        return s3fs.S3Map(root=self._root_pattern.format(store_id=store_id),
                          s3=self._s3,
                          check=False,
                          create=True)
