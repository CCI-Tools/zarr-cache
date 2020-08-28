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
from typing import Dict

from ._opener import StoreOpener


def new_memory_store_opener(store_collection: Dict[str, collections.MutableMapping] = None) -> StoreOpener:
    """
    Create a new memory store opener.

    :param store_collection: Optional dictionary in which the new stores will be kept.
    :return: A new store opener.
    """

    def open_store(store_id):
        store = dict()
        if store_collection is not None:
            store_collection[store_id] = store
        return store

    return open_store


def new_s3_store_opener(root_pattern: str = '${store_id}',
                        s3: 's3fs.S3FileSystem' = None,
                        **s3_kwargs) -> StoreOpener:
    """
    Create a new store opener in S3-compatible object storage.
    Requires the packages ``s3fs`` and ``boto3`` to be installed.

    :param root_pattern: A root path pattern that may contain the placeholder "${store_id}". Defaults to "${store_id}".
    :param s3: Optional ``s3fs.S3FileSystem`` instance.
    :param s3_kwargs: Keyword-arguments passed to ``s3fs.S3FileSystem`` constructor in case *s3* is not provided.
    :return: A new store opener.
    """
    import s3fs

    if s3 is None:
        s3 = s3fs.S3FileSystem(**s3_kwargs)
    elif s3_kwargs:
        raise ValueError(f'unexpected keyword(s): {s3_kwargs}')

    def open_store(store_id):
        return s3fs.S3Map(root=root_pattern.format(store_id=store_id), s3=s3, check=False, create=True)

    return open_store
