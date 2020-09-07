import collections
import unittest

import moto
import s3fs

from tests import helpers
from zarr_cache import MemoryStoreOpener
from zarr_cache import S3StoreOpener


class MemoryStoreOpenerTest(unittest.TestCase):

    def test_open(self):
        opener = MemoryStoreOpener()

        cube_1 = opener('cube_1')
        cube_2 = opener('cube_2')

        self.assertIsInstance(cube_1, collections.MutableMapping)
        self.assertEqual({}, cube_1)
        self.assertIsInstance(cube_2, collections.MutableMapping)
        self.assertEqual({}, cube_2)
        self.assertIsNot(cube_1, cube_2)

    def test_open_with_given_stores(self):
        cube_1 = helpers.make_test_store()
        stores = dict(cube_1=cube_1)
        opener = MemoryStoreOpener(stores=stores)

        cube_2 = opener('cube_2')
        cube_3 = opener('cube_3')

        self.assertIsInstance(cube_1, collections.MutableMapping)
        self.assertIsInstance(cube_2, collections.MutableMapping)
        self.assertIsInstance(cube_3, collections.MutableMapping)
        self.assertEqual(dict(cube_1=cube_1, cube_2=cube_2, cube_3=cube_3),
                         stores)


class S3StoreOpenerTest(unittest.TestCase):
    BUCKET_NAME = 'store-cache'

    def test_open_fs(self):
        with moto.mock_s3():
            s3 = s3fs.S3FileSystem(key='humpty', secret='dumpty')
            self._write_test_data(s3)
            self._test_opener(S3StoreOpener(root_pattern=self.BUCKET_NAME + '/{store_id}.zarr',
                                            s3=s3))
            self._delete_test_data(s3)

    def test_open_s3_kwargs(self):
        with moto.mock_s3():
            s3 = s3fs.S3FileSystem(key='humpty', secret='dumpty')
            self._write_test_data(s3)
            self._test_opener(S3StoreOpener(root_pattern=self.BUCKET_NAME + '/{store_id}.zarr',
                                            key='humpty', secret='dumpty'))
            self._delete_test_data(s3)

    def test_open_s3_and_s3_kwargs(self):
        with moto.mock_s3():
            s3 = s3fs.S3FileSystem(key='humpty', secret='dumpty')
            self._write_test_data(s3)

            with self.assertRaises(ValueError) as cm:
                S3StoreOpener(root_pattern=self.BUCKET_NAME + '/{store_id}.zarr',
                              s3=s3, key='humpty', secret='dumpty')
            self.assertEqual('Unexpected keyword arguments: key, secret', f'{cm.exception}')
            self._delete_test_data(s3)

    def _test_opener(self, opener):
        cube_1 = opener('cube_1')
        cube_2 = opener('cube_2')
        cube_3 = opener('cube_3')

        self.assertIsInstance(cube_1, collections.MutableMapping)
        self.assertIn('.zgroup', cube_1)
        self.assertIn('.zattrs', cube_1)
        self.assertEqual(8, len(cube_1))

        self.assertIsInstance(cube_2, collections.MutableMapping)
        self.assertIn('.zgroup', cube_2)
        self.assertIn('.zattrs', cube_2)
        self.assertEqual(8, len(cube_2))

        self.assertIsInstance(cube_1, collections.MutableMapping)
        self.assertEqual({}, cube_3)

    @classmethod
    def _write_test_data(cls, s3: s3fs.S3FileSystem):
        if not s3.isdir(cls.BUCKET_NAME):
            s3.mkdir(cls.BUCKET_NAME)

        data = helpers.make_test_store()
        s3map = s3fs.S3Map(root=cls.BUCKET_NAME + '/cube_1.zarr', s3=s3, create=True)
        s3map.update(data)
        s3map = s3fs.S3Map(root=cls.BUCKET_NAME + '/cube_2.zarr', s3=s3, create=True)
        s3map.update(data)

    @classmethod
    def _delete_test_data(cls, s3: s3fs.S3FileSystem):
        s3.rm(cls.BUCKET_NAME, recursive=True)
