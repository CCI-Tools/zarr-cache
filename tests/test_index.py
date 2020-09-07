import unittest

from zarr_cache import MemoryStoreIndex


class MemoryStoreIndexTest(unittest.TestCase):

    def test_fifo(self):
        index = MemoryStoreIndex(is_lifo=False)

        self.assertEqual(None, index.max_size)
        self.assertEqual(0, index.current_size)

        index.push_key('s1', 'k1', 100)
        self.assertEqual(100, index.current_size)

        index.push_key('s1', 'k2', 130)
        index.push_key('s1', 'k3', 120)
        index.push_key('s1', 'k4', 180)
        index.push_key('s1', 'k5', 160)
        self.assertEqual(100 + 130 + 120 + 180 + 160, index.current_size)

        r1 = index.pop_key()
        r2 = index.pop_key()
        r3 = index.pop_key()
        self.assertEqual(('s1', 'k5', 160), r1)
        self.assertEqual(('s1', 'k4', 180), r2)
        self.assertEqual(('s1', 'k3', 120), r3)
        self.assertEqual(100 + 130, index.current_size)

        index.push_key('s2', 'k1', 260)
        index.push_key('s2', 'k2', 220)
        index.push_key('s2', 'k3', 210)
        self.assertEqual(100 + 130 + 260 + 220 + 210, index.current_size)

        index.mark_key('s2', 'k2')
        r1 = index.pop_key()
        r2 = index.pop_key()
        self.assertEqual(('s2', 'k3', 210), r1)
        self.assertEqual(('s2', 'k1', 260), r2)
        self.assertEqual(100 + 130 + 220, index.current_size)

        s = index.delete_key('s2', 'k2')
        self.assertEqual(220, s)
        self.assertEqual(100 + 130, index.current_size)

        s = index.delete_store('s1')
        self.assertEqual(100 + 130, s)
        self.assertEqual(0, index.current_size)

        s = index.delete_store('s2')
        self.assertEqual(0, s)
        self.assertEqual(0, index.current_size)

    def test_lifo(self):
        index = MemoryStoreIndex(is_lifo=True)

        self.assertEqual(None, index.max_size)
        self.assertEqual(0, index.current_size)

        index.push_key('s1', 'k1', 100)
        self.assertEqual(100, index.current_size)

        index.push_key('s1', 'k2', 130)
        index.push_key('s1', 'k3', 120)
        index.push_key('s1', 'k4', 180)
        index.push_key('s1', 'k5', 160)
        self.assertEqual(100 + 130 + 120 + 180 + 160, index.current_size)

        r1 = index.pop_key()
        r2 = index.pop_key()
        r3 = index.pop_key()
        self.assertEqual(('s1', 'k1', 100), r1)
        self.assertEqual(('s1', 'k2', 130), r2)
        self.assertEqual(('s1', 'k3', 120), r3)
        self.assertEqual(180 + 160, index.current_size)

        index.push_key('s2', 'k1', 260)
        index.push_key('s2', 'k2', 220)
        index.push_key('s2', 'k3', 210)
        self.assertEqual(180 + 160 + 260 + 220 + 210, index.current_size)

        index.mark_key('s2', 'k2')
        r1 = index.pop_key()
        r2 = index.pop_key()
        self.assertEqual(('s1', 'k4', 180), r1)
        self.assertEqual(('s1', 'k5', 160), r2)
        self.assertEqual(260 + 220 + 210, index.current_size)

        s = index.delete_key('s2', 'k2')
        self.assertEqual(220, s)
        self.assertEqual(260 + 210, index.current_size)

        s = index.delete_store('s1')
        self.assertEqual(0, s)
        self.assertEqual(260 + 210, index.current_size)

        s = index.delete_store('s2')
        self.assertEqual(260 + 210, s)
        self.assertEqual(0, index.current_size)
