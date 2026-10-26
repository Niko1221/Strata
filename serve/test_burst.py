"""The chunked round-robin bookkeeping (serve/burst.py)."""
import unittest

from serve.burst import BurstCursor, ended_by_chunk, pass_limit, split


class Split(unittest.TestCase):
    def test_chunks(self):
        self.assertEqual(split(100, 64), [64, 36])
        self.assertEqual(split(64, 64), [64])
        self.assertEqual(split(10, 64), [10])
        self.assertEqual(split(0, 64), [])
        self.assertEqual(split(10, 0), [])


class Cursor(unittest.TestCase):
    def test_continuation_grows(self):
        c = BurstCursor(ids=[1, 2, 3], max_new=4, burst=2)
        self.assertEqual(c.next_chunk(), 2)
        self.assertTrue(c.accept([10, 11]))
        self.assertEqual(c.continuation(), [1, 2, 3, 10, 11])
        self.assertEqual(c.next_chunk(), 2)
        self.assertTrue(c.accept([12]))
        self.assertEqual(c.next_chunk(), 1)
        self.assertFalse(c.accept([13]))
        self.assertFalse(c.accept([13]))
        self.assertTrue(c.finished)
        self.assertEqual(c.next_chunk(), 0)

    def test_eos_stops_the_request(self):
        c = BurstCursor(ids=[1], max_new=100, burst=8)
        self.assertFalse(c.accept([5, 6, 1, 7], eos=1))
        self.assertEqual(c.ids, [1, 5, 6, 1])
        self.assertTrue(c.finished)

    def test_cancel_is_finishing(self):
        c = BurstCursor(ids=[1], max_new=100, burst=8)
        c.finished = True
        self.assertEqual(c.next_chunk(), 0)
        self.assertFalse(c.accept([9]))


class PassLimit(unittest.TestCase):
    def test_cap(self):
        self.assertEqual(pass_limit(100, 0, 0), 100)
        self.assertEqual(pass_limit(100, 0, 64), 64)
        self.assertEqual(pass_limit(100, 64, 64), 36)
        self.assertEqual(pass_limit(100, 100, 64), 0)

    def test_chunk_boundary(self):
        self.assertTrue(ended_by_chunk(100, 64, 64, 64))
        self.assertFalse(ended_by_chunk(100, 30, 64, 64))   # the model stopped early
        self.assertFalse(ended_by_chunk(100, 100, 100, 64)) # the whole reply is done
        self.assertFalse(ended_by_chunk(100, 64, 64, 0))    # burst off
