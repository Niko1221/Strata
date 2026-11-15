import ctypes
import os
import struct
import unittest
from unittest import mock

from serve.windows_process_snapshot import (FILETIME_EPOCH, MAX_BUFFER_BYTES,
    WindowsProcessSnapshot, decode_process_snapshot)


def image(base=0x100000, pids=(101,), created_ticks=None):
    raw = bytearray(320 * len(pids))
    for i, pid in enumerate(pids):
        at = i * 320
        struct.pack_into("<I", raw, at, 320 if i + 1 < len(pids) else 0)
        birth = int(FILETIME_EPOCH * 10_000_000) if created_ticks is None else created_ticks
        struct.pack_into("<qqq", raw, at + 32, birth, 12_500_000, 5_000_000)
        name = "editor.exe".encode("utf-16-le")
        struct.pack_into("<HH", raw, at + 56, len(name), len(name))
        struct.pack_into("<Q", raw, at + 64, base + at + 256)
        struct.pack_into("<QQ", raw, at + 80, pid, 77)
        struct.pack_into("<Q", raw, at + 144, 9 * 2**30)
        raw[at + 256:at + 256 + len(name)] = name
    return raw


class DecoderTests(unittest.TestCase):
    def test_exact_identity_and_counters(self):
        birth = int(FILETIME_EPOCH * 10_000_000) + 123
        parsed = decode_process_snapshot(image(created_ticks=birth), 0x100000)
        row = parsed[101]
        self.assertEqual(row["create_time_ticks"], birth)
        self.assertEqual(row["ppid"], 77)
        self.assertEqual(row["name"], "editor.exe")
        self.assertEqual(row["cpu_times"].user, 1.25)
        self.assertEqual(row["cpu_times"].system, .5)
        self.assertEqual(row["memory_info"].rss, 9 * 2**30)

    def test_multiple_processes_and_count_cap(self):
        raw = image(pids=(101, 102))
        self.assertEqual(list(decode_process_snapshot(raw, 0x100000)), [101, 102])
        with self.assertRaises(ValueError):
            decode_process_snapshot(raw, 0x100000, 1)

    def test_pointer_bounds_never_dereference_untrusted_addresses(self):
        for pointer in (0, 0xfffff, 0x100000 + 319, 0xffffffffffffffff):
            raw = image()
            struct.pack_into("<Q", raw, 64, pointer)
            with self.assertRaises(ValueError):
                decode_process_snapshot(raw, 0x100000)

    def test_malformed_offsets_lengths_and_identities(self):
        cases = [("<I", 0, 1), ("<I", 0, 256), ("<I", 0, 0xfffffff8),
                 ("<H", 56, 3), ("<H", 58, 2), ("<q", 32, -1),
                 ("<q", 40, -1), ("<Q", 80, 2**40), ("<Q", 88, 2**40)]
        for fmt, at, value in cases:
            with self.subTest(fmt=fmt, at=at):
                raw = image()
                struct.pack_into(fmt, raw, at, value)
                with self.assertRaises(ValueError):
                    decode_process_snapshot(raw, 0x100000)
        with self.assertRaises(ValueError):
            decode_process_snapshot(image(pids=(101, 101)), 0x100000)
        with self.assertRaises(ValueError):
            decode_process_snapshot(image()[:255], 0x100000)

    def test_empty_kernel_name_and_invalid_unicode(self):
        raw = image(pids=(0,))
        struct.pack_into("<HH", raw, 56, 0, 0)
        self.assertEqual(decode_process_snapshot(raw, 0x100000)[0]["name"], "System Idle Process")
        raw = image()
        raw[256:258] = b"\x00\xd8"  # unpaired UTF-16 surrogate
        with self.assertRaises(UnicodeDecodeError):
            decode_process_snapshot(raw, 0x100000)


class QueryTests(unittest.TestCase):
    def sampler(self, query):
        sampler = WindowsProcessSnapshot.__new__(WindowsProcessSnapshot)
        sampler._buffer = ctypes.create_string_buffer(256)
        sampler._query = query
        return sampler

    def test_one_bounded_resize_then_success_and_monotonic_midpoint(self):
        calls = []
        def query(kind, buffer, size, returned):
            calls.append(size)
            pointer = ctypes.cast(returned, ctypes.POINTER(ctypes.c_uint32))
            pointer[0] = 320
            if size < 320:
                return -1073741820  # STATUS_INFO_LENGTH_MISMATCH
            raw = image(ctypes.addressof(buffer), pids=(os.getpid(),))
            ctypes.memmove(buffer, bytes(raw), len(raw))
            return 0
        sampler = self.sampler(query)
        with mock.patch("serve.windows_process_snapshot.time.monotonic", side_effect=(1, 1.1, 1.2, 1.3, 1.4, 1.5)), \
             mock.patch("serve.windows_process_snapshot.time.time", side_effect=(100, 101)):
            sample = sampler.sample()
        self.assertEqual(len(calls), 2)
        self.assertAlmostEqual(sample.monotonic_time, 1.35)
        self.assertAlmostEqual(sample.wall_time, 101.05)
        self.assertIn(os.getpid(), sample.processes)

    def test_status_buffer_limit_and_latency_fail_closed(self):
        for status in (-1,):
            with self.assertRaises(OSError):
                self.sampler(lambda *_: status).sample()
        def too_large(kind, buffer, size, returned):
            ctypes.cast(returned, ctypes.POINTER(ctypes.c_uint32))[0] = MAX_BUFFER_BYTES + 1
            return -1073741820
        with self.assertRaises(ValueError):
            self.sampler(too_large).sample()
        with mock.patch("serve.windows_process_snapshot.time.monotonic", side_effect=(0, 0, 1)):
            with self.assertRaises(TimeoutError):
                self.sampler(lambda *_: 0).sample(max_seconds=.75)

    def test_no_unbounded_resize_loop(self):
        sampler = self.sampler(lambda *_: -1073741820)
        with self.assertRaises(RuntimeError):
            sampler.sample()


if __name__ == "__main__":
    unittest.main()
