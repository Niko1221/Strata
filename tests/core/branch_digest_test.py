"""Independent SHA-256 checks, with separately selectable CPU/CUDA test IDs."""
import hashlib
import json
import os
import random
import subprocess
import unittest


def fixtures():
    yield b''
    yield b'abc'
    yield b'abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq'
    yield b'a' * 1000000
    rng = random.Random(140014)
    lengths = [1, 2, 7, 8, 15, 16, 31, 32, 55, 56, 57, 63, 64, 65,
               119, 120, 127, 128, 129, 255, 256, 257, 4095, 4096, 4097,
               8191, 8192, 8193, 65535, 65536, 65537, 2097151, 2097152, 2097153]
    for length in lengths:
        yield rng.randbytes(length)


class HostDigestTests(unittest.TestCase):
    def test_sha256_known_vectors_and_padding_boundaries(self):
        binary = os.environ['CTT_DIGEST_HOST_BINARY']
        for data in fixtures():
            with self.subTest(bytes=len(data), input_sha256=hashlib.sha256(data).hexdigest()):
                result = subprocess.run([binary], input=data, capture_output=True, check=True)
                self.assertEqual(result.stdout.decode().strip(), hashlib.sha256(data).hexdigest())


class CudaDigestTests(unittest.TestCase):
    def test_actual_cuda_tiles_match_independent_sha256(self):
        binary = os.environ['CTT_DIGEST_GPU_BINARY']
        for data in fixtures():
            with self.subTest(bytes=len(data), input_sha256=hashlib.sha256(data).hexdigest()):
                result = subprocess.run([binary], input=data, capture_output=True, check=True)
                report = json.loads(result.stdout)
                chunks = [data[i:i + 4096] for i in range(0, len(data), 4096)] or [b'']
                self.assertEqual(report['digests'], [hashlib.sha256(chunk).hexdigest() for chunk in chunks])
                self.assertEqual(report['device_hashed_bytes'], len(data))
                self.assertEqual(report['digest_copy_bytes'], len(chunks) * 32)
                self.assertEqual(report['kernel_calls'], 1)
                self.assertEqual(report['workspace_allocations'], 1)
                self.assertEqual(report['workspace_releases'], 1)
                self.assertEqual(report['workspace_live_bytes'], 0)


if __name__ == '__main__':
    unittest.main()
