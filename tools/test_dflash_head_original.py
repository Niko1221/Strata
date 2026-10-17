"""Offline original-head fetch tests: range boundaries, byte preservation and cache corruption."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dflash_head_original as head


class OriginalHead(unittest.TestCase):
    def test_exact_bytes_cache_and_corruption(self):
        shape = [256, 3]
        raw = bytes(range(256)) * 6
        meta = {'dtype': 'BF16', 'shape': list(reversed(shape)), 'data_offsets': [17, 17 + len(raw)]}
        calls = []

        def get(url, start=None, end=None):
            if start is None:
                return json.dumps({'weight_map': {'lm_head.weight': 'shard.safetensors'}}).encode()
            calls.append((start, end))
            return raw[start - 117:end - 117 + 1]

        with tempfile.TemporaryDirectory() as d, patch.object(head, 'SHAPE', shape), \
                patch.object(head, 'CHUNK', 300), patch.object(head.source, 'get', side_effect=get), \
                patch.object(head.source, 'shard_header', return_value=(100, {'lm_head.weight': meta})):
            output = Path(d) / 'head.gguf'
            head.fetch(output)
            g = head.GGUFFile(output)
            self.assertEqual(output.read_bytes()[g.data_start:], raw)
            self.assertEqual(g.tensors[0].type_id, 30)
            self.assertEqual(calls[0][0], 117)
            self.assertEqual(calls[-1][1], 116 + len(raw))
            self.assertTrue(head.valid_cache(output))
            count = len(calls)
            head.fetch(output)
            self.assertEqual(len(calls), count)
            damaged = bytearray(output.read_bytes()); damaged[-1] ^= 1; output.write_bytes(damaged)
            self.assertFalse(head.valid_cache(output))
            head.fetch(output)
            self.assertEqual(output.read_bytes()[g.data_start:], raw)

    def test_refuses_quantized_or_wrong_geometry(self):
        with tempfile.TemporaryDirectory() as d, \
                patch.object(head.source, 'get', return_value=b'{"weight_map":{"lm_head.weight":"shard"}}'), \
                patch.object(head.source, 'shard_header', return_value=(100, {
                    'lm_head.weight': {'dtype': 'F16', 'shape': [248320, 2560], 'data_offsets': [0, 1271398400]}})):
            with self.assertRaisesRegex(ValueError, 'unexpected dtype'):
                head.fetch(Path(d) / 'head.gguf')


if __name__ == '__main__':
    unittest.main()
