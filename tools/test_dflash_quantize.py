#!/usr/bin/env python3
"""CPU checks for whole-drafter GGUF quantization, independent of a GPU/download."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dflash_quantize as d
from gguf_writer import GGUFWriter
from gguf.quants import quantize, dequantize
from gguf import GGMLQuantizationType as Q

class Export(unittest.TestCase):
    def test_matrix_quantization_and_exact_norms(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'original.gguf'
            w = GGUFWriter()
            w.add('general.architecture', 'dflash')
            w.add('dflash.target_layers', [3, 15, 23, 35, 43])
            x = np.random.default_rng(73).normal(0, .07, (13, 64)).astype(np.float32)
            norms = np.linspace(.5, 1.5, 64, dtype=np.float32)
            w.add_bf16('fc.weight', x, shape=[64, 13])
            w.add_bf16('output_norm.weight', norms, shape=[64])
            w.write(source)
            g = d.GGUFFile(source); raw = source.read_bytes()
            m, n = g.tensors
            bits = np.frombuffer(raw, '<u2', count=13*64, offset=g.data_start + m.offset).astype(np.uint32) << 16
            values = bits.view(np.float32).reshape(13, 64)
            for kind in ('Q8_0', 'Q5_0', 'Q4_0'):
                with self.subTest(kind=kind):
                    a, b = Path(folder)/f'{kind}-a.gguf', Path(folder)/f'{kind}-b.gguf'
                    d.export(source, a, kind, chunk_rows=3)
                    d.export(source, b, kind, chunk_rows=7)
                    self.assertEqual(a.read_bytes(), b.read_bytes())
                    out = d.GGUFFile(a); ma, na = out.tensors; payload = a.read_bytes()
                    self.assertEqual(out.metadata['dflash.target_layers'], [3, 15, 23, 35, 43])
                    self.assertEqual(out.metadata['strata.dflash.source_sha256'], hashlib.sha256(raw).hexdigest())
                    self.assertEqual(ma.type_id, Q[kind].value)
                    self.assertEqual(na.type_id, 30)
                    self.assertEqual(payload[out.data_start+na.offset:out.data_start+na.offset+na.expected_bytes()],
                                     raw[g.data_start+n.offset:g.data_start+n.offset+n.expected_bytes()])
                    packed = payload[out.data_start+ma.offset:out.data_start+ma.offset+ma.expected_bytes()]
                    self.assertEqual(packed, quantize(values, Q[kind]).tobytes())
                    back = dequantize(np.frombuffer(packed, np.uint8).reshape(13,-1),Q[kind])
                    self.assertLess(np.linalg.norm(back-values)/np.linalg.norm(values), .14)
                    with self.assertRaisesRegex(ValueError, 'original BF16'):
                        d.export(a, Path(folder)/'requant.gguf', 'Q8_0')
            truncated = Path(folder)/'truncated.gguf'
            truncated.write_bytes(raw[:-16])
            with self.assertRaisesRegex(ValueError, 'truncated'):
                d.export(truncated, Path(folder)/'bad.gguf', 'Q4_0')

class BF16Conversion(unittest.TestCase):
    def test_nan_sign_payload_infinities_and_ties(self):
        from gguf_writer import to_bf16
        # Signalling/quiet NaNs of both signs must never become infinities.
        raw = np.array([0x7f800001, 0xff800001, 0x7fc12345, 0xffc12345,
                        0x7f800000, 0xff800000, 0x00000000, 0x80000000,
                        0x3f808000, 0x3f818000], dtype='<u4')
        expected = np.array([0x7fc0, 0xffc0, 0x7fc1, 0xffc1, 0x7f80, 0xff80,
                             0, 0x8000, 0x3f80, 0x3f82], dtype='<u2')
        np.testing.assert_array_equal(to_bf16(raw.view('<f4')), expected)

if __name__ == '__main__': unittest.main()
