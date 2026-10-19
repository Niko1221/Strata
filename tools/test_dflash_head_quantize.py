#!/usr/bin/env python3
"""Offline GGUF format and quantization gates for draft-only head exports."""
import pathlib
import tempfile
import unittest
import numpy as np
import dflash_head_quantize as h
from gguf import GGUFWriter, GGMLQuantizationType as Q
from gguf.quants import quantize, dequantize

class HeadExport(unittest.TestCase):
    def test_chunked_copy_and_quantization(self):
        with tempfile.TemporaryDirectory() as d:
            source = pathlib.Path(d)/'head.gguf'
            values = np.random.default_rng(42).normal(0,.05,(19,2560)).astype(np.float32)
            original = quantize(values,Q.Q8_0)
            w = GGUFWriter(source,'qwen4exp')
            for key,value in [('block_count',48),('embedding_length',2560),('expert_count',512),('expert_used_count',10),('attention.head_count',24),('attention.head_count_kv',2)]:
                w.add_uint32('qwen4exp.'+key,value)
            w.add_tensor('output.weight',original,raw_dtype=Q.Q8_0)
            w.write_header_to_file();w.write_kv_data_to_file();w.write_tensors_to_file();w.close()
            before = source.read_bytes()
            with self.assertRaises(ValueError): h.export(source,source,'copy')
            self.assertEqual(source.read_bytes(),before)
            sidecar_source = pathlib.Path(d)/'sidecar.gguf.json'
            sidecar_source.write_bytes(before)
            with self.assertRaises(ValueError): h.export(sidecar_source,pathlib.Path(d)/'sidecar.gguf','copy')
            self.assertEqual(sidecar_source.read_bytes(),before)
            for rows in (0,-1):
                with self.assertRaises(ValueError): h.export(source,pathlib.Path(d)/'invalid.gguf','copy',rows)
            # A fixed ".part" name used to truncate a source with that name.
            collision = pathlib.Path(d)/'collision.gguf.part'
            collision.write_bytes(before)
            h.export(collision,pathlib.Path(d)/'collision.gguf','copy')
            self.assertEqual(collision.read_bytes(),before)
            for kind in ['copy','Q8_0','Q4_0']:
                output=pathlib.Path(d)/f'{kind}.gguf';r=h.export(source,output,kind,chunk_rows=7)
                g=h.GGUFFile(output);t=g.tensors[0]
                self.assertEqual(t.shape,[2560,19]);self.assertEqual(g.metadata['general.architecture'],'qwen4exp')
                self.assertEqual(g.metadata['strata.dflash_head.source_type'],'Q8_0')
                raw=output.read_bytes()[g.data_start:]
                if kind=='copy': self.assertEqual(raw,original.tobytes())
                else:
                    actual=dequantize(np.frombuffer(raw,np.uint8).reshape(19,-1),Q(t.type_id))
                    self.assertTrue(np.isfinite(actual).all())
                    self.assertLess(r['relative_l2'],.12 if kind=='Q4_0' else .02)
                self.assertEqual(r['payload_bytes'],t.expected_bytes())

if __name__=='__main__': unittest.main()
