#!/usr/bin/env python3
"""Synthetic DFlash2 export/quantization checks; no trained checkpoint or GPU needed."""
import copy
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dflash_gguf as d
from gguf_reader import GGUFFile

CONFIG = dict(architectures=['DFlash2DraftModel'], hidden_size=32, intermediate_size=32,
              head_dim=8, num_attention_heads=4, num_key_value_heads=1, num_hidden_layers=2,
              vocab_size=64, rope_parameters={'rope_theta': 1e7}, rms_norm_eps=1e-6,
              dflash_config=dict(block_size=3, mask_token_id=7, target_layer_ids=[0, 1],
                                 conv_kernel_size=2, conv_group_size=8, selector_rank=4,
                                 selector_top_k=3, sample_from_anchor=False))

class Export(unittest.TestCase):
    def source(self, root, bare=False):
        cfg = d.normalize_config(CONFIG)
        want = d.expected(cfg)
        header, chunks, offset = {}, [], 0
        for idx, (name, shape) in enumerate(want.items()):
            if bare and name.endswith('codebook.weight'):
                name = name.removesuffix('.weight')
            # Non-uniform finite BF16 values reveal reordering and wrong 3D shapes.
            count = int(np.prod(shape))
            raw = (np.arange(count, dtype=np.uint16) % 32 + 0x3e00 + idx).astype('<u2').tobytes()
            header[name] = dict(dtype='BF16', shape=shape, data_offsets=[offset, offset + len(raw)])
            chunks.append(raw); offset += len(raw)
        encoded = json.dumps(header).encode()
        source = root / 'model.safetensors'
        source.write_bytes(struct.pack('<Q', len(encoded)) + encoded + b''.join(chunks))
        (root / 'config.json').write_text(json.dumps(CONFIG))
        return source, want

    def test_export_and_quantization_preserve_extensions(self):
        # This import uses the same gguf-py dependency as the existing quantizer.
        import dflash_quantize as q
        for bare in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source, want = self.source(root, bare)
                out = root / 'draft.gguf'
                result = subprocess.run([sys.executable, str(Path(d.__file__)), str(source), '-o', str(out)],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                self.assertIn('SELF-CHECK PASS', result.stdout)
                g = GGUFFile(out)
                self.assertEqual(len(g.tensors), len(want))
                self.assertEqual(g.metadata['dflash.target_layer_inputs'], 'false')
                self.assertEqual(g.metadata['dflash.sample_from_anchor'], 'false')
                self.assertEqual(g.metadata['dflash.selector_rank'], 4)
                base = next(t for t in g.tensors if t.name == 'blk.0.attn_conv_base')
                self.assertEqual(base.shape, [32, 2, 2])
                raw = out.read_bytes()
                for kind in ('Q4_0', 'Q5_0', 'Q8_0'):
                    dest = root / (kind + '.gguf')
                    q.export(out, dest, kind, chunk_rows=3)
                    quant = GGUFFile(dest)
                    by = {t.name: t for t in quant.tensors}
                    packed = dest.read_bytes()
                    for t in g.tensors:
                        if 'conv_' in t.name or 'selector_' in t.name:
                            qt = by[t.name]
                            self.assertEqual(qt.type_id, 30)
                            self.assertEqual(raw[g.data_start+t.offset:g.data_start+t.offset+t.expected_bytes()],
                                             packed[quant.data_start+qt.offset:quant.data_start+qt.offset+qt.expected_bytes()])

    def test_unsupported_contracts_fail(self):
        cases = [('conv_group_size', 3), ('block_size', 9), ('selector_rank', 0),
                 ('selector_top_k', 65), ('conv_kernel_size', 9), ('sample_from_anchor', True),
                 ('output_multiplier', 2), ('final_logit_softcapping', 5)]
        for key, value in cases:
            cfg = copy.deepcopy(CONFIG); cfg['dflash_config'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): d.normalize_config(cfg)
        for key, value in [('rms_norm_eps', 1e-5), ('hidden_act', 'gelu'), ('use_sliding_window', True)]:
            cfg = copy.deepcopy(CONFIG); cfg[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): d.normalize_config(cfg)

    def test_official_sliding_contract(self):
        cfg = copy.deepcopy(CONFIG)
        cfg.update(is_causal=False, use_sliding_window=True, sliding_window=2048,
                   layer_types=['sliding_attention'] * cfg['num_hidden_layers'])
        self.assertEqual(d.normalize_config(cfg)['sliding_window'], 2048)
        self.assertFalse(d.normalize_config(cfg)['target_hc_mean'])
        official = copy.deepcopy(cfg)
        official.update(hidden_size=2560, head_dim=128, num_attention_heads=32, num_key_value_heads=8)
        official['dflash_config']['target_layer_ids'] = [5,15,25,35,45]
        self.assertTrue(d.normalize_config(official)['target_hc_mean'])
        official['is_causal'] = True
        with self.assertRaisesRegex(ValueError, 'non-causal'): d.normalize_config(official)
        cfg['layer_types'][0] = 'full_attention'
        with self.assertRaisesRegex(ValueError, 'uniform'): d.normalize_config(cfg)

    def test_math_changing_options_are_not_silently_dropped(self):
        for modern in (False, True):
            cfg = copy.deepcopy(CONFIG)
            if not modern:
                cfg.pop('architectures'); cfg.update(cfg.pop('dflash_config')); cfg['sample_from_anchor'] = True
            for key, value in [('input_embedding_scale', 2), ('attention_bias', True),
                               ('is_causal', True), ('rms_norm_eps', 1e-5), ('hidden_act', 'gelu'),
                               ('markov_rank', 16), ('enable_confidence_head', True)]:
                bad = copy.deepcopy(cfg); bad[key] = value
                with self.subTest(modern=modern,key=key), self.assertRaises(ValueError): d.normalize_config(bad)
            bad = copy.deepcopy(cfg); bad['rope_parameters']['rope_type'] = 'linear'
            with self.assertRaises(ValueError): d.normalize_config(bad)
            bad = copy.deepcopy(cfg); bad['layer_types'] = ['sliding_attention'] * bad['num_hidden_layers']
            bad['sliding_window'] = 2048; bad['use_sliding_window'] = False
            with self.assertRaises(ValueError): d.normalize_config(bad)

    def test_corrupt_safetensor_ranges_are_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for offsets in ([0, 1], [0, 100], [-2, 0]):
                h = json.dumps({'x': dict(dtype='BF16',shape=[1],data_offsets=offsets)}).encode()
                f = root/'bad.safetensors'; f.write_bytes(struct.pack('<Q',len(h))+h+b'\0\0')
                with self.subTest(offsets=offsets), self.assertRaises(ValueError): d.read_safetensors(f)

    def test_converter_does_not_replace_its_source(self):
        with tempfile.TemporaryDirectory() as folder:
            source,_ = self.source(Path(folder)); before=source.read_bytes()
            r = subprocess.run([sys.executable,d.__file__,str(source),'-o',str(source)],capture_output=True)
            self.assertNotEqual(r.returncode,0)
            self.assertEqual(source.read_bytes(),before)

    def test_plain_dflash_contract_retained(self):
        cfg = dict(CONFIG); cfg.pop('architectures'); cfg.update(cfg.pop('dflash_config')); cfg['sample_from_anchor'] = True
        self.assertEqual(len(d.expected(cfg)), 3 + 11 * cfg['num_hidden_layers'])
        self.assertIs(d.normalize_config(cfg), cfg)
        cfg['sample_from_anchor'] = False
        with self.assertRaisesRegex(ValueError, 'anchor'): d.normalize_config(cfg)

if __name__ == '__main__': unittest.main()
