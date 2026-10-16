"""CPU converter tests. Tests need numpy and torch; integration also needs
the pinned third_party/llama.cpp Python converter and its dependencies.

    python tools/test_nvfp4_csf_convert.py
"""
from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from nvfp4_csf import decode_scale_bytes, dequant_mxfp8, dequant_nvfp4
from nvfp4_csf_convert import (
    Checkpoint,
    bf16_bytes,
    conversion_plan,
    converter_policy,
    write_ple,
)
from test_nvfp4_csf import encode_reference


def safetensors(path, tensors):
    """Fixture writer retaining the actual float8/uint32 storage dtypes."""
    header, payload, offset = {}, [], 0
    for name, (dtype, array) in tensors.items():
        raw = array.tobytes()
        header[name] = {'dtype': dtype, 'shape': list(array.shape), 'data_offsets': [offset, offset+len(raw)]}
        payload.append(raw)
        offset += len(raw)
    encoded = json.dumps(header, separators=(',', ':')).encode()
    encoded += b' '*(-len(encoded) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack('<Q', len(encoded))+encoded+b''.join(payload))


def fixture(root, *, full=False):
    stem = 'model.language_model.layers.0.mlp.experts'
    recipes = {stem: {'quant_algo': 'NVFP4', 'group_size': 16, 'weight_scale_encoding': 'csf'}}
    tensors = {}
    rng = np.random.default_rng(19)
    for expert in range(2 if full else 1):
        for proj, shape in [('gate_proj', (640, 1280)), ('up_proj', (640, 1280)), ('down_proj', (2560, 320))]:
            if not full:
                shape = (16, 32)
            prefix = f'{stem}.{expert}.{proj}'
            raw = rng.integers(40, 56, (shape[0], shape[1]//8), dtype=np.uint8)
            raw.flat[::41] = 89
            fixed, exceptions = encode_reference(raw)
            tensors[prefix+'.weight'] = ('U8', rng.integers(0, 256, shape, dtype=np.uint8))
            tensors[prefix+'.weight_scale.nvfp4_csf_fixed'] = ('U8', fixed)
            tensors[prefix+'.weight_scale.nvfp4_csf_exceptions'] = ('U32', exceptions)
            tensors[prefix+'.weight_scale_2'] = ('F32', np.array(0.0001, dtype=np.float32))
    dense = 'model.language_model.layers.0.mlp.shared_expert.gate_proj'
    shape = (640, 2560) if full else (16, 64)
    tensors[dense+'.weight'] = ('F8_E4M3', rng.integers(40, 72, shape, dtype=np.uint8))
    tensors[dense+'.weight_scale'] = ('U8', np.full((shape[0], shape[1]//32), 119, dtype=np.uint8))
    recipes[dense] = {'quant_algo': 'MXFP8', 'group_size': 32}
    # Non-expert NVFP4 must be decoded, not sent through the expert pack path.
    embed = 'model.language_model.embed_tokens'
    tensors[embed+'.weight'] = ('U8', rng.integers(0, 256, (32, 1280), dtype=np.uint8))
    tensors[embed+'.weight_scale'] = ('F8_E4M3', np.full((32, 160), 48, dtype=np.uint8))
    tensors[embed+'.weight_scale_2'] = ('F32', np.array(.01, dtype=np.float32))
    recipes[embed] = {'quant_algo': 'NVFP4', 'group_size': 16}
    ple = 'model.language_model.layers.0.ple.ple_embedding.ngram_embedding'
    for i in range(2):
        tensors[f'{ple}.shard_{i}.weight'] = ('U8', rng.integers(0, 256, (17+i, 80), dtype=np.uint8))
        tensors[f'{ple}.shard_{i}.weight_scale'] = ('F8_E4M3', np.full((17+i, 10), 48+i, dtype=np.uint8))
    tensors[ple+'.weight_scale_2'] = ('F32', np.array([.025], dtype=np.float32))
    # MTP and vision are excluded from this text converter.
    tensors['mtp.layers.0.mlp.experts.0.gate_proj.weight'] = ('U8', np.zeros((16, 32), dtype=np.uint8))
    tensors['model.visual.blocks.0.attn.qkv.weight'] = ('F32', np.ones((8, 8), dtype=np.float32))
    config = {'model_type': 'qwen4_exp', 'architectures': ['Qwen4ExpForConditionalGeneration'],
              'text_config': {'num_hidden_layers': 1, 'num_experts': 2, 'num_experts_per_tok': 2, 'split_ngram_parts': 2,
                              'hidden_size': 2560, 'num_attention_heads': 24},
              'quantization_config': {'quant_method': 'modelopt', 'quant_algo': 'MIXED_PRECISION',
                                      'quantized_layers': recipes}}
    root.mkdir(parents=True, exist_ok=True)
    (root/'config.json').write_text(json.dumps(config))
    safetensors(root/'tensors/model-00001-of-00001.safetensors', tensors)
    (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {n: 'tensors/model-00001-of-00001.safetensors' for n in tensors}}))
    return tensors


class Conversion(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model = self.root/'model'
        self.tensors = fixture(self.model)
        self.checkpoint = Checkpoint(self.model)

    def tearDown(self):
        self.temporary.cleanup()

    def test_metadata_plan_reads_no_payloads(self):
        self.checkpoint.array = lambda *a, **k: self.fail('metadata plan read a payload')
        plan = conversion_plan(self.checkpoint)
        self.assertEqual(plan['csf_matrices'], 3)
        self.assertEqual(plan['ple_rows'], 35)
        self.assertEqual(plan['ple_payload_bytes'], 35*160*2)

    def test_streaming_ple_keeps_shard_and_row_order(self):
        a, b = self.root/'small.gguf', self.root/'large.gguf'
        write_ple(self.checkpoint, a, chunk_rows=5)
        write_ple(self.checkpoint, b, chunk_rows=64)
        self.assertEqual(a.read_bytes(), b.read_bytes())
        prefix = 'model.language_model.layers.0.ple.ple_embedding.ngram_embedding'
        expected = b''
        for i in range(2):
            w = self.tensors[f'{prefix}.shard_{i}.weight'][1]
            s = self.tensors[f'{prefix}.shard_{i}.weight_scale'][1]
            expected += bf16_bytes(dequant_nvfp4(w, s, .025))
        self.assertTrue(a.read_bytes().endswith(expected))
        self.assertEqual(a.read_bytes()[:4], b'GGUF')

    def test_rejects_missing_ple_shard(self):
        self.checkpoint.weights.pop(next(n for n in self.checkpoint.weights if '.shard_1.weight' in n))
        with self.assertRaisesRegex(ValueError, 'inventory'):
            conversion_plan(self.checkpoint)

    def test_rejects_unknown_quantization(self):
        first = next(iter(self.checkpoint.config['quantization_config']['quantized_layers'].values()))
        first['quant_algo'] = 'future-codec'
        with self.assertRaisesRegex(ValueError, 'unsupported quantization'):
            conversion_plan(self.checkpoint)

    def test_rejects_unpaired_csf_stream_in_metadata_plan(self):
        name = next(n for n in self.checkpoint.weights if n.endswith('.nvfp4_csf_exceptions'))
        self.checkpoint.weights.pop(name)
        with self.assertRaisesRegex(ValueError, 'unpaired'):
            conversion_plan(self.checkpoint)

    def test_tensor_read_above_four_gib(self):
        shard = self.model/'tensors/sparse.safetensors'
        start = 1 << 32
        name = 'large-offset'
        header = json.dumps({name: {'dtype':'U8', 'shape':[2,2], 'data_offsets':[start,start+4]}}).encode()
        with shard.open('wb') as out:
            out.write(struct.pack('<Q', len(header)))
            out.write(header)
            out.seek(8+len(header)+start)
            out.write(b'\x01\x02\x03\x04')
        self.checkpoint.weights[name] = 'tensors/sparse.safetensors'
        np.testing.assert_array_equal(self.checkpoint.array(name, 1, 1), np.array([[3,4]], dtype=np.uint8))

    def test_does_not_overwrite(self):
        out = self.root/'existing.gguf'
        out.write_bytes(b'keep me')
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            write_ple(self.checkpoint, out)
        self.assertEqual(out.read_bytes(), b'keep me')

    def test_rejects_checkpoint_escape(self):
        index = self.model/'model.safetensors.index.json'
        index.write_text('{"weight_map": {"x": "../../outside.safetensors"}}')
        with self.assertRaisesRegex(ValueError, 'checkpoint-local'):
            Checkpoint(self.model)

    def test_bf16_rounding_matches_torch(self):
        import torch
        values = np.array([0., -0., 1.+2**-8, 1.+3*2**-8, -1.-2**-8, 2**-130, 0.12345], dtype=np.float32)
        expected = torch.from_numpy(values).bfloat16().view(torch.uint16).numpy().tobytes()
        self.assertEqual(bf16_bytes(values), expected)

    def test_mixed_quantization_matches_torch(self):
        import torch
        dense = next(n for n in self.tensors if '.shared_expert.' in n and n.endswith('.weight'))
        w = self.tensors[dense][1]
        s = self.tensors[dense.removesuffix('.weight')+'.weight_scale'][1]
        expected = torch.from_numpy(w.copy()).view(torch.float8_e4m3fn).float()*torch.from_numpy(s.copy()).view(torch.float8_e8m0fnu).float().repeat_interleave(32, dim=1)
        np.testing.assert_array_equal(dequant_mxfp8(w, s), expected.numpy())


class PinnedConverter(unittest.TestCase):
    def test_interleaved_experts_flush_within_one_layer_budget(self):
        llama = Path(__file__).resolve().parents[1]/'third_party/llama.cpp'
        sys.path[:0] = [str(llama/'gguf-py'), str(llama)]
        import gguf
        from conversion.base import ModelBase
        from conversion.qwen4exp import Qwen4ExpTextModel as Model
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tensors = fixture(root/'model', full=True)
            expert_names = [n for n in tensors if '.layers.0.mlp.experts.' in n]
            tensors.update({n.replace('.layers.0.', '.layers.1.'):tensors[n] for n in expert_names})
            # As in the real checkpoint, an early shard has a few experts of
            # every layer; a later shard finishes those same layers.
            shard_paths = ['tensors/first.safetensors', 'tensors/second.safetensors']
            groups = [{}, {}]
            for name, value in tensors.items():
                second = '.mlp.experts.1.' in name
                groups[int(second)][name] = value
            for name, group in zip(shard_paths, groups):
                safetensors(root/'model'/name, group)
            (root/'model/tensors/model-00001-of-00001.safetensors').unlink()
            (root/'model/model.safetensors.index.json').write_text(json.dumps({'weight_map':{
                name:shard for shard, group in zip(shard_paths, groups) for name in group}}))
            config_path = root/'model/config.json'
            config = json.loads(config_path.read_text())
            config['text_config']['num_hidden_layers'] = 2
            recipes = config['quantization_config']['quantized_layers']
            recipes['model.language_model.layers.1.mlp.experts'] = recipes['model.language_model.layers.0.mlp.experts'].copy()
            config_path.write_text(json.dumps(config))
            buffered = []
            flush = ModelBase._flush_nvfp4_experts
            def observe(self, key, blocks, *args):
                buffered.append(sum(len(v) for v in blocks.values()))
                return flush(self, key, blocks, *args)
            checkpoint = Checkpoint(root/'model')
            with patch.object(ModelBase, '_flush_nvfp4_experts', observe), converter_policy(checkpoint):
                model = Model(checkpoint.root, gguf.LlamaFileType.MOSTLY_Q8_0, root/'result.gguf', use_temp_file=True)
                model.prepare_tensors()
                model.gguf_writer.write_header_to_file(root/'result.gguf')
                model.gguf_writer.write_kv_data_to_file()
                model.gguf_writer.write_tensors_to_file()
                model.gguf_writer.close()
            self.assertEqual(len(buffered), 6)
            self.assertLessEqual(max(buffered), 2*3)  # one layer's two experts x three projections
            experts = [t for t in gguf.GGUFReader(root/'result.gguf').tensors if t.name.endswith('_exps.weight')]
            self.assertEqual(len(experts), 6)

    def test_adapters_are_restored_after_an_exception(self):
        import torch
        llama = Path(__file__).resolve().parents[1]/'third_party/llama.cpp'
        sys.path[:0] = [str(llama/'gguf-py'), str(llama)]
        from conversion.base import LazyTorchTensor, ModelBase
        from conversion.qwen4exp import Qwen4ExpTextModel as Model
        original = (ModelBase.get_model_part_names, Model.prepare_tensors, Model.tensor_force_quant,
                    Model._place_ple_shard, LazyTorchTensor._dtype_str_map.get('U32'),
                    LazyTorchTensor._dtype_byteswap_map.get(torch.uint32))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root/'model')
            checkpoint = Checkpoint(root/'model')
            with self.assertRaisesRegex(RuntimeError, 'test conversion failure'), converter_policy(checkpoint):
                raise RuntimeError('test conversion failure')
        restored = (ModelBase.get_model_part_names, Model.prepare_tensors, Model.tensor_force_quant,
                    Model._place_ple_shard, LazyTorchTensor._dtype_str_map.get('U32'),
                    LazyTorchTensor._dtype_byteswap_map.get(torch.uint32))
        self.assertEqual(original, restored)

    def test_real_converter_writes_experts_dense_and_metadata_without_loading_ple(self):
        import torch
        llama = Path(__file__).resolve().parents[1]/'third_party/llama.cpp'
        sys.path[:0] = [str(llama/'gguf-py'), str(llama)]
        import gguf
        from conversion.base import LazyTorchTensor, ModelBase
        from conversion.qwen4exp import Qwen4ExpTextModel as Model
        original = (ModelBase.get_model_part_names, Model.prepare_tensors, Model.tensor_force_quant,
                    Model._place_ple_shard, LazyTorchTensor._dtype_str_map.get('U32'),
                    LazyTorchTensor._dtype_byteswap_map.get(torch.uint32))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tensors = fixture(root/'model', full=True)
            checkpoint = Checkpoint(root/'model')
            with converter_policy(checkpoint):
                model = Model(checkpoint.root, gguf.LlamaFileType.MOSTLY_Q8_0, root/'result.gguf', use_temp_file=True)
                model.prepare_tensors()
                model.gguf_writer.write_header_to_file(root/'result.gguf')
                model.gguf_writer.write_kv_data_to_file()
                model.gguf_writer.write_tensors_to_file()
                model.gguf_writer.close()
            restored = (ModelBase.get_model_part_names, Model.prepare_tensors, Model.tensor_force_quant,
                        Model._place_ple_shard, LazyTorchTensor._dtype_str_map.get('U32'),
                        LazyTorchTensor._dtype_byteswap_map.get(torch.uint32))
            self.assertEqual(original, restored)
            output = {t.name: t for t in gguf.GGUFReader(root/'result.gguf').tensors}
            self.assertNotIn('per_layer_token_embd.weight', output)
            self.assertEqual(model._ple_row_dim, 160)
            self.assertEqual(len(model._ple_shards), 2)
            self.assertFalse(any(n.startswith('mtp.') or 'ngram_embedding' in n for n in model.model_tensors))
            for proj, target in [('gate_proj', 'gate'), ('up_proj', 'up'), ('down_proj', 'down')]:
                data = output[f'blk.0.ffn_{target}_exps.weight']
                self.assertEqual(data.tensor_type, gguf.GGMLQuantizationType.NVFP4)
                self.assertEqual(int(data.shape[-1]), 2)
                self.assertIn(f'blk.0.ffn_{target}_exps.scale', output)
                for expert in range(2):
                    stem = f'model.language_model.layers.0.mlp.experts.{expert}.{proj}'
                    fixed = tensors[stem+'.weight_scale.nvfp4_csf_fixed'][1]
                    exceptions = tensors[stem+'.weight_scale.nvfp4_csf_exceptions'][1]
                    scales = torch.from_numpy(decode_scale_bytes(fixed, exceptions)).view(torch.float8_e4m3fn)
                    expected, _ = ModelBase._nvfp4_pack(torch.from_numpy(tensors[stem+'.weight'][1]), scales)
                    np.testing.assert_array_equal(data.data[expert], expected)
            self.assertEqual(output['blk.0.ffn_gate_shexp.weight'].tensor_type, gguf.GGMLQuantizationType.Q8_0)
            self.assertEqual(output['token_embd.weight'].tensor_type, gguf.GGMLQuantizationType.Q8_0)
            stem = 'model.language_model.embed_tokens'
            decoded = dequant_nvfp4(tensors[stem+'.weight'][1], tensors[stem+'.weight_scale'][1], .01)
            expected = gguf.quants.quantize(decoded, gguf.GGMLQuantizationType.Q8_0)
            np.testing.assert_array_equal(output['token_embd.weight'].data, expected)


if __name__ == '__main__':
    unittest.main()
