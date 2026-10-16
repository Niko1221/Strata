"""Opt-in text conversion of Qwen3.8-Flash-Next FP4-Local (CSF / mixed ModelOpt).

    python tools/nvfp4_csf_convert.py --model CHECKPOINT --dry-run
    python tools/nvfp4_csf_convert.py --model CHECKPOINT --outfile main.gguf --ple-outfile ple-bf16.gguf

Routed experts retain their NVFP4 codes/scales. Dense MXFP8 and non-expert
NVFP4 weights are decoded before nvfp4_convert.py's BF16/F16/Q8_0 policy.
The separate PLE table is decoded in row chunks and rounded to BF16. Vision
and MTP are excluded. No runtime, installer, or default conversion is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from nvfp4_csf import decode_scale_bytes, dequant_mxfp8, dequant_nvfp4

DTYPES = {'U8': '<u1', 'U32': '<u4', 'F32': '<f4', 'F8_E4M3': '<u1', 'F8_E4M3FN': '<u1'}


class Checkpoint:
    """Index/header access and bounded reads; payloads are never cached."""
    def __init__(self, model):
        self.root = Path(model).resolve()
        self.config = json.loads((self.root/'config.json').read_text(encoding='utf-8'))
        self.weights = json.loads((self.root/'model.safetensors.index.json').read_text(encoding='utf-8'))['weight_map']
        if not isinstance(self.weights, dict) or not self.weights:
            raise ValueError('checkpoint weight_map must be a non-empty object')
        self.headers = {}
        for shard in set(self.weights.values()):
            path = (self.root/shard).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                raise ValueError(f'checkpoint-local shard is missing: {shard}')

    def descriptor(self, name):
        path = self.root/self.weights[name]
        if path not in self.headers:
            with path.open('rb') as f:
                prefix = f.read(8)
                if len(prefix) != 8:
                    raise ValueError(f'short safetensors header: {path}')
                length, = struct.unpack('<Q', prefix)
                if not 0 < length <= min(100 << 20, path.stat().st_size-8):
                    raise ValueError(f'invalid safetensors header length: {path}')
                self.headers[path] = (8+length, json.loads(f.read(length)))
        offset, header = self.headers[path]
        desc = header[name]
        if desc['dtype'] not in DTYPES:
            raise ValueError(f'unsupported raw dtype for {name}: {desc["dtype"]}')
        shape = desc['shape']
        begin, end = desc['data_offsets']
        nbytes = int(np.prod(shape, dtype=np.int64))*np.dtype(DTYPES[desc['dtype']]).itemsize
        if begin < 0 or end-begin != nbytes or offset+end > path.stat().st_size:
            raise ValueError(f'invalid tensor byte range: {name}')
        return path, offset+begin, desc

    def array(self, name, start=None, rows=None):
        path, offset, desc = self.descriptor(name)
        dtype = np.dtype(DTYPES[desc['dtype']])
        shape = desc['shape']
        size = int(np.prod(shape, dtype=np.int64))*dtype.itemsize
        if start is not None:
            if len(shape) != 2 or rows is None or start < 0 or rows <= 0 or start+rows > shape[0]:
                raise ValueError('invalid tensor row slice')
            stride = shape[1]*dtype.itemsize
            offset += start*stride
            size, shape = rows*stride, [rows, shape[1]]
        with path.open('rb') as f:
            f.seek(offset)
            raw = f.read(size)
        if len(raw) != size:
            raise ValueError(f'short tensor read: {name}')
        return np.frombuffer(raw, dtype=dtype).reshape(shape)


def ple_plan(checkpoint):
    shards = {}
    for name in checkpoint.weights:
        match = re.search(r'\.ngram_embedding\.shard_(\d+)\.weight$', name)
        if match:
            index = int(match[1])
            if index in shards:
                raise ValueError('more than one PLE table is not supported')
            shards[index] = name
    text = checkpoint.config.get('text_config', checkpoint.config)
    count = text.get('split_ngram_parts')
    if not isinstance(count, int) or sorted(shards) != list(range(count)) or not shards:
        raise ValueError('PLE shard inventory does not match split_ngram_parts')
    scale_name = next(iter(shards.values())).split('.shard_')[0]+'.weight_scale_2'
    _, _, scale_desc = checkpoint.descriptor(scale_name)
    if scale_desc['dtype'] != 'F32' or scale_desc['shape'] not in ([], [1]):
        raise ValueError('PLE global weight_scale_2 must be one F32 value')
    rows = 0
    for name in shards.values():
        _, _, weight = checkpoint.descriptor(name)
        _, _, scales = checkpoint.descriptor(name.removesuffix('.weight')+'.weight_scale')
        if weight['dtype'] != 'U8' or len(weight['shape']) != 2 or weight['shape'][1] != 80:
            raise ValueError(f'PLE must have 160 unpacked values per row: {name}')
        if scales['dtype'] not in ('F8_E4M3', 'F8_E4M3FN') or scales['shape'] != [weight['shape'][0], 10]:
            raise ValueError(f'PLE block scale shape/dtype is wrong: {name}')
        rows += weight['shape'][0]
    return {'shards': shards, 'scale': scale_name, 'rows': rows, 'row_width': 160,
            'payload_bytes': rows*160*2}


def conversion_plan(checkpoint):
    config = checkpoint.config
    quant = config.get('quantization_config', {})
    recipes = quant.get('quantized_layers', {})
    if config.get('model_type') != 'qwen4_exp' or quant.get('quant_method') != 'modelopt':
        raise ValueError('this converter supports ModelOpt Qwen4Exp checkpoints only')
    if not any(recipe.get('weight_scale_encoding') == 'csf' for recipe in recipes.values()):
        raise ValueError('no CSF quantization recipe found; use nvfp4_convert.py for ordinary NVFP4')
    for name, recipe in recipes.items():
        if recipe.get('quant_algo') not in ('NVFP4', 'W4A16_NVFP4', 'MXFP8'):
            raise ValueError(f'unsupported quantization recipe: {name}: {recipe}')
        group = 32 if recipe['quant_algo'] == 'MXFP8' else 16
        if recipe.get('group_size') != group or recipe.get('weight_scale_encoding') not in (None, 'csf'):
            raise ValueError(f'unsupported block size or scale encoding: {name}: {recipe}')
        if recipe.get('weight_scale_encoding') == 'csf' and (recipe['quant_algo'] != 'NVFP4' or not name.endswith('.mlp.experts')):
            raise ValueError(f'CSF is supported only for routed expert NVFP4 scales: {name}')
    fixed = {n.removesuffix('.nvfp4_csf_fixed') for n in checkpoint.weights if n.endswith('.nvfp4_csf_fixed')}
    exceptions = {n.removesuffix('.nvfp4_csf_exceptions') for n in checkpoint.weights if n.endswith('.nvfp4_csf_exceptions')}
    if not fixed or fixed != exceptions:
        raise ValueError('CSF scale inventory has an unpaired stream')
    ple = ple_plan(checkpoint)
    return {'model': str(checkpoint.root), 'csf_matrices': sum(n.endswith('.nvfp4_csf_fixed') for n in checkpoint.weights),
            'ple_shards': len(ple['shards']), 'ple_rows': ple['rows'], 'ple_payload_bytes': ple['payload_bytes'],
            'routed_experts': 'NVFP4 codes/scales retained; lossless scale decoding',
            'dense': 'MXFP8/NVFP4 decoded, then existing BF16/F16/Q8_0 conversion policy',
            'ple': 'NVFP4 decoded in row chunks, rounded to BF16', 'vision': 'excluded', 'mtp': 'excluded'}


def bf16_bytes(values):
    """Round finite float32 to BF16, nearest with ties to even."""
    if not np.isfinite(values).all():
        raise ValueError('non-finite PLE value')
    bits = np.ascontiguousarray(values, dtype='<f4').view('<u4')
    rounded = ((bits+np.uint32(0x7FFF)+((bits >> 16) & 1)) >> 16).astype('<u2')
    if np.any((rounded & 0x7F80) == 0x7F80):
        raise ValueError('PLE value overflows BF16')
    return rounded.tobytes()


def write_ple(checkpoint, output, chunk_rows=16384):
    if chunk_rows <= 0:
        raise ValueError('chunk_rows must be positive')
    plan = ple_plan(checkpoint)
    output = Path(output)
    if output.exists():
        raise ValueError(f'refusing to overwrite {output}')
    def string(value):
        encoded = value.encode('utf-8')
        return struct.pack('<Q', len(encoded))+encoded
    kvs = [string('general.architecture')+struct.pack('<I', 8)+string('strata-ple'),
           string('general.name')+struct.pack('<I', 8)+string('NVFP4 PLE decoded to BF16')]
    head = b'GGUF'+struct.pack('<IQQ', 3, 1, len(kvs))+b''.join(kvs)
    head += string('per_layer_token_embd.weight')+struct.pack('<IQQIQ', 2, 160, plan['rows'], 30, 0)
    head += b'\0'*(-len(head) % 32)
    scale = checkpoint.array(plan['scale']).item()
    temporary = output.with_suffix(output.suffix+'.part')
    output.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open('xb') as out:
        out.write(head)
        for number, name in sorted(plan['shards'].items()):
            count = checkpoint.descriptor(name)[2]['shape'][0]
            for start in range(0, count, chunk_rows):
                rows = min(chunk_rows, count-start)
                weight = checkpoint.array(name, start, rows)
                scales = checkpoint.array(name.removesuffix('.weight')+'.weight_scale', start, rows)
                out.write(bf16_bytes(dequant_nvfp4(weight, scales, scale)))
            print(f'PLE shard {number+1}/{len(plan["shards"])}', flush=True)
    if temporary.stat().st_size != len(head)+plan['payload_bytes']:
        raise ValueError('PLE output size mismatch')
    temporary.rename(output)


@contextmanager
def converter_policy(checkpoint):
    """Scope the compatibility adapters to this explicit conversion only."""
    import nvfp4_convert as legacy
    import torch
    from conversion.base import LazyTorchTensor, ModelBase, TextModel
    from conversion.qwen4exp import Qwen4ExpTextModel as Model

    original_parts = ModelBase.get_model_part_names
    original_prepare = Model.prepare_tensors
    original_force, original_place = Model.tensor_force_quant, Model._place_ple_shard
    original_u32 = LazyTorchTensor._dtype_str_map.get('U32')
    original_u32_swap = LazyTorchTensor._dtype_byteswap_map.get(torch.uint32)
    recipes = checkpoint.config['quantization_config']['quantized_layers']
    normalized_recipes = {}
    for name, recipe in recipes.items():
        item = TextModel.filter_tensors((name+'.weight', lambda: None))
        if item:
            normalized_recipes[item[0].removesuffix('.weight')] = recipe
    def recipe_for(name):
        while name:
            if name in normalized_recipes:
                return normalized_recipes[name]
            name = name.rpartition('.')[0]
        return None
    def parts(directory, prefix, suffix):
        if Path(directory).resolve() == checkpoint.root and suffix == '.safetensors':
            return sorted(set(checkpoint.weights.values()))
        return original_parts(directory, prefix, suffix)
    def eager(gen):
        return LazyTorchTensor.to_eager(gen()).cpu().contiguous()
    def raw(gen):
        return eager(gen).view(torch.uint8).numpy()
    def prepare(self):
        tensors = self.model_tensors
        # The legacy converter writes no PLE payload. Record its logical row
        # width without materializing the 51.2-billion-value table.
        ple = ple_plan(checkpoint)
        self._ple_shards = dict(ple['shards'])
        self._ple_row_dim = ple['row_width']
        for name in list(tensors):
            if name.startswith('mtp.') or '.ngram_embedding.' in name:
                tensors.pop(name)
        fixed_suffix, exception_suffix = '.nvfp4_csf_fixed', '.nvfp4_csf_exceptions'
        fixed = {n.removesuffix(fixed_suffix): n for n in tensors if n.endswith(fixed_suffix)}
        exceptions = {n.removesuffix(exception_suffix): n for n in tensors if n.endswith(exception_suffix)}
        if fixed.keys() != exceptions.keys():
            raise ValueError('CSF scale stream without its partner')
        for scale, name in fixed.items():
            if scale in tensors:
                raise ValueError(f'both compressed and uncompressed scale found: {scale}')
            fg, eg = tensors.pop(name), tensors.pop(exceptions[scale])
            def load_scale(fg=fg, eg=eg):
                result = decode_scale_bytes(raw(fg), eager(eg).numpy())
                if np.any(result >= 127):
                    raise ValueError('CSF NVFP4 scales must be finite and non-negative E4M3')
                return torch.from_numpy(result).view(torch.float8_e4m3fn)
            tensors[scale] = load_scale
        for name in list(tensors):
            if not name.endswith('.weight'):
                continue
            stem = name.removesuffix('.weight')
            recipe = recipe_for(stem)
            if not recipe or ('.experts.' in name and recipe['quant_algo'] == 'NVFP4'):
                continue
            if stem+'.weight_scale' not in tensors:
                raise ValueError(f'quantized weight has no block scale: {name}')
            wg, sg = tensors[name], tensors.pop(stem+'.weight_scale')
            if recipe['quant_algo'] == 'MXFP8':
                def load(wg=wg, sg=sg):
                    return torch.from_numpy(dequant_mxfp8(raw(wg), raw(sg)))
            else:
                global_scale = tensors.pop(stem+'.weight_scale_2', None)
                if global_scale is None:
                    raise ValueError(f'NVFP4 weight has no global scale: {name}')
                def load(wg=wg, sg=sg, gg=global_scale):
                    return torch.from_numpy(dequant_nvfp4(raw(wg), raw(sg), eager(gg).item()))
            tensors[name] = load
            tensors.pop(stem+'.input_scale', None)
        # The checkpoint distributes each layer's experts over many shards.
        # Group names by layer before the pinned converter buffers/re-packs
        # experts, so it can flush a layer instead of retaining the whole model.
        self.model_tensors = dict(sorted(tensors.items()))
        return original_prepare(self)
    try:
        legacy.install_policy()
        ModelBase.get_model_part_names = staticmethod(parts)
        LazyTorchTensor._dtype_str_map['U32'] = torch.uint32
        LazyTorchTensor._dtype_byteswap_map[torch.uint32] = np.uint32
        Model.prepare_tensors = prepare
        yield
    finally:
        ModelBase.get_model_part_names = staticmethod(original_parts)
        Model.prepare_tensors = original_prepare
        Model.tensor_force_quant, Model._place_ple_shard = original_force, original_place
        if original_u32 is None:
            LazyTorchTensor._dtype_str_map.pop('U32', None)
        else:
            LazyTorchTensor._dtype_str_map['U32'] = original_u32
        if original_u32_swap is None:
            LazyTorchTensor._dtype_byteswap_map.pop(torch.uint32, None)
        else:
            LazyTorchTensor._dtype_byteswap_map[torch.uint32] = original_u32_swap


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--model', required=True)
    parser.add_argument('--outfile', help='main text GGUF, then pack with tools/iq_pack.py')
    parser.add_argument('--ple-outfile', help='separate BF16 PLE GGUF (~102.4 GB for FP4-Local)')
    parser.add_argument('--chunk-rows', type=int, default=16384)
    parser.add_argument('--dry-run', action='store_true', help='metadata-only plan; read no tensor payloads or write outputs')
    args = parser.parse_args()
    try:
        checkpoint = Checkpoint(args.model)
        plan = conversion_plan(checkpoint)
        if args.chunk_rows <= 0:
            raise ValueError('--chunk-rows must be positive')
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return 0
        outputs = [Path(p).resolve() for p in [args.outfile, args.ple_outfile] if p]
        if not outputs or len(set(outputs)) != len(outputs):
            raise ValueError('choose distinct --outfile and/or --ple-outfile')
        for path in outputs:
            if path.exists() or path.with_suffix(path.suffix+'.manifest.json').exists() or path.is_relative_to(checkpoint.root):
                raise ValueError(f'output must be a new file outside the checkpoint: {path}')
        plan['config_sha256'] = hashlib.sha256((checkpoint.root/'config.json').read_bytes()).hexdigest()
        plan['full_model_serving_validated'] = False
        if args.outfile:
            root = Path(__file__).resolve().parents[1]
            llama = root/'third_party/llama.cpp'
            sys.path[:0] = [str(llama/'gguf-py'), str(llama)]
            import convert_hf_to_gguf
            output = Path(args.outfile).resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix('.part.gguf')
            if temporary.exists():
                raise ValueError(f'partial output already exists: {temporary}')
            previous_argv = sys.argv
            try:
                sys.argv = ['convert_hf_to_gguf.py', str(checkpoint.root), '--outtype', 'q8_0',
                            '--outfile', str(temporary), '--use-temp-file']
                with converter_policy(checkpoint):
                    result = convert_hf_to_gguf.main()
                if result:
                    return result
            finally:
                sys.argv = previous_argv
            temporary.rename(output)
            output.with_suffix(output.suffix+'.manifest.json').write_text(json.dumps(plan, indent=2)+'\n')
        if args.ple_outfile:
            write_ple(checkpoint, args.ple_outfile, args.chunk_rows)
            path = Path(args.ple_outfile)
            path.with_suffix(path.suffix+'.manifest.json').write_text(json.dumps(plan, indent=2)+'\n')
        return 0
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f'error: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
