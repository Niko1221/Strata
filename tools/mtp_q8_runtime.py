"""Adapt the existing shared Q8 GGUF draft head to Strata's Q2 runtime.

Target-model weights are never changed. This adapter only accepts a blk.48 Q8 draft whose norms already include
the Gemma +1 offset. It preserves those norms; it does not add the offset.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from _paths import add_gguf_py
add_gguf_py()
import gguf
from mtp_pack import q2_0, dequant
from mtp_rt import blob_of, H, FF, NE

def bf16(x):
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype(np.uint16)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gguf', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--norms-already-offset', action='store_true',
                    help='Confirm that this GGUF already includes the Gemma +1 norm offset')
    a = ap.parse_args()
    if not a.norms_already_offset:
        ap.error('This adapter requires --norms-already-offset; use mtp_rt.py for raw-norm drafts')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    reader = gguf.GGUFReader(a.gguf)
    ts = {t.name.removeprefix('blk.48.'): t for t in reader.tensors}
    for name in ('ffn_gate_exps.weight', 'ffn_up_exps.weight', 'ffn_down_exps.weight'):
        if name not in ts or int(ts[name].tensor_type) != 8:
            raise ValueError('Expected blk.48 Q8_0 expert tensors: ' + name)
    report = {'source_name': Path(a.gguf).name, 'target_unchanged': True,
              'expert_conversion': 'Q8_0 -> Q2_0 (Strata MSE quantizer)',
              'norms': 'copied, GGUF Gemma +1 already applied', 'expert_error': []}
    with (out / 'experts.bin').open('wb') as f:
        for e in range(NE):
            floats = [gguf.dequantize(np.asarray(ts[n].data[e]), ts[n].tensor_type)
                      for n in ('ffn_gate_exps.weight', 'ffn_up_exps.weight', 'ffn_down_exps.weight')]
            gu = q2_0(np.concatenate(floats[:2])).reshape(2*FF, H//64*18)
            dn = q2_0(floats[2]).reshape(H, FF//64*18)
            f.write(blob_of(gu, dn))
            if e < 8:
                ref = np.concatenate([x.reshape(-1) for x in floats])
                got = np.concatenate([dequant('q2_0', gu, 2*FF*H), dequant('q2_0', dn, FF*H)])
                report['expert_error'].append(float(np.sqrt(np.mean((ref-got)**2))/np.sqrt(np.mean(ref**2))))
            if e % 32 == 0:
                print(f'Expert {e}/{NE}', flush=True)
    mapping = {
        'attn_q.weight': 'self_attn.q_proj.weight', 'attn_k.weight': 'self_attn.k_proj.weight',
        'attn_v.weight': 'self_attn.v_proj.weight', 'attn_output.weight': 'self_attn.o_proj.weight',
        'attn_q_norm.weight': 'self_attn.q_norm.weight', 'attn_k_norm.weight': 'self_attn.k_norm.weight',
        'ffn_down_shexp.weight': 'mlp.shared_expert.down_proj.weight',
        'ffn_gate_shexp.weight': 'mlp.shared_expert.gate_proj.weight',
        'ffn_up_shexp.weight': 'mlp.shared_expert.up_proj.weight',
        'ffn_gate_inp.weight': 'mlp.gate.weight', 'ffn_gate_inp_shexp.weight': 'mlp.shared_expert_gate.weight',
        'nextn.enorm.weight': 'pre_fc_norm_embedding.weight',
        'nextn.hnorm.weight': 'pre_fc_norm_hidden.weight',
    }
    for gg, rt in [('hc_attn', 'attn_hyper_connection'), ('hc_ffn', 'mlp_hyper_connection'),
                   ('nextn.hc_head', 'hyper_connection_mixer')]:
        for suffix, name in [('down','input_mix_weight_down'), ('up','input_mix_weight_up'),
                             ('norm','hc_norm')]:
            mapping[f'{gg}_{suffix}.weight'] = f'{rt}.{name}.weight'
        if not gg.startswith('nextn'):
            mapping[f'{gg}_inject.weight'] = f'{rt}.block_inject_weight.weight'
    entries = []
    for src, dest in mapping.items():
        t = ts[src]
        shape = tuple(int(x) for x in t.shape[::-1])
        rows, cols = (1, shape[0]) if len(shape)==1 else shape
        if int(t.tensor_type)==8 and ('self_attn.' in dest or 'mlp.shared_expert.' in dest):
            raw, kind = np.asarray(t.data).tobytes(), 'q8_0'
        else:
            x = gguf.dequantize(np.asarray(t.data), t.tensor_type)
            if 'norm' in dest:
                raw, kind = x.astype(np.float32).tobytes(), 'f32'
            else:
                raw, kind = bf16(x).tobytes(), 'bf16'
        entries.append((dest, kind, rows, cols, raw))
    # The GGUF combiner fuses [W_embedding | W_hidden] along columns.
    t = ts['nextn.eh_proj.weight']
    assert int(t.tensor_type)==8 and tuple(t.shape)==(2*H,H)
    fused = np.asarray(t.data).reshape(H,2,H//32*34)
    for i, name in enumerate(('fc_embedding.weight','fc_hidden.weight')):
        entries.append((name,'q8_0',H,H,np.ascontiguousarray(fused[:,i,:]).tobytes()))
    off, lines = 0, []
    with (out/'dense.bin').open('wb') as f:
        for name, kind, rows, cols, raw in entries:
            pad = (-off)%256
            f.write(b'\0'*pad)
            off += pad
            lines.append(f'{name} {kind} {rows} {cols} {off} {len(raw)}')
            f.write(raw)
            off += len(raw)
    (out/'dense.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    report['files'] = {p.name: {'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                       for p in out.iterdir() if p.is_file()}
    (out/'conversion-local.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'Done: {len(entries)} dense tensors, {NE} experts, {off} dense bytes',flush=True)

if __name__=='__main__': main()
