"""Create the synthetic actual-width Q8 regression fixture (about 4.5 GB).
Requires numpy and a C++17 compiler. No GPU or model download is needed.
"""
from pathlib import Path
import argparse, hashlib, json, subprocess, sys, tempfile
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools import gguf_writer as writer
writer.TYPE_IDS['Q8_0'] = 8
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--cxx', default='g++')
a = ap.parse_args()
CPP = r'''
#include "strata/core/family_layout.hpp"
#include <iostream>
int main() {
    strata::core::ModelGeometry g;
    g.family=strata::ModelFamily::Qwen35Moe;
    g.n_layers=4;g.n_mtp_layers=1;g.n_embd=2048;g.n_vocab=256;
    g.n_head=16;g.n_head_kv=2;g.head_dim=256;g.n_expert=256;g.n_expert_used=8;
    g.n_ff=g.n_ff_shexp=512;g.ssm_state_size=128;g.ssm_k_heads=16;g.ssm_v_heads=32;
    g.ssm_value_dim=4096;g.ssm_d_conv=4;g.ssm_conv_channels=8192;g.qsa_interval=4;
    g.hc=g.hc_lr=0;g.idx_q_heads=g.idx_key_dim=0;
    auto emit=[](const std::string& prefix,const auto& ts) {
        for(const auto& t:ts) {std::cout<<prefix<<t.name;for(auto d:t.shape)std::cout<<" "<<d;std::cout<<"\n";}
    };
    emit("",strata::core::global_tensors(g));
    for(int i=0;i<5;i++)emit("blk."+std::to_string(i)+".",strata::core::layer_tensors(g,i));
}
'''
with tempfile.TemporaryDirectory(prefix='sage-fixture-') as tmp:
    tmp = Path(tmp)
    (tmp/'layout.cpp').write_text(CPP)
    subprocess.run([a.cxx, '-O2', '-std=c++17', '-I'+str(ROOT/'include'), str(tmp/'layout.cpp'), '-o', str(tmp/'layout')], check=True)
    table = subprocess.check_output([str(tmp/'layout')], text=True)
layout = [(line.split()[0], list(map(int, line.split()[1:]))) for line in table.splitlines()]
meta = {'block_count': 5,
 'embedding_length': 2048,
 'attention.head_count': 16,
 'attention.head_count_kv': 2,
 'attention.key_length': 256,
 'attention.value_length': 256,
 'expert_count': 256,
 'expert_used_count': 8,
 'expert_feed_forward_length': 512,
 'expert_shared_feed_forward_length': 512,
 'ssm.conv_kernel': 4,
 'ssm.state_size': 128,
 'ssm.group_count': 16,
 'ssm.time_step_rank': 32,
 'ssm.inner_size': 4096,
 'full_attention_interval': 4,
 'nextn_predict_layers': 1,
 'rope.dimension_count': 64,
 'rope.freq_base': 10000000.0,
 'attention.layer_norm_rms_epsilon': 1e-06}
rng = np.random.default_rng(314159)
gdn_rng = np.random.default_rng(20261021)
w = writer.GGUFWriter()
w.add('general.architecture', 'qwen35moe')
w.add('general.name', 'synthetic-qwen35moe-Q8_0')
w.add('tokenizer.ggml.tokens', [f't{i}' for i in range(256)])
for key, value in meta.items(): w.add('qwen35moe.'+key, value)
# GDN projections use their own seed. Advance the base stream by a fixed block
# for them so their larger shapes do not change the other seeded tensors.
for name, shape in layout:
    size = int(np.prod(shape))
    gdn_projection = name.endswith('attn_gate.weight') or name.endswith('ssm_out.weight')
    values = rng.normal(0, .025, 2048*128 if gdn_projection else size).astype(np.float32)
    if gdn_projection: values = gdn_rng.normal(0, .025, size).astype(np.float32)
    if 'norm' in name: values.fill(1)
    if name.endswith('.ssm_a') or name.endswith('ssm_dt.bias'): values.fill(-1)
    f32 = len(shape)==1 or any(x in name for x in ['ssm_conv1d','ssm_alpha','ssm_beta','ffn_gate_inp'])
    kind = 'F32' if f32 else 'Q8_0'
    if f32: raw = values.astype('<f4').tobytes()
    else:
        blocks = values.reshape(-1,32)
        scales = np.max(np.abs(blocks),axis=1)/127
        codes = np.clip(np.rint(blocks/scales[:,None]),-127,127).astype(np.int8)
        encoded = np.empty((len(scales),34),np.uint8)
        encoded[:,:2] = scales.astype('<f2').view(np.uint8).reshape(-1,2)
        encoded[:,2:] = codes.view(np.uint8)
        raw = encoded.tobytes()
    w.tensors.append(writer._Tensor(name,shape,kind,raw))
a.out.parent.mkdir(parents=True, exist_ok=True)
w.write(a.out)
with a.out.open('rb') as f: digest = hashlib.file_digest(f, 'sha256').hexdigest()
print(json.dumps({'bytes':a.out.stat().st_size, 'sha256':digest}, indent=2))
