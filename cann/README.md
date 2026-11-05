# `cann/` - the Ascend port of the engine

This directory is to the CANN backend what `sycl/` is to the Intel backend: a separate project
that grows next to the CUDA engine until it can take the engine's sources.

| phase | scope | state |
|---|---|---|
| P1 | device, stream, HBM buffers, first aclnn ops, CPU oracle | done, parity on hardware |
| P2 | Q4_K / Q5_K / Q6_K / Q8_0 decoders, dequantize then matmul on the NPU | done, bit-exact vs ggml |
| P2b | GGUF reader, real model file, real-weight GEMV on the NPU | done |
| P2c | one real block from real weights: blk.0 shared expert (SwiGLU) | done |
| P3 | GDN op, the GDN layer block, and state carried across tokens | done |
| P3b | rope, the full-attention layer of blk.3, and its KV cache across tokens | done |
| P4 | MoE: top-8 routing, expert matmuls, shared expert | done for one layer's FFN |
| P5 | `serve/` wiring, 18789, benchmarks | not started |
| P4b | the whole model: 40 layers from a token id to logits | done; matches llama.cpp's top-5 |

## Files

| file | what |
|---|---|
| `include/strata/cann/device.hpp` | `Device`, `Stream`, `Buffer`, `DeviceInfo`, `Error`; one place that turns `aclError` into an exception |
| `include/strata/cann/ops.hpp` | `matmul`, `rms_norm`, `silu`, `add`, `sub`, `mul`, `muls`, `exp`, `softmax` on fp16 device tensors, each with a CPU oracle |
| `include/strata/cann/quants.hpp` | `block_q8_0` (34 B), `block_q4_K` (144 B), `block_q5_K` (176 B), `block_q6_K` (210 B), plus `dequantize()` |
| `include/strata/cann/gdn.hpp` | gated delta net: float reference and the device path |
| `include/strata/cann/gguf.hpp` | GGUF v3 reader: header, metadata, tensor index, on-demand tensor bytes |
| `include/strata/cann/tensor_load.hpp` | any tensor as floats: F32 / F16 / BF16 / Q8_0 / Q4_K / Q5_K / Q6_K |
| `src/` | the same, implemented: aclnn two-phase calls, decoders, GDN, reader, loader |
| `tests/parity.cpp` | random fp16 inputs, device result against the CPU reference |
| `tests/quant_parity.cpp` | decoders against ggml's `to_float`, then a dequantized matmul on the NPU |
| `tests/gdn_parity.cpp` | GDN against the float reference, both decay modes |
| `tests/gguf_parity.cpp` | the real 22.85 GB model: header, weight mix, real GEMV, real-tensor decoders |
| `tests/layer_parity.cpp` | one real block on the NPU: blk.0 shared expert, `silu(W_gate x) * (W_up x)` then `W_down` |
| `tests/moe_parity.cpp` | the whole FFN of blk.0: router, top-8, 8 expert slabs, weighted sum, shared expert |
| `tests/gdn_block_parity.cpp` | the whole GDN block of blk.0: qkv, short conv, L2 norms, delta net, gated norm, ssm_out |
| `tests/gdn_state_parity.cpp` | the same block across T tokens, with the conv history and the 128x128 per-head state carried on the device |
| `tests/rope_parity.cpp` | rotary embedding (NeoX pairing, n_rot of head_dim) against the formula |
| `tests/attn_layer_parity.cpp` | the full-attention layer of blk.3: q/k/v, per-head q/k norms, rope, gate, output projection |
| `tests/attn_cache_parity.cpp` | the same layer across T tokens with a KV cache: score, softmax, GQA broadcast, weighted mix |
| `tests/model_forward.cpp` | the whole stack: token id to logits, NPU against a CPU reference |
| `tests/model_seq.cpp` | the whole stack over a prompt, with the KV cache and the recurrent state carried across tokens |
| `tests/run_*.sh` | build and run each test |

## Build and run

```bash
# P1 only
cmake -S cann -B build-cann && cmake --build build-cann -j
./build-cann/cann_parity

# with ggml's decoders as the oracle (needed by quant_parity and gguf_parity)
cmake -S cann -B build-cann \
  -DSTRATA_CANN_GGML_DIR=/path/to/llama.cpp \
  -DSTRATA_CANN_GGML_LIB_DIR=/path/to/llama.cpp/build/bin
cmake --build build-cann -j
./build-cann/quant_parity
./build-cann/gguf_parity /path/to/model.gguf
./build-cann/layer_parity /path/to/model.gguf
```

`ASCEND_HOME_PATH` defaults to `/usr/local/Ascend/cann-9.0.0`. The ggml oracle links
`libggml-base.so` for the tests only; nothing in the port depends on ggml.

## Tolerances

Inputs are rounded through fp16 before the CPU oracle runs, because that is what the device
receives.

| check | tolerance |
|---|---|
| matmul 256x512x384 | 0.5 (512-term fp16 dot) |
| rms_norm 512x2048 / silu / add / softmax | 0.02 / 0.02 / 0.01 / 0.01 |
| Q8_0 / Q4_K / Q5_K / Q6_K decode | bit-exact against ggml |
| GDN, one head | attn 0.05, state 0.05 |
| real-weight GEMV 512x2048 | 0.5 |
| blk.0 shared expert | gate/up/act 0.2, output 0.5 and relative L2 0.02 |
| blk.0 FFN (MoE + shared) | router 0.05, output 1.0 and relative L2 0.02 |
| blk.0 GDN block | relative L2 0.01 through the norms, 0.02 at the output |
| blk.0 GDN across tokens (T=1/4/8) | relative L2 0.02 per token and on the final state |
| rope | max abs 0.01, relative L2 0.01; the un-rotated tail must be bit-identical |
| blk.3 attention, one token | relative L2 0.02 |
| blk.3 attention with KV cache (T=4/8) | relative L2 0.02 per token |
| whole model, 2 layers | hidden state relative L2 0.02 |
| whole model, 40 layers | hidden state relative L2 0.02 (measured 0.0011) |
| whole model, 40 layers | top-5 next tokens match llama.cpp; logprobs within 0.2 |
| whole model over a 5-token prompt | same top-5 in the same order; logprobs within 0.07 |

## Two API facts worth writing down

- aclnn tensor descriptors are **row-major**: `shape = {rows, cols}`, `stride = {cols, 1}`.
  Handing it a ggml-style column-major descriptor (`{cols, rows}` / `{1, cols}`) fails with
  `161002` on the first matmul.
- `aclnnRmsNorm` takes an extra `rstdOut` tensor (the reciprocal std per row), and `gamma` must
  be 1-D. A 2-D `{1, C}` gamma fails with `561002`.

## What the model file is made of

`gguf_parity` prints the real mix, and it is not "Q4_K and Q6_K":

```
weight mix :
  f32    368 tensors      0.10 GB
  q8_0   260 tensors      2.61 GB
  q4_K    80 tensors     12.08 GB
  q5_K    40 tensors      7.38 GB
  q6_K     3 tensors      0.66 GB
  bf16     2 tensors      0.00 GB
  sum                    22.84 GB
```

All four formats now decode, and each is bit-exact against ggml on a real tensor from this file.
F32 and BF16 tensors are read directly (norms, the router, `ssm_a`).

## What a layer looks like in this model

`blk.0` is a hybrid layer, and the header says what the port has to support:

```
blk.0.attn_norm.weight        F32    [2048]
blk.0.attn_qkv.weight         Q8_0   [2048, 8192]     fused QKV
blk.0.attn_gate.weight        Q8_0   [2048, 4096]
blk.0.ssm_a / ssm_alpha / ssm_beta / ssm_conv1d / ssm_dt / ssm_norm    F32    gated delta net
blk.0.ffn_gate_inp.weight     F32    [2048, 256]      router
blk.0.ffn_gate_exps.weight    Q4_K   [2048, 512, 256] experts
blk.0.ffn_down_exps.weight    Q5_K   [512, 2048, 256]
blk.0.ffn_gate_shexp.weight   Q8_0   [2048, 512]      shared expert
blk.0.post_attention_norm.weight F32 [2048]
```

So 3 of 4 layers run a GDN block (`ssm_*`), and the FFN is 256 experts with one shared expert.

## The model's shape, as the header describes it

`qwen35moe` has 41 blocks, but the last one is not part of the main stack: `blk.40` carries
`nextn.eh_proj`, `nextn.enorm`, `nextn.hnorm` and `nextn.shared_head_norm`, which is the
multi-token-prediction draft head. The main model is 40 layers, and every 4th of them is a
full-attention layer (3, 7, ... 39); the other 30 are recurrent. `blk.40` has `attn_q/k/v`
instead of the fused `attn_qkv`, which is how the missing-tensor error showed it.

Its router tensors are BF16 (`ffn_gate_inp`, `ffn_gate_inp_shexp`), the only BF16 in the file.

## Checking against llama.cpp, not only against ourselves

Every test here compares the NPU against a CPU reference written from reading llama.cpp's source.
That proves the CANN port matches *my transcription*; it does not prove the transcription is right.
For that, `llama-eval-callback` (built from the same checkout) dumps llama.cpp's own intermediate
tensors for the same single token:

```
llama.cpp  attn_norm-0    = [-0.0858,  0.6080, -3.4649, ...,  0.0278, -0.2714,  0.1871]
cann/      attn_norm      = [-0.0858,  0.6079, -3.4648, ...,                       ...]

llama.cpp  linear_attn_out-0 = [-0.0061, -0.0046,  0.0089, ...]
cann/      attn_out          = [-0.0061, -0.0045,  0.0089, ...]

llama.cpp  attn_residual-0   = [-0.0067,  0.0003, -0.0187, ...]
cann/      residual          = [-0.0067,  0.0003, -0.0187, ...]

llama.cpp  ffn_moe_out-0     = [ 0.0239, -0.0100, -0.0117, ...]
cann/      ffn_out           = [ 0.0295, -0.0106, -0.0099, ...]     <- still differs
```

That comparison found a real bug: the short conv feeds its inputs oldest-first, so the **current
token pairs with the last tap**, not the first. My transcription used the first tap, and the CPU
reference used the same wrong convention, so the two agreed with each other and both were wrong.

Layer by layer against llama.cpp's `l_out-<n>`:

| layer | llama.cpp | here |
|---|---|---|
| 0 (gdn) | 0.0230, -0.0104, -0.0286 | 0.02273, -0.01032, -0.02859 |
| 1 (gdn) | 0.0293, -0.0038, -0.0339 | 0.02888, -0.00381, -0.03406 |
| 2 (gdn) | 0.0215, -0.0053, -0.0386 | 0.02096, -0.00531, -0.03854 |
| 3 (attn) | 0.0370, 0.0148, -0.0211 | 0.05063, 0.01886, -0.03329 |
| 4 (gdn) | 0.0127, 0.0168, -0.0321 | 0.02386, 0.01603, -0.03848 |

The three GDN layers match; the first attention layer did not. Splitting the gated attention
vector head by head against llama.cpp's full dump (the debug printer was patched locally to print
whole tensors) showed the pattern: heads 0-7 matched on even heads only, heads 8-15 on odd ones.
That is a blocked mapping, not an interleaved one:

```
q head h -> kv head h % 2     (what I had)      heads 0,2,4,6 ... share kv 0
q head h -> kv head h / 8     (llama.cpp)       heads 0-7 share kv 0, 8-15 share kv 1
```

After that fix, the layer-by-layer trace and the whole model agree:

| layer | llama.cpp | here |
|---|---|---|
| 0 (gdn) | 0.0230, -0.0104, -0.0286 | 0.02273, -0.01032, -0.02859 |
| 3 (attn) | 0.0370, 0.0148, -0.0211 | 0.03610, 0.01498, -0.01968 |
| 5 (gdn) | 0.0064, 0.0216, -0.0313 | 0.00608, 0.02287, -0.03059 |

and the same token's next-token distribution matches llama.cpp's top five exactly, with logprobs
within 0.2 nats:

```
llama.cpp  8(-2.607)  283(-2.615)  13(-3.162)  19845(-3.256)  4927(-3.842)
cann/     283(-2.507)  8(-2.515)  19845(-3.116)  13(-3.226)  4927(-3.648)
```

The top two trade places because they are 0.008 apart in both. The 40-layer forward also now
matches the CPU reference to 1.1e-3, down from 9.2e-3 before this fix: most of what looked like
fp16 drift was this one wrong index.

## A prompt, token by token

`model_seq` walks a whole prompt through the 40 layers with the conv history, the 128x128 per-head
state and the KV cache live on the device. For `The capital of France is` (5 tokens), the next-token
distribution is the same as llama.cpp's, in the same order:

```
                           llama.cpp      here
token 11751 ' Paris'       -0.5844      -0.5468
token   264 ' a'           -2.2250      -2.2186
token   279 ' the'         -3.1078      -3.0624
token  3750 ' known'       -3.4281      -3.4686
token   271 '\n\n'         -3.7641      -3.8280
```

Five tokens took 166 s, so about 33 s per token. That was the honest number for the first version:
every `gemv` dequantized its weights on the host and uploaded them again, so nothing amortized
across tokens.

## Making it faster

Two changes, both measured on the same prompt:

| change | 5 tokens |
|---|---|
| first version | 166 s |
| weights and the aclnn workspace stay resident | 64 s |
| threaded dequantization | 41 s |
| decoding a layer's 24 expert slabs concurrently | 12.7 s |

Profiling paid for itself here. A single GEMV measures 0.043 ms, and a whole token makes about
12000 aclnn calls, so the ops were never the problem: `print_op_stats` shows 2.5 s for 122559
calls across the five tokens, and `print_xfer_stats` shows 1.2 s for 11 GB of transfers. The rest
was **host-side dequantization** of expert slabs.

The first attempt threaded *inside* each slab: 128 threads over a megabyte was slower than the
decode. `print_slab_stats` showed it plainly — `3348 new, disk 0.5 s, dequant 22.3 s`. The fix
was to decode the 24 slabs a layer needs **concurrently, one thread each**:

```
slabs: 3348 new, disk 0.5 s, dequant 1.5 s, upload 3.9 s
```

Dequantization went from 22.3 s to 1.5 s and the five tokens from 37 s to 12.7 s, with identical
logits.

The weight cache keeps every non-expert tensor (2.4 GB) in HBM and caches expert slabs with a
budget. The workspace cache removed an `aclrtMalloc`/`aclrtFree` pair per op. Dropping the
per-op `stream.synchronize()` as well made the numbers wrong — host uploads go through
`aclrtMemcpy`, which is not ordered against queued kernels — so the sync stayed.

## What is not here yet

- The engine's own kernels: `src/` still compiles for CUDA only.
- `ggml-cann` still has no `GGML_OP_GATED_DELTA_NET`, so the llama.cpp server keeps 3 of 4 layers
  on the CPU. This directory has a working device path, but wiring it into that backend is a
  separate change.
- Fused dequant+matmul. The decoder works on the host today; a fast engine wants it fused on the
  device.
- GDN state carry between tokens: done, verified for T=1, 4 and 8 in `gdn_state_parity`.
- RoPE and the KV cache are done for the full-attention layers. What is left before a whole model
  runs: the 41-layer loop, the token embedding, the final norm and the lm head.
- Top-k on the device. The router logits come from the NPU, but the selection runs on the host.
- Weights resident on the device: done for everything except unselected expert slabs, which are
  cached with a budget. A token still costs 13 s, dominated by ~1200 aclnn calls with a sync each.
- Fusion: the eight selected experts share one input, so their gate and up projections could be one
  matmul over a stacked weight instead of sixteen. That is the next large win.
- Serving: this stack is not wired to `serve/`; the running server is llama.cpp with `ggml-cann`.

## Verified on the machine

`Ascend910B2C`, CANN 9.0.0, driver 26.0.rc1, 2026-11-05:

```
device: Ascend910B2C, HBM 61.0 GiB total / 58.0 GiB free

op                          max|diff|   mean|diff|  tolerance  result
matmul 256x512x384           0.002529     0.000266     0.5000  PASS
rms_norm 512x2048            0.000977     0.000152     0.0200  PASS
silu 4096                    0.001947     0.000319     0.0200  PASS
add 4096                     0.000977     0.000209     0.0100  PASS
softmax 64x1024              0.000004     0.000000     0.0100  PASS
muls 8x8 by 0.5              0.000000     0.000000     0.0100  PASS
matmul 8x8x1                 0.000205     0.000088     0.0100  PASS
mul 8x1 * 1x8                0.000237     0.000062     0.0100  PASS
```

```
Q8_0   vs ggml: max|diff| = 0  BIT-EXACT
Q4_K   vs ggml: max|diff| = 0  BIT-EXACT
Q5_K   vs ggml: max|diff| = 0  BIT-EXACT
Q6_K   vs ggml: max|diff| = 0  BIT-EXACT
```

```
case                 attn|diff|  state|diff|  result
kda    S=64  T=8       0.000360     0.003886  PASS
scalar S=64  T=8       0.000360     0.001168  PASS
kda    S=128 T=4       0.000175     0.001516  PASS
```

```
gguf : v3, 753 tensors, arch qwen35moe, 41 layers, 256 experts (top 8), embedding 2048

real GEMV  blk.0.ffn_gate_exps.weight [512 x 2048] on the NPU: max|diff| = 0.000116  PASS

blk.0.ffn_gate_exps.weight   Q4_K  vs ggml: BIT-EXACT
blk.0.ffn_down_exps.weight   Q5_K  vs ggml: BIT-EXACT
blk.34.ffn_down_exps.weight  Q6_K  vs ggml: BIT-EXACT
output.weight                Q8_0  vs ggml: BIT-EXACT
```

```
block      : blk.0 shared expert (SwiGLU)

stage                     max|diff|       rel L2  result
gate  = W_gate x           0.000240     0.000202  PASS
up    = W_up x             0.000215     0.000194  PASS
act   = silu(g) * u        0.000172     0.000428  PASS
y     = W_down act         0.000034     0.000404  PASS
```

```
block  : blk.0 GDN (recurrent) layer, one token, zero state
shapes : q/k 16x128 heads, v 32x128, conv k=4

stage                          rel L2    max|diff|  result
conv = silu(w0 * qkv)        0.000367     0.000045  PASS
q after l2 norm              0.000406     0.000511  PASS
k after l2 norm              0.000460     0.000550  PASS
delta net out                0.002496     0.000000  PASS
gated rms norm               0.003742     0.000040  PASS
y = ssm_out                  0.004751     0.000005  PASS
```

```
rope : head_dim 256, n_rot 64, freq_base 1e+07, 5 tokens (pos 0 1 2 7 129)

check           max|diff|       rel L2  result
rotated          0.001307     0.000237  PASS
tail copy        0.000000     0.000000  PASS
```

```
layer  : blk.3 attention, one token at position 5

stage                  rel L2  result
y = attn_output      0.000512  PASS
```

```
layer  : blk.3 attention across 4 tokens with a KV cache
cache  : 2 kv heads x 256 dim, GQA 8:1

token        y rel L2  result
0            0.000501  PASS
1            0.000598  PASS
2            0.000881  PASS
3            0.000706  PASS
```

```
block  : blk.0 GDN across 4 tokens, state carried on the device

token          y rel L2  result
0              0.001930  PASS
1              0.002137  PASS
2              0.000963  PASS
3              0.001056  PASS
state          0.001513  PASS
```

```
block   : blk.0 FFN = 8 of 256 experts + shared expert
routing : 213(0.1542) 71(0.1505) 230(0.1398) 130(0.1233) 201(0.1118) 205(0.1094) 135(0.1074) 25(0.1037)

stage                     max|diff|       rel L2  result
router logits              0.000457     0.000211  PASS
ffn out                    0.000013     0.000439  PASS
```

```
layer  0/40  gdn   top1 expert 101  1.3 s
layer 10/40  gdn   top1 expert 8  9.6 s
layer 20/40  gdn   top1 expert 64  17.7 s
layer 30/40  gdn   top1 expert 231  25.9 s
layer 39/40  attn  top1 expert 213  33.3 s

final hidden: 0.95312 2.02734 3.54102 -5.52344
top 5 logits:  token 21750 = 8.4375  token 30458 = 8.2188  token 31590 = 7.8789  token 17023 = 7.5938

CPU reference in 31.4 s, hidden state rel L2 = 0.016679  PASS
```

Two things to read carefully there: the 0.0167 is **drift, not noise** (2 layers give 0.0022, 40
layers give 0.0167), because the residual stream and the KV cache are fp16 end to end. And the
5th logit entry was not stable across two runs, which is still unexplained.

Reproduce with `bash cann/tests/run_parity.sh`, `run_quant.sh`, `run_gdn.sh`, `run_gguf.sh`,
`run_layer.sh`, `run_moe.sh`, `run_gdn_block.sh`, `run_gdn_state.sh`, `run_attn.sh`, `run_attn_cache.sh`.
