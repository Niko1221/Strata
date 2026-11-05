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
| P4 | MoE: top-8 routing, expert matmuls, shared expert | done for one layer's FFN |
| P5 | `serve/` wiring, 18789, benchmarks | not started |

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

## What is not here yet

- The engine's own kernels: `src/` still compiles for CUDA only.
- `ggml-cann` still has no `GGML_OP_GATED_DELTA_NET`, so the llama.cpp server keeps 3 of 4 layers
  on the CPU. This directory has a working device path, but wiring it into that backend is a
  separate change.
- Fused dequant+matmul. The decoder works on the host today; a fast engine wants it fused on the
  device.
- GDN state carry between tokens: done, verified for T=1, 4 and 8 in `gdn_state_parity`.
- RoPE and KV cache, which belong to the full-attention layers (3 of 4 layers are recurrent).
- Top-k on the device. The router logits come from the NPU, but the selection runs on the host.
- The attention half of a layer, which is where RoPE and the KV cache live.

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

Reproduce with `bash cann/tests/run_parity.sh`, `run_quant.sh`, `run_gdn.sh`, `run_gguf.sh`,
`run_layer.sh`, `run_moe.sh`, `run_gdn_block.sh`.
