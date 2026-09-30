# NVFP4 routed experts

The engine can serve a Qwen3.8-Flash-Next whose routed experts are NVFP4, as NVIDIA's ModelOpt stores them: 64-value
blocks of E2M1 codes with four UE4M3 sub-block scales, plus one FP32 `weight_scale_2` per expert and projection.
Everything else (attention, GDN, shared experts, routers, the head) stays in the formats the native path already
reads. Measured on Windows 11, RTX 5090 (32 GB), Ryzen 9 9950X3D, 128 GB DDR5-5600.

## Getting a model

```
python tools/nvfp4_convert.py --model <ModelOpt NVFP4 checkpoint dir> --outfile <model>.gguf
python tools/iq_pack.py --gguf <model>.gguf --out <pack dir>
```

- `nvfp4_convert.py` runs llama.cpp's own converter (`third_party/llama.cpp`, the commit setup pins) with the type
  policy the engine needs. The experts are repacked without loss: `tools/nvfp4_verify.py` compares them with the
  checkpoint bit for bit, and `tools/nvfp4_verify_gguf.py` checks the GGUF.
  - The small projections the BF16 kernels read (routers, SSM gates, indexer, PLE and hyper-connection projections)
    are written as BF16.
  - Everything served natively goes to Q8_0: attention, `ssm_out`, the shared experts, the output head, and the
    token embedding.
  - The 51.2e9-value PLE table is left out.
- `iq_pack.py` writes `experts.bin` for an NVFP4 GGUF by itself. Each blob is `[gate | up | down]` plus a 16-byte
  tail `{s_gate, s_up, s_down, 0}`. The GGUF has no room for these scales, so the engine reads NVFP4 experts from
  `experts.bin` only.

Run it like an IQ pack, with the GGUF as its own dense-weight source and the PLE table from an IQ GGUF's PLE shard:

```
strata --pack <pack dir> --native <model>.gguf --native-dense-gguf <model>.gguf \
       --ple-gguf <an IQ model's shard with per_layer_token_embd> --mtp <mtp/rt> \
       --expert-profile data/expert-profile.bin --expert-cache auto --prefill auto --spec 4 ...
```

The NVFP4 checkpoints change only the routed experts. The PLE table, the MTP layer and the expert profile of the
original model therefore work as they are.

Tested end to end with `jpezzulli/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4`. By its
`hf_quant_config.json`, NVIDIA's `nvidia/Qwen3.8-Flash-Next-NVFP4` quantizes the routed experts the same way
(NVFP4, group 16; the other layers BF16, the PLE table FP8, the MTP experts FP8), so it takes the same path. It has
not been run here.

## Measured

RTX 5090, 64K context, int8 K/V, `--vram-reserve-mib 1500`:

| | NVFP4 pack | IQ2_XS, same engine |
| --- | ---: | ---: |
| experts in RAM | 63.3 GiB | 33.0 GiB |
| expert cache slots | ~8,350 | ~17,300 |
| decode, 400-token answer | 114-116 tokens/s | 132-139 tokens/s |
| a 32K prompt | 3,900 tokens/s | 5,600 tokens/s |

An NVFP4 expert blob is 2.76 MB, against 1.51 MB for IQ2_XS: the cache holds half as many, and a miss costs twice
the bytes over PCIe. The PCIe share of the missed experts is therefore 0.25 for an NVFP4 pack, not the usual 0.55.
At 0.55 the copy kernel was 40% of all GPU kernel time (nsys, 300-token decode). A sweep at 262K context gave the
following decode rates:

| PCIe share | short prompt | 32K prompt |
| --- | ---: | ---: |
| 0.55 | 103 tokens/s | 105 |
| 0.40 | 110 | 117 |
| 0.25 | 117 | 124 |
| 0.10 | 112 | 118 |

## Where the global scales go

Each scale multiplies its own projection's FP32 output, as llama.cpp applies `.scale`:
`h = silu(s_gate * (G x)) * (s_up * (U x))`, `y = s_down * (D h)`.

The scales are ~1e-4 each. Folding `s_down` into up is algebraically the same, but it leaves the hidden near 1e-5.
The hidden is then rounded to Q8_0 (CPU) or q8_1 (GPU) blocks with an FP16 scale, and `amax / 127` lands deep in
FP16's subnormals: one expert's output was off by 2-12% instead of 1.1% (`nvfp4_avx512_parity --expert`). The FP16
prompt path had the same problem in its dequantized up weights (~1e-8). Any future fold must be checked against
FP16's range.

## Prompt path precision

`STRATA_PREFILL_NVFP4` picks how the batched prompt path multiplies the experts:

| mode | kernel | activations | one product vs FP64 (`mmq_nvfp4_parity`) |
|---|---|---|---|
| `w4a8` (default) | llama.cpp MMQ, int8 tensor cores | q8_1, float block scales | 0.53% |
| `w4a4` | llama.cpp MMQ, Blackwell FP4 MMA (sm_120a only) | NVFP4 | 8.6% |
| `fp16` | dequantize + cuBLAS FP16 GEMM | FP16 | reference |

`src/prefill/mmq_nvfp4_w4a8.cu` compiles mmq.cuh's int8 NVFP4 path with Blackwell's FP4 MMA hidden from both the
device code and the host's tile config, so it is the same on every architecture; decode already multiplies at this
precision. The FP4 MMA exists only in the arch-specific target, so `src/prefill/mmq_nvfp4_w4a4.cu` (the FP4 x FP4
kernel and its FP4 activation quantizer) is the one unit CMake builds with `12x` -> `12xa` (`strata_mmq_w4a4`);
the rest of the engine keeps the architectures as given and runs on sm_121 too. A card without that unit's image
(another architecture, or sm_121 against a 120a build) falls back to `w4a8` with a note.

First-token KL divergence against `fp16`, 8 prompts (code and prose, 1K-8K tokens). The noise floor is the same
`fp16` path cut into 4096-token chunks instead of 8192: another summation order, same arithmetic.

| mode | KL mean | KL median | KL max | top-1 agreement | prefill, 4-8K prompts |
|---|---|---|---|---|---|
| noise floor | 0.00023 | 0.00003 | 0.0016 | 8/8 | - |
| `w4a8` | 0.0018 | 0.0010 | 0.0096 | 8/8 | 2,503 tok/s |
| `w4a4` | 0.0080 | 0.0027 | 0.040 | 7/8 | 2,904 tok/s |
| `fp16` | reference | | | | 1,905 tok/s |

`w4a8` is the default. It is 4.5x closer to the reference than `w4a4`, at 86% of its speed, and it is the precision
every generated token already runs at. `fp16` is the choice when the prompt must be read exactly.

`STRATA_DUMP_MOE_INPUT` + `mmq_nvfp4_parity --real` test one product on a real prompt's rows: layer 20, its 12
busiest experts, max|x|/rms 4 median and 9 at most. The errors:

| mode | gate/up | down (heavier SwiGLU tails) |
| --- | ---: | ---: |
| `w4a8` | 0.46% | 0.90% |
| `w4a4` | 7.2% | 8.4% |
| `fp16` | 0.017% | 0.021% |

## CPU experts

`src/kernels/cpu/nvfp4_avx512.cpp` computes the CPU pool's NVFP4 rows in 512-bit lanes. Each 64-value block is
decoded once for every token of a verify window: `vpdpbusd` on |E2M1| codes, with the sign moved onto the
activation. The arithmetic is ggml-cpu's `ggml_vec_dot_nvfp4_q8_0`, with the float additions ordered differently
(worst 4.5e-7 relative, `nvfp4_avx512_parity`). `STRATA_NO_NVFP4_512=1` falls back to ggml-cpu, which is also the
path of CPUs without AVX-512.

| tokens | ggml-cpu | AVX-512 | speedup |
|---|---|---|---|
| 1 | 0.102 ms | 0.058 ms | 1.76x |
| 4 | 0.410 ms | 0.125 ms | 3.29x |
| 7 | 0.719 ms | 0.196 ms | 3.66x |

These are one expert's gate+up (1280 rows of 2560, cache-resident). The pool itself is bound by DRAM.

## Other GPUs

Only the optional `w4a4` needs Blackwell. The default `w4a8` is ggml's int8 MMQ (sm_75 and up), and decode uses the
engine's own q8_1 kernels, which decode UE4M3/E2M1 in software. The engine was built for sm_75/86/89 as PTX and run
on the RTX 5090 with the host answering as that generation. On those builds `mmq_nvfp4_parity` and
`nvfp4_expert_gpu_parity` gave results identical to sm_120.

HIP: the decode kernels use no CUDA-only intrinsics. The NVFP4 MMQ instance and the W4A8 unit are CUDA-only and
guarded (`GGML_USE_HIP`). Neither was built for HIP here.

## Tests

- `nvfp4_expert_gpu_parity`: the decode kernels against ggml's reference.
- `nvfp4_avx512_parity`: the CPU rows against ggml-cpu (with `--expert`: one whole expert).
- `mmq_nvfp4_parity`: the prompt path's products against FP64 (with `--real`: rows dumped by `STRATA_DUMP_MOE_INPUT`).
- `tools/nvfp4_verify.py` and `tools/nvfp4_verify_gguf.py`: the conversion, bit for bit.
