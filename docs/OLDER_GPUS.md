# Older GPUs (experimental)

This page is for people with an NVIDIA card older than the RTX 20 series, an older AMD card, or a Turing card they want to tune.
For a normal install use the app installer for your hardware or the [Terminal](INSTALL.md#the-terminal). Pascal and Volta use the
CUDA 12 engine package. The `--cuda`, `--gpus`, `START-HERE.bat` and `UPDATE.bat` examples below are
[source-setup](SOURCE_INSTALL.md) options, not Terminal launch flags.

Strata's supported cards are NVIDIA RTX 20 / 30 / 40 / 50 and the AMD cards in [AMD_HIP.md](AMD_HIP.md). The cards below run
through opt-in, experimental paths that community members wrote and measured. The maintainers own one of these cards, a Tesla
P100, and test on it. Every other path is compile-checked and unit-tested, and the ready-made engines and their output are
unchanged. Numbers are the reporters' own, on one machine each, unless the text says the maintainers measured them.

> **On this page:** [Support matrix](#support-matrix) · [CUDA 12 engine](#nvidia-the-cuda-12-engine) ·
> [Tesla P100](#tesla-p100-sm_60) · [GP100 Q8_0 kernel](#pascal-gp100-a-q8_0-decode-kernel-opt-in) ·
> [Tesla P40](#tesla-p40-sm_61-what-was-measured-and-changed) · [Turing switches](#quadro-rtx-8000-and-other-turing-cards) ·
> [AMD](#amd-building-gfx906-and-gfx1012) · [History](#history--older-measurements)

## Support matrix

### NVIDIA

| Cards | Compute capability | How it runs | What is different on it |
| --- | --- | --- | --- |
| Tesla P100 | 6.0 | the CUDA 12 engine | `__dp4a` emulated (bit-exact); BF16 projections through fp32; seven scheduling switches on by default ([below](#tesla-p100-sm_60)); opt-in `STRATA_Q8_SM60=1` ([below](#pascal-gp100-a-q8_0-decode-kernel-opt-in)) |
| Tesla P40 / P4, GTX 10 series | 6.1 | the CUDA 12 engine | BF16 projections through fp32 (cuBLAS has no BF16 GEMM there); dense weights dequantized by a wide kernel; expert mode 8 and its own interleaved-matvec table; opt-in `STRATA_DENSE_F32=1` ([below](#tesla-p40-sm_61-what-was-measured-and-changed)) |
| Tesla V100, Titan V | 7.0 | the CUDA 12 engine | BF16 projections on the FP16 tensor cores; prompt attention on `mma.m8n8k4`; a leaner attention kernel. See [NVIDIA_V100.md](NVIDIA_V100.md) |
| RTX 20 (Turing) | 7.5 | **supported**, the ready-made engine | opt-in `STRATA_BF16_TC=1` runs the BF16 projections on the FP16 tensor cores |

A card needs enough VRAM to be useful. 12 GB or more is recommended, as for every card. A 2 GB GT 1030 is compute capability 6.1
too and cannot hold any of the model.

### AMD

| Cards | Architecture | How it runs |
| --- | --- | --- |
| RX 6800 / 6900 series | gfx1030 | setup (`--backend hip`), unvalidated. The #540 attention kernel is the default there (`STRATA_ATTN_PRE75=0` runs the standard kernel). See [AMD_HIP.md](AMD_HIP.md#rdna2-gfx1030) |
| RX 6700 XT | gfx1031 | setup (`--backend hip`), unvalidated; used daily by its reporter on one card |
| RX 5500 XT (RDNA1) | gfx1012 | built by hand: `-DCMAKE_HIP_ARCHITECTURES=gfx1012` (HIP 5.7 or 7) |
| RX 5700 XT (RDNA1) | gfx1010 | built by hand: `-DCMAKE_HIP_ARCHITECTURES=gfx1010` on ROCm 7.14's `gfx101X-dgpu` wheels; needs `ROCR_VISIBLE_DEVICES=0` on a PC that also has an AMD iGPU |
| RX 5700, 6 GB RX 5600 (RDNA1) | gfx1010 | as above; a 6 GB card leaves about 50 expert slots |
| Radeon PRO V520 / Pro 5600M (RDNA1) | gfx1011 | built by hand: `-DCMAKE_HIP_ARCHITECTURES=gfx1011`; builds with 0 errors, untested on hardware |
| Instinct MI50 / MI60, Radeon VII | gfx906 (wave64) | built by hand: `-DSTRATA_HIP_GFX906=ON` |

## NVIDIA: the CUDA 12 engine

CUDA 13 dropped Pascal and Volta and cannot compile for them. Setup therefore keeps a second engine, built with CUDA 12.9 and
`-DSTRATA_EXPERIMENTAL_SM60=ON`, in its own folder (`engine-cuda12\`, beside `engine\`). One engine runs per model, so the
choice is made per model, by the oldest card that model runs on:

- Every card the model uses is RTX 20 or newer: the ready-made CUDA 13 engine.
- A card is Pascal or Volta: the CUDA 12 engine. Setup says so (`CUDA 12: sm_70 is older than CUDA 13 supports ...`).

Release builds deliver engine components through the signed feed (`strata update`). For a source checkout, `--build` compiles with
a CUDA 12.x toolkit ([source installation](SOURCE_INSTALL.md)).

### Opting in

An older card is used only when you choose it. A PC with a newer card keeps recommending the newer one.

| You | Setup |
| --- | --- |
| have only Pascal / Volta NVIDIA cards (and no AMD card it can use) | uses them, with the CUDA 12 engine |
| name the card: `START-HERE.bat --setup --gpu 1`, or `--gpus 0,1` with a newer card | uses it; the model gets the CUDA 12 engine |
| `--cuda 12` (or `STRATA_CUDA=12`) | the CUDA 12 engine for this model, on any card |
| `--cuda 13` | the CUDA 13 engine even with an older card (a warning: it has no code for that card) |
| `STRATA_EXPERIMENTAL_SM60=1` | the older cards are listed as usable |

The choice is kept in the model's config (`"cuda": 12`). Starts and `UPDATE.bat` keep it, and other models keep their own engine.
Setting the model up again chooses again (add `--cuda 12` to keep a forced choice). A Pascal or Volta card added to a model at a
start (`--gpus`) moves that model to the CUDA 12 engine.

`--cuda 12` is also the way to run Strata with an NVIDIA driver older than 580. The CUDA 12 engine needs driver 528 or newer on
Windows and 525 on Linux (`CUDA12_MIN_DRIVER` in `setup.py`); setup's "driver too old" stop says so. The Docker `cuda12` image
asks for 575 ([DOCKER.md](DOCKER.md)). Drivers that old were not tested here.

### Mixed cards

A model that shares a V100 with an RTX 30 / 40 card runs both on the CUDA 12 engine (code for sm_60 to sm_89, plus PTX). An RTX
50 card (sm_120) in a CUDA 12 engine is a warning, not a stop: engines built with 12.8 crashed on long prompts there (#220,
#224). Keep the RTX 50 card on its own model (`--gpu N`), where it runs the CUDA 13 engine.

### Building it yourself

```sh
# Linux (Windows: the same with the toolkit's nvcc.exe)
cmake -S . -B build-cuda12 -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_EXPERIMENTAL_SM60=ON \
      -DCMAKE_CUDA_ARCHITECTURES="61;70" -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc
cmake --build build-cuda12 --target strata -j
```

Setup does the same when it compiles: it looks for the newest CUDA 12.x toolkit. `STRATA_NVCC=<path to nvcc>` picks one. On
glibc 2.43 use 12.8 ([TROUBLESHOOTING.md](TROUBLESHOOTING.md)). On GCC 12.3 with nvcc (openEuler 24.03, CUDA 12.8) one report
needed `-D_BITS_OPT_RANDOM_H` added to the host flags (#1074, not reproduced here).

### What the flag changes

`-DSTRATA_EXPERIMENTAL_SM60=ON` lowers the runtime floor to compute capability 6.0 and compiles the older cards' code paths into
that build only: the Volta prompt attention, the #540 attention kernel (used below sm_75), and the Pascal BF16 path. The
ready-made CUDA 13 engine has none of them, so its kernels and output are unchanged.

The FP16 path for BF16 projections is in every build but runs by default only below sm_75. RTX 20 owners can try it with
`STRATA_BF16_TC=1` (`=0` turns it off on a V100). It is not bitwise the same as cuBLAS's BF16 kernel, because FP16 tensor-core
sums round differently.

### Switches by card

| Variable | Card | Default | What it does |
| --- | --- | --- | --- |
| `STRATA_SM60_PERF=0` | P100 | set of 7 on | Turns off the seven scheduling switches below. |
| `STRATA_Q8_SM60=1` | GP100 | off | Q8_0 dense projections and head on a GP100 kernel. |
| `STRATA_IQ_STAGE_TABLES=1` | Pascal | off | IQ3_XXS / IQ3_S tables staged in shared memory (bitwise the same output). |
| `STRATA_DEQUANT_WIDE=0` | sm_61 | on | Back to the old dequantization. |
| `STRATA_SM61_TABLE=0` | sm_61 | on | Back to the CUDA expert layout and `native_mmvq`. |
| `STRATA_DENSE_F32=1` | Pascal | off | Large dense GEMMs as `cublasSgemm`. |
| `STRATA_BF16_TC=0\|1` | sm_7x | on below sm_75 | BF16 projections on the FP16 tensor cores. |
| `STRATA_PROMPT_ATTN_OLD=1` | Volta | off | The decode kernel for prompts instead of the Volta kernel. |
| `STRATA_ATTN_PRE75=0` | sm_70, gfx103x | on | Turns the #540 attention kernel off. `=1` turns it on for another wave32 AMD card. |

[NVIDIA_V100.md](NVIDIA_V100.md) has the V100 build, its switches and the parity test.

## Tesla P100 (sm_60)

On a Tesla P100 the engine turns on seven switches that change only how the work is scheduled, so answers stay bit for bit the
same: `STRATA_ADAPT_NOWAIT=1`, `STRATA_ADAPT_LAG=2`, `STRATA_SIDE_MIXER=2`, `STRATA_FUSE_WAIT=2`, `STRATA_B6_MMVQ_ROWS=1`,
`STRATA_FUSE_DECODE=1` and `STRATA_GR_UP75=1`. `STRATA_SM60_PERF=0` turns the whole set off, and a switch you set yourself, to
any value, is kept. The set is used when every visible card has compute capability 6.0. The full list and per-switch results are
in [DETAILS.md](DETAILS.md#rtx-30-series-sm_86-and-tesla-p100-sm_60-bitwise-decode-switches-on-by-default).

## Pascal GP100: a Q8_0 decode kernel (opt-in)

GP100 (P100, PH402) is the one Pascal chip without `__dp4a`. Its dense projections read their weights at 40-80 GB/s of about
730, decoding K- and i-quant blocks one int at a time. Two opt-in changes, both measured on a PH402 only:

- **`STRATA_Q8_SM60=1`** packs every Q8_0 dense matrix and the head into the planes of `STRATA_Q8_PACKED` and runs them on
  `src/kernels/cuda/q8_sm60.cuh`: a warp per row, 16-byte weight loads, activations copied once per workgroup into shared memory,
  the dot on `STRATA_DP4A`, no barrier per row. Shapes it declines keep the usual kernel. It is not bitwise the exact kernels.
  The packed copies cost their size in VRAM (Flash-Next: 2.9 GiB of dense projections over the stages and 0.6 GiB for the head).
- The GSQ-RCO packs have no Q8_0 dense matrices, so the kernel needs a copy whose dense projections and head are Q8_0:
  `python tools/q8_dense_gguf.py <model>-00001-of-00002.gguf <new folder>` writes one (+3.5 GiB; experts keep their offsets,
  so the native pack is reused). Point `--native` and `--ple-gguf` at the new folder.
- **`STRATA_IQ_STAGE_TABLES=1`** stages IQ3_XXS and IQ3_S codebooks in shared memory in the grouped expert kernels. Bitwise the
  same output. Interleaved A/Bs on a Tesla P100 showed no end-to-end change, so it is off by default. On an RTX 3090 Ti, forced on,
  the same layers took 3-13% longer.

Neither change is measured on compute capability 6.1, which has `__dp4a`. Two practical notes for GP100 cards: the default
application clock is 759 MHz and a layer split leaves each die idle most of a window, so lock the clocks
(`nvidia-smi -i <dies> -ac 715,1050` as administrator). And a second engine pinned to the same logical CPUs halves this one's
decode; keep two engines on different logical CPUs.

`tools/q8_sm60_check.cu` checks the kernel against a CPU reference on these shapes (1-8 columns).

## Tesla P40 (sm_61): what was measured and changed

Three things run differently on compute capability 6.1 by default and a fourth is opt-in. Each gives the same bits as what it
replaces.

| Change | What it replaces | Switch |
| --- | --- | --- |
| Wide dequantization (a thread per 8 values, 16-bit loads, 16-byte stores) for Q4_K / Q5_K / Q6_K / IQ4_XS | a thread per 32 values with byte loads | on; `STRATA_DEQUANT_WIDE=0` |
| Expert mode 8 for the experts in VRAM (decode and verify windows) | the CUDA layout | on; `STRATA_SM61_TABLE=0` (or `STRATA_EXP_MODE=0`) |
| The interleaved small-batch matvec with a P40 table (`kIlRows_61`) | `native_mmvq` | on; `STRATA_SM61_TABLE=0` (or `STRATA_MMVQ_IL=0`) |
| The large dense GEMMs as `cublasSgemm` on FP32 copies of the FP16 operands | cuBLAS's FP16-operand kernels (4.3 TFLOPS instead of 7.6 for most chunk sizes) | **opt-in**, `STRATA_DENSE_F32=1` |

The GEMM route is opt-in because its equality is a property of the cuBLAS build, not of the code. It was measured with cuBLAS
12.4.5.8 only. Run `gemm_f32_parity` (a build with `-DSTRATA_BUILD_TESTS=ON`; exit 0 = every size bitwise) with your toolkit's
library before you set it.

`STRATA_GR_FAST=1` is slower on the P40 below 5 tokens, and `STRATA_PREFILL_CPU_SHARE=auto` and `STRATA_ADAPT_OVERLAP=1` gave
nothing there.

Two things to know when measuring on such a card:

- **Temperature.** A P40 has no fan of its own. At 89 C the driver reports `SW Thermal Slowdown: Active` and the clock falls from
  1,531 to about 1,300 MHz (`nvidia-smi -q -d PERFORMANCE`). Start every arm of an A/B at the same temperature, or the later arm loses.
- **Greedy answers are not identical run to run on a CPU without AVX-512**, even with `--prompt-cache 0 --adapt-swaps 0
  --pcie-frac 0`. Add `STRATA_IQ_MT_MIN=1` (and `STRATA_PREFILL_CPU_SHARE=0`, `STRATA_ROUTE_TAIL_SKIP=0`) to compare two builds'
  answers byte for byte.

Profiling switches: `STRATA_PREFILL_TIMING=1`, `STRATA_DECODE_TIMING=1`, `STRATA_VERIFY_PROFILE=1`.

## Quadro RTX 8000 and other Turing cards

Turing (sm_75) has FP16 tensor cores but no native BF16 or TF32 tensor cores. Two existing, opt-in switches are worth comparing
on your own workload:

- `STRATA_BF16_TC=1` converts BF16 projection operands to FP16 and uses FP16 tensor cores with FP32 accumulation. Default on sm_75
  is off; `=0` restores the cuBLAS BF16 path. Values outside FP16 range clamp to +-65504, and differently ordered sums can round
  differently.
- `STRATA_SELECT_SIMT=1` uses the tiled FP32 QSA block scorer on the pre-sm_80 CUDA path instead of the warp scorer. `=0` keeps
  the warp scorer. It does not enable TF32 and its summation order differs.

Set the variables before starting the engine, or in the model config's `env` object, then restart. Merge these fields into the
existing config; this is not a complete run config:

```json
{
  "env": {
    "STRATA_BF16_TC": "1",
    "STRATA_SELECT_SIMT": "1"
  }
}
```

`STRATA_BF16_TC=1` improved prompt throughput by 20-34% on one Quadro RTX 8000 and changed replies in 3 of 9 paired requests, so
use it only if different rounding is acceptable. It did not improve decode. Numbers are in the history section. These switches are
existing code ([gemm.cu](../src/prefill/gemm.cu), [qsa_select.cu](../src/kernels/cuda/qsa_select.cu)), not new kernels.

## AMD: building gfx906 and gfx1012

Setup does not build these. Build by hand and run `serve/server.py` with a config, as on any other card.

- **gfx906** (MI50 / MI60 / Radeon VII, wave64): a separate opt-in build, `-DSTRATA_HIP_GFX906=ON
  -DCMAKE_HIP_ARCHITECTURES=gfx906` (not `STRATA_ENABLE_HIP`). Current ROCm no longer ships gfx906 libraries; the reporter used a
  community ROCm 7.14 image. Recipe, kernels and measurements:
  [AMD_HIP.md](AMD_HIP.md#gfx906-instinct-mi50--mi60-radeon-vii-wave64-built-from-source).
- **gfx1012** (RX 5500 XT): the wave32 backend, `-DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1012`. HIP 5.7 (Ubuntu's
  packages) works: older hipBLAS is used through rocBLAS, legacy HIP names and the missing `__syncwarp` are version-gated, and
  RDNA1's missing signed dot4 uses llama.cpp's SDWA sequence (`-DSTRATA_GFX1012_PORTABLE_DOT=ON`: the portable one).

## Reports welcome

These paths stay experimental until more people run them. A report with the card, the driver or ROCm version, the model and the
engine log (`strata-<model>.log`) in an issue helps. [COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md) has the format.

## History / older measurements

Dated numbers. Each row is one machine and one engine version, as labelled. They are not a promise for your card.

**Community reports by card**

| Card | Measured on | Result |
| --- | --- | --- |
| 2x Tesla P100 | IQ3_S, engine 0.1.39 ([report](../bench/results/2026-10-06-community-2x-p100/README.md), #1157) | prompt 526-532 tok/s; decode 28-32 tok/s at 128K prompt tokens; 3 of 3 needle checks at 122K |
| PH402 (4 GP100 dies) | IQ3_XXS, engine 0.1.40.2 | decode 28.3 -> 38.9 tok/s with `STRATA_Q8_SM60=1` and a Q8_0 dense shard |
| Tesla P40 | IQ3_S, engine 0.1.30 (#395) | prompt 217-374 tok/s; decode 30-33 tok/s |
| Tesla P40 | IQ3_S, engine 0.1.43.1 with the sm_61 changes | prompt 255 tok/s at 565 tokens (was 205); decode 37-38 tok/s (was 34-35); with `STRATA_DENSE_F32=1` 464-474 tok/s at 7K tokens (was 360-432) |
| V100-PCIE-32GB | UD-IQ4_XS (#600) | prompt 1,123-1,251 tok/s |
| V100 32GB | IQ2_XS (#540) | prompt +22% |
| RTX 2080 Ti | Q2_0 (#655) | prompt +15-18% with `STRATA_BF16_TC=1` |
| RX 6900 XT | IQ3_S ([report](../bench/results/2026-10-04-rdna2-pre75-attention/README.md)) | prompts +4-6% alone, +7-12% with #835 |
| RX 5500 XT, 8 GB | IQ3_S, 8K prompt (#442) | decode 15.3 tok/s |
| RX 5700 XT, 8 GB | Coder IQ1_M, 32K context | prompt 115 / 144 / 149 tok/s and decode 18.0 / 20.3 / 23.3 tok/s at 4K / 16K / 30K; 6 of 6 needle checks; 607 of 12,288 experts in VRAM |
| 2x MI50 | Coder IQ1_M, 128K context (#677) | decode 50.1 / 47.8 / 45.7 tok/s at 4K / 32K / 128K; prompt about 520 tok/s |

**Tesla P100 default switch set.** Measured on: Tesla P100 16 GB, IQ3_XXS, default flags, engine 0.1.45.3, 8 interleaved pairs of
fresh engines: decode 36.56 -> 41.76 tok/s (+14.2%, faster in 8 of 8 pairs), same number of expert-cache slots. The identity gate
(10 prompts and 4K / 20K prompt dumps) is identical with the set on.

**GP100 Q8_0 kernel.** Measured on: PH402 SKU 200 (two boards, four GP100 dies, 48 SMs and 32 GB HBM2 each), application clocks
locked at 1050 MHz, Windows 11, driver 581.80 (TCC), CUDA 12.9, engine 0.1.40.2, Flash-Next IQ3_XXS, 32K, `--layer-split
12,24,36 --prefill 2048`, greedy, 256 tokens, medians of 2-3 interleaved starts, decode tok/s.

| | ~10.7K-token document | the same, cached | short reasoning | code | mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| GSQ-RCO shard, exact kernels | 28.8 | 29.4 | 28.4 | 26.8 | 28.3 |
| Q8_0 dense shard, `STRATA_Q8_PACKED=1` | 27.5 | 31.0 | 34.4 | 31.8 | 31.1 |
| Q8_0 dense shard, `STRATA_Q8_SM60=1` | 37.5 | 38.4 | 41.2 | 38.5 | 38.9 |

GPU time per verify window: dense projections 19.2 -> 11.2 ms, head 4.25 -> 2.1 ms. Prompt reads are unchanged (437-438 tok/s at
10.7K). Locking clocks at 1050 MHz took one case from 25.0 to 29.1 tok/s at 10.7K context.

**Tesla P40 sm_61 kernels.** Measured on: one Tesla P40 24 GB in a 20-vCPU VM (Xeon E5-2695 v4, 98 GB RAM), CUDA 12.4 (cuBLAS
12.4.5.8), IQ3_S with MTP (`--spec 4`), int8 KV, `--max-context 262144`, engine 0.1.43.1 ([report](../bench/results/2026-10-12-p40-sm61-kernels/README.md)).

| Change | Result on the P40 |
| --- | --- |
| Wide dequantization | Q6_K 10240 x 2560: 7.2 -> 0.33 ms (10 -> 227 GB/s); Q4_K 5.2x, Q5_K 3.8x, IQ4_XS 3.6x |
| Expert mode 8 | an 18-expert call at 2 tokens: IQ3_S 390 -> 255 us, IQ3_XXS 345 -> 252, IQ2_S 290 -> 238, IQ4_XS 328 -> 261 |
| Interleaved matvec table | IQ4_XS 2560 x 10240 at 2 columns 112 -> 65 us; Q6_K 12288 x 2560 at 4 columns 189 -> 127 us |
| `STRATA_DENSE_F32=1` | 8.4-9.8 TFLOPS at every size from 512 tokens; qkv at 7,041 tokens 85 -> 45 ms |

Through the server, medians of 3 requests, two runs per build alternated, card at 65 C. By default: a 565-token prompt 203 / 205
-> 256 / 255 tok/s, decode 35.0 / 34.2 -> 38.2 / 37.2 tok/s. With `STRATA_DENSE_F32=1`: a 7,047-token prompt 430 / 364 -> 474 / 473
tok/s. Of a 7K-token prompt, 36% is the dense projections' GEMMs, 28% the experts, 13% the hyper-connection read and 12%
attention.

**Quadro RTX 8000 (sm_75).** Measured on: one 48 GiB Quadro RTX 8000, 260 W limit, Xeon W-2295, CUDA 12.4, GCC 13, 2026-10-07,
source build based on v0.1.40.3 with other local changes; IQ3_S with MTP (`--spec 4`), `--max-context 262144`, INT8 KV with
`--kv-resident 32768`, `--pcie-frac 0.20`, `--expert-cache 19000`, `--prefill auto`, prompt cache and adaptive swaps off. Three
seeded synthetic prompts per size, 128 output tokens, median engine-reported rates.

| Switches | 537 tokens: prompt / decode tok/s | 4,057 tokens | 32,057 tokens |
| --- | --- | --- | --- |
| BF16_TC=0, SELECT_SIMT=0 | 493 / 86.8 | 1,055 / 79.9 | 1,135 / 70.6 |
| BF16_TC=1, SELECT_SIMT=0 | 661 / 83.7 | 1,262 / 75.4 | 1,416 / 68.4 |
| BF16_TC=1, SELECT_SIMT=1 | 679 / 87.1 | 1,290 / 78.2 | 1,501 / 71.6 |

`SELECT_SIMT` added about 6% to the 32K prompt rate over BF16_TC alone. `--prefill auto:16384` gained about 0.6% at 32K for 3 GiB
more prompt-cache loan, and `STRATA_PREFILL_CPU_SHARE=auto` gave no gain, so neither is recommended. All five basic arithmetic and
JSON checks passed in each arm; that is a smoke check, not a quality study.
