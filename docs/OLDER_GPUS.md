# Older GPUs (experimental)

Strata's supported cards are NVIDIA RTX 20 / 30 / 40 / 50 and the AMD cards in [AMD_HIP.md](AMD_HIP.md). The cards
below run through **opt-in, experimental** paths that community members wrote and measured on their own machines. The
maintainers have none of these cards: each path is compile-checked and unit-tested here, and the ready-made engines
and their output stay exactly as they were. Numbers are the reporters' own, on one machine each.

## Support matrix

### NVIDIA

| Cards | Compute capability | How it runs | What is different on it | Reported |
| --- | --- | --- | --- | --- |
| Tesla P100 | 6.0 | the CUDA 12 engine | `__dp4a` emulated (bit-exact); BF16 projections through fp32 | not measured |
| Tesla P40 / P4, GTX 10 series | 6.1 | the CUDA 12 engine | BF16 projections through fp32 (cuBLAS has no BF16 GEMM there, #395) | P40, IQ3_S, engine 0.1.30: prompt 217-374 tok/s, decode 30-33 tok/s (#395) |
| Tesla V100, Titan V | 7.0 | the CUDA 12 engine | BF16 projections on the FP16 tensor cores (#655, #540); the prompt attention on `mma.m8n8k4` (#600); a leaner attention kernel (#540) | V100-PCIE-32GB, UD-IQ4_XS: prompt 1,123-1,251 tok/s (#600); V100 32GB, IQ2_XS: prompt +22% from #540 |
| RTX 20 (Turing) | 7.5 | **supported**, the ready-made engine | opt-in: `STRATA_BF16_TC=1` runs the BF16 projections on the FP16 tensor cores | RTX 2080 Ti, Q2_0: prompt +15-18% (#655) |

A card needs enough VRAM to be useful: 12 GB or more is recommended, as for every card (a 2 GB GT 1030 is compute
capability 6.1 too, and cannot hold any of the model).

### AMD

| Cards | Architecture | How it runs | Reported |
| --- | --- | --- | --- |
| RX 6800 / 6900 series | gfx1030 | setup (`--backend hip`), unvalidated; #540's attention kernel is the default there, with 8 cells per step and DPP lane exchanges (bit-exact; `STRATA_ATTN_PRE75=0` runs the standard kernel) | [AMD_HIP.md](AMD_HIP.md#rdna2-gfx1030); RX 6900 XT, IQ3_S: prompts +4-6% alone, +7-12% with #835 ([bench/results/2026-10-04-rdna2-pre75-attention](../bench/results/2026-10-04-rdna2-pre75-attention/README.md)) |
| RX 6700 XT | gfx1031 | setup (`--backend hip`), unvalidated (#524) | used daily by its reporter, one card |
| RX 5500 XT (RDNA1) | gfx1012 | built by hand: `-DCMAKE_HIP_ARCHITECTURES=gfx1012` (HIP 5.7 or 7) | 8 GB card, IQ3_S, 8K prompt: 15.3 tok/s decode (#442) |
| RX 5700 XT (RDNA1) | gfx1010 | built by hand: `-DCMAKE_HIP_ARCHITECTURES=gfx1010`, on ROCm 7.14's `gfx101X-dgpu` wheels; needs `ROCR_VISIBLE_DEVICES=0` in a PC that also has an AMD iGPU | 8 GB card, Coder IQ1_M, 32K context: prompt 115 / 144 / 149 tok/s and decode 18.0 / 20.3 / 23.3 tok/s at 4K / 16K / 30K prompt tokens, 6 of 6 needle checks at 8K and 30K, 61 of 61 ctest on the card, 607 of 12,288 experts in VRAM |
| RX 5700 and the 6 GB RX 5600 (RDNA1) | gfx1010 | as above | the same Navi 10 silicon as the RX 5700 XT, so the same build; a 6 GB card leaves ~50 expert slots, expect the RX 5500 XT's range, not this one's |
| Radeon PRO V520 / Pro 5600M (RDNA1) | gfx1011 | built by hand: `-DCMAKE_HIP_ARCHITECTURES=gfx1011` | untested on hardware: it is the same RDNA1 ISA as gfx1010 (Navi 12), and the engine builds for it with 0 errors on the same wheels |
| Instinct MI50 / MI60, Radeon VII | gfx906 (wave64) | built by hand: `-DSTRATA_HIP_GFX906=ON` | 2x MI50, Coder IQ1_M, 128K context: decode 50.1 / 47.8 / 45.7 tok/s at 4K / 32K / 128K prompt tokens, prompt ~520 tok/s (#677) |

## NVIDIA: the CUDA 12 engine

CUDA 13 dropped Pascal and Volta: it cannot compile for them. Setup therefore keeps a **second engine**, built with
CUDA 12.9 and `-DSTRATA_EXPERIMENTAL_SM60=ON`, in its own folder (`engine-cuda12\`, beside `engine\`). One engine runs
per model, so the choice is made per model, by the oldest card that model runs on:

- **Every card the model uses is RTX 20 or newer:** the ready-made CUDA 13 engine, as always.
- **A card is Pascal or Volta:** the CUDA 12 engine. Setup says so (`CUDA 12: sm_70 is older than CUDA 13 supports
  ...`). On Windows it downloads `strata-windows-x64-cuda12.zip` with NVIDIA's CUDA 12 libraries (from pip, like the
  CUDA 13 ones); on Linux, or with `--build`, it compiles the engine with a CUDA 12.x toolkit.

### Opting in

An older card is used only when you choose it; a PC with a newer card keeps recommending the newer one.

| You | Setup |
| --- | --- |
| have only Pascal / Volta NVIDIA cards (and no AMD card it can use) | uses them, with the CUDA 12 engine |
| name the card: `START-HERE.bat --setup --gpu 1`, or `--gpus 0,1` with a newer card | uses it; the model gets the CUDA 12 engine |
| `--cuda 12` (or `STRATA_CUDA=12`) | the CUDA 12 engine for this model, on any card |
| `--cuda 13` | the CUDA 13 engine even with an older card (a warning: it has no code for that card) |
| `STRATA_EXPERIMENTAL_SM60=1` | the older cards are listed as usable (the setting from #295 still works) |

The choice is kept in the model's config (`"cuda": 12`): its starts and `UPDATE.bat` keep it, and other models keep
their own engine. Setting the model up again chooses again (by its cards; add `--cuda 12` to keep a forced choice). A Pascal / Volta card added to a model at a start (`--gpus`) moves that model to the
CUDA 12 engine.

`--cuda 12` is also the way to run Strata with an NVIDIA driver older than 580: CUDA 12 needs 528 or newer on Windows
(527.41, NVIDIA's minor-version compatibility) and 525 on Linux; setup's "driver too old" stop says so. Such old
drivers were not tested here.

### Mixed cards

A model that shares a V100 with an RTX 30 / 40 card runs both on the CUDA 12 engine (it has code for sm_60 to sm_89,
plus PTX). An RTX 50 card (sm_120) in a CUDA 12 engine is a warning, not a stop: CUDA 12.8 and newer compile for it,
but engines built with 12.8 crashed on long prompts there (#220, #224). Keep the RTX 50 card on its own model (`--gpu
N`), where it runs the CUDA 13 engine.

### Building it yourself

```sh
# Linux (Windows: the same with the toolkit's nvcc.exe)
cmake -S . -B build-cuda12 -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_EXPERIMENTAL_SM60=ON \
      -DCMAKE_CUDA_ARCHITECTURES="61;70" -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc
cmake --build build-cuda12 --target strata -j
```

Setup does the same when it compiles: it looks for the newest CUDA 12.x toolkit (`STRATA_NVCC=<path to nvcc>` picks
one, #601; on glibc 2.43 use 12.8, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md)).

On GCC 12.3 with nvcc (openEuler 24.03, CUDA 12.8, 2x V100) the build needed `-D_BITS_OPT_RANDOM_H` added to the host flags (#1074; one
report, not reproduced here). The `size_t` error in `vmm.hpp` that the same report hit is fixed in 0.1.40.2.

### What the flag changes, and what it does not

`-DSTRATA_EXPERIMENTAL_SM60=ON` lowers the runtime floor to compute capability 6.0 and compiles the older cards' code
paths into that build only: the Volta prompt attention (#600), #540's attention kernel (used below sm_75), and the
Pascal BF16 path. The ready-made CUDA 13 engine has none of them, so its kernels and its output are unchanged. The
FP16 path for BF16 projections is in every build but runs by default only below sm_75; RTX 20 owners can try it with
`STRATA_BF16_TC=1` (`=0` turns it off on a V100). It is not bitwise the same as cuBLAS's BF16 kernel (FP16 tensor-core
sums round differently): #540 measured a mean KL of 8.4e-3 on the next-token distribution of 24 code prompts on a V100
(the same top-1 in 23), the size of other summation-order changes; #655 a worst relative difference of 3.5e-5 per
product on an RTX 2080 Ti.

A/B switches: `STRATA_BF16_TC=0|1`, `STRATA_PROMPT_ATTN_OLD=1` (the decode kernel for prompts), `STRATA_ATTN_PRE75=0`
(#540's kernel off; on gfx103x with HIP that kernel is the default, see the AMD table above, and `=1` turns it on for
another wave32 AMD card). [NVIDIA_V100.md](NVIDIA_V100.md) has the V100 build, its measurements and the parity test.

## AMD: building gfx906 and gfx1012

Setup does not build these; build by hand and run `serve/server.py` with a config, as on any other card.

- **gfx906** (MI50 / MI60 / Radeon VII, wave64): a separate opt-in build, `-DSTRATA_HIP_GFX906=ON
  -DCMAKE_HIP_ARCHITECTURES=gfx906` (not `STRATA_ENABLE_HIP`). Current ROCm no longer ships gfx906 libraries; the
  reporter used a community ROCm 7.14 image. Recipe, kernels and measurements: [AMD_HIP.md](AMD_HIP.md#gfx906-instinct-mi50--mi60-radeon-vii-wave64-built-from-source).
- **gfx1012** (RX 5500 XT): the wave32 backend, `-DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1012`. HIP 5.7
  (Ubuntu's packages) works: older hipBLAS (0.x) is used through rocBLAS, the legacy HIP names and the missing
  `__syncwarp` are version-gated, and RDNA1's missing signed dot4 uses llama.cpp's SDWA sequence
  (`-DSTRATA_GFX1012_PORTABLE_DOT=ON`: the portable one).

## Turing (sm_75): opt-in interleaved verify projections

`STRATA_MMVQ_IL=1` enables an exact-shape table for dense IQ4_XS, Q4_K, Q5_K and Q6_K projections in
2-4-token verify windows. All sm_75 cards are eligible, including RTX 20 and Quadro RTX cards, but the table was
tuned and measured **only on a Quadro RTX 8000**. An unset value, `0`, or `true` keeps the existing sm_75 path.
Unlisted shapes also keep it. The sm_80-and-newer automatic table is unchanged; Pascal, Volta and HIP do not use
this path. It reorders the same Q8_1 bytes and preserves the multi-column kernel's output bit for bit.

On that RTX 8000, CUDA 12.4 / GCC 13, the table retains 46 of 69 dense shape/window cells: each saved at least
3% in every alternating A/B pair in two seven-pair sweeps, including the full interleave launch cost. Weights
rotate between copies to avoid measuring only L2-resident data. These are not speed guarantees for other sm_75 cards.

Controlled whole-model decode on the same card at 260 W, with a Xeon W-2295, IQ3_S + MTP, `--spec 4`, fixed
19,000 expert slots, automatic prefill (8,192), INT8 KV, maximum context 262,144 / resident 32,768 and PCIe fraction
0.20, gave the following medians of three runs with 128 output tokens each. Prompt caching and adaptation were disabled;
the baseline already enabled the BF16-to-FP16 tensor-core projection path, and only `STRATA_MMVQ_IL=1` changed.

| Prompt tokens | baseline decode (tok/s) | interleaved decode (tok/s) |
| --- | ---: | ---: |
| 537 | 83.7 | 88.9 |
| 4,057 | 75.4 | 80.4 |
| 32,057 | 68.4 | 72.1 |

For a CUDA build with `-DSTRATA_BUILD_TESTS=ON`, `STRATA_MMVQ_IL=1 ./build/mmvq_il_parity` checks bitwise
parity and the actual dispatch path for table, fallback and forced 1/2/4-row kernels. Repeat with `=0` and `=true`
to check strict opt-in behavior; `./build/mmvq_il_parity --sm75-bench` runs the paired shape sweep on sm_75.
No model is needed. Select the intended card with `CUDA_VISIBLE_DEVICES`; other sm_75 hardware needs its own measurements.

## Reports welcome

These paths stay experimental until more people run them. A report with the card, the driver / ROCm version, the
model and the engine log (`strata-<model>.log`) in an issue helps; [COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md)
has the format for measurements.
