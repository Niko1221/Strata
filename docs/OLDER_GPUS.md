# Older GPUs (experimental)

Strata's supported cards are NVIDIA RTX 20 / 30 / 40 / 50 and the AMD cards in [AMD_HIP.md](AMD_HIP.md). The cards
below run through **opt-in, experimental** paths that community members wrote and measured on their own machines. The
maintainers have none of these cards: each path is compile-checked and unit-tested here, and the ready-made engines
and their output stay exactly as they were. Numbers are the reporters' own, on one machine each.

## Support matrix

### NVIDIA

| Cards | Compute capability | How it runs | What is different on it | Reported |
| --- | --- | --- | --- | --- |
| Tesla P100 | 6.0 | the CUDA 12 engine | `__dp4a` emulated (bit-exact); BF16 projections through fp32 | 2x P100, IQ3_S, engine 0.1.39: prompt 526-532 tok/s, decode 28-32 tok/s at 128K prompt tokens, 3 of 3 needle checks at 122K ([report](../bench/results/2026-10-06-community-2x-p100/README.md), #1157) |
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

### Quadro RTX 8000: measurements of existing switches

The Quadro RTX 8000 is Turing (sm_75): it has FP16 tensor cores, but no native BF16 or TF32 tensor cores. Two
**existing, opt-in** CUDA switches are worth comparing on your own workload:

- `STRATA_BF16_TC=1` converts BF16 projection operands to FP16 and uses FP16 tensor cores with FP32 accumulation.
  The default on sm_75 is off; `=0` restores the cuBLAS BF16 path. The switch applies to 7.x cards, not the native
  BF16 path on sm_80 and newer. Conversion clamps finite values outside FP16's range to ±65504; small values and
  differently ordered sums can round differently.
- `STRATA_SELECT_SIMT=1` uses the tiled FP32 QSA block scorer on the pre-sm_80 CUDA path instead of the default
  warp scorer. `=0` keeps the warp scorer. It does not enable TF32 on Turing, and its summation order differs.

The existing paths are in [gemm.cu](../src/prefill/gemm.cu) and [qsa_select.cu](../src/kernels/cuda/qsa_select.cu).

Set the environment before starting the engine, or set the switches in the model config's `env` object, then
restart. For example, this config fragment opts into both (keep the config's other settings):

```json
"env": {
  "STRATA_BF16_TC": "1",
  "STRATA_SELECT_SIMT": "1"
}
```

This section records an investigation, **not a new kernel or a change to defaults**. Measurements on 2026-10-07
used one 48 GiB Quadro RTX 8000 (physical GPU 3), a 260 W power limit, a Xeon W-2295, CUDA 12.4 and GCC 13.
The source-built engine was based on v0.1.40.3 with other local changes integrated; unrelated opt-in changes
were disabled in these arms. These are not measurements of an otherwise clean documentation-only checkout,
not results at the launcher's 200 W setting, and not a speed promise for every sm_75 card.

The model was IQ3_S with MTP (`--spec 4`), `--max-context 262144`, INT8 KV with `--kv-resident 32768`,
`--pcie-frac 0.20`, a fixed `--expert-cache 19000` request and `--prefill auto` (8,192-token ceiling).
The engine reported 21,189 cache slots including the prompt loan. Prompt caching and adaptive swaps were off
(`--prompt-cache 0 --adapt-swaps 0`). Each size used three fresh, seeded synthetic prompts and 128 output tokens
per request; the table gives median engine-reported rates, not end-to-end server latency.

| Existing switch settings | 537 tokens: prompt / decode tok/s | 4,057 tokens: prompt / decode tok/s | 32,057 tokens: prompt / decode tok/s |
| --- | --- | --- | --- |
| BF16_TC=0, SELECT_SIMT=0 | 493 / 86.8 | 1,055 / 79.9 | 1,135 / 70.6 |
| BF16_TC=1, SELECT_SIMT=0 | 661 / 83.7 | 1,262 / 75.4 | 1,416 / 68.4 |
| BF16_TC=1, SELECT_SIMT=1 | 679 / 87.1 | 1,290 / 78.2 | 1,501 / 71.6 |

BF16_TC improved prompt throughput by about 20–34%, but did not improve decode in these comparisons. Adding
SELECT_SIMT improved the 32K prompt rate by about 6% over BF16_TC alone; this is not a universal decode win.
BF16_TC changed replies in 3 of the 9 paired requests, so use it only if different rounding is acceptable.
All five basic arithmetic/JSON checks passed in each arm, which is a smoke check, **not a model-quality study**.
A separate synthetic QSA scorer check at 131,072 context tokens matched selected IDs for all 256 queries and
passed its FP64 accuracy gate. That check does not prove identical selections on arbitrary prompts or model quality.

Two other existing tuning options did not justify a recommendation on this machine:

- `--prefill auto:16384`, with BF16_TC on and SELECT_SIMT off, increased the prompt's cache loan from 4.38 to
  7.38 GiB for only about 0.6% more 32K prompt throughput (1,425 versus 1,416 tok/s). Just 24 MiB of VRAM remained
  at the last graph capture. Keep the 8,192 ceiling rather than spend that memory margin for this small gain.
- `STRATA_PREFILL_CPU_SHARE=auto` gave no gain at the two tested sizes (537 and 4,057 tokens): prompt rates were
  647 and 1,253 tok/s versus 661 and 1,262 with BF16_TC alone; decode was also lower. Larger prompts were not
  tested for this setting.

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
2-4-token verify windows. All sm_75 cards are eligible, including RTX 20 and Quadro RTX cards, but the original
table tuning and author measurements used **only a Quadro RTX 8000**. An unset value, `0`, or `true` keeps the existing sm_75 path.
Unlisted shapes also keep it. The other architectures' tables, including Pascal/Volta and the sm_86 Q3_K and
one-column choices, are unchanged; HIP has no interleaved path. The sm_75 table is registered in `kIlArch`,
but uses exact input/output dimensions rather than output-row classes. `STRATA_MMVQ_IL_ROWS` does not override
this exact-shape policy on sm_75; it still works on the other eligible architectures. On sm_75 an unspecified
input width, Q3_K or one column falls back. It reorders the same Q8_1 bytes and preserves the multi-column
kernel's output bit for bit.

In the original RTX 8000 measurements, CUDA 12.4 / GCC 13, the table retained 46 of 69 dense shape/window cells: each saved at least
3% in every alternating A/B pair in two seven-pair sweeps, including the full interleave launch cost. Weights
rotate between copies to avoid measuring only L2-resident data. These are not speed guarantees for other sm_75 cards.

Previously recorded controlled whole-model decode on the same card at 260 W, with a Xeon W-2295, IQ3_S + MTP, `--spec 4`, fixed
19,000 expert slots, automatic prefill (8,192), INT8 KV, maximum context 262,144 / resident 32,768 and PCIe fraction
0.20, gave the following medians of three runs with 128 output tokens each. Prompt caching and adaptation were disabled;
the baseline already enabled the BF16-to-FP16 tensor-core projection path, and only `STRATA_MMVQ_IL=1` changed.
These are historical measurements, not a whole-model retest of the rebased PR.

| Prompt tokens | baseline decode (tok/s) | interleaved decode (tok/s) |
| --- | ---: | ---: |
| 537 | 83.7 | 88.9 |
| 4,057 | 75.4 | 80.4 |
| 32,057 | 68.4 | 72.1 |

For a CUDA build with `-DSTRATA_BUILD_TESTS=ON`, `STRATA_MMVQ_IL=1 ./build/mmvq_il_parity` checks bitwise
parity and the actual dispatch path for table, fallback and forced 1/2/4-row kernels. Repeat with `=0` and `=true`
to check strict opt-in behavior; `./build/mmvq_il_parity --sm75-bench` runs the paired shape sweep on sm_75.
No model is needed. Select the intended card with `CUDA_VISIBLE_DEVICES`; other sm_75 hardware needs its own measurements.
`./build/mmvq_il_parity --tables-only` checks architecture selection without initializing CUDA; with
`STRATA_MMVQ_IL_ROWS` set, `--sm75-tables-only` checks that class overrides cannot broaden the sm_75 table.

### Community reports (not tests of the rebased PR head)

- [ioridev's RTX 2080 Ti report](https://github.com/Niko1221/Strata/pull/1437#issuecomment-6108554652)
  ports the unchanged shape table, strict opt-in and actual input widths onto 0.1.42 (`61b3fb5d`).
  It does **not** use the original PR head or port its extended parity harness/last-launch observation.
  On two RTX 2080 Ti cards (22 GB main / 11 GB peer, 250 W, i9-9820X, CUDA 12.4/GCC 15.2),
  IQ3_S + MTP, spec 4, INT8 KV, BF16_TC and SELECT_SIMT enabled, the reporter found bitwise kernel parity
  with 0.1.42's harness and mean decode 68.3 → 71.3 tok/s (+4.4%) across five workloads. Medians use
  four runs per cell with reversed arm order. The ~8K result was noisy; baseline temperature-zero
  replies already varied, so text equality was not a correctness check. This is independent evidence
  for a port, not our own measurement or validation of the rebased code.
- [xrip's CMP 50HX report](https://github.com/Niko1221/Strata/pull/1437#issuecomment-6122392200)
  uses a **different** output-row-class table, with Q3_K and one-column choices, not this PR's exact-shape table.
  On two CMP 50HX cards (TU102, 2100 MHz, 56 SMs, CUDA 13.3/GCC 14), IQ3_XXS, layer split,
  spec 4 + MTP, INT8 KV and PCIe fraction 0.17, the reporter found bitwise T=1–4 kernel parity
  and +7.0% decode (72.62/72.57 → 77.85/77.46 tok/s) for a 35.7K-token prompt followed by six
  512-token questions. That table is not adopted here. The same comment's `STRATA_EXP_MODE=8`
  results concern the expert layout and are outside this PR.

## Reports welcome

These paths stay experimental until more people run them. A report with the card, the driver / ROCm version, the
model and the engine log (`strata-<model>.log`) in an issue helps; [COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md)
has the format for measurements.
