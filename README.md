<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="exstrata-logo-dark.svg">
    <img src="exstrata-logo.svg" alt="exstrata" width="484">
  </picture>
</p>

<h3 align="center">Strata, reworked to use less memory on one consumer PC.</h3>
<p align="center">An experimental fork of <a href="https://github.com/Niko1221/Strata">Strata</a> for large MoE models whose experts live in system RAM.</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-experimental-orange" alt="Status: experimental">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT license"></a>
  <a href="https://github.com/Niko1221/Strata/commit/3142b66b9"><img src="https://img.shields.io/badge/based%20on-Strata%203142b66-555" alt="Based on Strata 3142b66"></a>
</p>

**exstrata** keeps Strata's inference engine and reduces the memory footprint of large MoE models—so far cutting combined VRAM + RAM by **14.1%** on Qwen3.8-Flash-Next with no loss in task accuracy. Every change is opt-in: with nothing enabled, exstrata runs identically to Strata.

> exstrata is an independent fork and is not affiliated with or endorsed by the Strata project.

---

## Results: Qwen3.8-Flash-Next

Measured against Strata using the same executable and model, differing only in the opt-in switches and the compressed n-gram table (`T58e`, 17.0 GiB vs. Strata's 26.8 GiB `IQ4_NL` table).

### Memory

| Metric | Strata | exstrata | Change |
| --- | ---: | ---: | ---: |
| VRAM (peak minus idle baseline) | 14,173 MiB | 13,487 MiB | −4.8% |
| RAM (working-set peak) | 71,526 MiB | 60,103 MiB | −16.0% |
| **VRAM + RAM** | **85,699 MiB** | **73,590 MiB** | **−14.1%** |

**Where the savings come from** (measured on one machine):

- **n-gram table (`T58e`):** ~−9.8 GiB RAM (57 bytes/row instead of 90; decoded bit-for-bit as encoded).
- **Fixed VRAM (`STRATA_DENSE_ARENA=1` + `STRATA_MTP_HEAD_INDEX=1`):** −0.37 GiB VRAM.
- **Lazy CUDA module loading (`CUDA_MODULE_LOADING=LAZY`):** −0.31 GiB VRAM and ~−1.3 GiB RAM (a standard CUDA setting).

### Accuracy and speed

Changes are accepted only if they pass a two-stage verification gate: a fast teacher-forcing KL screen (`≤ 0.10`), followed by a paired comparison on 914 verifiable items (accuracy `≥ 91%` and within 2 points of Strata) and a decode speed floor (`≥ 0.97×` Strata at the 95% CI lower bound).

| Overall accuracy (914 items) | Strata | exstrata | Change |
| --- | ---: | ---: | ---: |
| Correct answers | 836 | 843 | +7 |
| Accuracy | 91.5% | 92.2% | +0.77 pts (+0.84% rel.) |
| Wrong answers | 78 | 71 | −9.0% |

The difference is within the noise of a 914-item test (95% CI: −0.44 to +1.97 points; 12 items only Strata got right, 19 only exstrata), so read this as **no accuracy loss**, not an improvement.

| Benchmark (items) | Strata | exstrata |
| --- | ---: | ---: |
| GSM8K (200) | 96.0% | 95.5% |
| MATH-500 (150) | 86.7% | 90.0% |
| HumanEval (164) | 94.5% | 95.7% |
| MMLU-Pro (200) | 82.5% | 83.5% |
| JCommonsenseQA (200) | 97.0% | 96.5% |

| Speed & divergence check | Strata | exstrata | Gate |
| --- | ---: | ---: | --- |
| Teacher-forcing KL (agent segment) | – | 0.071 | ≤ 0.10 |
| Decode speed (relative to Strata) | 1.00 | 0.994 (95% CI: 0.983–0.997) | lower bound ≥ 0.97 |
| Prompt speed (relative, long prompt) | 1.00 | 0.980 (95% CI: 0.976–0.984) | reported only |

**Test environment:** Windows, single RTX 5070 Ti (16 GB), Qwen3.8-Flash-Next `IQ3_XXS`, MTP drafting, 30 alternating rounds with a 3,000-word prompt, measuring peak VRAM minus idle baseline and peak RAM working set. Results will vary across hardware and settings.

**What did not make the cut:** Reaching ~−15% while keeping speed. Keeping each routed expert in only one place (VRAM or RAM) reached −15.2% in the same test, but decode speed dropped to `0.974×` (95% CI lower bound `0.964`, below the `0.97` gate), so it is not included.

---

## What exstrata changes

Four opt-in changes reduce memory on Qwen3.8-Flash-Next. Three are bit-for-bit identical in output; only the compressed n-gram table changes output slightly (verified above).

| Change | Switch | Memory saved | Output |
| --- | --- | --- | --- |
| [1. Smaller n-gram table](#1-smaller-n-gram-table---ple-gguf) | `--ple-gguf` (`T58e`) | ~9.8 GiB RAM | Slight diff (checked) |
| [2. Dense weights arena](#2-single-allocation-for-dense-weights-strata_dense_arena1) | `STRATA_DENSE_ARENA=1` | 186 MiB VRAM | Bit-identical |
| [3. Shared draft head](#3-draft-head-without-a-copy-strata_mtp_head_index1) | `STRATA_MTP_HEAD_INDEX=1` | 178 MiB VRAM | Bit-identical |
| [4. Lazy CUDA loading](#4-lazy-cuda-module-loading-cuda_module_loadinglazy) | `CUDA_MODULE_LOADING=LAZY` | ~0.31 GiB VRAM, ~1.3 GiB RAM | Bit-identical |

### 1. Smaller n-gram table (`--ple-gguf`)

Every token reads rows from a large per-layer n-gram (PLE) table: 320,001,536 rows of 160 values. Strata stores it as `IQ4_NL` (90 bytes/row, 26.8 GiB in RAM). exstrata stores the same table in 57 bytes/row (17.0 GiB) using a 3-bit trellis format called `T58e`:

- **Trellis coding:** A row's 160 values are encoded as a single 448-bit string (2.8 bits/value). Each value reads a sliding 12-bit window into a shared table of 4,096 values. Neighboring windows overlap, so each value adds only 2–3 new bits while still selecting from 4,096 levels. The string wraps around so the first value reads from the end of the row.
- **Row scale:** One extra byte stores the row's scale as an 8-bit float (FP8).
- **Fast decode:** The offline encoder runs a Viterbi search to find the bit string closest to the original row. At inference time, the engine only decodes (reads a row, looks up values, applies the scale). Reading, memory mapping, and the prompt path work unchanged.
- **Approximation:** Reconstructed values are close to `IQ4_NL` rather than exact (relative squared error 0.024), which is why full task accuracy is verified.

**Usage:**

```bash
# 1. Fit the shared 4,096-value codebook on a slice of the table
#    (outputs slice.gguf.codebook.json)
python terminal/tools/ple_lowbit_emulate.py \
  --format t58e --native \
  --src <IQ4_NL_shard_2.gguf> \
  --out slice.gguf \
  --rows 0:200000

# 2. Encode the full table (takes a few hours on CPU)
python terminal/tools/ple_lowbit_emulate.py \
  --format t58e --native \
  --src <IQ4_NL_shard_2.gguf> \
  --codebook slice.gguf.codebook.json \
  --out ple-t58e.gguf
```

Then start the engine with `--ple-gguf ple-t58e.gguf --ple-io ram`.

To verify that the engine decodes a table identically to the encoder, run `ple_fp8_parity` (built with the engine) with `PLE_EXPECT_FORMAT=T58e` and pass the table, its `.ref.bin`, and the `IQ4_NL` shard.

<details>
<summary>Other table formats evaluated (C1–C6, T58)</summary>

Other sizes were built and screened on the way to `T58e`. `T58e` is recommended; the rest remain available:

| Format | Bytes/row | Teacher-forcing KL, agent segment (≤ 0.10) |
| --- | ---: | ---: |
| `IQ4_NL` (Strata) | ~90 | reference |
| `C1` | 82 | 0.079 |
| `C2`, `C2s` | 70 | 0.101, 0.098 |
| `C3` | 65 | 0.096 |
| `C4`, `C5` | 62 | 0.116, 0.097 |
| `C5e` | 61 | 0.069 |
| `T58` | 58 | 0.078 |
| **`T58e`** | **57** | **0.071** |
| `C6` | 50 | 0.133 |

The KL numbers carry about ±0.03 of noise between nearly identical tables, so they separate pass from fail rather than giving an exact ranking. `C4` and `C6` fail the limit; `T58e` is the smallest format that passes with margin and was therefore evaluated in full.

</details>

### 2. Single allocation for dense weights (`STRATA_DENSE_ARENA=1`)

Native dense weights (projection matrices) were previously allocated one matrix at a time. Enabling this switch places them in a single VRAM allocation, saving 186 MiB. Both the bytes and model output are unchanged; the engine aborts at startup if copies do not land where the sizing pass expected.

### 3. Draft head without a copy (`STRATA_MTP_HEAD_INDEX=1`)

The MTP draft head previously kept its own 178 MiB copy of rows already present in the main output head. Enabling this switch reads those rows directly from the main head by row index. Model output is unchanged.

### 4. Lazy CUDA module loading (`CUDA_MODULE_LOADING=LAZY`)

A standard CUDA environment variable rather than an exstrata switch: CUDA kernels load on first use instead of at startup. Saves ~0.31 GiB of VRAM and ~1.3 GiB of RAM without changing output.

### 5. Adaptive VRAM & RAM tiers in partial-RAM mode (`STRATA_ADAPT_FETCH_RESIDENT=1`, `STRATA_RAM_ADAPT_COLD=1`) & low-memory presets (`v0.2.20-ex.2`)

For systems targeting **VRAM ≤ 12 GiB** and/or **RAM ≤ 32 GiB**, exstrata (`v0.2.20-ex.2`, based on Strata v0.2.20) adds two opt-in, bit-identical / distribution-preserving algorithms:

- **`STRATA_ADAPT_FETCH_RESIDENT=1`:** In Strata's `--resident-experts --resident-budget-gib N` mode (where a subset of MoE experts stays pinned in RAM and the coldest tail reads from SSD), `STRATA_ADAPT_FETCH=2` is disabled by default. Enabling `STRATA_ADAPT_FETCH_RESIDENT=1` re-enables adaptive VRAM admission for RAM-resident experts only, avoiding file reads on hot swaps.
- **`STRATA_RAM_ADAPT_COLD=1` (Dynamic RAM Tier Promotion):** Between decode windows (`adapt()`), promotes frequently routed cold-tier experts from the OS page cache into the fixed `--resident-budget-gib N` pinned RAM buffer by evicting never/rarely-used RAM-resident experts of the same layer. Uses **0 extra RAM**, preserves **bit-identical** weights, and cuts cold-tier file reads by up to **−47% to −78%** (boosting long-context decode by **+25.4%** in partial-RAM mode).
- **Windows `--native` + `--resident-experts` fix:** Skips the unbuffered startup view-unmap path when reading native GGUF shards in place on Windows so partial-RAM mode runs cleanly on Strata v0.2.20.

#### Preset Comparison (RTX 5070 Ti 16 GB, Qwen3.8-Flash-Next `IQ3_XXS`)

| Preset | Key Options / Switches | VRAM | Effective RAM | Total vs. Strata | Decode (med / max) |
| :--- | :--- | ---: | ---: | ---: | ---: |
| **Strata baseline** | `--ple-io ram` (`IQ4_NL`) | 14.0 GiB | ~70.8 GiB | baseline | ~99 tok/s |
| **`F3H` (Speed)** | `T58e` + `--ple-io ram` + `--mtp-hnorm stream` | 13.17 GiB | ~55.2 GiB | −19.4% | **105.6 tok/s** |
| **`D16MH` (Fast-SSD)** | `F3H` + `--ple-io direct --ple-row-cache 16777216` | 13.17 GiB | ~38.8 GiB | **−33.3%** | **105.9 tok/s** |
| **`V12` (VRAM ≤ 12 GiB)** | `D16MH` + `--expert-cache 3000` (`ExpertArena`, v0.2.20) | **11.92 GiB** | ~38.8 GiB | −34.8% | **95.8 / 99.7 tok/s** |
| **`V12_R30` (VRAM ≤ 12 GiB & RAM ≤ 32 GiB)** | `V12` + `--resident-experts --resident-budget-gib 30` + `STRATA_ADAPT_FETCH_RESIDENT=1` (+ `STRATA_RAM_ADAPT_COLD=1`) | **11.92 GiB** | **~31.9 GiB** | **−47.7%** | **73.6 / 80.4 tok/s** |

> **Versioning:** exstrata releases follow `v<upstream-strata>-ex.<N>` (e.g., `v0.2.20-ex.2` for Strata v0.2.20 + exstrata revision 2).

### Tools

- `terminal/tools/ple_lowbit_emulate.py` — encodes the PLE tables above (and, without `--native`, writes a `Q8_0` copy for accuracy tests).
- `terminal/tools/ple_tf_screen.py` — writes a sparse table containing only the rows read during a teacher-forcing run (a few GB instead of 17–26 GB), allowing formats to be screened before encoding the full table.
- `terminal/tools/iq_eval/` — runs the 914-item paired accuracy evaluation reported in [Results](#results-qwen38-flash-next).

---

## Building and running

Building and running follow Strata's workflow—see [Build from source](docs/SOURCE_INSTALL.md) and the [install guide](docs/INSTALL.md).

To enable all exstrata memory reductions, set the environment variables `STRATA_DENSE_ARENA=1`, `STRATA_MTP_HEAD_INDEX=1`, and `CUDA_MODULE_LOADING=LAZY`, and pass `--ple-gguf ple-t58e.gguf --ple-io ram` with a `T58e` table. See [docs/SETTINGS.md](docs/SETTINGS.md) for where each switch applies.

---

## Design principles and roadmap

exstrata follows two core constraints:

- **Reduce, do not offload.** Make stored data smaller rather than spilling to disk, which hurts machines with slow SSDs.
- **Preserve capability.** Any change that alters output must pass both teacher-forcing screening and paired accuracy checks across GSM8K, MATH-500, HumanEval, MMLU-Pro, and JCommonsenseQA.

### Work in progress: drafting for CPU-resident experts

Every token in a verify window reads experts from system RAM whether or not the token is accepted, so a wrong draft wastes memory bandwidth. Ongoing work focuses on tuning drafting for that cost:

- tuning the MTP drafter's settings;
- comparing the MTP drafter against the DFlash2 drafter;
- ending a verify window early when expected accepted tokens no longer justify the RAM reads.

---

## Relationship to Strata

exstrata is a GitHub fork of Strata and preserves Strata's full commit history and contributors.

| Branch | Contents |
| --- | --- |
| `exstrata` (default) | exstrata itself |
| `main` | Unchanged mirror of Strata's `main`, updated via GitHub's "Sync fork" |

Upstream changes from Strata reach `exstrata` via merges from `main`. The "based on" badge at the top names the Strata commit currently merged in.

## License and credits

exstrata is released under the MIT license, matching Strata. [LICENSE](LICENSE) retains Strata's copyright notice: `"Copyright (c) 2026 Niko1221 and the Strata contributors"`. The engine, tools, and documentation originate from [Strata](https://github.com/Niko1221/Strata).

Please report issues that also occur in Strata upstream to Strata, and report issues specific to exstrata here.
