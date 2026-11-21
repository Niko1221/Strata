# RTX 3060 12 GB — Strata 0.1.63 / Qwen3.8-Flash-Next Q2_0 / 260K benchmark

> **Headline:** A single RTX 3060 12 GB eGPU processed **260,050 input tokens** at **1,160.3 prefill tok/s** and **42.8 decode tok/s** with **Thinking XHIGH**. The dedicated synthetic recall checks at this context passed **2/2**. All tests used **GPU Unlock ON** and **drafting OFF**.

## Quick overview — the numbers that matter

| Configured context | Actual long input | Prefill tok/s | Decode tok/s | TTFT | Recall |
|---|---:|---:|---:|---:|---:|
| 32K | 32,051 | **1,422.1** | 49.3 | 22.57 s | 2/2 |
| 64K | 64,044 | **1,386.9** | 49.6 | 46.23 s | 2/2 |
| 128K | 128,030 | **1,308.1** | 47.8 | 97.95 s | 2/2 |
| **262K** | **260,050** | **1,160.3** | **42.8** | **224.25 s** | **2/2** |

**What stands out:**

- **All 8 test runs completed:** four upstream standard pilot runs and four separate long-input runs. **84/84 required standard requests**, **48/48 standard sanity checks**, and **20/20 long-context requests** completed; dedicated synthetic recall passed **8/8**.
- **Prefill throughput declined 18.4% from 32K to 260K actual input**, while input length increased about eightfold. Median speed is from **three repetitions per context**, not a best single run.
- **262K hardware observations:** maximum sampled **72°C**, **168.92 W**, minimum **10.70 GiB available RAM** and **562/562 PCIe samples at Gen4 ×4**.
- **No speedup claim from GPU Unlock:** there was **no Unlock-OFF comparison**. The recall test is synthetic key retrieval, not a general reasoning benchmark.

**Machine:** Minisforum AI X1-255 / AMD Ryzen 7 255 / ~60 GiB usable RAM / RTX 3060 12 GB via DEG1 OCuLink PCIe Gen4 ×4 / Ubuntu 26.04.1 / Strata 0.1.63 CUDA / NVIDIA 595.99.02. **Model:** Qwen3.8-Flash-Next Q2_0; **settings:** thinking ON (`xhigh`), drafting OFF, vision OFF, parallel 1, GPU Unlock ON.

**How to read this report:** the long-input table above is from a **non-standard synthetic extension** that actually read ~32K–260K prompts. The **official** `strata-qwen38-thinking-v1-draft.1` protocol uses fixed workloads of up to ~16K tokens even when 262K is configured. `draft.1` is an upstream **protocol identifier**, not this PR's status. See [standard runs](#standard-protocol--configured-context-fixed-requests-up-to-16k), [full long-context measurements](#non-standard-extension--actual-long-inputs-thinking-xhigh), [GPU telemetry](#gpu-memory-pcie-telemetry-long-context-sweep) and [limitations](#limitations-and-publication-notes).

---

## Detailed evidence and methodology

**Measured on 2026-11-21 (UTC). Completed evidence submission, not an official leaderboard approval.** Four runs of the upstream standard protocol and a separate four-context, **non-standard synthetic** long-context extension. All runs used **Thinking ON (`reasoning_effort=xhigh`)**, **drafting OFF**, **GPU Unlock ON**, vision OFF, parallel capacity 1. This PR is intended as a regular results-only PR for maintainer review.

## Hardware

- Minisforum AI X1-255, AMD Ryzen 7 255; ~60 GiB usable memory; Ubuntu 26.04.1 LTS, kernel 7.0.0-43-generic.
- RTX 3060 12 GiB (12,288 MiB), NVIDIA 595.99.02, CUDA 13.2, 170 W power cap, DEG1 OCuLink eGPU; PCIe 4.0 ×4 under active load.
- Strata 0.1.63 CUDA **release** binary; standard runner commit `12d491da4a6904eb49aaa6f2f63ec4b1a17582e3` recorded in four original standard JSON artifacts.
- Model: Qwen3.8-Flash-Next Q2_0 from `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, declared quant-source revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`. Original GGUF SHA256 values recorded per run. The base model source revision is not given for some standard runs; preserve these unknowns (see Limitations).
- Experimentally enabled GPU Unlock: ON in all measurements. **No Unlock-OFF control**, so no causal Unlock speedup is claimed.

## Standard protocol — configured context (fixed requests up to ~16K)

The exact upstream pilot protocol ID is **`strata-qwen38-thinking-v1-draft.1`**. The `draft.1` string names the protocol version; **all four benchmark runs are completed**. It does not mean this submission is a Draft pull request.

| Configured context | Median prefill 4K tok/s | Median prefill 16K tok/s | Median 16K decode tok/s | Sanity | Raw original standard files |
|---|---:|---:|---:|---:|---|
| 32K (32,768) | 780.3 | 1378.0 | 46.8 | 12/12 | [Evidence](../2026-11-21-standard-rtx-3060-q2-0-32k/) |
| 64K (65,536) | 775.2 | 1319.4 | 46.5 | 12/12 | [Evidence](../2026-11-21-standard-rtx-3060-q2-0-64k/) |
| 128K (131,072) | 772.8 | 1319.6 | 45.7 | 12/12 | [Evidence](../2026-11-21-standard-rtx-3060-q2-0-128k/) |
| 262K (262,144) | 757.2 | 1310.7 | 46.1 | 12/12 | [Evidence](../2026-11-21-standard-rtx-3060-q2-0-262k/) |

Each standard run: **21/21 required requests**, 0 failed, 0 skipped; all 12 sanity checks passed, optional concurrency ran. Thinking and sampling follow the fixed upstream protocol. For individual samples, cap status, optional rounds, model metadata, templates and original byte-level artifact hashes, open each linked folder. A context setting of 262K **does not imply** that the standard workload had 262K-length prompts.

## Non-standard extension — actual long inputs (thinking xhigh)

| Configured context | Actual speed input tokens | Actual recall input tokens | Median prefill tok/s | Median decode tok/s | Median TTFT s | Recall 10%+90% | Evidence |
|---|---:|---:|---:|---:|---:|---:|---|
| 32K (32,768) | 32,051 | 30,510 | 1422.1 | 49.3 | 22.57 | 2/2 | [Evidence](runs/32k/) |
| 64K (65,536) | 64,044 | 63,262 | 1386.9 | 49.6 | 46.23 | 2/2 | [Evidence](runs/64k/) |
| 128K (131,072) | 128,030 | 128,030 | 1308.1 | 47.8 | 97.95 | 2/2 | [Evidence](runs/128k/) |
| 262K (262,144) | 260,050 | 259,889 | 1160.3 | 42.8 | 224.25 | 2/2 | [Evidence](runs/262k/) |

- Each configuration had 3 speed samples (50%-depth synthetic key, 512 maximum output tokens) and 2 recall samples (10% and 90% key depth, 2048 maximum output tokens), using the same Qwen thinking/sampling parameters as the standard protocol. All 20 ended with `finish_reason=stop`, and the 8 dedicated recall samples matched the expected synthetic key.
- The 262K configured context (`262144`) **actually processed 260050 input tokens on speed repetitions** (259889 on recall). Median TTFT at that size was about 224.25 s including prefill. Throughput is a read/decode measurement, not an estimate from total request duration.
- TTFT is first **streamed token including reasoning**; `first_answer_s` is tracked separately per run. Stable small synthetic key recall is not a demonstration of general reasoning quality.
- Exact experiment scripts: [Python v3](reproduce/strata_long_context_bench.py) and [shell v3](reproduce/strata_long_context_bench.sh). They rely on the locally installed Strata tokenizer and Q2_0 pack; the shell script assumes the scripts are placed under `$HOME/strata-test/` and an already-running matching-context engine at `127.0.0.1:18080`.
- Source-derived [performance.csv](performance.csv) and [GPU telemetry summary](gpu-telemetry-summary.csv) are included. Per-context original JSONL, summary, manifest, telemetry and GPU details are under `runs/`.

## GPU, memory, PCIe telemetry (long-context sweep)

| Context | Observed peak temp | Observed peak power | Max VRAM | Min MemAvailable | PCIe Gen4 ×4 samples |
|---|---:|---:|---:|---:|---:|
| 32K | 69 °C | 169.71 W | 11406 MiB | 19.45 GiB | 66/67 |
| 64K | 71 °C | 168.58 W | 11408 MiB | 16.32 GiB | 124/124 |
| 128K | 72 °C | 168.19 W | 11406 MiB | 10.91 GiB | 252/252 |
| 262K | 72 °C | 168.92 W | 11408 MiB | 10.70 GiB | 562/562 |

Monitoring sampled approximately once every two seconds. The long-context wrapper aborts at GPU temperature ≥75°C and on severe RAM/swap pressure. All test runs completed without triggering a guard. The 262K test retained at least ~10.7 GiB `MemAvailable` in sampled telemetry and recorded Gen4 ×4 on every sample. These data do not prove absence of transient events between samples.

## Method and reproducibility

- All measurements came from the same eight-run evidence archive, generated on the operator's machine. Raw official `results.json`, `outputs.jsonl`, `config.json`, the frozen standard corpus hash file and chat templates are included **unchanged**. Original artifact SHA256 references inside `results.json` were revalidated against the exact committed files in each standard folder.
- Extra information from the runner's `metrics-start.json` and sampled CSV provides support for some dynamic cache/KV settings. Standard `configuration` properties not exposed by the running server remain `null` in the original JSON; they were **not backfilled** from later or unrelated runs.
- Custom Python script generates deterministic synthetic prompts locally with Strata's BPE tokenizer; seeds and code markers, prompt SHA256, actual server-reported usage, raw per-case timing/finish status are in `long-results.jsonl`. This is **not** the upstream `terminal/tools/needle_bench.py` workload.
- Output/telemetry files are local copies of evidence, not server-side GitHub-generated data. `SHA256SUMS` lists exact package files. Full engine runtime logs were not included in the original archive, so sanitized launcher output is provided in the standard folders, explicitly labelled as such.

## Limitations and publication notes

- **Pilot protocol:** upstream identifier is `strata-qwen38-thinking-v1-draft.1`. Submitting complete evidence does not constitute independent verification or inclusion in the upstream approved list. Maintainer review is required.
- **Missing fields:** Standard results explicitly lack tokenizer revision, build options, and several runtime configuration fields; `model.source_revision` is also absent in the 128K and 262K run JSON. These are **not measured / not given**, not inferred from the other runs.
- **Cold/warm status:** The standard runner used an already-started engine; the long synthetic sweep followed it for each context. This is not a clean cold-cache comparison, and no caching-effect claim is made.
- **No controlled comparison:** All eight runs used GPU Unlock ON; no off/on speedup claim. Avoid direct speed comparisons with other quantizations, versions, buses, or OS without acknowledging differences.
- **Privacy:** The original source archive included private host facts and unredacted launcher logs. These are **not included** here. The committed official raw benchmark JSON and outputs were checked for identifiable paths, addresses and obvious secret patterns; private binary/model files are excluded.

## Exact evidence links

All paths are relative to this report. Standard folders are siblings under `docs/bench-results/`; synthetic long-context runs are [32K](runs/32k/), [64K](runs/64k/), [128K](runs/128k/), and [262K](runs/262k/). See [SHA256SUMS](SHA256SUMS) for hashes of all files in this report folder.
