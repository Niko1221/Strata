# Community speed and long-context recall benchmark — RTX 5070 / DeepSeek V4.1 Flash 1017 (Huihui, operator-reported)

Measured **2026-11-05** on **Windows 11 Pro**, from an already running local Strata server (`127.0.0.1:8070`). This is a **community speed report**, not a `strata bench run` standard-protocol result and not a quantization-fidelity study.

The model was identified by the operator as **DeepSeek-V4.1-Flash-1017 Huihui abliterated, ISTA-DAS GSQ-RCO IQ2_XXS 2.00 bpw**. The benchmark server reported only the generic model ID `deepseek-v4.1-flash`. A **later local collection at 17:23 +01:00** supplies four on-disk GGUF SHA-256 hashes and an allowlisted expected configuration (see [PROVENANCE.md](PROVENANCE.md)), but **does not prove those files or settings were loaded by the earlier benchmark process**. The exact active model identity therefore remains unverified.

## Hardware and software

- **GPU:** NVIDIA GeForce RTX 5070, **12,227 MiB** reported VRAM; NVIDIA driver **621.14**; NVIDIA-reported CUDA *driver maximum* **13.5** (not a verified CUDA toolkit/build version).
- **GPU snapshot after each run:** PCIe **Gen4 ×16**, reported configured power limit **250 W**; GPU draw snapshots **81.04 W (OFF)** and **79.12 W (ON)**, neither is average or peak inference power.
- **CPU:** AMD Ryzen 7 9700X, **8 cores / 16 threads**. Installed RAM **125.6 GiB**.
- **Storage detected:** Samsung SSD 990 EVO 1 TB and Samsung SSD 990 PRO 2 TB, both SSD/NVMe. The benchmark does not identify which drive actually held all model assets.
- **OS:** Windows 11 Pro, version `10.0.26300`. **Python:** 3.12.15.
- **Strata:** project **0.1.56**, checkout commit **`ffa80d73133ea5c548aefe57889a16e8085a938f`**; server engine version **`ds4-0.1`**. The original benchmark environment summary reported a **modified Git working tree**; a **later** snapshot at 17:23 reported a **clean tree** at the same commit. The intervening history and any changes present *during* the benchmark are unknown. Build flags were not provided.
- **Background workload:** not recorded. No continuous VRAM/RAM utilization trace, temperature or power time series was collected.

## Model and runtime as observed

- Reported model ID: `deepseek-v4.1-flash`; quantization (`machine.quant`): **null**. A later on-disk inventory contains four 2.00 bpw IQ2_XXS shards: three byte hashes matching the original ISTA launcher inventory and one changed fourth-shard hash (details in `PROVENANCE.md`). This supports the existence of a modified model file set but **does not tie it to the server that served these requests**.
- Context capacity: **131,072 tokens**. Engine-reported KV setting: **`fp4`**, `kv_resident=131072`.
- **321 GPU expert slots**, **15,039 CPU experts**, 0 reported RAM experts. **Drafting: off** (no active MTP drafter recorded).
- Vision subsystem advertised and lazy-loaded; vision quality/speed were **not tested**.
- Server configured for **1 parallel slot**, despite the client's four concurrent requests.
- The GGUF filenames, sizes and hashes were collected **after** the run from the expected Huihui directory. The expected config allowlist contains `--max-context 131072`, `--vision`, `--reserve-mib 900` and port `8070`; this is not an attestation of the active process. Exact loaded executable/build, source weight revision, expert preparation/profile, patch identity, full memory/offload policies, environment flags, effective launch arguments and the DeepSeek reasoning budget remain **unverified for the measured run**.

## Post-run provenance

A separate read-only local collector captured four GGUF shards at **2026-11-05 17:23:42 +01:00**: **138,238,893,056 bytes (128.75 GiB)**. All filenames and SHA-256 checksums are preserved in [`provenance.json`](provenance.json). The first three hashes match the original ISTA base hashes declared by the operator's earlier launcher; the fourth shard has the same byte length but a **different** SHA-256, consistent with modifications. This does **not** independently prove the exact Huihui patch, nor that the earlier server loaded those files. See [`PROVENANCE.md`](PROVENANCE.md) for the complete comparison and Git snapshot.

## Method

All performance measurements use unmodified **`tools/strata_bench.py`** from the stated Strata checkout, Python 3.12, loopback HTTP, temperature **0**, **3 repeats** per test. Warm-up and a 1K calibration probe are excluded from the speed samples. The script uses unique prompt tags and reports server timings (`timings.prompt_per_second`, `timings.predicted_per_second`) together with client-measured time to first non-empty streamed token (TTFT). Model loading excluded. Fresh prompts had zero reported cached tokens.

Two sweeps ran **sequentially on the same already-loaded server**, **without an engine restart between sweeps**:

1. **Thinking forced OFF** — `--url http://127.0.0.1:8070 --runs 3 --long --clients 4`.
2. **Thinking allowed / not forcibly disabled** — same flags plus `--think`.

`--think` simply omits the explicit disable-thinking override. The uploaded report lacks generated text/reasoning traces; it does **not independently prove** that reasoning tokens were emitted or what reasoning effort/budget took effect. Sweeps should not be treated as a controlled same-cache-state A/B experiment.

Cases per sweep: **9 short chats** (3 prompts × 3 repeats, generation cap 100), **3 fresh 4K**, **3 fresh 16K**, **3 fresh 32K** prompts (generation cap 24), **3 multi-turn agent sessions** (six turns/session, per-turn generation cap 64), and **3 rounds** of **4 simultaneous clients** (generation cap 128 per client). The four requests queue on the server's **single slot**; this is not four-way parallel decoding.

### Measured performance

All rates below are medians with **minimum–maximum** in parentheses unless explicitly marked p50/p95. Prompt rates are server-measured prefill rates; decode is server-measured generation rate. TTFT is client-side.

| Metric | Thinking forced OFF | Thinking allowed (`--think`) |
| --- | ---: | ---: |
| Short chat decode | 15.30 (6.80–16.10) | 15.90 (13.70–16.40) |
| Short chat TTFT | 1.14 (0.94–1.32) | 1.84 (1.74–1.99) |
| Fresh 4K prompt tok/s | 390.5 (386.9–405.6) | 389.9 (387.3–393.8) |
| Fresh 16K prompt tok/s | 771.4 (767.9–774.2) | 749.9 (746.2–750.5) |
| Fresh 32K prompt tok/s | 819.6 (812.7–832.9) | 836.0 (834.8–840.2) |
| Agent cache hit (%) | 69.5 (69.4–69.5) | 69.4 (69.4–69.5) |
| Agent decode tok/s | 15.3 (13.1–16.6) | 15.8 (13.2–16.8) |
| 4-client aggregate tok/s | 12.53 (11.97–12.94) | 12.78 (12.27–13.05) |
| 4-client TTFT p50 (s) | 16.817 | 17.528 |
| 4-client TTFT p95 (s) | 32.800 | 33.147 |

Each of the 4K/16K/32K prompt tests has 3 successful requests per sweep and **0 reported reused tokens**. Short-chat decode has 9 successful requests per sweep. Reported aggregate concurrency rates use **4 queued clients**. No API errors or skipped cases were recorded in either speed sweep.

**Important variability:** OFF short-chat decode includes a **6.8 tok/s** request (other OFF chat requests range **14.1–16.1 tok/s**). This outlier is preserved in the original JSON; no post hoc filtering. The ON/OFF difference should not be claimed as a speed-up.

### Actual prompt, reused and generated tokens

Each cell is median **prompt / reused / generated** tokens, derived directly from per-request records (not the output cap).

| Case | OFF | Thinking allowed |
| --- | ---: | ---: |
| chat | 32 / 0 / 100 | 58 / 0 / 100 |
| 4K | 4,125 / 0 / 24 | 4,151 / 0 / 12 |
| 16K | 16,562 / 0 / 24 | 16,588 / 0 / 12 |
| 32K | 33,126 / 0 / 24 | 33,152 / 0 / 12 |

Notably, the long-prompt ON requests produced a median of **12 tokens** even though the **configured cap was 24**; do not confuse the cap with actual output. Short-chat requests reached the 100-token cap in the captured runs. This limits conclusions about sustained long-answer decode throughput.

### Agent cache and multi-client behavior

- Agent cache reuse: **69.5%** OFF and **69.4%** thinking allowed (tool-provided summary); CPU/GPU expert-cache residency is distinct from conversation prefix reuse.
- Agent follow-up decode: OFF and ON medians/ranges in the table above. Agent requests and reuse counts are in each original `results.json`.
- 4 clients: **1 server slot**; median aggregate throughput **12.53 tok/s** OFF and **12.78 tok/s** ON; zero reported request errors. This is primarily **queuing/latency under load**.

## Long-context correctness: needle-in-a-haystack

After the speed sweeps, **unmodified `tools/needle_bench.py`** was run with `--lengths 32k,64k,128k --depths 10,50,90` against the existing local server, at temperature 0 with thinking forcibly disabled. The haystack consists of deterministic repeated repository text (docs/source files from this checkout); reported prompt lengths differ from nominal labels, partly because the tool estimates tokens by characters.

| Nominal length | **Actual** input tokens | Needle depth (%) | Found | Median request wall time (s) |
| --- | ---: | --- | ---: | ---: |
| 32K | 30,001–30,002 | 10 / 50 / 90 | 3/3 | 46.2 |
| 64K | 61,225–61,227 | 10 / 50 / 90 | 3/3 | 86.0 |
| 128K | 117,794–117,795 | 10 / 50 / 90 | 3/3 | 168.5 |

**Overall: 9/9 needles found (100%)**, with 0 recorded errors, at actual prompts up to **117,795 tokens**; there is **no 131,072-token recall measurement**. These nine synthetic items assess retrieval for these inputs only, not general reasoning quality, long-context robustness on arbitrary inputs, or quantization fidelity. Each depth is measured once per nominal length; no confidence interval was estimated.

Full per-needle word, answer, actual tokens, depth and time are in [`long-context-recall/needles.json`](long-context-recall/needles.json).

## Evidence files

- [`reasoning-off/results.json`](reasoning-off/results.json) and [`reasoning-on/results.json`](reasoning-on/results.json): original Strata per-request speed records, unchanged from the upload.
- [`reasoning-off/official-tool-README.md`](reasoning-off/official-tool-README.md), [`reasoning-on/official-tool-README.md`](reasoning-on/official-tool-README.md): original script-generated README reports.
- [`long-context-recall/needles.json`](long-context-recall/needles.json): unmodified recall outputs.
- [`ENVIRONMENT-original.md`](ENVIRONMENT-original.md): original local environment summary provided by the operator.
- [`VERIFICATION.json`](VERIFICATION.json): independent recomputation/checks of case counts, output medians, known failures and the separate post-run provenance cross-check.
- [`provenance.json`](provenance.json): unchanged operator-supplied SHA-256 and allowlisted configuration snapshot taken after the benchmark.
- [`PROVENANCE.md`](PROVENANCE.md): original ISTA hash comparison, model identity caveats and later clean-tree observation.
- [`REPRODUCE.md`](REPRODUCE.md): exact benchmark commands and omissions.

## Correctness and limitations

- Community speed test only; **not** standard Qwen benchmark, AIME, GPQA, MMLU-Pro or `quant_quality.py` PPL/KL fidelity.
- Different models, quants, hardware, software, context and temperatures cannot be compared as a matched head-to-head without control runs. Do **not** attribute higher rates to Huihui abliterating weights or to an engine tuning change without a baseline on the **same machine** using the **same workload**.
- Both sweeps reused a resident server. Fresh model restart, background utilization, power/thermals while active and exact CPU memory utilization are missing. **Model file hashes are now documented from a later snapshot**, but the server-to-file linkage at benchmark time remains unverified.
- The original environment file reported a **modified** working tree; a later collection reports a **clean** tree at the same commit. No benchmark-time diff was preserved. The verified-on-disk shard hashes identify files after the run, but not the active pack or engine build; `engine_build` and quantization metadata were missing from the server response.
- ON suite did not capture reasoning content; thinking state is a request-level configuration label, not independently validated model behavior.
- Inference API exposed 1 parallel slot: concurrency is **queue throughput**, not 4-way parallel inference.
- Needle test uses code/docs corpus from the checked-out tree; because the tree was reported modified at benchmark time and no per-source hash snapshot was captured, byte-perfect corpus provenance is not established.
- No engine startup logs were collected because the benchmark attached to an already-running server. Later SHA-256 evidence is included, but the **active benchmark-time** GGUF revision, pack identity and sanitized launch configuration remain unconfirmed.

## Provenance and remaining uncertainties

The operator's earlier launcher declares upstream repository revisions for **`huihui-ai/Huihui-DeepSeek-V4.1-Flash-1017-abliterated-GGUF`** (`c59352db18acc847d9db6763df992d41b1d7d419`) and original **`ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF`** (`cb02422c900df489cea5833e8a92171224059b3c`), with a 40-tensor patch over IQ2_XXS. These revisions and patch details are **launcher-declared**, not independently attested by the benchmark server. After the run, the operator's collector verified existence of the expected config/model directory, gathered four complete GGUF checksums, and reported a clean checkout. **A post-run disk snapshot is not a loaded-process record.** Details, including the exact comparison to the original base hashes, are in [`PROVENANCE.md`](PROVENANCE.md). Remaining reproducibility gaps are listed in the package's `MISSING_PROVENANCE.md`.
