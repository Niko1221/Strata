# Community benchmark: Quadro RTX 8000 48 GB, Xeon E5-2687W 0, Unsloth UD-Q4_K_XL, 131K context

Measured on 2026-10-11 by [aspineux](https://github.com/aspineux)

**What was tested:** fresh-prompt prompt and decode throughput at 4K / 32K / 128K prompt
tokens (three runs each, 256-token cap, greedy), long-context recall at 32K and 128K,
concurrency from 1 to 4 clients, a 75-minute / 708-request soak for drift and stability, and a
cold-engine restart with its load time and first-touch runs, on Strata v0.1.42 with the
Unsloth UD-Q4_K_XL pack.

**Main limitations, stated up front:**

1. **The measurement agent shared the engine.** The machine drives the agent that ran these
   scripts (`PI_PROVIDER=strata`, `PI_MODEL=qwen3.8-flash-next-ud-q4_k_xl`), so the client and
   the subject are the same server (`PI_PROVIDER=strata`,
   `PI_MODEL=qwen3.8-flash-next-unsloth-ud-q4_k_xl`). See [Method](#method) for how this was
   bounded; **0 of 9 sweep runs were flagged contaminated** by the built-in check.
2. **Warm expert cache.** Every run ran in one long-lived engine session that had already
   served ~20 requests. These are not cold-start numbers. Run 1 of each length is measurably
   slower than runs 2-3 (see [Cache warming](#cache-warming-is-visible-in-the-data)).
3. **The card is power-capped to 200 W** against a 260 W default (`power.limit` 200.00 W,
   `power.default_limit` and `power.max_limit` 260.00 W). Compare rows in
   [COMMUNITY.md](../COMMUNITY.md) with that in mind.
4. **Not tested:** a **cold boot** (power-on with an empty OS file cache - what is measured
   here is a cold *engine* with the model files still in RAM), `--batch` slots, multi-GPU,
   `--calibrate`, vision requests, low-RAM mode, contexts beyond 131,072, and any quality
   benchmark (HumanEval or similar). Recall here is needle-in-a-haystack only.
   The long soak **is** done - see [Long soak](#long-soak-708-requests-in-75-minutes).

## Hardware and software

| | |
| --- | --- |
| GPU | NVIDIA Quadro RTX 8000, 49,152 MiB (Turing, sm_75), VBIOS 90.02.4A.00.01 |
| GPU clocks | 1905 MHz under load, 2100 MHz max reported |
| GPU power | **limit 200 W, default/max 260 W**; observed peak 198.7-224.3 W, mean 87-98% utilisation |
| PCIe | gen3 x16 (current = max); engine's startup probe: **11.5 GB/s host-to-device**, best of 4 |
| CPU | Intel Xeon E5-2687W 0 @ 3.10 GHz, 8 physical cores / 16 logical, governor `schedutil` |
| CPU ISA | **no AVX2, no AVX-512** (`gcc -march=native` resolves to `sandybridge`); engine `isa_floor: "avx"` |
| RAM | 125.72 GiB total, 8 GiB swap, no hugepages; rated speed not measured (needs root) |
| OS / kernel | Ubuntu 24.04.4 LTS, kernel 6.8.0-146-generic |
| Driver / CUDA | driver 580.178.04; engine built against `/usr/local/cuda-12.2`, `nvcc` on PATH reports 12.0 |
| Strata | tag `v0.1.42`, commit `61b3fb5d`, engine 0.1.42, **source build** (`engine/BUILD.json`: `source: local`, `archs: [75]`, `src: 1e0c97588848ea0f`) |
| Background load | the Strata server itself, and the agent driving the measurement; no other GPU processes |

Full machine detail: [`environment.json`](environment.json). Engine-reported figures:
[`metrics-snapshot.json`](metrics-snapshot.json), [`props.json`](props.json).

## Model and configuration

- Model: Unsloth **Qwen3.8-Flash-Next UD-Q4_K_XL**, 4 GGUF shards
  (`Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf` … `-00004-`), ~111 GB of shards
- Native-experts pack `packs/unsloth-ud-q4_k_xl` (`native_experts.txt`, `dense.bin`,
  `compat-bf16.json`), expert profile `data/expert-profile.bin`, MTP draft pack `packs/mtp/rt`
- Vision encoder configured (`mmproj-Qwen3.8-Flash-Next-BF16.gguf`, `strata-vision`, GPU) but
  **no image request was sent in this report**
- Context 131,072; KV `int8` with `--kv-resident 32768`; `--resident-budget-gib 71`;
  expert cache `auto`; `--prefill auto`; `--spec 4`; `--pcie-frac 0.32`; `--spec-min-p 0.70`;
  `--vram-reserve-mib 700`; 7 pool workers + host thread on core 0

```text
engine/strata --serve --pack /opt/models/gguf/packs/unsloth-ud-q4_k_xl \
  --native /opt/models/gguf/models/unsloth-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --expert-profile data/expert-profile.bin --expert-cache auto --prefill auto --spec 4 \
  --mtp /opt/models/gguf/mtp/rt --max-context 131072 --kv int8 --kv-resident 32768 \
  --resident-budget-gib 71 --vision --vram-reserve-mib 700 --pcie-frac 0.32 --spec-min-p 0.70
```

Exact file: [`run-config.json`](run-config.json); launcher: [`run-script.sh`](run-script.sh).
No API key is set and the server binds `127.0.0.1:8080` only.

**What the engine said about its own memory at startup** ([`engine-session.log`](engine-session.log)):

```text
expert cache auto: 39.22 GiB free, 700 MiB reserved (+281 MiB for the draft head) -> 10286 slots
expert cache 13099 slots, 38.26 GiB of VRAM
FileExpertSource: mapped pinned cache complement ready: resident 33.47 GiB, pinned 33.47 GiB in 37 s
strata serve: 335 MiB of VRAM free with everything loaded
```

## Method

[`benchmark.py`](benchmark.py) sends non-streaming greedy (`temperature: 0`) requests with a
fixed `max_tokens: 256` and reads **prompt and decode throughput from the server's own
`timings` block** — never by dividing generated tokens by total request time.

- **Prompts are fresh, not reused.** Each prompt is built from a 20-sentence varied corpus and
  carries a nonce at the head, a marker in the middle and a tag at the end, so the conversation
  prefix cannot be reused. The engine confirms it: **`reused 0` on all nine runs**, and
  `usage.prompt_tokens_details.cached_tokens` is 0.
- **Actual token counts** are recorded per run and the characters-per-token ratio is re-fitted
  after each run (`ratio.json`), so the ladder lands near its target.
- **Timing boundaries.** `prompt_tps` and `gen_tps` are the engine's. `ttft_s_proxy` is
  `wall - predicted_ms` measured at the client of a **non-streaming** request, so it includes
  the HTTP round trip and is a proxy, not a streamed first token. `total_s` is client wall time.
- **GPU sampling** every 0.5 s in a side thread (`nvidia-smi`): peak VRAM, mean/max utilisation,
  peak power and temperature.

**Handling the agent sharing the engine.** The agent that ran these scripts is served by the
same server, so its own chat requests compete for the one slot. Two controls:

1. Every measurement ran in a **blocking foreground call**, one at a time, never in parallel
   and never in the background — while the script runs, the agent is waiting and generates no
   tokens.
2. Each run is checked afterwards: the engine must have been **idle before** the request
   (`/status` `busy` and `queued` both false), and the newest record in `/metrics` must be this
   request with a matching `prompt_total`. Otherwise the run is written with
   `"contaminated": true` and excluded from the medians.

**Result: 0 of 9 runs flagged.** Still treat this as a single-tenant measurement made by a
process that is itself a tenant, and re-run on a machine where the client is elsewhere before
quoting the numbers as clean.

**State between runs.** No restart, no cache reset, 5-8 s idle between runs. The expert cache
carried over from the agent's own conversation, so the hit rates below are a *warm* machine.
VRAM is preallocated by the expert cache, so `vram_peak_mib` (48,180 MiB on every run) is a
startup snapshot, not an inference peak.

## Results

All runs clean, all finished `stop`, all generated the full 256 tokens.

| Configuration | Actual prompt tokens | Reused | Generated | Runs | Prompt tok/s (median, range) | Decode tok/s (median, range) | TTFT proxy s (median) | Total s (median) |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| 4K fresh prompt | 3,452 / 3,977 / 4,125 | 0 | 256 | 3 | **728.4** (683.5-744.6) | **61.5** (47.4-61.5) | 5.47 | 9.72 |
| 32K fresh prompt | 31,808 / 32,955 / 32,621 | 0 | 256 | 3 | **1,012.8** (962.5-1,025.1) | **53.1** (37.7-58.1) | 31.88 | 38.23 |
| 128K fresh prompt | 127,823 / 128,167 / 127,817 | 0 | 256 | 3 | **896.5** (893.9-896.8) | **53.2** (44.8-57.4) | 143.10 | 147.92 |

Per-run data: [`runs.jsonl`](runs.jsonl) (one JSON object per run; `prompt_*` and `gen_*` are
the engine's `timings`, `engine_*` come from the server's `/metrics` request record, `gpu` is
the sampler's peak over that run) and [`results.json`](results.json) (aggregates).

Draft acceptance was high throughout: 89.0% median at 4K, 87.7% at 32K, 86.4% at 128K.
GPU utilisation rose with context length: 87-88% mean at 4K, 93-95% at 32K, 98% at 128K.
Peak temperature 69-71 °C.

### Cache warming is visible in the data

Run 1 of every length is slower than runs 2-3, and its expert-cache hit rate is lower:

| Length | Run 1 decode tok/s (hit rate) | Run 2 | Run 3 | Warm-only median (runs 2-3) |
| --- | --- | --- | --- | --- |
| 4K | 47.4 (96.2%) | 61.5 (98.5%) | 61.5 (98.9%) | 61.5 |
| 32K | 37.7 (92.6%) | 53.1 (97.9%) | 58.1 (98.4%) | 55.6 |
| 128K | 44.8 (96.2%) | 53.2 (98.9%) | 57.4 (98.9%) | 55.3 |

The prompt was byte-fresh each time, so this is the **expert** cache adapting to the new
prompt's routing, not prefix reuse. PCIe reads over the same runs were small
(`pcie_share` 0.00-0.60% of routed experts), which fits: the resident complement already holds
most of what this workload routes to.

### Concurrency

[`tools/serve_load.py`](../../../tools/serve_load.py), streamed, greedy, 256-token cap, fresh
nonce per client and round, 3 rounds per client count. Raw: [`concurrency.json`](concurrency.json).

| Clients | Total tok/s | Per-request tok/s | First token p50 / p95 s | Latency p50 / p95 s | Errors |
| ---: | ---: | ---: | --- | --- | ---: |
| 1 | 45.0 | 51.5 | 0.72 / 1.10 | 5.69 / 8.00 | 0 |
| 2 | 52.7 | 58.5 | 3.08 / 5.96 | 7.57 / 10.28 | 0 |
| 4 | 50.8 | 58.9 | 7.99 / 15.84 | 12.41 / 20.29 | 0 |

Total throughput is flat from 1 to 4 clients while first-token latency grows about 11x: this
server runs **one slot** (no `--batch`), so extra clients queue rather than decode in parallel.
That is the expected shape, recorded here so nobody reads the flat line as a scaling result.

### Recall

[`tools/needle_bench.py`](../../../tools/needle_bench.py), code word at 10 / 50 / 90% depth.
Raw: [`needles-32k.json`](needles-32k.json), [`needles-128k.json`](needles-128k.json).

| Length | Actual prompt tokens | Result |
| --- | ---: | --- |
| 32k | 32,804 | **3 of 3 found** (34 s each) |
| 128k | 125,536 | **3 of 3 found** (143, 143, 128 s) |

No misses, no errors, no skipped cases. This measures recall on these nine inputs; it says
nothing about answer quality generally.

### Long soak: 708 requests in 75 minutes

This is the answer to the open row in
[TEST_REQUESTS.md](../../../docs/TEST_REQUESTS.md) - *"a long soak (an hour or more of
back-to-back requests) on a split or resident setup"*. This is a resident setup:
`--resident-budget-gib 71`, `--kv-resident 32768`, and at startup
`mapped pinned cache complement ready: resident 33.47 GiB, pinned 33.47 GiB in 37 s`.

[`soak.py`](soak.py) ran back-to-back greedy requests from 14:08:21 to 15:23:15 UTC
(74.9 min, 9.5 requests/min), each a fresh ~1,329-token prompt with a per-request nonce and a
192-token cap. Raw: [`soak.jsonl`](soak.jsonl) (one object per request, hardware sampled every
30 s), summary: [`soak.out`](soak.out).

| Measure | First decile | Median | Last decile | Quarters 1 / 2 / 3 / 4 |
| --- | ---: | ---: | ---: | --- |
| Decode tok/s | 73.95 | **74.2** | 75.0 | 73.9 / 74.3 / 74.4 / 74.2 |
| Prompt tok/s | 341.85 | **342.1** | 342.3 | 342.1 / 342.0 / 342.1 / 342.2 |

| Watch | Result |
| --- | --- |
| Requests | **708 sent, 708 succeeded, 0 failures, 0 timeouts** |
| Finish reasons | `length` 491 (the 192-token cap), `stop` 217 (answer ended early, 121-191 tokens) |
| Request latency | median 6.4 s, p95 6.6 s, worst 7.4 s - and the worst was **request 1**, not a late one |
| RAM used | 42.93 - 42.99 GiB over the whole run (**+0.06 GiB, no leak**) |
| VRAM used / free | 48,180 MiB used and **335 MiB free at every one of the 142 samples** - no erosion |
| GPU | 97.9% mean utilisation, peak 205.6 W, peak 69 °C |
| Conversation cache | 4 slots, 0 MiB used, unchanged |

**No drift.** Decode moved +1.4% from the first decile to the last and prompt +0.1%, both
within run-to-run noise; the slowest request of the 708 was the very first one (44.7 tok/s,
cold expert cache) and the run's floor afterwards was 61.7 tok/s. Nothing paged, nothing grew,
the engine's own free-VRAM figure never moved off 335 MiB.

**These soak numbers are not comparable to the sweep table above.** The soak re-sends nearly
identical text (only the nonce differs) at ~1.3K prompt tokens, so the expert cache locks onto
one routing pattern and the KV stays small; the sweep sends byte-fresh varied prompts at
4K-128K. That is why soak decode reads 74.2 tok/s against the sweep's 53-61, and soak prompt
342 tok/s against the sweep's 728-1,013 - short, repetitive prompts are dominated by fixed
overhead. The soak measures *stability over time*, not peak throughput.

### Cold engine restart: 65.7 s to load, first-touch runs after it

[`cold_start.py`](cold_start.py) restarts the **engine process only** - `POST /unload` ends it
(and the image encoder), `POST /load` starts a fresh one and returns once it is ready - so the
Python server and its endpoint never change. Proof it really restarted: the engine PID went
`27516` -> `106524`, and GPU memory went **48,180 MiB -> 1 MiB** between the two calls.
Raw: [`cold/cold-start.json`](cold/cold-start.json), [`cold/engine-cold-session.log`](cold/engine-cold-session.log),
[`cold/runs.jsonl`](cold/runs.jsonl).

| | |
| --- | --- |
| `POST /unload` | **6.79 s**, GPU given back to 1 MiB used |
| `POST /load` | **65.7 s** wall, blocking until ready |
| Weights | 1,416 MiB of pack weights in 0.3 s (303 canonical tensors skipped, served natively) |
| Expert cache | 13,099 slots, 38.26 GiB of VRAM, `R4 hit path ON` |
| Resident complement | `resident 33.47 GiB, pinned 33.47 GiB in 37 s; page-locked and mapped` - **37 s of the 65.7 s** |
| After load | 48,064 MiB used, `335 MiB of VRAM free with everything loaded` (same as the long session) |

First touch after that restart, one run per length, all `reused 0`, none contaminated:

| Length | Prompt tokens | Prompt tok/s | Decode tok/s | Expert hit | Draft accept | Wall s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ~4K | 2,986 | 612.2 | 49.1 | 97.1% | 76.6% | 10.11 |
| ~32K | 30,770 | 1,019.0 | 59.6 | 98.9% | 90.7% | 34.55 |
| ~128K | 128,270 | 899.1 | 59.5 | 99.3% | 84.3% | 147.15 |

**This is a cold engine, not a cold boot.** The 111 GB of shards stay in the OS page cache
across an unload, so the file reads are warm; the 65.7 s is dominated by re-creating the
page-locked, mapped complement (37 s) and re-establishing the cache. A reboot would be a larger,
different number and is **not** measured here.

**One comparison to take carefully.** The cold 4K prompt is 2,986 tokens against the warm 4K
runs' 3,452-4,125 (the cold arm re-fitted its characters-per-token ratio from scratch), so its
612 tok/s is not the same measurement as the warm 728 tok/s. At 32K and 128K the prompts are
within 3% and 0.1%, and there the cold engine is **not** slower: 1,019.0 and 899.1 prompt tok/s
against warm medians of 1,012.8 and 896.5.

## Observations worth a maintainer's attention

- **A freshly restarted engine beat the first run inside the long session.** First touch after
  the cold restart hit 59.6 and 59.5 decode tok/s at 32K and 128K with 98.9% and 99.3% expert
  hit rates; the first run of each length in the long-running session managed 37.7 and 44.8
  tok/s at 92.6% and 96.2% hit. Both are first-touch-for-that-prompt runs, so the difference
  points at the expert cache settling differently after a clean profile-driven start than after
  ~1,400 mixed requests. **n = 1 per cell, and prompt lengths differ slightly** - it is a
  question, not a finding: does the complement degrade over a long session, and would a periodic
  re-place help? Not answered here.
- **`335 MiB of VRAM free with everything loaded`** on this config (`--expert-cache auto`,
  `--vram-reserve-mib 700`). [DETAILS.md](../../../docs/DETAILS.md) reports that an expert cache
  leaving no free VRAM cost 7x decode on two other cards, and names this log line as the
  reliable tell. Decode here is healthy (53-61 tok/s on the sweep, 74.2 tok/s steady over a
  75-minute soak) and hit rates are 96-99%, and the free figure **never moved during the soak**,
  so this is **not a reported degradation** - but 335 MiB is inside the margin the docs warn
  about, and a `--vram-reserve-mib` sweep on this box would say whether the auto sizer is
  leaving too little. Not measured here.
- **No AVX2 at all.** The CPU expert pool on this machine runs the AVX1 path
  (`isa_floor: "avx"`, `-march=sandybridge`). Most no-AVX-512 community reports are still AVX2
  parts; this one is a step older.
- **Power cap 200 W vs 260 W default**, yet instantaneous draw peaked at 210-224 W. Worth
  knowing when comparing against uncapped Turing cards.
- **Prompt throughput peaks at 32K, not 128K** (1,012.8 vs 896.5 tok/s). The 128K runs are
  extremely tight (893.9-896.8, 0.3% spread), so this looks real rather than noise.

## Reproducing

```bash
# from the repository root, against a running server on 127.0.0.1:8080
python3 bench/results/2026-10-11-community-quadro-rtx-8000-ud-q4kxl/collect_env.py
python3 bench/results/2026-10-11-community-quadro-rtx-8000-ud-q4kxl/benchmark.py --lengths 4k    --runs 3
python3 bench/results/2026-10-11-community-quadro-rtx-8000-ud-q4kxl/benchmark.py --lengths 32k   --runs 3
python3 bench/results/2026-10-11-community-quadro-rtx-8000-ud-q4kxl/benchmark.py --lengths 128000 --runs 3
python3 bench/results/2026-10-11-community-quadro-rtx-8000-ud-q4kxl/benchmark.py --summary
python3 tools/needle_bench.py --url http://127.0.0.1:8080 --lengths 32k,128k --depths 10,50,90
python3 tools/serve_load.py http://127.0.0.1:8080 --clients 1,2,4 --rounds 3 --max-tokens 256
python3 bench/results/2026-10-11-community-quadro-rtx-8000-ud-q4kxl/cold_start.py \
  --out bench/results/2026-10-11-community-quadro-rtx-8000-ud-q4kxl/cold
python3 bench/results/2026-10-11-community-quadro-rtx-8000-ud-q4kxl/soak.py --minutes 75 \
  --out bench/results/2026-10-11-community-quadro-rtx-8000-ud-q4kxl/soak.jsonl
```

Run the soak with nothing else touching the engine, including any chat page or agent served by
it: its per-request timings are exactly what a stray request would move.

`--lengths 128000` is written without the `k` on purpose: `128k` would mean 131,072 tokens,
which does not fit a 131,072-token context with room to answer.

## Not tested

A **cold boot** (the cold arm here restarts the engine with the model files still in the OS file
cache, which is the cheaper of the two numbers); `--batch` slots and true
parallel decode; multi-GPU and `--layer-split`; `--calibrate`; vision requests; low-RAM mode;
contexts beyond 131,072; quality benchmarks; and any comparison against another engine build or
another server.

## Files

| File | Contents |
| --- | --- |
| `benchmark.py` | the sweep, with the contamination check |
| `collect_env.py` | hardware, driver, PCIe, CPU flags, build provenance |
| `soak.py` | long-soak harness |
| `soak.jsonl` / `soak.out` | 708-request soak: per-request records and the run summary |
| `cold_start.py` | engine-only restart, load timing, first-touch runs |
| `cold/` | `cold-start.json`, `runs.jsonl`, `results.json`, `engine-cold-session.log` |
| `runs.jsonl` / `results.json` | per-run and aggregate sweep data |
| `concurrency.json` | `serve_load` raw output |
| `needles-32k.json` / `needles-128k.json` | recall raw output |
| `environment.json` | machine and build provenance |
| `metrics-snapshot.json` / `props.json` | engine-reported state and model properties |
| `engine-session.log` | the measured engine session, from its start line |
| `run-config.json` / `run-script.sh` | exact launch configuration |
