# Community benchmark: RTX 5090 Laptop GPU (24 GB), engine 0.1.44.1

Measured on 2026-10-16 (15:05–15:12 local time) by [wolffahrer](https://github.com/wolffahrer) on the same Windows 11
gaming laptop (Schenker XMG NEO 16, A25) as the earlier entries
[0.1.40.1](../2026-10-06-community-rtx-5090-laptop/README.md),
[0.1.40.2/0.1.40.3](../2026-10-08-community-rtx-5090-laptop-engine-0.1.40.3/README.md),
[0.1.41](../2026-10-08-community-rtx-5090-laptop-engine-0.1.41/README.md) and
[0.1.43](../2026-10-11-community-rtx-5090-laptop-engine-0.1.43/README.md). Same model files, scripts and request
order. Two things changed besides the Strata version, see "What changed on this machine" below. 0.1.44 was not measured
separately (0.1.44.1 contains it).

Median decode throughput with **0.1.44.1** was **128.4 tok/s at 4,096 prompt tokens, 137.9 tok/s at 32,768, and
124.6 tok/s at 128,000**; median prompt throughput 1,804.1, 2,745.6 and 2,660.9 tok/s. All nine speed requests
succeeded with zero reused tokens.

**Read the numbers with the laptop's power profile in mind.** They are 3–9% below the 0.1.43 entry from 2026-10-11,
but the engine is not the reason: a before/after run on the same afternoon (0.1.43 at 13:43, 0.1.44.1 at 13:55, same
configuration, same prompts, greedy, 1K–100K prompts) landed within ±2% for decode and prompt throughput. What did
change is the laptop's CPU/GPU power profile (below).

## What changed on this machine since the 0.1.43 entry

- **Power profile:** the vendor "Overboost" profile was switched from *Extreme* to *Normal* on 2026-10-13 (the CPU
  ran too hot in daily use). Sampled GPU power draw during this run: median 162.3 W, peak sample 219.1 W; GPU
  temperature peaked at 85 °C. The 0.1.43 entry was measured on *Extreme*.
- **Conversation cache:** 12 GiB / 8 slots instead of 6 GiB / 2 slots (`--conversation-cache-mib 12288
  --conversation-cache-slots 8`). It parks finished conversations in host RAM and does not take part in these
  single-conversation requests (zero reused tokens); it raises the host RAM figures below.

## Hardware and software

Unchanged from the earlier entries unless noted:

- NVIDIA GeForce RTX 5090 **Laptop** GPU, 24,463 MiB, 175 W limit (Dynamic Boost). PCIe Gen 4 x8 under load.
- AMD Ryzen 9 9955HX3D, `--pool-workers 10`. 64 GB RAM (61.68 GiB usable). Samsung SSD 990 PRO 1TB.
- Windows 11 Home, build 26200. NVIDIA driver 610.88. CUDA toolkit 13.3.
- Strata **v0.1.44.1**, commit `296b59d2f73c8f04a805df29c2fa97655adace78`. Source archive downloaded by commit
  (`/archive/<commit>.zip`, sha256 starting `11750730df62c841`), every file checked against the git tree, then built with
  the unmodified installer for CUDA architecture 120, GPU vision helper included ([BUILD.json](BUILD.json)). The
  installer's suggested arguments differ from this configuration only in `--spec-min-p` (0.5 suggested, 0.7 kept) and
  `--vram-reserve-mib` (700 suggested, 1000 kept), both carried over from the earlier entries.
- Background: a WSL2 VM with an idle local agent stack (its chat model was routed to a cloud API for the afternoon);
  the regular inference server was stopped. Not an isolated OS.

## Model and configuration

`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` **Q2_0**, context 131,072, INT8 KV with 32,768 resident cells, expert
cache `auto` (12,493 slots requested, shrunk to 11,685 slots / 15.04 GiB once written; decode lookups hit the VRAM
cache 67.7–99.4% of the time per request), prefill `auto`, MTP `--spec 4 --spec-min-p 0.7`, `--pcie-frac 0`,
`--mmap-experts`, GPU vision, conversation cache 12 GiB / 8 slots, `reasoning_budget_tokens: 12000`,
`reasoning_loop_recovery: "stop"`. Configuration with placeholders: [strata-q2_0.json](strata-q2_0.json).
Per-request timings: [engine.log](engine.log).

## Method

Same as the earlier entries: [benchmark.py](benchmark.py), the unchanged `tools/needle_bench.py` and
[coding_check.py](coding_check.py). One warm-up excluded; the server was healthy 21.1 s after start. Memory and GPU
state were sampled by [monitor.py](monitor.py) ([memory-summary.json](memory-summary.json): 190 samples over 415.5 s,
no unreadable GPU samples; raw `telemetry.jsonl` left out).

```text
.venv\Scripts\python.exe serve\server.py --engine strata --config strata-q2_0.json --port 18096
.venv\Scripts\python.exe benchmark.py --root <strata> --pack <strata-data>\packs\q2_0 --url http://127.0.0.1:18096 --out results
.venv\Scripts\python.exe tools\needle_bench.py --url http://127.0.0.1:18096 --lengths 32k,128k --depths 10,50,90 --out needles.json
.venv\Scripts\python.exe coding_check.py --url http://127.0.0.1:18096 --out coding-check.json
```

## Results (0.1.44.1)

Median **[minimum–maximum]** of three runs; every request generated 256 tokens; no failed or cancelled requests.

| Prompt tokens | Reused | Prompt tok/s | Decode tok/s | TTFT seconds | Total seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 0 | 1,804.1 [1,530.6–1,846.5] | 128.4 [114.4–135.3] | 2.290 [2.240–2.696] | 4.225 [4.171–5.145] |
| 32,768 | 0 | 2,745.6 [2,730.1–2,760.0] | 137.9 [128.1–139.1] | 11.986 [11.907–12.052] | 13.893 [13.813–13.899] |
| 128,000 | 0 | 2,660.9 [2,632.5–2,689.5] | 124.6 [119.2–124.8] | 48.274 [47.704–48.784] | 50.409 [49.846–50.827] |

Raw records: [results.json](results.json); aggregates: [summary.json](summary.json). Peak GPU memory 23,432 MiB; host
RAM in use (system-wide, includes the WSL2 VM and the 12 GiB conversation cache) 17.22 GiB before load, 59.07 GiB peak;
page file 1.02 GiB throughout.

**Across versions on this laptop** (median decode / prompt tok/s, same procedure, one run of three requests each;
0.1.44.1 on the *Normal* power profile, the others on *Extreme*):

| Prompt tokens | 0.1.40.1 | 0.1.40.3 | 0.1.41 | 0.1.43 | 0.1.44.1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 136.9 / 1,952.5 | 138.1 / 1,949.7 | 142.5 / 1,949.8 | 140.5 / 1,906.9 | 128.4 / 1,804.1 |
| 32,768 | 137.7 / 2,877.8 | 144.4 / 2,893.3 | 141.5 / 2,888.3 | 141.5 / 2,871.2 | 137.9 / 2,745.6 |
| 128,000 | 134.6 / 2,764.1 | 131.0 / 2,780.1 | 127.8 / 2,776.8 | 131.6 / 2,767.5 | 124.6 / 2,660.9 |

## Correctness and limitations

- **Long-context recall:** all six needles found (prompt lengths 32,778–32,779 and 125,010–125,012 tokens, 12.4–14.0 s
  and 42.3–47.3 s). See [needles.json](needles.json).
- **Coding check: 6/10 tests passed**, 500 generated tokens, 6.4 s. See [coding-check.json](coding-check.json). The
  Roman-numeral function handles the subtractive forms wrongly (IV → 5, IX → 10, MCMXCIV → 2105, CDXLIV → 555); the
  other six cases pass. 0.1.43 passed 10/10 on the same prompt (greedy). We treat a single greedy sample as one data
  point, not as evidence of a regression, because the larger agent suite below moved the other way; the check will be
  run twice on the next release.
- **Own agent task suite** (private, 44 tasks x 3 runs, German prompts, production configuration including the
  conversation cache): **126 of 132 runs** with 0.1.44.1, against 122 (0.1.43), 124 (0.1.41) and 123 (0.1.40.3); the
  best result so far on this machine, the deep multi-step set passed 24/24 for the first time. Same task fingerprints
  as the earlier runs.
- **Conversation cache (production configuration, 61,775-token conversation):** restoring a parked conversation after a
  foreign request took 0.7 s (61,928 of 61,956 tokens reused), the same as with 0.1.43 (0.8 s).
- Not evaluated: long outputs, sampled decoding, thinking speed, vision, tool use, concurrency, sustained thermal load.

The author documents local AI on consumer hardware (in German) on the KI SOUVERÄN channels:
[YouTube](https://www.youtube.com/channel/UCE6Ch6g6Bo8v4ROpYDzOOxA) and [Telegram](https://t.me/lokale_ki).
