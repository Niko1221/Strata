# Community benchmark: RTX 5090 Laptop GPU (24 GB), engine 0.1.43

Measured on 2026-10-11 (19:34–19:40 local time) by [wolffahrer](https://github.com/wolffahrer) on the same Windows 11
gaming laptop (Schenker XMG NEO 16, A25) as the earlier entries
[0.1.40.1](../2026-10-06-community-rtx-5090-laptop/README.md),
[0.1.40.2/0.1.40.3](../2026-10-08-community-rtx-5090-laptop-engine-0.1.40.3/README.md) and
[0.1.41](../2026-10-08-community-rtx-5090-laptop-engine-0.1.41/README.md). Same machine, model files, configuration,
scripts and request order; only the Strata version changed. 0.1.42 was not measured separately (0.1.43 contains it).

Median decode throughput with **0.1.43** was **140.5 tok/s at 4,096 prompt tokens, 141.5 tok/s at 32,768, and
131.6 tok/s at 128,000**; median prompt throughput 1,906.9, 2,871.2 and 2,767.5 tok/s. All nine speed requests
succeeded with zero reused tokens. On this machine that is within run-to-run variation of 0.1.41.

**Why the 0.1.42 decode gain does not show here.** The route tail skip is on (`STRATA_ROUTE_TAIL_SKIP=7`, the engine
log confirms it and counts skipped experts), but this laptop keeps 93.9–99.2% of the decode expert lookups in its
VRAM cache (11,739 slots, 15.11 GiB), so there is little tail to skip. That matches the release notes, which put the
gain on 12–16 GB cards that miss experts.

## Hardware and software

Unchanged from the earlier entries unless noted:

- NVIDIA GeForce RTX 5090 **Laptop** GPU, 24,463 MiB, 175 W limit (Dynamic Boost). Sampled draw median 160.6 W, peak
  sample 168.3 W; GPU temperature peaked at 84 °C. PCIe Gen 4 x8 under load.
- AMD Ryzen 9 9955HX3D, `--pool-workers 10`. 64 GB RAM (61.68 GiB usable). Samsung SSD 990 PRO 1TB.
- Windows 11 Home, build 26200. NVIDIA driver 610.88. CUDA toolkit 13.3.
- Strata **v0.1.43**, commit `7695859aa13afb248bdb4d36f9b2e95707db079d`. Source archive downloaded by commit
  (`/archive/<commit>.zip`, sha256 `f41eeb0f91d6ec4d8daf88629373048b7fc6c5c5f8e8a104295f5f9aa18259dc`), every file
  checked against the git tree, then built with the unmodified installer for CUDA architecture 120, GPU vision helper
  included ([BUILD.json](BUILD.json)). The installer suggested the same arguments as for 0.1.41, so the configuration
  was carried over unchanged.
- Background: a WSL2 VM with an idle local agent stack; the regular inference server was stopped. Not an isolated OS.

## Model and configuration

`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` **Q2_0**, context 131,072, INT8 KV with 32,768 resident cells, expert
cache `auto`, prefill `auto`, MTP `--spec 4 --spec-min-p 0.7`, `--pcie-frac 0`, `--mmap-experts`, GPU vision,
conversation cache 6 GiB / 2 slots, `reasoning_budget_tokens: 12000`, `reasoning_loop_recovery: "stop"`.
Configuration with placeholders: [strata-q2_0.json](strata-q2_0.json). Per-request timings: [engine.log](engine.log).

## Method

Same as the earlier entries: [benchmark.py](benchmark.py), the unchanged `tools/needle_bench.py` and
[coding_check.py](coding_check.py). One warm-up excluded; the server was healthy 17.1 s after start. Memory and GPU
state were sampled by [monitor.py](monitor.py) ([memory-summary.json](memory-summary.json): 97 samples over 301.6 s,
9 GPU samples unreadable and skipped; raw `telemetry.jsonl` left out).

```text
.venv\Scripts\python.exe serve\server.py --engine strata --config strata-q2_0.json --port 18096
.venv\Scripts\python.exe benchmark.py --root <strata> --pack <strata-data>\packs\q2_0 --url http://127.0.0.1:18096 --out results
.venv\Scripts\python.exe tools\needle_bench.py --url http://127.0.0.1:18096 --lengths 32k,128k --depths 10,50,90 --out needles.json
.venv\Scripts\python.exe coding_check.py --url http://127.0.0.1:18096 --out coding-check.json
```

## Results (0.1.43)

Median **[minimum–maximum]** of three runs; every request generated 256 tokens; no failed or cancelled requests.

| Prompt tokens | Reused | Prompt tok/s | Decode tok/s | TTFT seconds | Total seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 0 | 1,906.9 [1,555.9–1,959.7] | 140.5 [121.0–150.5] | 2.180 [2.106–2.662] | 4.076 [3.954–4.854] |
| 32,768 | 0 | 2,871.2 [2,862.9–2,908.7] | 141.5 [136.3–142.3] | 11.445 [11.296–11.478] | 13.242 [13.084–13.343] |
| 128,000 | 0 | 2,767.5 [2,758.7–2,795.0] | 131.6 [126.2–145.3] | 46.339 [45.895–46.502] | 48.252 [47.829–48.357] |

Raw records: [results.json](results.json); aggregates: [summary.json](summary.json). Peak GPU memory 23,213 MiB; host
RAM in use (system-wide, includes the WSL2 VM) 14.86 GiB before load, 55.56 GiB peak; page file 0.81–1.88 GiB.

**Across versions on this laptop** (median decode / prompt tok/s, same procedure, one run of three requests each):

| Prompt tokens | 0.1.40.1 | 0.1.40.3 | 0.1.41 | 0.1.43 |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 136.9 / 1,952.5 | 138.1 / 1,949.7 | 142.5 / 1,949.8 | 140.5 / 1,906.9 |
| 32,768 | 137.7 / 2,877.8 | 144.4 / 2,893.3 | 141.5 / 2,888.3 | 141.5 / 2,871.2 |
| 128,000 | 134.6 / 2,764.1 | 131.0 / 2,780.1 | 127.8 / 2,776.8 | 131.6 / 2,767.5 |

## Correctness and limitations

- **Long-context recall:** all six needles found (prompt lengths 32,804–32,805 and 124,840–124,842 tokens; two 128K
  cases reused 16,384 prompt tokens). See [needles.json](needles.json).
- **Coding check:** **10/10 tests passed**, 942 generated tokens, 8.6 s. See [coding-check.json](coding-check.json).
- **Own agent task suite** (private, 44 tasks x 3 runs, German prompts): 122 of 132 runs with 0.1.43, against 124
  (0.1.41) and 123 (0.1.40.3); the differences are single runs of the same tasks and within our run-to-run spread.
- **A note for anyone benchmarking with a small `max_tokens`:** with reasoning on and `max_tokens` below the thinking
  budget plus the answer reserve, 0.1.43 cuts the thinking at the reduced budget and continues with the answer (#984).
  The engine log then shows two timing lines for one request. Scripts that read only the last line will under-count
  that request; this report's speed table is unaffected (reasoning off).
- Not evaluated: long outputs, sampled decoding, thinking speed, vision, tool use, concurrency, sustained thermal load.

The author documents local AI on consumer hardware (in German) on the KI SOUVERÄN channels:
[YouTube](https://www.youtube.com/channel/UCE6Ch6g6Bo8v4ROpYDzOOxA) and [Telegram](https://t.me/lokale_ki).
