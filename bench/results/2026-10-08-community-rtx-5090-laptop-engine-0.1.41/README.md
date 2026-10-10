# Community benchmark: RTX 5090 Laptop GPU (24 GB), engine 0.1.41

Measured on 2026-10-08 (19:54–20:01 local time) by [wolffahrer](https://github.com/wolffahrer) on the same Windows 11
gaming laptop (Schenker XMG NEO 16, A25) as
[2026-10-06-community-rtx-5090-laptop](../2026-10-06-community-rtx-5090-laptop/README.md) (0.1.40.1, #1263) and
[2026-10-08-community-rtx-5090-laptop-engine-0.1.40.3](../2026-10-08-community-rtx-5090-laptop-engine-0.1.40.3/README.md)
(0.1.40.2 and 0.1.40.3, #1462). Same machine, model files, configuration, scripts and request order. Only the Strata
version changed.

Median decode throughput with **0.1.41** was **142.5 tok/s at 4,096 prompt tokens, 141.5 tok/s at 32,768, and
127.8 tok/s at 128,000**. Median prompt throughput was 1,949.8, 2,888.3 and 2,776.8 tok/s. All nine speed requests
succeeded with zero reused tokens. On this machine that is within run-to-run variation of 0.1.40.3. That matches the
release notes: the large 0.1.41 gains are for multi-GPU setups and Windows PCs with little RAM, and this laptop is a
single GPU with 64 GB.

## Hardware and software

Unchanged from the earlier entries unless noted:

- NVIDIA GeForce RTX 5090 **Laptop** GPU, 24,463 MiB, 175 W limit (Dynamic Boost). Sampled draw during this run
  median 163.4 W, one peak sample 212.9 W; GPU temperature peaked at 84 °C. PCIe Gen 4 x8.
- AMD Ryzen 9 9955HX3D, `--pool-workers 10`. 64 GB RAM (61.68 GiB usable). Samsung SSD 990 PRO 1TB.
- Windows 11 Home, build 26200. NVIDIA driver 610.88. CUDA toolkit 13.3.
- Strata **v0.1.41**, tag commit `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`, built from the release source archive
  (sha256 `e5e9a45ed42536c792157654b4d98309565f2d3797d1819c5cebe61c4d94d611`) with the unmodified installer for CUDA
  architecture 120, GPU vision helper included. See [BUILD.json](BUILD.json). The installer suggested the same
  arguments as for 0.1.40.3, so the configuration was carried over unchanged.
- Background: a WSL2 VM with an idle local agent stack; the regular inference server was stopped. Not an isolated OS.

## Model and configuration

`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` **Q2_0**, context 131,072, INT8 KV with 32,768 resident cells, expert
cache `auto` (**11,739 slots, 15.11 GiB**; 0.1.40.3 had 11,742), 497 MiB VRAM free with everything loaded, prefill
`auto` (borrows 2,082 slots), MTP `--spec 4 --spec-min-p 0.7`, `--pcie-frac 0`, `--mmap-experts`, GPU vision,
conversation cache 6 GiB / 2 slots, `reasoning_budget_tokens: 12000`, `reasoning_loop_recovery: "stop"`.
Configuration with placeholders: [strata-q2_0.json](strata-q2_0.json). Timings per request: [engine.log](engine.log).

## Method

Same as the earlier entries: [benchmark.py](benchmark.py), the unchanged `tools/needle_bench.py` and
[coding_check.py](coding_check.py). One warm-up excluded; the server was healthy 11.5 s after start. Memory and GPU
state sampled once per second by [monitor.py](monitor.py), summarised in [memory-summary.json](memory-summary.json)
(289 samples over 381.0 s; raw `telemetry.jsonl` left out).

```text
.venv\Scripts\python.exe serve\server.py --engine strata --config strata-q2_0.json --port 18096
.venv\Scripts\python.exe benchmark.py --root <strata> --pack <strata-data>\packs\q2_0 --url http://127.0.0.1:18096 --out results
.venv\Scripts\python.exe tools\needle_bench.py --url http://127.0.0.1:18096 --lengths 32k,128k --depths 10,50,90 --out needles.json
.venv\Scripts\python.exe coding_check.py --url http://127.0.0.1:18096 --out coding-check.json
```

## Results (0.1.41)

Median **[minimum–maximum]** of three runs; every request generated 256 tokens; no failed or cancelled requests.

| Prompt tokens | Reused | Prompt tok/s | Decode tok/s | TTFT seconds | Total seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 0 | 1,949.8 [1,798.7–1,956.3] | 142.5 [117.8–148.6] | 2.133 [2.112–2.301] | 3.962 [3.921–4.461] |
| 32,768 | 0 | 2,888.3 [2,887.6–2,909.6] | 141.5 [141.1–149.5] | 11.376 [11.297–11.385] | 13.095 [13.086–13.180] |
| 128,000 | 0 | 2,776.8 [2,766.3–2,802.2] | 127.8 [123.9–142.0] | 46.187 [45.760–46.364] | 48.154 [47.888–48.240] |

Raw records: [results.json](results.json); aggregates: [summary.json](summary.json). Decode expert-cache hit rates
92.1–98.5%. Peak GPU memory 23,565 MiB (0.1.40.3: 23,221 MiB); host RAM in use (system-wide, includes the WSL2 VM)
12.38 GiB before load, 54.51 GiB peak; page file 0.12–0.14 GiB.

**Across versions on this laptop** (median decode / prompt tok/s, same procedure, one run of three requests each):

| Prompt tokens | 0.1.40.1 | 0.1.40.2 | 0.1.40.3 | 0.1.41 |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 136.9 / 1,952.5 | 139.3 / 1,960.0 | 138.1 / 1,949.7 | 142.5 / 1,949.8 |
| 32,768 | 137.7 / 2,877.8 | 141.6 / 2,890.0 | 144.4 / 2,893.3 | 141.5 / 2,888.3 |
| 128,000 | 134.6 / 2,764.1 | 137.2 / 2,764.1 | 131.0 / 2,780.1 | 127.8 / 2,776.8 |

The decode medians move by a few percent in both directions between versions, inside the single-run min–max ranges;
prompt throughput is flat. A separate before/after run with our own house benchmark (13 fresh prompts from 1K to 113K
tokens, 0.1.40.3 measured directly before 0.1.41 on the same evening) gave the same picture: mean decode +0.4%, prompt
throughput above 30K tokens +0.5%.

## Correctness and limitations

- **Long-context recall:** all six needles found (prompt lengths 33,275–33,276 and 126,551–126,553 tokens; one 32K and
  two 128K cases reused 16,384 prompt tokens). See [needles.json](needles.json).
- **Coding check:** **10/10 tests passed**, 827 generated tokens, 7.8 s. See [coding-check.json](coding-check.json).
- **Own agent task suite** (multi-step tool use under a local agent, German prompts, 44 tasks x 3 runs, private):
  124 of 132 runs passed with 0.1.41, against 121 (0.1.40.1), 122 (0.1.40.2) and 123 (0.1.40.3).
- Not evaluated: long outputs, sampled decoding, thinking speed, vision, tool use, concurrency, sustained thermal load.

The author documents local AI on consumer hardware (in German) on the KI SOUVERÄN channels:
[YouTube](https://www.youtube.com/channel/UCE6Ch6g6Bo8v4ROpYDzOOxA) and [Telegram](https://t.me/lokale_ki).
