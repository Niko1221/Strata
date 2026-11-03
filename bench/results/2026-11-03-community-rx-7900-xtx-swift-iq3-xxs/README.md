# Community benchmark on AMD Radeon RX 7900 XTX (swift-iq3-xxs)

Measured on 2026-11-03 by raufaser. Stock engine 0.1.55, track default, one model configuration. Main limitation: single evening, one machine, warmed (not cold) measurements.

## Hardware and software

- GPU: AMD Radeon RX 7900 XTX (24.0 GiB dedicated); CPU: AMD Ryzen 9 7900X 12-Core Processor; RAM: 63.1 GiB
- Storage of the model files: ssd Seagate FireCuda 530 ZP2000GM30013 (nvme)
- OS: Microsoft Windows 11 Pro, version 10.0.26200; driver: gpu-driver 32.0.32015.12012; runtime: not measured
- Strata commit: `c62d5e3a15b40ea51c96aae9e8d36928965f53ab`; engine 0.1.55; release build
- GPU tuning (recorded variable, set in Adrenalin, not stock): max clock 3000 MHz, voltage 1000 mV (stock voltage 1150 mV)
- Background workloads: not measured

## Model and configuration

- Model: swift-1.5-iq3_xxs; quantization IQ3_XXS; revision not measured
- `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` sha256 `3bddaa667c750f63...`
- `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` sha256 `b0b15f782af71eb4...`
- Track: **default** (0.1.55, setup defaults + 128k/int8/vision-off, RX 7900 XTX Windows)
- Context 131072 tokens; server capacity 3 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, reasoning_effort xhigh; thinking on, preserved

```text
strata launch --family swift --model IQ3_XXS --context 128k --kv int8 --vision off --yes --no-tui
strata bench run --url http://127.0.0.1:8080 --track default --track-notes "0.1.55, setup defaults + 128k/int8/vision-off, RX 7900 XTX Windows"
```

## Method

The frozen requests of `bench/prompts/standard-v1` (strata-qwen38-thinking-v1-draft.1) were sent over loopback HTTP, streaming, one at a time: 1 warm-up (excluded), 3 repetitions each of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first delta. No request was retried. Expert cache warm and evolving; the cache was not emptied between requests. These are not cold-model measurements.

## Results

| Case | Prompt tok/s median (min-max) | Decode tok/s median (min-max) | TTFT s median (min-max) | Total s median (min-max) |
| --- | --- | --- | --- | --- |
| short | 72.5 (70.1-91.0) n=3 | 83.7 (82.7-92.6) n=3 | 1.08 (0.88-1.14) n=3 | 5.6 (5.4-6.4) n=3 |
| prompt-4k | 1071.4 (1065.2-1071.9) n=3 | 83.1 (74.5-84.5) n=3 | 3.86 (3.84-3.88) n=3 | 10.0 (9.9-10.7) n=3 |
| prompt-16k | 1335.7 (1330.0-1337.1) n=3 | 85.5 (81.8-87.3) n=3 | 12.31 (12.30-12.36) n=3 | 18.3 (18.1-18.5) n=3 |

Performance requests: 9, 7 reached the 512-token cap (reasoning and answer together; a capped request's total time is the time to the cap). Failed: 0. Per-request machine data: [`results.json`](results.json) (`summaries` for the medians above, `requests` for every request, `missing` for values the runner could not read).

## Correctness and limitations

Sanity checks: **12 of 12** (instruction 4/4, retrieval 4/4, tool calls 4/4). Output caps include reasoning. Untested: cold starts, other contexts/KV types, concurrent load, power draw, other clocks/voltages.
