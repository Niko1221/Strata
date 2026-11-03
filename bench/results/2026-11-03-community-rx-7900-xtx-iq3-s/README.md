# Community benchmark on AMD Radeon RX 7900 XTX (iq3-s)

Measured on 2026-11-03 by raufaser. Stock engine 0.1.55, track default, one model configuration. Main limitation: single evening, one machine, warmed (not cold) measurements.

## Hardware and software

- GPU: AMD Radeon RX 7900 XTX (24.0 GiB dedicated); CPU: AMD Ryzen 9 7900X 12-Core Processor; RAM: 63.1 GiB
- Storage of the model files: ssd Seagate FireCuda 530 ZP2000GM30013 (nvme)
- OS: Microsoft Windows 11 Pro, version 10.0.26200; driver: gpu-driver 32.0.32015.12012; runtime: not measured
- Strata commit: `c62d5e3a15b40ea51c96aae9e8d36928965f53ab`; engine 0.1.55; release build
- GPU tuning (recorded variable, set in Adrenalin, not stock): max clock 3000 MHz, voltage 1000 mV (stock voltage 1150 mV)
- Background workloads: not measured

## Model and configuration

- Model: qwen3.8-flash-next-iq3_s; quantization IQ3_S; revision not measured
- `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` sha256 `4c1eb2ceb4915e11...`
- `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` sha256 `316b46f3a2dbd68c...`
- Track: **default** (0.1.55, setup defaults + 128k/int8/vision-default-on, RX 7900 XTX Windows)
- Context 131072 tokens; server capacity 3 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, reasoning_effort xhigh; thinking on, preserved

```text
strata launch --model IQ3_S --context 128k --kv int8 --yes --no-tui
strata bench run --url http://127.0.0.1:8080 --track default --track-notes "0.1.55, setup defaults + 128k/int8/vision-default-on, RX 7900 XTX Windows"
```

## Method

The frozen requests of `bench/prompts/standard-v1` (strata-qwen38-thinking-v1-draft.1) were sent over loopback HTTP, streaming, one at a time: 1 warm-up (excluded), 3 repetitions each of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first delta. No request was retried. Expert cache warm and evolving; the cache was not emptied between requests. These are not cold-model measurements.

## Results

| Case | Prompt tok/s median (min-max) | Decode tok/s median (min-max) | TTFT s median (min-max) | Total s median (min-max) |
| --- | --- | --- | --- | --- |
| short | 51.9 (50.1-56.9) n=3 | 61.7 (59.1-66.8) n=3 | 1.55 (1.41-1.58) n=3 | 8.2 (6.7-9.6) n=3 |
| prompt-4k | 839.5 (838.4-843.3) n=3 | 57.9 (55.8-58.8) n=3 | 4.91 (4.89-4.91) n=3 | 13.7 (13.6-14.1) n=3 |
| prompt-16k | 1073.8 (1071.6-1079.1) n=3 | 58.0 (55.6-61.1) n=3 | 15.30 (15.23-15.33) n=3 | 24.1 (23.7-24.4) n=3 |

Performance requests: 9, 6 reached the 512-token cap (reasoning and answer together; a capped request's total time is the time to the cap). Failed: 0. Per-request machine data: [`results.json`](results.json) (`summaries` for the medians above, `requests` for every request, `missing` for values the runner could not read).

## Correctness and limitations

Sanity checks: **12 of 12** (instruction 4/4, retrieval 4/4, tool calls 4/4). Output caps include reasoning. Untested: cold starts, other contexts/KV types, concurrent load, power draw, other clocks/voltages.
