# Community benchmark on NVIDIA GeForce RTX 5090

Measured on 2026-10-25 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5090, 32 GiB VRAM
- CPU: AMD Ryzen 7 9800X3D 8-Core Processor (8 cores, 16 threads); installed RAM: 62.2 GiB; storage: not measured
- OS: Linux 7.3.0-5-cachyos; GPU driver: 615.78.08; CUDA: not read
- Strata: project 0.1.49.2, commit 7dd5cf18; engine 0.1.49.2
- Background workloads and power limits: background load not recorded. The card runs a custom LACT profile: a **400 W cap** (the card's minimum; the stock default is 575 W), locked 300-2500 MHz core, +600 MHz VF offset at p-state 0. It was in force for this run: that session's lactd started at 06:54:09 and applied its configuration throughout (27 "configuration applied" lines, the last at 09:54:55, and the run at 08:09 falls between them), `/etc/lact/config.yaml` is unchanged since 2026-10-19, and the card's owner confirms the 400 W cap. The wattage itself was not read at the time of the run: after the machine rebooted at 09:59:07 the same day, `nvidia-smi --query-gpu=power.limit` reported 400.00 W. These are a 400 W card's numbers, not a stock 5090's.

## Model and configuration

- Model and quantization: qwen3.8-flash-next-gyro-gyro-m
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack gyro-gyro-m --native Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf --ple-gguf Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 262144 --kv int8 --kv-resident 32768 --vision --vram-reserve-mib 700 --pcie-frac 0.35 --spec-min-p 0.5
```

Environment of the config: `STRATA_DEQUANT_WIDE=1`, `STRATA_GDN_CHUNKED=1`

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 181.1 (156.4-253.2) | 0.16 (0.14-0.17) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 5265.3 (4558.2-5272.9) | not measured | 0.78 (0.77-0.89) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 6318.3 (6141.2-6318.8) | not measured | 2.56 (2.56-2.63) |
| agent session, follow-up turns | 15442 (15422-15471) at the last turn | see below | up to 64 | 3 | see below | 190.2 (152.7-247.0) | 0.76 (0.43-1.17) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 70.5 % (70.4-70.5); earlier-turn tokens read again per session: 3 (3-72) (earlier turns that the cache did not hold; the 14603 (14602-14604) new tokens of the files are read once); first turn (nothing cached) first token: 0.32 s (0.32-0.38).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 1659.9 | 0.379 | 126.9 |
| 1 | 4976 | 621 | 4355 | 3790.3 | 1.165 | 190.2 |
| 2 | 6509 | 4969 | 1540 | 2643.8 | 0.597 | 174.1 |
| 3 | 9942 | 6573 | 3369 | 3531.8 | 0.97 | 193.4 |
| 4 | 12700 | 10006 | 2694 | 3250.5 | 0.845 | 196.4 |
| 5 | 15471 | 12754 | 2717 | 3495.9 | 0.792 | 183.1 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded, and the power limit in the section above was read by hand after this machine rebooted, not at the time of the run. Total time of the suite: 35 s.
