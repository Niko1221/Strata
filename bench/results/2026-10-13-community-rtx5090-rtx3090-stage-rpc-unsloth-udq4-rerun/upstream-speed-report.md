# Community benchmark on NVIDIA GeForce RTX 5090

Measured on 2026-10-13 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5090, 32 GiB VRAM
- CPU: AMD Ryzen 9 9950X3D 16-Core Processor (32 cores, 32 threads); installed RAM: 42.6 GiB; storage: not measured
- OS: Linux 7.0.0-38-generic; GPU driver: 615.71.09; CUDA: not read
- Strata: project 0.1.43, commit not a git checkout; engine 0.1.43
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-ud-q4_k_xl-rpc
- Context: 196608 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack unsloth-ud-q4_k_xl --native Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --ple-gguf Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill 2048 --spec 4 --spec-min-p 0.5 --mtp rt --max-context 196608 --kv int8 --resident-experts --vision --vram-reserve-mib 1000 --rpc-prefix-layers 24 --rpc-model-id 68e268b1d5e7127b25a0103c0d5e99a79612a37a00e3d84034f1ead9d5377f79 --rpc-ca <LOCAL-CREDENTIAL> --rpc-stage <ip>:50056 --rpc-cert <LOCAL-CREDENTIAL> --rpc-key <LOCAL-CREDENTIAL>
```

Environment of the config: `STRATA_RESIDENT_HEADROOM_GIB=6`

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 66.4 (51.7-80.4) | 0.42 (0.39-0.44) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1177.2 (1062.7-1182.2) | not measured | 3.43 (3.42-3.80) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1362.8 (1319.7-1366.2) | not measured | 11.81 (11.78-12.19) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1399.1 (1395.0-1399.1) | not measured | 22.94 (22.94-23.01) |
| agent session, follow-up turns | 14501 (14446-14505) at the last turn | see below | up to 64 | 3 | see below | 81.1 (72.7-114.1) | 2.96 (1.92-4.49) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8 % (68.7-68.8); earlier-turn tokens read again per session: 2 (1-4) (earlier turns that the cache did not hold; the 13633 (13633-13633) new tokens of the files are read once); first turn (nothing cached) first token: 1.12 s (1.11-1.20).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 514.5 | 1.203 | 70.4 |
| 1 | 3704 | 623 | 3081 | 690.5 | 4.493 | 77.6 |
| 2 | 5237 | 3767 | 1470 | 587.6 | 2.523 | 80.9 |
| 3 | 8681 | 5301 | 3380 | 804.4 | 4.224 | 80.2 |
| 4 | 11594 | 8734 | 2860 | 866.5 | 3.326 | 81.1 |
| 5 | 14501 | 11657 | 2844 | 970.6 | 2.959 | 78.5 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 197 s.
