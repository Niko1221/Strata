# Community benchmark on NVIDIA GeForce RTX 5060 Ti + NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 3080 Ti

Measured on 2026-10-15 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5060 Ti + NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 3080 Ti, 39 GiB VRAM
- CPU: Intel(R) Xeon(R) CPU E5-2683 v4 @ 2.10GHz (32 cores, 64 threads); installed RAM: 184.5 GiB; storage: not measured
- OS: Linux 7.2.6-070206-generic; GPU driver: 615.71.09; CUDA: not read
- Strata: project 0.1.44, commit ed4773eb; engine 0.1.44
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_xxs
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack sc117_iq3_xxs --native Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.65 --mtp rt --max-context 262144 --kv k8v4 --trim-stage-weights --host-core last --pipeline-windows 2 --vision
```

Environment of the config: `STRATA_STAGE_TRIM=1`, `STRATA_SPLIT_MTP_BATCH=1`, `STRATA_CACHE_MESSAGE_BOUNDARY=1`

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 104.0 (89.9-117.7) | 0.45 (0.41-0.47) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1033.8 (991.3-1034.6) | not measured | 3.91 (3.91-4.08) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2046.8 (2022.2-2058.0) | not measured | 7.88 (7.84-7.98) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 2788.0 (2776.3-2795.5) | not measured | 11.55 (11.52-11.60) |
| agent session, follow-up turns | 14573 (14544-14588) at the last turn | see below | up to 64 | 3 | see below | 108.9 (64.5-154.8) | 3.62 (2.92-4.05) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.6-68.8); earlier-turn tokens read again per session: 4 (4-75) (earlier turns that the cache did not hold; the 13730 (13729-13731) new tokens of the files are read once); first turn (nothing cached) first token: 1.93 s (1.92-1.94).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 322.2 | 1.915 | 81.0 |
| 1 | 3760 | 638 | 3122 | 783.0 | 4.02 | 105.5 |
| 2 | 5293 | 3823 | 1470 | 483.4 | 3.075 | 99.6 |
| 3 | 8704 | 5324 | 3380 | 842.4 | 4.049 | 87.4 |
| 4 | 11609 | 8747 | 2862 | 757.1 | 3.819 | 120.1 |
| 5 | 14544 | 11645 | 2899 | 791.2 | 3.704 | 87.3 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 157 s.
