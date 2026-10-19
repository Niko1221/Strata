# Community benchmark on NVIDIA GeForce RTX 5060 Ti + NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 3080 Ti

Measured on 2026-10-19 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5060 Ti + NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 3080 Ti, 39 GiB VRAM
- CPU: Intel(R) Xeon(R) CPU E5-2683 v4 @ 2.10GHz (32 cores, 64 threads); installed RAM: 184.5 GiB; storage: not measured
- OS: Linux 7.2.6-070206-generic; GPU driver: 615.71.09; CUDA: not read
- Strata: project 0.1.45.3, commit e88ee39d; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_xxs
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack sc117_iq3_xxs --native Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00002-of-00002.gguf --ple-io ram --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.65 --mtp rt --max-context 262144 --kv k8v4 --trim-stage-weights --host-core last --pipeline-windows 2 --conversation-cache-park-gain 1024 --vision
```

Environment of the config: `STRATA_STAGE_TRIM=1`, `STRATA_SPLIT_MTP_BATCH=1`, `STRATA_CACHE_MESSAGE_BOUNDARY=1`, `STRATA_SM75_PERF=1`, `STRATA_VRAM_RESERVE_MIB=700,550,750`, `STRATA_BODY_STALL_S=60`, `STRATA_PEER_STALL_S=60`

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 81.1 (76.0-102.5) | 0.44 (0.41-0.47) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1035.1 (1025.0-1047.7) | not measured | 3.91 (3.86-3.95) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2052.6 (2042.6-2059.9) | not measured | 7.86 (7.84-7.90) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 2814.2 (2797.2-2815.6) | not measured | 11.44 (11.44-11.51) |
| agent session, follow-up turns | 14684 (14633-14704) at the last turn | see below | up to 64 | 3 | see below | 99.3 (79.1-125.0) | 3.47 (2.89-3.79) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.3 % (69.3-69.5); earlier-turn tokens read again per session: 143 (5-145) (earlier turns that the cache did not hold; the 13814 (13812-13815) new tokens of the files are read once); first turn (nothing cached) first token: 1.95 s (1.90-1.96).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 325.5 | 1.9 | 87.6 |
| 1 | 4128 | 640 | 3488 | 930.5 | 3.785 | 79.1 |
| 2 | 5661 | 4121 | 1540 | 531.4 | 2.928 | 91.5 |
| 3 | 9094 | 5724 | 3370 | 905.5 | 3.756 | 89.7 |
| 4 | 11841 | 9087 | 2754 | 802.4 | 3.467 | 112.5 |
| 5 | 14684 | 11877 | 2807 | 820.4 | 3.455 | 100.2 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 159 s.
