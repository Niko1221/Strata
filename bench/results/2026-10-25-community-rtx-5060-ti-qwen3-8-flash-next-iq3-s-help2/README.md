# Community benchmark on NVIDIA GeForce RTX 5060 Ti

Measured on 2026-10-25 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5060 Ti, 16 GiB VRAM
- CPU: 11th Gen Intel(R) Core(TM) i5-11400F @ 2.60GHz (6 cores, 12 threads); installed RAM: 62.7 GiB; storage: not measured
- OS: Linux 6.8.0-138-generic; GPU driver: 595.71.05; CUDA: 13.2
- Strata: project 0.1.51, commit 4b9e05a2; engine 0.1.51
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s-help2
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 262144 --kv q4_0 --vision --vram-reserve-mib 700 --control-vector-scaled Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer --remote-expert-opt --spec-min-p 0.70 --adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97 --pcie-frac 0.35 --expert-cache-device1 auto --expert-profile-save expert-profile-learned.bin --kv-resident 20480
```

Environment of the config: `CUDA_VISIBLE_DEVICES=0,1`

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 68.2 (51.1-97.0) | 0.41 (0.36-0.45) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1066.6 (1018.4-1068.6) | not measured | 3.79 (3.78-3.97) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1539.9 (1517.6-1542.3) | not measured | 10.46 (10.44-10.61) |
| agent session, follow-up turns | 12778 (12774-12782) at the last turn | see below | up to 64 | 3 | see below | 80.3 (61.4-96.0) | 3.96 (3.73-4.39) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.1 % (69.1-69.3); earlier-turn tokens read again per session: 74 (2-75) (earlier turns that the cache did not hold; the 11897 (11897-11898) new tokens of the files are read once); first turn (nothing cached) first token: 2.00 s (1.98-2.05).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 301.2 | 2.048 | 43.8 |
| 1 | 3212 | 633 | 2579 | 611.3 | 4.245 | 67.2 |
| 2 | 4745 | 3275 | 1470 | 379.6 | 3.899 | 61.4 |
| 3 | 7989 | 4808 | 3181 | 729.1 | 4.389 | 66.0 |
| 4 | 10017 | 7982 | 2035 | 487.6 | 4.199 | 80.2 |
| 5 | 12778 | 10072 | 2706 | 644.1 | 4.228 | 73.0 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 146 s.
