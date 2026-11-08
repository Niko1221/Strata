# Community benchmark on Tesla V100-PCIE-32GB + Tesla V100-PCIE-32GB

Measured on 2026-11-08 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no full correctness suite was run; one spot-check (multi-robot simultaneous-move BFS, answer 2) was answered correctly.

## Hardware and software

- GPU and VRAM: Tesla V100-PCIE-32GB + Tesla V100-PCIE-32GB, 64 GiB VRAM
- CPU: Intel(R) Xeon(R) Platinum 8168 CPU @ 2.70GHz (48 cores, 96 threads); installed RAM: 251.5 GiB; model files on NFS from a second host (single spindle archive drives)
- OS: Linux 6.8.0-147-generic; GPU driver: 580.178.04; CUDA (driver supports up to): 13.0
- Strata: project 0.1.58; engine: self-built sm_70 build of jmnargi/Strata-V100 (CUDA 12.0 toolkit, gcc-12, STRATA_EXPERIMENTAL_SM60 + a host-side float->half cast patch for q36_engine.cu)
- GPU detail (after the run, near idle): power_limit_w=250.00, power_draw_w=45.11, pcie=gen3 x16, pstate=P0, temp_c=47
- GPU detail (after the run, near idle): power_limit_w=250.00, power_draw_w=47.45, pcie=gen3 x16, pstate=P0, temp_c=47
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: Qwen3.8-Flash-Next, unsloth UD-Q4_K_XL (native experts, MTP drafter)
- Benchmarked by: thegliffy
- Context: 131072 tokens; parallel slots: 16
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack <install>/packs/unsloth-ud-q4_k_xl --native <models>/unsloth-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --expert-profile <install>/data/expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp <install>/mtp/rt --max-context 131072 --kv int8 --ple-io ram --gpus 0,1
```

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 48.7 (39.7-57.9) | 0.60 (0.51-0.67) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 629.0 (627.6-629.2) | not measured | 6.43 (6.42-6.44) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1441.2 (1435.0-1441.6) | not measured | 11.20 (11.19-11.25) |
| agent session, follow-up turns | 13582 (13571-13582) at the last turn | see below | up to 64 | 3 | see below | 61.8 (48.9-66.7) | 6.64 (6.30-6.74) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.1 % (69.1-69.1); earlier-turn tokens read again per session: 301 (292-301) (earlier turns that the cache did not hold; the 12707 (12705-12707) new tokens of the files are read once); first turn (nothing cached) first token: 4.26 s (4.22-4.28).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 145.7 | 4.218 | 33.3 |
| 1 | 3822 | 602 | 3220 | 486.6 | 6.667 | 49.3 |
| 2 | 5351 | 3815 | 1536 | 245.3 | 6.304 | 48.9 |
| 3 | 8580 | 5344 | 3236 | 483.5 | 6.74 | 56.6 |
| 4 | 10804 | 8573 | 2231 | 345.0 | 6.513 | 63.9 |
| 5 | 13571 | 10797 | 2774 | 420.4 | 6.644 | 50.9 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 222 s.
