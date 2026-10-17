# Community benchmark on 4x Radeon Instinct MI50 32GB

Measured on 2026-10-17 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: Radeon Instinct MI50 32GB + Radeon Instinct MI50 32GB + Radeon Instinct MI50 32GB + Radeon Instinct MI50 32GB, 128 GiB VRAM
- CPU: Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz (14 cores, 28 threads); installed RAM: 62.7 GiB; storage: not measured
- OS: Linux 6.8.0-110-generic; GPU driver: not read; CUDA: not read
- Strata: project 0.1.45, commit 477ac13c; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-iq4_xs
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata --pack unsloth-ud-iq4_xs --native Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --pcie-frac 0 --max-context 131072 --kv int8
```

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 33.7 (33.3-35.5) | 0.61 (0.56-0.68) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 490.2 (489.8-490.7) | not measured | 8.24 (8.23-8.25) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 778.3 (770.2-788.5) | not measured | 20.69 (20.43-20.91) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1010.4 (995.8-1045.9) | not measured | 31.80 (30.73-32.26) |
| agent session, follow-up turns | 14598 (14589-14608) at the last turn | see below | up to 64 | 3 | see below | 33.6 (30.2-35.4) | 7.00 (4.06-7.95) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8 % (68.8-68.8); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 13720 (13719-13721) new tokens of the files are read once); first turn (nothing cached) first token: 2.43 s (2.43-2.44).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 254.0 | 2.441 | 33.4 |
| 1 | 3759 | 636 | 3123 | 432.2 | 7.279 | 33.6 |
| 2 | 5292 | 3822 | 1470 | 357.6 | 4.158 | 33.2 |
| 3 | 8725 | 5355 | 3370 | 426.6 | 7.953 | 33.2 |
| 4 | 11644 | 8782 | 2862 | 414.8 | 6.954 | 35.4 |
| 5 | 14598 | 11699 | 2899 | 417.5 | 6.997 | 30.2 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 366 s.

## Custom patch

A custom patch was needed due to a compilation error specific to gfx906. See issue [#2459](https://github.com/Niko1221/Strata/issues/2459#issuecomment-6233746023) for more information.
