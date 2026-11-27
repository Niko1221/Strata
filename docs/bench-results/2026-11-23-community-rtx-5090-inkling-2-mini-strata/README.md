# Community benchmark on NVIDIA GeForce RTX 5090

Measured on 2026-11-23 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5090, 32 GiB VRAM
- CPU: Intel(R) Core(TM) Ultra 9 285K (24 cores, 24 threads); installed RAM: 60.9 GiB; storage: nvme0n1 KINGSTON SNV3S4000G nvme
- OS: Linux 7.0.0-43-generic; GPU driver: 595.99.02; CUDA (driver supports up to): 13.2
- Strata: project 0.1.66, commit not a git checkout; engine strata-inkling-0.1.66
- GPU detail (after the run, near idle): power_limit_w=575.00, power_draw_w=574.05, pcie=gen5 x16, pstate=P1, temp_c=73
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: inkling-2-mini-sq-20
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 44 (43-50) | 0 | 100 (100-100) | 9 | not measured | 161.3 (143.6-243.9) | 0.05 (0.05-0.05) |
| 4K prompt, fresh | 4075 (4075-4075) | 0 (0-0) | 24 cap | 3 | 8712.9 (8705.4-8737.1) | not measured | 0.49 (0.49-0.49) |
| 16K prompt, fresh | 16128 (16128-16128) | 0 (0-0) | 24 cap | 3 | 7860.0 (7847.0-7872.7) | not measured | 2.08 (2.08-2.09) |
| 32K prompt, fresh | 32208 (32208-32208) | 0 (0-0) | 24 cap | 3 | 6742.4 (6734.1-6763.7) | not measured | 4.82 (4.80-4.82) |
| agent session, follow-up turns | 16033 (12834-16084) at the last turn | see below | up to 64 | 3 | see below | 141.8 (98.5-189.3) | 1.89 (1.03-2.65) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 0.0 % (0.0-0.0); earlier-turn tokens read again per session: 37125 (24280-37216) (earlier turns that the cache did not hold; the 15195 (12041-15235) new tokens of the files are read once); first turn (nothing cached) first token: 0.45 s (0.37-0.46). Corpus: `repo_docs` (7c81d715a005) [README.md, docs/HOW_IT_WORKS.md, docs/DETAILS.md, docs/INSTALL.md, docs/MODELS.md].

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 556 | 0 | 556 | 5138.6 | 0.453 | 186.8 |
| 1 | 5625 | 0 | 5625 | 6905.2 | 1.137 | 138.1 |
| 2 | 7245 | 0 | 7245 | 6532.9 | 1.148 | 140.7 |
| 3 | 10592 | 0 | 10592 | 6549.6 | 1.916 | 189.3 |
| 4 | 12825 | 0 | 12825 | 6336.5 | 2.073 | 140.9 |
| 5 | 16033 | 0 | 16033 | 6250.0 | 2.643 | 98.5 |

### Skipped and failed

- failed agent: no token came back

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 63 s.
