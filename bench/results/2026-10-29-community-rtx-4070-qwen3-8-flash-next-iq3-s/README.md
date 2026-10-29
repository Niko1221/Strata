# Community benchmark on 2x NVIDIA GeForce RTX 4070 (layer split)

Measured on 2026-10-29 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: 2x NVIDIA GeForce RTX 4070, 12 GiB VRAM each, layer split (the tool records one card; see Operator notes)
- CPU: AMD Ryzen 9 5950X 16-Core Processor (16 cores, 32 threads); installed RAM: 125.7 GiB; storage: Crucial T500 NVMe (model files)
- OS: Linux 7.0.0-38-generic; GPU driver: 595.91.07; CUDA: 13.2
- Strata: project 0.1.51, commit 4b9e05a2; engine 0.1.51
- Background workloads and power limits: power limit 200 W per card; idle server, no other GPU work (see Operator notes)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s
- Context: 131072 tokens; parallel slots: 1
- Settings: a tuned configuration, not the install's default (the tool wrote "unchanged" because it measured a running server; see Operator notes); thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 82.1 (66.2-89.2) | 0.44 (0.41-0.49) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1113.3 (1111.8-1154.7) | not measured | 3.63 (3.50-3.63) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2333.3 (2333.0-2363.2) | not measured | 6.91 (6.82-6.91) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 2809.6 (2808.9-2838.3) | not measured | 11.45 (11.33-11.45) |
| agent session, follow-up turns | 12767 (12744-12775) at the last turn | see below | up to 64 | 3 | see below | 99.9 (84.8-112.7) | 2.62 (1.85-3.65) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.2 % (69.2-69.3); earlier-turn tokens read again per session: 2 (2-3) (earlier turns that the cache did not hold; the 11897 (11896-11897) new tokens of the files are read once); first turn (nothing cached) first token: 1.83 s (1.64-1.86).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 376.7 | 1.636 | 63.3 |
| 1 | 3215 | 636 | 2579 | 927.2 | 2.806 | 84.8 |
| 2 | 4748 | 3278 | 1470 | 750.3 | 1.98 | 87.8 |
| 3 | 7992 | 4812 | 3180 | 876.5 | 3.651 | 88.7 |
| 4 | 10017 | 8053 | 1964 | 900.7 | 2.203 | 95.8 |
| 5 | 12767 | 10061 | 2706 | 965.5 | 2.827 | 84.9 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 142 s.

## Operator notes

Added by the operator; the tool cannot see these. Same machine, same freshly loaded engine and the same notes as the
standard run in [`../2026-10-29-standard-rtx-4070-dc2aa70e/OPERATOR-NOTES.md`](../2026-10-29-standard-rtx-4070-dc2aa70e/OPERATOR-NOTES.md)
(hardware, overclock, full engine arguments and environment). This suite ran right after that run (09:09-09:11 UTC),
with no other requests in between; the 32K-prompt and agent-session numbers here are not in the standard run.

- 2x Palit RTX 4070 12 GB, both PCIe Gen4 x8 on CPU lanes, `--layer-split 25`, **overclocked** (NVML memory +2200 /
  core +75), 200 W power limit each; ASUS PRIME X570-PRO, Ryzen 9 5950X, 4x 32 GB DDR4-3200.
- Strata v0.2.6 CUDA source build (`4b9e05a2`, engine 0.1.51), original Flash-Next IQ3_S, 131,072 context, `--kv int8`,
  `--spec 3 --spec-min-p 0.70`, conversation cache 32 GiB, `STRATA_STAGE_TRIM=1 STRATA_PREFILL_CPU_SHARE=auto
  STRATA_SPLIT_MTP_BATCH=1`.
- The owner of this machine directed the work; an AI agent (Anthropic's Claude Code) ran it and wrote these notes.
