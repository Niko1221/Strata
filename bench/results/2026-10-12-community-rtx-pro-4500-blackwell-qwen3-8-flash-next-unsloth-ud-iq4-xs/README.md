# Community benchmark on NVIDIA RTX PRO 4500 Blackwell

Measured on 2026-10-12 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA RTX PRO 4500 Blackwell, 32 GiB VRAM
- CPU: AMD EPYC 7A23 48-Core Processor (48 cores, 48 threads); installed RAM: 123.4 GiB; storage: not measured
- OS: Linux 7.0.0-34-generic; GPU driver: 615.71.09; CUDA: not read
- Strata: project 0.1.43, commit 7695859a; engine 0.1.43
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-iq4_xs
- Context: 600000 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack unsloth-ud-iq4_xs --native Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto:32768 --spec 4 --spec-min-p 0.5 --mtp rt --kv int8 --resident-budget-gib 55 --vision --vram-reserve-mib 989 --ple-io ram --conversation-cache-mib 16384 --prompt-cache 12 --conversation-cache-slots 8 --pcie-frac 0.20 --prompt-cache-tail --adapt-async 1 --expert-profile-save expert-profile-learned.bin --expert-profile-save-every 2 --lookup-chain 2 --max-context 600000 --rope-scaling yarn --rope-scale 2.29 --kv-resident 300000
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 82.6 (55.1-120.1) | 0.33 (0.29-0.57) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1718.3 (1651.7-1730.5) | not measured | 2.36 (2.35-2.46) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 3057.4 (2627.4-3075.8) | not measured | 5.30 (5.25-6.14) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 3241.5 (2921.5-3267.8) | not measured | 9.93 (9.85-11.09) |
| agent session, follow-up turns | 14461 (14441-14472) at the last turn | see below | up to 64 | 3 | see below | 97.9 (54.6-133.3) | 2.13 (1.92-2.29) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8 % (68.6-68.8); earlier-turn tokens read again per session: 4 (4-74) (earlier turns that the cache did not hold; the 13635 (13633-13636) new tokens of the files are read once); first turn (nothing cached) first token: 1.13 s (1.12-1.24).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 561.8 | 1.12 | 84.2 |
| 1 | 3712 | 630 | 3082 | 1465.7 | 2.155 | 110.3 |
| 2 | 5245 | 3775 | 1470 | 780.2 | 1.927 | 105.4 |
| 3 | 8689 | 5238 | 3451 | 1537.1 | 2.293 | 103.9 |
| 4 | 11613 | 8752 | 2861 | 1355.6 | 2.177 | 68.5 |
| 5 | 14461 | 11615 | 2846 | 1340.6 | 2.155 | 54.6 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 119 s.
