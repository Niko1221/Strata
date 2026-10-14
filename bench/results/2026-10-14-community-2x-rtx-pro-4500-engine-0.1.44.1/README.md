# Community benchmark on NVIDIA RTX PRO 4500 Blackwell + NVIDIA RTX PRO 4500 Blackwell

Measured on 2026-10-14 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA RTX PRO 4500 Blackwell + NVIDIA RTX PRO 4500 Blackwell, 64 GiB VRAM
- CPU: AMD Ryzen Threadripper 7960X 24-Cores (24 cores, 48 threads); installed RAM: 61.0 GiB; storage: not measured
- OS: Linux 7.0.0-38-generic; GPU driver: 595.91.07; CUDA: 13.2
- Strata: project 0.1.44.1, commit 296b59d2; engine 0.1.44.1
- Power limit: 200 W per card (`nvidia-smi`); GPU memory clock 14,000 MHz under load (set with an offset; sampled in the runs before and after this one on the same boot, not during this run); no other workload on the machine
- Source: Linux source build of the v0.1.44.1 tag with one local patch, PR #1891 (`logit_bias` list normalized once per request); `logit_bias` is not used by this suite, so it has no effect on these numbers

## Model and configuration

- Model and quantization: swift-1.5-iq3_xxs
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration (production): two cards, `layer_split` 25, `--pipeline-windows 2`, `--spec 5 --spec-min-p 0.5`, `--kv k8v4`, `--vision`, draft vocabulary `de_nocjk`; all experts resident in VRAM (100 %); thinking off for the suite; temperature 0

```text
strata --pack swift-iq3_xxs --native Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 5 --spec-min-p 0.5 --mtp rt --max-context 262144 --kv k8v4 --vision --vram-reserve-mib 700 --mtp-draft-vocab draft_vocab_de_nocjk.bin --pipeline-windows 2
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 172.2 (153.0-244.8) | 0.15 (0.14-0.16) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 3279.7 (3082.1-3280.3) | not measured | 1.24 (1.24-1.32) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 4732.4 (4630.3-4734.9) | not measured | 3.41 (3.41-3.49) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 5530.5 (5527.2-5540.0) | not measured | 5.82 (5.81-5.83) |
| agent session, follow-up turns | 14567 (14546-14597) at the last turn | see below | up to 64 | 3 | see below | 163.7 (113.1-203.2) | 1.10 (0.69-1.33) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.7-68.8); earlier-turn tokens read again per session: 3 (3-4) (earlier turns that the cache did not hold; the 13729 (13728-13730) new tokens of the files are read once); first turn (nothing cached) first token: 0.47 s (0.47-0.47).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 1330.0 | 0.471 | 121.3 |
| 1 | 3746 | 625 | 3121 | 2452.5 | 1.289 | 176.6 |
| 2 | 5279 | 3809 | 1470 | 1920.3 | 0.788 | 151.9 |
| 3 | 8723 | 5342 | 3381 | 2571.7 | 1.332 | 170.9 |
| 4 | 11635 | 8773 | 2862 | 2413.6 | 1.205 | 142.4 |
| 5 | 14597 | 11700 | 2897 | 2526.8 | 1.165 | 171.7 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 64 s.
