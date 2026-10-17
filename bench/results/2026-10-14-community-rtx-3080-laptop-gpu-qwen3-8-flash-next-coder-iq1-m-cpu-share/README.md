# Community benchmark on NVIDIA GeForce RTX 3080 Laptop GPU

Measured on 2026-10-14 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 3080 Laptop GPU, 8 GiB VRAM
- CPU: 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz (8 cores, 16 threads); installed RAM: 31.0 GiB; storage: not measured
- OS: Linux 7.0.0-34-generic; GPU driver: 580.178.04; CUDA: 13.0
- Strata: project 0.1.44.1, commit 296b59d2; engine 0.1.44.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-coder-iq1_m
- Context: 32768 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack coder-iq1_m --native Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf --expert-profile expert-profile-coder.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 32768 --kv int8 --resident-experts
```

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 25.3 (22.3-35.0) | 1.31 (1.06-1.44) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 541.1 (537.5-570.7) | not measured | 7.48 (7.08-7.52) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 627.5 (611.2-633.1) | not measured | 25.65 (25.42-26.33) |
| agent session, follow-up turns | 14506 (14495-14541) at the last turn | see below | up to 64 | 3 | see below | 21.1 (13.2-24.6) | 10.63 (5.73-12.53) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.5-68.6); earlier-turn tokens read again per session: 4 (3-74) (earlier turns that the cache did not hold; the 13729 (13728-13729) new tokens of the files are read once); first turn (nothing cached) first token: 3.24 s (3.16-3.27).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 197.2 | 3.237 | 17.8 |
| 1 | 3751 | 629 | 3122 | 279.4 | 11.364 | 21.1 |
| 2 | 5284 | 3814 | 1470 | 256.8 | 5.818 | 20.9 |
| 3 | 8688 | 5308 | 3380 | 280.6 | 12.138 | 17.4 |
| 4 | 11582 | 8720 | 2862 | 287.5 | 10.047 | 24.3 |
| 5 | 14495 | 11596 | 2899 | 283.7 | 10.29 | 23.0 |

### Skipped and failed

- skipped 32K: the context is 32768 tokens

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 350 s.

## Added by the submitter, not measured by bench.sh

- **Machine:** MSI GP66 laptop. RTX 3080 Laptop GPU, 8 GB (sm_86), 140 W max power limit, driver 580.178.04, CUDA 13.0. i7-11800H (8 cores / 16 threads, AVX-512), 32 GB RAM (31 GiB usable), 8 GiB swap. Micron 3400 1 TB NVMe (PCIe 4.0). Ubuntu 24.04.5, kernel 7.0.0-34. On AC power.
- **Build:** `./update.sh --yes --no-start` from 0.1.43.1 (`a0116b27`); it compiled the engine from source. Setup left both model configs byte-identical to the 0.1.43.1 ones.
- **Model files:** setup's downloads. sha256 of the GGUFs: Coder IQ1_M `e11083ba…087fad` (part 1), `316b46f3…61e113` (part 2); IQ2_XS `92cee27a…ec0a49d7` (part 1), `316b46f3…61e113` (part 2). Expert profile and packs are setup's own.
- **Runs:** one engine start per configuration, each from GPU ≤ 50 °C (the first started at 40 °C; the others waited 7-11.5 min to cool). Within a start, bench.sh's 9 short chats and 3 runs per prompt size. Chat decode spreads within a start are wide (e.g. 22.7-29.0 tok/s), so one start per configuration does not separate decode differences of under ~20%. Prompt throughput is tight (within ±6% of its median).
- **Power profile matters on this laptop.** Ubuntu's `power-saver` was active by default even on AC. All numbers here are in `performance`. On 0.1.43.1 (not repeated on 0.1.44.1) the same Coder config gave 17.7 tok/s chat decode in power-saver vs 25.6 in performance.
- **Heat:** sustained use is slower than a cool start. On 0.1.43.1 (not repeated on 0.1.44.1), a 10-minute back-to-back run went 27.9 → 22.1 tok/s while the CPU package sat at 94-95 °C and its power fell from 37 to 24 W. On 0.1.44.1 the CPU package reached 99-100 °C in every run here.
- **Memory at start:** the first of the three runs (Coder, setup's config) started with 24.4 GiB of file cache (earlier model reads) and 4.1 GiB of swap in use; nothing else was running. See issue B (https://github.com/Niko1221/Strata/issues/2478) for why this matters on a 32 GB machine.
- **PCIe:** the engine's probe reads 7.4 GB/s host→device on a link that nvidia-smi reports as Gen4 x16 (pcie_frac 0.20).
- **IQ2_XS on 32 GB:** `./setup.sh --check` lists IQ2_XS as "needs ~48 GB RAM: does not fit", yet the earlier install chose mapped mode and it runs. In a 16K prompt the OS read up to 101 GB from the NVMe (525k major faults) for 197 GB of expert reads ("file tier I/O" lines in the engine log).
- **Environment:** The server was started with `STRATA_PREFILL_CPU_SHARE=auto` in its environment. bench.sh does not record environment variables, so this folder's own README and config look the same as the default run's.
- **Not tested:** prompts longer than 16K (the 32K test is skipped at a 32K context), 128K context, `--calibrate`, `--clients`, the optional correctness checks, battery power, Windows.
