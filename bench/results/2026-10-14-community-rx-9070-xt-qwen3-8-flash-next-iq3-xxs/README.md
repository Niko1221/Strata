# Community benchmark on AMD Radeon RX 9070 XT

Measured on 2026-10-14 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

Same machine, same day and the same configuration as the A/B arm in `../2026-10-14-community-rx-9070-xt-qwen3-8-flash-next-iq3-xxs-2/`; the two differ only in `--kv-resident` (this run has it, that one does not). Removing it cost 1.9-2.5% on prompt throughput and 2.0-2.5% on first-token time, with the decode medians also lower (chat -4.4%, agent -6.8%) but with overlapping run-to-run ranges, so the decode part is directional only.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 9070 XT, 16 GiB VRAM
- CPU: AMD Ryzen 7 9800X3D 8-Core Processor (8 cores, 16 threads); installed RAM: 46.8 GiB; storage: not measured
- OS: Windows 11 (10.0.26200); GPU driver: 32.0.31036.15; CUDA: not read
- Strata: project 0.1.44.1, commit 0f2ed512; engine 0.1.44.1
- Storage: the model shards and the prepared pack on an SK hynix PVC10 HFS001TEM9X173N NVMe SSD (954 GiB volume)
- PCIe: Gen5 x16; the engine's own start-up probe measured 54.9 GB/s host->device
- GPU power limit and clocks: not recorded (card at stock settings)
- Background: the desktop was in normal use; no game, other model or benchmark ran during the suite, but nothing was isolated or pinned
- Runtime state of this run: 5293 experts in the GPU cache (8.58 GiB of VRAM), 32.32 GiB of experts resident in RAM (pageable), Windows commit limit about 124 GiB (a 77 GiB page file)
- The reported commit is a local branch: upstream `main` at `296b59d2` plus two local patches (a Simplified Chinese UI translation and the Windows AMD telemetry of PRs #1434/#1152). Neither touches the engine or the model path, so the numbers are the engine's own.

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_xxs
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata.exe --pack iq3_xxs --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 262144 --kv k8v4 --kv-resident 32768 --resident-experts --vram-reserve-mib 256
```

Environment of the config: `STRATA_HIPBLASLT_TUNING=gfx1201-hipblaslt-100500.txt`, `STRATA_RESIDENT_PIN=0`

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 65.7 (60.5-79.3) | 0.39 (0.33-0.44) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 800.5 (799.4-808.5) | not measured | 5.04 (4.99-5.05) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 921.1 (920.6-926.5) | not measured | 17.46 (17.36-17.47) |
| agent session, follow-up turns | 14573 (14532-14592) at the last turn | see below | up to 64 | 3 | see below | 68.8 (40.0-87.3) | 4.67 (3.88-7.27) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.6-68.7); earlier-turn tokens read again per session: 73 (4-74) (earlier turns that the cache did not hold; the 13729 (13728-13730) new tokens of the files are read once); first turn (nothing cached) first token: 2.98 s (2.98-3.19).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 192.1 | 3.19 | 55.7 |
| 1 | 3754 | 632 | 3122 | 675.1 | 4.649 | 68.8 |
| 2 | 5287 | 3818 | 1469 | 202.8 | 7.269 | 58.4 |
| 3 | 8731 | 5280 | 3451 | 709.1 | 4.892 | 61.1 |
| 4 | 11637 | 8775 | 2862 | 616.5 | 4.673 | 76.8 |
| 5 | 14573 | 11674 | 2899 | 623.7 | 4.676 | 48.2 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 186 s.
