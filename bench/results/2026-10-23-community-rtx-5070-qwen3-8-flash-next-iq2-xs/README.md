# Community benchmark on NVIDIA GeForce RTX 5070

Measured on 2026-10-23 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5070, 12 GiB VRAM
- CPU: Intel(R) Core(TM) i5-14600KF (14 cores, 20 threads); installed RAM: 31.8 GiB (2 x 16 GB DDR4-3600); storage: the model files are on an aigo P2000 NVMe drive, which read at about 1.4-1.6 GB/s in an earlier measurement on this machine
- PCIe: the GPU is on PCIe 5.0 x16; Resizable BAR is off
- OS: Windows 10 Pro 22H2 (10.0.19045), Chinese locale; GPU driver: 591.74; CUDA: 13.1 (driver)
- Strata: release 0.2.3.1 (project 0.1.48.1), commit 7ddc17a3; engine 0.1.48.1 built from that commit on this PC (CUDA 13.0, sm_120, MSVC 2022, `STRATA_PORTABLE=ON`), not the release package's binary
- Background workloads and power limits: not recorded; the GPU also drives the display

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq2_xs (the original Flash-Next IQ2_XS pack, 35.5 GB of experts)
- Context: 32768 tokens; parallel slots: 1
- Settings: the daily configuration of this PC (low-RAM mode, `--mmap-experts`, because the experts do not fit in 31.8 GiB of RAM), with one change: the optional control-vector file that setup offers (`--control-vector-scaled`, not a speed setting) was left out so the numbers compare with other reports; thinking off for the suite; temperature 0

```text
strata.exe --pack iq2_xs --native Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 32768 --kv int8 --mmap-experts --pcie-frac 0.55 --spec-min-p 0.70
```

Environment of the config: `STRATA_FETCH_THREADS=16`

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

How it was run on this PC; neither changes what is measured:

- The server was started separately with the release's `serve/server.py` and the tool was pointed at it with `--url http://127.0.0.1:8080 --config <the config>`, the same way as the 0.1.45 report of this PC, so the tool did not save the engine log itself. `engine-log-excerpt.txt` was cut from the benchmark run's part of the engine log with the tool's own `log_excerpt()` and `redactor()`, and local folders were replaced with placeholders.
- The tool was run with `PYTHONUTF8=1`: on a Chinese-locale Windows it still reads `CMakeLists.txt` with the default code page (`tools/strata_bench.py`, the `project(strata VERSION ...)` lookup) and stops with `UnicodeDecodeError: 'gbk' codec` (reported in #2792).

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 55.8 (14.6-66.8) | 0.73 (0.53-2.12) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 87.0 (56.9-96.7) | not measured | 46.27 (41.63-70.81) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 101.3 (66.7-105.3) | not measured | 158.67 (152.52-240.99) |
| agent session, follow-up turns | 14601 (14581-14621) at the last turn | see below | up to 64 | 3 | see below | 5.7 (0.7-14.4) | 62.82 (41.85-80.74) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.5 % (69.4-69.5); earlier-turn tokens read again per session: 73 (73-144) (earlier turns that the cache did not hold; the 13729 (13729-13730) new tokens of the files are read once); first turn (nothing cached) first token: 24.91 s (24.12-29.21).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 20.9 | 29.207 | 9.8 |
| 1 | 4109 | 641 | 3468 | 55.3 | 62.821 | 9.4 |
| 2 | 5642 | 4173 | 1469 | 31.4 | 46.985 | 2.0 |
| 3 | 9075 | 5635 | 3440 | 64.5 | 53.442 | 0.7 |
| 4 | 11828 | 9120 | 2708 | 52.0 | 52.172 | 1.2 |
| 5 | 14601 | 11884 | 2717 | 34.1 | 80.738 | 2.5 |

Per-request data: `results.json`. Engine log lines from the benchmark run: `engine-log-excerpt.txt` (see Method).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 2161 s.

## Notes

- **Against 0.1.45 on this PC** (the report this one replaces, same settings and tool, 2026-10-18): 4K prompt 58.3 tok/s, 16K prompt 127.5 tok/s, agent follow-up turn first token 48.2 s, agent decode 6.4 tok/s. On 0.2.3.1 the 4K median is higher (87.0) but the ranges overlap (56.9-96.7 against 51.7-98.9); the 16K prompt is slower (101.3, range 66.7-105.3 against 124.3-173.4); the agent turns are not faster.
- **Low RAM and a slower drive.** The RTX 5070 Ti report with 31.6 GiB of RAM (#2791, 0.1.44.1, Windows 11, WD SN770) reads a 4K prompt at 270 tok/s and a 16K prompt at 448 tok/s. The GPU (16 GB against 12 GB) and the drive differ, so this run alone does not say which one matters more.
- **What earlier measurements on this PC showed** (0.1.39, with Windows performance counters during a 20K-token prompt): the drive read only 0.2-0.5 GB/s while about 300 MB of RAM stayed available, so the prompt path was limited by page faults on the mapped `experts.bin`, not by the drive's speed. Those pages also push the decode's cached experts out of RAM, which matches the slow decode in the agent turns (5.7 tok/s). Opt-in changes for this setup and their measurements: #3065.
- **VRAM.** With these settings the engine logs 337 MiB of VRAM free with everything loaded, and warns that the card is nearly full; the suite completed without an error.
- **The short-chat decode** (median 55.8 tok/s) includes the first run right after the warm-up (14.6-21.0 tok/s): in low-RAM mode the first requests after a start read many experts from the drive.
- Not tested here: a context above 32K (`--long`), `--clients`, a RAM budget (`--resident-budget-gib`), the release package's own engine binary, other models.
