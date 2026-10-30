# Community benchmark on AMD Radeon RX 7900 XTX

Measured on 2026-10-30 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 7900 XTX, 24 GiB VRAM
- CPU: AMD Ryzen 9 5900X 12-Core Processor (12 cores, 24 threads); installed RAM: 63.9 GiB; storage: SAMSUNG MZ7LM1T9HMJP-00005 SSD/SATA; Msft Virtual Disk SSD/File Backed Virtual; TOSHIBA MG09ACA18TE HDD/SATA; CT500MX500SSD1 SSD/SATA
- OS: Windows 10 (10.0.19044); GPU driver: 32.0.32015.7029; CUDA (driver supports up to): not read
- Model files and packs: on a KINGSTON SKC3000D2048G NVMe SSD (the list above leaves this drive out)
- Strata: project 0.1.54, commit 870aedd9; engine ds4-0.1
- Engine binary (added by hand): strata-dsv41 0.1.54, built on this machine from tag v0.2.8 (commit 870aedd9e) with `build_hip_win.bat` for gfx1100, without source changes. No upstream feed has a Windows gfx1100 strata-dsv41 package. Toolchain: ROCm 10.2.0a20260930 wheels `rocm[libraries,devel,device-gfx1100]` in a Python venv.
- Background workloads (added by hand): llama-swap proxy idle with a 230M llama.cpp title generator loaded on the CPU (-ngl 0); qBittorrent; ProtonVPN service; Signal; opencode; Microsoft SQL Server (sqlservr); WizTree; Windows Defender
- Power limit: not measured; stock settings

## Model and configuration

- Model and quantization: deepseek-v4.1-flash
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata-dsv41.exe --pack deepseek-v4.1-flash-ist-das-2.25bpw --max-context 131072 --eos 1 --vision off --dspark auto --dspark-conf 0.65
```

Setup ran with `--context 131072`. At this context the engine keeps 1415 of 15360 experts in VRAM. At 32768, engine 0.1.52 kept 1431. No tuning switches are set for this engine.

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 32 (29-38) | 0 | 100 (100-100) | 9 | not measured | 6.0 (5.6-6.9) | 6.74 (5.87-7.91) |
| 4K prompt, fresh | 4125 (4125-4125) | 0 (0-0) | 24 cap | 3 | 222.6 (207.0-226.9) | not measured | 18.54 (18.20-19.93) |
| 16K prompt, fresh | 16562 (16562-16562) | 0 (0-0) | 24 cap | 3 | 371.8 (363.8-375.9) | not measured | 44.57 (44.10-45.55) |
| 32K prompt, fresh | 33126 (33126-33126) | 0 (0-0) | 24 cap | 3 | 415.0 (412.4-416.5) | not measured | 79.87 (79.57-80.38) |
| agent session, follow-up turns | 12677 (12667-12678) at the last turn | see below | up to 64 | 3 | see below | 7.1 (5.8-9.0) | 30.55 (29.34-33.11) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.7 % (69.6-69.7); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 11798 (11797-11799) new tokens of the files are read once); first turn (nothing cached) first token: 27.69 s (27.45-29.40).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 602 | 0 | 602 | 20.5 | 29.399 | 6.4 |
| 1 | 3338 | 638 | 2700 | 81.6 | 33.107 | 5.8 |
| 2 | 4856 | 3401 | 1455 | 48.9 | 29.801 | 6.4 |
| 3 | 7968 | 4919 | 3049 | 92.3 | 33.068 | 7.0 |
| 4 | 10053 | 8031 | 2022 | 66.4 | 30.488 | 7.7 |
| 5 | 12678 | 10100 | 2578 | 81.5 | 31.688 | 6.6 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

### Needle test at 120K (added by hand)

One case of `tools/needle_bench.py --lengths 120k --depths 50 --timeout 7200`, run after the suite against a fresh server with the same configuration on another port. Result: found (`meadow-copper-504`), 110,147 actual prompt tokens, 314 s for the request. Data: `needle-120k.json`. Only one length and one depth were run.

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 1416 s. The background workloads above were added by hand after the run.

Storage note (added by hand): an earlier run of this suite on the same day, with the model files NTFS-compressed (the Windows compression attribute on the model folder, ratio 1.0 to 1), measured decode 1.4 tok/s short chat and 1.9 tok/s agent turns, prompt 69.2 / 138.4 / 163.4 tok/s at 4K / 16K / 32K, and took 6150 s; its 120k needle took 1490 s. Unbuffered 11.4 MB reads of a compressed shard ran at 0.56-0.59 GB/s (7.1-7.6 GB/s after `compact /u`), and the RAM store's 35.7 misses a token then cost 748 ms a token (112 ms after). This run is after `compact /u`.
