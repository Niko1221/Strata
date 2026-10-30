# Community benchmark on AMD Radeon RX 7900 XTX

Measured on 2026-10-30 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 7900 XTX, 24 GiB VRAM
- CPU: AMD Ryzen 9 5900X 12-Core Processor (12 cores, 24 threads); installed RAM: 63.9 GiB; storage: SAMSUNG MZ7LM1T9HMJP-00005 SSD/SATA; Msft Virtual Disk SSD/File Backed Virtual; TOSHIBA MG09ACA18TE HDD/SATA; CT500MX500SSD1 SSD/SATA
- OS: Windows 10 (10.0.19044); GPU driver: 32.0.32015.7029; CUDA (driver supports up to): not read
- Model files and packs: on a KINGSTON SKC3000D2048G NVMe SSD (KC3000 2 TB, M.2 2280, PCIe 4.0 x4, Phison E18, rated 7,000 MB/s sequential read; Windows reports the link at PCIe 4.0 x4; also the Windows system drive). The list above leaves this drive out.
- Strata: project 0.1.54, commit 870aedd9; engine 0.1.54
- Strata release v0.2.8 (a clone of tag v0.2.8); setup installed the ready-made engine 0.1.54-windows-x86_64-hip-rdna3 from the signed update feed
- Background workloads (added by hand): llama-swap proxy idle with a 230M llama.cpp title generator loaded on the CPU (-ngl 0); qBittorrent; ProtonVPN service; Signal; opencode; Microsoft SQL Server (sqlservr); WizTree; Windows Defender
- Power limit: not measured; stock settings

## Model and configuration

- Model and quantization: swift-1.5-iq3_xxs
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata.exe --pack swift-iq3_xxs --native Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill 16384 --spec 4 --spec-min-p 0.5 --mtp rt --max-context 131072 --kv int8 --kv-resident 32768
```

Environment of the config: `STRATA_HIPBLASLT_TUNING=gfx1100-hipblaslt-100500.txt`, `STRATA_PF_FUSED=1`, `STRATA_PF_GEMM=1`, `STRATA_PF_SWITCH_MIN_T=4096`, `STRATA_HIP_WMMA=1`, `STRATA_PA_FAST=1`

Local tuning (added by hand): the config is not the setup default. A local installer adds five switches (`STRATA_PF_FUSED`, `STRATA_PF_GEMM`, `STRATA_PF_SWITCH_MIN_T`, `STRATA_HIP_WMMA`, `STRATA_PA_FAST`) and replaces `--prefill auto` with `--prefill 16384`. Setup chose `--context 131072 --kv int8`. Engine 0.1.54 warns that `--prefill 16384` stopped with 0xC0000409 on an RX 7900 XTX for chunks of 8192 or more. That crash did not occur in this run, including the three 32K prompts.

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 72.0 (57.2-78.3) | 0.49 (0.42-0.53) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 827.8 (822.0-833.1) | not measured | 4.88 (4.85-4.92) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1781.1 (1773.7-1784.2) | not measured | 9.05 (9.03-9.09) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1777.6 (1768.0-1781.2) | not measured | 18.08 (18.05-18.20) |
| agent session, follow-up turns | 12942 (12917-12945) at the last turn | see below | up to 64 | 3 | see below | 89.8 (62.3-102.9) | 4.37 (3.93-4.79) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.3 % (69.1-69.3); earlier-turn tokens read again per session: 73 (73-144) (earlier turns that the cache did not hold; the 12063 (12063-12065) new tokens of the files are read once); first turn (nothing cached) first token: 3.04 s (2.84-3.09).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 215.9 | 2.844 | 44.2 |
| 1 | 3350 | 623 | 2727 | 627.4 | 4.374 | 77.7 |
| 2 | 4883 | 3414 | 1469 | 374.5 | 3.97 | 62.3 |
| 3 | 8127 | 4876 | 3251 | 683.5 | 4.795 | 83.6 |
| 4 | 10196 | 8190 | 2006 | 487.2 | 4.155 | 85.2 |
| 5 | 12942 | 10259 | 2683 | 602.6 | 4.502 | 75.7 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 210 s. The background workloads above were added by hand after the run.

Storage note (added by hand): an earlier run of this suite on the same day, with the model files NTFS-compressed (the Windows compression attribute on the model folder, ratio 1.0 to 1), measured the same fresh prompts (821.7 / 1764.5 / 1766.1 tok/s at 4K / 16K / 32K) but slower agent turns: prompt 147.5 to 335.7 tok/s in run 1 against 215.9 to 683.5 here. The engine's PLE direct-read probe then measured about 5,100 rows/s, against about 138,000 rows/s for this run. This run is after `compact /u`.
