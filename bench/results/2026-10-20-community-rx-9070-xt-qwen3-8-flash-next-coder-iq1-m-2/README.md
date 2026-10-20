# Community benchmark on AMD Radeon RX 9070 XT (second run, RAM at 5600 MT/s)

Measured on 2026-10-20 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

The same PC, engine, model and configuration as [the first run](../2026-10-20-community-rx-9070-xt-qwen3-8-flash-next-coder-iq1-m/README.md), measured the same day after AMD EXPO was turned on in the BIOS (RAM 4800 -> 5600 MT/s) and the PC restarted. **It is not a clean A/B of EXPO**: the restart also left more memory free, so this engine start got a larger GPU cache and RAM copy (see [Compared with the first run](#compared-with-the-first-run)).

## Hardware and software

- GPU and VRAM: AMD Radeon RX 9070 XT, 16 GiB VRAM (HIP device 1). PCIe: the engine's probe read 54.9 GB/s host->device (52.1 in the first run)
- CPU: AMD Ryzen 9 9950X 16-Core Processor (16 cores, 32 threads); installed RAM: 31.1 GiB (2x 16 GB Kingston KF556C40-16, **now at 5600 MT/s with AMD EXPO on**, the BIOS change made for this run); motherboard MSI PRO X870-P WIFI
- Storage: the model files are on a Kingston NV3 2 TB NVMe (SNV3S2000G)
- OS: Windows 11 (10.0.26300); GPU driver: 32.0.31041.1004; ROCm: 10.2.0a20260930 (inside the engine package)
- Strata: commit db53461d (the first run's commit e88ee39d plus that report's files); engine 0.1.45.3, the same binary as the first run (built on this PC with `tools\hip\build_windows.bat`, gfx1201 only)
- Background workloads: the Windows desktop and the Claude desktop app were open; other background load was not controlled. Which GPU drives the display: not recorded. Power limits: not checked

## Model and configuration

- Model and quantization: qwen3.8-flash-next-coder-iq1_m (ISTA-DASLab's Coder, IQ1_M)
- Context: 65536 tokens; parallel slots: 1; KV int8
- Low-RAM mode: on (`--resident-experts`). This start kept 14.25 GiB of experts in RAM (first run: 10.86 GiB) and 3,263 in the GPU cache (first run: 2,934)
- **Experimental speed projection: ON**, as in the first run
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata.exe --pack coder-iq1_m --native Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf --expert-profile expert-profile-coder.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 65536 --kv int8 --resident-experts --vision --control-vector-scaled Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer
```

Environment of the config: `STRATA_HIPBLASLT_TUNING=gfx1201-hipblaslt-100500.txt`, `STRATA_RESIDENT_PIN=0`

## Method

Run by `python tools/strata_bench.py --config strata-coder-iq1_m.json --url http://127.0.0.1:8080 --long --compare <first run>` against the HTTP API of a Strata server started the same way as in the first run (the tool's own `start_server()`, because of #3235); model loading is not timed. One suite run, on a freshly started server. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 60.2 (23.5-71.1) | 0.51 (0.41-0.58) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1186.6 (930.2-1189.1) | not measured | 3.42 (3.41-4.36) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1314.3 (1309.6-1315.0) | not measured | 12.27 (12.24-12.31) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1397.9 (1281.4-1452.0) | not measured | 22.96 (22.12-25.06) |
| agent session, follow-up turns | 14662 (14653-14673) at the last turn | see below | up to 64 | 3 | see below | 67.8 (59.9-83.8) | 3.08 (2.46-4.33) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.4 % (69.4-69.6); earlier-turn tokens read again per session: 73 (3-74) (earlier turns that the cache did not hold; the 13812 (13811-13813) new tokens of the files are read once); first turn (nothing cached) first token: 1.91 s (1.90-2.03).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 327.0 | 1.902 | 54.4 |
| 1 | 4132 | 646 | 3486 | 1084.9 | 3.264 | 61.6 |
| 2 | 5665 | 4195 | 1470 | 588.5 | 2.522 | 70.7 |
| 3 | 9098 | 5729 | 3369 | 973.8 | 3.508 | 67.8 |
| 4 | 11832 | 9149 | 2683 | 885.6 | 3.085 | 60.9 |
| 5 | 14673 | 11866 | 2807 | 923.7 | 3.077 | 63.0 |

Per-request data: `results.json`. Engine log lines from the start of this server: `engine-log-excerpt.txt` (paths replaced by `<HOME>`, `<STRATA>`, `<DATA>`).

## Compared with the first run

`tools/strata_bench.py --compare` against the first run (one suite each, medians):

| Test | First run (4800 MT/s) | This run (5600 MT/s) | Change |
| --- | ---: | ---: | ---: |
| chat decode tok/s | 46.90 | 60.20 | +28.4% |
| chat first token (s) | 0.64 | 0.51 | -20.3% |
| 4K prompt tok/s | 908.60 | 1186.60 | +30.6% |
| 16K prompt tok/s | 1271.40 | 1314.30 | +3.4% |
| 32K prompt tok/s | 1237.30 | 1397.90 | +13.0% |
| 32K first token (s) | 25.95 | 22.96 | -11.5% |
| agent decode tok/s | 60.50 | 67.80 | +12.1% |
| agent cache hit % | 69.58 | 69.40 | -0.3% |
| agent re-read tokens/session | 4 | 73 | |

What else differed between the two engine starts (from the two `engine-log-excerpt.txt` files):

| Engine start | First run | This run |
| --- | ---: | ---: |
| GPU expert cache | 2,934 slots, 5.61 GiB | 3,263 slots, 6.23 GiB |
| Experts kept in RAM (resident complement) | 10.86 GiB, capped by the Windows commit limit | 14.25 GiB, capped by available RAM |
| Windows commit left at start | 4.7 of 49.1 GiB | 8.2 of 50.1 GiB |
| Prefill plan: drive reads per 32,768-token prompt | about 64.0 GB | about 38.2 GB |
| PCIe probe host->device | 52.1 GB/s | 54.9 GB/s |

So the gains above come from the faster RAM **and** from a larger GPU cache and RAM copy after the restart; this pair of runs cannot separate the two. The chat decode range is wide in this run (23.5-71.1, n=9). A clean EXPO comparison would toggle only EXPO with the same free memory at engine start, several runs each.

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; the hand-added lines come from Windows (WMI) and the engine log. Total time of the suite: 216 s.
