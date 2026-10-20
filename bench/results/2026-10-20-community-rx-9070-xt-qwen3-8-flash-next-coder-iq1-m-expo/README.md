# Community benchmark on AMD Radeon RX 9070 XT (RAM at 5600 MT/s, host-aware prompt plan off)

Measured on 2026-10-20 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

The configuration this PC now runs: the same PC, engine, model and settings as [the first run](../2026-10-20-community-rx-9070-xt-qwen3-8-flash-next-coder-iq1-m/README.md), with two changes: AMD EXPO on (RAM 4800 -> 5600 MT/s), and `STRATA_PREFILL_HOST_PLAN=0` in the config's `env`, because on this PC the host-aware prompt plan made 16K-32K prompts slower whenever it engaged (#3518). With it off the engine keeps setup's 6144-token prompt chunk and 96-slot ring, so this run's prompt path is the same as the first run's (which also kept 6144/96). It is the "after" reference for the first run.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 9070 XT, 16 GiB VRAM (HIP device 1). PCIe: the engine's probe read 55.1 GB/s host->device
- CPU: AMD Ryzen 9 9950X 16-Core Processor (16 cores, 32 threads); installed RAM: 31.1 GiB (2x 16 GB Kingston KF556C40-16 at 5600 MT/s, AMD EXPO on); motherboard MSI PRO X870-P WIFI
- Storage: the model files are on a Kingston NV3 2 TB NVMe (SNV3S2000G)
- OS: Windows 11 (10.0.26300); GPU driver: 32.0.31041.1004; ROCm: 10.2.0a20260930 (inside the engine package)
- Strata: commit 66db9516 (e88ee39d plus this PR's report files); engine 0.1.45.3, the same binary as the first run (built on this PC with `tools\hip\build_windows.bat`, gfx1201 only)
- Background workloads: the Windows desktop and the Claude desktop app were open; other background load was not controlled. Which GPU drives the display: not recorded. Power limits: not checked

## Model and configuration

- Model and quantization: qwen3.8-flash-next-coder-iq1_m (ISTA-DASLab's Coder, IQ1_M)
- Context: 65536 tokens; parallel slots: 1; KV int8
- Low-RAM mode: on (`--resident-experts`). This start kept 16.45 GiB of experts in RAM and 3,081 in the GPU cache (first run: 10.86 GiB and 2,934)
- **Experimental speed projection: ON**, as in the first run
- Settings: the install's own configuration plus `STRATA_PREFILL_HOST_PLAN=0` (hip backend); thinking off for the suite; temperature 0

```text
strata.exe --pack coder-iq1_m --native Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf --expert-profile expert-profile-coder.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 65536 --kv int8 --resident-experts --vision --control-vector-scaled Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer
```

Environment of the config: `STRATA_HIPBLASLT_TUNING=gfx1201-hipblaslt-100500.txt`, `STRATA_RESIDENT_PIN=0`, `STRATA_PREFILL_HOST_PLAN=0`

## Method

Run by `python tools/strata_bench.py --config strata-coder-iq1_m.json --url http://127.0.0.1:8080 --long --compare <first run>` against the HTTP API of a Strata server started the same way as in the first run (the tool's own `start_server()`, because of #3235); model loading is not timed. One suite run, on a freshly started server. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 56.6 (49.3-79.4) | 0.45 (0.40-0.54) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1219.2 (904.8-1243.0) | not measured | 3.32 (3.25-4.48) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1314.0 (1277.0-1314.9) | not measured | 12.27 (12.25-12.62) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1322.9 (1294.5-1330.6) | not measured | 24.30 (24.12-24.81) |
| agent session, follow-up turns | 14642 (14642-14699) at the last turn | see below | up to 64 | 3 | see below | 69.1 (53.1-76.9) | 3.07 (2.38-3.61) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.5 % (69.4-69.5); earlier-turn tokens read again per session: 4 (3-74) (earlier turns that the cache did not hold; the 13812 (13811-13812) new tokens of the files are read once); first turn (nothing cached) first token: 1.89 s (1.86-1.96).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 316.6 | 1.964 | 59.9 |
| 1 | 4124 | 636 | 3488 | 1079.0 | 3.287 | 61.0 |
| 2 | 5657 | 4188 | 1469 | 624.0 | 2.381 | 62.5 |
| 3 | 9066 | 5697 | 3369 | 1030.0 | 3.312 | 64.9 |
| 4 | 11785 | 9102 | 2683 | 893.6 | 3.03 | 70.4 |
| 5 | 14642 | 11835 | 2807 | 929.3 | 3.068 | 72.0 |

Per-request data: `results.json`. Engine log lines from the start of this server: `engine-log-excerpt.txt` (paths replaced by `<HOME>`, `<STRATA>`, `<DATA>`).

Two more starts with the same settings, run earlier the same evening as the "off" arms of the A/B in #3518 (one suite each, medians): chat decode 56.6 and 58.3 tok/s, prompt 1,179 / 1,160 (4K), 1,296 / 1,306 (16K) and 1,281 / 1,277 tok/s (32K), agent decode 68.0 and 64.8 tok/s. Their per-request data is not in this PR.

## Compared with the first run

`tools/strata_bench.py --compare` against the first run (RAM at 4800 MT/s; one suite each, medians):

| Test | First run (4800 MT/s) | This run (5600 MT/s, plan off) | Change |
| --- | ---: | ---: | ---: |
| chat decode tok/s | 46.90 | 56.60 | +20.7% |
| chat first token (s) | 0.64 | 0.45 | -28.9% |
| 4K prompt tok/s | 908.60 | 1219.20 | +34.2% |
| 16K prompt tok/s | 1271.40 | 1314.00 | +3.4% |
| 32K prompt tok/s | 1237.30 | 1322.90 | +6.9% |
| 32K first token (s) | 25.95 | 24.30 | -6.3% |
| agent decode tok/s | 60.50 | 69.10 | +14.2% |
| agent cache hit % | 69.58 | 69.48 | -0.1% |

Both runs read prompts with the same 6144-token chunk and 96-slot ring. They still differ in more than the RAM speed: this start had 3,081 GPU cache slots (5.89 GiB) and 16.45 GiB of experts in RAM, the first run 2,934 slots (5.61 GiB) and 10.86 GiB, because more memory was free after the restart (`engine-log-excerpt.txt` of each run). The gains above cannot be split between the faster RAM and the larger caches with these runs.

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; the hand-added lines come from Windows (WMI) and the engine log. Total time of the suite: 216 s.
