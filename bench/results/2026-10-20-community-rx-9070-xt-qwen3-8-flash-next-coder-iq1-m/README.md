# Community benchmark on AMD Radeon RX 9070 XT

Measured on 2026-10-20 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 9070 XT, 16 GiB VRAM (HIP device 1; device 0 is the CPU's integrated Radeon, gfx1036, which this engine does not run on). PCIe: the engine's probe read 52.1 GB/s host->device (`engine-log-excerpt.txt`); link width and generation not read
- CPU: AMD Ryzen 9 9950X 16-Core Processor (16 cores, 32 threads); installed RAM: 31.1 GiB (2x 16 GB Kingston KF556C40-16, DDR5-5600 modules running at 4800 MT/s, EXPO off); motherboard MSI PRO X870-P WIFI
- Storage: the model files are on a Kingston NV3 2 TB NVMe (SNV3S2000G)
- OS: Windows 11 (10.0.26300); GPU driver: 32.0.31041.1004 (AMD Software); ROCm: 10.2.0a20260930 (TheRock wheels, inside the engine package)
- Strata: project 0.1.45.3, commit e88ee39d; engine 0.1.45.3, **built on this PC from that commit** with `tools\hip\build_windows.bat` and `STRATA_HIP_ARCHS=gfx1201` (gfx1201 code only, hipBLASLt 1.5.0), installed with `START-HERE.bat --backend hip --prebuilt dist\` because no ready-made AMD engine was published for 0.1.45.3. Its `BUILD.json` therefore says `"source": "prebuilt"`
- Background workloads: the Windows desktop and the Claude desktop app were open; other background load was not controlled. Which GPU drives the display: not recorded. Power limits: not checked

## Model and configuration

- Model and quantization: qwen3.8-flash-next-coder-iq1_m (ISTA-DASLab's Coder, IQ1_M, the two GGUF shards below), set up by `START-HERE.bat`
- Context: 65536 tokens; parallel slots: 1; KV int8
- Low-RAM mode: on (setup chose `--resident-experts` for 31 GB of RAM). At start the engine kept 10.86 GiB of the experts the GPU cache does not hold in RAM and reads the rest from the model folder; the GPU cache held 2,934 slots (5.61 GiB). See `engine-log-excerpt.txt`
- **Experimental speed projection: ON** (the control vector in the command below, chosen in setup). It changes the model's answers; it was not compared against a run without it
- Not changed: calibration not run; `STRATA_HIP_WMMA` unset (the engine says the prompt attention on matrix cores is off by default on Windows); vision on, CPU reader
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata.exe --pack coder-iq1_m --native Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf --expert-profile expert-profile-coder.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 65536 --kv int8 --resident-experts --vision --control-vector-scaled Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer
```

Environment of the config: `STRATA_HIPBLASLT_TUNING=gfx1201-hipblaslt-100500.txt`, `STRATA_RESIDENT_PIN=0`

## Method

Run by `python tools/strata_bench.py --config strata-coder-iq1_m.json --url http://127.0.0.1:8080 --long` against the HTTP API of a Strata server; model loading is not timed. The server was started with the tool's own `start_server()` (the same command and config copy it uses), in a separate process, because starting it from the tool failed at e88ee39d (see Correctness and limitations). One suite run, on a freshly started server. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 46.9 (42.5-58.8) | 0.64 (0.48-0.86) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 908.6 (846.5-1106.9) | not measured | 4.46 (3.65-4.79) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1271.4 (1258.2-1273.5) | not measured | 12.67 (12.65-12.80) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1237.3 (1218.2-1251.3) | not measured | 25.95 (25.67-26.34) |
| agent session, follow-up turns | 14683 (14570-14702) at the last turn | see below | up to 64 | 3 | see below | 60.5 (25.2-69.1) | 3.41 (2.70-4.39) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.6 % (69.4-69.6); earlier-turn tokens read again per session: 4 (4-5) (earlier turns that the cache did not hold; the 13812 (13812-13812) new tokens of the files are read once); first turn (nothing cached) first token: 2.19 s (2.09-2.56).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 297.4 | 2.091 | 49.8 |
| 1 | 4124 | 636 | 3488 | 999.5 | 3.559 | 53.8 |
| 2 | 5644 | 4175 | 1469 | 346.1 | 4.289 | 50.2 |
| 3 | 9077 | 5707 | 3370 | 933.9 | 3.665 | 66.1 |
| 4 | 11762 | 9079 | 2683 | 867.9 | 3.134 | 25.2 |
| 5 | 14570 | 11763 | 2807 | 836.1 | 3.409 | 50.9 |

Per-request data: `results.json`. Engine log lines from the start of this server: `engine-log-excerpt.txt` (paths replaced by `<HOME>`, `<STRATA>`, `<DATA>`). Device checks asked for in `docs/AMD_HIP.md`: `strata-device.txt`.

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; the lines above marked as hand-added come from Windows (WMI) and the engine log. Total time of the suite: 239 s.

Not tested: a run without the experimental speed projection; `STRATA_HIP_WMMA=1`; more than one suite run per server start; a long generation. A second run the same day, the configuration this PC now uses (RAM at its rated 5600 MT/s with AMD EXPO on, and `STRATA_PREFILL_HOST_PLAN=0`, #3518), is in [`../2026-10-20-community-rx-9070-xt-qwen3-8-flash-next-coder-iq1-m-expo/`](../2026-10-20-community-rx-9070-xt-qwen3-8-flash-next-coder-iq1-m-expo/README.md); its engine start also got more free memory, so it is not a clean A/B of EXPO.

Seen while preparing this run (reported, not fixed here):

- `engine\strata-device.exe --selftest`, as `docs/AMD_HIP.md` gives it, stops on a PC whose integrated Radeon is HIP device 0: it tests device 0 and refuses gfx1036. With `HIP_VISIBLE_DEVICES=1` it passes on the RX 9070 XT (`strata-device.txt`).
- `python tools/strata_bench.py` (and so `BENCH.bat`) exits with `NameError: name 'port' is not defined` at `tools/strata_bench.py:990` whenever it has to start the server itself, as already reported in #3235; reproduced here at e88ee39d. The `--url` path against a running server worked on this PC.
- The engine warned that the Windows commit limit (RAM + page file, 49.1 GiB) had 4.7 GiB left at start (#1607). No crash happened during this run.
