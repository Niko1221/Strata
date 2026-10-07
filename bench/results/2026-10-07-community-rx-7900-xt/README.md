# Community benchmark on RX 7900 XT (Windows 11)

Measured on 2026-10-07 by matrix9neonebuchadnezzar2199-sketch. Strata engine 0.1.40.2, original Flash-Next IQ3_S and IQ3_XXS, on Windows with the ready-made AMD engine (not a source build). Each size: three fresh runs at about 100 prompt tokens and three at about 38,184. The two sizes were not loaded together. IQ3_S used context 131,072 and KV q4_0. IQ3_XXS used context 262,144 and KV int8. Main limitations: one machine, three runs per configuration, non-streaming requests (no time-to-first-token), and GPU power, temperature, PCIe link width, and VRAM during the runs were not measured.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 7900 XT (gfx1100), GPU 0. The engine reports 20464 MiB on the card and a Windows WDDM budget of 19653 MiB for this process. No second GPU was selected.
- CPU: AMD Ryzen 7 9800X3D, 8 cores / 16 threads. AVX level: not recorded in this start log.
- Installed RAM: `/v1/status` reported 95.6 GiB total (`Win32_ComputerSystem.TotalPhysicalMemory` 102666436608 bytes). Speed: not measured.
- Storage: Windows is on a CT1000T500SSD8 1 TB NVMe (C:). Model files (`Strata-data`) are on a FIKWOT FX991 4 TB NVMe (H:). An 8 TB SATA HDD is also installed and was not used for the model.
- PCIe link: width and generation not recorded. The engine's startup probe measured 28.4 GB/s host to device (best of 28.3, 28.2, 28.4, 28.3) and `pcie_frac` 0.55.
- OS: Windows 11 Pro, version 10.0.26300 (build 26300).
- Driver: AMD display driver 32.0.23033.1002, dated 2026-03-09 (`Win32_VideoController`).
- ROCm / HIP: runtime bundled with the engine (`amdhip64_7.dll`). `engine/BUILD.json` records ROCm `10.2.0a20260930` and hipBLASLt 100500. No separate ROCm install.
- Strata commit: `e8ca9afd03d839d4f8dbbe82dffce7f8a3bafd7a` (2026-10-07). Engine version 0.1.40.2 from `/v1/status` and `engine/BUILD.json` (`source: prebuilt`, `platform: windows-x64`, `backend: hip`). Not built from source.
- Background workloads: the desktop was in interactive use (Cursor was open). No other LLM server was called during the runs. GPU power limit: not measured.

## Model and configuration

- Model: Qwen3.8-Flash-Next, original, GSQ-RCO quantization by ISTA-DASLab, size IQ3_S.
- GGUF files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` (54,817,524,224 bytes) and `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` (28,800,138,432 bytes). Repository revision: not recorded.
- Vision encoder: off (`/v1/status` `vision.enabled` false; `BUILD.json` `vision: none`). Custom packs or profiles: none. The pack was prepared by setup (`packs\iq3_s`) and the shipped `data/expert-profile.bin` was used (24,576 ranked pairs).
- Context: 131,072. KV: `q4_0` with `--kv-resident 32768` (the start log: 32,768 of 131,072 cells per QSA layer in VRAM, K/V in 0.84 GiB of pinned RAM). Prefill: `auto`. The start log then chose 8,192-token prompt chunks (`prompt chunk auto: 8192 tokens, a 384-slot ring`) and borrowed 2,227 cache slots (4.19 GiB) for the prompt path. Expert cache: `auto`, which settled at 5,512 slots (10.48 GiB of VRAM) after the startup shrink. Low-RAM mode: not used. Conversation cache: not set.
- MTP draft layer: on (`--spec 4 --spec-min-p 0.5`). The draft head loaded at 820 MiB of VRAM. Reasoning: no `reasoning_effort` was sent, and there is no `strata-*.shared-settings.json`. Sampling: server defaults. Calibration: not run. Experimental speed projection: off.
- hipBLASLt tuning table: `gfx1100-hipblaslt-100500.txt`, set through `STRATA_HIPBLASLT_TUNING`. The start log says tuning enabled, 32 rows, gfx1100, version 100500.
- After load, the engine reported 323 MiB of VRAM free and a WDDM warning that little room remained for verify-window buffers. The process did not exit. All eight requests in this report completed.

```text
H:\CURSOR\Strata\Strata-main\run-iq3_s.bat
```

The run config is [strata-iq3_s.json](strata-iq3_s.json). It contains local paths and no credentials. The engine was already up (started 2026-10-07 22:45, engine 0.1.40.2) before these requests. Model loading is not included in the timings.

Startup snapshot from `strata-iq3_s.log`, before any request in this report:

```text
GPU 0: AMD Radeon RX 7900 XT (gfx1100)
expert cache 5512 slots, 10.48 GiB of VRAM
session is up (engine 0.1.40.2)
prompt chunk auto: 8192 tokens
323 MiB of VRAM free with everything loaded
```

## Method

Script: [run_bench.py](run_bench.py). IQ3_S rows: [runs.csv](runs.csv) and [runs.json](runs.json) (`--model qwen3.8-flash-next-iq3_s --out runs.json`). IQ3_XXS rows are in the section below.

- Requests: non-streaming `POST /v1/chat/completions` with `max_tokens` 256. No other sampling fields. Each prompt starts with a fresh `Run id:` UUID.
- Timings: `GET /v1/status` `last_timings` immediately after each request. Prompt tok/s is `prompt_per_second` (freshly read tokens, `prompt_n`, over `prompt_ms`). Decode tok/s is `predicted_per_second` (engine decode clock). Reused tokens are `cache_n`. These are not generated tokens divided by the whole request.
- Wall time is the client clock around the HTTP call, so it includes request overhead. It is not the decode rate.
- Order: one short warm-up, one long warm-up (neither is in the summary table), then 3 short runs and 3 long runs. The server was not restarted between them. The expert cache had been filled at startup (5,512 of 5,512 slots) and had then served the warm-ups.
- Short prompt: "Write a short story about a lighthouse keeper." (94 to 99 tokens including the run id and template).
- Long prompt: 1,100 synthetic log lines from `random.Random(42)`. Line 550 carries the passphrase `amber-keel-2904`. A two-part question follows. Counted runs were 38,183 to 38,186 tokens. `cache_n` was 0 on every run, so the prefix was not reused.
- Memory: system RAM used, from `/v1/status` after each request (a snapshot, not a peak). VRAM during the runs: not measured.
- Nothing else was sent to this server during the runs.

## Results

Summary of the three counted runs. Warm-up rows are below, not in this table.

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| Short | 94 to 99 | 0 | 256 (cap) | 3 | 29.5 (28.0 to 31.8) | 37.3 (34.0 to 37.3) | not measured |
| Long, 38K | 38,183 to 38,186 | 0 | 134 to 139 | 3 | 1,015.8 (1,015.0 to 1,016.3) | 24.3 (21.4 to 25.4) | not measured |

The short-prompt "prompt tok/s" is dominated by fixed per-request overhead on a ~100-token prompt and is not a prefill speed. The long-prompt figure is.

Total wall-clock time (client, includes overhead): short median 10.2 s (9.8 to 11.1); long median 43.3 s (42.9 to 44.2).

RAM used after each counted request rose from 72.6 GiB to 77.9 GiB (`/v1/status`). Before the first warm-up the same field was 70.5 GiB. Unit is GiB as reported by the server.

Per-run data, including warm-ups (also in `runs.csv`). Draft accepted % is `100 * draft_n_accepted / draft_n`.

| Config | Counted | prompt_n | cache_n | prompt_ms | prompt tok/s | generated | decode tok/s | draft accepted % | total s | RAM used GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short | warm-up | 96 | 0 | 3,695.8 | 26.0 | 256 | 45.4 | 61.1 | 9.3 | 70.5 |
| long-38K | warm-up | 38,182 | 0 | 32,954.6 | 1,158.6 | 138 | 51.7 | 67.2 | 35.7 | 70.7 |
| short | yes | 99 | 0 | 3,350.3 | 29.5 | 256 | 37.3 | 62.2 | 10.2 | 72.6 |
| short | yes | 99 | 0 | 3,540.3 | 28.0 | 256 | 34.0 | 66.0 | 11.1 | 74.1 |
| short | yes | 94 | 0 | 2,959.9 | 31.8 | 256 | 37.3 | 61.7 | 9.8 | 75.0 |
| long-38K | yes | 38,185 | 0 | 37,591.9 | 1,015.8 | 139 | 21.4 | 68.6 | 44.2 | 77.9 |
| long-38K | yes | 38,183 | 0 | 37,617.9 | 1,015.0 | 135 | 24.3 | 66.7 | 43.3 | 77.6 |
| long-38K | yes | 38,186 | 0 | 37,574.5 | 1,016.3 | 134 | 25.4 | 66.4 | 42.9 | 77.5 |

The three counted short runs hit the 256-token cap (`finish_reason` `length`). The three counted long runs stopped on their own (`stop`) at 134 to 139 tokens. The long warm-up decoded at 51.7 tok/s; the three counted long runs decoded at 21.4 to 25.4 tok/s. Both figures are `predicted_per_second`. The cause was not isolated.

No request failed. None was skipped.

## Correctness and limitations

The long prompt asks for the passphrase planted on line 550. The warm-up and all three counted long runs quoted `amber-keel-2904` (`answer_has_passphrase` true in `runs.json`). That is one synthetic log, not a needle sweep and not a quality score.

`tools/needle_bench.py`: not measured.

Untested on IQ3_S, and not estimated here:

- Time to first token, streaming, GPU power, temperature, PCIe width, and VRAM during generation.
- Images (vision is off).
- Other sizes besides IQ3_S and IQ3_XXS (Q2_0, IQ2_XS, the Coder, Swift, and Unsloth variants).
- Contexts other than this run's 131,072. A 128k needle was not run.
- A restarted server between runs. The counted runs follow two warm-ups on an already loaded engine.

## IQ3_XXS

Same PC, driver, commit, and engine binary as above. The server was started with [run-iq3_xxs.bat](../../../run-iq3_xxs.bat) and [strata-iq3_xxs.json](strata-iq3_xxs.json) after IQ3_S had been stopped. `/v1/status` reported `qwen3.8-flash-next-iq3_xxs`, engine 0.1.40.2, context 262,144.

- GGUF files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` (47,039,860,096 bytes) and `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` (28,800,138,432 bytes). Repository revision: not recorded. Vision: off. Pack: `packs\iq3_xxs`, shipped `data/expert-profile.bin`.
- Context: 262,144. KV: `int8` with `--kv-resident 32768` (start log: 32,768 of 262,144 cells per QSA layer in VRAM, K/V in 3.09 GiB of pinned RAM). Prefill: `auto`, which the start log ran as 8,192-token chunks. Expert cache: `auto`, 6,376 slots, 10.31 GiB of VRAM. MTP: `--spec 4 --spec-min-p 0.5`. Low-RAM mode, conversation cache, calibration, and experimental speed projection: off.
- After load the engine reported 328 MiB of VRAM free and the same WDDM warning as IQ3_S. The process did not exit. All eight requests completed.

```text
python run_bench.py --model qwen3.8-flash-next-iq3_xxs --out runs-iq3_xxs.json
```

Same prompts, `max_tokens` 256, fresh run id, and `last_timings` fields as IQ3_S. One short warm-up and one long warm-up, then three counted runs of each. The server was not restarted between those eight requests. Rows: [runs-iq3_xxs.csv](runs-iq3_xxs.csv) and [runs-iq3_xxs.json](runs-iq3_xxs.json).

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| Short | 97 to 99 | 0 | 256 (cap) | 3 | 90.5 (90.3 to 92.4) | 74.1 (66.2 to 76.3) | not measured |
| Long, 38K | 38,182 to 38,185 | 0 | 135 to 256 | 3 | 1,400.1 (1,399.7 to 1,400.3) | 72.3 (59.4 to 73.1) | not measured |

The short-prompt figure is overhead on a ~100-token prompt, not a prefill speed. The long-prompt figure is.

Wall-clock time (client): short median 4.5 s (4.4 to 4.9); long median 29.6 s (29.2 to 30.9). RAM used after the counted requests stayed at 65.5 to 65.8 GiB.

| Config | Counted | prompt_n | cache_n | prompt_ms | prompt tok/s | generated | decode tok/s | draft accepted % | total s | RAM used GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short | warm-up | 99 | 0 | 1,342.6 | 73.7 | 256 | 60.9 | 66.3 | 5.5 | 65.6 |
| long-38K | warm-up | 38,186 | 0 | 27,198.7 | 1,404.0 | 135 | 64.1 | 73.0 | 29.4 | 65.8 |
| short | yes | 97 | 0 | 1,072.2 | 90.5 | 256 | 66.2 | 57.5 | 4.9 | 65.5 |
| short | yes | 99 | 0 | 1,072.0 | 92.4 | 256 | 76.3 | 70.7 | 4.4 | 65.5 |
| short | yes | 97 | 0 | 1,074.0 | 90.3 | 256 | 74.1 | 63.5 | 4.5 | 65.5 |
| long-38K | yes | 38,182 | 0 | 27,279.1 | 1,399.7 | 135 | 59.4 | 71.6 | 29.6 | 65.8 |
| long-38K | yes | 38,184 | 0 | 27,273.1 | 1,400.1 | 256 | 73.1 | 73.9 | 30.9 | 65.8 |
| long-38K | yes | 38,185 | 0 | 27,268.7 | 1,400.3 | 135 | 72.3 | 69.2 | 29.2 | 65.7 |

Counted short runs hit the 256 cap (`length`). Two counted long runs stopped at 135 tokens and quoted the passphrase. The third hit the cap (`length`); its visible text was only the cut-off start `"amber-keel-2`, so the full passphrase is not in that answer. No request failed.

`tools/needle_bench.py`: not measured on IQ3_XXS either. TTFT, streaming, power, temperature, PCIe width, and VRAM during generation: not measured.

A separate report, [2026-10-06 RX 7900 XTX](../2026-10-06-community-rx-7900-xtx/README.md), used IQ3_S and a similar 38k log on a different PC, with context 262,144, KV int8, and `--prefill auto:32768`. This report does not replace that one.
