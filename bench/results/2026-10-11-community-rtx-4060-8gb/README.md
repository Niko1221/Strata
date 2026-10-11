# Community benchmark on RTX 4060 8 GB with 32 GB RAM

Measured on 2026-10-11 and 2026-10-12 by @a5372935. Coder IQ1_M and the original
Flash-Next IQ2_XS on an 8 GB GPU that also drives the desktop, with 31.9 GB of RAM
(below the 48 GB that `setup.py` suggests for IQ2_XS, so IQ2_XS ran in low-RAM mode).
Context is 32,768 tokens only; no 128K runs.

## Hardware and software

- GPU: NVIDIA GeForce RTX 4060 8 GB (sm_89), driving the monitor; PCIe 3.0 x8, 6.4 GB/s host to device (engine probe)
- CPU: Intel Core i7-8700, 6 cores / 12 threads, AVX2, no AVX-512
- RAM: 31.9 GB installed (DDR4); page file on a separate HDD
- Storage: models on an NVMe SSD (XPG GAMMIX S50 Core)
- OS: Windows 10 Pro 22H2 (19045); NVIDIA driver 610.88
- Strata `61b3fb5` (engine 0.1.42), official prebuilt Windows engine (`strata-windows-x64.zip`, SHA-256 verified), CUDA 13 runtime from the venv
- Background: desktop only; browsers, game launchers, Discord and Wallpaper Engine closed. The desktop holds about 700 MiB of VRAM before the engine starts.

## Model and configuration

| | Coder IQ1_M | Flash-Next IQ2_XS |
|---|---|---|
| Repository | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF` @ `5348543e0147355ac9cbcb031184a3546350988e` | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` @ `ed59f92082b1e93c0e96d60a8b11aab089b52f09` |
| GGUF | `Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-0000{1,2}-of-00002.gguf` | `Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-0000{1,2}-of-00002.gguf` |
| Expert placement | `--resident-experts`: 318 GPU slots (0.61 GiB) + about 18.5 GB locked RAM, rest from the file cache | `--mmap-experts` (low-RAM mode chosen by setup): 1,226–1,231 GPU slots (1.67 GiB), all other experts from the SSD through the file cache |
| Prompt chunk (auto) | 512 tokens (256 in a second session, see below) | 2,560 tokens |
| Expert profile | `data/expert-profile-coder.bin` | `data/expert-profile.bin` |

Both: `--max-context 32768 --kv int8 --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5`,
MTP draft layer from `Qwen/Qwen3.8-Flash-Next` @ `de4b8e4d43b917e7706784d8bb445c9af86a3540` (cjk draft vocabulary),
`--vision` with the BF16 mmproj on the CPU (6 threads), `--vram-reserve-mib 300`, and
`STRATA_RESIDENT_HEADROOM_GIB=6` through the config `env` key. The speed and soak requests send `reasoning_effort: "none"`;
the speed runs use `temperature` 0.6, everything else the server defaults. No calibration, no experimental speed projection.

The full run configurations are `strata-coder-iq1_m.json` and `strata-iq2_xs.json` in this folder.
With the default `--vram-reserve-mib 700`, or with vision on the GPU, Coder IQ1_M got 0 GPU expert slots and the engine exited during the startup check.

## Method

- Server: `python -m serve.server --engine strata --config <config> --port 8080`, one model at a time.
- `api_suite.py`: functional checks (OpenAI and Anthropic APIs, streaming, reasoning effort, JSON, tool call, cancel, context overflow, vision) and speed.
  - `speed_short`: an ~82-token prompt, `max_tokens` 256, streamed, 3 runs.
  - `speed_long`: ~9.7K tokens of `docs/DETAILS.md`, `max_tokens` 256, streamed, 3 runs. Each run starts with a random tag, so no prefix is reused (`cached` is 0 in every run).
  - Conversation-cache check: the same 4,479-token prompt sent twice.
- Needles: `tools/needle_bench.py --lengths 8k,28k --depths 10,50,90`. 28K is the largest that fits 32,768 with the tool's output cap; 32K was rejected with HTTP 400 because the prompt was 32,803 tokens.
- `soak.py`: 60 minutes of back-to-back requests rotating a short answer (`max_tokens` 200), a streamed summary of 8,000 characters of `docs/DETAILS.md` (`max_tokens` 300) and a short JSON answer (`json_object`, `max_tokens` 200). Every 10th request is cancelled after 20 streamed chunks. Hardware sampled every 30 s.
- Prompt and decode throughput come from the engine timing lines (`engine_log_*.txt`, prompt/answer text removed). TTFT and total latency are measured at the client and exclude model loading. Expert caches were warm from earlier requests in the same session.
- `runs.json` holds every run, including failures; its `fields` key documents the units.

## Results

### Speed (engine timings, 3 runs each, median and range)

| | Coder IQ1_M | Flash-Next IQ2_XS |
|---|---|---|
| Load to `/health` ready | 29.4–29.8 s (153 s on the very first load) | 20.8 s |
| Decode, ~82-token prompt | 15.5 (15.4–15.8) tok/s | 20.7 (13.4–24.1) tok/s |
| Prompt, ~9.7K fresh tokens | 134.9 (130.9–135.5) tok/s | 31.3 and 35.7 tok/s (2 of 3 runs; the third stalled) |
| TTFT, ~9.7K tokens | 72.7 (72.1–74.8) s | 276 and 335 s |
| Decode after the ~9.7K prompt | 16.8 (14.6–18.2) tok/s | 3.1 and 7.6 tok/s |
| ~28K fresh tokens (needle runs), client seconds per request | 212–215 s | 564–765 s (engine prompt 37.0–50.4 tok/s) |
| Same 4,479-token prompt again | TTFT 34.5 s → 0.53 s (4,472 reused) | TTFT 137 s → 0.94 s (4,472 reused) |
| Peak system RAM during the API suite | 31.8 of 31.9 GiB | 31.8 of 31.9 GiB |
| Peak VRAM during the API suite (incl. desktop) | 7,645 MiB | 7,769 MiB |

### Correctness checks

| | Coder IQ1_M | Flash-Next IQ2_XS |
|---|---|---|
| `api_suite.py` checks | 23 / 23 | 22 / 23 (one `speed_long` run stalled, see below) |
| Needles 8K | 3 / 3 | 3 / 3 |
| Needles 28K | 3 / 3 | 3 / 3 |
| Vision (CPU), three values read from a chart | 2 / 3 and 3 / 3 (+1.49% read as +1.69% once) | 3 / 3 on both APIs |

### 60-minute soak

| | Coder IQ1_M | Flash-Next IQ2_XS |
|---|---|---|
| Requests (OK) | 179 (168) | 74 (74) |
| Failures | 11 HTTP 502 `structured_output_failed`: the model repeated a phrase until the cap in a short Chinese JSON answer | none |
| Decode median, first 10 → last 10 requests | 17.1 → 16.7 tok/s (−2.3%) | 8.15 → 7.35 tok/s (−9.8%) |
| Server RSS | 25.6 GB at start, then 26.8–27.0 GB | 27.1–27.7 GB |
| System RAM | 30.3–31.6 GiB | 31.5–31.8 GiB |
| GPU temperature, max | 63 °C | 63 °C |

## Notes and limitations

- **One IQ2_XS stall.** During a ~9.7K-token prompt, the engine reported no progress for 60 s at token 9,714, with 357 MiB RAM available and 39.6 million page faults. The watchdog stopped and restarted the engine and later requests succeeded. The log at startup said "0 MiB of VRAM free with everything loaded - LOW: requests may stall; add --vram-reserve-mib 812". The stall dump (`strata-stall-44412.dmp`) is kept locally and can be attached on request.
- **VRAM headroom changes the prompt chunk on an 8 GB display GPU.** After a restart of Coder IQ1_M with about 70 MB less free VRAM, the automatic prompt chunk dropped from 512 to 256 tokens and ~9.7K-token prompt throughput fell from 131–136 to 76–78 tok/s (−43%). Decode was unchanged. These runs are under `second_session_prompt_chunk_256` in `runs.json`.
- IQ2_XS used the low-RAM mode because this machine has 31.9 GB of RAM. Prompt throughput with experts resident in RAM was not measured.
- Only one machine and one engine version. No 128K runs.
