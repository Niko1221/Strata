# Community benchmark on 2x AMD Radeon RX 7900 XT

Measured on 2026-11-01 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: 2x AMD Radeon RX 7900 XT 20 GiB (40 GiB total VRAM, dual GPU layer split 22/26)
- CPU: AMD Ryzen 9 3900X 12-Core Processor (12 cores, 24 threads); installed RAM: 63.9 GiB; storage: WDC WD20EZRZ-22Z5HB0 HDD/SATA; HFS256GD9TNG-62A0A SSD/NVMe; WDC WD40EZRZ-22GXCB0 HDD/SATA; Samsung SSD 980 1TB SSD/NVMe
- OS: Windows 11 (10.0.26200); GPU driver: 32.0.23033.1002; CUDA (driver supports up to): not read
- Strata: project 0.1.55, commit c62d5e3a; engine 0.1.48 (prebuilt windows-x64 hip, commit 6f8800aefe27, ROCm 10.2.0a20260930, hipBLASLt 100500)
- Background workloads: not recorded (GPU power limit and clocks not read on Windows HIP)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s (Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S, 2 GGUF shards + native pack)
- Context: 196608 tokens (192K); parallel slots: 1; prefill chunk: 8192, prefill pipe: 1
- Settings: custom configuration (not setup defaults, see below); thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K / 32K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 41.2 (35.8-49.5) | 0.57 (0.51-0.64) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 403.9 (401.2-408.2) | not measured | 10.00 (9.90-10.05) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 798.2 (793.0-817.9) | not measured | 20.18 (19.67-20.29) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 886.1 (870.5-909.2) | not measured | 36.23 (35.31-36.88) |
| agent session, follow-up turns | 12936 (12924-12941) at the last turn | see below | up to 64 | 3 | see below | 54.0 (36.7-66.8) | 8.99 (8.66-9.46) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.3 % (69.1-69.3); earlier-turn tokens read again per session: 74 (73-144) (earlier turns that the cache did not hold; the 12064 (12063-12065) new tokens of the files are read once); first turn (nothing cached) first token: 7.19 s (7.12-7.20).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 86.3 | 7.12 | 28.2 |
| 1 | 3355 | 626 | 2729 | 304.6 | 9.004 | 42.6 |
| 2 | 4888 | 3348 | 1540 | 172.7 | 8.961 | 36.7 |
| 3 | 8132 | 4952 | 3180 | 347.4 | 9.193 | 44.6 |
| 4 | 10199 | 8194 | 2005 | 232.3 | 8.685 | 51.0 |
| 5 | 12941 | 10258 | 2683 | 305.1 | 8.851 | 51.7 |

Per-request data: `results.json`. Sanitized engine start-up log of this session: `engine-log-excerpt.txt`.

### Non-default settings and notes (added by the author)

- `tools/strata_bench.py` was run with `--url http://127.0.0.1:8080 --long` without `--config`, so `results.json` did not
  load `strata-iq3_s.json` or `engine/BUILD.json` (`config: {}`, `launch: null`, `engine_build: null`) and sliced `gpu` to
  GPU 0 (`20464` MiB, `#1985`). `engine-log-excerpt.txt` records the 2-GPU split of this session:
  `layer split across 2 GPUs: CUDA0, then CUDA1 (split 22)` (CUDA0 layers 0-21 with 6,472 expert slots / 11.56 GiB,
  CUDA1 layers 22-47 with 6,539 slots / 13.06 GiB, plus 29.03 GiB pageable resident RAM complement).
- PCIe host->device probe in `engine-log-excerpt.txt`: GPU 0 `27.6 GB/s` (best of 27.3, 27.3, 27.6, 27.6; `pcie_frac 0.55`),
  GPU 1 `6.5 GB/s` (best of 6.5, 6.4, 6.4, 6.4; `pcie_frac 0.18`).
- Engine flags beyond setup defaults: `--layer-split 22 --prefill 8192 --prefill-pipe 1 --pipeline-windows 0
  --max-context 196608 --kv q4_0 --spec 4 --spec-min-p 0.5 --mtp <dir>` (`engine_info` reports verify window `spec: 6`,
  `mtp_max: 4`, `lookup: 3`) `--resident-experts --trim-stage-weights --vram-reserve-mib 2624 --vram-reserve-later-mib 1400
  --vision`, and the experimental speed projection control vector (`--cvec-mode project --cvec-dir per-layer`, layers 4-44,
  scale 1.0).
- Environment: `STRATA_HIPBLASLT_TUNING` (`gfx1100-hipblaslt-100500.txt` shipped with the engine),
  `STRATA_HIP_ADAPT_KERNEL_COPY=1`, `STRATA_RESIDENT_PIN=0`.
- `--prefill 8192`: the engine warns at start-up that prompt chunks of 8,192 tokens or more crashed on Windows HIP on an
  RX 7900 XTX (#1630). No crash occurred during this run, but one session is not evidence that `8192` is safe.
- The working tree had one 4-line local edit in `serve/frontend.py` (image input: accept a base64 field that already
  starts with `data:`). It is not used by the text-only requests in this suite.
- The server was already loaded and had answered a few earlier requests in the same session (verify graphs captured,
  adaptive swaps active). No A/B comparison against setup defaults was run, so individual flag effects are not isolated
  here. Context was set to 196,608 tokens; the longest prompt in this suite is about 32K tokens.

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 423 s.
