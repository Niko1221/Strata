# Community benchmark on NVIDIA RTX PRO 6000 Blackwell Workstation Edition

Measured on 2026-11-08 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Long-context recall was checked afterwards with `tools/needle_bench.py`: 6/6 found at 32k and 128k. The card runs under a 350 W power limit (of its 600 W maximum); no clock capture was taken during the runs.

## Hardware and software

- GPU and VRAM: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GiB VRAM, PCIe gen5 x16 (current = max)
- CPU: Intel(R) Core(TM) Ultra 7 265KF (20 cores, 20 threads); installed RAM: 127.6 GiB; model files on a Samsung SSD 990 PRO with Heatsink 4TB (NVMe)
- OS: Windows 10 Pro (10.0.19045); GPU driver: 596.72; CUDA (driver supports up to): 13.2; engine built for CUDA 13.0
- Strata: v0.2.12 plus a local logging branch (commit 3feea4f4; the local change adds JSONL run logs to the server, the engine is untouched); engine 0.1.58, official release binary (`strata-engine-0.1.58-windows-x86_64-cuda-cuda13.tar.gz`)
- GPU detail (after the run, near idle): power_limit_w=350.00 (user-set; factory max_limit is 600 W), power_draw_w=148.42, pcie=gen5 x16, pstate=P1, temp_c=53
- Background workloads: none on the GPU during the suite (only the Strata engine and its vision helper held GPU contexts); no other capture

## Model and configuration

- Model and quantization: qwen3.8-flash-next-nvfp4-nvfp4-BF16 — Qwen3.8-Flash-Next in NVFP4 (GPTQ), a custom Strata pack from the Hugging Face repository `Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata` (download revision not recorded)
- Pack files (sizes in bytes, sha256 re-verified on disk the same day):
  - `pack/experts.bin` 67,948,118,016 `325e5e0d85c262a8d8278daf73b20696ec877d9a3ff584e57391fa69ad10bc64`
  - `pack/dense.bin` 1,538,035,200 `974df6dcefcae85e3fd2c19fcc12eeaacaadde8261ab732d5af5af6ce2e25a7a`
  - `qwen-nvfp4-gptq-dense.gguf` 5,991,569,472 `abdf74336b53d8f7ca3d99947f1eac8f19b8a544966750127cb61e561b127e3b`
  - `ple-fp8.gguf` 51,200,246,176 `37f89b8ce5a3425932763257c40229ec02723ad2ab14b4bbfc26a486f4725c9c`
  - `token-embd-bf16.gguf` 1,271,398,656 `386203af9070c7b43acb24ded8922876721d97bbad1a709b25cb49d694a7f51d`
  - `expert-profile.bin` 196,632 `8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf` (locally produced expert profile, not from the repository)
- Vision encoder: on (`--vision`); MTP drafter: local directory `mtp/rt` (MTP files, source revision not recorded)
- Context: 262144 tokens; parallel slots: 2; KV int8, `--kv-resident 0`; expert cache auto with `--mmap-experts`; conversation cache 16,384 MiB x 4 slots (min 8,192 tokens); `--prefill auto`; `--spec 4 --spec-min-p 0.5`; thinking off for the suite; temperature 0

```text
strata.exe --pack pack --native qwen-nvfp4-gptq-dense.gguf --native-dense-gguf qwen-nvfp4-gptq-dense.gguf --ple-gguf ple-fp8.gguf --embd-gguf token-embd-bf16.gguf --expert-profile expert-profile.bin --expert-cache auto --mmap-experts --prefill auto --spec 4 --mtp rt --max-context 262144 --kv int8 --kv-resident 0 --vision --vram-reserve-mib 700 --spec-min-p 0.5 --session-save-reclaim --conversation-cache-mib 16384 --conversation-cache-slots 4 --conversation-cache-min-tokens 8192
```

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 228.1 (209.3-327.2) | 0.14 (0.14-0.16) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 4445.8 (4329.0-4474.0) | not measured | 0.94 (0.93-0.96) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 5053.8 (4895.9-5264.1) | not measured | 3.21 (3.08-3.31) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 5126.3 (5119.4-5130.3) | not measured | 6.30 (6.29-6.31) |
| agent session, follow-up turns | 13531 (13518-13587) at the last turn | see below | up to 64 | 3 | see below | 266.4 (243.7-321.5) | 0.82 (0.57-0.96) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.8 % (69.8-69.8); earlier-turn tokens read again per session: 4 (3-4) (earlier turns that the cache did not hold; the 12724 (12723-12724) new tokens of the files are read once); first turn (nothing cached) first token: 0.66 s (0.61-0.68).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 1037.5 | 0.61 | 200.0 |
| 1 | 3823 | 620 | 3203 | 3507.1 | 0.948 | 266.6 |
| 2 | 5356 | 3887 | 1469 | 2607.8 | 0.593 | 243.7 |
| 3 | 8600 | 5419 | 3181 | 3427.8 | 0.957 | 266.4 |
| 4 | 10820 | 8656 | 2164 | 3048.7 | 0.737 | 261.9 |
| 5 | 13587 | 10877 | 2710 | 3301.7 | 0.852 | 321.5 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

### Long-context recall (`tools/needle_bench.py`, same server, same day)

| Length | Depths 10/50/90 % | Actual prompt tokens | Notes |
| --- | --- | ---: | --- |
| 32k | 3/3 found | 31,634–31,635 | 6–7 s per request |
| 128k | 3/3 found | 123,305–123,307 | 17–22 s per request |

6 of 6 needles found; per-run data in `needles.json`. A needle test measures recall on those inputs, not overall model quality.

## Correctness and limitations

Checks: 6/6 needle recall at 32k and 128k (table above). No coding, tool-call or image-quality checks were run.

Not measured: throughput under concurrency (`--clients` was not run); vision; a fresh 128K prompt in the speed suite (the 128K numbers above are needle requests, not the fixed suite); decode throughput at long prompts (the suite caps fresh prompts at 24 new tokens); GPU clocks and power draw during the runs (only the near-idle reading above); RAM/VRAM usage was not captured during the suite. Total time of the suite: 59 s.

Comparability: engine 0.1.58 / Strata v0.2.12 changed server-side speed accounting (#5318, #5307, #4006 — admission time now counts); the suite uses the engine's own timings, but older community reports may differ in convention. Token counts differ a little between runs because the agent session feeds the model's own replies back. This report was produced on a local logging branch; the engine binary is the unmodified official release.
