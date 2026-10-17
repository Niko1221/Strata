# Community benchmark on NVIDIA GeForce RTX 4090

Measured on 2026-10-16 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4090, 48 GiB VRAM
- CPU: 13th Gen Intel(R) Core(TM) i9-13900KS (24 cores, 32 threads); installed RAM: 127.7 GiB; storage: not measured
- OS: Windows 11 (10.0.22631); GPU driver: 617.42; CUDA: not read
- Strata: project 0.1.44.1, commit not a git checkout; engine 0.1.44.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata (NVFP4/GPTQ; served as `qwen3.8-flash-next-nvfp4`)
- Context: 524288 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0
- Server arguments (key ones): `--prefill auto:32768 --spec 6 --spec-min-p 0.7 --max-context 524288 --kv int8 --kv-resident 0 --kv-grow --pool-workers 15 --rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 262144 --rope-freq-base 10000000 --vram-reserve-mib 1792 --batch-mtp --mtp D:\Strata\data\mtp\rt`.
- Model files: `D:\Models\Maximilian228\Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata\` (pack, dense GGUF as --native/--native-dense-gguf, PLE FP8, token embedding); expert profile `D:\Strata\data\expert-profile.bin`; a local control vector (`--control-vector-scaled <local vector>:1.0 --control-vector-layer-range 8 33 --cvec-mode project --cvec-dir per-layer`).
- Environment: `STRATA_PREFILL_RING=384`, `STRATA_PREFILL_CPU_SHARE_MAX=4096`, `STRATA_GDN_CHUNKED=1`.
- Tuning note: the 32768-token prefill chunk, the 384-slot prompt ring, the 4096-token CPU-share limit, the 1792 MiB VRAM reserve, the elastic K/V (`--kv-resident 0 --kv-grow`) and the chunked GDN recurrence were tuned on this machine (A/B runs).

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 140.7 (104.6-188.8) | 0.19 (0.16-0.22) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 3697.0 (3473.3-3709.6) | not measured | 1.11 (1.10-1.18) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 5022.8 (4988.8-5098.9) | not measured | 3.22 (3.18-3.25) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 5240.0 (5204.0-5242.6) | not measured | 6.15 (6.15-6.20) |
| agent session, follow-up turns | 14624 (14598-14649) at the last turn | see below | up to 64 | 3 | see below | 144.1 (122.6-172.2) | 1.21 (0.78-1.39) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.9 % (68.8-68.9); earlier-turn tokens read again per session: 4 (3-73) (earlier turns that the cache did not hold; the 13729 (13728-13730) new tokens of the files are read once); first turn (nothing cached) first token: 0.64 s (0.64-0.66).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 661 | 0 | 661 | 1062.9 | 0.641 | 121.3 |
| 1 | 3797 | 674 | 3123 | 2368.6 | 1.34 | 127.8 |
| 2 | 5330 | 3861 | 1469 | 1819.4 | 0.829 | 122.6 |
| 3 | 8767 | 5387 | 3380 | 2476.6 | 1.388 | 128.1 |
| 4 | 11677 | 8815 | 2862 | 2165.7 | 1.349 | 137.8 |
| 5 | 14598 | 11699 | 2899 | 2351.9 | 1.26 | 136.5 |

Per-request data: `results.json`. Engine log lines: `engine-startup-lines.txt` (this run used a server that was already running; the file carries that server's start-up lines).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 71 s.

Environment note: on this zh-CN Windows host the standard tool needed `PYTHONUTF8=1` (the GBK locale default against the UTF-8 BOM of `CMakeLists.txt` in `project_version()`); raw run log: `bench-run-output.txt`.
