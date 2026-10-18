# Community benchmark on Apple M4 Pro

Measured on 2026-10-17 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: Apple M4 Pro, 37 GiB VRAM
- CPU: arm (12 cores, 12 threads); installed RAM: 48.0 GiB; storage: not measured
- OS: Darwin 27.0.0; GPU driver: not read; CUDA: not read
- Strata: project 0.1.45, commit 477ac13c; engine 0.1.45-metal
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-q2_0
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 26.0 (23.0-26.8) | 1.35 (1.35-1.52) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 247.4 (246.3-264.6) | not measured | 16.28 (15.23-16.36) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 280.4 (249.8-284.8) | not measured | 57.31 (56.42-64.32) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 280.2 (279.6-280.3) | not measured | 114.42 (114.40-114.67) |
| agent session, follow-up turns | 14513 (14512-14531) at the last turn | see below | up to 64 | 3 | see below | 24.4 (20.9-27.8) | 12.46 (6.57-14.54) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.7-68.7); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 13719 (13719-13719) new tokens of the files are read once); first turn (nothing cached) first token: 3.82 s (3.75-3.83).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 160.9 | 3.825 | 23.7 |
| 1 | 3750 | 627 | 3123 | 221.0 | 14.176 | 20.9 |
| 2 | 5283 | 3813 | 1470 | 203.8 | 7.259 | 24.4 |
| 3 | 8716 | 5346 | 3370 | 232.5 | 14.544 | 22.2 |
| 4 | 11599 | 8737 | 2862 | 230.6 | 12.455 | 22.5 |
| 5 | 14512 | 11613 | 2899 | 233.5 | 12.458 | 22.9 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 850 s.

## Contributor notes (outside the tool's knowledge)

- **Source build with local patches:** upstream HEAD (`477ac13c`, project 0.1.45) did not compile for the Metal engine on this Mac out of the box. Two classes of fixes were needed, all in code the PC/CUDA builds do not touch: (1) the Metal compat header (`metal/include/metal_compat/cuda_runtime.h`) was missing shims for API added by yesterday's peer-copy fix (#2666): `cudaMemcpyPeer`, `cudaDeviceDisablePeerAccess`, the `cudaDevAttrComputeCapabilityMajor/Minor` enum constants, `cudaDeviceGetStreamPriorityRange` and `cudaStreamCreateWithPriority`; (2) the generated `metal/src/engine/stubs.cpp` was stale against the headers (missing `x_f16` on `Gemm::bf16`, `f16img` on the `gr_norm` family, `gr_write_cvec_norm_rs` absent) and was regenerated with `metal/tools/gen_stubs.py` from a fresh link log, plus three manual one-liners (a runtime no-op for `fused_gr_persist_init`, `fused_gr_supported -> true`, destructor + `metadata_bytes` for `RemoteExpertOpt`, and Metal-capability probes returning false). With those, `ninja` in `build-metal/` links cleanly.
- **Streaming config:** 48 GB unified memory cannot hold the 46 GB Q2_0 pack resident; the engine streams experts from the SSD automatically (`STRATA_METAL_STREAM=1`, the `.experts` one-request-per-expert copy written by setup, `--mmap-experts`, expert cache 64 slots, `--spec 2` with the server defaulting the draft to the MTP layer, context fitted to the full 262,144). KV streaming keeps the whole window affordable (~13.7 KB/token).
- **Fidelity note:** authoritatively measured with the suite's fixed suite; a real-world hunting-course re-drive on the same install (agent-style long-prompt reading) was subjectively far faster than the resident dense 27B it replaced, consistent with the prefill numbers above.
- **Real-world use case:** replaced the resident oMLX-served dense Qwen3.8-27B as default for an interactive per-candidate "hunter course test" agent workload and delivered results several times faster than the 27B equivalent, at equivalent answer quality for that task.
