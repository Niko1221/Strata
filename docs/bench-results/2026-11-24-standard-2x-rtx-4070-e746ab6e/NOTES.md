# Notes: 2x RTX 4070, layer split, tuned production config

The `strata bench run` files are unedited. This file adds what the runner could not see.

## Corrections to the generated README
- **2 GPUs, not 1:** 2x RTX 4070 12 GB (Palit), layer split 25 (layers 0-24 on CUDA0, 25-47 on CUDA1). The runner lists only CUDA0.
- **Source build, not release:** a git worktree of v0.2.20 plus 2 local commits. `terminal/` has no `.git` and no `COMMIT` file, so the runner found no commit. That is the only reason for `nonconforming`.
- **CUDA:** built with toolkit 13.5 (sm_89). The 13.4 in the README is what the driver reports.

## Machine
- Ryzen 9 5950X; 4x 32 GB DDR4-3200; ASUS PRIME X570-PRO; both GPUs at PCIe 4.0 x8; models on a Crucial T500 1 TB NVMe.
- Ubuntu 24.04.5, kernel 7.0.0-43, NVIDIA 615.71.09.
- GPU OC on both cards: core +75 MHz, memory +1100 MHz (Afterburner scale). Power limit 200 W (stock). A third-party GPU unlock kernel module is loaded.
- This is a production server with one client. The engine was restarted just before the run; the engine log shows exactly the 22 benchmark requests.

## Build: v0.2.20 (engine 0.1.66) + 2 patches
1. Upstream PR #6714 (`fused_gr` Q8 TC kernels: no `__ldg` on shared memory). Without it, 0.1.66 fails "verify: layer 0 never rang" on UD-* GGUFs.
2. Thinking-loop guard (server side): if one token repeats 256 times inside the thinking, the thinking is closed and the model answers. It did not trigger in this run.

## Settings (full file: `strata-config.json`, paths shortened to `~`)
- Model: Unsloth UD-IQ4_XS v3.1 GGUF (4 shards, SHA-256 in `results.json`), Strata native pack, stock `expert-profile.bin`.
- Context 131072; `--kv int8`; `--kv-resident 32768`; `--expert-cache auto`; `--prefill auto` with `STRATA_NV_CHUNK=1` (chunks up to 32768).
- `--spec 3 --spec-min-p 0.70`, MTP draft head; `--pipeline-windows 2`; `STRATA_SPLIT_MTP_BATCH=1`; `STRATA_STAGE_TRIM=1`.
- `--vram-reserve-mib -57 --vram-reserve-later-mib 0`: VRAM filled to about 100-250 MiB free.
- `STRATA_PREFILL_CPU_SHARE=auto` (max 3072); `STRATA_ROUTE_TAIL_SKIP=0` (exact routing); `STRATA_ROUTE_PRIOR=0`; adaptive expert cache every request (80 swaps, decay 0.97, lag 2).
- **Output-changing:** refusal-direction projection (`--control-vector-scaled` 1.0, layers 4-44, `--cvec-mode project`, per-layer).
- Conversation cache 27 GiB in RAM (20 slots); vision encoder on the CPU (8 threads, idle during the run).

## Results (median, min-max, n=3)
| Case | Prompt tok/s | Decode tok/s | TTFT s |
| --- | ---: | ---: | ---: |
| short | n/a | 70.7 (69.2-82.6) | 0.79 |
| prompt-4k | 920.5 (919.9-920.6) | 79.7 (72.9-79.9) | 4.47 |
| prompt-16k | 2177.0 (2174.9-2177.4) | 75.1 (73.3-80.6) | 7.56 |

- Sanity checks 10/12. T1 and T2 failed: the model answered in text as if it had already called the tool (0 structured calls, `finish_reason: stop`). It was one sample each at temperature 1.0, and T3 and T4 passed.
