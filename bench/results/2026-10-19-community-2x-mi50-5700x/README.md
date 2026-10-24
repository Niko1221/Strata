# Community benchmark: 2× AMD Instinct MI50 16 GB (gfx906), Ryzen 7 5700X

Measured on 2026-10-19 by a machine owner (reported here for the project). This is
**Strata v0.1.43.1** (`a0116b2`), the gfx906 HIP build, with the original
Flash-Next **IQ3_XXS** at the model's full **131,072-token context**.

The point of this report is not a single throughput number but a **tuning result**:
on this two-card gfx906 split, `--pipeline-windows 2` lifts decode by **+36%**
(49.2 → 66.9 tok/s) with prompt throughput unchanged, and `--trim-stage-weights`
adds another **+5%** on prompt reading. Both options are off by default, and
neither is mentioned in `docs/AMD_HIP.md` for gfx906.

## Hardware and software

- 2× AMD Instinct MI50 16 GB (Vega 20, gfx906, wave64). The VBIOS reports
  `1002:66af`, so `lspci` and the engine print "Radeon VII"; the cards are sold as
  MI50 16 GB. 16 GiB BAR each (Re-Size BAR on).
- ASUS ROG Strix X570-E Gaming, BIOS 5055; Ryzen 7 5700X (8C/16T, no AVX-512);
  64 GB DDR4-3600 (2×32 GB, FCLK 1:1) — see the RAM section below.
- Ubuntu 24.04.5, kernel `6.8.0-147-generic` (amdgpu from the kernel).
- ROCm from the community image `mixa3607/rocm-gfx906:7.14-complete`
  (rocBLAS/hipBLAS with gfx906 kernels; hipBLASLt is not used).
- Engine built in that image with
  `-DSTRATA_HIP_GFX906=ON -DCMAKE_HIP_ARCHITECTURES=gfx906`,
  clang from `/opt/rocm/llvm/bin`. Container runs with
  `--device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --ulimit memlock=-1`.
- Power limit 200 W per card (`power1_cap`, 80% of the 250 W default).
- Model on NVMe; pack and MTP built with `tools/iq_pack.py` / `tools/mtp_*.py`.

Note: **0.1.44+ does not build for gfx906** on this machine (separate issue): a
`cublas*` symbol error in `src/qwen35/q36_engine.cu` and a `cuda.h`/`cuda_bf16.h`
error in the `strata_mmq_w4a4` target. `v0.1.43.1` builds cleanly, so that is
what the measurements use.

## Model and configuration

`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, revision
`ed59f92082b1e93c0e96d60a8b11aab089b52f09`, IQ3_XXS (71 GB, two shards).

Base configuration (everything below is one change against it):

```
--pack packs/iq3_xxs --native <shard1> --ple-gguf <shard2>
--expert-profile data/expert-profile.bin --expert-cache auto
--prefill 2048 --mtp mtp/rt --max-context 131072
--kv int8 --kv-resident 32768 --layer-split 27
--spec 3 --spec-min-p 0.9 --vram-reserve-mib 1024
```

The engine reports 918 MiB of VRAM free with everything loaded; the prompt path
borrows cache slots across both cards.

## Method

Two measurements per configuration, both served through `serve.server`, greedy
(`temperature 0`), one request at a time:

- **decode**: a fresh 1,647-token prompt (a unique nonce in every run, so nothing
  is reused), `max_tokens 128`; the number is the engine's own
  `predicted_per_second`.
- **prompt**: the same request's `prompt_per_second`.
- **short chat**: "Explain in three sentences what a CPU cache does",
  `max_tokens 256`; note that its length varies (83–133 tokens), so it is noisier.

Each configuration: one discarded warm-up pass, then **7 long and 5 short runs**,
median reported. A configuration change means a full container restart and model
reload (38–42 s, so the VRAM tier is rebuilt from the same profile).

Reproduction data: [`results.json`](results.json) (every run, not just medians).

## Results

Medians of 7 long / 5 short runs; tok/s.

| Configuration | decode (1.6K prompt) | prompt read | short chat |
| --- | ---: | ---: | ---: |
| base | 49.2 | 353.1 | 51.5 |
| `STRATA_GFX906_ATTN_QUERY_SWIZZLE=1` + `STRATA_GFX906_ATTN_REDUCE12=1` | 49.4 | 356.1 | 53.4 |
| `--prefill 4096` | 48.9 | 353.3 | 51.6 |
| **`--pipeline-windows 2`** | **66.9** | 352.3 | **74.1** |
| **`--trim-stage-weights`** | 50.1 | **370.8** | 52.9 |
| `STRATA_ARENA_MMAP=1` | 49.7 | 352.1 | 51.4 |
| **`--pipeline-windows 2 --trim-stage-weights`** | **66–68** | **371** | **73–74** |
| `... --trim-stage-weights` + the two attn switches | 69.5 | 377.7 | 72.4 |

- `--pipeline-windows 2`: **decode +36%**, prompt unchanged. On a two-stage layer
  split it overlaps the verify windows (stage 0 computes the next window while
  stage 1 verifies the current one), which is exactly the decode path here.
- `--trim-stage-weights` (with the explicit `--layer-split 27`): **prompt +5%**,
  decode +2%. It frees dense weights of foreign layers and the cache grows.
- The two attn switches add a further ~1.5–2% (69.5 / 377.7), which is at this
  machine's run-to-run noise; not adopted because they change the attention
  summation order.
- `--prefill 4096` and `STRATA_ARENA_MMAP=1`: no measurable change (the latter
  also makes the model load 98 s instead of ~38 s).
- Combined result against the tuple measured before tuning: decode **48.0 → 66–68
  tok/s (+38%)**, prompt **352.5 → 371 tok/s (+5%)**.

## Concurrency (`--batch`)

Four requests fired at once, 128 tokens each, aggregate tokens / wall time:

| Configuration | 1 request | 2 concurrent | 4 concurrent |
| --- | ---: | ---: | ---: |
| no `--batch` | 20.2 | 20.6 | 21.1 |
| `--batch 2` | 20.4 | 23.0 | 21.6 |
| `--batch 4` | 20.0 | 22.5 | 21.8 |

Aggregate throughput barely moves while per-request decode collapses (70 → 11–37
tok/s at `--batch 2`/`4`). For a single interactive user batching is a net loss on
this machine, so it is off.

## `--kv-resident` sweep

Short context (the 1.6K prompt): `0` (all K/V in VRAM), `20480`, `32768`,
`49152`, `65536` — decode 65.4–67.3 tok/s and prompt 370.9–371.4 tok/s for every
value, i.e. inside the run-to-run noise. At 1.6K tokens nothing streams at any
setting, so the only effect is VRAM taken away from the expert cache.

Deep context (~31.3K prompt tokens, 64 tokens out):

| `--kv-resident` | prompt tok/s | decode tok/s |
| --- | ---: | ---: |
| 0 (all in VRAM) | 712.7 | 50.9 |
| 20480 | 706.0 | 51.4 (4 runs: 50.9 / 52.0 / 51.4 / 73.7) |
| 32768 | 704.5 | 53.2 (3 runs: 51.8 / 73.7 / 53.2) |
| 65536 | 704.1 | 52.5 |

A 73.7 outlier appeared once per configuration and did not reproduce; it is MTP
draft acceptance, not the KV setting. Net effect: **no measurable difference**, so
the default 32768 stays.

Worth noting separately: prompt reading at 31K tokens runs at **~700 tok/s**, about
twice the 371 tok/s measured with a 1.6K prompt — the chunked prefill path is more
efficient on long prompts than on short ones.

## RAM speed: 2400 → 3600 MT/s

The same machine, same configuration, JEDEC 2400 vs DOCP 3600 (FCLK 1:1):
decode 48.0 → 49.1 tok/s (**+2.3%**), prompt 352.5 → 353.3 (**unchanged**). The
CPU-side expert pool is only part of the work here, so memory bandwidth is not
the lever it is on a CPU-only setup.

## Stability and thermals

- 200 W cap per card; under sustained generation: edge 35–40 °C, junction 45–49 °C,
  memory 43–46 °C, 23–122 W per card (the cards never approach the cap).
- ~1.5 h of continuous benchmarking across 12 restarts: no errors, no faults, no
  OOM, `dmesg` clean (no MCE/EDAC reports at 3600 MT/s).
- 128K context with `--kv int8 --kv-resident 32768` holds the K/V in 0.77 GiB of
  pinned RAM and leaves ~1 GiB of VRAM for the expert cache.

## What this suggests for the project

1. `--pipeline-windows 2` deserves a mention for **two-GPU layer splits on
   gfx906** (and probably any two-stage HIP split): +36% decode for one flag.
2. `--trim-stage-weights` is a cheap +5% on prompt reading whenever an explicit
   `--layer-split` is used.
3. `--batch` on this workload is not worth it (one user, MTP on).
4. Fix the 0.1.44+ gfx906 build (separate issue) — without it, every MI50 owner
   who follows `docs/AMD_HIP.md` on a current checkout hits a wall.
