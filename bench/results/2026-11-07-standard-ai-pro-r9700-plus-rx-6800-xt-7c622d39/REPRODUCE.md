# How to reproduce

Machine: AMD Ryzen 7 7700X, 48 GB RAM, AMD Radeon AI PRO R9700 32 GB (gfx1201) + AMD Radeon RX 6800 XT 16 GB (gfx1030), WD_BLACK SN850X 4000 GB NVMe, Windows 11 x64.
Drivers: R9700 32.0.32015.7029 (2026-10-18), RX 6800 XT 32.0.21045.16010 (2026-10-16).

## 1. Build the engine (required: no official engine carries gfx1030 and gfx1201)

From a Strata checkout at `508227b52fc179c9a001b8c72cfe589359b8e39a` (includes the PR #4883 setup change as `6f901345`):

```
set STRATA_HIP_ARCHS=gfx1100;gfx1101;gfx1102;gfx1200;gfx1201;gfx1030;gfx1031;gfx1151
tools\hip\build_windows.bat
```

Produces `dist\strata-windows-x64-hip.zip` (sha256 `bba59f5e…0838dd`). Install with setup's `--prebuilt dist\`.

## 2. Install the model

unsloth UD-IQ3_XXS of Qwen3.8-Flash-Next (4 GGUF shards), converted to a native pack with `tools/iq_pack.py`; the PLE table is read from shard 4 via `--ple-gguf`. Expert routing profile at `data/expert-profile.bin`.

## 3. Serve


`strata-unsloth-ud-iq3_xxs-bench.json` (in this repo's root at submission time), started by the runner itself via `--profile unsloth-ud-iq3-xxs-bench`. Key flags as measured, not defaults:

```
--max-context 262144 --kv int8 --spec 4 --spec-min-p 0.5 --mtp <rt>
--resident-experts --resident-budget-gib 20 --prefill auto:16384 --layer-split auto
--vram-reserve-mib 1792 --vram-reserve-later-mib 3400 --host-core last --expert-cache auto
```

Key environment: `STRATA_RESIDENT_PIN=0` (pageable, not page-locked), `STRATA_HIP_PROMPT_F16=1`, `STRATA_PF_FUSED=1`, `STRATA_HIP_ADAPT_KERNEL_COPY=1`, `STRATA_PCIE_SLOT_GENS=4,5`, `STRATA_RDNA2_SGEMM=1`, `STRATA_HIP_WMMA=1`.

## 4. Benchmark

```
.venv\Scripts\python.exe strata.py bench run --profile unsloth-ud-iq3-xxs-bench ^
  --credential-env STRATA_API_KEY --quantization iq3_xxs ^
  --custom-pack "native-experts pack unsloth-ud-iq3_xxs; expert profile data/expert-profile.bin" ^
  --track custom --track-notes "engine 0.1.57 HIP self-built multi-arch (gfx1030+gfx1201); vram-reserve 1792/3400; prefill auto:16384; resident 20 GiB pageable; kv int8; MTP drafter" ^
  --build-options "hip, gfx1030+gfx1201 self-built multi-arch engine, setup from PR 4883" ^
  --background none
```

Run three times; each run starts its own fresh engine (see `run_order` in `provenance.json`).
