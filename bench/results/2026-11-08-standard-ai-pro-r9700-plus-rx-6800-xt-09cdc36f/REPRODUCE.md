# How to reproduce

Machine: AMD Ryzen 7 7700X, 48 GB RAM, AMD Radeon AI PRO R9700 32 GB (gfx1201) + AMD Radeon RX 6800 XT 16 GB (gfx1030), WD_BLACK SN850X 4000 GB NVMe, Windows 11 x64.
Drivers: R9700 32.0.32015.7029 (2026-10-18), RX 6800 XT 32.0.21045.16010 (2026-10-16).

## 1. Build the engine (required: no official engine carries gfx1030 and gfx1201)

From a Strata checkout at `588b9d341155688fbe23bbcfc81b0bf2b096fed1` (branch `development`; carries the PR #4883 setup change as `9746cece` and the PR #5562 fused-layout fix as `588b9d34`):

```
set STRATA_HIP_ARCHS=gfx1100;gfx1101;gfx1102;gfx1200;gfx1201;gfx1030;gfx1031;gfx1151
tools\hip\build_windows.bat
```

Produces `dist\strata-windows-x64-hip.zip` (sha256 `73beee33…c4e16c3a5f` — see `provenance.json`). Install with setup's `--prebuilt dist\`.

PR #5562 is required for `STRATA_PF_FUSED=1` on this pack. Without it the run aborts on every prompt of 4084 tokens or more with `prefill: the fused layout's MoE buffers are too small for layer 13's expert path`, because the fused-layout buffer shrink asks only the formats, not every layer's path.

## 2. Install the model

unsloth UD-IQ4_XS of Qwen3.8-Flash-Next (4 GGUF shards, Hugging Face path `UD-IQ4_XS` at revision `814db05`), converted to a native pack with `tools/iq_pack.py`; the PLE table is read from shard 4 via `--ple-gguf`. Expert routing profile at `data/expert-profile.bin`.

## 3. Serve

`strata-unsloth-ud-iq4_xs-v31-bench.json` (a copy of the running `strata-unsloth-ud-iq4_xs-v31.json` with `host` set to `127.0.0.1`, a separate `log`), started by the runner itself via `--profile unsloth-ud-iq4-xs-v31-bench`. Key flags as measured, not defaults:

```
--max-context 262144 --kv k8v4 --spec 4 --spec-min-p 0.5 --mtp rt
--resident-experts --resident-budget-gib 18 --prefill auto:16384 --layer-split auto
--vram-reserve-mib 1792 --vram-reserve-later-mib 3400 --host-core last --expert-cache auto
```

The split lands as layers 0-12 on the RX 6800 XT and 13-47 on the R9700.

Key environment: `STRATA_PF_FUSED=1`, `STRATA_PF_FUSED_Q50=1` (the fused int8 expert kernels; Q5_0 down is opt-in on gfx12 and never taken on gfx1030), `STRATA_RESIDENT_PIN=0` (pageable, not page-locked), `STRATA_HIP_PROMPT_F16=1`, `STRATA_HIP_ADAPT_KERNEL_COPY=1`, `STRATA_PCIE_SLOT_GENS=4,5`, `STRATA_RDNA2_SGEMM=1`, `STRATA_HIP_WMMA=1`, `STRATA_SELECT_WMMA=1`, `STRATA_SSD_GBPS=3.5` (measured drive read speed; the planner otherwise assumes 1.5 GB/s on Windows), `STRATA_HIPBLASLT_TUNING=engine/tuning/gfx1201-hipblaslt-100500.txt`.

The fused path engaged: every `engine.log` contains `strata: prompt experts on the fused int8 kernels (STRATA_PF_FUSED=1, #136)`.

## 4. Benchmark

```
.venv\Scripts\python.exe strata.py bench run --profile unsloth-ud-iq4-xs-v31-bench ^
  --credential-env STRATA_API_KEY --quantization iq4_xs ^
  --model-revision 814db05 --tokenizer-revision 814db05 ^
  --custom-pack "native-experts pack unsloth-ud-iq4_xs-v31; expert profile data/expert-profile.bin" ^
  --track custom --track-notes "engine 0.1.58 HIP self-built multi-arch gfx1030+gfx1201 from source 588b9d34 (includes PR 5562, fused-layout shrink fix); vram-reserve 1792/3400; prefill auto:16384; resident 18 GiB pageable; kv k8v4; MTP spec 4; STRATA_SSD_GBPS=3.5; STRATA_PF_FUSED=1; STRATA_PF_FUSED_Q50=1" ^
  --build-options "hip, self-built multi-arch gfx1030+gfx1201 (arch set from BUILD.json), ROCm 10.2.0a20260930, hipBLASLt 100500, setup from PR 4883" ^
  --background none
```

Run three times; each run starts its own fresh engine (see `run_order` in `provenance.json`).
