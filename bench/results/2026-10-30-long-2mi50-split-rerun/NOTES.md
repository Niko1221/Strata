# Long prompts only: MI50 #0 prefill, MI50 #1 decode (rerun)

Not a standard-protocol run. In the earlier run of this setup the prefill engine exited on the last retrieval sanity check
and was not started again, so its long prompts were all read by the decode engine; this folder is a fresh engine start
running only the long prompts.

Contents: `engine-config.json` (the run config, paths redacted), `long-prompts/` (per-request records, summary,
stderr with two failed requests), `split-lines.txt`, `logs/server.log`.

Environment: `HIP_VISIBLE_DEVICES=0,1 STRATA_PIN_RESERVE_MIB=12288 HSA_OVERRIDE_GFX_VERSION=9.0.6 STRATA_GR_SPLIT=1
STRATA_GFX906_ATTN_QUERY_SWIZZLE=1 STRATA_GFX906_ATTN_REDUCE12=1`.

## Machine (not visible to the runner)

- Host: Ubuntu 22.04, kernel 6.8, ROCm 7.14 with amdgpu DKMS 6.10.5, NVIDIA driver 580 / CUDA 13.0. The engines ran in
  Ubuntu 24.04 containers, which is the OS the runner reports.
- CPU AMD Threadripper 1920X, 96 GB RAM (the OS reports 94.3 GiB), PCIe 3.0, model files on a SATA SSD.
- GPUs: 2x AMD Instinct MI50 32 GB (gfx906); NVIDIA RTX 3070 on a modified 16 GB board (8 GB from the factory), FP16/BF16
  issue rate raised with Zagitta/gpu-unlock, clocks locked to 1,500 to 2,100 MHz. MI50 clocks capped at 1,339 MHz and
  170 W by the host.
- Background: one request at a time; no other inference process on the GPUs during the run.

## Build

Strata 0.2.7 (`5e8a4cca`) with #3986 (gfx906 build fix), commit `93f866a7` as the runner reports; #3987 was not in
this build. How to rebuild both engines, with the configs and the launcher, is on
[`bench/0.2.7-split-gfx906`](https://github.com/MarcoTool/Strata/tree/bench/0.2.7-split-gfx906/bench/split-gfx906). Track `custom`; the runner's track note names the setup by an internal label,
described in words above.
