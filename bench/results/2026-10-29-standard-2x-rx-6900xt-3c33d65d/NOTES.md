# Notes for the 2026-10-29 runs

**Machine:** 2x AMD Radeon RX 6900 XT (gfx1030, 16 GB each, PCIe 4.0 x8 each, both at 245 W and -81 mV), Ryzen 5 5600X, 128 GB DDR4-3200, Ubuntu 26.04 (kernel 7.0.0-38), ROCm 10.0. No display attached. Model files on an NVMe SSD.

**Builds:**
- *Release package:* `strata-engine-0.1.52-linux-x86_64-hip-rx6000.tar.gz` from the v0.2.7 release, sha256 `d653dbd1...` as published, unpacked unchanged.
- *Source builds* (custom arms "P"/"K"): v0.2.7 (`5e8a4cca`) merged into a local patch set - #3381 (`STRATA_PROMPT_ATTN_RDNA2`), #3382 (`STRATA_PREFILL_CPU_SHARE_BATCH`), #3073 (`STRATA_SELECT_SIMT=2`), #3095 (`STRATA_PW_WITH_BATCH`) - with ggml from a llama.cpp branch carrying RDNA2 MMQ patches, setup's HIP build options plus `STRATA_PREFILL_MMQ`, and the rocBLAS 5.6.0 tuning table. The two-card production run adds #3821 (`STRATA_PREFILL_PIPE_FLOOR`) cherry-picked unchanged. The full patch set, the build script and configuration, and the A/B behind each step are in [dual-6900xt-flash-next-tuning](https://github.com/Vop3p/dual-6900xt-flash-next-tuning/blob/main/docs/linux-dual-6900xt.md) (the llama.cpp/ggml patches in [`patches/llama.cpp/`](https://github.com/Vop3p/dual-6900xt-flash-next-tuning/tree/main/patches/llama.cpp)).

**Two things a user upgrading on Linux may hit (both on this box, 2026-10-29):**
1. An engine installed by an earlier Terminal is not replaced when the Terminal is updated: `strata launch` from the v0.2.7 checkout reported "Engine 0.1.47 (hip) is installed" and ran it. `strata update` then fetched 0.1.52 (`rx6000`, 200 MB) and test-started it fine. The default-track runs above are on 0.1.52.
2. On 0.2.7, `strata launch --save-preset NAME` stores the benchmark profile as `launch-NAME`, so `strata bench run --profile NAME` reports "profile is not saved"; `--profile launch-NAME` works. (Also: the profile is written after the server reports healthy, so a script that stops the launch as soon as `/health` answers loses it.)

**The release package's rocBLAS** is 5.4.0 and the gfx1030 table shipped in `tools/hip` is for 5.6.0, so the release-package arms log `cannot open rocBLAS tuning file ... rocBLAS's default kernels run`, as on 0.2.2 (#3790). #3343 adds a 5.4.0 table.
