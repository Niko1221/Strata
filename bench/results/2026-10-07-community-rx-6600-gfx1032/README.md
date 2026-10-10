# Community benchmark: RX 6600 8 GB (gfx1032), EPYC 7232P, Linux, IQ3_XXS at 128K

Measured on 2026-10-06 and 2026-10-07 by [cryptedx](https://github.com/cryptedx).

An 8 GB RDNA2 card that setup does not list runs Flash-Next IQ3_XXS with 128K context as an always-on LAN server.
It is used every day as a coding-agent backend (prompts of 10K to 47K tokens). Since 2026-10-04 there has been no
service failure and no engine restart; the only start failures were during the 0.1.39 setup that night.

```text
GPU      AMD Radeon RX 6600 8 GB (gfx1032), headless, PCIe 4.0 x8 (engine probe 14.0 GB/s)
         runs as gfx1030 with HSA_OVERRIDE_GFX_VERSION=10.3.0 (see "gfx1032" below)
CPU      AMD EPYC 7232P, 8 cores / 16 threads, AVX2, no AVX-512
RAM      128 GB DDR4-3200 ECC RDIMM (8x 16 GB, all 8 channels), Supermicro H12SSL-NT
OS       Ubuntu Server 24.04.5, kernel 6.8.0-146 (in-box amdgpu, no DKMS)
ROCm     7.2.4 from repo.radeon.com (system packages, /opt/rocm), no hipBLASLt for gfx1030
Strata   v0.1.40.2 (e8ca9af), built with ./setup.sh --update --backend hip; earlier 0.1.38 / 0.1.39 / 0.1.40
Model    ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF IQ3_XXS (IQ3_S and IQ2_XS also installed)
Config   --max-context 131072 --kv int8 --kv-resident 32768 (KV streaming), MTP --spec 4 --spec-min-p 0.70,
         --pcie-frac 0.38 (calibrated), --expert-cache auto, conversation cache 4 slots / 8 GiB
Env      STRATA_HIP_PROMPT_F16=1, STRATA_PREFILL_CPU_SHARE=auto
```

## Results

Fixed 8.6K-token prompt, cold (new prefix every run, 0 reused tokens), then 512 tokens of output, no thinking.
Two runs per row, run one right after the other.

| Engine / setting | Prompt tok/s | Output tok/s | MTP accepted |
| --- | ---: | ---: | ---: |
| 0.1.39 (6f32ec0) | 171.8 / 171.7 | 22.6 / 23.3 | 67 / 71 % |
| 0.1.40 (82f46a8) | 171.7 / 171.5 | 23.8 / 23.5 | 70 / 69 % |
| 0.1.40 + `STRATA_HIP_PROMPT_F16=1` | 243.0 / 241.8 | 23.7 / 23.1 | 69 / 63 % |
| **0.1.40.2 (e8ca9af) + `STRATA_HIP_PROMPT_F16=1`** | **262.6 / 265.6** | **24.0 / 23.8** | 70 / 67 % |

- `STRATA_HIP_PROMPT_F16=1` is +41 % prompt speed on this card. 0.1.40.2 adds another +9 %, likely from the pre75
  attention kernel that is now the default on gfx103x. Together that is +54 % over 0.1.39.
- IQ3_S on 0.1.39, same test: prompt 102.9 / 103.3, output 20.2 / 20.9 tok/s. IQ3_XXS stays the default here.

**`STRATA_PREFILL_CPU_SHARE=auto`** (0.1.40.2, F16 on): A/B with a restart between arms, 2 rounds per arm. Each
session has a warm 8.6K prefix, then 9 follow-up chunks shaped like tool results, with 8-token answers. Medians:

| Chunk size (median tokens off / auto) | off | auto | Change |
| --- | ---: | ---: | ---: |
| under 500 (332 / 336) | 3.93 s | 2.64 s | -33 % |
| 500 to 1,023 (626 / 725) | 4.87 s | 3.67 s | -25 % (with more tokens) |
| 1,024 and up (1,097 / 1,338) | 5.39 s | 5.43 s | same (with more tokens) |

The upstream NVIDIA numbers carry over to this AMD card with an 8-core CPU.

**Quality** (own task set, 38 checkable German tasks x 2 runs, thinking on, server default sampling):
F16 alone 73/76, F16 + CPU share 71/76. The extra misses are one ambiguous date task and one off-by-one weekday.
I read both as sampling noise, but the sample is small.

## Memory on an 8 GB card

- Expert cache auto: 985 slots, 1.62 GiB VRAM; MTP draft layer 835 MiB; draft head 178 MiB
- 620 MiB VRAM free with everything loaded (KV streaming: a 32K-token VRAM cache of the attention-selected blocks, older ones included; the full KV copy stays in RAM)
- 39.97 GiB of experts loaded into a pinned RAM arena at about 11 GiB/s; 7 pool workers + host thread
- KV streaming in normal use: 98.7 to 99.8 % of block reads hit VRAM

## gfx1032

`setup.py` (`AMD_ARCHS`, `amd_gpus`) knows gfx1030 and gfx1031, not gfx1032. With `HSA_OVERRIDE_GFX_VERSION=10.3.0`
the gfx1030 build runs fine on the RX 6600. The only change needed is this local patch to `amd_gpus()`:

```diff
         arch = f"gfx{ver // 10000}{(ver // 100) % 100:x}{ver % 100:x}"
+        if arch in ("gfx1031", "gfx1032") and os.environ.get("HSA_OVERRIDE_GFX_VERSION") == "10.3.0":
+            arch = "gfx1030"
```

Upstream support for gfx1032 (RX 6600 / 6600 XT / 6650 XT) could be added the same way gfx1031 was.

## Limitations

- One machine; 2 runs per configuration; only an 8.6K prompt measured (no 32K / 128K ladder yet)
- No power or temperature numbers in this report
- IQ2_XS has not been benchmarked here
