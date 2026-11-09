# Which model runs on which hardware

This page is built by `tools/capability_table.py` from [docs/data/capabilities.json](data/capabilities.json) and the package names of release v0.2.13 ([docs/data/release.json](data/release.json)). Do not edit it by hand.

- **Packaged**: the engine packages in the release for that hardware. Read from the release's file names.
- **Supported**: what these docs commit to. *untested* means a package exists and nothing is claimed.
- **Measured**: where numbers for that combination are published. Empty means none are published; it does not mean the combination fails.

App and Terminal 0.2.13, Control and engines 0.1.59. App, Control and engine versions differ on purpose.

## At a glance

| Model | NVIDIA on Windows | NVIDIA on Linux | NVIDIA DGX Spark | AMD on Windows | AMD on Linux | Intel Arc on Linux | Apple silicon |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [DeepSeekV4.1-Flash-1017](#deepseek41) | yes | yes | yes | yes | yes | no | yes |
| [Qwen3.8-Flash-Next](#qwen38) | yes | yes | yes | yes | yes | yes | yes |
| [Qwen3.6-35B-A3B](#qwen36) | yes | yes | untested | yes | yes | no | no |

<a id="deepseek41"></a>

## DeepSeekV4.1-Flash-1017

Model guide: [MODELS.md](MODELS.md#deepseek-v41-flash).

| Hardware | Packaged | Supported | Limits | Measured |
| --- | --- | --- | --- | --- |
| NVIDIA on Windows | `cuda-cuda13` | supported: RTX 30, 40 and 50 | The package carries RTX 40 code (sm_89); we have not run an RTX 40 card. | [README](../README.md) speed table (RTX 5070, RTX 3060); [DEEPSEEK41.md](DEEPSEEK41.md) |
| NVIDIA on Linux | `cuda-cuda13` | supported: RTX 30, 40 and 50 | The package carries RTX 40 code (sm_89); we have not run an RTX 40 card. | [DEEPSEEK41.md](DEEPSEEK41.md) |
| NVIDIA DGX Spark | `cuda-cuda13` | supported: GB10 | - | [DGX_SPARK_LINUX.md](DGX_SPARK_LINUX.md) (2.00 bpw) |
| AMD on Windows | `hip-rdna3` | supported: Ryzen AI Max / Strix Halo, gfx1151 | The package is built for gfx1151. Other AMD cards on Windows: nothing is claimed. | [STRIX_HALO.md](STRIX_HALO.md#deepseek-v41-flash) |
| AMD on Linux | `hip-rx9070` | supported: Radeon AI PRO R9700 | The RX 9070 (gfx1200) uses the same package and is untested. Ryzen AI Max (Strix Halo) on Linux has no DeepSeek package. | [README](../README.md) speed table (1 and 4 cards); [AMD_HIP.md](AMD_HIP.md) |
| Intel Arc on Linux | none | does not run | No package. DeepSeek does not run on Intel ([INTEL.md](INTEL.md)). | - |
| Apple silicon | `metal` | supported: 64 GB or more, macOS 15 or newer | Text; pictures only with an engine that carries the Metal picture reader (marker STRATA_DS4_METAL_VISION_V1, present in the 0.1.58 package). Nothing under 128 GB was run. | [DEEPSEEK_MAC.md](DEEPSEEK_MAC.md) (M5 Max, 128 GB) |

<a id="qwen38"></a>

## Qwen3.8-Flash-Next

Model guide: [MODELS.md](MODELS.md#qwen38-flash-next).

| Hardware | Packaged | Supported | Limits | Measured |
| --- | --- | --- | --- | --- |
| NVIDIA on Windows | `cuda-cuda12`, `cuda-cuda13` | supported | - | [README](../README.md) speed table; [DETAILS.md](DETAILS.md) |
| NVIDIA on Linux | `cuda-cuda12`, `cuda-cuda13` | supported | - | [DETAILS.md](DETAILS.md) |
| NVIDIA DGX Spark | `cuda-cuda13` | supported: GB10 | - | [DGX_SPARK_LINUX.md](DGX_SPARK_LINUX.md) |
| AMD on Windows | `hip-rdna2`, `hip-rdna3`, `hip-rx9070` | supported: RX 6000, 7000 and 9000, Ryzen AI Max | - | [STRIX_HALO_WINDOWS.md](STRIX_HALO_WINDOWS.md); [AMD_HIP.md](AMD_HIP.md) |
| AMD on Linux | `hip-halo`, `hip-rx6000`, `hip-rx7900`, `hip-rx9070` | supported: RX 6000, 7000 and 9000, Radeon AI PRO R9700, Ryzen AI Max | RX 6000 is validated differently from the newer cards ([AMD_HIP.md](AMD_HIP.md)). | [AMD_HIP.md](AMD_HIP.md); [STRIX_HALO.md](STRIX_HALO.md) |
| Intel Arc on Linux | `sycl` | supported: Arc A-series and B-series | Linux only (Ubuntu 24.04 or newer); the model is prepared for the card first, with per-card limits ([INTEL_ARC.md](INTEL_ARC.md)). | [INTEL_ARC.md](INTEL_ARC.md) |
| Apple silicon | `metal` | supported: M1 or newer, macOS 15 or newer | - | [MAC.md](MAC.md#measured-m5-max-128-gb) (M5 Max, 128 GB) |

<a id="qwen36"></a>

## Qwen3.6-35B-A3B

Model guide: [QWEN36.md](QWEN36.md).

| Hardware | Packaged | Supported | Limits | Measured |
| --- | --- | --- | --- | --- |
| NVIDIA on Windows | `cuda-cuda12`, `cuda-cuda13` | supported | - | [QWEN36.md](QWEN36.md) |
| NVIDIA on Linux | `cuda-cuda12`, `cuda-cuda13` | supported | - | [QWEN36.md](QWEN36.md) |
| NVIDIA DGX Spark | `cuda-cuda13` | package present, untested | The engine package is the same one; no page states a Qwen3.6 run on the Spark. | - |
| AMD on Windows | `hip-rdna2`, `hip-rdna3`, `hip-rx9070` | supported | - | [QWEN36.md](QWEN36.md) |
| AMD on Linux | `hip-halo`, `hip-rx6000`, `hip-rx7900`, `hip-rx9070` | supported | - | [QWEN36.md](QWEN36.md) |
| Intel Arc on Linux | none | does not run | The Intel Arc engine does not run this family ([QWEN36.md](QWEN36.md)). | - |
| Apple silicon | none | does not run | The Mac engine does not run this family ([QWEN36.md](QWEN36.md)). | - |
