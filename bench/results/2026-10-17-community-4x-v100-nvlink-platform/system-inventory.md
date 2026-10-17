# System inventory

```
captured_utc:        2026-10-16T23:28:34Z
host:                <HOSTNAME>
method:              scripts/collect-inventory.sh   (sha256 eb897bd238a7fe0466ae8655d3685b9be0f472be974b773382feba2c24ef1b65)
raw:                 raw/inventory/2026-10-16T232834Z/   raw/topology/2026-10-16T232834Z/
change to the system: none (read-only; no compilation or installation was performed)
units:               temperature degC, power W (nvidia-smi, W), clocks MHz, link speeds in GB/s as defined by
                     nvidia-smi (1 GB/s = 10^9 B/s), PCIe width in lanes, memory sizes in MiB (1 MiB = 2^20 B).
status:              measured, except items explicitly marked not checked / not possible.
limits:              the Strata commit SHA is unavailable on the measuring machine (no .git and no git); the
                     values were taken while the Strata engine was running (see §6) - this is a working
                     background, not an idle server.
```

## 1. Agreement with the stated configuration

| claim | check | result |
|---|---|---|
| 4x Tesla V100 SXM2 16 GB | `nvidia-smi -L`, `--query-gpu` | **confirmed**: 4x `Tesla V100-SXM2-16GB`, 16384 MiB per card |
| Volta, CC 7.0, SM70 | `--query-gpu=compute_cap`, `nvidia-smi -q` (`Product Architecture`) | **confirmed**: CC `7.0`, architecture `Volta` |
| NVLink 2.0 | `nvidia-smi nvlink --status`, `-c`, `-R` | **confirmed**: 6 active links per card, 25.781 GB/s per link, a full mesh (see §4) |
| PCIe Gen3, x8 seen earlier | `--query-gpu=pcie.link.*`, `nvidia-smi -q`, sysfs | **confirmed**: Gen3, width **x8** (device max x16, host max Gen5) |
| P2P about 38.8 GB/s | not measured at this step | **not measured here** (measured later, `p2p-results.md`) |
| the host uplink is limited to PCIe x8 | sysfs + `-q` | **confirmed at the link-configuration level**; its effect on Strata was not checked here (quantified later, §8) |
| Ubuntu 26.04.1 LTS | `/etc/os-release` | **confirmed**: `Ubuntu 26.04.1 LTS (Resolute Raccoon)` |
| driver 580.178.04 | `nvidia-smi`, `--query-gpu=driver_version` | **confirmed**: 580.178.04 |
| "CUDA 13.0" | `nvidia-smi` (header) vs `nvcc --version` | **clarified**: 13.0 is the CUDA version **supported by the driver**. The installed toolkit is **CUDA 12.8** (private, `<HOME>/strata185/cuda-12.8`), and that is what the working build was compiled with |
| the stack: vLLM, Strata, Qwen | processes, units, configs | **partially confirmed**: Strata runs (PID 32197). vLLM does **not** take part in the current run; there is a separate installation `<HOME>/1cat` and a disabled unit `vllm-1cat.service` |
| context up to 260 331 tokens | process arguments | **partial**: the working run uses `--max-context 262144`; the actual length of past tests was not checked at this step |

## 2. Hardware

### 2.1 Platform

| parameter | value | how it was obtained |
|---|---|---|
| OS | Ubuntu 26.04.1 LTS (Resolute Raccoon) | `cat /etc/os-release` |
| kernel | `7.0.0-38-generic` #38-Ubuntu SMP PREEMPT_DYNAMIC, x86_64 | `uname -a` |
| boot parameters | `ro quiet splash crashkernel=…`; there are **no** `nvidia.*`, `pci=`, `iommu` parameters | `cat /proc/cmdline` |
| CPU | 13th Gen Intel Core i9-13900H, 20 logical CPUs, 1 socket | `lscpu`, `/proc/cpuinfo` |
| CPU frequencies | 400-5400 MHz (rated) | `lscpu` |
| NUMA | 1 node (node0: CPU 0-19, 93972 MB) | `numactl --hardware`, `lscpu` |
| RAM | 91 GiB total, 81 GiB available at capture time, swap 8 GiB (376 MiB used) | `free -h`, `/proc/meminfo` |
| disk | `/dev/nvme0n1p2` 1.9 TiB, 554 GiB used, **1.2 TiB free** - room for benches and traces | `df -h` |
| host model / server platform | see §8 (filled in from PR #2525's `env.json`) | `dmidecode` needs root |

### 2.2 GPU (a snapshot while the Strata engine runs)

| # | model | UUID | PCI bus id | CC | VRAM | used | T, degC | power W (limit) | SM clock MHz (max) | ECC | ECC errors (agg/vol) | VBIOS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | Tesla V100-SXM2-16GB | `GPU-c4a285b6-4b3f-e33d-b215-96e0d6ea6f6e` | `00000000:03:00.0` | 7.0 | 16384 MiB | 15250 MiB | 40 | 67.29 / 300 | 1530 / 1530 | Enabled | 0 / 0 | 88.00.4F.00.09 |
| 1 | Tesla V100-SXM2-16GB | `GPU-856a23fb-2a0d-12bb-10af-7a31a071322d` | `00000000:04:00.0` | 7.0 | 16384 MiB | 15590 MiB | 40 | 54.20 / 300 | 1312 / 1530 | Enabled | 0 / 0 | 88.00.4F.00.09 |
| 2 | Tesla V100-SXM2-16GB | `GPU-121f2dc1-2feb-3d01-b0ff-57b876562bc7` | `00000000:05:00.0` | 7.0 | 16384 MiB | 15200 MiB | 40 | 68.55 / 300 | 1530 / 1530 | Enabled | 0 / 0 | 88.00.4F.00.09 |
| 3 | Tesla V100-SXM2-16GB | `GPU-df670e96-a76d-034a-8640-a3b6f6171696` | `00000000:06:00.0` | 7.0 | 16384 MiB | 16072 MiB | 40 | 67.59 / 300 | 1530 / 1530 | Enabled | 0 / 0 | 88.00.4F.00.09 |

Additional:

- `Persistence Mode: Enabled` on all cards; `nvidia-persistenced` is active.
- `compute_mode: Default` (not exclusive) - several processes can run on one card.
- InfoROM: `G503.0201.00.03`, OEM `1.1`.
- **Throttling:** `clocks_throttle_reasons.active = 0x0`, every reason (`hw_slowdown`, `sw_power_cap`, `hw_thermal_slowdown`, `gpu_idle`) is `Not Active`. No power or thermal performance limit at capture time.
- **ECC:** enabled on all cards; `uncorrected`/`corrected` (volatile and aggregate) = 0; `Retired Pages: None`, `Remapped Rows: N/A` (Volta does not support row remapping - expected, not a fault).
- GPU1 had SM clock 1312 MHz at capture time - an idle/frequency-transition state, not a limit: `clocks.max.sm = 1530`, no throttling active.
- 37-40 degC, 54-70 W per card at idle against a 300 W limit.

### 2.3 PCIe

| # | current gen | max gen (device) | host max | current width | device max width | NUMA node | cpulist | driver |
|---|---|---|---|---|---|---|---|---|
| 0 (`03:00.0`) | 3 | 3 | 5 | **8** | 16 | `-1` (not reported) | 0-19 | nvidia |
| 1 (`04:00.0`) | 3 | 3 | 5 | **8** | 16 | `-1` | 0-19 | nvidia |
| 2 (`05:00.0`) | 3 | 3 | 5 | **8** | 16 | `-1` | 0-19 | nvidia |
| 3 (`06:00.0`) | 3 | 3 | 5 | **8** | 16 | `-1` | 0-19 | nvidia |

Sources: `nvidia-smi --query-gpu=pcie.link.*`, `nvidia-smi -q` (the `GPU Link Info` section), `/sys/bus/pci/devices/*/current_link_*`.

Conclusion (a configuration fact, not a performance measurement): the links run at **Gen3 x8**, about 7.88 GB/s of usable bandwidth per card toward the host; the GPUs and the host themselves are capable of x16/Gen5. `Replays Since Reset: 0`, `Replay Number Rollovers: 0` - no PCIe retransmission errors recorded.

### 2.4 NVLink topology

`nvidia-smi topo -m`: every GPU pair is **`NV2`** (a bundle of 2 NVLink links). CPU affinity for all cards is CPU 0-19, NUMA affinity 0.

The physical connection map (`nvidia-smi nvlink -R`), 24 links in total (4x6), exactly 2 links per pair:

| GPU (bus) | link -> remote device |
|---|---|
| GPU0 `03:00.0` | L0->`05:00.0`:L3, L1->`05:00.0`:L2, L2->`04:00.0`:L0, L3->`04:00.0`:L1, L4->`06:00.0`:L4, L5->`06:00.0`:L5 |
| GPU1 `04:00.0` | L0->`03:00.0`:L2, L1->`03:00.0`:L3, L2->`06:00.0`:L1, L3->`06:00.0`:L0, L4->`05:00.0`:L4, L5->`05:00.0`:L5 |
| GPU2 `05:00.0` | L0->`06:00.0`:L2, L1->`06:00.0`:L3, L2->`03:00.0`:L1, L3->`03:00.0`:L0, L4->`04:00.0`:L4, L5->`04:00.0`:L5 |
| GPU3 `06:00.0` | L0->`04:00.0`:L3, L1->`04:00.0`:L2, L2->`05:00.0`:L0, L3->`05:00.0`:L1, L4->`03:00.0`:L4, L5->`03:00.0`:L5 |

| parameter | value | source |
|---|---|---|
| active links per card | 6 of 6 | `nvlink --status` |
| link speed | 25.781 GB/s (all 24) | `nvlink --status` |
| theoretical bandwidth per pair | 2 links x 25.781 = **51.56 GB/s in one direction** (Volta NVLink 2.0 is full duplex by direction) | calculated from `--status` |
| theoretical total per card | 6 x 25.781 = 154.7 GB/s one way | calculated |
| capabilities of each link | P2P: **true**, P2P atomics: **true**, access to system memory: **true**, atomics in system memory: **true**, SLI: false | `nvlink -c` (24/24 links) |
| topology | a direct mesh without NVSwitch: `Fabric State/Status: N/A`, `GPU Fabric GUID: N/A` | `nvidia-smi -q` |
| NVLink Enhanced (NVLE) | disabled, Firmware N/A | `nvlink -info` |
| link width / bandwidth mode | `Not supported` / `not supported` - the query is not supported on Volta | `nvlink -gLWidth`, `-gBwMode` |
| NVLink data counters (`Data Tx/Rx`) | **`N/A` - not available** | `nvlink -gt d`, `-gt r` |
| error counters (Replay / Recovery / CRC, including per lane) | all 0 | `nvlink -e`, `nvlink -ec` |

An important method limit: **NVLink traffic cannot be measured with `nvidia-smi` on this platform** (only link state and zero error counters). The moment the error counters were last cleared is unknown, so "0" must not be read as "no errors ever during operation".

## 3. Software

### 3.1 Driver and CUDA

| component | value | how it was obtained |
|---|---|---|
| NVIDIA driver | **580.178.04** | `nvidia-smi`, `--query-gpu=driver_version` |
| CUDA supported by the driver | **13.0** (the line in the `nvidia-smi` header) | `nvidia-smi` |
| CUDA toolkit actually used | **12.8, V12.8.93**, `release 12.8.r12.8`, build `cuda_12.8.r12.8/compiler.35583870_0` | `<HOME>/strata185/cuda-12.8/bin/nvcc --version` |
| toolkit location | `<HOME>/strata185/cuda-12.8` (private, **not** a system package) | `ls`, `BUILD.json` |
| `nvcc` in `PATH` | **none** (`nvcc`, `nvcc-12`, `nvcc-13` NOT FOUND) | `command -v` |
| system `/usr/local/cuda` | absent | directory search |
| compilers | system `gcc/g++ 15.2.0`; **private `gcc-14/g++-14 14.3.0`** in `<HOME>/strata185/gcc185/usr/bin` (the Strata build used gcc-14) | `gcc --version`, `gcc-14 --version`, `BUILD.json` |
| CMake / Ninja | **not** in `PATH`; present in Strata's venv: `cmake 4.4.3`, `ninja 1.13.2` | `command -v`, venv |
| profilers | `nsys` none; `ncu` and `compute-sanitizer` only in the private toolkit `<HOME>/strata185/cuda-12.8/bin`; `dcgmi` at `/usr/bin/dcgmi` (the `nvidia-dcgm` service is **inactive**), `dcgmproftester12` present | `command -v`, `systemctl is-active` |
| `git` | **not installed** (`dpkg -l git` - 0) | `command -v`, `dpkg -l` |
| Python | system `3.14.4` (`/usr/bin/python3`, no pip); Strata's venv `3.14.4` | `python3 -V` |
| packages in Strata's venv | numpy 2.5.3, cmake 4.4.3, ninja 1.13.2, pillow 12.3.0, psutil 7.2.2, pyyaml 6.0.3, regex, jinja2 3.1.6, requests 2.34.2, tqdm and others. **torch/vllm/triton/transformers are absent** | a `site-packages` listing, import attempts |
| PyTorch / vLLM in the working Strata stack | **not used** (Strata is its own CUDA engine) | imports in system and venv python: `ModuleNotFoundError` |
| a separate vLLM installation | `<HOME>/1cat` + the unit `vllm-1cat.service` (disabled). Not inventoried | `ls`, `systemctl list-unit-files` |
| local copies of third-party projects | `<HOME>/opt/v100-skinny` (CUDA kernels for V100), `<HOME>/1cat` (1Cat-vLLM) | `ls` |

### 3.2 Strata: version and the architecture targets of the binaries

| parameter | value | source |
|---|---|---|
| version | **0.1.44.1** | `CMakeLists.txt: project(strata VERSION 0.1.44.1)`, `engine-cuda12/BUILD.json: "version": "0.1.44.1"` |
| commit SHA | **not available on this machine**: `<HOME>/strata185/Strata` is a copy without `.git`, and `git` is not installed | `ls -a`, `dpkg -l`; the tag SHA is given in §8 |
| the working engine binary | `<HOME>/strata185/Strata/engine-cuda12/strata`, 71 186 856 B, mtime 2026-10-14 16:01 | `ls -la` |
| SHA-256 `strata` | `bdfec9f553d07c2b6017748802a614b48fb847efa9ec727cbe111e2ab2280de6` | `sha256sum` |
| SHA-256 `strata-vision` | `932969ee6308f30130917355ccd598823c3d2b974f8b819d0179fd9888687700` | `sha256sum` |
| build architecture targets | `"archs": [70]`, `"toolkit": 12`, `STRATA_EXPERIMENTAL_SM60=ON`, the private toolkit and gcc-14 | `engine-cuda12/BUILD.json` |
| actual binary content | **62 cubins and 62 PTX, all `sm_70`**; no other architecture | `cuobjdump --list-elf/--list-ptx` |
| `strata-vision` | contains **no** built-in CUDA modules (0 cubin / 0 PTX) | `cuobjdump` |
| CMake default | `CMAKE_CUDA_ARCHITECTURES` defaults to `120`; building below sm_75 is refused, sm_60/sm_70 are allowed only with `STRATA_EXPERIMENTAL_SM60`; the comments record that **CUDA 13 removed sm_60/sm_70** | `CMakeLists.txt` (lines 160-199) |

The presence of sm_70 code in the working engine is a **fact**; its sufficiency and correctness (whether every needed kernel is specialized, whether there are fallback paths) is the subject of `sm70-compatibility.md`.

### 3.3 The working run (for reproducing future measurements)

The active processes at inventory time:

- PID 32197: `<HOME>/strata185/Strata/engine-cuda12/strata --serve … --layer-split auto --max-context 262144 --kv int8 --spec 4 --mtp … --mmap-experts --vision --vram-reserve-mib 512 --pool-workers 12 …` (the full line: `raw/inventory/2026-10-16T232834Z/strata-process-env.txt`)
- PID 32185: `strata-vision --mmproj … --gpu --max-tokens 1024`
- VRAM used: 15234 / 15574 / 15184 / 16056 MiB on cards 0-3 - **only 73…945 MiB free per card**.
- Process environment: `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=0,1,2,3`, `STRATA_STAGE_TRIM=1`, `LD_LIBRARY_PATH=<HOME>/strata185/cuda-12.8/bin:…/lib64`. No other CUDA/NCCL/TORCH variables. Secret values are not written out.
- The systemd unit `strata-iq3-s.service` is `enabled` but **inactive (dead)** since 2026-10-15 19:09 local; the current process was started **manually** (agent-session variables are visible in its environment). The units `strata.service`, `vllm-1cat.service`, `nvidia-dcgm.service` are disabled.
- The run config: `<HOME>/strata185/Strata/strata-185-iq3_s.json` (its SHA-256 and content without secrets are in `raw/inventory/2026-10-16T232834Z/strata-run-config.txt`).
- Model: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S` (2 files, 54 817 524 224 B + 28 800 138 432 B), pack `<HOME>/strata185/Strata-data/packs/iq3_s`, MTP `<HOME>/strata185/Strata-data/mtp/rt`. Model checksums were **not taken at this step** (they are taken before benchmarks).

A practical consequence for the P2P work: **there is almost no free VRAM** (a minimum of 73 MiB on GPU3). Any 256 MiB P2P probe needs either the engine stopped or smaller buffers - this has to be agreed with the operator.

## 4. What could not be obtained at this step (honest gaps)

| gap | why | the minimum action to close it |
|---|---|---|
| the Strata commit SHA | no `.git` in the copy and no `git` on the system | install `git`, or download the release archive and record the SHA from its metadata |
| vLLM/PyTorch versions of the working stack | they are not in the Strata stack; vLLM is a separate installation | inventory `<HOME>/1cat` separately if it is in scope |
| the server model/platform (the SXM2 platform, its maker) | not requested at this step | `dmidecode` (needs root) - closed in §8 from PR #2525's data |
| NVLink traffic counters | not supported by the driver/Volta in `nvidia-smi` | DCGM (`dcgmi`, the service is off) or an NVML query through a custom probe; or proof through CUDA tracing and path comparison |
| `nsys` (Nsight Systems) | not installed | install by agreement, or use `ncu` from the private 12.8 toolkit |
| model checksums | not part of this step | `sha256sum` before the first benchmark |
| the "38.8 GB/s P2P" figure | not measured | the P2P work |

## 5. Summary of this step

The configuration is recorded completely except the Strata commit SHA. The key facts for further work:

1. **NVLink 2.0 is present and fully enabled**: 24 active links, a full mesh, P2P and P2P atomics supported on every pair. This means the absence of an NVLink effect in Strata **cannot** be explained by missing hardware or disabled links.
2. **The host-traffic bottleneck is confirmed configurationally**: PCIe Gen3 **x8** (about 7.88 GB/s) while x16/Gen5 is possible.
3. **NVLink counters are unavailable**, so the proof that "data goes over NVLink" has to be built from the CUDA API, tracing and control measurements.
4. **The working build already contains sm_70 code** (62 cubins + 62 PTX) and was built with CUDA 12.8 because CUDA 13 removed sm_70. This is a detail worth recording for anyone building for a V100: a V100 build needs a toolkit <= 12.x.
5. **There is almost no free VRAM** - the P2P method has to account for that.

## 6. The conditions under which the snapshot was taken

The capture was made with the Strata engine running (4 cards occupied, utilization 0-30 % at polling time, no throttling). This is "the background of a working server", not an idle stand. For bandwidth measurements the conditions must be recorded separately, with an explicit statement of whether the engine was running during the measurement (see `p2p-results.md`).

## 7. Snapshot log

| directory | what it is | status |
|---|---|---|
| `raw/inventory/2026-10-16T232506Z`, `raw/topology/…232506Z` | a first attempt: some fields (`-d PCIE`, `nvlink -e 0`) were unsupported | **the directory is not on disk** (checked: only 232551Z, 232721Z, 232834Z exist) - the attempt is documented only in the log, its output was not saved |
| `raw/inventory/2026-10-16T232551Z`, `raw/topology/…232551Z` | full topology and GPU, but without Strata artifacts | superseded |
| `raw/inventory/2026-10-16T232721Z`, `raw/topology/…232721Z` | complete except `--query-gpu` (an error on the `vbios.version` field) | superseded |
| **`raw/inventory/2026-10-16T232834Z`, `raw/topology/…2026-10-16T232834Z`** | **the authoritative snapshot this report is built from** | current |

---

## 8. Clarifications after auditing PR #2525 (2026-10-16T23:45Z)

The audit of existing materials closed part of §4's gaps. This report is **not rewritten** - below are only
clarifications to what is already written; the machine-readable version is `system-inventory.json` ->
`corrections_after_pr2525_audit`.

| item of this report | clarification | source |
|---|---|---|
| §4 "server model/platform not checked" | The host is a **Minisforum MS-01** (board AHWSA, BIOS AHWSA.1.22 of 2024-03-12); the GPU panel is a **Spring V100 SXM2 4-GPU, model V100-SXM2-NVLINK_D V2.10**; the PCIe switch is a **Broadcom PEX8748** (Gen3) with two external SFF-8654 channels, one cable connected; the MS-01 slot is electrically x8 (`/sys/bus/pci/devices/0000:00:01.0 max_link_width = 8`); the SXM2 cooling is a 360 mm AIO | PR #2525's `env.json` -> `host_platform`, its Hardware section |
| §4 "the Strata commit SHA is unavailable" | `v0.1.44.1` = **`296b59d2f73c8f04a805df29c2fa97655adace78`**, `v0.1.43` = **`7695859a…`**. Provenance: taken from release tags, **not** from a working tree (there is no `.git` on the machine) - a caveat to keep in any reference | PR #2525's `env.json` -> `strata_commit`, `strata_branch` |
| §2.4 "NVLink counters are unavailable" | The cause is established: DCGM 4.7.0 on this host has **no profiling module**, and in DCGM 4.x the NVLink byte counters go through **NVML Sampling & Reporting**, which exists in neither 580.178.04 nor 595.91.07 (checked by reading the symbol tables of both drivers). The check was repeated with root rights | `p2p-results.md` §5, `raw/nvlink/2026-10-16T234500Z/` |
| §2.3 "the host-traffic bottleneck is a configuration fact" | Quantified: 6.29 GB/s per card alone -> **1.56 GB/s per card with four simultaneous streams** (a 4:1 oversubscription measured), and during a long prefill the sum over four cards gives a median of 5.25 GB/s and 9.33 GB/s in the worst second | PR #2525 `interconnect-experiments/host-bw.txt`, `uplink-saturation.json` |
| §3.2 "commit SHA is a gap" | Partially closed: the SHAs of the tested versions are known (above); git is still needed only for matching against `1Cat-vLLM`/`v100-skinny` and for a stand pinned to a SHA | - |

No new measurements were made for this section: all clarifications come from PR #2525's materials.
