# `--shared-expert-arena`: page size and GPU queue stops on an R9700 (2026-10-13)

The measurements behind the `--shared-expert-arena` section of [DETAILS.md](../../../docs/DETAILS.md): how the memory
behind a host arena affects the CPU's reads, the host->device copies and, on AMD, the stops of the GPU queues when the
kernel compacts memory.

## Setup

- AMD Radeon AI PRO R9700 32 GB, PCIe 4.0 x16; Ryzen 9 3900 (Zen 2, one NUMA node), 128 GB DDR4; IOMMU off.
- Ubuntu 24.04, Linux 6.8.0, ROCm 7.14.1 (HIP 7.14.60850). THP `enabled=madvise`, `shmem_enabled=never`, defaults for
  `compact_unevictable_allowed` (1) and swap (8 GB file).
- `arena_probe.cpp` (this folder): maps 16 GiB the way an arena would be (`anon`: anonymous + `MADV_HUGEPAGE`, as the
  engine's default arena; `tmpfs PATH`: a `MAP_SHARED` file, the first process allocates and fills it, a second maps
  it), registers it with `hipHostRegister`, then measures:
  - the CPU: 12 threads read whole 3 MiB blocks at random places, one load per cache line (a warm-up pass, then three
    3 s passes);
  - host->device: 16 random 3 MiB copies per batch, and 256 MiB copies, 3 s each;
  - the stops: 15 s of batches of 16 random 3 MiB copies while `echo 1 > /proc/sys/vm/compact_memory` runs 4 s in;
    the batch latencies and the process's KFD `evicted_ms` (`/sys/class/kfd/kfd/proc/<pid>/stats_<gpu>/evicted_ms`).
- Build: `hipcc -O2 -std=c++17 --offload-arch=gfx1201 -pthread arena_probe.cpp -o arena_probe`; run e.g.
  `HIP_VISIBLE_DEVICES=1 ./arena_probe tmpfs /mnt/arena/a.bin 16 --register --hold 15` (tmpfs mounted
  `size=18G,huge=always,noswap`).

## Results (three rounds, the arms interleaved A, B, D)

| arm | on 2 MiB pages | `VmPin` | CPU reads, GB/s | host->device 3 MiB / 256 MiB, GB/s | queues stopped | worst batch | pages migrated by the compaction |
|---|---|---|---|---|---|---|---|
| A anonymous + THP | 16 of 16 GiB | 0 | 45.5 / 45.4 / 45.4 | 27.1-27.3 / 28.7 | 51 / 54 / 80 ms | 27.0 / 27.9 / 17.0 ms | 726 / 82 / 546 |
| B tmpfs `huge=always` | 16 of 16 GiB | 0 | 45.4 / 45.4 / 45.4 | 27.0-27.3 / 28.7 | 0 / 18 / 0 ms | 1.9 / 10.6 / 1.9 ms | 814 / 1,054 / 225 |
| D `/dev/shm` (4 KB pages) | 0 | 0 | 45.3 / 45.3 / 45.4 | 27.2-27.3 / 28.7 | 194 / 295 / 206 ms | 98.4 / 149.7 / 104.8 ms | 735 / 1,687 / 2,227 |

A median batch took 1.84-1.85 ms in every arm. A second process attaching to B's file: `MADV_POPULATE_READ` 0.11 s,
`hipHostRegister` 0.18 s, 0 wrong pages of 200,000 checked; `Shmem` grew by 16 GiB once.

Earlier single runs on the same machine (one compaction each, so only a direction):

- anonymous memory with 2 GiB of it still on 4 KB pages (THP fell short): 761 ms stopped, worst batch 350 ms; the same
  with `mlock` (all on 2 MiB pages): 65 ms; tmpfs + `mlock`: 110 ms. With every page huge, `mlock` added nothing visible.
- hugetlbfs (`vm.nr_overcommit_hugepages`): a file shared by two processes worked (populate 0.01 s); 63 ms stopped;
  anonymous `MAP_HUGETLB`: 57 ms.
- `hipHostMalloc` is a shared mapping of `/dev/zero` on 4 KB pages on this ROCm (`VmPin` 0): 653 ms stopped; its
  dma-buf export (`hipMemGetHandleForAddressRange`) returned "invalid argument".
- `hipMemCreate` with `hipMemLocationTypeHost` allocated VRAM, not host memory (`mem_info_vram_used` +2,246 MiB,
  `mem_info_gtt_used` +2 MiB for 2 GiB; with a full card it fails "out of memory").

## Reading it

- The page size did not change the CPU's reads (the memory's bandwidth bounds them) or the copies (no IOMMU here).
- It changed the stops: compaction moves 4 KB pages and leaves huge ones, `hipHostRegister` does not pin, and every
  moved page in a registered range stops the process's queues on every GPU while the driver maps it again. Where is
  the rest with every page huge (0-80 ms, more for anonymous memory than for tmpfs) from? Not measured further.

## Two engines with the engine itself (this PR)

UD-IQ4_XS (55.4 GiB of experts), `--spec 4 --mtp --expert-cache auto`, 512 greedy tokens per engine; the R9700 above
(x16) and a second R9700 on PCIe 3.0 x4, both 32 GB; the arena file on tmpfs `size=58G,huge=always,noswap` or on
hugetlbfs `pagesize=2M` (`vm.nr_overcommit_hugepages=29000`).

| arena file | x16 alone | x4 alone | both at once, run 1: x16 / x4 | run 2: x16 / x4 | queue stops during a forced compaction mid-decode, x16 / x4 |
|---|---|---|---|---|---|
| tmpfs | 63.3 tok/s (loads) | 42.2 (attaches) | 63.9 / 41.3 | 66.9 / 42.1 | 0 / 713 ms, 0 / 569 ms |
| hugetlbfs | 67.2 (loads) | - | 66.8 / 37.4 | 66.9 / 41.9 | 0 / 1,664 ms (+1,357 / 2,829 ms before it), 0 / 645 ms |

Arena checksum `92a4fc0d243d7402` in every run; each engine's 512 tokens the same alone and in the pair. RAM with both
running: `Shmem` 58.3 GB (tmpfs) or 28,384 surplus huge pages (hugetlbfs, back to 0 when the file was deleted);
`MemAvailable` 69 GB of 125. Loads from the file cache: 3.4 GiB/s into tmpfs, 4.1 GiB/s into hugetlbfs.

## Not tested

NVIDIA (`cudaHostRegister` pins there, so the stops should not exist), other ROCm versions, a fragmented machine where
`huge=always` falls short (the engine reports `N of M GiB on 2 MiB pages` and tries `MADV_COLLAPSE`).
