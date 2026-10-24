# Community benchmark on RX 7900 XT 20 GB (gfx1100, Linux), Unsloth UD-Q5_K_XL

Measured on 2026-10-15 by loeeeee (runs and write-up done with an AI coding agent). Strata v0.1.44.1 built from
source for gfx1100 on Linux with a system ROCm 7.2.3, serving Unsloth UD-Q5_K_XL at the full 262,144-token context.
It compares the configuration setup writes with two added settings, both measured the same day on the same build
and machine, and reports the screening runs behind them. Main limitation: one machine, shared with other guests,
one day; thinking off; no concurrency.

## Hardware and software

- GPU: AMD Radeon RX 7900 XT 20 GB (XFX, Navi 31, gfx1100), stock power limit and clocks; PCIe 4.0 x16 at every hop
  from the root port to the card (read from sysfs during the runs; the card's PCIe DPM level stayed at its top level,
  16 GT/s x16, for 100% of a 10-minute 1 Hz sample under load)
- CPU: AMD EPYC 7K62 48-core (96 threads, 12 NUMA nodes on the host). Strata runs in a Proxmox LXC container limited
  to 48 logical CPUs (spread over all nodes, not pinned to whole cores) and 128 GiB of memory; the host has more RAM
  and runs other guests, whose load was not controlled. The GPU is passed to the container as `/dev/dri` + `/dev/kfd`
  device nodes on the host's `amdgpu` driver (no VFIO)
- Storage: ZFS-backed container volume, not measured
- OS: NixOS 26.11 (container) on a Proxmox VE host, kernel 7.0.14-19-pve; driver: the kernel's amdgpu
- ROCm: 7.2.3 from nixpkgs, given to setup as a system ROCm through `ROCM_PATH` (clr, hipBLAS, hipBLASLt 1.2.2,
  rocBLAS, device libs; no TheRock wheels). Setup picked `tools/hip/gfx1100-hipblaslt-100202.txt` for that hipBLASLt
- Strata: tag v0.1.44.1 (`9791ea3`), HIP engine compiled by setup for gfx1100 (`--backend hip`). One local change
  to `setup.py` only, not to the engine: `check_shards` skips shards that hold no tensors, because setup refused the
  current UD-Q5_K_XL `00001-of-00006` file (10,946,618 bytes, sha256 `610922cd…`, identical to setup's own pin) as
  6 bytes short. That is the bug of #2346 / #2470 / #2587 / #2635, fixed the same way in #2505 / #2632

## Model and configuration

- Model: `unsloth/Qwen3.8-Flash-Next-GGUF`, `UD-Q5_K_XL`, six shards (Hugging Face commit `766911a`), installed by
  `./setup.sh --backend hip --family unsloth --model UD-Q5_K_XL --context 262144 --kv int8 --vision no --yes`;
  native pack and MTP draft layer built by setup; no vision encoder
- All 98 GB of experts in RAM (arena 93,812 MiB, page-locked; no huge pages: `vm.nr_hugepages=0`, THP `madvise`)
- Setup's configuration (the **setup default** run below): `--expert-cache auto --prefill auto --spec 4
  --spec-min-p 0.5 --mtp … --max-context 262144 --kv int8 --kv-resident 32768`, env
  `STRATA_HIPBLASLT_TUNING=…/gfx1100-hipblaslt-100202.txt`
- **Tuned** run: the same plus `--vram-reserve-mib 1024 --prefill auto:32768`. Nothing else changed
- Engine facts from the runs (`results.json` → `machine.engine_info`):

| | setup default | tuned |
| --- | ---: | ---: |
| VRAM free after start | 562 MiB | 888 MiB |
| GPU expert slots (cache) | 3,225 (12,306 MiB) | 3,140 (11,981 MiB) |
| pool workers | 40 | 40 |
| PCIe share (auto) | 0.55 | 0.55 |

With setup's configuration the server printed `WARNING: only 498 MiB of VRAM free at the start of this request
(below 512 MiB)` and suggested `--vram-reserve-mib 768`; the tuned run gives up 85 expert slots for the headroom.

## Method

`python tools/strata_bench.py --url http://127.0.0.1:8080 --runs 3 --long` against the running server (model
loading not timed), once per configuration, both on 2026-10-15. Each folder holds the tool's own `README.md` and
`results.json`; their "Settings: the install's own configuration, unchanged" line is the tool's default wording,
because it was pointed at an already-running server: `setup-default/` ran setup's configuration and `tuned/` the
tuned one. No engine-log excerpt, for the same reason.

Recall: `tools/needle_bench.py --lengths 32k,128k --depths 10,50,90` on the tuned configuration (`needles.json`).

The two settings were chosen from deterministic screening runs (`micro/mybench.py`, data in `micro/sweeps.json`):
four fixed coding prompts at temperature 0 with exactly 256 new tokens (engine decode tok/s, draft acceptance),
fresh ~4K and ~16K prompts (prompt tok/s), and on the 16K context a cached follow-up turn and a turn that appends
~2K tokens of tool output (client-side time to first token). Each arm started from setup's configuration with only
its own overrides, and the server was restarted per arm.

## Results

Same build, same machine, same day; medians with the range over the runs, from the two `results.json` files:

| Test | setup default | tuned | change |
| --- | --- | --- | ---: |
| short chat decode tok/s (n=9) | 17.2 (8.2-23.0) | 22.7 (18.2-31.3) | +32% |
| short chat first token s | 1.61 (1.34-5.62) | 1.47 (0.98-1.69) | -9% |
| 4K prompt tok/s, fresh (n=3) | 334.9 (233.3-888.1) | 576.7 (263.2-603.2) | +72% |
| 4K first token s | 12.05 (4.57-17.27) | 7.01 (6.74-15.33) | -42% |
| 16K prompt tok/s, fresh (n=3) | 564.6 (470.8-579.1) | 887.7 (766.7-1123.0) | +57% |
| 16K first token s | 31.30 (27.93-34.32) | 18.14 (14.35-21.00) | -42% |
| 32K prompt tok/s, fresh (n=3) | 1018.9 (1018.7-1028.5) | 1115.9 (1001.6-1116.4) | +10% |
| 32K first token s | 31.52 (31.23-31.53) | 28.79 (28.77-32.07) | -9% |
| agent session cache hit % (n=3) | 68.6 (68.6-68.7) | 68.8 (68.7-68.8) | +0.2 pt |
| agent follow-up first token s (n=15) | 7.17 (4.26-10.69) | 5.36 (4.43-12.94) | -25% |
| agent follow-up decode tok/s (n=15) | 15.1 (5.5-41.8) | 26.1 (11.9-43.2) | +73% |

Needles on the tuned configuration: 6 of 6 found (32,778-32,779 prompt tokens at depths 10/50/90 %, 32-39 s each;
125,044-125,046 prompt tokens at the same depths, 82-126 s each).

### Screening and soak (`micro/sweeps.json`)

Medians; decode over 8 samples per arm (2 reps) in sweep 1 and 16 (4 reps) in sweep 2, prefill and first-token
values over the reps. The first request after each restart is cold and is the lowest decode sample of an arm.

| Arm (over setup's configuration) | Decode tok/s | Draft accept | 4K tok/s | 16K tok/s | +2K tool turn, first token |
| --- | ---: | ---: | ---: | ---: | ---: |
| sweep 1: none (setup default) | 25.5 | 0.83 | 559 | 266 | not measured |
| sweep 1: `--vram-reserve-mib 1024` | 32.3 | 0.84 | 386 | 661 | not measured |
| sweep 1: + `STRATA_PREFILL_MMQ=1` | 31.1 | 0.84 | 342 | 547 | not measured |
| sweep 1: + MMQ + `--prefill auto:32768` | 34.7 | 0.85 | 422 | 825 | not measured |
| sweep 1: + that + `--pool-workers 47` | 31.6 | 0.84 | 253 | 493 | not measured |
| sweep 1: + that + `--adapt-every 0` | 20.8 | 0.84 | 384 | 494 | not measured |
| sweep 2: vram + MMQ | 26.4 | 0.84 | 267 | 528 | 5.59 s |
| sweep 2: vram + `--prefill auto:32768` (**kept**) | 34.8 | 0.83 | 434 | 829 | 4.62 s |
| sweep 2: vram + MMQ + `--prefill auto:32768` | 33.4 | 0.83 | 437 | 620 | 6.79 s |
| sweep 2: vram + `--prefill auto:16384` | 26.8 | 0.82 | 405 | 882 | 5.38 s (one 23.5 s) |
| soak, kept config, 8 rounds without restarts | 35.1 (24.4-42.9) | 0.84 | 594 | 975 | 4.70 s (4.31-6.42) |

What this says on this machine, with these few runs:

- `--vram-reserve-mib 1024` is the largest single effect (decode +27% in sweep 1; 7 of the 8 same-prompt samples
  higher).
- `--prefill auto:32768` (the setup tip for 128 GB machines) gave the best 16K prompts and agent tool turns.
- `STRATA_PREFILL_MMQ=1` was slower or no faster in every pairing here (ROCm 7.2.3, hipBLASLt 1.2.2, K-quant
  experts), unlike the IQ3_XXS ROCm 7.1 measurements in `docs/AMD_HIP_PERFORMANCE.md`; not kept.
- `--pool-workers 47` (all 48 container CPUs but one) and `--adapt-every 0` were slower; not kept.

### Failures

- Sweep 2's control arm (`--vram-reserve-mib 1024` only) lost its run: a 38-token prompt took 15 s, decode fell to
  0.6-1.4 tok/s, and the engine stopped with `Unrecoverable native verification failure: verify: timed out at
  layer 17; its GPU waits were released and the GPU finished (#267)`; the server restarted the engine on the next
  request. During a first, aborted screening attempt the first request after a server restart failed twice
  (`0 tokens in 31-32 s (error)`, about 35 s after `ready:`). No failure in the 8-round soak or in either suite run. The cause is not established; it looks like the
  low-VRAM-headroom hang of #2065 / #2477, but sweep 2's failure had 1 GiB reserved.

## Not tested

Thinking on; concurrency (`--clients`, `--batch`, `parallel`); vision; prompts longer than ~125K tokens; other quants;
`--calibrate`; power and clock capture; answer quality beyond the needle check. The host's other guests were not
stopped, and the container's 48 CPUs are not pinned to whole cores, so CPU contention may add noise (container CPU
pressure reached 43% at times during the soak).
