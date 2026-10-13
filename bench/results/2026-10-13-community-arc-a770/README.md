# Community benchmark on one Intel Arc A770 16 GB (`xe`)

Measured on 2026-10-12/13 by aslater3. The SYCL port (`sycl/`) on an Arc A770 16 GB under the `xe` driver, Coder IQ1_M at a
131,072-token context, one request at a time through the engine's `--tokens-file` path (no server). It is a report of what
current `main` does on this card and of the A770 changes sent as pull requests, each measured against a same-day build
without it. Every A770 result in `docs/INTEL.md` so far is either an A750 on `i915` or a Battlemage card on `xe`; this is
an Alchemist card on `xe`.

## Hardware and software

- Intel Arc A770 16 GB (`8086:56a0`, DG2), PCIe 4.0 x16 (the engine's PCIe probe: 3.9 GB/s host to device); AMD Ryzen 7 3800X (Zen 2, 8 cores / 16 threads, AVX2, no AVX-512);
  30 GB usable DDR4; NVMe SSD. A desktop session, a browser and a few small services were running (about 7 GB of RAM).
- Ubuntu 26.04.1, kernel 7.0.0, `xe` driver (`xe.force_probe=56a0 i915.force_probe=!56a0`), compute runtime (NEO,
  `libze-intel-gpu1`) 26.05.37020.3, oneAPI DPC++ 2026.1, oneMKL 2026.1.
- Strata `a0116b27` (0.1.43.1), native JIT build (`cmake -S sycl -G Ninja -DCMAKE_BUILD_TYPE=Release`; no `AOT=`, as
  `docs/INTEL.md` says for the A-series), and the same tree with the pull requests below.
- Each run in its own transient systemd unit (`LimitMEMLOCK=infinity`, a 330-600 s watchdog). Swap off.

## Configuration

Coder IQ1_M (`Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M`, two shards) with its native pack, the PLE shard and the MTP layer:

```
--expert-profile <24-trace profile, see below> --expert-cache auto --resident-experts
--max-context 131072 --kv int8 --kv-resident 20480 --vram-reserve-mib 1200 --prefill-borrow --prefill auto
--spec 4 --spec-min-p 0.5 --adapt-every 8 --tokens-file <prompt> --greedy --max-new 256 (128 at 32K) --stats
env: STRATA_VERIFY_STEPPED=1 STRATA_RESIDENT_HEADROOM_GIB=5..7 UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
     STRATA_PARALLEL_ARGMAX=0 STRATA_LOOKAHEAD=0 ONEAPI_DEVICE_SELECTOR=level_zero:0
```

Prompts: an 8K code prompt (7,987 tokens) and a 32K one (31,680 tokens). Prompt speed is the engine's own line
(`prefill N tokens in K chunks, ... (X tok/s)`), decode is `--stats`' `decode` line. "ids" is a hash of the generated token ids.

### Why not setup's own `xe` config

`setup_intel.py` writes `--stream-experts` (a pinned RAM mirror of every expert the card does not hold) for an `xe` card.
On this 30 GB machine that does not fit Coder IQ1_M at 128K: the mirror needs ~23.5 GB pinned, 15.68 GB fit, and the engine
stops with `REFUSED: 2137 experts are neither in VRAM nor mirrored`. In the same run the 4.86 GiB expert-cache allocation
failed (`cudaMalloc(4.86 GiB) failed`, the cache retried at 1,891 slots) because `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1`
is not set by the wrapper. So this card runs `--resident-experts`: the experts are in RAM, the CPU computes the ones the card
does not hold, and the stepped verify window (`STRATA_VERIFY_STEPPED=1`, from #1713) serves them between graph segments.

## What `main` does on this card

1. **It does not start.** `xe` here has no `ext_intel_free_memory`; the free-memory query returns 0 and
   `--expert-cache auto` sizes the cache from 0 bytes: `expert cache auto: 0.00 GiB free ... -> 0 slots`, then
   `--spec needs the device residency table`, exit 2. Fixed by **#2190** (size from the DRM fdinfo when the driver says 0);
   every other number below includes it.
2. **Decode crawls: 5.4 tok/s.** The verify window's `pool` phase is 183-208 ms per round although the CPU pool call itself
   is 26 ms. The cause is the fused GR read copying its 1.2 KB by-value kernel argument into every work-item's private
   memory on DG2. Fixed by **#2193** (5.4 -> 22 tok/s, the same tokens). The 2x B60 report's `wait for rings 186.8 ms per
   round, about 3.9 ms per layer` may be worth re-checking with it.
3. **Decode intermittently never starts** (**#2194**): the host waits in `Verifier::run` on a queue that does not drain, RSS
   grows until the OOM killer. 6 of 7 default runs of `main` + #2190 hit it; with the changes below, roughly 1 run in 3.
   Runs that hung are left out of every table here; they are counted in #2194.
4. Every run, on every build including our 2026-10-09 one, ends with a segmentation fault after the results are printed (exit
   code 139). `gr_parity` reports one failure on `main` (`QFUSE: T=1 rep 0: not written`).

## Results

All rows: same day, 8K prompt unless noted, each change measured against the row it builds on. Output ids were identical to
`main` + #2190 in every run where the expert placement matched (8K `72b762d5b2a3`, 32K `eb3432fea981`).

| Build | Prompt tok/s (8K) | Decode tok/s (8K) | Kernel benches |
| --- | --- | --- | --- |
| `main` + #2190 | 461 | 5.37 (5.88, 5.93 with `STRATA_DOORBELL_CHECK=1` / `STRATA_LEND_MIRROR=0`) | |
| + #2193 GR_PICK, large GRF on Alchemist | 466 | 22.12, 22.35, 22.55, 22.65 | `gr_bench` 205-258 -> 119-184 us |
| + #2203 more GR work-groups | 507 | 23.24, 23.61, 23.62 | `gr_bench` -> 95-148 us |
| + #2207 wide MMVQ in large GRF | 472-502 | 25.99, 27.10, 27.49 (base 23.47-23.77) | Q6_K 2-8 columns 1.3-7.4x |
| + #2199 GDN Xe recurrence | 8K +3-5%; 32K 478-480 -> 511-518 (bench lines) | unchanged | `gdn_parity` 51.0 -> 24.5 ms (8K), 204-236 -> 110-113 ms (32K) |
| + #2215 one-token CPU gate/up | unchanged | unchanged (pool ~ -3%) | one token: IQ3_S -27%, IQ3_XXS -6% per expert |
| all of the above (one run) | 532 | 27.49 | |

All kernel changes are bit-identical (checksums / parity tests in each PR) and on by default on Alchemist only, with a switch.

Not sent as pull requests:
- **Spill-free routers, QSA select and grouped gate/up** (our `STRATA_SPILL_FREE`): bit-identical, but on `main` 32K prompt
  510-516 -> 512-528 tok/s and 8K decode 23.47 -> 23.49-23.64, within run-to-run spread. Branch `aslater3:a770/up-spill`.
- **Stager threads.** Running the engine directly, the GGUF-in-place stager uses 32 threads: 8K prompt 465-485 tok/s; with
  `STRATA_STAGER_THREADS` 6, 8 or 12: 524-537 (+11-13%), 32K 515-525 -> 577-585, the same ids. `strata-sycl.sh` already sets 12.
- **A lock on the prefill ring's release events** (`aslater3:a770/up-stager-race`): on our fork a 32K prompt with 6-8 stager
  threads aborted with `urEventWait must not be called for an internal event`; not reproduced on `main` in 6 runs.

## Measured on this card earlier and not taken (our fork, 2026-10-07/08)

Same card and model, our fork on top of `fb58e0db`; listed so nobody repeats them.

| Change | Result |
| --- | --- |
| DG2 XMX prompt attention | 0.13-0.88x speed and ~1e-3 error |
| Tiled HC read kernels | bit-identical, HC 1.72-1.97 -> 2.09-2.22 s (slower) |
| GDN state in SLM | no spills, bit-identical, ~4x slower |
| Key-head GDN | ~7x slower |
| Grouped oneMKL GEMM | ~2x faster at equal tokens per expert, 15-30% slower with real routing |
| 2-4 GPU queues, a larger staging ring | no gain |
| Whole-build large-GRF mode | ~2.5x slower (per-kernel only) |
| `--spec 3` / `5`, `--spec-min-p 0.4` / `0.6` | 3-4% slower than 4 / 0.5 |
| Level Zero settings (`UR_L0_USE_DRIVER_INORDER_LISTS`, counter-based events, immediate-append, `SYCL_UR_USE_LEVEL_ZERO_V2=0`) | within noise |
| Prompt chunks above 8,192 at 128K | the engine keeps 8,192 (not enough slots) |
| `--kv-resident` 40,960 / 65,536, `--vram-reserve-mib` below 1,200 | no gain |
| Q6_K MMVQ load-ahead and 2-4 rows per sub-group | 1.2-5.7x slower at 2-6 columns |
| CPU pool: single gather, row prefetch, unroll, paired gate/up, other task sizes | -22% to +0.8% |

A 24-trace expert profile (the 6 prompts of an earlier profile, 14 short prompts in 9 languages/tasks, 4 long source files) with `--adapt-every 8`
was used throughout; against a 6-prompt profile it gave +4.8% decode on an 8K code prompt and +14% on chat, -3 to -4% on two
short code prompts (fork, 2026-10-08). Not compared with 0.1.43's `expert-profile-sycl.bin`.

## Not tested

Setup's `--stream-experts` config on a machine with enough RAM; the server path; images; i915 on this card; AOT
(`dg2-g10` fails to build here in kernels that declare `double`); Windows.
