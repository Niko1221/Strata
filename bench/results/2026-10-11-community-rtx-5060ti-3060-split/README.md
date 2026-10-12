# Community benchmark: RTX 5060 Ti 16 GB + RTX 3060 12 GB, unequal layer split

Measured on 2026-10-11/12 (one continuous evening session, 22:39-00:46) by
@mmbeanz. An unequal 16+12 GB NVIDIA pair with **unequal weak PCIe links**
(gen5 x8 + gen3 x4, measured under load), answering `docs/TEST_REQUESTS.md`
row 1 (two NVIDIA cards of different sizes, `--layer-split auto`), plus A/B
runs of the 0.1.43 `STRATA_ADAPT_OVERLAP` and `STRATA_ROUTE_TAIL_SKIP`
defaults on a **split** (the open threads measured them single-GPU), and a
needle suite at 32K/128K/256K. Main limitation: one rig, no soak, no
batching, no Windows/HIP.

Sub-reports per run: [`A/`](A/README.md) (baseline), [`B1/`](B1/README.md)
(`auto`), [`B1c/`](B1c/README.md) (explicit `6`), [`B1d/`](B1d/README.md)
(explicit `42`), [`C/`](C/README.md) (overlap off), [`D/`](D/README.md)
(tail-skip off), [`needles/`](needles/README.md). What we edited in the raw
artifacts and why: [`NOTES.md`](NOTES.md). Artifact hashes:
[`sha256sums.txt`](sha256sums.txt).

## Hardware and software

Hand-written from `nvidia-smi`, `lspci -vv` **under load**, `dmidecode` and
`nvcc` on the host - do not use the auto-generated `machine` block (see
#2062 and NOTES.md).

- GPUs: **RTX 5060 Ti 16,311 MiB** on PCIe **5.0 x8** (measured H2D 28.2-28.7
  GB/s) + **RTX 3060 12,288 MiB** on PCIe **3.0 x4** (measured H2D 3.15-3.2
  GB/s), the 3060 behind the 600-series chipset; the board cannot negotiate
  better widths. Idle `lspci` shows gen1 (ASPM), so only loaded readings are
  meaningful.
- CPU: AMD Ryzen 9 7900X (12 cores / 24 threads). RAM: 64 GB DDR5-6400
  dual-channel (61.9 GiB visible to the container). Storage: Samsung 990 EVO
  Plus 4 TB NVMe, DRAM-less (HMB); serial redacted.
- OS: Linux 7.2.9-1-cachyos, bare metal; the engine runs in a
  **systemd-nspawn** container (shared host kernel: real links, real CPU
  topology). GPU driver 615.71.09; CUDA 13.4 (nvcc V13.4.92).
- Strata 0.1.43, engine built from upstream `7695859a` (tag v0.1.43); the
  install carries one local commit changing **only `setup.py` (pip -> uv)** -
  no engine code changes.
- Background workloads: none (dedicated inference box); no power-limit
  changes. Bench client on a second machine over LAN (RTT 0.25 ms; the 32K
  prompt number matched an on-box run to 0.06%).

## Model and configuration

- Model: Qwen3.8-Flash-Next **IQ3_S**, ISTA-DASLab GSQ-RCO pack (sha256 in
  `sha256sums.txt`), 512K context via yarn factor 2.0, `--vision`.
- Baseline (run A): `--layer-split 40` + `--trim-stage-weights`, `--pcie-frac
  0.00`, `--kv int8 --kv-resident 20480`, `--spec 4 --spec-min-p 0.70`,
  `--mtp rt`, draft vocab `en`, pinned `data/expert-profile.bin` (24,576
  ranked pairs) with **no** `--expert-profile-save` - the profile never
  mutates between runs. Exact launch lines are in each run's
  `strata-iq3_s.log` and `config.json`.
- Boot facts (`GET /metrics`): `expert_slots 8842` (17,910 MiB) across both
  cards, `arena_mib 47962`, `pool_workers 11`, `vram_free_mib 324`.
- `--remote-expert-opt` is present but **inert** with a layer split (it feeds
  the helper-cache path only, #578).

## Method

`tools/strata_bench.py --long --runs 3` against our own server (we own the
lifecycle), thinking off, fixed output caps, 0 failures in all six runs.
`/metrics?requests=all` snapshotted before and after each run for the
per-request `hit_rate`, `pcie_share`, `decode_tok_s` (n=40 per run). Between
configs: full engine restart, same pinned profile, warm expert cache from the
same boot profile. One continuous session; see NOTES.md.

## Results

Medians with min-max over 3 runs; prompt tok/s and decode tok/s are the
engine's own timings, TTFT is the client's.

| run | change vs A | prompt 4K | prompt 16K | prompt 32K | TTFT 32K | chat decode | agent decode | hit_rate (engine, decode) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | baseline, K=40 | 868 (866-875) | 1,413 (1,412-1,415) | 1,721 (1,720-1,722) | 18.7 s | 74.4 (73.2-92.9) | 80.7 | 0.859 |
| B1 | `--layer-split auto` (chose K=3) | 836 | 1,227 | **1,291 (-25.0%)** | 24.9 s (+33%) | 66.6 (-10%) | 73.3 | 0.865 |
| B1c | explicit `--layer-split 6` | 264 | 346 | 408 (-76%) | 78.6 s | 41.4 | 38.6 | 0.669 |
| B1d | explicit `--layer-split 42` | 889 | 1,397 | 1,683 (-2.2%) | 19.1 s | 76.2 | 80.3 | 0.818 |
| C | `STRATA_ADAPT_OVERLAP=0` | 868 | 1,413 | 1,723 (+0.1%) | 18.6 s | 75.3 | 83.6 | 0.839 (mean) |
| D | `STRATA_ROUTE_TAIL_SKIP=0` | 871 | 1,411 | 1,723 (+0.1%) | 18.6 s | 72.0 | 76.0 | 0.785 (see finding 4) |

Memory: boot snapshots only (`vram_free_mib 324` on the 3060 with everything
loaded, arena 46.8 GiB of 61.9 GiB); peak RSS/VRAM during inference not
measured - a soak with sampling is planned as a follow-up.

## Findings

1. **`auto` on an unequal pair with unequal links (B1).** The probes are
   right (3.2 / 28.7 GB/s -> miss cost x6.26 / x1.00, matching our
   `h2d_bench`), but the choice costs 25% of 32K prompt throughput: K=3
   strands 3,548 MiB on the 3060 and puts 45 layers on the 5060 Ti, which
   clamps the prompt chunk to 3,328 (#1616). The engine's own caveat ("prices
   decode, not the prompt") was exactly right. Related: #1238, #880, #1616.
2. **The suggested split is printed in the wrong frame (B1c vs B1d).** `auto`
   reorders cards so the faster one is last (#1352), but an explicit
   `--layer-split N` does not reorder - so following the engine's printed
   `--layer-split 6` literally put 42 layers on the slow 3060: -76% prompt,
   hit_rate 0.669, ~7 GiB stranded on the 16 GB card. The as-given-frame
   equivalent (`42`, run B1d) is a wash on speed with a significantly worse
   hit rate (0.818 vs 0.859, permutation p=0.024). **Question: should the
   suggestion be printed in the as-given frame, or should explicit splits
   inherit the reorder?**
3. **Hit rate is per-layer coverage, not held count.** A (8,842 experts held)
   and B1 (5,935) are statistically indistinguishable in hit rate (p=0.93)
   despite a 33% difference in held experts; no count-based coverage curve
   (old or new `H(f)`) survives this. What tracks is each stage caching its
   own layers' experts. Details and permutation tests in A/B1/B1c/B1d.
4. **`hit_rate` silently changes meaning with `STRATA_ROUTE_TAIL_SKIP`
   (D).** From our two logs: A's reported lookups/token (564.3) **exclude**
   the 98,217 tail-skipped misses; on a common denominator skip ON measures
   **0.752 vs 0.775** - the sign reverses. Worth a doc line for `/metrics`
   and Monitor consumers; and when the flag is off the log prints nothing at
   all, so the off state is only inferable from absence.
5. **Tail-skip on a split: ~+5%, not +10-20%, and it decays with context
   (D).** +9.3% at 4K, +4.8% at 16K, +1.0% at 32K (decode mean 71.10 vs
   67.58, p=0.20); prompt unaffected; draft acceptance identical. Keep the
   default ON here. The unpriced cost is quality ("answers differ slightly")
   - needles pass 9/9 with it on. Related: #2007 (a T10 split measured -1.4%
   for `=7`), #1964.
6. **`STRATA_ADAPT_OVERLAP` is null on this split (C).** 32K prompt +0.14%,
   decode inside run-to-run spread (p=0.39), hit_rate identical (p=0.92):
   the -21%/-26% single-GPU regressions (#2010, #1975) do not reproduce on
   an unequal two-card split with `--pcie-frac 0.00`. The flag's cost is
   topology-dependent - input for #2011's opt-in-everywhere proposal.
7. **Tool findings (measurements of the tool, not the engine):** #2062
   confirmed and extended - `gpu_slug()` drops the second card
   (`2x-rtx-5060-ti`), the remote `machine` block blends client fields (3.8
   GiB RAM, null driver), and the generated README cites
   `engine-log-excerpt.txt` in `--url` mode where no such file is written.
   Separately: `--expert-profile-save` writes on server QUIT, and
   `strata_bench.py` starts **and stops** its own server - a bench run
   silently overwrites the live profile with what it learned (we hit this;
   pin the profile and drop `--save`).

## Correctness and limitations

Needles: **9/9 found** at 32K/128K/256K x depths 10/50/90 on the daily
driver (cold boot), `needles/`. Not tested: long soak (planned follow-up),
batching/concurrency, vision latency, the mmap file tier, Windows, HIP,
single-card configs on this box, and the answer-quality cost of tail-skip
beyond needles. `hit_rate` is the engine's decode-phase expert-cache metric;
prefill routing is not covered. Permutation p-values treat requests as
independent, which the adaptive tier correlates - they are optimistic.


*Drafted with AI assistance; I reviewed every line and ran the measurements myself.*
