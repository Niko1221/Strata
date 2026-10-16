# GTX 1080 Ti (sm_61): eight-knob screen, the adopted stack, long context 32K-250K with needles, and a one-hour soak

Measured on 2026-10-16 on the same rig as [#2642](https://github.com/Niko1221/Strata/pull/2642) and
[#2684](https://github.com/Niko1221/Strata/issues/2684). One day of A/B runs: eight engine knobs screened one at a
time, the winners stacked, then long-context behavior (32K/128K/250K prompts, KV streaming int8) with needle checks,
and a 60-minute back-to-back soak (an answer to the soak row of
[TEST_REQUESTS.md](../../../docs/TEST_REQUESTS.md)). Speed only; the correctness evidence is two needles and a fixed
arithmetic probe, not a quality suite. Main limitations: one round per knob (decode is noisy run-to-run on this rig,
±20%; verdicts rest on whether all three requests of an arm move together), 250K is one run per arm.

## Hardware and software

- GPU: NVIDIA GeForce GTX 1080 Ti 11 GiB (GP102, sm_61, PCIe negotiates Gen3 x8 here), driving the desktop; CPU:
  2x Intel Xeon E5-2696 v4 (44c/88t, no AVX-512), 125.8 GiB RAM, NVMe SSD
- OS/driver/toolkit: CachyOS Linux LTS 6.18, driver 580.178.04, CUDA 12.8 (cuBLAS 12.8.4.1)
- Strata: engine 0.1.44.1 source build (`-DCMAKE_CUDA_ARCHITECTURES=61 -DSTRATA_EXPERIMENTAL_SM60=ON`), same binary
  in every arm; clock cap −200 MHz (1733 MHz under load); every arm started at 57-58 C
- Both arms of every pair bound to the GPU's NUMA node (`numactl --preferred=1 --cpunodebind=1`, per #2684) through
  a root-level cpuset cgroup (`numa-cgroup-launch.sh`); one instance at a time, arms alternated

## Model and configuration

Qwen3.8-Flash-Next IQ3_S (setup's pack, native experts, shipped expert profile and MTP head), `--max-context 262144
--kv int8 --kv-resident 32768 --spec 4 --mtp ... --expert-cache auto --prefill auto` (engine chose a 4,608-token
prompt chunk). The base arm of the screen is this config with `STRATA_DENSE_F32=1` (measured in
[#2695](https://github.com/Niko1221/Strata/pull/2695) the same day); each variant changes exactly one thing.

## Method

`bench.py` of [2026-10-12-p40-sm61-kernels](../2026-10-12-p40-sm61-kernels/README.md): a ~565-token prompt + 200
tokens out and a ~7,047-token + 60, three requests each, greedy, fresh nonce per request, engine's own timings
(`arm-run.sh`, `pair-run.sh`). A fixed arithmetic probe (17×23+41×19) runs in every arm: **all 20 arms answered the
same (1158)**. Long context: `longbench.py` (32K/128K/250K fresh prompts, 60 tokens out) and
`tools/needle_bench.py --lengths 128k,250k --depths 50`. Soak: `soak.py`, alternating short/long for 60 minutes.

## Results: the eight knobs (base → variant; prompt short/long tok/s; decode short/long tok/s)

| Knob | prompt | decode | verdict |
| --- | --- | --- | --- |
| `--mtp-hnorm stream` | 266/379 → 259/378 (same) | 27.7/25.2 → 35.4/32.3 (**+28%/+28%**, arms disjoint) | **adopted** |
| `--adapt-async 1` | 260/381 → 258/380 | 31.6/31.6 → 33.8/32.1 (+7%/+2%) | weak, not adopted |
| `--aux-cpus auto` | 258/380 → 256/386 | 34.8/31.0 → 26.7/26.2 (**−23%/−15%**, 3 of 3 down) | rejected |
| `--kv k8v4` (was int8) | 259/378 → 260/380 | 31.0/28.3 → 31.3/31.3 (same) | not adopted (precision change, no gain) |
| `--ple-io ram` | 258/380 → 264/378 | 30.6/27.3 → 33.3/30.5 (**+9%/+12%**) | **adopted** (mlock refused by `ulimit -l`; pages faulted in, 28.8 GB table loads in 8.2 s) |
| `--kv-grow` + `--expert-profile-save` | 262/380 → 256/381 | 25.8/28.5 → 25.6/24.3 (same) | same; profile-save kept, `--kv-grow` dropped (long-context screen below found it negative too) |
| no `--pcie-frac` (engine auto = 0.32; base had 0.20) | 257/384 → 256/384 | 29.2/28.4 → 32.7/34.0 (**+12%/+20%**, 3 of 3 up) | **adopted** |
| `--batch 2 --batch-mtp` | 257/383 → 142/203 (**−45%**) | 31.9/29.2 → 25.6/23.4 | rejected; 2-client burst 40.6 s queued vs 71.4 s batched. The #2126 Linux startup hang did not reproduce |
| the three adopted together | 258/384 → 266/384 | 31.2/28.6 → 34.4/33.1 (**+10%/+16%**) | confirmed |

Model load also improved: ~2.8 min → 56 s with `--ple-io ram` (the PLE table stops being read from SSD at load).

## Long context (fresh prompts, 60 tokens out)

| Depth | prompt tok/s (no `--kv-grow`) | prompt tok/s (`--kv-grow`) | decode tok/s (no / with) |
| --- | ---: | ---: | ---: |
| 32K (32,442 tok) | 415 (410-420) | 398 (395-401) | 28.1 / ~27 |
| 128K (130,346 tok) | 372 (370-373) | 363 (363-364) | 32.4 / 31.5 |
| 250K (248,245 tok) | 333 | 318 | 29.5 / 24.2 (n=1) |

Prompt throughput falls only 20% from 32K to 250K and decode does not fall with depth. `--kv-grow` is negative at
every depth here (−2 to −4.5% prompt; its #2650 benefit is decode drift inside one long growing session, a workload
these fresh-prompt runs do not exercise). The 128K run logged KV streaming doing 431,340 block reads with **88.27%
served from VRAM, only 203.9 MiB read from RAM**.

**Needles** (`tools/needle_bench.py`, depth 50%): 128k — FOUND at 124,782 tokens (342 s); **250k — FOUND at 251,546
tokens (732 s)**, exact word (`needles-lg-grow2.json`). First correctness evidence at 250K on this card: streaming
int8 KV with `--kv-resident 32768` holds up at the far end of the 262K window.

## Soak (TEST_REQUESTS: "an hour or more of back-to-back requests" on a resident setup)

Adopted stack, 60 minutes, 237 requests (short/long alternating), **0 failures, no drift**: short prompt 221.1 →
221.0 tok/s, long 390.9 → 391.9, decode 32.9 → 33.7 / 31.5 → 31.9 (first third vs last third medians). End temp
73 C, clocks 1708-1733 MHz (no thermal throttling; the card runs a −200 MHz cap). Per-request data:
`soak-results.json`.

## Correctness and limitations

- Correctness evidence is thin by design: two needles (both found), one arithmetic probe identical in all 20 arms.
  No answer-identity suite, no HumanEval; `--kv k8v4` was judged on speed only (it was not adopted anyway).
- Single round per knob: decode runs are noisy on this rig (25-35 tok/s between runs of the same config); a verdict
  was taken only when all three requests of an arm moved together, and the adopted stack was confirmed in one final
  pair. Prompt numbers are much steadier (±3%).
- 250K long-context is one run per arm; the needle is the correctness anchor there.
- Not tested: Windows, other models/quants, `--batch` at 3+ clients (its value is real concurrency on other
  workloads), sessions that grow to 262K inside one conversation (#2650's drift), prompts between 7K and 32K.

## Files

Pair transcripts (`pair-t*.out` incl. the per-request engine lines and quality probes), `burst.out`, `lg-*.out`
(long-context arms), `needles-lg-grow2.json`, `soak-results.json` (237 rows), and the scripts
(`arm-run.sh`, `pair-run.sh`, `longbench.py`, `burst.py`, `soak.py`, `numa-cgroup-launch.sh` — the last one is the
root-level cpuset cgroup wrapper this rig needs for node binding, see #2684).
