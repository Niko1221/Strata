# Windows processor groups and CPU-pool workers on two machines

This file records two Windows measurements of Strata's CPU expert pool. The report below, from 2026-10-03,
compares a before/after build on a Threadripper 3990X, where the two processor groups were the problem. A
later section, from 2026-10-29, sweeps `--pool-workers` on an RX 6600M laptop with one processor group, where
the pool computes most of the experts that are not in the 8 GiB of VRAM.

On 2026-10-03, a before/after run compared Strata on Windows 11 Pro (build 10.0.26300) with a Ryzen Threadripper
3990X (64 physical cores, 128 logical processors) and Radeon RX 6900 XT (16 GiB, gfx1030). Windows reported two
processor groups of 64 logical processors each, but one NUMA node spanning both groups. The original pool code
encoded each CPU as `group * 64 + processor`, then used `SetThreadAffinityMask` with `cpu & 63`. That discarded the
group number, so CPUs from both groups aliased to the same group-relative masks. The experimental group-aware build
at [source commit fba17b6](https://github.com/Yasei-no-otoko/Strata/commit/fba17b6ec6e50104aa194443eaee17728db14135)
uses group-aware thread affinity and preserves group identity when pinning workers and the session host.

The standard automatic worker count remains 63 on this 3990X. No hardware-specific worker cap or default-selection
change is part of this fix. All comparison runs below explicitly passed `--pool-workers`; the 63-worker comparison
targets the affinity correction, while the older 31-worker rows are retained as a manual-count reference. Explicit
worker settings remain available as before.

## Measurement

The workload was Qwen3.8-Flash-Next Coder IQ1_M, 65,536 context, INT8 KV with 32,768 resident cells, automatic
prefill, fixed expert-cache budget, MTP depth 4, and no images. Both builds used the same Windows HIP release build
configuration: ROCm 10.2.0a20260930, Clang 24.0.0, pinned llama.cpp commit
`3cf03257f219afbe7334045ff7c6a06ac68c627d`, and expert-cache argument 2,500 (3,245 effective slots in the measured
settings). Each short and 4K result is the median of three fresh prompts generating 256 tokens apiece; each session's
warm-up was excluded and measured prompts reused zero prefix tokens. Prompt and decode rates below use the engine's
token counts and timings. TTFT and total latency are client-side medians in seconds.

| Pool build | Explicit workers | Input tokens | Prompt tok/s | Decode tok/s | TTFT s | Total s | Initial VRAM free |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Before group-aware build | 63 | 94 | 38.08 | 12.69 | 2.64 | 22.66 | 425 MiB |
| Experimental group-aware build | 63 | 94 | 46.84 | 46.09 | 2.05 | 7.57 | 214 MiB (LOW) |
| Before group-aware build | 63 | 4,022 | 312.02 | 13.08 | 13.13 | 32.49 | 425 MiB |
| Experimental group-aware build | 63 | 4,022 | 316.11 | 47.43 | 12.79 | 18.16 | 214 MiB (LOW) |
| Before group-aware build | 31 | 94 | 47.47 | 46.97 | 2.02 | 7.44 | 423 MiB |
| Experimental group-aware build | 31 | 94 | 43.55 | 34.78 | 2.21 | 9.52 | 145 MiB (LOW) |
| Before group-aware build | 31 | 4,022 | 318.13 | 44.99 | 12.72 | 18.42 | 423 MiB |
| Experimental group-aware build | 31 | 4,022 | 315.39 | 34.60 | 12.83 | 20.13 | 145 MiB (LOW) |

Both paired binaries contained an earlier automatic-worker proposal, bypassed by the explicit counts in every comparison. The final [code PR #626](https://github.com/Niko1221/Strata/pull/626) removes that proposal and leaves automatic worker sizing unchanged.

At 63 explicit workers, the experimental group-aware build's decode medians were about 3.63 times the prior build's
in both measured cases. The 4K prompt throughput changed little. The old 31-worker run decoded at 46.97/44.99 tok/s, while
the experimental 31-worker build measured 34.78/34.60 tok/s. Its starting free VRAM was only 145 MiB and Strata
marked it LOW, versus 423 MiB before; background load and VRAM conditions were not controlled, so this regression's cause remains unresolved. The experimental 63-worker results were similar to the prior 31-worker results in this short/4K sample. The
final PR preserves the original automatic worker selection (63 on this machine); all paired comparison runs
explicitly passed `--pool-workers`.

A separate experimental-build 32K request read 32,685 fresh prompt tokens, generated 256 tokens, and measured 350.70 prompt
tok/s, 48.87 decode tok/s, 93.34 seconds TTFT, and 98.55 seconds total latency. This is one observation, not a
three-run median, and is not included in the table.

## Automatic-settings check before the review follow-up

The final build at `d828e8e2cea24908f9d24597e987257ed293acc1` selected **63 workers** without `--pool-workers`.
With `--expert-cache auto`, it selected 2,675 effective slots and had 1,143 MiB of VRAM free after startup.
Three fresh requests per short/4K case produced decode medians of **43.13 / 57.91 tok/s** and client TTFT medians of
**2.20 / 14.29 seconds**. These are separate validation of the final automatic path: cache capacity differs from
the paired runs, and outputs/draft acceptance also differ, so this row is not an additional affinity-only speedup.
The [final configuration and metadata](../../bench/results/2026-10-03-windows-groups/final-auto-metadata.json)
and raw data are preserved with the paired results.

## One processor group with streamed experts: RX 6600M (gfx1032)

An RX 6600M has 8 GiB of VRAM, and its expert cache held 1,165 of this model's 12,288 (layer, expert) pairs, so
no more than 9.5% of the pairs are resident on the card and most routed experts are read and computed by the
host. That makes the worker count the same lever this file measures, in the regime where it costs the most. The
machine also has one processor group (Windows creates one per 64 logical processors, and this CPU has 20), so
the group path is not exercised here. This section is a single-group control, not a test of the affinity fix.

> **Measured on:** RX 6600M, 8 GiB (gfx1032), 13th Gen Intel Core i5-13600 (6 performance cores and 8
> efficiency cores, 14 physical, 20 logical, one processor group), 95.7 GB RAM, Windows 11 build
> 10.0.26100.3037, an engine built from this tree at commit `46c152a1` (engine 0.1.53) with ROCm
> 10.2.0a20260930, the Qwen3.8-Flash-Next GSQ-RCO IQ1_M coder pack (256 experts, 48 layers) as a native pack,
> 2,048-cell context, `--spec 4`, automatic prefill, `--expert-cache auto` (1,165 slots, 2.25 GiB of VRAM),
> greedy sampling, and the same five-token prompt in every run.

Without `--pool-workers` the engine chose:

```
strata generate: CPU pool: 9 workers by default (14 physical cores, 20 logical processors; bounded by the performance cores and half of the efficiency cores)
strata generate: 9 pool workers on logical processors 2,4,6,8,10,12,13,14,15, the host thread on 0
```

Five performance cores after the host's, plus four of the eight efficiency cores. 558 MiB of VRAM stayed free
after the expert cache was filled. The table below generates 64 tokens per run; every row ran twice, and the
rows other than Automatic passed `--pool-workers` explicitly. The per-round figures are the engine's own
`verify window` line.

| Explicit `--pool-workers` | Decode tok/s (run 1 / run 2) | Pool ms/round (run 1 / run 2) | Wait for rings ms/round (run 1 / run 2) |
| --- | --- | --- | --- |
| Automatic (9) | 3.89 / 4.08 | 53.6 / 52.3 | 214.8 / 207.8 |
| 1 | 3.81 / 3.72 | 107.9 / 117.8 | 173.8 / 170.9 |
| 3 | 4.16 / 4.07 | 63.9 / 74.8 | 192.4 / 175.0 |
| 5 | 4.16 / 4.06 | 49.8 / 54.6 | 205.3 / 207.0 |
| 9 | 4.15 / 4.01 | 51.3 / 54.7 | 204.9 / 210.5 |
| 13 | 4.17 / 4.02 | 46.2 / 49.8 | 208.4 / 213.7 |

The pool's share of a round falls from 108 to 46 ms as the workers rise from 1 to 13; the wait for the rings
rises from 174 to 208 ms, which absorbs about half of it, and a round ends about 25 ms shorter (about 300
against 275 ms). Two runs at the same setting already differ by up to 7% (automatic 3.89 against explicit 9 at
4.15), so the sweep separates only one count: 1 worker averages 3.77 tok/s where 3, 5, 9 and 13 average 4.08 to
4.12 and cannot be told apart. On this card the count above one worker is not what moves decode, and the
automatic 9 is not a bottleneck. All twelve runs produced the same 64 token ids, byte for byte, which matches
the flag's help text: it moves threads only, never a result.

**Decode over a longer run is slower.** Same session, 9 workers, same prompt and settings, at three lengths
(the 16-token row repeats the 5.70 tok/s recorded for this machine in its [card report](../RX6600M.md)):

| New tokens | Rounds | Decode tok/s | Decode ms/round | Pool ms/round | Wait for rings ms/round | CPU experts per layer, distinct / routings |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 16 | 15 | 5.71 | 187 | 76.3 | 88.4 | 13.88 / 22.12 |
| 32 | 29 | 4.63 | 238 | 62.0 | 155.6 | 11.95 / 17.43 |
| 64 | 56 | 4.15 / 4.01 | 276 / 285 | 51.3 / 54.7 | 204.9 / 210.5 | 9.57 / 12.86 |

The 16- and 32-token rows are one run each. The 64-token row is the two explicit-9 runs from the table above,
and its CPU-experts cell is run 1 (run 2: 9.84 distinct / 13.21 routings).

The average round grows from 187 to about 280 ms while the pool's share of it falls. During the 64-token runs
the adaptive tier moved 1,087 to 1,102 experts into the VRAM tier, and the experts computed on the CPU fell
from 13.88 to 9.57 distinct per layer (22.12 to 12.86 routings). This report did not separate these effects,
so read the 64-token rows as rates over a longer run, not as a different build.

These are two runs per setting on a desktop that was not idle, with VRAM state and ambient conditions not
controlled. They do not establish results for other CPUs, cards, models or worker counts, and they say nothing
about the group-affinity fix. The raw engine output of every run is preserved at
[the measurement directory](../../bench/results/2026-10-29-community-rx6600m-workers/README.md).

## Scope and checks

The NUMA/group probe found one NUMA node across the two Windows processor groups. It demonstrated the legacy mask
alias on probe-owned threads. The inference comparison used the same model, explicit worker count, build settings,
and expert-cache settings before and after. It ran on a normal desktop; other applications were not stopped and free
memory varied, so the measurements are evidence for this machine and workload rather than a controlled claim about
all systems. They do not establish results for other CPUs, operating systems, GPUs, or models. The RX 6600M
section has the same caveat, with two runs per worker count, one processor group, and no control of ambient
conditions; it measures the worker count only.

The experimental Windows HIP build compiled, and the processor-group affinity API test passed. Both the AVX-512
pool self-test and `pool_stress` skipped their workloads on this CPU. `pool_stress` returns zero when skipped, so
the original CTest summary's "Passed" is not a stress-workload pass. A Linux build could not be run because the WSL virtual disk was unavailable. Microsoft's
[processor-group documentation](https://learn.microsoft.com/en-us/windows/win32/procthread/processor-groups)
describes the group-relative processor numbers and Windows 11 primary-group behavior;
[SetThreadAffinityMask](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadaffinitymask)
documents that the legacy mask applies to the thread's current primary group. The
[Windows Containers issue #409](https://github.com/microsoft/Windows-Containers/issues/409) concerns a different
container environment and multiple NUMA nodes; this Strata measurement used a native process on one NUMA node and
does not claim an operating-system or Docker fix.

The full probe, raw inference runs, configs, build metadata, and reproduction script are preserved at
[the measurement directory](../../bench/results/2026-10-03-windows-groups/README.md).
The [original official-release measurements](../../bench/results/2026-10-03-rx6900xt-coder/README.md) retain the earlier manual-worker sweep as historical evidence.

After these measurements, PR #626 changed the host path to reversible CPU Set selection while keeping hard group
affinity on pool-owned workers. The saved throughput figures above belong to their recorded revisions, not this
later host-restoration change. Its [review validation](../../bench/results/2026-10-03-windows-groups/review-validation.md)
records the build and regression tests; inference throughput was not remeasured for that revision.
