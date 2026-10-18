# DFlash review fixes, 2026-10-17

Base: `434ecabc` (`fork/feat/dflash`, Strata 0.1.45), with the working-tree changes
in this review. This is a partial response to
[Niko's review](https://github.com/Niko1221/Strata/pull/1426#issuecomment-6239944066).
The exact prompt behind his greedy mismatch was not available. No fix for that
specific mismatch or general DFlash speed advantage is claimed.

## Changes

- Read diagnostic FP16 bits on the host with `memcpy`, rather than the device-only
  `__ushort_as_half` intrinsic. CUDA builds. HIP and SYCL cannot be built here:
  ROCm and `icpx` are missing; see the configure logs in this directory.
- Keep the final committed draft context when generation ends, including the
  token limit and EOS. Do not commit accepted draft rows whose outputs were not
  emitted. The final emitted token remains an unconsumed anchor.
- Allow prompt checkpoints and live continuation with DFlash. Park and restore
  all five FP16 draft pools with the target state. Include their allocations in
  the RAM budget and physical-memory admission check. Validate every pool before
  overwriting a live session. DFlash takes parked entries whole; it does not
  borrow a donor's prefix. Disk sessions remain refused.
- Use target-only decoding for sampled requests. The server still maintains draft
  context for a subsequent greedy request; it synchronizes this consumption even
  when no proposal follows. Sampled CLI runs skip loading the drafter.
- Make the target-logit probe work with Off, MTP and DFlash, in the CLI and server.
  It reports only kept rows and can save the full F32 logit row.
- Remove links to four historical reports absent from this branch. Their older
  numbers are not evidence for the current build.

## Sampled regression and target logits

Hardware: RTX 4070 Ti SUPER 16 GiB, Ryzen 9 5900X, 128 GiB installed RAM,
Linux 6.12.107, CUDA 13.4, GCC 14.2, sm89. Target: GSQ-RCO IQ3_XXS, shared Q5_K
head. Drafter: PixelML Q4_0, 106299 selected vocabulary rows. Fixed 3500 MiB
expert cache (4675 slots), adaptation off, prefill CPU sharing off,
`STRATA_IQ_MT_MIN=1`. These are CPU/GPU mixed decode runs, not all-experts-in-VRAM
runs on Niko's Blackwell card.

Before the fallback, `tools/dflash_sampling.py --count 128` failed its cold,
filtered and penalty cases. Greedy and greedy-repeat passed. The first cold
mismatch was output index 3, token 16 versus 17, with temperature 0.3, seed 1234,
top-k 20 and top-p 0.95. Full token sequences are in
[before-fallback-sampling.json](before-fallback-sampling.json).

At the corresponding absolute input position 26, the target logits changed:

| Token ID | Off | DFlash speculative verification |
|---|---:|---:|
| 16 | 20.4672947 | 20.5032387 |
| 17 | 20.5256424 | 20.4684105 |

The top-two margin changes sign. Across the whole vocabulary the maximum absolute
difference is 0.5808825 and RMS difference is 0.106162. A sampled pick need not be
the argmax. This is evidence for changed target logits, not a claim that only the
sampler or the draft head is responsible. See [near-tie.json](near-tie.json) and
[before-fallback-probe.json](before-fallback-probe.json).

With `--spec-min-p 1`, draft computations and taps remain enabled but the verified
prefix is empty. All 248320 target logits at that position are bit-identical to
Off, and the eight emitted tokens match. See [gated-probe.json](gated-probe.json).
This control isolates the discrepancy to speculative verification/history rather
than a tap-only write to the target. It does not distinguish arithmetic differences
from a verifier rollback defect, and does not explain Niko's unknown prompt.

The sampled fallback is a correctness restriction, not a sampled speedup. The
gate now requires zero offered DFlash candidates for sampled requests, real draft
execution for greedy requests, equality to Off, and a sampled-to-greedy reset.

## Greedy near-tie fixture

The integration-by-parts prompt reproduces a greedy difference at output index
72 (absolute input position 108). The first 72 emitted tokens match. At that row:

| Token ID | Off | DFlash |
|---|---:|---:|
| 5170 (`First`) | 24.5071468 | 24.3672791 |
| 332 (`**`) | 24.4744987 | 24.6652279 |

Off's winning margin is 0.0326481. Both runs have the same top-two IDs, but their
order swaps: the continuations read `First application:` and `**First application:**`.
Full-vocabulary RMS drift is 0.0873765 and maximum drift is 0.635169.
These runs have the default `STRATA_ROUTE_TAIL_SKIP=7` enabled. Missing
experts are skipped according to the whole verify window, so changing its
width can change the computed function, not just floating-point rounding.
This is a confound in this mixed-residency reproduction and is absent from
Niko's all-experts-resident Blackwell case.

This documents a near tie in the baseline, whose margin is smaller than the
observed target-logit perturbation. It does not prove a tap defect or a universal
bound on numerical error. See [greedy-near-tie.json](greedy-near-tie.json) and
[greedy-probe-results.json](greedy-probe-results.json).

Control on 2026-10-18: rerunning the same fixture with
`STRATA_ROUTE_TAIL_SKIP=0` produced identical 73-token Off/DFlash outputs,
including index 72. At position 108, both select `First`; the target margin
is 0.1052551, full-vocabulary RMS drift 0.0401626 and maximum drift 0.2241893.
See [no-tail control](no-tail-probe/results.json) and its engine logs. Removing
the window-dependent expert skip removes this observed mismatch; residual logits
still differ. This establishes a local confound, not a fix for Niko's
all-experts-in-VRAM divergence or exactness on arbitrary longer continuations.
Reproduce the command below with `STRATA_ROUTE_TAIL_SKIP=0` in its environment.

The fixture-specific regression is `tools/dflash_greedy_probe.py`. It passes exact
outputs or this narrowly bounded case: all preceding tokens equal, the same top
two IDs, target margin at most 0.05, RMS drift at most 0.1 and maximum drift at
most 1.0. The thresholds describe this measured fixture; they are not a general
exactness exemption. A different first mismatch fails and leaves the evidence.

```sh
python tools/dflash_greedy_probe.py --config DFLASH_CONFIG.json --engine build/strata \
  --fixtures bench/results/2026-10-17-dflash-review/math-fixtures.json \
  --case integration --index 72 --output /tmp/dflash-greedy-probe
```

## Cache checks

The GPU pool regression saves five distinct FP16 pools through a partial page,
overwrites them, restores them, verifies their bytes and rejects foreign geometry.
No target checkpoint or model download is needed for this test.

`tools/dflash_cache.py` uses one-candidate, split verify windows and fixed expert
residency to keep arithmetic consistent between fresh and resumed reads. A
checkpoint is pinned at token 16. It checks continuation, parking, rewind and
sampled-to-greedy continuation against fresh requests. It also checks the reused
token counts, so silently reading the whole prompt cannot pass. This controlled
test does not establish byte equality for every default-width cached conversation.
An earlier four-candidate fresh-versus-resumed comparison differed; that broader
arithmetic/verification parity requirement remains open.

## Math comparison

The raw runs, resolved engine arguments and generated token IDs are in
[math-results.json](math-results.json), with the prompts and exact input IDs in
[math-fixtures.json](math-fixtures.json). Each arm starts a fresh engine, warms up
on 16 tokens, then requests 300 greedy tokens per prompt. Order alternates
Off/MTP/DFlash and DFlash/MTP/Off. Startup and prefill are excluded from decode
throughput. Other GPU tests and compilation were stopped during the comparison.

All arms use the same IQ3_XXS target, 4675 resident expert slots, 3500 MiB cache,
256-token prefill, 4096-token context, INT8 target KV, PCIe fraction 0.20,
adaptation off and prompt/suffix drafting off. MTP uses `--spec 4`; DFlash uses
`--spec 8 --dflash-block 5`. Both use `--spec-min-p 0`. This is a fixed ungated
comparison, not a sweep of every calibration setting.

Three interleaved rounds gave these median decode rates (tok/s):

| Prompt | Off | MTP | DFlash | DFlash / MTP |
|---|---:|---:|---:|---:|
| Induction | 51.48 | 71.70 | 40.92 | 0.571 |
| Quadratic formula | 49.00 | 78.79 | 61.66 | 0.783 |
| Integration by parts | 47.23 | 78.28 | 50.16 | 0.641 |

DFlash won none of the nine prompt/round comparisons against MTP. Aggregate
throughput across the nine requests per arm was 48.96 / 76.07 / 49.55 tok/s for
Off / MTP / DFlash. See [math-summary.json](math-summary.json).

The continuations are not byte-identical to Off: the first greedy differences in
round 0 occur at output indices 108 / 167 / 216 for MTP and 108 / 156 / 72 for
DFlash. Every arm produced 300 tokens. These timings describe those actual
continuations; they are not an output-equivalence or quality gate. The tests do
not meet the review's requirement for a DFlash win over MTP.

Reproduction with existing model artifacts:

```sh
python tools/dflash_math_bench.py --config DFLASH_CONFIG.json --engine build/strata \
  --mtp MTP_RUNTIME_DIR --fixtures bench/results/2026-10-17-dflash-review/math-fixtures.json \
  --output /tmp/dflash-math --rounds 3
```

## Confidence thresholds and automatic VRAM budget

Validation continued on 2026-10-18. An exploratory sweep used the same fixed
3500 MiB expert cache, three fixtures and 300 generated tokens, with confidence
thresholds 0, 0.5, 0.7 and 0.9 for both drafters. Each arm ran in one process,
with a 16-token warmup per threshold; these are single observations, not paired
repeated measurements. Raw token IDs and timing are in
[confidence-sweep.json](confidence-sweep.json).

At threshold 0.5, MTP / DFlash throughput was 77.80 / 69.08 tok/s for induction,
83.36 / 73.01 for the quadratic formula, and 80.41 / 66.33 for integration.
DFlash improved over its zero-threshold run, but did not beat MTP on any fixture
at any tested threshold. This does not justify changing the engine defaults.

A further single serial round used automatic expert-cache sizing, the same
1024 MiB reserve for every arm and threshold 0.5. Final allocated expert slots
were 6210 / 5612 / 6008 for Off / MTP / DFlash. DFlash therefore had more expert
capacity than MTP. Decode tok/s were:

| Prompt | Off | MTP | DFlash |
|---|---:|---:|---:|
| Induction | 53.31 | 81.88 | 67.62 |
| Quadratic formula | 51.47 | 84.32 | 75.74 |
| Integration by parts | 49.39 | 80.21 | 66.02 |

There was no DFlash win. This exploratory round is not a repeated performance
claim. See [raw results and resolved arguments](auto-cache/math-results.json)
and [input configuration](auto-fixture-config.json). Reproduce with the math
benchmark's `--auto-expert-cache --spec-min-p .5 --rounds 1` flags, passing a
configuration containing `--vram-reserve-mib 1024`. Generated continuations may
differ, as in the fixed-cache comparison.

## Validation

| Check | Result |
|---|---|
| CUDA engine build | Passed |
| DFlash GPU CTest checks | 4 passed |
| Artifact loader and draft policy checks | 2 passed |
| Conversation cache, file, transfer and snapshot CTest checks | 8 passed |
| Installer, calibration, quantization and Python server tests | 392 passed |
| 128-token server sampling/greedy gate, seed 7, filters and penalties | Passed |
| 128-token sampled CLI versus Off | Passed |
| Eight cache scenarios, including sampled-to-greedy continuation | Passed |
| Greedy integration fixture | Passed as the documented near tie |
| HIP / SYCL build | Blocked at configure: ROCm / icpx unavailable |

Evidence: [sampling-test.log](sampling-test.log), [sampling-results.json](sampling-results.json),
[cache-test.log](cache-test.log), [cache-results.json](cache-results.json),
[greedy-fixture-test.log](greedy-fixture-test.log),
[greedy-fixture-results.json](greedy-fixture-results.json),
[cuda-tests.log](cuda-tests.log), [conversation-tests.log](conversation-tests.log),
[loader-policy-tests.log](loader-policy-tests.log), [python-tests.log](python-tests.log).
The Python server suite uses simulated engines; its intentional error log lines
are test fixtures, not failures of these model runs.

The two 128-token DFlash-off greedy controls also match the parent generation
source at `434ecabc`, compiled with the same current libraries and headers. This
isolates the generation-program changes; it is not a clean full-release rebuild
or a rerun of Niko's ten prompts at 4K and 20K. His original DFlash-off result is
reported in the linked review. No inference arithmetic was changed for Off.

## Remaining review requirements

The exact Niko greedy prompt, template/thinking setting and launch/request
parameters are needed for his reported token-17 divergence. The probe can collect
its target logits by setting `STRATA_DF_TARGET_PROBE` to the absolute input
position and `STRATA_DF_TARGET_PROBE_FILE` to an output path. Compare the same
target configuration and prefix on both sides.

HIP and SYCL compilation, Niko's Blackwell/RTX 3060 measurements, long-context
cache pressure, and a measured DFlash win over MTP remain unverified here. MTP
remains the default drafter. No DFlash/MTP combination was added.
