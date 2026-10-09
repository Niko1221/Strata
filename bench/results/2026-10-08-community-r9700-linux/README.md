# Single R9700 Flash Next speed and short prefill measurements

Measured on 2026-10-08 by @zihaomu on one 32 GB Radeon AI PRO R9700, with two EPYC 9334 CPUs and
about 503 GiB of system RAM. This report covers original ISTA GSQ-RCO Qwen3.8-Flash-Next IQ3_S,
with IQ2_XS as a separate quantization contrast. [中文加速总结](ACCELERATION.zh-CN.md)

The product-style IQ3_S configuration generated **87.6–101.9 tokens/s**, taking the median at each
of four input lengths from 1K through 128K. Separately, five independent pairs on a controlled
configuration found **16.18% / 14.07% lower native TTFT** for 512 / 900 new tokens after 32K retained
history with the short-input grouped gather from [#1107](https://github.com/Niko1221/Strata/pull/1107).
There are also regressions: the change is **not accepted as an all-workload performance improvement**.

These are measurements of a frozen experimental source based on release commit `d5ea713`, not of
the newer upstream revision carrying this report. This PR adds results and reproduction artifacts;
it does not apply its archived patches to the engine or change runtime defaults.

## Hardware and software

| Item | Recorded environment |
| --- | --- |
| Selected GPU | Radeon AI PRO R9700, 32 GB, gfx1201, BDF `0000:63:00.0` |
| Host | Two EPYC 9334 CPUs, 128 logical CPUs, approximately 503 GiB OS-visible RAM |
| Placement | CPU NUMA nodes 0,1; memory preferred node 1 with fallback; 15 engine workers |
| OS and runtime | Ubuntu 24.04.1 LTS, kernel 7.0.0-31-generic, system ROCm 7.2.3, HIP 7.2.53211, hipBLASLt 100202 |
| Storage and PCIe | NVMe SSD for model/PLE; sysfs reported 32.0 GT/s x16 during the current-speed run; startup H2D probe 36.9 GB/s |
| Build | Release, gfx1201, native experts and HIP prefill MMQ enabled, pinned llama.cpp `3cf03257` |

The host contains eight GPUs, but ROCr UUID isolation and a HIP API check required exactly one visible
device with the recorded BDF. Per-device memory snapshots show the model allocations on that card;
other cards have 32 KiB each of runtime bookkeeping. One current-speed activity snapshot showed
100% on the selected card and 0% on the others. This is a shared host, so host CPU interference is
not excluded. No GPU power cap, clock, driver or global OS setting was changed. RAM speed and GPU
power cap were not recorded. See [hardware.json](hardware.json).

Single GPU here still uses CPU experts, system RAM and PCIe. These measurements do not establish
the rate on a desktop CPU or the minimum RAM needed to run the model.

## Measured source and model

The preserved baseline includes a gfx1201 Q2_0 signed-zero fix and an inactive first-logit diagnostic
hook. The candidate additionally applies #1107 at `1a58f780c5d112a10509eb7478778f1712cc41a6`.
Its implementation change is confined to `src/prefill/prefill.cpp`: short staged reads gather native
experts in groups of up to 16, with 32 staging slots and group-level copy waits/release events.
Long streamed reads already supported grouped gather in the baseline.

| Arm | Original source reference | Preserved executable SHA-256 |
| --- | --- | --- |
| Baseline | `9fd910f4bc6513558b133f7faf78b57f5874fd35` | `e90845e090cb51ee836c967e9e2fa01890a06cabcb389a1707778235c7a84b96` |
| Candidate | `269a140b473a36686661103b9649f8383fe88b77` | `ffc88a3da34fc83158c8cd64fae549d928d9f2953a62234c1d7d0f1967d63b1d` |

The later product-style speed run recorded checkout `25d2e33`; its engine source was unchanged from
the candidate and it reused that executable. The original local commit IDs are provenance, not a
requirement to fetch an unpublished branch: [source-layout.json](source-layout.json) and the two
[archived patches](patches/) reconstruct the measured engine source from public release commit
`d5ea7133741e67743c0e886bb426c0ce8d69cf6c`. Rebuilding can change executable hashes and requires new validation.

GGUFs come from `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` at
`ed59f92082b1e93c0e96d60a8b11aab089b52f09`, with the original IQ3_S and IQ2_XS two-shard filenames.
The MTP runtime uses the original Qwen checkpoint, Q2_0 draft experts and the repository draft
vocabulary. [Model manifest](model-manifest.json) and the
[IQ3_S](evidence/iq3s-assets.json) / [IQ2_XS](evidence/iq2xs-assets.json) inventories record hashes
for the GGUFs, pack, tokenizer, MTP and expert profile. No vision or experimental speed projection
was enabled; reasoning was disabled. No new calibration table was selected.

## Method and configurations

All performance requests are native engine protocol, concurrency 1, temperature 0, top-k 1,
top-p 1, min-p 0, seed 42, with exactly 256 generated tokens. TTFT is submission to the first native
token; it excludes HTTP/SSE, client networking and model loading. Prompt rate uses freshly read
tokens divided by engine prompt time; decode rate uses generated tokens divided by engine decode
time. Total latency includes both stages. Warmups are retained but excluded from the reported values.

| Setting | Product-style fresh speed | Controlled code comparison |
| --- | --- | --- |
| BLAS / attention | Tensile; WMMA off; Lt tuning empty | Same |
| Native prefill MMQ | Enabled | Enabled |
| Expert cache | Auto: 13,021 slots, 24.71 GiB on IQ3_S | Budget 8,000 largest-expert slots |
| Chunk / ring | Auto: 8,192 / 384 | Fixed 4,096 / 96 |
| PCIe / adaptation | Defaults retained; probe selected PCIe fraction 0.55 | PCIe=0, adaptation disabled |
| IQ kernel selection | Default | `STRATA_IQ_MT_MIN=1` |
| MTP | `--spec 4 --spec-min-p 0.5`, default suffix/lookup policies | Same MTP limit; suffix/chain disabled |
| KV / context | int8, 32,768 resident cells, 147,456 maximum context | Same |
| Workers / reserve | 15 / 1,024 MiB requested reserve | Same |
| Prompt cache | 0, no reuse | 0 for fresh input; incremental runner sets 6 |

The models use the native in-RAM expert arena, not an added mmap/resident-CPU-expert experiment.
Both configurations select `ROCBLAS_USE_HIPBLASLT=0`, `STRATA_PREFILL_MMQ=1`,
`STRATA_HIP_WMMA=0` and an empty `STRATA_HIPBLASLT_TUNING`.
The exact [current-speed config](current-speed/iq3_s-config.json) and
[controlled configs](evidence/configs-tensile-repro-fixed/) are included.

The local default/preferred hipBLASLt path produced large errors and nonfinite outputs when replaying
captured FP16 T249/N512/K2560 GEMM operands in a separate program. Tensile completed 1,000 identical
repetitions, with maximum absolute error about 1.719e-5 against CPU float64. This is why the final
comparisons use Tensile. It is a result for this installation and capture, not a universal hardware
or ROCm diagnosis. Earlier default-backend speeds are not mixed into the comparisons below.
[Replay results](evidence/gemm-repeat-summary.json), [numerical checks](evidence/gemm-repeat-numerics.json).

## Fresh IQ3_S speed

One engine session: one short warmup, one full generation at each shape, then three rounds over
1,024 / 4,096 / 32,768 / 131,072 input tokens. The synthetic task asks for a record-format summary
and Python parser. Every measured input is fully processed (`reused=0`), with 256 outputs.
Filesystem and expert caches are warmed. Brackets show minimum–maximum across all three repeats.

| Input tokens | Prompt tok/s median [range] | Decode tok/s median [range] | Native TTFT seconds median [range] | Total seconds median |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 632.9 [514.8–634.0] | 101.9 [101.2–102.8] | 1.64 [1.64–2.01] | 4.15 |
| 4,096 | 768.1 [716.3–768.8] | 96.4 [94.4–100.6] | 5.35 [5.35–5.74] | 8.05 |
| 32,768 | 750.5 [750.4–775.4] | 95.6 [90.4–98.2] | 43.69 [42.29–43.70] | 46.29 |
| 131,072 | 718.4 [713.9–719.0] | 87.6 [85.3–91.4] | 182.48 [182.33–183.64] | 185.39 |

All 12 formal requests finished at the 256-token cap. Generated IDs and draft counts vary between
repeats under the retained product policies. This is absolute speed on this workload, not a
before/after attribution of these rates to #1107. Startup took 34.56 seconds, excluded above;
that is not a cold-disk loading benchmark. READY reported 940 MiB free VRAM. The largest process
VRAM observation at request boundaries was 30.84 GiB; this is not a continuously sampled peak.
[All requests](current-speed/iq3_s-session/results.json),
[engine log](current-speed/iq3_s-session/engine.log.gz), [summary](current-speed/summary.json).

## Controlled short increment comparison

Five IQ3_S independent-engine pairs comprise a two-pair screen and three unchanged confirmation
pairs. Each engine warms the entire shape sequence once, then measures each increment once after
rewinding to the same 32,768-token history. Every corresponding input/output ID, reuse/read count
and MTP accepted/offered count matches. The controls above hold in both arms.

| New tokens after 32K | Baseline TTFT median | Candidate TTFT median | Median paired time reduction | Paired range |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 1.327 s | 1.100 s | 16.42% | −23.87% to +33.90% |
| 512 | 1.868 s | 1.570 s | **16.18%** | +15.07% to +25.58% |
| 900 | 2.545 s | 2.185 s | **14.07%** | +13.49% to +14.27% |
| 2,048 | 4.045 s | 4.060 s | −0.32% | −0.80% to −0.11% |
| 4,096 | 6.623 s | 6.640 s | −0.30% | −4.00% to −0.19% |

Paired reduction is `100 * (1 - candidate_time / baseline_time)` per pair, then the median;
it need not equal the ratio of the displayed time medians. All five pairs are faster at 512/900,
with whole-request median paired time reductions of 7.03%/7.53%. The positive median at 256 is
not a stable improvement: two pairs regress by more than 23%.
[Five-pair summary](evidence/pr1107-incremental-five-pairs-iq3s/summary.json),
[screen requests](evidence/pr1107-incremental-screen-iq3s/),
[confirmation requests](evidence/pr1107-incremental-confirm-iq3s/).

The separate two-pair IQ2_XS contrast finds 16.21%/14.96% median paired TTFT reductions at
512/900 tokens, with both pairs faster. Its 256-token range is −26.94% to +36.15%.
This supports the direction at 512/900 but is only two pairs, not five-pair acceptance.
[IQ2_XS requests and summary](evidence/pr1107-incremental-screen-iq2xs/).

## Regressions and rejected configurations

Two fresh-input pairs per quantization also covered 4K/32K/128K with 256 outputs and no reuse.
Corresponding outputs and MTP work match. IQ3_S decode time is **3.5–4.7% longer at 4K/32K in
both pairs**. IQ2_XS decode changes direction between pairs, and one 128K TTFT is **9.67% longer**.
All slow samples remain in [IQ3_S](evidence/pr1107-fresh-screen-iq3s/) and
[IQ2_XS](evidence/pr1107-fresh-screen-iq2xs/). Neither these pairs nor the product-style speed table
establish an all-workload improvement.

Four Lt/WMMA numerical candidates failed the predeclared stable-reference gate: all logits finite,
mean KL ≤0.001 nat, per-position KL ≤0.01 nat, top-1 agreement ≥98% over 48 fixed teacher-forced
positions. For example, K/V-only Lt with Tensile fallback passes the KL bounds but matches top-1
at only 47/48 positions. No thresholds were relaxed.
[Criteria](evidence/quality-criteria.json), [final results](evidence/stable-quality-screen-summary.json).

Historical PLE RAM, worker-count, NUMA/helper-affinity and inline-issuer screens did not establish
a reliable gain. They are retained as rejected screens, not timings to combine with the stable
backend. The complete decision list is in [pr1107-decisions.json](evidence/pr1107-decisions.json).
No full chunk/ring/MTP/KV/PCIe sweep or strict memory-binding/interleaving comparison was performed.

## Correctness and limitations

The gfx1201 signed-zero fix recorded 991,300 FP16 mismatches in three real IQ3_S format combinations,
all involving zero signs; after the fix, all seven format combinations had zero FP16 bit mismatches.
That is a correctness fix, without a speed claim. This observation relates to
[#1474](https://github.com/Niko1221/Strata/issues/1474) and
[#1540](https://github.com/Niko1221/Strata/pull/1540); the archived local fix is architecture-scoped
and is not a measurement of #1540's broader patch.
[Parity log](evidence/zero-fix-native.log.gz),
[state comparison](evidence/zero-fix-state-comparison.json),
[generation comparison](evidence/zero-fix-generation-comparison.json).

The two quantizations and two engine versions each passed eight product-style HTTP task checks:
JSON, arithmetic, Python execution, tool-call, multi-turn reuse, 32K/128K three-position retrieval,
and SSE cancellation/retry. That is **32/32 task checks**, not 32 HTTP requests. The original-setting
strict state/work mismatches remain failures; task success does not override them. These are local
regressions, not a general answer-quality benchmark.
[IQ3_S responses](evidence/pr1107-product-http-iq3s/),
[IQ2_XS responses](evidence/pr1107-product-http-iq2xs/).

Native IQ/MMQ, MTP, auto cache and KV streaming were existing engine capabilities; this work does
not assign separate gains to them. Windows, other GPUs, concurrent serving, other variants, and the
newer upstream engine were not tested by this report. Resource observations did not establish the
cause of the timing bands. No confidence intervals or universal throughput claims are made.

## Reproduction and archive checks

[REPRODUCE.md](REPRODUCE.md) describes the frozen source, build, assets, single-device launcher,
and performance commands. [harness/](harness/) contains the recorded measurement scripts.
The short-gather environment switch also disables the baseline's existing long-input grouping;
use separate reconstructed baseline/candidate binaries for a code-only comparison.

Run the archive check without a GPU, from the repository root:

```sh
python3 bench/results/2026-10-08-community-r9700-linux/verify_report.py --check-sources
```

It checks exported hashes, reconstructs changed measured sources, recomputes medians and paired
reductions from every formal request, validates the 94 paired + 12 absolute-speed requests, and
checks the four eight-task HTTP result sets. [coverage.json](coverage.json) records this audit.
It does not rerun inference or turn an experimental result into performance acceptance.

Absolute checkout/asset/home prefixes are replaced with placeholders, and JSON arrays/metric snapshots
use compact layout without changing their values. Hashes inside original records still identify
the original bytes; [export-manifest.json](export-manifest.json) separately
records original and published hashes. Included compressed logs/fixtures are lossless after that
stated path substitution. Large model files, engine binaries, raw logits/tensors, profiler traces,
build logs and raw process/resource streams are omitted. All formal request results, warmups and
the regressions used by the performance tables are included.
