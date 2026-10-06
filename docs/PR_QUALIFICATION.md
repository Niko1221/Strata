# Combined change: validation and evidence

This submission combines the reviewed QFUSE correctness, opt-in concurrent
throughput, and prefill-monitoring changes. Its runtime source matches the
previously tested integrated candidate exactly; packaging adds documentation
and a portable workload driver, not new engine behavior.

## Fresh pre-publication checks, 2026-10-06

- 64 server/setup tests passed in 32.492 seconds, run from the combined source.
- Six native regression executables passed: batch segment layouts, partial
  recurrent-state commits, expert multi-row processing, GR, QFUSE quantization
  boundaries, and QFUSE GDN. The segment tests cover 9,840 layouts and 100
  partial commits across 16 two-slot layouts.
- Native tests reuse the previously built integrated binaries, whose source
  matches this branch. They were rerun, not rebuilt. Binary hashes and logs
  are in [the qualification evidence](evidence/pr-qualification/publication-checks.json).
- The diff passes whitespace checks. The daily service is unchanged.

Run the server checks with:

```sh
python -m unittest serve.test_reasoning_rescue serve.test_reasoning_tools serve.test_restart_waiters serve.test_parallel serve.test_prefill_phase tools.test_setup_hotfix_tag
```

With `STRATA_BUILD_TESTS=ON`, build and run `batch_segments_test`,
`batch_segment_gdn_test`, `expert_multi_test`, `gr_parity`, `qfuse_quant_test`,
and `qfuse_gdn_test`. CUDA kernel tests require a supported GPU/toolchain.

## Model and performance evidence

[Essay and tool-driven coding benchmarks](PR_WORKLOAD_BENCHMARKS.md) report
three paired runs per workload on an undervolted RTX 5090, Ryzen 9950X3D,
48 GB DDR5-6000. The baseline is v0.1.40.1 plus a two-line MTP metadata fix.
Four-stream decode improves from 146 to 200 TPS for essays and 167 to 326 TPS
for coding. These are output throughput measurements, not equal-quality
task-completion claims; coding structure checks passed 12/12 versus 11/12.

Earlier model evidence is deliberately distinguished from that fresh A/B:

- Controlled QFUSE off/on generation matched 7,638/7,638 tokens across solo,
  upstream concurrent MTP, and wide batches. Expert placement and CPU
  arithmetic were controlled. See [production summary](evidence/pr-qualification/production-summary.json).
- The QFUSE correction restored all 150 teacher-forced full-vocabulary
  positions byte-for-byte after isolating producer quantization differences.
  See [QFUSE rationale](QFUSE_CORRECTNESS.md) and
  [trace provenance](evidence/pr-qualification/provenance.json).
- Normal adaptive comparisons against the corrected older v039 candidate
  passed the recorded absolute screens: mean KL <= .001, maximum KL <= .02,
  mean TV <= .01. The stricter .0001 mean-KL trigger still flags differences,
  including self-repeats. These are not pristine-upstream quality comparisons.
  [Comparison summaries](evidence/pr-qualification/numerical-comparison-summary.json)
  include source report hashes; full traces remain in the local audit archive.
- A small 16-question screen scored 15/16 on both builds, with the same error.
  [Results](evidence/pr-qualification/answer-comparison.json) are a smoke check,
  not a broad model-quality evaluation.

Normal adaptive outputs are not claimed to be bitwise identical. Two numerical
repetitions and these bounded workloads do not establish universal quality
equivalence or long-context smoothness. AMD and multi-GPU fallback paths are
preserved in code but were not tested on that hardware. No new c=1 speedup is
claimed. Wide speculation remains opt-in.
