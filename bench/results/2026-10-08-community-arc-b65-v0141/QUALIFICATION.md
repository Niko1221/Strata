# Scope of validation

Official v0.1.41 builds all 201 native targets. Its complete registered CTest
suite reports 30 tests: 26 pass, four fail. These fresh failures were preserved
and investigated before model benchmarks:

| Test | Finding |
| --- | --- |
| `iq_parity` | CTest supplies `--selftest` as a fixture directory. Separate actual IQ fixtures pass; native GGUF expert oracles over eight layers also pass. |
| `ple_parity` | The standalone Q2_0 fixture is unavailable. This case remains unqualified. The measured profile uses original IQ2_XS/native model artifacts. |
| `gr_parity` | Ordinary/native subchecks pass. Optional QFUSE and multi-QFUSE checks fail: the SYCL implementation explicitly returns false for those outputs. QFUSE is absent. |
| `s2_expert_grouped_parity` | Four old/new grouped cases differ bitwise. Every CPU-oracle comparison has zero rows outside tolerance; worst normalized error is 1.03e-7. Independent exact target-only/one-draft/MTP4 model-stream checks gate throughput. |

No engine code was patched or tolerance relaxed. The full suite does not pass;
these results qualify only the measured original IQ2_XS profiles.

Each 8K profile passes 21 native reference and 24 API/lifecycle checks, including
cancellation, restart, tools, queued isolation and near-8K capacity. Checks
compare actual token streams. Benchmark repetitions within a profile match
exactly. Different prefill sizes can change continuations and draft acceptance;
a decode-rate difference across them is not a pure kernel comparison. Arithmetic
accuracy is recorded separately from runtime validity.

The 262K profile separately passes five native reference checks and 12 API
checks at progressive/full context, including exact speculative/reference
agreement and overflow rejection. Individual full-window capacity checks are
separate from the three-run matched-input benchmarks.

Independent guards enforce exclusive GPU ownership, Gen4 x16, zero watched
PCIe errors, 30.5 GiB resident-VRAM ceiling, 32 GiB host-memory margin, an 80 GiB
cgroup without swap, bounded execution and sensor-specific thermal margins.
Generated content, native/API logs and temporary request configuration remain in RAM. Sanitized launch profiles and build metadata persist for reproduction. Core dumps, persistent SYCL cache and
crash capture during generation are disabled. Cleanup verifies payload exit
before deleting RAM content and restoring normal services and crash capture.
Only numeric/hash receipts persist. qualification.json records sample counts,
observed peaks and fault counts for every successful phase.

Before the measured build, the new per-device spin helper selected the UHD
display GPU's i915 driver, which chose 2000000 reads instead of the intended
20000 for the B65/xe. That attempt was stopped cleanly; its incomplete benchmark
is excluded. There was no hardware fault. After independent idle recovery, the
release was rebuilt with its documented STRATA_SYCL_SPIN_MAX=20000 option and
the full qualification repeated under new run IDs. No source patch or relaxed
test threshold. A separate pre-payload controller status-file race was repaired;
failed receipts remain local, and neither event contributes throughput data.
