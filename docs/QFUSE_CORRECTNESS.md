# Preserve the separate quantizers' arithmetic in QFUSE

Base: upstream v0.1.40.1 (82f46a8). This change is independent of the wide
speculative batch implementation. QFUSE defaults and platform eligibility stay
as upstream defines them.

## Changes and reasons

- GR and GDN fused producers use the existing finite-half protection. Without
  it, large finite inputs can overflow the half-precision block scale or sum.
- Native CUDA quantization is compiled with fast math; the fused producers are
  not. Explicit native division reproduces that consumer's integer rounding at
  half-step boundaries. GR selects native versus IQ arithmetic per consumer;
  GDN follows its native projection. HIP retains its division convention.
- Batched GDN recurrence does not emit a fused q8 image, so it must still run
  the separate quantizer. Otherwise downstream projections read stale data.
- The one-token commit shortcut now uses the same QFUSE exclusion as capture.
  A window which did not self-commit must not skip the commit graph.

## Reproduce without model files

Configure the normal CUDA or HIP build with STRATA_BUILD_TESTS=ON and
STRATA_NATIVE_EXPERTS=ON. Build qfuse_gdn_test and gr_parity; on CUDA also build
qfuse_quant_test. Run:

```
ctest --test-dir <build> --output-on-failure -R "^(qfuse_quant_test|qfuse_gdn_test|gr_parity)$"
```

The CUDA boundary test constructs 8192 blocks close to quantizer rounding
boundaries and compares actual q8 bytes against native_quantize_q8_1. The GDN
test compares both floating output and q8 images over 64 combinations of row
width, output range, input magnitude and direct/graph execution. The GR test
covers both native and IQ consumers, widths 1..8 and repeated graph replay.

## Prior model evidence and limits

The integrated candidate at production source 3cfff22 was tested on NVIDIA
RTX 5090 with Swift IQ2_XS. The original mismatch was isolated to producer
quantization: 13 attention and 11 GDN mismatch events. Correcting finite guards
alone did not fix that trace; correcting division did. All 150 full-vocabulary
positions then matched the unfused reference byte for byte. Synthetic boundary
bytes differed 31603 times with legacy arithmetic and zero with the correction.

Controlled QFUSE off/on generation matched 7638/7638 tokens across solo MTP,
upstream concurrent MTP and the separate wide-batch candidate. Both concurrent
lifecycle suites matched eight transitions to their respective solo reference.
Those runs fixed expert placement and CPU arithmetic. They are historical
integrated-candidate evidence, not a fresh model qualification of this isolated
PR or a claim of normal adaptive determinism. Full evidence is preserved locally
at candidate tag candidate-v040-numerics-20261006; the review bundle identifies
the source files and original reports. Public evidence summaries and fresh
test logs are linked in [PR_QUALIFICATION.md](PR_QUALIFICATION.md).

No AMD or multi-GPU hardware test was available. The arithmetic fix deliberately
does not change global compiler math flags or enable QFUSE on new configurations.
