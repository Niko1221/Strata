# Real-weight MoE layer probe on Ascend

This is a standalone extension of the S2 projection experiment. It checks the
complete **MoE sublayer** of layer zero: the 512-way router and normalized top
10, all three projections and SwiGLU in each selected expert, the gated shared
expert, and the final weighted sum. Attention, gated residual connections,
state/cache, model loading and token generation are outside its boundary.

The weights are real. The FP32 hidden-state inputs are deterministic synthetic
fixtures; they are not activations captured from a prompt. The probe uses the
Q2_0 GGUF at the revision pinned by Strata's installer, not original BF16 model
weights. It is not a model-quality evaluation or default-engine parity claim.

## Weight and execution contract

`fetch_layer.py` downloads only eight tensor byte ranges from shard one at
`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, revision
`ed59f92082b1e93c0e96d60a8b11aab089b52f09`. It parses the header with the
repository's existing `tools/gguf_reader.py`, requires exact HTTP 206 ranges
and sizes, re-reads each tensor's boundaries, and records SHA-256 values. The
header prefix hash is pinned from an independent Hugging Face origin read.
The whole 37.6 GB shard is not downloaded or hashed. Weight files are external
test inputs and must not be committed to this repository.

The real shared-expert weights have mixed formats: IQ4_XS gate, Q3_K up and
Q5_0 down. The router and scalar shared gate are BF16. The existing Strata
scalar decoders convert these weights to FP32 on the host, then the probe uses
FP32 NPU matrix operations and activations for this shared branch. This does
not reproduce the engine's legacy quantized shared-activation path.

The routed experts retain all 512 real Q2_0 gate/up/down matrices as packed
codes and FP32 scale planes on logical NPU zero. The Ascend C kernel
from `../fused` computes their projections. Its optional
`STRATA_S2_COMPENSATED` build setting retains product and addition low parts
when accumulating the 32-element chunk contributions. It is OFF by default.
Its finite-normal-input assumptions are in the [projection README](../fused/README.md).
Framework operators implement
32-element activation quantization, SwiGLU, routing and accumulation on the
NPU. Equal router scores choose the lower expert ID. Expert dispatch is a
Python host loop that reads the selected IDs. Each raw ACL launch synchronizes
the framework host queue. This is a correctness harness, not a throughput
implementation.

## Independent checks

The CPU reference uses the existing C++ scalar weight decoders and FP64 NumPy
products, sigmoid, routing, shared branch and weighted accumulation. It
independently transcribes the existing scalar FP32 activation quantizer.
Routed intermediate values are rounded to FP32 before the next quantization
boundary. Different FP32 reduction orders can change a subsequent quantized
code, so full-layer checks use explicit numerical budgets instead of a
bitwise claim.

The budgets were set before the first real-weight run:

- Exact router expert IDs, including the all-zero tie fixture.
- End-to-end maximum error divided by the oracle's largest output magnitude
  at most `1e-3`; relative RMS error at most `5e-4`. The denominator floor is
  `1e-12`. Router logits, route weights, shared output and routed output are
  also reported separately with the same budgets.
- Every routed projection is independently checked against FP64 using its
  actual device quantized input, with the existing cancellation-aware bound
  `gamma(4*K/64+4) * sum(abs(products))`. This local check isolates projection
  error from earlier quantization boundaries; it does not replace the
  independent full-layer oracle.
- Device activation codes must match the host quantizer. Scale differences
  are recorded and limited to one ULP: the NPU division preflight failed an
  initial exact-scale assertion, with 31 of 640 scales differing by one ULP.
  That failure is retained; the end-to-end budgets were not changed.
  Output guards, unchanged inputs and unchanged resident weights are checked.
- An explicit all-512-expert sweep tests each real expert with a forced ID.
  That sweep checks expert computation and does not claim natural router
  coverage of all experts.

## Run

Build the original decoder and fused projection for the target as documented
in the A3 report. In a separate scratch checkout, set `POC_ROOT` and run:

```sh
g++ -std=c++17 -O2 -shared -fPIC -I"$POC_ROOT/include" \
  "$POC_ROOT/bench/ascend_a2_s2/moe/decode_reference.cpp" \
  -o "$POC_ROOT/bench/ascend_a2_s2/moe/libstrata_moe_reference.so"
python3 "$POC_ROOT/bench/ascend_a2_s2/moe/fetch_layer.py" \
  --reader-root "$POC_ROOT/tools" --output "$POC_ROOT/real-moe-weights"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python3 \
  "$POC_ROOT/bench/ascend_a2_s2/moe/probe.py" \
  --weights "$POC_ROOT/real-moe-weights" \
  --build "$POC_ROOT/bench/ascend_a2_s2/fused/build-a3-release" \
  --reference "$POC_ROOT/bench/ascend_a2_s2/moe/libstrata_moe_reference.so" \
  --report "$POC_ROOT/real-moe-evidence/run-1.json" --sweep
```

For the optional accuracy path, configure a fresh projection directory with
`-DSTRATA_S2_COMPENSATED=ON -DSOC_VERSION=Ascend910_9362`, build it, and pass
that directory as `--build`. Do not reuse the OFF build directory. Run the
same fixtures and limits for both builds. `diagnose_expert.py` takes the same
weight/build/reference/report arguments and reproduces expert 423 by default.
It holds stages fixed to distinguish projection error from a later discrete
quantization change. It does not mark the full test suite as passed.

`probe.py` finishes numerical checks and saves the report before returning
exit 1 if any contract fails; success returns 0. Earlier collected failures
returned 0, so their original JSON `passed` fields remain authoritative.

The optional downloader flags `--endpoint https://hf-mirror.com --ipv4` affect
only that process. ModelScope is also supported, pinned to the source file's
commit `c529e4fc7fccb4b91a0257f22eaf89189416cba9`. Using
`--endpoint https://modelscope.cn` requires `--verify-manifest` pointing to an
independent Hugging Face origin `WEIGHTS.json`; all eight tensor hashes must
match. Keep `WEIGHTS.json`, raw reports and failure logs. Existing
probe reports are never overwritten. A suggested command is not evidence of
passing execution; measured outcomes belong in `RESULTS.md` after the runs.
