# Real-weight MoE layer on Ascend A3

## Optional compensated path, 2026-10-22

With `STRATA_S2_COMPENSATED=ON`, three fresh A3 processes passed the original
test suite: **21/21 complete MoE fixtures and 1,536/1,536 forced experts**.
All input hashes, real weights and error budgets match the same-day OFF
baseline. That baseline reproduced expert 423's RMS failure and returned
exit 1. The three ON processes returned 0. See the
[comparison](evidence/compensated/COMPARISON.json) and
[summary](evidence/compensated/SUMMARY.json).

| Check, three final ON processes | Observed result |
| --- | --- |
| Complete MoE fixtures | 21/21 passed |
| Forced expert fixtures | 1,536/1,536 passed |
| Matched-input FP64 projection bounds | 6,480/6,480 passed |
| Quantizer codes / scales | 4,320/4,320 exact codes, scales within one ULP |
| Maximum forced-expert relative RMS error | 1.279868e-7 |
| Maximum complete-layer relative RMS error | 2.161990e-6 |
| Input, weight and guard checks | Passed |
| All layer/expert output hashes across processes | Stable |

Expert 423's relative RMS error fell from `7.467131e-4` to `7.509644e-8`;
its intermediate codes now match the independent FP64 chain. The
[diagnostic](evidence/compensated/diagnostic-compensated.json) retains the
stage comparisons. The full-layer maximum error did not improve uniformly:
its maximum normalized error rose from the original `2.749228e-6` to
`2.790524e-6`, still within the unchanged `1e-3` limit. This is an accuracy
regression fix for the tested expert path, not a model-quality claim.

The optional kernel first multiplies the FP16-derived scale by each exact
integer chunk dot, then compensates both product and addition rounding.
Compensating addition alone fixed 423 but introduced a failure at expert 328
(`9.297341e-4` relative RMS). That candidate was rejected. The final path
passes both experts and the entire sweep; no expert-specific branch exists.
See the [algorithm and input assumptions](../fused/README.md).

Both OFF and ON clean A3 builds also passed all 21 original synthetic
projection cases. Two additional cancellation regressions have independently
known results, `1` and `-2^-22`: ON returned both exactly; OFF lost both and
returned exit 1 as an expected negative control. The rebuilt OFF shared
library is **byte-identical** to the original A3 library, SHA-256
`e49fd7ae0bf22b8cc80ffc1b76acbc42aa1a6f2eabd7220e897326fc3dcf86b9`.
The ON library is
`e3ba27847b24579c3180ce844f0835592f6b826b0bc31e2e7d08bba03e2933a7`.

Failed attempts are retained in `evidence/compensated/`: ordinary CMake
`target_compile_*` calls did not reach CANN's external device compiler, and
three runs of that purported ON build reproduced the old failure. Its binary
was identical to OFF. CANN's `ascendc_compile_*` interface fixed the build
configuration. The subsequent sum-only candidate failed expert 328. Archives
contain these reports, logs and source versions, not just the passing runs.
The probe now saves all numerical failures and returns exit 1; previously it
printed FAIL and wrote `passed: false` but returned 0.

Build commands, actual device flags, source/binary hashes, raw reports and
rejected attempts are indexed by the new
[evidence manifest](evidence/compensated/EVIDENCE_MANIFEST.json).
The environment and weights are the same as below. ON was not tested on A2,
and its performance was not measured. Full inference and the other limits
below remain untested. The option defaults to OFF; production engine,
installer and root build files are unchanged.

## Original OFF path, 2026-10-21

Measured on 2026-10-21. Seven complete layer-zero MoE fixtures passed in three
fresh processes. A forced sweep covered all 512 experts. It reproduced one
strict FP64-chain RMS-budget failure, expert 423, in every process.
**The complete test set is not an all-pass result.**

| Check, three complete processes | Observed result |
| --- | --- |
| Complete MoE sublayer fixtures | 21/21 passed |
| Forced single-expert fixtures | 1,533/1,536 met both original error budgets |
| Native S2 projections vs FP64 with actual quantized inputs | 6,480/6,480 within the forward bound |
| Activation quantizer checks | 4,320/4,320: exact INT8 codes, scales within one ULP |
| Router IDs, including zero-input ties | Exact in all complete-layer fixtures |
| Inputs and resident weights | Unchanged |
| Layer and expert output hashes | All stable across three complete processes |

Maximum complete-layer relative RMS error was `2.160978e-6`; maximum absolute
error divided by the largest reference output magnitude was `2.749228e-6`.
The largest projection error consumed `0.044429` of its FP32 forward bound.
The projection binary is unchanged from the earlier A3 experiment:
`e49fd7ae0bf22b8cc80ffc1b76acbc42aa1a6f2eabd7220e897326fc3dcf86b9`.

## Environment and real weights

Logical NPU zero: `Ascend910_9362`, CANN 9.0.0, driver 25.5.5, aarch64 Ubuntu
20.04.5, Python 3.11.4, torch 2.7.1+cpu, torch_npu 2.7.1.post4, NumPy 1.26.4;
HF32 disabled. Only one of the two exposed logical chips executes the probe.
See the earlier [A3 environment/build report](../a3/RESULTS.md).

Eight real tensors total 713,116,160 bytes. They come from Strata's pinned
Q2_0 GGUF revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`. The three
routed tensors contain all 512 experts. The shared expert retains its
IQ4_XS/Q3_K/Q5_0 weights, decoded on the host with existing Strata references
and executed with FP32 NPU operations. Router weights are BF16 widened to
FP32. This shared branch does not emulate legacy shared-activation quantization.

Independent Hugging Face origin reads on the PC and ModelScope reads on A3
matched **all eight tensor hashes** and the GGUF header-prefix hash. Exact
HTTP ranges, sizes and independent boundary re-reads were checked. ModelScope
file commit: `c529e4fc7fccb4b91a0257f22eaf89189416cba9`. The full 37.6 GB shard
was not downloaded or independently hashed. Both manifests are in
[evidence](evidence/).

Inputs are synthetic FP32 hidden states: zero, normal batches of 1/2/4/8,
three small-scale inputs and five large-scale inputs. They are not prompt
activations. The boundary is router → ten routed experts → weighted sum plus
gated shared expert. Each expert includes gate/up → SwiGLU → down. Framework
operators execute routing, nonlinearities, quantization and accumulation on
the NPU; a Python host loop dispatches selected experts. This is a correctness
harness with no throughput claim.

## Expert 423: quantization boundary

The predeclared limits were maximum normalized error `1e-3` and relative RMS
`5e-4`. Expert 423 had maximum normalized error `5.895191e-4` and relative RMS
`7.467131e-4`, failing the RMS limit. The limits were not increased. Input and
output hashes reproduced exactly across processes.

The [targeted diagnostic](evidence/expert-423-diagnostic.json) found one
intermediate code difference, column 196:

| Quantity | Native projection chain | Independent FP64 chain |
| --- | --- | --- |
| FP32 SwiGLU intermediate | -0.10586678236722946 | -0.10586674511432648 |
| Scaled value before rounding | -19.500003814697266 | -19.499990463256836 |
| INT8 code | -20 | -19 |

The native quantizer matched the host quantizer on the actual native
intermediate. Holding both native gate/up outputs also removed the code
difference in the host nonlinear calculation. Native down with its actual
codes used `0.012265` of the projection bound. Feeding the reference codes
through native down reduced maximum normalized error to `9.101843e-8`.
These checks support a projection-rounding-induced crossing of a discontinuous
quantization boundary. They do not establish a production-engine bug or
strict FP64-chain parity for every expert.

## Passed, failed and not run

Passed: the table's checks; eight cross-source hashes; the new CPU reference
build and execution; Python syntax and source/binary/evidence hashes. Maximum
torch_npu peak allocation: 828,427,776 bytes; reservation: 861,929,472 bytes.
These are framework allocator counters, not board-wide HBM peaks.

Failed/interrupted, retained:

- The first precheck required bit-identical scales. Of 640 scales, 31 differed
  by one ULP; codes matched. Scale checks were changed to record and allow
  one ULP before the first real-weight run. End-to-end limits were unchanged.
- The initial real-weight run stopped at expert 423. The collector was changed
  to finish all numerical fixtures and record failures. Three subsequent
  fresh processes each completed all 512 experts and reproduced that failed
  expert. All three reports retain `passed: false`. Exit zero means collection
  completed; numerical success is the JSON `passed` field and PASS/FAIL marker.
- A3's direct Hugging Face transfer failed; hf-mirror returned HTTP 429 for
  the weight transfer. A slow WebIDE upload was canceled and the page
  reconnected. Fixed-commit ModelScope reads then matched all origin hashes.
  Transport failures are separate from model execution.

Not run: original BF16 weights, other layers, prompt activations, attention,
GDN/QSA state, gated residual connections, default AVX/shared-activation
parity, Strata routing/cache integration, full inference, vision, HCCL, the
second chip, long context, model-quality scores or performance comparisons.
Production engine, installer and root build files are unchanged.

## Reproduce and inspect

The [README](README.md) contains build/fetch/probe commands. Use three fresh
processes and distinct reports. `summarize.py` aggregates reports and was
executed against the measured raw reports after GPU execution. Measured source
and binary hashes are in [SUMMARY.json](evidence/SUMMARY.json).

`run-1.json.gz` preserves the stopped run. `run-2.json.gz` through
`run-4.json.gz` are complete runs with unchanged JSON bytes. For example,
`gzip -dc evidence/run-2.json.gz` reads a raw report. Logs, earlier sources,
weight manifests and the diagnostic are included. Public logs replace
installation/workspace paths. The [manifest](evidence/EVIDENCE_MANIFEST.json)
records original export and public file checksums. Weight binaries are excluded.
