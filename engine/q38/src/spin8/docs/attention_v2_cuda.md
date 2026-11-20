# CUDA attention_v2 reference (bounded B2)

This is an opt-in standalone CUDA reference. It implements attention on the GPU.
It makes no speed, model-quality or memory-saving claim. Full phase B, runtime
integration and qualification remain pending separate review.

The public header is `spin8/attention_v2_cuda.h`. Link `spin8_cuda`; configure
`SPIN8_ENABLE_CUDA=ON`. This slice requires the CUDA 13.5 runtime API and accepts
only compute capability 12.0 (sm_120) with unified addressing. Other devices
return `SPIN8_UNSUPPORTED`. B1's CPU ABI and the older CUDA API are unchanged.

## Shared contract and supported inputs

`spin8_attention_v2_cuda_plan_create` takes the existing B1 config, its exact
byte size, the current device ordinal and an output handle. Set the config's
`dtype` to `SPIN8_F32` for this CUDA function. B1 CPU creation still accepts only
FP64. The CUDA constructor uses B1 plan creation internally to validate the same
geometry, source identity, mapping and fixed-basis semantics. It exports actual
FP64 coefficients through `spin8_plan_create` and `spin8_plan_matrix`. Those
coefficients have passed the existing Spin8 audit. No replacement rotation or
rounded F32 coefficient table is substituted.

Both config and request headers need the exact version, `struct_bytes`, and zero
reserved fields. The CUDA request has its own header and contains the B1 request
as `attention`. Its `operation` is `SPIN8_ATTENTION_V2_CUDA_REFERENCE`. Unknown
versions, operations, codecs, bases, logit orders, dtypes and position semantics
fail explicitly. Exceptions are contained at the C boundary.

Q, bias, gate and output use F32. K and V independently use F32 or F16, selected
by `k_dtype` and `v_dtype`. Each tensor has independent positive byte strides and
a byte capacity. Axes are `[token,head,channel]`; mask and bias axes are
`[query,query-head,key]`. Mask uses bytes, with zero excluded and any nonzero
value allowed. Mask is required. Optional bias and gate descriptors must be
entirely zero when absent. All Q/K/V/bias/gate elements must be finite, including
masked positions and unused KV heads.

The plan copies an arbitrary Q-to-KV head mapping. Contiguous groups and head
divisibility are not assumed. This permits MHA, GQA and MQA, including
noncontiguous mapping. Q/K widths and V widths can differ and need not be
multiples of eight or 128. Tests cover Q/K widths 64, 80, 96, 128, 130, 192 and
256 with different V widths, plus small boundary shapes.

The source ID is a nonzero caller assertion that Q and KV belong to one
self-attention source. Both request IDs must match the plan. Q/K must already
have position handling applied. Cross-attention, position processing, codecs,
learned controllers, multiple views and sessions are unsupported here.

## Numerical order

F32 and F16 inputs are decoded into caller-owned FP64 workspace. In fixed mode,
each Q, K and V row first takes optional normalized H128 on complete 128-wide
blocks, then the selected actual Spin8 matrix on complete eight-wide blocks.
Each stage leaves its own incomplete tail unchanged. Q and K always share the
same basis. All three audited Spin8 views are selectable, one per plan.

Scores use the caller's explicit finite `scale`, including zero or negative
scale. There is no implicit `1/sqrt(d)` replacement. The two logit orders match
B1: scale then bias then softcap, or scale then softcap then bias. A zero softcap
disables it; otherwise it must be finite and positive. Mask exclusion is
structural and never passed through softcap. There is one stable softmax per
query/head row and one weighted V accumulation.

The result takes the transpose Spin8 transform and then inverse H128, followed
by the optional elementwise gate in original coordinates. All intermediate and
final values are checked. A finite FP64 result outside the finite F32 range is
also an error before the final cast. Arithmetic uses FP64 workspace and
accumulation, with explicit separately rounded dot products. This is a
correctness-first reference; universal bitwise CPU/GPU agreement is not claimed.

## Enqueue, completion and errors

Every function returns `{status, cuda_error}`. Semantic or descriptor rejection
has a non-OK `spin8_status` and `cudaSuccess`. A CUDA runtime failure returns
`SPIN8_INVALIDATED` and the actual CUDA error. Creation and workspace-size query
leave their output argument unchanged on failure. No CPU attention fallback is
used.

Enqueue requires a real explicitly created stream on the plan's current device.
Null, legacy and per-thread default streams are rejected. All geometry, spans,
alignment, aliases, versions, semantics, pointer memory types and devices are
validated before enqueueing any work. Host validation failures leave output,
workspace and device status unchanged. Enqueue does not allocate, synchronize or
copy data to the host. It submits a status reset and actual CUDA kernels.

An OK return means only that submission succeeded. After successful stream or
event completion, inspect the caller-owned `uint32_t` device status in stream
order. Zero means numerical success. Bit 1 (`NONFINITE`) reports nonfinite input,
intermediate arithmetic or F32 output overflow. Bit 2 (`ALL_MASKED`) reports an
empty allowed set. Bits may combine. The status is reset on each invocation and
each captured graph replay; it is not a persistent monitor.

All output is staged. A separate final kernel reads the completed status and
copies payload values only if it is zero. A late error in any row therefore
preserves every output byte, including earlier rows. Successful publication
does not alter padding. This is transactional with respect to the checked
numerical errors, not an atomic multi-element write visible to racing consumers.
Consumers must wait for completion. On CUDA submission or execution failure,
discard the invocation, even if its status appears zero; runtime failure can
interrupt publication. Drain submitted work before reusing its storage. The
library cannot recover a broken CUDA context or guarantee output on such faults.

For a host-side consumer, enqueue a status copy to caller-owned host memory in
the same stream, synchronize/check that stream, and then check both the CUDA
result and copied status before using output. For device consumers, establish
the stream/event dependency and check status before consuming output.

## Ownership and memory

Config/request descriptors and host output arguments must be accessible and
naturally aligned host objects. Mapping and planes are copied at creation;
their storage may then be changed or released. Request descriptors may be
released after enqueue returns because kernel arguments are copied by value.
The immutable plan is device-bound. The library never changes the current
device. Creation allocates and copies once and belongs outside graph capture.
It uploads on a private nonblocking stream and checks stream completion before
publishing the handle or releasing its host upload buffer. Successful creation
therefore permits immediate use on any supported caller stream, without an
extra wait. Submission, completion and stream-destruction errors prevent
publication and trigger device-storage cleanup; `*out` remains unchanged. This
initialization wait is confined to construction. Enqueue remains asynchronous
and has no added synchronization.
Free the plan on its device only after all uses and graph replays are complete.
`cudaFree` may synchronize; plan destruction is outside the hot path.

Tensor bases and all strides must be aligned to their item size: four bytes for
F32, two for F16, one for masks. Workspace needs eight-byte alignment and status
four-byte alignment. All device buffers must be ordinary device memory on the
plan's device; host, managed and foreign-device memory are rejected. The caller
must provide live allocations whose true accessible capacity is at least the
declared capacity. The library checks arithmetic, required extent and address
wrap; pointer attributes identify memory type/device but do not prove allocation
length or pointer lifetime. Stale handles, dishonest capacities and concurrent
caller mutation are outside the contract.

Output, workspace and status must be disjoint from each other, every input,
descriptors and plan. Alias checks use entire declared capacities, including
padding. Input-only aliases are permitted. Output accepts padded/permuted
layouts that pass a conservative sorted-stride nonoverlap proof. Some unusual
injective layouts can therefore be rejected. Keep all device allocations live
and input data stable until completion. Independent invocations may share a plan
and read-only inputs, but must use distinct workspace, output and status.

For `Q=n_query`, `K=n_key`, `H=n_q_heads`, `J=n_kv_heads`, `D=d_qk`, `V=d_v`, the
workspace query returns exactly:

```text
8 * (Q*H*D + K*J*D + K*J*V + Q*H*V + Q*H*K) bytes
```

These are packed Q, packed K, packed V, staged output and the full score array,
in that order. All staging and scores are included. FP64 elements are contiguous
in their logical axis order. Scores are overwritten with unnormalized softmax
exponentials; row normalization is applied in the V accumulation. Bias, gate and
mask are scanned/read directly and have no hidden packed allocation. The tests
inspect packed Q/K/V after completion to verify actual transforms. Workspace
contents after an error have no numerical validity guarantee.

Add the caller's actual capacities for Q/K/V/mask/bias/gate/output and at least
four bytes for status to obtain per-invocation device memory. Each live plan adds
`512 + 4*H` device bytes for its matrix and mapping, including identity plans.
Persistent host storage is the opaque handle (`sizeof` its internal struct);
there is no retained host tensor or mapping. Startup also temporarily owns the
B1 validation plan and copied mapping, audited Spin8 plans, copied planes, and
a `512 + 4*H` byte upload buffer. Those startup allocations are released before
creation returns. The private initialization stream is also destroyed before
publication. CUDA runtime/context/stream/graph metadata and allocator overhead
are additional, implementation-dependent costs, not included in tensor bytes.

Kernels use scalar loops, with only eight FP64 local values for a Spin8 block.
Hadamard butterfly state stays in workspace. In the measured NVCC 13.5.37
sm_120 Release compile, all five kernels reported zero stack-frame bytes and
zero spill loads/stores. This observation is compiler/build specific.

## Capture and evidence boundary

Allocate all buffers, create the plan and finish uploads before capture. Enqueue
can be captured on the explicit stream. A graph retains device addresses and
by-value request scalars; changing the original host descriptor does not update
the graph. Keep plan, tensors, workspace and status live through all replays.
Serialize replays that share writable storage. The tests exercise capture,
multiple successful replays, a late NaN replay, and subsequent recovery.

The tests predeclare an absolute output tolerance of `2e-5` for both F32 and F16
against an independent FP64 oracle using the actual rounded inputs. They also
check transforms against dense matrix/Hadamard arithmetic, masks, all-masked
rows across CTAs, late NaN/Inf, output overflow, sentinels, ABI/geometry errors,
original-scale and gate-order negative controls, and independent streams/plans.
The first-use regression prepares the invocation storage, nonblocking stream and
uploads before creating each of 32 plans, then immediately enqueues with no
intervening caller allocation or synchronization. It checks output and actual
transformed Q staging. It is coverage of that ordering contract, not a guarantee
that every unfixed scheduling race will reproduce in a bounded run.
Windows tests enforce at least 3 GiB available physical RAM and 1 GiB GPU free
before allocations, with aggregate explicit device allocations below 256 MiB.
Insufficient resources print `BLOCKED` and return CTest skip code 77.

Exact build commands, observed results, warnings and remaining checks are in
[`../../_docs/2026-11-06_spin8-attention-v2-cuda_OpenCode.md`](../../_docs/2026-11-06_spin8-attention-v2-cuda_OpenCode.md).
Only bounded local fixtures have been exercised. Long contexts, throughput,
model loading, real-model quality, other architectures, multi-device execution,
allocation/runtime-failure injection and production integration are not qualified.
