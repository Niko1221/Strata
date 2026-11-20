# tqcodec reference primitives

Build independently with CMake and run CTest. This library currently provides a
CPU FP64 reference, explicit normalized scalar codebooks, LSB-first packed
indices at 2 through 6 bits, raw compressed-key dot products, common-coordinate
three-view residuals, and packed-sign Gaussian QJL residual correction.

The scalar primitives perform no rotation. The caller supplies the exact
transform order and codebook. The generic packed payload plus separate FP64 norm is a new reference
layout, not a byte-compatible implementation of any legacy GGUF/TurboQuant
block. It must not be labeled TQ4_1S, PQ2_0 or turbo4 in a model loader.
Byte-exact dialect adapters, CUDA codec kernels and model-engine integration are pending.

The separate CPU reference cache in cache.h owns three physically independent
packed K arrays and one shared quantized V array, snapshots caller transforms
and codebooks, rotates keys at insertion and queries at scoring, and uses one
online softmax. It supports clone, clear, masked attention and copied debug
readback. Its dense transforms are a correctness reference. They do not imply
that a supplied orthogonal matrix is a strict Spin(8) representation. Configure
with -DTQCODEC_SPIN8_TESTS=ON to test composition with the actual strict Spin8
library and its query controller. Cache persistence and positional relocation
remain outside the base cache API.

QJL accepts an explicit projection matrix, with sign(0)=+1. The caller must
version/hash the actual matrix rather than assume a C++ random distribution is
portable across implementations. The correction is sqrt(pi/2)*norm/m times the
signed query projections, scaled by beta in [0,1]. Beta=0 is the baseline.
Unbiasedness applies to fixed vectors and an independent ideal Gaussian draw;
it is not a softmax or autoregressive quality guarantee.

The size API includes payload and the separate FP64 norm. Full cache accounting
must additionally include rotations, codebooks, projection matrices, controller
weights, all three K buffers, V, alignment, allocator reservations and scratch.

Tests cover golden bytes, cross-byte packing, compressed vs decoded scores,
zero residual, correction normalization, rejected overlapping outputs, and
independent-sketch mean/variance. Successful primitive tests do not establish
runtime/model compatibility or performance improvement.

The v3 statistics helper reports beta/Attention-scale-aware score error
moments under independent Gaussian-sketch assumptions. distill.h provides a
detached-teacher normalized-output-MSE and KL loss plus student-query gradient.
It recomputes row scores using caller-owned scratch of 2*value_width plus
query_width doubles. No full token-square Attention matrix is allocated.
This is a CPU training reference; model captures and adapter optimization
are separate integration work.

The versioned cache create_v2 API can opt into one shared residual sketch per
key, with a copied Gaussian matrix, LSB-first signs and FP64 residual norms.
The correction is added before the existing Attention scale and the single
softmax. beta=0 allocates none of these arrays. bytes_v2 accounts for their
allocated capacities; the old bytes API rejects active QJL to avoid reporting
an incomplete total. Clone/clear and failed insertion include residual state.

calibrate.h fits fixed simplex consensus weights from captured per-view error
vectors, with optional shrinkage toward uniform weights. It uses streaming
Givens QR and direct residual comparisons to retain small independent error
directions. It belongs to calibration, outside decoding. QJL is disabled for
this linear fit; changing alpha requires residual/cache reconstruction.
The first-order objective and the adapter training example establish no
independent model/task quality improvement.

transport.h adds a separate owning FP64 plan for one proper orthogonal S
shared by every selected logical key. It evaluates attention with S^T*q,
preserving all physical K, V and residual-sketch bytes. New logical keys must
first be canonicalized with S^T. Existing QJL signs are retained; the query
projection becomes G*S^T*q. The API does not infer sign(G*S*e), handle different
S per row, persist position state, or establish model/RoPE relocation semantics.
Its external plan ledger is separate from both cache ledger versions.
