# Standalone Spin(8) reference library

This directory builds independently of Strata and GGML. It does not enable an
engine path by itself. CPU transforms support F32 and F64 storage with FP64
coefficients and accumulation. F16 and BF16 CPU operations return
`SPIN8_UNSUPPORTED`; their policy tolerances are not a support claim.

Build and test on Windows:

```powershell
cmake -S spin8 -B H:\Strata\build\spin8-cpu -G "Visual Studio 17 2022" -A x64 -DSPIN8_ENABLE_CUDA=OFF
cmake --build H:\Strata\build\spin8-cpu --config Release
ctest --test-dir H:\Strata\build\spin8-cpu -C Release --output-on-failure
```

The optional CUDA target requires `SPIN8_ENABLE_CUDA=ON`. It is off by default.
Its API and supported storage formats are declared separately in `spin8/cuda.h`.
The standalone CUDA test build also includes the Strata native adapter test.

## Mathematical convention

The real octonion basis is `1,e1,...,e7`. The positive oriented triples are
`(123),(145),(176),(246),(257),(347),(365)`. Each imaginary left multiplication
matrix `Li` is skew symmetric. Alternativity gives
`Li Lj + Lj Li = -2 delta(i,j) I`. The real symmetric generators are

```text
Gamma0 = [ 0   I ]        Gammai = [ 0    Li ]
         [ I   0 ]                 [ -Li  0  ]
```

They satisfy the Cl(8,0) anticommutator `Gammaa Gammab + Gammab Gammaa =
2 delta(a,b) I`. Chirality is the ordered product `Gamma0 ... Gamma7`.
A fixed block permutation places its +1 eigenspace before its -1 eigenspace.
Both have dimension 8. The diagnostic generator API returns this same basis.

A plane `(a,b,theta)` means `U=cos(theta/2)I-sin(theta/2)Gammaa Gammab`.
Factors act on column vectors in array order; the last factor is leftmost in
the product. Conjugation gives `U Gamma(v) U^T=Gamma(Rv)`. In particular,
`R e_a=cos(theta)e_a+sin(theta)e_b`. The two diagonal blocks of the SAME `U`
give the positive and negative half-spin matrices. They are not copies of the
vector matrix. Vector rotation by 2pi is identity while both half-spin matrices
are -I; all three are identity at 4pi.

Each plan is audited before publication. The audit calculates
`EF=||M^T M-I||F/sqrt(8)`, the maximum Gram entry error, and a pivoted LU signed
log determinant independently for each view. It requires determinant sign +1,
`abs(logabsdet)<=1e-12`, both Gram errors <=1e-12, and the generator intertwiner
error <=1e-12. The public matrix audit also reports negative/singular matrices;
it does not silently replace determinant sign with an absolute value.

## Buffer and reduction contract

Public C declarations are in `include/spin8/spin8.h`. Strides count scalar
elements. The caller owns buffer capacity and lifetime. The library checks
arithmetic overflow and overlapping ranges, and rejects partial aliasing.
Exact in-place transforms and reduction are supported. Validation and staged
results ensure no output write occurs on a failed CPU operation. Immutable
plans can be shared across threads; monitor state requires caller serialization.
The caller must not pass invalid memory or concurrently modify input buffers.

Every complete group of 8 coordinates uses the chosen view matrix. A remaining
1..7 coordinates use identity. This policy supports arbitrary head widths.
F32 forward/inverse results use a 1e-5 tolerance and F64 results use 1e-12.
The declared F16/BF16 policy coefficients are 2e-3 and 2e-2; CPU operations for
those storage formats remain unsupported.

`spin8_reduce` averages or weights three equally shaped scalar views. Weights
are finite, nonnegative, and sum to one within 1e-12. Null weights mean exactly
the stored three double values `1/3`. This convex sum is not itself a group
representation and does not preserve norm in general.

The reference matrix multiplication evaluates each view dot product and then
weights those logits. Reference attention applies that weighted reduction before
one stable softmax, then multiplies the original V. Its optional byte mask uses
nonzero for allowed positions. An all-masked row is an error. Each call handles
one attention head; callers invoke it for arbitrary numbers of heads and map
GQA query heads to their K/V heads explicitly. The reference performs real
view transforms even though exact orthogonality implies all three logits equal
the original-space dot product. A performance or model-quality gain is not
implied by this identity.

## Signed normalized Hadamard

The eight-dimensional convenience API uses an 8-bit negative-sign mask.
The general API supports power-of-two blocks from 1 through 1024, including
Bonsai's 128-element block, with explicit +/-1 signs or all positive signs.
Forward is `HD`; inverse is `DH`, where H is normalized Sylvester Hadamard and
D is diagonal signs. Complete blocks transform, and incomplete tails stay
unchanged. The implementation uses FP64 butterflies and final storage casts.

General factor audit uses the actual rounded coefficient `c` for the requested
F32/F64 coefficient model: `EF=max_error=abs(n*c*c-1)` and
`logabsdet=n*log(abs(c)*sqrt(n))`. The sign is the product of D's signs times
the Sylvester determinant sign (-1 for n=2, +1 for n=1 or n>=4). These are
analytical factor results for rounded coefficients. They are distinct from the
8x8 numerical LU audit and measured runtime forward/inverse error. F32
butterfly arithmetic is not claimed to be an exactly linear matrix operation.

## Monitoring and evidence boundaries

`spin8_monitor` is an explicit scheduler. Its caller supplies monotonic token
counts and elapsed seconds. Audit is initially due, then due at 1024 tokens OR
60 seconds since the last record. Every typed non-OK result invalidates the
monitor until an explicit reset; a later success cannot silently clear failure.
This standalone scheduler does not launch a background thread or intercept
engine execution. Integration must check it and gate its own cache admission.

The CPU test independently calculates Clifford anticommutators, chirality,
random group products and triality intertwining; tests the 2pi/4pi cover;
checks all 256 eight-element Hadamard sign masks; compares general Hadamard
against a dense parity oracle; checks strided inverse, tails and F32 storage;
compares matmul/attention against original-space arithmetic; and exercises
invalid inputs, aliasing, nonfinite data, overflow and sticky monitor failure.
A separate C translation unit verifies the ABI. These tests do not prove
engine integration, CUDA support, model loading, throughput or quality.

## Versioned query adapter

query_v3.h derives its seven skew generators from the same audited Clifford
blocks, and audits the normed-tensor polarization identities. It supplies the
bounded rational tangent chart, a biased rank-eight controller consuming
already normalized Q, and its analytic CPU VJP. Each complete block8 uses
A(c)=c0*I+sum(ca*Ja), with a shared c per head row. A copied tail still feeds
the controller. Zero output weights/bias give exact identity. This family is
seven dimensional, not all SO(8); its query-dependent Jacobian need not be
orthogonal. The older query.h controller/chart remains versioned separately.

The adapter belongs after model Q normalization and before position handling.
The three-view codec belongs after position handling. Those ordering contracts
are not a claim that a qualified full-model v3 path already exists. See
../_docs/2026-10-23_integrated_triality_algorithm_Codex.md for the combined
operator, error objective, memory comparison and remaining acceptance gates.

query_v3_cuda.h provides an opt-in fused F32 GPU controller/chart/transform.
Each call shares one parameter set across its rows. Byte strides/capacities,
explicit stream and caller-owned device buffers are required. The API enqueues
one kernel without allocation, copy or synchronization. Double arithmetic and
explicit separate product/sum rounding favor reference accuracy. Nonzero device
error flags require discarding all output. CPU/GPU tanh differences still
require profile-specific numerical checks; universal bitwise equivalence is
not asserted. GPU codec/VJP, per-head multi-controller fusion and model wiring
are separate work. Standalone CUDA tests and sanitizers do not prove throughput.
