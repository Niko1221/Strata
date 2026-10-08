# gfx906 Q2_0 down: exact bit spreading (opt-in mode 15)

Base: PR #1187 head 2a7503d82f39ac445617d61a9fd82282ebaeef33.
Mode 13 is the baseline: the previously qualified Q2_0 gate/up optimization.
Mode 15 keeps that gate/up implementation and changes only Q2_0 down unpacking,
for Flash-Next H=2560 / FF=640 on gfx906. Upstream default mode 7 is unchanged.

## Implementation

The down kernel retains its 256-thread launch, 64-row block, four-entry LDS
staging, weight addresses, dp4a accumulation, floating multiplication and
warp reduction order. It replaces four byte-permutation operations per
packed word with integer bit spreading. Two shift/OR/mask stages place the
four 2-bit values into bytes. Adding 127 and XORing 128 maps them to signed
{-1,0,1,2}; each byte is at most 3, so no carry crosses bytes.

Gate/up deliberately retains its existing helper. Other down formats retain
their existing path. No quantization, sampling, model or KV change is required.
Set STRATA_EXP_MODE=15 to opt in; STRATA_EXP_MODE=13 restores the prior path.

An initial variant reused gate/up's exact unpacker directly (experimental
mode 14). Its 64K/1024-output ABBA showed only +0.4565% TG. The two-stage
bit-spread version was faster in every one of 72 component comparisons with
that variant. Mode 14 is not retained in this patch.

## Evidence

Two gfx906 16 GiB GPUs, Xeon E5-2698B v3 (AVX2), 128 GiB RAM.
Model: shefowl Hybrid IQ3_XXS-Q2_0, revision
55568c1b41d2e381a59a281447fb10935c3c6d0a. Same software, container,
profile, MTP artifacts and build settings as the preceding
[gate/up report](../2026-10-06-gfx906-q2-gu-global/README.md).
SATA model/runtime, split at layer 27, 204800 context capacity,
INT8 KV with 32768 resident cells, PLE locked in RAM, prefill 4096,
MTP4 plus suffix3, 19078 resident experts, no prompt reuse.
Vision is disabled for native text measurements only.
Clocks were not locked. Exact peak power/RSS/VRAM and TTFT were not measured.

### Component correctness and performance

- Exhaustive integer witness: all 65536 16-bit words, both packed outputs
  (131072 checks), match the original {-1,0,1,2} mapping.
- 432 real-weight cases versus mode 13, on both GPUs: all 48 layers with
  mixed group lengths, padded grids and two seeds; plus layers 0/23/47,
  G=1/2/8/32/64 and T=1..8.
- Every gate/up/SwiGLU value, quantized activation byte and final output
  matched bit-for-bit.
- Down speedup: minimum 1.0254x, median 1.1255x, maximum 1.2658x.
- Whole-expert speedup: minimum 1.0118x, median 1.0586x.
- Component timing uses three interleaved rounds and their minimum,
  not a statistical confidence interval.

### Full model: controlled comparison

Fresh process each run; A13-B15-B15-A13-A13-B15; 2048 outputs per run.
Fixed expert placement (--adapt-every 0), greedy sampling (temperature 0,
top_p 1, top_k 1), seed 12345. All six outputs are identical per workload.
Every candidate repetition is faster than every baseline repetition.

| Prompt | Baseline aggregate TG | Candidate aggregate TG | Change |
|---|---:|---:|---:|
| 4096 | 50.078778 | 50.695579 | +1.2317% |
| 65536 | 47.836124 | 48.464846 | +1.3143% |

Aggregate TG is total generated tokens / total decode time, not the arithmetic
mean of rates. One 4K candidate and one 64K baseline offered one extra draft
(cause not isolated); accepted counts and emitted token sequences matched.
Per-arm rates, PP, counters and output hashes are in results.json.

### Production-like adaptive comparison: important limit

A separate six-run series per prompt used normal adaptation and sampling
temperature 1 / top_p .95 / top_k 20, 4096 output tokens each.

| Prompt | Baseline aggregate TG | Candidate aggregate TG | Observed change |
|---|---:|---:|---:|
| 4096 | 50.276978 | 51.495628 | +2.4239% |
| 65536 | 44.822750 | 44.834803 | +0.0269% |

Long outputs differed, including between baseline repetitions. Draft counts
and routed work therefore differ. These rows are observational, not
equal-work speedup or quality-equivalence evidence. In particular, no stable
production-like long-output 64K gain has been established. The fixed-placement
result must not be presented as a guaranteed production gain.
Context capacity 204800 does not qualify a fully occupied 200K prompt.

## Reproduce

Use the pinned HIP build and model artifacts in the preceding report.
Build strata and native_expert_bench. Run the supplied unpack_exact.py.
Generate the same synthetic prompts with the preceding report's fixtures.py.
For component comparisons use native_expert_bench with reference mode 13 and
candidate mode 15; STRATA_BENCH_MIXED, STRATA_BENCH_PAD and STRATA_BENCH_SEED
exercise the same layouts documented there.

run.py launches one native --serve measurement and records output hashes.
Pass the engine command and the model arguments after --. For the controlled
comparison add --greedy to the helper and --adapt-every 0 to the engine.
Use mode order 13,15,15,13,13,15 and --tokens 2048 for each prompt. For the
adaptive comparison omit those two controls and use --tokens 4096.
Keep split, KV, profile, model, MTP and runtime flags constant between arms.
Do not run competing GPU workloads.

## Scope

No AMD architecture other than gfx906, other expert geometry, full-window
responsiveness or broad model-quality qualification is claimed.
This patch does not contain deployment configuration, secrets, user prompts,
earlier experimental kernels or changes from PR #1209.

## Cleaned-source validation

The initial mode14 branch was removed; mode15 retains the two-stage unpacker.
The cleaned source rebuilt and passed another116 component cases: all48 layers
on both GPUs, singleton/small-group checks and two non-Q2 Coder controls.
A fresh controlled4K pair emitted the same2048 IDs as the preceding series:
50.156740 ->50.477167 tok/s (+0.639%). This pair is a smoke/consistency check,
not a replacement three-repeat performance study of the cleaned binary.

Clean executable SHA256:
3fdb12152c5e148c7af241cc4cde0e4f71494f657bba48f6ca9bb363b90757f6
Kernel source SHA256:
56fa5ff433b3c7a4eb9af208b005027affb2987c625a14f63a7f403a8cf01c36

The exact-integer witness and helper parser's positive/three negative cases
passed. The helper was prepared after the main measurements; the recorded
series used the local bounded native-protocol driver with the same controls.
