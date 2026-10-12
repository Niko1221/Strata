# Real-model backend adapter selection and conformance

## Order of implementation

Choose the backend after inspecting what is installed and which version exposes reliable save/restore. Favor one backend for the first real certification. Do not build a generic adapter layer for imaginary engines; a *small protocol* is enough to isolate the actual backend's quirks.

Potential sources for a concrete adapter include llama.cpp's per-sequence context/state APIs, a pinned Transformers checkpoint with an explicit no-cache forward path, vLLM's prefix-caching and KV management primitives, or an engine already used on the target host. Their available APIs, state lifetimes, and claimable copy-on-write behavior vary with version. Inspect installed headers, source and tests; record exact commit.

## Minimal required backend capabilities

```python
class BackendAdapter(Protocol):
    def identify_model_contract(self) -> str: ...
    def identify_execution_profile(self) -> str: ...
    def full_prefill_independent(self, token_ids: tuple[int, ...]) -> tuple[float, ...]: ...
    def make_lease(self, prefix: tuple[int, ...], checkpoint=None): ...
    def read_next_logits(self, lease) -> tuple[float, ...]: ...
    def consume(self, lease, tokens: tuple[int, ...]): ...
    def save_complete_state(self, lease) -> bytes: ...
    def restore_complete_state(self, data: bytes): ...
    def report_counters(self, lease) -> dict: ...
```

This is a semantic sketch, **not** a prescriptive method-for-method SDK. A backend may implement `full_prefill_independent` in another separately trusted model runtime. If the supposedly fresh oracle simply loads cached blocks from the same SUT path, it is not independent.

If a model uses recurrent state, query and checkpoint it explicitly. If save-state omits frontier logits, restore by reconstructing the correct last consumed boundary, not by double-feeding an already-consumed final token.

## Independent numerical oracle

Choose a fixed tokenized fixture, not a regenerated chat template. For each exact history `P||A[:i]`, run a no-cache/full-prefill oracle in a fresh request with all prefix-cache sharing disabled. Compare the entire final-step logit vector against the branch continuation's next logits before sampling.

For a frozen execution profile record at least:

- model weight file SHA-256 and adapter weight hashes;
- model architecture, tokenization identity, fixed chat template/input artifact hashes;
- attention/position configuration including RoPE and YaRN;
- inference backend commit/build flags, dtype and KV dtype, attention kernel;
- hardware identifiers, deterministic settings and `atol`, `rtol`;
- observed worst-case absolute/relative error, max normalized error and (optional) KL/top-1 changes;
- individual mismatch artifacts with exact token IDs/positions and branch lineage.

Never tune tolerance based on failing branch fixtures. Freeze based on independent baseline comparisons (e.g. no-cache vs fresh prefills under the same numerical profile) *before* running the actual branching tests. A tolerance broader than normal numerical variability needs investigation, not rationalization.

## Critical real-engine traps

- **Delayed token ingestion:** some generation loops sample/display a token and only consume it on the next iteration. Public snapshots cannot lie about consumed position.
- **Paged KV partial tail:** if the prefix ends mid-page, sibling writes must not mutate shared live bytes, quantizer scales or page metadata.
- **Cache identity:** equal last block or equal token count is not equal history; hash/reuse keys need full preceding prefix and semantic contract.
- **RNG leakage:** branch scheduler/batch order must not perturb branch-local sampling. Save draw cursor or full PRNG state.
- **Stop/grammar:** generation call work limits must not restart the generation scope. Saved mid-grammar states must survive.
- **Profile mismatch:** incompatible cache layout or model contract must be rejected or fully replayed under a compatible semantic model; never treat a shape check as identity.
- **Aborted decode:** partial backend writes must be isolated until fully consumed/validated and published.
- **Partial context:** sliding-window and recurrent models may have state beyond simple `KV[:n]`; do not truncate blindly.

## Choose guarantee per execution profile

A. **Semantic:** correct history and controller, next logits within frozen numerical tolerance.

B. **Deterministic tokens:** same profile, sampler stream and scheduling contract reproduces exact accepted tokens/controller bytes.

C. **Bitwise:** only claim if a specific pinned device, kernel, scheduling and serialization environment proves it; not a default guarantee.

D. **Hot zero-copy:** only claim after measuring shared pages and bounded tail copies. A fallback that reruns prefill is still a correct semantic implementation, not a zero-copy implementation.

**No backend installed?** G0 can pass; mark G1+ BLOCKED. Do not fabricate engine output or call mock tests a certification.

## Current draft matrix (2026-10-12)

The machine-readable boundary is `support-matrix.json`. Its one draft target is the pinned Qwen Q8_0 model, FP16 KV, token-at-a-time operator, Linux/ext4/CUDA and one active execution lease. llm-60 is primary; llm-49 provides a separate profile and agreement checks. Each executed result must bind its own binary, model-file, hardware and frozen calibration identities. A result on one host does not qualify another host.

Status is DRAFT_VALIDATION_PENDING with no certified guarantees. Batched-prefill equivalence failed and remains outside this target. Q4 weights, quantized KV, HIP/SYCL and other models are unmeasured; native Windows storage, multimodal inputs and concurrent GPU leases are unsupported by the current implementation. Process termination tests do not establish power-loss durability. The registered matrix test executes both forced suffixes at every boundary and checks cold restoration, but cannot be counted until actually run under a frozen profile.
