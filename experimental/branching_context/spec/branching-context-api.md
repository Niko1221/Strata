# Context Time Travel
## Mathematics and test specification for an immutable branching-context API

**Version:** 0.1.0  
**Date:** October 11, 2026  
**Status:** Proposed contract, mathematical derivations, worked examples, and acceptance-test specifications. This is not a claim that a production inference backend has implemented or passed this contract.

> Read a prefix once. Save a decision point. Explore several continuations independently. Return to the decision point without retaining an abandoned future's influence.

The central invariant is:

> **History determines what can be continued. Checkpoints determine how quickly it can be continued.**

A saved conversation must not disappear merely because its GPU cache was evicted. Conversely, retaining tokens alone does not promise immediate restoration of a live model state or bit-identical regeneration on different hardware.

---

## 1. The design in plain English

Think of a game with saved positions, except that every accepted move can have an immutable save record.

A **snapshot** identifies exactly what has happened and the decoder's state at that point. A **branch** is a movable bookmark pointing to one snapshot. A **checkpoint** is the already-computed numerical model state that makes returning to a snapshot faster. An **evaluation** is a separate report about a specific snapshot, not an edit to it.

For example:

```text
                         /-- A1 -- A2     candidate A
P: shared saved prefix --+-- B1 -- B2     candidate B
                         \-- C1 -- C2     candidate C
```

All three candidates inherit `P`. None inherits another candidate's suffix. To try another future, create another branch at `P`, or move an existing branch bookmark back to `P` and continue.

The previous future remains a valid immutable object while retained. A rewind is not deletion, a summary, a positional trick, or an inverse neural-network calculation.

### What is deliberately different from “save the KV cache”

This contract does **not** make a KV blob the authoritative conversation. That would make cache eviction equivalent to losing history, and it would omit random-number state and other decoding decisions.

Instead there is one authoritative immutable history, with necessary decoder state, and optional compiled model-state checkpoints. These are a source and its materialization, not competing copies of the conversation. Moving between GPU memory, RAM, disk, and recomputation does not create different kinds of branch.

The initial scope is fixed-weight, causal inference. It includes branching at every retained user/assistant message boundary and every retained token boundary. It does not include undoing filesystem changes, rerunning tools during replay, changing model weights, or merging two divergent KV suffixes.

---

## 2. Assumptions and notation

Let:

| Symbol | Meaning |
|---|---|
| `f` | Immutable model/input-semantics contract |
| `e` | Numerical execution and checkpoint-format profile |
| `X_t = (x_1, ..., x_t)` | Exact model inputs consumed so far; token IDs in the text-only case |
| `P || A` | Concatenation of prefix `P` and suffix `A` |
| `C_t` | Complete model continuation state after consuming `X_t` |
| `ell_t` or $\ell_t$ | Logits predicting input token `x_(t+1)` |
| `D_t` | Complete decoder/controller state at this boundary |
| `S_t` | Immutable logical snapshot |
| `B = (id, revision, tip)` | A named branch reference |
| `J` | An optional materialized checkpoint |

The token count `t` includes every model token actually ingested, including special tokens. Logical token count, absolute position IDs, physical cache slots, and a sliding window's retained length are not interchangeable.

### Conditions for the mathematical identities

Hold fixed the weights and adapters, inference behavior, attention visibility rules, and positional semantics. Disable training-time stochastic operations. Any randomness that affects execution must be part of the declared state or execution contract.

In particular, adding a future suffix must not retroactively change the interpretation of earlier positions. A position-scaling scheme that silently changes its frequencies when total sequence length changes does not satisfy this assumption without additional treatment.

A batch must not allow one branch to change another branch's mathematical computation. Backend floating-point scheduling can still affect numerical reproducibility; Section 10 separates that issue from causal correctness.

**Scope limit:** no cache-management scheme guarantees continuation beyond an unsupported model context limit, after required weights or input artifacts are lost, or without enough eventual compute resources. Cache eviction alone must not make a retained, otherwise valid history uncontinuable.

---

## 3. Why branching is mathematically valid

### 3.1 Causal prefix invariance

For a transformer layer, write attention at position `i` as:

$$
a_i = \sum_{j\le i}\alpha_{ij}v_j,
\qquad
\alpha_{ij} =
\frac{\exp(q_i^\top k_j/\sqrt d)}
{\sum_{r\le i}\exp(q_i^\top k_r/\sqrt d)}.
$$

The attention mask excludes future positions. Consequently, a representation at position `i` depends on inputs at positions at most `i`, not on a future suffix. This is the causal property underlying KV reuse. [1]

For any prefix `P` of length `n` and suffixes `A` and `B`:

$$
K^{(l)}_{1:n}(P\Vert A)
=K^{(l)}_{1:n}(P\Vert B)
=K^{(l)}_{1:n}(P),
$$

$$
V^{(l)}_{1:n}(P\Vert A)
=V^{(l)}_{1:n}(P\Vert B)
=V^{(l)}_{1:n}(P).
$$

These equations are ideal-arithmetic statements under Section 2's assumptions.

**Proof sketch.** At the input layer, the prefix embeddings and positions are identical. Assume the prefix representations entering layer `l` are identical. Their queries, keys, and values are identical; their causal visibility sets are identical; therefore their attention and subsequent position-local transformations are identical. Induct over layers.

This proves that a shared prefix may be reused. It does not prove that arbitrary non-prefix fragments can be reused independently of their preceding histories.

### 3.2 Continuation as a state transition

Define a one-input model transition:

$$
U_f(C_t,x_{t+1})=C_{t+1}.
$$

The next-token logits are an observable of that state:

$$
\ell_t = L_f(C_t).
$$

The complete state must be sufficient: future computation may not depend on hidden mutable session state outside `C_t`, the supplied inputs, and the declared execution profile.

Define continuation by folding the transition over a suffix:

$$
\operatorname{Extend}_f(C,\epsilon)=C,
$$

$$
\operatorname{Extend}_f(C,Y\Vert y)
=U_f(\operatorname{Extend}_f(C,Y),y).
$$

Then:

$$
\boxed{
\operatorname{Extend}_f(\operatorname{Prefill}_f(P),A)
\simeq \operatorname{Prefill}_f(P\Vert A)
}
$$

Here $\simeq$ means identical continuation behavior in the mathematical model. It does not require two physical memory allocators to produce identical padding bytes or addresses.

### 3.3 The branching law

Let `C_P = Prefill_f(P)`. Independent continuations are:

$$
C_A=\operatorname{Extend}_f(C_P,A),
\qquad
C_B=\operatorname{Extend}_f(C_P,B).
$$

The implementation must behave as though each transition receives an immutable `C_P`, even if physical prefix pages are shared.

The forbidden implementation is:

$$
C_B=\operatorname{Extend}_f(C_A,B).
$$

That computes `P || A || B`, not the requested alternate future `P || B`.

### 3.4 The rewind law

For an ancestral snapshot `S_P`:

$$
\operatorname{History}(\operatorname{Rewind}(B,S_P))=P.
$$

After an alternate continuation `Y`:

$$
\boxed{
\operatorname{Continue}(\operatorname{Rewind}(B,S_P),Y)
\simeq\operatorname{Fresh}(P\Vert Y)
}
$$

This is restoration or reconstruction of an earlier state. It does **not** require:

$$
U_f^{-1}(C_{t+1},x_{t+1}).
$$

No such inverse is assumed to exist.

---

## 4. What a snapshot must identify

### 4.1 Immutable logical record

Use an immutable parent-linked representation:

```text
SnapshotRecord {
    format_version
    model_contract_id
    parent_snapshot_id          # absent only at a root
    appended_model_inputs       # exact IDs/positions/artifact references
    decoder_state               # immutable, complete continuation controller
    boundary_metadata           # message/turn boundaries, not rewritten text
}
```

The parent link and appended inputs define the authoritative history:

$$
H(S)=H(\operatorname{parent}(S))\Vert\operatorname{inputs}(S).
$$

A decoder-only transition, such as selecting another random stream, has an empty input delta. Every accepted generated token creates a new stable logical boundary. Input blocks may create larger deltas, provided all advertised message/token boundaries can be addressed without changing their exact prefix.

A snapshot identity is:

$$
\operatorname{id}(S)=
\operatorname{Hash}(\operatorname{CanonicalEncode}(S)).
$$

Use versioned, unambiguous serialization. Include lengths and field types; do not hash ambiguous string concatenations. Verify content and compatibility when loading. A cryptographic hash is an engineering integrity mechanism, not a mathematical proof that collisions are impossible.

Timestamps, storage locations, GPU addresses, branch names, and evaluation scores are not part of the model's input history. Put operational audit information in separate immutable records.

Two snapshots may have identical model inputs but different decoder states. They may share a model-state checkpoint while representing different future experiments. They must not be confused as complete decoding snapshots.

### 4.2 Decoder state is more than a seed

`D_t` must include or losslessly reference everything that can affect the next decoding decision:

- The sampler specification, processing order, random-generator algorithm, stream key, and current draw position or full generator state.
- Grammar/constraint progress, repetition-history state, adaptive sampler state, stop-detection progress, and generation-scope counters.
- Pending output-decoder bytes or equivalent reconstruction information, and whether the generation is active or stopped.

This is not redundant with the token prefix: the same tokens can have been reached through different random draws, policies, or adaptive sampler histories.

Opaque backend state is acceptable only with a versioned serializer and a demonstrated round-trip contract. An unsupported mutable sampler cannot silently be treated as stateless.

### 4.3 Model state is more than KV

A complete materialization may contain:

$$
C_t=
(\mathrm{KV}_t,\mathrm{recurrent}_t,\mathrm{convolution}_t,
\mathrm{positions}_t,\mathrm{visibility}_t,
\mathrm{codec\ state}_t,\mathrm{next\ output}_t).
$$

Not every model has every component. The point is completeness, not mandatory empty fields.

For a conventional full-attention transformer, KV and continuation metadata may suffice, together with the next-logit frontier. A recurrent or hybrid model may require additional state that cannot be recovered by slicing KV. A sliding-window cache might have discarded tokens needed to reconstruct a much earlier boundary. Cache implementations already differ in how they retain and update prior positions. [1]

### 4.4 One boundary convention prevents an off-by-one bug

Every published snapshot uses this convention:

> **All inputs recorded in the snapshot have been logically consumed. The available next logits predict the next input after them. There is no hidden, already-sampled-but-unconsumed token.**

In notation:

$$
H(S_t)=x_{1:t},\qquad \ell_t\text{ predicts }x_{t+1}.
$$

An engine may internally delay ingesting its last sampled token. Its adapter must normalize that internal state before claiming this public boundary, or reconstruct the correctly normalized state on materialization.

When the final token has been sampled but not fed through the model, pretending the KV cache already includes it creates a corrupt checkpoint even though the displayed text looks correct.

A checkpoint may retain `ell_t`, retain a sufficient final hidden state to recompute it, or restore an earlier boundary and replay through `x_t`. It must not obtain missing logits by feeding `x_t` twice onto a cache that already contains `x_t`.

---

## 5. Checkpoints: an optional accelerator, never the history

A checkpoint is:

```text
Checkpoint {
    input_prefix_identity
    model_contract_id
    execution_profile_id
    consumed_input_count
    position_and_visibility_identity
    state_manifest_and_checksums
    complete_model_state
}
```

The snapshot's authoritative decoder state is attached when materializing a usable continuation. A checkpoint may duplicate it for convenience, but that copy must match the snapshot and cannot override it.

### 5.1 Separate semantic compatibility from binary compatibility

`f`, the semantic contract, includes the actual weights, adapters, architecture, vocabulary/input interpretation, attention rules, positional configuration, and other settings that change the modeled function. Preserve tokenizer/template provenance at the text-to-input boundary.

`e`, the execution profile, includes the backend build, checkpoint layout, cache representation, relevant kernel/math settings, and hardware/runtime information needed for its declared replay guarantee.

Changing a seed does not change `f`. Changing an adapter or RoPE/YaRN semantics does. A checkpoint from incompatible `f` must never be used.

A checkpoint from an incompatible `e` cannot be deserialized merely because its tensor shapes match. Recompute under the requested compatible semantic contract, or use an explicitly implemented and validated conversion. A lossy conversion is not exact restoration.

### 5.2 Materialization rule

Given snapshot `S` with `t` inputs:

$$
\operatorname{Materialize}(S)=
\begin{cases}
\operatorname{ValidateAndUse}(J_t), & \text{valid exact-prefix checkpoint exists},\\
\operatorname{Extend}(J_m,H(S)_{m+1:t}), & \text{compatible ancestor at }m<t,\\
\operatorname{Prefill}(H(S)), & \text{otherwise}.
\end{cases}
$$

Attach exactly `D(S)` afterward. During replay, previously accepted generated tokens are fixed input data. They are not sampled again. Do not advance the live random stream merely because the backend reconsumed historical tokens.

If multiple ancestors exist, choose one using a cost estimate such as:

$$
J^*=\arg\min_J
\left[T_{\mathrm{load}}(J)+T_{\mathrm{replay}}(J\rightarrow S)\right].
$$

The nearest ancestor is not necessarily fastest if it is on slow storage. This is a performance decision; it cannot change which history is reconstructed.

### 5.3 Eviction invariant

For retained `S`, compatible model resources, and sufficient eventual capacity:

$$
\operatorname{EvictAllCheckpoints}(S)
\implies H(S)\text{ and }D(S)\text{ remain unchanged}.
$$

Materialization may become slower. The branch remains valid.

The baseline storage policy retains logical snapshots until explicit history deletion. Cache eviction and history garbage collection are separate operations with separate authorization and retention rules.

---

## 6. The smallest useful API

The following is a proposed backend-independent interface, not the signature of an existing engine.

```text
create(contract, exact_inputs, decoder_state, boundaries) -> SnapshotID
snapshot(branch_id) -> (revision, SnapshotID)
fork(snapshot_id, new_branch_name) -> BranchRef
apply(snapshot_id, action) -> TransitionResult
move(branch_id, expected_revision, target_snapshot_id, operation_id) -> BranchRef
checkpoint(snapshot_id, execution_profile) -> CheckpointID
materialize(snapshot_id, execution_profile) -> StateLease
evaluate(snapshot_id, evaluation_spec) -> EvaluationRecord
```

`StateLease` is a private execution workspace. It may share immutable cache pages, but mutation through one lease must not alter another snapshot's observable state.

### 6.1 One transition operation

`apply` accepts typed domain actions:

```text
Feed(exact_inputs)
StartGeneration(policy, stream_key)
Generate(max_steps)
SetRandomStream(stream_key)
```

`Feed` appends actual supplied inputs and updates input-sensitive controller state according to the declared policy. It cannot edit an existing prefix. `StartGeneration` explicitly establishes a new generation scope and its grammar, stop rules, counters, and sampling policy. `Generate` continues that scope. `SetRandomStream` changes only its random stream, preserving grammar, penalties, and stop progress.

All actions return new immutable snapshots. None mutates the supplied snapshot or automatically moves a named branch.

`Generate(max_steps)` is a work bound, not a new sampler initialization. Returning because the work bound was reached pauses the same generation scope. An EOS/stop rule instead creates a stopped decoder state. Continuing a stopped scope must not silently generate more text; starting another scope is an explicit action.

### 6.2 Forking copies a reference, not the prefix

$$
\operatorname{fork}(S,\text{name})
=(\text{new id},0,\operatorname{id}(S)).
$$

A fork initially inherits the **same** decoder state, including its RNG state. Therefore two forks can reproduce the same future. A new future can come from different inputs, a different explicitly selected random stream, or a new declared generation policy.

There is no hidden rule that assigns a new seed simply because a branch was created.

### 6.3 Rewind and winner adoption use the same reference operation

`move` atomically compares the revision and changes the branch tip:

$$
\operatorname{CAS}((v,s),\mathrm{expected}=v,\mathrm{target}=s')
=(v+1,s').
$$

If the current revision differs, return a conflict without changing the branch.

Moving to an ancestor is a rewind. Moving to an evaluated descendant adopts that candidate. Moving does not merge histories or recompute model state. Access checks, retained-history checks, and the branch's model-contract restrictions still apply.

Use a monotonically increasing revision, not just the expected snapshot ID. Otherwise a move `P -> A -> P` would allow an old operation that expected `P` to succeed despite intervening changes: the ABA problem.

### 6.4 Retry and failure semantics

An `operation_id` identifies one requested branch update. Atomically store its request digest and result with the reference update. A repeated identical request returns the original result; reuse with different arguments is an error.

If generation fails, already published immutable snapshots remain valid. An uncommitted candidate cannot partially move a branch. If a branch update fails, its candidate snapshot may still exist and be adopted or discarded later.

The core API returns accepted token boundaries. A streaming wrapper must publish a durable snapshot before promising that a displayed token is resumable, and must distinguish a produced candidate from an updated named branch.

---

## 7. Randomness: repeat the future or explore another one

### 7.1 Sampling is part of the transition

Let the processed next-token distribution be:

$$
p_t=\operatorname{Normalize}(\operatorname{Process}(\ell_t,D_t)).
$$

For categorical sampling using cumulative probabilities:

$$
x_{t+1}=\min\left\{j: \sum_{i\le j}p_t(i)>u_t\right\},
\qquad u_t\in[0,1).
$$

Specify token ordering, ties, and the comparison convention. Empty allowed-token sets are errors, not permission to bypass a grammar.

After accepting the token, update both the model state and decoder state:

$$
C_{t+1}=U_f(C_t,x_{t+1}),
$$

$$
D_{t+1}=\operatorname{DecoderUpdate}(D_t,\ell_t,x_{t+1},\text{draws consumed}).
$$

Publish the new snapshot only at that consistent boundary.

### 7.2 A schedule-independent random stream

A convenient design is a specified counter-based generator:

$$
u_j=\operatorname{Uniform}(\operatorname{PRF}(k,j)).
$$

Store the generator specification, key `k`, and next draw index `j` in `D`. A conventional generator with fully serialized state is also valid.

Count random draws, not merely generated tokens: a sampler may consume multiple draws per token. Backend samplers must not use a process-global random stream that lets branch A consume branch B's future draws.

For exploration, derive a distinct reproducible key from an experiment seed and a stable candidate label:

$$
k_b=\operatorname{KDF}(k_{\mathrm{experiment}},
\text{“context-branch”}\Vert\operatorname{id}(S_P)\Vert\mathrm{label}_b).
$$

Store the resolved key. Do not make it depend on worker scheduling or wall-clock arrival order.

### 7.3 Two different guarantees

**Repeat the same future:** restore the same snapshot, policy, RNG position, inputs, and compatible deterministic execution. The same future should result within the certified guarantee.

**Explore another future:** retain the same context and choose another stream or action. Different streams do **not** guarantee different text. A model can choose the same token under many random draws; greedy generation may use no draws at all.

For a test that requires different branches, force distinct suffix tokens or inject known test variates into a toy sampler. Do not assert that two arbitrary seeds must produce different answers.

---

## 8. Copy-on-write sharing and isolation

### 8.1 The invariant

For any physical cache page reachable from more than one retained state:

$$
\operatorname{shared}(p)\implies\operatorname{immutable}(p).
$$

More generally, a page reachable from an immutable checkpoint cannot be overwritten just because its current reference count happens to be one. Future restoration still depends on its contents.

An append may allocate new pages and copy a partially filled tail into private storage. It must never rewrite the saved prefix in place. Mutable quantization scales, page metadata, position maps, and recurrent buffers are also subject to isolation—not just K/V tensor bytes.

PagedAttention established practical shared KV storage; vLLM's documented prefix caching keys blocks by their prefix history and additional input identities. That is an implementation precedent, not proof that an arbitrary application already supports durable branch snapshots. [2][3]

### 8.2 Worked page boundary

Suppose pages hold four tokens and the saved prefix has six tokens:

```text
saved P:     page 0 [p1 p2 p3 p4]   page 1 [p5 p6 -- --]
```

After two branches append different tokens:

```text
saved P:     page 0 [p1 p2 p3 p4]   page 1 [p5 p6 -- --]
branch A:    page 0 [shared]         page 2 [p5 p6 a1 --]
branch B:    page 0 [shared]         page 3 [p5 p6 b1 --]
```

The tail copies are bounded by page size, not prefix length. A backend that shares only full pages can reconstruct or copy this bounded tail; it must report that work honestly.

The implementation may choose a different physical layout as long as the same logical isolation holds. Tests should inspect logical inputs, active attention visibility, page checksums where applicable, and allocation counters—not require these exact page IDs.

### 8.3 Memory and work saved

For full-attention KV storage, define per-token KV bytes:

$$
b_{\mathrm{KV}}=\sum_l h_{\mathrm{KV},l}
(d_{K,l}b_{K,l}+d_{V,l}b_{V,l}).
$$

For a prefix of `n` tokens and `k` branches with suffix lengths `m_1,...,m_k`, ideal unique KV storage is:

$$
M_{\mathrm{shared}}\approx b_{\mathrm{KV}}
\left(n+\sum_{i=1}^{k}m_i\right),
$$

instead of:

$$
M_{\mathrm{separate}}\approx b_{\mathrm{KV}}
\left(kn+\sum_{i=1}^{k}m_i\right).
$$

Thus the ideal saved prefix storage is:

$$
\Delta M\approx(k-1)n\,b_{\mathrm{KV}}.
$$

These formulas exclude page rounding, copies of partial tails, logits, decoder state, recurrent-state copies, workspaces, and allocator overhead. Retaining the original partial tail while `k` branches each copy it can duplicate up to `k` partial prefixes; do not assume that overhead is zero.

With `n=100,000`, `k=8`, and every `m_i=2,000`, the ideal storage counts are `116,000` token-equivalents shared versus `816,000` separate. These are arithmetic examples, not measured VRAM results.

Similarly, shared materialization can avoid roughly `(k-1)` repeated prefix prefills. It does **not** eliminate each suffix token's attention over the prefix, nor promise an eightfold generation speedup.

---

## 9. Evaluation without contaminating the candidates

An evaluator consumes an immutable candidate and a fixed evaluation specification:

$$
R_b=E(H(S_b),\mathcal{T},\mathcal{V},\mathcal{W}).
$$

Here `T` is the test/fixture set, `V` is the evaluator implementation and version, and `W` is a fixed or recorded evaluation environment.

An evaluation record should identify the candidate snapshot, common task root, evaluator version, fixture hashes, observed results, unresolved checks, and resource measurements. It may refer to output artifacts; it must not overwrite the candidate's history or decoder state.

The noninterference requirement is:

$$
\boxed{\operatorname{SnapshotAfterEvaluate}(S)=S.}
$$

A judge that needs to generate text must use its own branch and random stream. A test runner must use an isolated workspace or immutable inputs. A judgment cannot be inserted into the candidate and then presented as an evaluation of its original state.

### 9.1 Selecting a winner

Separate mandatory validity from preferences. Let:

$$
\mathcal{A}=\{b:\operatorname{MandatoryChecks}(R_b)=\mathrm{PASS}\}.
$$

If `A` is empty, return **no validated candidate**. Otherwise choose according to a declared ordering, for example higher verified score, then fewer generated tokens, then a stable snapshot-ID tie-break.

`UNKNOWN`, a timeout, a missing test, and a failed test are distinct results. None automatically counts as a pass. Average model log-probability is not a substitute for an external correctness check.

Adopting the winner is simply:

```python
winner = choose_validated(evaluation_records)
new_ref = move("main", expected_revision, winner.snapshot_id, operation_id)
```

There is no operation that combines the hidden states of the winning and losing futures.

### 9.2 External effects are not time-travelled

Replaying a saved tool result means feeding the recorded result bytes back into the model. It does not mean executing the tool again.

A context rewind cannot unsend an email, restore a real filesystem, reverse a robot motion, or undo a payment. Experiments with tools need an independently isolated or versioned environment. Environment rollback is outside this context-state contract.

If a successful branch creates code changes, adopting its context is not by itself permission to apply those changes elsewhere. Keep artifact adoption explicit.

---

## 10. Exact mathematics versus numerical reproducibility

Do not use a single vague word such as “same” for all tests.

### 10.1 Three separate checks

**Structural identity:** history, decoder state, declared positions, ancestry, checksums, and visibility membership must match exactly. A floating-point tolerance never excuses the wrong token or an overwritten snapshot.

**Exact replay under a certified execution profile:** identical state, input, and controlled randomness must reproduce the declared deterministic outputs. Certification must specify hardware, backend, kernels, batching conditions, and restoration/recomputation paths. A seed alone is not a cross-platform guarantee. PyTorch's reproducibility guidance explicitly cautions against assuming identical results across releases and platforms. [5]

**Numerical equivalence:** compare model outputs on the same forced input continuation under a tolerance fixed before judging the implementation. Full prefill and incremental execution can use different arithmetic paths. Report their differences rather than pretending mathematical equality implies bitwise GPU equality.

These checks are not interchangeable. Approximate numerical agreement does not certify exact sampled replay.

### 10.2 Compare logits before sampling

For a fixed suffix `Y=(y_1,...,y_m)`, compare the candidate and an independent fresh reference at every shared history:

$$
\ell^{\mathrm{candidate}}(P\Vert Y_{<i})
\quad\text{versus}\quad
\ell^{\mathrm{reference}}(P\Vert Y_{<i}).
$$

Then force both systems to consume `y_i`. This prevents an early sampling difference from causing all later comparisons to use different contexts.

A componentwise numerical criterion is:

$$
|a_j-b_j|\le\epsilon_{\mathrm{abs}}+
\epsilon_{\mathrm{rel}}|b_j|.
$$

Require identical allowed-token support when comparing grammar-masked distributions. Reject NaNs and unexplained infinities. Do not subtract infinities and call the resulting NaN a successful comparison.

For probabilities, useful diagnostics include:

$$
\mathrm{TV}(p,q)=\tfrac12\sum_j|p_j-q_j|,
$$

and, where defined:

$$
D_{\mathrm{KL}}(p\Vert q)=\sum_jp_j\log\frac{p_j}{q_j}.
$$

There is no universal BF16/FP16 tolerance suitable for every model and kernel. Establish the unchanged-backend baseline first, specify the intended guarantee, freeze thresholds, and retain the worst cases. Do not widen thresholds merely to hide branching bugs.

### 10.3 Why nearly identical logits can generate different text

Suppose each raw logit differs from reference by at most `epsilon`. If the reference's top-two margin is:

$$
\ell_{(1)}-\ell_{(2)}>2\epsilon,
$$

the greedy winner is preserved. Without that margin, a very small numerical change can swap the winner.

For sampled decoding, let `F` and `G` be the two cumulative distributions. The same variate `u` chooses the same token when it stays on the same side of every relevant boundary. A sufficient condition is:

$$
\min_j|u-F(j)|>\max_j|F(j)-G(j)|.
$$

Therefore tiny probability errors do not imply identical sampled strings for every seed. Report structural failures, numerical drift, and unstable sampling boundaries separately.

---

## 11. A completely worked miniature example

This example is deliberately small enough to verify by hand. It proves the branch-management example, not the correctness of a real transformer's cache implementation.

Use one scalar attention head with every key equal to zero:

$$
k_i=0,\qquad v_i=x_i.
$$

Every attention score is zero, so the weights are uniform:

$$
\alpha_i=\frac1t,\qquad a(X_t)=\frac1t\sum_{i=1}^t x_i.
$$

Take the saved prefix:

$$
P=(1,3),\qquad a(P)=2.
$$

Explore two futures:

$$
A=P\Vert(5)=(1,3,5),\qquad a(A)=3,
$$

$$
B=P\Vert(9)=(1,3,9),\qquad a(B)=13/3.
$$

Now rewind branch A's bookmark to `P` and take another future:

$$
C=P\Vert(-1)=(1,3,-1),\qquad a(C)=1.
$$

The expected final observations are:

| Retained snapshot | Exact inputs | Exact attention output |
|---|---|---|
| `P` | `(1, 3)` | `2` |
| Old A future | `(1, 3, 5)` | `3` |
| B future | `(1, 3, 9)` | `13/3` |
| New A future C | `(1, 3, -1)` | `1` |

A broken shared-list implementation can accidentally compute B as `(1,3,5,9)`, giving `18/4`, or C as `(1,3,5,-1)`, giving `8/4`. Neither is an alternate future of `P`.

### 11.1 A tiny executable reference model

The Python block below uses only the standard library. It checks immutable history, exact rational attention, sampled replay, stream isolation, chunked generation, and cold reconstruction in a toy model. It intentionally does not implement GPU KV pages, real model logits, durable transactions, or backend checkpoint formats.

The two-token distribution is defined as:

$$
z=a(X),\qquad
p(1\mid X)=\frac12+\frac{z}{4(1+|z|)},
\qquad p(-1\mid X)=1-p(1\mid X).
$$

Its range is strictly inside `(1/4, 3/4)`. It is a toy readout, not a claim about a trained model.

Save the block as `test_context_time_travel_toy.py`, then run `python test_context_time_travel_toy.py`.

```python
from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib
import hmac
import unittest


@dataclass(frozen=True)
class Snapshot:
    # Tuples and bytes make the contained values immutable as well.
    tokens: tuple[int, ...]
    stream_key: bytes
    next_draw: int = 0

    def __post_init__(self):
        if not self.tokens or any(type(x) is not int for x in self.tokens):
            raise ValueError("Use a nonempty tuple of integer toy values")
        if type(self.tokens) is not tuple or type(self.stream_key) is not bytes:
            raise TypeError("Snapshot storage must be immutable")
        if (not self.stream_key or type(self.next_draw) is not int
                or not 0 <= self.next_draw < 2**128):
            raise ValueError("Invalid stream key or counter")


def attention(s: Snapshot) -> Fraction:
    # Fresh oracle: compute directly from the authoritative input tuple.
    return Fraction(sum(s.tokens), len(s.tokens))


def feed(s: Snapshot, values: tuple[int, ...]) -> Snapshot:
    if type(values) is not tuple or any(type(x) is not int for x in values):
        raise TypeError("Expected a tuple of integer toy inputs")
    # This toy has no grammar or penalties; feeding inputs consumes no RNG.
    return replace(s, tokens=s.tokens + values)


def next_uniform(s: Snapshot) -> Fraction:
    # Exact rational variate in (0, 1); no process-global random state.
    payload = b"context-time-travel-toy-v1:" + s.next_draw.to_bytes(16, "big")
    bits = hmac.new(s.stream_key, payload, hashlib.sha256).digest()
    integer = int.from_bytes(bits, "big")
    return Fraction(2 * integer + 1, 2**257)


def step(s: Snapshot) -> tuple[Snapshot, int]:
    z = attention(s)
    p_positive = Fraction(1, 2) + z / (4 * (1 + abs(z)))
    token = 1 if next_uniform(s) < p_positive else -1
    child = replace(s, tokens=s.tokens + (token,), next_draw=s.next_draw + 1)
    return child, token


def generate(s: Snapshot, count: int) -> tuple[Snapshot, tuple[int, ...]]:
    if type(count) is not int or count < 0:
        raise ValueError("count must be a nonnegative integer")
    output = []
    for _ in range(count):
        s, token = step(s)
        output.append(token)
    return s, tuple(output)


class TimeTravelTests(unittest.TestCase):
    def setUp(self):
        self.p = Snapshot((1, 3), b"fixed-experiment-key")

    def test_worked_attention_outputs(self):
        a = feed(self.p, (5,))
        b = feed(self.p, (9,))
        self.assertEqual(attention(self.p), Fraction(2))
        self.assertEqual(attention(a), Fraction(3))
        self.assertEqual(attention(b), Fraction(13, 3))

    def test_rewind_does_not_delete_old_future(self):
        old_a = feed(self.p, (5,))
        branch_a = old_a
        branch_a = self.p  # Rewind only moves this bookmark.
        branch_a = feed(branch_a, (-1,))
        self.assertEqual(branch_a.tokens, (1, 3, -1))
        self.assertEqual(attention(branch_a), Fraction(1))
        self.assertEqual(old_a.tokens, (1, 3, 5))
        self.assertEqual(self.p.tokens, (1, 3))

    def test_same_snapshot_repeats_sampled_future(self):
        mid, _ = generate(self.p, 7)
        final_a, out_a = generate(mid, 20)
        final_b, out_b = generate(mid, 20)
        self.assertEqual(out_a, out_b)
        self.assertEqual(final_a, final_b)

    def test_chunking_does_not_reset_random_stream(self):
        all_at_once, whole = generate(self.p, 19)
        mid, first = generate(self.p, 7)
        in_chunks, second = generate(mid, 12)
        self.assertEqual(whole, first + second)
        self.assertEqual(all_at_once, in_chunks)

    def test_sibling_scheduling_does_not_change_a(self):
        a = self.p
        b = replace(self.p, stream_key=b"another-recorded-stream")
        a, first = generate(a, 4)
        b, _ = generate(b, 50)  # Sibling activity must not consume A's draws.
        a, second = generate(a, 8)
        isolated, expected = generate(self.p, 12)
        self.assertEqual(first + second, expected)
        self.assertEqual(a, isolated)

    def test_fresh_input_oracle_matches_each_forced_future(self):
        for suffix in ((5,), (9,), (-1,), (5, 9, -1), ()):
            child = feed(self.p, suffix)
            fresh = Snapshot((1, 3) + suffix, self.p.stream_key)
            self.assertEqual(attention(child), attention(fresh))

    def test_eviction_preserves_history_and_rng_position(self):
        mid, _ = generate(self.p, 9)
        # Simulated derived cache; it is not the authoritative history.
        checkpoints = {mid: (sum(mid.tokens), len(mid.tokens))}
        checkpoints.clear()
        reconstructed = Snapshot(mid.tokens, mid.stream_key, mid.next_draw)
        self.assertEqual(generate(mid, 11), generate(reconstructed, 11))

    def test_other_stream_does_not_mutate_parent(self):
        other = replace(self.p, stream_key=b"candidate-B", next_draw=0)
        generate(other, 30)
        self.assertEqual(self.p.tokens, (1, 3))
        self.assertEqual(self.p.next_draw, 0)
        # Deliberately do not assert that different streams produce different text.

    def test_feed_associativity(self):
        left = feed(feed(self.p, (5, 9)), (-1,))
        right = feed(self.p, (5, 9, -1))
        self.assertEqual(left, right)


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

The rational arithmetic makes the toy's expected attention outputs exact. Its uniform attention does not exercise real position encodings, multiple layers, dynamic masks, or low-precision kernels. Those require the independent backend tests below.

---

## 12. What working real-backend tests should look like

These are acceptance-test specifications and pseudocode. They have not been run against your inference engines. The oracle must not secretly reuse the implementation's checkpoint path.

Use exact saved token IDs from a fixed fixture, a pinned checkpoint/model build, fixed positions, and a declared execution profile. Begin with short sequences and then repeat at supported long lengths. Never invent valid token IDs without checking the model's vocabulary.

### Test A — Every future matches its own fresh history

**Given:** a fixed prefix `P`, two explicitly different suffixes `A` and `B`, and a trusted no-cache or independently implemented fresh-prefill reference.

**When:** create a snapshot of `P`, then feed `A` on one child and `B` on another, interleaving their steps.

**Then:** each child's next logits match the reference for its own exact history; neither child contains the other suffix.

```python
# Pseudocode: references denote independent backend test adapters.
p = sut.create(f, P, idle_decoder, message_boundaries)
root_digest = sut.logical_digest(p)
a, b = p, p

for index in range(max(len(A), len(B)) + 1):
    if index <= len(A):
        expected_a = reference.forward_without_cache(P + A[:index]).next_logits
        assert_logits_close(sut.next_logits(a), expected_a, frozen_tolerances)
    if index <= len(B):
        expected_b = reference.forward_without_cache(P + B[:index]).next_logits
        assert_logits_close(sut.next_logits(b), expected_b, frozen_tolerances)
    if index < len(A):
        a = sut.apply(a, Feed((A[index],))).snapshot_id
    if index < len(B):
        b = sut.apply(b, Feed((B[index],))).snapshot_id

assert sut.inputs(a) == P + A
assert sut.inputs(b) == P + B
assert sut.logical_digest(p) == root_digest
```

Compare next logits **before the first suffix token** as well as after every feed. Otherwise a missing/stale prefix-frontier logit can go undetected.

Use nontrivial fixtures: different positions, repeated tokens in different histories, near-boundary cache lengths, and ordinary text. The all-zero-key toy is not enough.

**Bug detected:** branch leakage, incorrect attention masks, wrong positions, shared mutable pages, stale next logits, or an invalid cache hit.

### Test B — Snapshot halfway through sampling, then repeat exactly

**Given:** a generation scope with a fixed sampler, grammar, stop rules, and random stream.

**When:** generate seven accepted tokens, snapshot that state, generate twenty more, restore the intermediate snapshot, and regenerate those twenty.

```python
ready = sut.apply(p, StartGeneration(policy, stream_key)).snapshot_id
mid = sut.apply(ready, Generate(7)).snapshot_id
saved_decoder = sut.decoder_bytes(mid)
j = sut.checkpoint(mid, execution_profile)

first = sut.apply(mid, Generate(20))
sut.release_all_live_leases()
sut.verify_checkpoint(j)
second = sut.apply(mid, Generate(20))

assert first.accepted_tokens == second.accepted_tokens
assert sut.decoder_bytes(first.snapshot_id) == sut.decoder_bytes(second.snapshot_id)
assert sut.decoder_bytes(mid) == saved_decoder
```

Exact token assertions belong to the certified deterministic profile. Also compare per-step logits and random draw coordinates to diagnose failure.

Repeat with a **mid-generation** snapshot; starting both tests from a fresh seed would not detect a failure to restore the draw cursor. Stop rules may produce fewer than twenty tokens, but both runs must stop at the same accepted boundary.

**Bug detected:** seed-only restoration, lost grammar state, sampler resets, stale frontier logits, or output-stop progress loss.

### Test C — A sibling's activity cannot consume this branch's future

Run candidate A alone from a saved generation state for 64 tokens. Repeat from the same state, but interleave many steps from candidate B, checkpoint writes, and independent evaluations.

Require the same logical inputs and controller transitions for A. Under the certified schedule-independent execution profile, require the same A logits and generated tokens. Under a profile that only promises numerical equivalence, compare forced logits and explicitly report that exact scheduling-invariant generation is not certified.

Do not downgrade a structural isolation error to “GPU nondeterminism.” A different RNG draw cursor, grammar cursor, or visible token set is always a contract failure.

**Bug detected:** shared RNG, shared sampler, shared stop detector, mutable global sequence state, or cross-sequence attention.

### Test D — Rewind after a divergent suffix, including partial pages

Take a snapshot with `n = page_size + 2` inputs. Pin it. Run A for enough tokens to cross the next page boundary. Record A's final snapshot. Move A's bookmark back to the saved prefix. Feed a different forced suffix C.

Require:

```text
history(saved P)       = P
history(retained A)    = P || A
history(new A tip)     = P || C
history(sibling B)     = P || B
logits(new A tip)      ~= fresh_reference(P || C)
```

Audit the original published prefix pages and mutable metadata. None may have been overwritten. Test empty suffixes and exact page boundaries too.

**Bug detected:** partial-tail overwrite, leaked abandoned tokens, refcount/lifetime errors, or a rewind that merely changes displayed text.

### Test E — Destroy every checkpoint and still resume

Create `mid` halfway through sampling. Record its authoritative history and decoder state. Produce an expected continuation. Evict GPU state, RAM state, and persisted numerical checkpoints, but retain the logical snapshot records and input artifacts.

Materialize `mid` from history, then continue.

Require identical history and decoder state, and next logits matching fresh reconstruction under the declared numerical guarantee. Require exact sampled replay only when that reconstruction path is certified for it.

Instrument the backend to prove it actually recomputed historical inputs rather than finding an unnoticed cache. Historical generated tokens must be consumed as fixed inputs, and the restored RNG draw cursor must not advance during that replay.

**Bug detected:** treating a cache blob as the only history, rerolling old outputs, or losing sampler state during eviction.

### Test F — Evaluating and adopting one future does not alter the others

Use three fixed candidate artifacts from the same task root. A small deterministic verifier yields:

```text
candidate A: all mandatory checks PASS; verified score 7
candidate B: one mandatory check FAIL; verified score 10
candidate C: all mandatory checks PASS; verified score 8
```

The selected candidate must be C, not B. Keep B's failure evidence. Evaluate in several orders; the result must not depend on evaluation order for this deterministic fixture.

Hash every snapshot's history and decoder state before and after evaluation. Require exact identity. Move the `main` branch pointer to C with compare-and-swap. Require A and B to remain independently continuable while retained.

Then test an all-failed or all-unknown batch: selection must return **no validated candidate**, not silently choose the least bad output.

**Bug detected:** evaluator contamination, mutable winner selection, unverified “best” claims, or destructive adoption.

---

## 13. Acceptance matrix

Every row needs explicit fixtures and observations. `PASS`, `FAIL`, `NOT RUN`, and `UNSUPPORTED` are distinct. An unsupported requirement blocks certification for that capability; it is not a pass.

| ID | Test | Required observation |
|---|---|---|
| T01 | Causal prefix invariance | Same logical prefix K/V or equivalent observable state under different suffixes; match an independent fresh reference. |
| T02 | Fork identity | Immediately after fork, both branches point to the same snapshot and inherit the same decoder state without recomputing the prefix. |
| T03 | Divergent forced futures | A sees only `P || A`; B sees only `P || B`; both match their own fresh-reference logits. |
| T04 | Rewind and repeat | Returning to a saved sampling boundary restores the full controller and reproduces its future under the certified profile. |
| T05 | Schedule isolation | Sibling generation/evaluation changes no logical state or RNG coordinates of the target branch. |
| T06 | Next-token frontier | Restore before/after one sampled token, including EOS; no duplicated, dropped, or unconsumed token. |
| T07 | Partial-page copy-on-write | Appending on either branch cannot overwrite the saved partial tail or sibling metadata. |
| T08 | Full cache eviction | Retained history and controller survive; fresh materialization remains possible without resampling the past. |
| T09 | Ancestor checkpoint replay | Restore `m < n`, replay exact inputs `m+1:n`, and recover snapshot `n` without advancing its saved RNG or rerunning tools. |
| T10 | Checkpoint corruption | Truncation, wrong lengths, checksums, or metadata fail validation; discard/recompute or report error without touching the branch. |
| T11 | Contract mismatch | Wrong weights, adapter, positional policy, multimodal identity, or unsupported checkpoint format is never silently reused. |
| T12 | Middle edit | New history recomputes from its actual common prefix; later contextual states from the old history are not reused as though unchanged. |
| T13 | False suffix match | Identical suffix text after different prefixes must not collide as equivalent model state. |
| T14 | Recurrent/hybrid state | Use a fixture where non-KV state matters; restoring KV alone must fail the test, while complete-state restoration succeeds. |
| T15 | Sliding-window rewind | A boundary older than the retained window is rebuilt from an adequate earlier checkpoint/history, not a crop of insufficient current state. |
| T16 | Grammar, stops, output bytes | Snapshot across partial grammar output, a multi-token stop string, and split UTF-8 output; continuation and stopping remain correct. |
| T17 | CAS race and ABA | Two concurrent updates at one revision have at most one winner; a stale update cannot succeed after `P -> A -> P`. |
| T18 | Retry identity | Repeating one branch-update operation ID returns its original result; different arguments under that ID are rejected. |
| T19 | Allocation/crash failure | Failure cannot publish a branch pointing to missing or partially written history; a failed decode cannot mutate the saved parent. |
| T20 | Evaluation purity | Evaluation leaves every candidate's history, decoder state, and future unchanged; judge state and workspace are separate. |
| T21 | Validated selection | Mandatory failures/unknowns cannot win; all-invalid candidates produce no validated winner. |
| T22 | Message framing and artifacts | All retained message boundaries reconstruct exact model inputs; changed templates or image embeddings cannot masquerade as the old prefix. |
| T23 | Actual reuse | With a hot compatible checkpoint, fork and reference movement perform no model-token computation; append work/copies are measured and bounded. |
| T24 | Stateful randomized commands | A command-sequence model finds no divergence across append, fork, reseed, generate, rewind, checkpoint, eviction, evaluation, retry, and release. |

### 13.1 Mandatory boundary fixtures

For a page size `B`, test prefix lengths `1`, `B-1`, `B`, `B+1`, `2B-1`, and `2B`. Include the smallest valid BOS-only prefix when relevant. An empty-prefix request must either have defined model semantics or be rejected before mutation.

For a sliding window `W`, test saved points on both sides of the eviction boundary. For long-context certification, extend the same tests to representative supported lengths such as 8K, 32K, and 128K; do not claim an unsupported 256K test passed by silently truncating the input.

For branch counts, include one, two, and enough simultaneous branches to force allocation pressure. Use both identical and deliberately different random streams.

### 13.2 Make sure the tests can detect a broken implementation

Intentionally inject these faults into a test adapter:

```text
share a mutable prefix list
omit the RNG draw position
restore stale next-token logits
forget the recurrent state
reuse a suffix after a different prefix
write into a published partial cache page
let replay rerun a recorded tool call
accept a stale branch revision
```

Each injected fault must be caught by a named test. Otherwise a passing suite may only mean the fixtures never exercised the relevant dependency.

For T14, a simple negative-control recurrence is:

$$
r_{t+1}=2r_t+x_{t+1},\quad r_0=0.
$$

After prefix `(1,3)`, `r=5`; after suffix `(5)`, `r=15`. Rewinding to `(1,3)` must restore `r=5`. Keeping `r=15` and appending `-1` yields `29`, while the correct alternate future yields `9`. This catches incomplete-state restoration without needing an actual hybrid checkpoint.

---

## 14. Property-based laws for the reference state machine

Generate random, reproducible sequences of API commands and compare the optimized implementation with a simple immutable reference model. Keep the failing command sequence and shrink it to a minimal counterexample.

### P1 — Parent preservation

For every valid action `a`:

$$
S'=\operatorname{apply}(S,a)
\implies \operatorname{bytes}(S)_{\mathrm{after}}
=\operatorname{bytes}(S)_{\mathrm{before}}.
$$

Apply this to immutable logical records and published numerical page payloads. Physical allocator bookkeeping may change only if it cannot change a retained state's interpretation.

### P2 — Feed associativity

For supplied input blocks `A` and `B`:

$$
\operatorname{Feed}(\operatorname{Feed}(S,A),B)
\equiv\operatorname{Feed}(S,A\Vert B).
$$

Here $\equiv$ compares model inputs and resulting controller semantics, not necessarily snapshot IDs: different parent-link groupings can represent the same final history. This law assumes the declared `Feed` controller update is compositional and neither call implicitly starts a new generation scope.

### P3 — Generation chunk invariance

Within one active generation scope, with the same deterministic execution and randomness:

$$
\operatorname{Generate}(\operatorname{Generate}(S,a),b)
\equiv\operatorname{Generate}(S,a+b).
$$

Compare concatenated accepted tokens and final state. Stop/EOS is absorbing within that scope, so both paths accept the same prefix of at most `a+b` tokens. A call's work limit cannot reset grammar, penalties, random draws, or stopping progress.

### P4 — Fork commutation

For operations on independent branch references, with resources available:

$$
\operatorname{advance}_A\circ\operatorname{advance}_B
\equiv
\operatorname{advance}_B\circ\operatorname{advance}_A.
$$

Logical outcomes must commute. Latency and temporary allocation order need not. Exact logits across batch schedules require an execution profile that actually certifies that property.

### P5 — Checkpoint transparency

Inserting or removing valid checkpoint creation, offloading, and eviction operations does not change authoritative histories or controllers. Materialized outputs must satisfy their declared exact or numerical equivalence guarantee.

### P6 — Evaluation noninterference

For an evaluation operation `E` and a continuation action `a`:

$$
\operatorname{apply}(S,a)
\equiv\operatorname{apply}(\operatorname{unchangedSnapshotAfter}(E,S),a).
$$

An evaluator's own output and compute cost are separate observations.

---

## 15. Backend adapter requirements and known integration traps

Existing engines offer useful primitives, but the application must verify that they implement this complete contract.

**llama.cpp.** Its public interface exposes sequence memory copy/removal and sequence state serialization, alongside separate sampler clone/copy functions. Partial sequence removal can fail. Its current on-device sequence-state flag explicitly says another save for that sequence invalidates prior saves using that flag. A wrapper cannot call such transient objects independent immutable snapshots without owning/copying the required state. The decode interface also documents possible partially processed state after an abort. Treat backend workspaces as disposable until a valid public boundary is established. [4]

**vLLM.** Its documented prefix-cache design includes the preceding prefix in a block's identity, with additional identities for features such as adapters or multimodal inputs, and describes full-block caching. Those mechanisms are relevant to physical reuse. They do not, on their own, specify this application's durable decoder snapshots, reference revisions, or evaluator isolation. [3]

**Hugging Face Transformers.** Cache objects and generation loops provide an approachable correctness-adapter surface. In the documented loop, a sampled token is passed to the model on the next iteration; a wrapper must account for that pending-token frontier rather than equate displayed token count with consumed KV length. [1]

Pin the exact backend revision used in every certification record. The source pages below are pointers to upstream behavior inspected for this specification, not evidence that your installed versions expose identical capabilities.

### Adapter interface obligations

An adapter must answer these questions through tests, not assumptions:

```text
What exactly has been consumed when a save operation returns?
Are next logits included, or how are they reconstructed?
Does the save survive subsequent saves and decode operations?
Which model, sampler, position, and recurrent states are excluded?
Can partial removal fail or omit older sliding-window history?
Does state restoration require the identical backend/model layout?
What survives an aborted decode or out-of-memory condition?
Can two leases genuinely share immutable prefix pages?
```

If a backend cannot provide immutable sharing, an owned deep copy or reconstruction can still satisfy semantic correctness. It must not be advertised as zero-copy hot branching. Correctness and the fast-path performance claim are separate gates.

---

## 16. Durability, identity, and failure boundaries

A minimal persistent implementation needs immutable history objects, optional checkpoint objects, and an atomic branch-reference store. It does not require a distributed database or a new network protocol.

Before publishing a reference to a snapshot, make its required history and decoder objects durable according to the storage contract. Before accepting a checkpoint, validate its complete manifest and payloads. Temporary or partially written objects must never look complete after a restart.

A native file-based implementation can use staged immutable files and atomic publication, with appropriate synchronization and durability barriers. The actual filesystem/platform guarantees still need crash tests; a successful file write alone is not proof of durable atomic publication.

Branch revision updates and retry receipts must commit together. Record old and new tips in the audit log. History deletion must check retained roots and references; numerical cache eviction must never silently remove those roots.

A content hash is not an authorization token. Snapshot identifiers and checkpoint contents may reveal sensitive conversation data. Check access before reading, branching, sharing, or restoring them.

---

## 17. What counts as success

Use cumulative gates rather than a single “works” label.

| Gate | Evidence required |
|---|---|
| G0 — Reference semantics | Exact worked examples and toy state-machine tests pass; tests detect deliberately broken implementations. |
| G1 — Real-model continuation correctness | Fresh-reference logits, branch isolation, frontier, grammar/RNG, and rewind tests pass for a pinned model/backend profile. |
| G2 — Persistence and failure safety | Cold reconstruction, corruption rejection, restart/crash, retries, and atomic reference updates pass. |
| G3 — Efficient sharing | Counters prove hot branching avoids repeated prefix computation and full-prefix copying; COW, allocation pressure, and cleanup pass. |
| G4 — Evaluated exploration | Multiple retained futures can be evaluated independently and a validated candidate adopted without altering the others. |

Report the actual model/weight hashes, backend revision, cache precision, execution profile, input-fixture hashes, sequence lengths, branch counts, tolerances, maximum errors, allocation/work counters, and failed/unsupported cases.

Do not claim G1 from the toy model, G2 from an in-memory test, or G3 from a visually convincing chat interface. Reusing a prefix may save work while increasing coordination overhead; measure wall-clock performance separately from mathematical correctness.

### Minimal end-to-end demonstration

The first complete demonstration should be:

```text
1. Ingest one exact prefix and publish its snapshot.
2. Materialize and checkpoint it once.
3. Create three branch references to that snapshot.
4. Give each branch an explicit continuation policy or random stream.
5. Generate independent candidates; preserve every accepted boundary.
6. Rewind one branch and explore a different future.
7. Evict numerical state and reconstruct a retained candidate.
8. Compare each candidate with an independent fresh-history reference.
9. Evaluate all candidates in isolated, reproducible conditions.
10. Adopt a validated winner by atomically moving one reference.
```

**The final correctness statement is:** every candidate behaves like a continuation of its own exact saved history—not like a continuation of whichever sibling happened to run first.

---

### Validation performed for this artifact

The executable toy block in Section 11 was extracted verbatim and run locally with Python 3.13.5. All **9 toy tests passed**. This checks the included reference example only. The 24 backend acceptance tests, mutation-test campaign, real-model numerical comparisons, persistence/crash behavior, and GPU memory-sharing claims are **not run** here. No production certification gate is asserted from the toy results alone.

## 18. Sources and evidence boundaries

The algebra, API, invariants, worked fixtures, and proposed tests in this document are the design being specified here. References support the underlying cache behavior and current integration cautions; they do not certify this proposed implementation.

Sources inspected October 11, 2026. Upstream `main`/`master` and current documentation may change; pin a concrete revision when implementing.

[1]: https://huggingface.co/docs/transformers/main/en/cache_explanation
[2]: https://arxiv.org/abs/2309.06180
[3]: https://docs.vllm.ai/en/latest/design/prefix_caching/
[4]: https://raw.githubusercontent.com/ggml-org/llama.cpp/master/include/llama.h
[5]: https://docs.pytorch.org/docs/stable/notes/randomness.html

- **[1] Hugging Face Transformers, “How caching works.”** Causal prefix reuse, layer cache behavior, and the generation loop's consumed-versus-pending token distinction.
- **[2] Kwon et al., “Efficient Memory Management for Large Language Model Serving with PagedAttention” (2023).** Shared/paged KV storage as an implementation precedent.
- **[3] vLLM, “Automatic Prefix Caching,” design documentation.** Prefix-dependent block identity, additional input identities, and block reuse mechanics.
- **[4] llama.cpp, public `llama.h` interface.** Sequence memory/state APIs, sampler APIs, snapshot lifetime limitations, and aborted-decode state behavior.
- **[5] PyTorch, “Reproducibility.”** Limits of reproducibility across execution environments and releases.
