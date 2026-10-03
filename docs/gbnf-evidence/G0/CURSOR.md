# G0 cursor audit before native qualification

At the R4 base, the persistent loop uses the same `Verifier::run` and
`Verifier::commit` as speculative execution. For prompt length `n`, its first
window starts at `p = n - 1`. The prompt path has consumed `[0, n - 1)` and
`x = prompt[n - 1]` is pending feedback. The output selected after consuming
that token is not itself in the model's committed state yet.

For a target-only window, `T = 1`, accepted drafts `a = 0`, `commit(1)` retains
one input position, and one output token is emitted. After `m > 0` outputs,
the consumed state is the prompt plus the first `m - 1` outputs. The last
output is pending feedback. Thus `consumed.size() = n + m - 1`. This includes
the recurrent state, PLE history and attention state, not just the KV count.
The next window reads that pending token exactly once. A continuation that
supplies the full prompt and outputs reuses the consumed prefix and feeds the
pending token through the ordinary prompt/decode path.

`Verifier::run` keys sampling row `i` by absolute position `p + i`, including
the prompt offset. Penalty rows include consumed history, pending feedback,
then preceding draft inputs. No grammar matcher exists at G0. G2 must advance
its matcher when an output is committed, not again when that same token later
becomes model input.

Prompt cancellation leaves no reusable live session. Decode cancellation
drains the native `DONE cancel` through the existing `StrataEngine` iterator.
Prompt checkpoints contain native recurrent/PLE/index state and remain usable
without a drafter. Parked conversation snapshots additionally require draft
KV state, so G0 rejects their enabled configuration for target-only serving.

The existing speculative loop may commit an evaluated input tail before its
budget/EOS-limited emission. G0 does not qualify that behavior for a grammar
matcher. G5 must bound retained model and matcher progress together before
claiming constrained speculative execution. A parser checkpoint alone cannot
restore model state.

`STRATA_TRACE=1` now records each window's prompt, consumed and output counts,
pending token, window size and selection position. The real native probe
asserts the target-only relation for every recorded decode window. Native
results are separate artifacts; this audit is not itself passing evidence.
