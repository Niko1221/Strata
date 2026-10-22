# Small Context Output Budget Validation

Measured on 2026-10-22, native Windows, Node v24.12.0, RTX 5080 (16,303 MiB
reported VRAM), 61 GiB RAM. Source base: `011c9e600a93bbc9fd451c563501b5831c32bbca`.
App 0.2.2's shipped Host, Harness 0.2.1-alpha.3 and pi-ai 1.1.0 were copied to
isolated acceptance folders. The installed components were left unchanged.
The candidate used the source-owned `patchStagedHarnessBudget` helper on its
copied Harness. The complete `pnpm deploy` staging pipeline was not run.

## Actual Default-Agent Turn

Both runs used the shipped Host's `/api/chat/stream` endpoint and real Harness,
with `agent: true` in the start frame, `demo: false`, Context 8192, Medium
effort and Manual mode. Neither `STRATA_AGENT=off` nor the direct-chat fallback
was used. This is Host API acceptance, not a native App rendering test.

The existing Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS model ran on engine 0.1.47,
through the same public Tray-managed loopback API. The saved API model had
`contextMax: 8192`, `maxOutput: null`, `vision: null`, `reasoning: null`.
Both fresh sessions sent exactly `Reply with exactly OK.` with empty history.
No effort, permission, retry, model or Context setting changed between runs.

| Observed Field | Shipped Baseline | Candidate Harness Copy |
| --- | --- | --- |
| Wire `max_completion_tokens` | 1 | 104 |
| Wire Thinking | Enabled, Medium | Enabled, Medium |
| Message Count / Tool Count | 5 / 26 | 5 / 26 |
| Serialized Message / Tool Characters | 3436 / 24071 | 3436 / 24071 |
| Engine Prompt Tokens | 6540 | 6540 |
| Completion Tokens, Including Thinking | 1 | 25 |
| Finish Reason | `length` | `stop` |
| Model Answer | Empty | `OK` |

The baseline's UI frames contained the empty-answer notice. The candidate
emitted an answer delta of `OK`. A transparent loopback recorder captured the
wire caps and token usage without recording prompts or credentials. The
candidate reused 6110 cached prompt tokens; these runs establish behavior,
not a speed comparison.

The synthetic unit fixture uses a prompt estimate of 6545 and yields 1135
output tokens after the patch. That is not the live prompt's estimate: the
live candidate had only 104 estimated output tokens after its safety margin.
The native tokenizer reported 6540 prompt tokens in both real runs. A short
answer now succeeds; long answers, long reasoning and full tool conversations
at 8K have not been validated.

## Source-Owned Checks

Passed on native Windows with the same Node runtime:

```powershell
# STRATA_TEST_PI_AI_DIR names the read-only installed pi-ai package.
node --test --test-isolation=none apps/host/tests/harness-budget.test.mjs
# 9 passed, 0 failed, 0 skipped.

node --test --test-isolation=none apps/host/tests/harness-staging.test.mjs apps/host/tests/agent-compaction.test.mjs apps/host/tests/agent-shape.test.mjs apps/host/tests/agent-events.test.mjs
# 24 passed, 0 failed, 3 skipped.

node --check apps/desktop/scripts/stage-harness.mjs
git diff --check
```

The three skips are the real Harness compaction turn, real response-settings
turn and staged-component inventory without `STRATA_HARNESS_DIR`. The arithmetic
tests stub the prompt estimator; they do not establish native token counts.
They cover the 8K failure, caller caps of 1 and 128, no remaining room, 32K,
64K/128K behavior, unknown windows, patch provenance, rejection of unexpected
dependencies and shared files, and preservation of the installed original.

The pre-fix regression intentionally failed (`1` instead of `1135`). The
post-fix source checks and the real candidate turn passed. No attributable
post-fix check failure was observed.

## Patch And Limits

The staged module subtracts `min(4096, floor(contextWindow / 16))` as safety
room, retaining the caller's output cap and the original one-token floor.
At 8K the margin is 512; at 64K and above the old calculation is retained.
This is the common pi-ai clamp, so the staged patch affects small windows
across its protocols. Only the local OpenAI chat route was exercised live.

`BUNDLE.json.patches` records `strata-pi-ai-context-safety-v1`, pi-ai 1.1.0,
the patched file and both hashes. A different version, unexpected bytes or a
shared module stops staging for explicit source review. This does not repair
an already-installed Harness or modify its files at App startup.

Not run: full Harness build/deploy, signed installer build, native App
rendering with the candidate, other operating systems or providers, long
conversations, and CUDA/HIP/SYCL builds (no engine source changed).

Machine-readable evidence: [output-budget-validation.json](output-budget-validation.json).
