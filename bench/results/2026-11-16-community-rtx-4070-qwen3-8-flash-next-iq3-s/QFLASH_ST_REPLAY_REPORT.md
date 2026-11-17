# qflash-st agentic coding replay trial (2026-11-16)

A follow-up to the [coder-st replay trial](../2026-11-15-community-rtx-4070-qwen3-8-flash-next-coder-iq1-m/CODER_ST_REPLAY_REPORT.md). That trial replayed six completed tickets on the **coder** variant of qwen3.8-flash-next at IQ1_M; it passed three and failed three. This trial re-ran the **three failed tickets** on the **full** qwen3.8-flash-next at IQ3_S, served by the same engine on the same machine.

**The question is suitability for coding work:** can the model reliably turn tickets into reviewable PRs, and could it replace or add to the local qwen3.8-27b? Speed and energy are secondary and are reported for context.

## How this was tested

### The harness

The test harness is unchanged from the coder-st trial; that report describes it in full. In short, a private **Jira dev orchestrator** plugin for the **Hermes** agent runtime turns a ticket into a pull request through a fixed pipeline:

1. A ticket analyzer writes a spec.
2. An implementer writes code and tests in a fresh worktree and proves each acceptance check.
3. Deterministic build, test and scope gates run.
4. A PR maker opens the PR.
5. An automated reviewer runs follow-up rounds.

Each agent session is limited to **80 model API calls**, and Hermes compacts the context as it nears the model's window. If a session ends without a parseable result, the orchestrator starts a **format-repair** session.

The implementer model is chosen per ticket, and that model also handles the review follow-ups. The analyzer, PR maker and reviewer always use the same hosted model (gpt-5.6-sol), so the implementer is the only variable.

### Models

| Profile | Model | Where it runs |
|---|---|---|
| `qflash-st` | qwen3.8-flash-next (full model), IQ3_S quantization | local, Strata inference engine 0.1.60 |
| `coder-st` | qwen3.8-flash-next **coder** variant, IQ1_M quantization | local, Strata inference engine 0.1.59 (same machine) |
| `qwen27` | qwen3.8-27b | local, a different machine |
| `sol` | gpt-5.6-sol | hosted (OpenAI) |
| `opus55` | Claude Opus 5.5 | hosted (Anthropic) |

Both Strata models ran on a 12 GB consumer GPU in a machine with about 61 GB of RAM, using a 131,072-token context.

**The machine's memory speed changed between the two trials.** Its RAM had been running below its rated speed. It was raised from 2133 to 3000 MT/s about two hours after the last coder-st run and four hours before the first qflash-st run; 3200 MT/s was not stable. So **every coder-st run was at 2133 MT/s and every qflash-st run at 3000 MT/s.** That is about 40% more memory bandwidth for qflash-st. It matters here because the engine reads the experts that are not cached on the GPU from system RAM. Each request had thinking on, with a thinking budget of 4096 tokens (maximum 8192) and up to 32k output tokens. No sampling parameters were sent, so Strata used its defaults for thinking mode: temperature 1.0, top_p 0.95 and top_k 20, which match the model's recommended settings.

IQ3_S holds about 47 GiB of expert weights, about twice the coder IQ1_M pack. On this machine they all fit in system RAM, and a few GiB of the most-used experts are cached on the GPU. The Strata profile adds `--vram-reserve-mib 768`. Without it, the engine warned that only 493 MiB of VRAM was free, which risks the driver paging GPU memory to system RAM on long prompts.

### Replay method

As before, each replay started from the **same pinned baseline commit** as the original run, and the analyzer re-ran on the hosted model. Each PR then went through the automated review and follow-up loop and was closed unmerged; the original PRs had already shipped.

### The three tickets

These are the three tickets coder-st failed. All of them need tests that depend on real infrastructure.

| # | Repository type | Work | Original model, PR size |
|---|---|---|---|
| T4 | ASP.NET Core app | Switch to `MapStaticAssets` (precompressed, fingerprinted assets) and load some scripts only on the pages that use them. Includes an integration test of compressed static-asset delivery. | qwen27, +263/−22, 8 files |
| T5 | .NET service + SQL Server | A newer successful run supersedes stale pending records it no longer produces. Uses SQL lock hints, with Tier-3 tests against a real SQL Server in Docker. | sol, +426/−29, 9 files |
| T6 | .NET service + SQL Server | Hold a run for review when its changes exceed a configured share. New rule class, options, config, unit and Tier-3 SQL tests. | opus55, +366/−15, 9 files |

## Verdict

**qflash-st passed all 3 tickets that coder-st failed.**

- **Call counts matched the originals.** Its implementer used 22, 56 and 37 API calls, against 28, 52 and 36 for the original models. coder-st hit the 80-call limit on all three.
- **It wrote the infrastructure tests itself.** That includes the compressed-asset integration test (T4) and Tier-3 SQL Server tests (T5, T6), which coder-st never got to.
- **It needed no more review rounds than the originals.** Fix rounds were 1, 2 and 1, against 3, 3 and 1.
- **On the one ticket that allows a direct comparison with qwen3.8-27b (T4)**, it needed fewer calls and fewer review rounds. It was also faster and used less energy, though those are secondary.

## Results per ticket

Times are for the implementer stage only. Calls are model API calls, which count against the 80-call limit. "Fix rounds" are review follow-ups that pushed a change.

| Ticket | Original run | coder-st run | qflash-st run | qflash-st fix rounds |
|---|---|---|---|---|
| T4 | qwen27: 28 calls, 12.9 min, 3 rounds | failed at 80/80, 26.5 min | **passed: 22 calls, 7.9 min** | 1 (20 calls, 4.1 min) |
| T5 | sol: 52 calls, 12.0 min, 3 rounds | failed at 80/80, 36.8 min | **passed: 43 + 13 calls, 24.6 min** | 2 (23 calls, 11.0 min; 12 calls, 2.5 min) |
| T6 | opus55: 36 calls, 8.2 min, 1 round | failed at 80/80, 29.7 min | **passed: 37 calls, 15.0 min** | 1 (24 calls, 7.3 min) |

"43 + 13" means a first session followed by a format-repair session; see below.

**Output and input tokens (implementer, from the session usage):**

| Ticket | Original | coder-st (failed) | qflash-st |
|---|---|---|---|
| T4 | 26.6k out / 1.39M in | 59.5k / 4.93M | **20.4k / 0.96M** |
| T5 | 17.8k / 4.08M | 76.8k / 5.62M | 51.7k / 3.24M |
| T6 | 22.0k / 4.69M | 51.9k / 5.73M | 34.5k / 2.59M |

qflash-st used fewer input tokens than every other model on all three tickets. Every call re-sends the tool definitions and the growing history, so input tokens track the call count and the context size. It generated more output than the hosted models on T5 and T6; that output is mostly thinking.

**Final PR sizes:**

| Ticket | Final size | Commits |
|---|---|---|
| T4 | +191/−23, 8 files | implementer + 1 fix |
| T5 | +514/−18, 8 files | implementer + 2 fixes |
| T6 | +352/−17, 10 files | implementer + 1 fix |

## Inference metrics

The local proxy logs every request to a time-series store. Each record holds:

- prompt and generation token counts;
- prompt-processing and generation time;
- prefix-cache reuse;
- GPU power.

The figures below cover every request in each implementer or follow-up window: 196 requests in total, none outside those windows. Throughput is token-weighted, meaning total tokens divided by total time. The coder-st and qwen3.8-27b figures are taken from the coder-st report.

**Hardware:**

- qflash-st and coder-st ran on the **same machine and GPU**, but one engine release apart (0.1.60 and 0.1.59) and at **different memory speeds** (3000 and 2133 MT/s). Their per-token speeds therefore don't isolate the model.
- qwen3.8-27b ran on a **different machine**, so that column compares the setups as deployed.
- The hosted models have no comparable data.

### Overall

| | qflash-st (IQ3_S) | coder-st (IQ1_M), thinking on | qwen3.8-27b originals |
|---|---|---|---|
| Requests | 196 | 529 | 104 |
| Generation speed | **61.2 tok/s** | 57.7 tok/s | 47.9 tok/s |
| Prompt processing (uncached tokens) | 1,177 tok/s | 1,230 tok/s | 762 tok/s |
| Prefix-cache reuse | 94.7% | 96.2% | 95.6% |
| Median context per request | 48k tokens | 58k tokens | 50k tokens |
| Max context | 97k tokens | 98k tokens | 82k tokens |
| Median generated tokens per request | 374 | 350 | 302 |
| Share of model time spent generating | 86% | 90% | 85% |
| Median GPU power | ~124 W | ~125 W | ~240 W |

Generation speed by context size, for qflash-st:

| Context | Requests | Gen tok/s | Prompt tok/s |
|---|---|---|---|
| < 32k | 66 | 62.6 | 1,231 |
| 32–64k | 69 | 60.6 | 1,324 |
| 64–96k | 59 | 60.5 | 841 |

Generation speed barely drops as the context grows: 62.6 tok/s below 32k and 60.5 tok/s at 64–96k. In the coder-st report, coder-st fell from 60.8 to 56.3 over the same range, and qwen3.8-27b from 54.0 to 42.7.

### Per run

"Model time" is the sum of per-request prompt and generation time. "Generated tokens" here is the proxy's count, which includes thinking.

| Run | Requests | Gen tok/s | Prompt tok/s | Model time | Generated tokens | GPU energy |
|---|---|---|---|---|---|---|
| T4 qflash-st | 22 | 62.2 | 1,089 | 6.1 min | 20k | **13.3 Wh** |
| T4 coder-st (failed) | 90 | 56.7 | 1,087 | 25.3 min | 78k | 52.7 Wh |
| T4 qwen3.8-27b | 29 | 47.1 | 641 | 10.8 min | 27k | 44.7 Wh |
| T5 qflash-st | 57 | 61.3 | 1,203 | 19.1 min | 61k | **42.3 Wh** |
| T5 coder-st (failed) | 93 | 58.4 | 1,470 | 35.1 min | 112k | 74.7 Wh |
| T6 qflash-st | 38 | 61.1 | 1,399 | 12.5 min | 39k | **27.8 Wh** |
| T6 coder-st (failed) | 84 | 58.1 | 1,583 | 29.0 min | 89k | 63.6 Wh |
| T4 fix round, qflash-st | 20 | 60.4 | 892 | 3.7 min | 11k | 7.7 Wh |
| T5 fix round 1, qflash-st | 23 | 61.7 | 941 | 4.8 min | 16k | 10.0 Wh |
| T5 fix round 2, qflash-st | 12 | 60.9 | 901 | 1.6 min | 5k | 3.3 Wh |
| T6 fix round, qflash-st | 24 | 59.9 | 1,071 | 6.1 min | 19k | 13.1 Wh |

### What the metrics show

- **It was a little faster per token than the coder variant.** It generated at 61 tok/s against 58 despite twice the expert data. Its runs also had about 40% more memory bandwidth (3000 against 2133 MT/s) and a newer engine release, so this report doesn't attribute the gain to the model. The speed comparison is the least reliable part of this report; the call counts, tokens and pass/fail results don't depend on memory speed.
- **Total output decided run time, again, and this time it was low.** On T4 it generated 20k tokens, against 78k for coder-st and 27k for qwen3.8-27b, so it finished first.
- **Energy per ticket fell sharply.**
  - T4 used 13 Wh, against 53 Wh for coder-st and 45 Wh for qwen3.8-27b.
  - All three implementer runs plus all four fix rounds together used **118 Wh**. coder-st's three failed runs alone used 191 Wh.
- **Tool execution took a larger share of each stage than in the coder-st trial.** Model time was 77–83% of implementer time, against about 95% for coder-st. The rest went to builds and test runs, including starting SQL Server containers for the Tier-3 tests. These runs got far enough to execute real infrastructure tests, which coder-st's runs never did.
- **Prefix caching stayed high,** at 94–96% reuse, so prompt processing was never the bottleneck.

## Session behaviour

| Run | Sessions | Compactions | Ended with |
|---|---|---|---|
| T4 implementer | 1 | 0 | a valid result |
| T5 implementer | 2 | 1 | a format repair, then a valid result |
| T6 implementer | 1 | 1 | a valid result |
| All four fix rounds | 1 each | 0 | a valid result |

- **One premature end of turn.** T5's first session finished the work in 43 calls, then ended its turn with "All checks green. Marking todos complete and reporting:" and no result. The 13-call format-repair session returned it. coder-st showed the same failure mode, but early in a run, before any work. Here it cost 13 calls at the very end.
- **Compactions did no harm.** T5 and T6 each compacted once, at about 97k tokens of context, and both finished.
- **No token-level corruption.** None of the runs showed coder-st's misspelled identifiers, malformed JSON, or tool calls printed as plain text. The fix rounds also followed the orchestrator's standing review guidance, for example checking sibling code paths for the same defect.

## Review rounds

| Ticket | Review finding | Fix |
|---|---|---|
| T4 | The deploy pipeline's cleanup pattern would also delete the static web assets **endpoints manifest** that `MapStaticAssets` needs at runtime. | Narrowed the pattern and added a regression test that reads the published output. |
| T5, round 1 | Tier-3 regression coverage was missing for the three supersede outcomes the ticket names. The first commit had Tier-3 tests, but not for every outcome. | Added five Tier-3 tests against SQL Server. This is the coverage coder-st never started. |
| T5, round 2 | A summary formatter left doubled spaces when optional fields were missing. | Fixed. |
| T6 | Two findings in one round: the order of the new "Held" check against the existing job-failure checks, and how held runs are counted in the summary. | Fixed both. |

Every finding was real but ordinary, and each was fixed in one round. T5's first finding is worth noting: like coder-st, qflash-st under-delivered on the Tier-3 tests in its first pass. The difference is that it wrote most of them, and finished the rest when asked.

## Conclusions

**On coding suitability, the full qwen3.8-flash-next at IQ3_S is a different proposition from the coder variant at IQ1_M.**

- **It passed the tickets the coder variant couldn't.** These are the hardest three of the six: real infrastructure tests, an HTTP pipeline test and Tier-3 SQL Server tests. It wrote those tests, its call counts matched the original models, and it showed none of the token-level errors behind coder-st's failures. That supports the coder-st report's view that low-bit quantization, not the model family, was the main problem.
- **Its PRs needed no more review than the originals.** Fix rounds were 1, 2 and 1, against 3, 3 and 1. Every finding was ordinary and fixed in one round.
- **Against qwen3.8-27b, the evidence is encouraging but thin.** On T4, the only ticket qwen3.8-27b originally did, it finished in fewer calls (22 against 28) and needed fewer review rounds (1 against 3). That is one ticket. On T5 and T6 it matched gpt-5.6-sol and Claude Opus 5.5 on calls and review rounds, which qwen3.8-27b has not been tested against here.
- **Speed is secondary but not a problem.** Wall time was about 2× the hosted models' on T5 and T6, and faster than qwen3.8-27b on T4.

**Caveats:**

- **It's a small sample:** three tickets, one run each.
- **It's only the three hardest tickets.** T1–T3, which coder-st passed, were not re-run.
- **Only T4 has a qwen3.8-27b original.** The other two originals ran on hosted models.
- **The memory speed changed between the trials** (2133 → 3000 MT/s), as did the engine release (0.1.59 → 0.1.60). This affects the speed, time and energy comparisons with coder-st, not the pass/fail results or the call and token counts.

On these results qflash-st looks suitable as a local implementer model, which coder-st was not. Whether it is **better** than qwen3.8-27b is not settled. That needs both models run on the same unseen tickets (see Next steps). It runs on a different machine from qwen3.8-27b, so adopting it could also add a second local job lane rather than replace one.

## Next steps

- Re-run T1–T3 on qflash-st for a full six-ticket comparison with the coder-st trial.
- Run a set of new tickets on both qflash-st and qwen3.8-27b to compare first-attempt quality and review rounds on unseen work.
- Watch for premature end of turn. It happened once in seven implementer and fix-round runs; a format repair recovers it, but each one costs extra calls.
