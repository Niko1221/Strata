# coder-st agentic coding replay trial (2026-11-16)

**The question is suitability for coding work:** can the model reliably turn tickets into reviewable PRs, and could it replace or add to the local qwen3.8-27b? Speed and energy are secondary and are reported for context.

## How this was tested

### The test harness: a Jira-driven dev orchestrator on Hermes

The models were tested inside a private **Jira dev orchestrator**. It is a plugin for the **Hermes** agent runtime that turns Jira tickets into GitHub pull requests with no human in the loop until review. Each ticket runs through a fixed pipeline of agent roles and deterministic gates:

1. **Ticket analyzer:** reads the ticket and the target repository and writes an implementation spec. The spec covers scope (which files may change), coding patterns to follow, and measurable acceptance checks.
2. **Implementer:** works in a fresh git worktree. It edits code, adds or updates tests, runs the build and tests, and proves each acceptance check with captured command output. It returns a structured JSON result envelope.
3. **Deterministic gates:** the orchestrator, not the model, runs the repository's configured build and test commands. It also checks that only in-scope files changed.
4. **PR maker:** pushes the branch and opens the PR with a test summary.
5. **Review follow-ups:** every PR gets an automated first-pass code review. A reviewer agent validates each finding, and valid findings go back to the agent in a follow-up round, which pushes fixes to the same PR. "Rounds" below counts these follow-ups.

Each agent role is a one-shot Hermes session with the usual coding tools: file read/write/patch, terminal, Python execution and search. Two Hermes settings matter for reading the results:

- **80 API calls per session.** A session that reaches 80 model calls without returning a result fails with `max_iterations_reached`.
- **Context compaction.** When the conversation nears the model's context window, Hermes summarises earlier turns and continues. coder-st ran with a 131,072-token context.

If a session ends without a parseable result envelope, the orchestrator starts a "format repair" session that tells the model to continue and return a valid result.

**The model is a per-ticket setting.** A ticket label picks the implementer model, and that model also handles the follow-up rounds. The analyzer, the PR maker and the follow-up reviewer always use the same hosted model (gpt-5.6-sol), so the only variable between runs is the implementer.

The models used in the comparison:

| Profile | Model | Where it runs |
|---|---|---|
| `coder-st` | qwen3.8-flash-next **coder** variant, Strata IQ1_M quantization | local, served by the Strata inference engine (0.1.59) |
| `qwen27` | qwen3.8-27b | local |
| `sol` | gpt-5.6-sol | hosted (OpenAI) |
| `opus55` | Claude Opus 5.5 | hosted (Anthropic) |

### Replay method

- **The tickets:** six real tickets, each already completed by one of the other models (`qwen27`, `sol` or `opus55`), were replayed with coder-st as the implementer.
- **Same starting point:** each replay started from the **same pinned baseline commit** as the original run, in a new worktree and branch, so both models faced identical code.
- **The analyzer re-ran:** it uses the same hosted model, so the specs were equivalent but not byte-identical.
- **After review:** the replay PRs went through the same automated review and follow-up loop. They were then closed unmerged; the original PRs had already shipped.
- **Thinking on:** coder-st ran with thinking enabled (budget 4096, maximum 8192) and up to 32k output tokens per call.

### The six tickets

The original work spans three repositories: a React admin front end, an ASP.NET MVC web app, and a .NET background service backed by SQL Server.

| # | Repository type | Work | Original PR size |
|---|---|---|---|
| T1 | React admin app | Rework a table's columns to previous/new/change; hide one status unless it is filtered for. React + Jest. | +206/−66, 2 files |
| T2 | ASP.NET MVC app | Queue enable/disable clicks into a pending-changes list with per-item remove, and apply them all on Save. Controller, view model, Razor/JS, xUnit. | +371/−79, 5 files |
| T3 | React admin app | Add a signed "Change" column next to a quantity column. React + Jest. | +125/−3, 2 files |
| T4 | ASP.NET Core app | Switch to `MapStaticAssets` (precompressed, fingerprinted assets) and load some scripts only on the pages that need them. Includes an integration test of compressed static-asset delivery. | +263/−22, 8 files |
| T5 | .NET service + SQL Server | A newer successful run supersedes stale pending records it no longer produces. Uses SQL lock hints, with Tier-3 tests against a real SQL Server in Docker. | +426/−29, 9 files |
| T6 | .NET service + SQL Server | Hold a run for review when its changes exceed a configured share. New rule class, options, config, unit and Tier-3 SQL tests. | +366/−15, 9 files |

## Verdict

**coder-st passed 3 of the 6 replays.**

- **Passed: T1, T2 and T3.** These are front-end and controller work with plain unit tests.
  - Each pass needed **1.6–3.2×** the API calls of the original model and **1.8–6×** the implementer wall time.
  - Each passing PR needed **one** review round. The originals needed two.
- **Failed: T4, T5 and T6.** All three hit the 80-call limit, and every one needed tests that depend on infrastructure: an HTTP pipeline/compression test, or Tier-3 tests against SQL Server.
  - In T5 and T6 the production code was mostly written. The budget ran out before the tests were started.

On this sample coder-st is slower and less reliable than the local qwen3.8-27b on the same work, and well behind the two hosted frontier models. When it does finish, the PRs are of fair quality.

## Results per ticket

Times are for the implementer stage only. Calls are model API calls, which count against the 80-call limit. "Rounds" are review follow-ups that pushed a fix.

| Ticket | Original run | coder-st run | Outcome |
|---|---|---|---|
| T1 | qwen27: 27 calls, 12.7 min, 2 rounds | **67 calls, 22.3 min**, 1 round (30 calls) | **Passed** |
| T2 | sol: 19 calls, 5.5 min, 2 rounds | **31 calls, 10.6 min**, 1 round (58 calls, 20.3 min); the second review was clean | **Passed** |
| T3 | opus55: 11 calls, 1.6 min, 2 rounds | **35 calls, 9.7 min**, 1 round (34 calls, 7.3 min); the second review was clean | **Passed** (see note) |
| T4 | qwen27: 28 calls, 12.9 min, 3 rounds | 6 + 80 calls, 26.5 min | **Failed** at 80/80 |
| T5 | sol: 52 calls, 12.0 min, 3 rounds | 8 + 80 calls, 36.8 min | **Failed** at 80/80 |
| T6 | opus55: 36 calls, 8.2 min, 1 round | 80 calls, 29.7 min | **Failed** at 80/80 |

"6 + 80" means a first session that ended early, followed by a format-repair session.

**Note on T3:** the PR-maker step, which runs on the hosted model, failed with a transient provider `HTTP 400` after it had already opened the PR. The orchestrator's Continue re-ran that step against the existing PR, and coder-st's work was unaffected.

**Input tokens** rose with the call count. T1 used 4.19M tokens against qwen27's 1.64M, and T2 used 1.77M against sol's 0.82M. Every call re-sends the roughly 28k-token tool definitions plus the growing history.

## Inference metrics

The local proxy logs every request to a time-series store. Each record has prompt and generation token counts, prompt-processing and generation time, prefix-cache reuse, and GPU power. The figures below come from the requests made during each job's implementer or follow-up window. Throughput is token-weighted: total tokens divided by total time.

**Hardware caveat:** the two local models ran on **different machines**, so throughput and energy compare the two setups as deployed, not the models on equal hardware. The hosted models (gpt-5.6-sol, Claude Opus 5.5) have no comparable data.

**Memory speed:** the machine running coder-st had RAM running below its rated speed, at **2133 MT/s**, for every run in this report. It was later raised to 3000 MT/s, about 40% more bandwidth, and the engine reads uncached experts from system RAM. coder-st's figures here are therefore probably lower than that machine gives now.

### Overall

| | coder-st, thinking on | coder-st, thinking off (void runs) | qwen3.8-27b originals (T1, T4 + T1 fixes) |
|---|---|---|---|
| Requests | 529 | 282 | 104 |
| Generation speed | **57.7 tok/s** | 66.2 tok/s | 47.9 tok/s |
| Prompt processing (uncached tokens) | **1,230 tok/s** | 746 tok/s | 762 tok/s |
| Prefix-cache reuse | 96.2% | 98.3% | 95.6% |
| Median context per request | 58k tokens | 54k tokens | 50k tokens |
| Max context | 98k tokens | 92k tokens | 82k tokens |
| Median generated tokens per request | 350 | 155 | 302 |
| Share of model time spent generating | 90% | 75% | 85% |
| Median GPU power | ~125 W | ~121 W | ~240 W |

Generation speed falls as the context grows, but more slowly for coder-st than for qwen3.8-27b. These figures cover all requests in the window, including sessions outside this trial:

| Context | coder-st gen tok/s | qwen3.8-27b gen tok/s | coder-st prompt tok/s | qwen3.8-27b prompt tok/s |
|---|---|---|---|---|
| < 32k | 60.8 | 54.0 | 1,723 | 954 |
| 32–64k | 57.0 | 47.3 | 1,274 | 726 |
| 64–96k | 56.3 | 42.7 | 833 | 638 |
| > 96k | 55.3 | 40.0 | 770 | 727 |

### Per run

"Model time" is the sum of per-request prompt and generation time. Generated tokens ≈ generation time × generation speed.

| Run | Requests | Gen tok/s | Prompt tok/s | Model time | ≈ Generated tokens | GPU energy |
|---|---|---|---|---|---|---|
| T1 coder-st | 69 | 56.8 | 1,008 | 21.3 min | 66k | 45.2 Wh |
| T1 qwen3.8-27b | 28 | 47.6 | 772 | 11.9 min | 29k | 49.5 Wh |
| T4 coder-st (failed) | 90 | 56.7 | 1,087 | 25.3 min | 78k | 52.7 Wh |
| T4 qwen3.8-27b | 29 | 47.1 | 641 | 10.8 min | 27k | 44.7 Wh |
| T2 coder-st | 32 | 61.1 | 1,107 | 10.2 min | 34k | 22.3 Wh |
| T3 coder-st | 36 | 57.2 | 964 | 9.1 min | 28k | 19.3 Wh |
| T5 coder-st (failed) | 93 | 58.4 | 1,470 | 35.1 min | 112k | 74.7 Wh |
| T6 coder-st (failed) | 84 | 58.1 | 1,583 | 29.0 min | 89k | 63.6 Wh |
| T1 fix round, coder-st | 31 | 56.0 | 1,061 | 4.9 min | 14k | 10.0 Wh |
| T2 fix round, coder-st | 59 | 56.1 | 817 | 11.1 min | 33k | 23.5 Wh |
| T3 fix round, coder-st | 35 | 57.5 | 978 | 5.0 min | 14k | 10.2 Wh |
| T1 fix rounds ×2, qwen3.8-27b | 47 | 49.3 | 848 | 10.0 min | 24k | 40.8 Wh |

### What the metrics show

- **Per token, coder-st is the faster setup.** It generated about 20% faster than qwen3.8-27b (57.7 against 47.9 tok/s), processed prompts about 60% faster, and held its speed better at long context, at about half the GPU power.
- **Its runs were slower because it generated far more.** On the same tickets it produced 2.3–2.9× the tokens (T1: 66k against 29k; T4: 78k against 27k), spread over 2.5–3.1× as many requests. Generation took about 90% of model time, so total output decided the run time.
- **Tool execution was a small share of each stage.** Model time accounts for nearly all of each implementer stage (T1: 21.3 of 22.3 minutes).
- **Energy per ticket ended up about even.** The lower power draw roughly offset the extra tokens: T1 used 45 Wh against 49.5 Wh, and T4 used 53 Wh against 45 Wh. The three failed runs alone used 191 Wh and produced no PR.
- **Prefix caching worked well throughout,** at 95–98% reuse. Each request re-processed only about 2k new tokens on average, out of a 50–98k context, so prompt processing was never the bottleneck.
- **Thinking roughly doubled the median output per request** (155 → 350 tokens) and lowered generation speed from 66 to 58 tok/s. Without thinking, the model did not finish any of the three tickets.

### Void first attempt

The first three replays (T1, T4 and T2) all failed at 80/80 within 7–11 minutes. The local OpenAI-compatible proxy in front of the model server had no per-model policy for coder-st yet. Its fallback policy disabled thinking (`enable_thinking=false`) and capped output at 8192 tokens.

After a coder-st policy was added (thinking on, budget 4096/8192, 32k output), every run had thinking enabled. The thinking-off runs are excluded from all results above. **Lesson:** check that a reasoning model actually has thinking enabled end to end before benchmarking it.

## How the failures happened

### T4: invented APIs, then test churn

1. **First session (6 calls):** gathered context, then ended its turn with "Context gathered. Setting up the todo list, then making the edits." That reply had no tool call and no result, so the orchestrator started a format-repair session.
2. **Repair session, about calls 1–15:** made the real code change, which was small.
3. **About calls 15–45:** looked for NUnit and HTTP-client test types that don't exist in the repo. It ran `find /` and `strings` on DLLs instead of copying the repo's existing test patterns.
4. **The rest:** rewrote the gzip-compression test over and over, with web searches in between. It finished at 80 calls with one test still failing.

There was one compaction, at about turn 63. The derail had started around turn 33, so compaction did not cause it. qwen3.8-27b finished this ticket in 28 calls.

### T5: production code done, tests never written

This is the largest ticket in the set; even gpt-5.6-sol needed 52 calls and three review rounds.

1. **First session (8 calls):** returned an empty reply, which triggered a format-repair session.
2. **Repair calls 1–50:** a good start. It wrote the supersede logic, including a SQL candidate select with `UPDLOCK/HOLDLOCK`, the status update and an audit insert, across 5 files (+106/−8), and the build was clean.
3. **Calls 51–88: stuck on its own misreadings.** It never wrote the six required Tier-3 tests.
   - **Its typos looked like corrupted output.** The model misspelled identifiers in its own tool calls: `grep "I gnored"`, then `NeesReview` and `NeessReview` for a field named `NeedsReview`. Its searches with those spellings came back empty, so it decided the *tool output display* was corrupting identifiers.
   - **It spent calls checking for corruption that wasn't there.** Calls 77–80 went to dumping character codes and hex to work out how `NeedsReview` is "really" spelled.
   - **Its own coding slips added more calls.** It passed the wrong parameter name to a tool, and passed file contents as `re.compile` flags. It also wrote a regex that matched nothing and so concluded that test helpers it had just read did not exist. Two calls later it found that they did.
   - **It ran out of calls.** All 7 acceptance checks were reported `blocked`.

There were two compactions, at turns 29 and 62. Both compaction summaries spell `NeedsReview` correctly, and the bad spellings first appear in the model's own reasoning afterwards. So, again, compaction did not cause the failure.

### T6: re-reads and malformed calls, no tests

This was a single session (no early stop) with three compactions.

- **About calls 1–14, setup misfires:**
  - It called directly available tools through the deferred-tool wrapper.
  - It searched old sessions for the analyzer's spec instead of reading its own prompt.
  - It tried to create a throwaway "skill".
- **Calls 15–44, reading:** mostly re-reads. One central file was read 12 times across the run.
- **Calls 45–50, todo lists:** six attempts to write a todo list, most rejected as invalid JSON.
- **Calls 51–89, code:** reasonable code: a new rule class, options, wiring, config and docs (+72/−13 plus the new file), with many patch retries along the way.
- **The end:** while fixing a compile error it had introduced, its 80th reply was a Qwen-style `<tool_call><function=…>` block printed as plain text, not a real tool call. The session ended with exit reason `unknown`.

No tests were written; the original PR had about 280 lines of unit and Tier-3 tests. This is the same pattern as T5: the code was mostly done, and the budget went before the tests were started.

### T3 review round

The first review of coder-st's T3 PR found the **same defect** that the opus55 original got in its first round: a decimal difference was left unrounded, so fractional quantities displayed values like `200.44399999999996`. coder-st fixed it in one round. Its PR was about twice the size of the original (+245 against +125 lines).

## Cross-cutting findings

1. **Premature end of turn.** In 4 of the 7 coder-st implementer runs, the first session ended early with a "now I'll do X" message or an empty reply, and no tool call. That count includes the void thinking-off runs. The passing T1 and T2 runs did not do this. Each one costs a format-repair restart and a re-read of the context.
2. **Token-level errors.** Misspelled identifiers, malformed JSON, tool calls printed as text and wrong parameter names all point to the low-bit quantization damaging exact token reproduction. That matters most in C# work, where exact identifiers drive grep, edits and tests. The model then treated its own mistakes as facts about the environment ("the display is corrupting text") and used up its call budget investigating them.
3. **Knowledge gaps on .NET test tooling.** In T4 it went looking for test types that don't exist rather than copying the repository's existing patterns.
4. **Compaction is a symptom, not a cause.** coder-st needs more calls, so its context fills and compacts; the originals never compacted. Compactions cost a few re-reads each but did not trigger any failure.
5. **When it finishes, the output is fair.** All three passing PRs needed one review round, against two for each original. Their review findings were real but ordinary, and in T3 the same finding came up for the frontier model.
6. **Throughput.** A passing implementer stage took 10–22 minutes, against 2–13 minutes for the original models. A failing one used 26–37 minutes before stopping.

## Conclusions

**For this workload, the coder variant at this quantization is not a good fit.** The local **qwen3.8-27b** has done better so far.

- **Reliability:** qwen3.8-27b completed the replayed tickets it originally owned (T1 and T4) in 27–28 calls. coder-st needed 67 calls for T1 and failed T4 at the 80-call limit.
- **Speed:** qwen3.8-27b's implementer stages took about 13 minutes. coder-st took 10–22 minutes when it passed and 26–37 minutes when it failed. Because each local model runs one ticket at a time, those failed runs also block the queue.
- **Fewer review rounds don't make up for it.** coder-st's one round per passing PR, against two for the originals, is a real result but a small sample. It does not offset failing half the tickets.

On these results, coder-st is unlikely to stay as an implementer model for this orchestrator. The one remaining question was whether the full (non-coder) qwen3.8-flash-next Strata build does better. The follow-up [qflash-st replay trial](../2026-11-16-community-rtx-4070-qwen3-8-flash-next-iq3-s/QFLASH_ST_REPLAY_REPORT.md) re-ran the three failed tickets on it at IQ3_S, and it passed all three.

## Next steps

- coder-st at this quantization can handle small to medium UI or controller tickets with plain unit tests. It is not reliable on tickets whose tests need real infrastructure: databases, HTTP pipelines, or containerised SQL Server.
- The clearest failure modes are token-level (T5, T6), which suggests a higher-bit quantization may help.
- **Next trial (done):** re-run the three failed tickets (T4, T5, T6) on the Strata build of the full qwen3.8-flash-next, not the reduced coder variant. See the [qflash-st replay trial](../2026-11-16-community-rtx-4070-qwen3-8-flash-next-iq3-s/QFLASH_ST_REPLAY_REPORT.md).
