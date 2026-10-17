# Community quality baseline: Codex gpt-6.1-sol

Measured on 2026-10-13 using Codex CLI 0.162.1, configured with
`gpt-6.1-sol` and reasoning effort `ultra`. This runs the seven quality suites
from the [RTX 5070 Ti quantization study (#1850)](https://github.com/Niko1221/Strata/pull/1850)
through a Codex CLI adapter. It provides an external quality baseline for that
study. The model ran through the hosted Codex service; Strata inference and GPU
performance were not measured.

The source study is its
[published README](https://github.com/Zauberio/Strata/blob/study/quant-battery-rtx5070ti/bench/results/2026-10-10-community-quant-study-rtx5070ti/README.md)
at Strata revision `98f9d9288a02e694f1b4005b8d6d0393d76c8ab2`.
The adapter and some protocol controls differ, as described below, so these
results do not isolate the model as the only changed variable.

## Results

Percentages use the displayed task denominators. HumanEval+ keeps first attempts
separate from the final result after one allowed repair per failed task.
Five Bugs also records the final result after corrections identified during
the debugging and validation workflow, alongside the initial independent runs.

| Suite | Measured result | Coverage |
| --- | --- | --- |
| HumanEval+ | **156/164 (95.1%) first; 160/164 (97.6%) final** | All 164 tasks; eight repairs attempted, four succeeded |
| GSM8K | **98/100 (98.0%)** | 100 sampled test questions, 8-shot |
| MMLU abstract algebra | **96/100 (96.0%)** | First 100 `abstract_algebra` questions, 5-shot; not full MMLU |
| IFEval | **45/50 prompts (90.0%); 70/76 instructions (92.1%)** | First 50 prompts |
| Five Bugs | **Final after corrections: 25/25 visible and 25/25 hidden (100.0% each)**; initial: 25/25 visible, 20/25 hidden | Initial: five independent sessions per file; final: five corrected files, each graded five times |
| Needle recall | **25/25 (100.0%)** | Five lengths and five depths; largest measured input about 222K tokens |
| Tool Eval Bench | **90.2% mean, 0.4 percentage-point sample SD** | 92 scenarios, five trials; 460 scored cells after one infrastructure recovery |

The tool trials scored `90, 91, 90, 90, 90` after rounding each trial's
points percentage, following the study's aggregation. Their points were
`165, 168, 165, 165, 166` out of 184 each. The bootstrap 95% interval for the
mean of those rounded scores is 90.0–90.6%. The designated primary result
(trial five) is **90%, Excellent**, with no safety cap or warnings and 100% on
the critical group. This primary result is distinct from the five-trial mean.

- Tool Pass@5: **81/92 (88.0%)** scored full marks in at least one trial.
- Tool Pass^5: **77/92 (83.7%)** scored full marks in all five trials.
- Tool Hard Mode P: **85.4% mean, 2.6 percentage-point sample SD**, on TC-70
  through TC-92 (23 scenarios).
- HumanEval base tests passed **163/164 (99.4%)** both before and after repair.
  The expanded-test final failures were HumanEval/32, /91, /99 and /141.

[benchmark-scores.json](benchmark-scores.json) contains the exact counts,
per-scenario tool grades, needle grid, original and recovered tool summaries,
protocol qualifications and artifact SHA-256 hashes. It uses the separate
`codex-battery-quality-v1` schema, not the GPU speed report schema. Raw prompts,
answers and execution traces are not included in this score export.

[five-bugs-final-validation.json](five-bugs-final-validation.json) records
the final Five Bugs validation after corrections, including all 25 visible
and 25 hidden check outcomes and the corrected solution hashes.

## Harnesses and configuration

| Component | Version or source revision |
| --- | --- |
| Codex CLI | `0.162.1` |
| Configured model / effort | `gpt-6.1-sol` / `ultra` |
| BenchKit | `9636e3cd2f382ef1727984201f29baead56cfed8` |
| EvalPlus | `0.3.1`, HumanEvalPlus `v0.1.10` |
| Tool Eval Bench | `f7c34a1303c6fe9bcb52bdfc348de19082912f48` |
| Five Bugs | `503d58367672841e0511c8e3180f40538c8c53a1` |

The public pinned harnesses supplied the tasks and graders. The reference
study's tracked BenchKit changes and local Five Bugs runner were unavailable,
so public harnesses were adapted for the CLI transport. Generation used fresh
benchmark sessions with a short benchmark instruction scaffold, rather than
the existing desktop conversation. During the initial benchmark, no canonical
coding solutions or hidden tests were supplied to the solver. The later Five
Bugs debugging phase used documentation and checker feedback, as detailed
below. Model hardware, model weights/revision and
quantization are unavailable for this hosted run.

The adapter buffered each CLI answer and exposed it as an OpenAI-compatible
chat completion. Tool definitions and conversation messages were serialized
into the prompt, with a JSON output schema for tool calls. Tool execution used
the benchmark's mock environment. This was not the service's native
function-call transport. Tool scenarios ran with concurrency four; the
reference study used one. Requested temperature zero, seed 42 and output-token
budgets were not enforced by the Codex CLI. The reasoning effort was fixed to
`ultra`.

GSM8K sampling used seed 42. The original study's sampling seed was not
retained, so this choice is inferred rather than a verified identical sample.
MMLU covers only the stated subject and IFEval only the first 50 prompts. The
initial Five Bugs benchmark used one generated candidate per session, with no
checker feedback or repair round.

## Coding validation

Each HumanEval+ task received one first attempt. Each of the eight failures
then received exactly one verifier-guided repair opportunity, with the same
feedback protocol as the pinned coding harness. There were 172 generated
candidates in total, and no infrastructure exclusions from the 164-task
denominator.

The Windows grader's recorded outcomes were checked independently on Linux
using stock EvalPlus 0.3.1, reusing every generated candidate without another
model call. Pass/fail agreed for **172/172** candidates. Exact status labels
agreed for 171/172: HumanEval/39's first candidate was a timeout on Windows and
a failure on Linux; both were unsuccessful, and its repair passed.

Stock EvalPlus's `find_zero` progress-accounting behavior was retained in the
published score. A separate diagnostic correcting only that progress marker
still failed the expanded tests for both existing HumanEval/32 candidates, so
it did not change the combined HumanEval+ result.

## Five Bugs final validation after corrections

The initial run passed **25/25 visible** and **20/25 hidden** checks. All five
hidden failures were in t4: the generated tag-stripping solutions removed
literal comparison text such as `a < b and c > d`.

Following that run, Codex corrected the five task files during the debugging
and validation workflow. The fixes cover insertion at the end of a sorted
list, shared mutable defaults, decimal half-up rounding, tag stripping that
preserves comparison text, and month arithmetic with day clamping. Final
validation ran each corrected solution through the repository's existing
visible and hidden checkers five times: **25/25 visible and 25/25 hidden
(100.0% in both suites)**.

These are five repetitions of the same five corrected solutions. The
debugging phase consulted the repository README, which describes the t4
blind spot and published fixes. After the first post-correction validation,
the hidden t4 checker was inspected to diagnose a quoted-attribute expectation.
The final result is therefore feedback-assisted validation after corrections;
the initial independent-candidate score remains in `benchmark-scores.json`.

The t4 checker expects `b.png'>x` for `<img src='a>b.png'>x`, following a
first-closing-angle rule. The graded correction matches that checker contract.
A separate quote-aware variant returns `x` for the same input; both preserve
literal comparison text. The validation JSON records this qualification and
each graded solution's SHA-256 hash. No benchmark checker was changed.

## Long-context transport

The needle grid used the harness's estimated context targets of 1,024,
65,792, 130,560, 195,328 and 260,096 tokens, at depths 0%, 25%, 50%, 75% and
100%. These targets are labels from the harness's token estimate, not the
model's actual input-token counts.

In the JSON grid, the inherited field name `depth_percent` stores a fraction
from 0 to 1: `0.25` means 25%, and `1` means 100%.

The first 20 cells used the regular CLI bridge. The five largest prompts hit
the CLI frontend's 1,048,576-character aggregate input limit before inference. Those five were sent
through supported same-role history injection, preserving the exact logical
prompt and making one inference per cell. The originally valid 20 results
were retained. Actual input counts for the five largest cells were
221,680, 221,877, 221,595, 221,816 and 221,731 tokens. This shows recall at
approximately **222K actual input tokens**, with a different transport for
those five cells; it does not establish recall at 260K actual model tokens.

## Infrastructure recovery and measurement limits

Interrupted controllers resumed using exact archived response replay in
FIFO order. Replay preserved the original response identity and used zero
new model tokens. Later tool trials used distinct generated responses.
Replay durations are not model timing measurements.

In tool trial two, TC-56 completed two turns before the next request hit the
300-second transport timeout (331.09 seconds for the case). The original
native report excluded that infrastructure failure and retained its score of
166/182 points, rounded to 91%. That original record is preserved separately
in the JSON.

After all five primary trials ended, a validated supplement restored the two
exact original replies and generated the previously unanswered third reply
once, with a 600-second transport timeout. The logical request, tool state,
model/effort and protocol hashes were checked. The original eight-turn limit
and native grader were retained. The recovered case received two points,
giving trial two 168/184; its rounded score remains 91%. The headline tool
matrix includes this explicitly recorded supplement. Model failures were
not retried within the tool suite.

These results describe this adapted battery and configured model. They are
not a same-day Strata engine comparison: no GPU throughput, memory, power,
real streaming TTFT or inference-speed measurements were made. Buffered CLI
responses, synthetic streaming and replay timing cannot supply those
measurements. No normalized OVERALL score is reported. Differences in
transport, sampling controls and unavailable study runner changes limit
direct comparison with the reference study.
