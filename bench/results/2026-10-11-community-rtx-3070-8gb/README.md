# 2026-10-11 community report: RTX 3070 8 GB, Ryzen 9 5900X, 96 GB RAM

One 8 GB card, Linux, Docker, Strata engine 0.1.41, the original Flash-Next in three quantizations (IQ2_XS, IQ3_S, IQ3_XXS), driven by OpenCode 1.18.16 on agent-coding tasks, plus a sampling A/B for #1274.

## What this report has, and what it does not

- **Section 1:** six local configurations on the same eight agent-coding tasks (quantization and thinking effort), one run each.
- **Section 2:** the same eight tasks with three paid hosted agents, **for reference only** (different tools, different hardware, not a fair speed comparison).
- **Section 3:** greedy decoding against the "thinking" sampling preset that PR #1274 adds as `--thinking` (opt-in in its current version): speed, draft acceptance, repetition loops, tool-call validity, on synthetic prompts.
- **Not tested:** the 4,096 / 32,768 / 128,000-token prompt sweep, recall checks, vision, more than one GPU, AMD/Intel, the instruct preset, batch mode (`parallel`), and engines 0.1.42 and 0.1.43 (released after these runs).

## Hardware

| Part | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3070, 8 GB (8192 MiB), PCIe 4.0 x16, power limit 240 W |
| CPU | AMD Ryzen 9 5900X, 12 cores / 24 threads |
| RAM | 96 GB DDR4-3200: 2 x 16 GB + 2 x 32 GB (93 GiB visible to the OS) |
| Storage | NVMe SSD (model files on a 1.8 TB volume) |
| Other GPU work | Other GPU services on this machine (speech and similar) were stopped while the model ran. Other GPU users during single runs were not logged. |

## Software

| Part | Value |
| --- | --- |
| OS | Ubuntu 24.04.4 LTS, Linux 7.0.0-34-generic |
| Driver | NVIDIA 580.178.04 (CUDA 13.0 as reported by `nvidia-smi`) |
| Strata | engine 0.1.41, built from source in Docker (a local Dockerfile derived from the repository's `Dockerfile`) |
| Agent (section 1) | OpenCode 1.18.16 through the OpenAI-compatible endpoint |
| In front of the engine | a small FIFO queue proxy (one request at a time) that forwards requests unchanged and keeps streams alive while they wait |

## Model and settings

- Model: original Flash-Next GSQ-RCO GGUF, `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` at the revision pinned by `setup.py` of 0.1.41 (`ed59f92`). Files `Qwen3.8-Flash-Next-GSQ-RCO-{IQ2_XS,IQ3_S,IQ3_XXS}-0000{1,2}-of-00002.gguf`.
- Sections 1 and 2 engine settings (all runs): `--max-context 196608 --kv int8 --kv-resident 20480 --spec 4 --spec-min-p 0.5 --mtp`, `--expert-cache auto --prefill auto`, one request at a time, sampling left at the server default. What OpenCode itself puts into its requests was not recorded. `m1` and `m2` ran on an experimental service unit without idle unload; `m3` to `m5` ran on the production unit, which unloads the model after 600 s without requests (the tasks ran back to back). `m1` to `m5` started on a freshly started engine; `m0` is the local entry of the section 2 comparison, run earlier, and its engine start state was not recorded.
- Section 3 engine settings: IQ2_XS (`qwen3.8-flash-next-iq2_xs`), `--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp --max-context 262144 --kv int8 --kv-resident 32768`, one request at a time, idle unload after 1,800 s, same host. (Section 3 runs on IQ2_XS, not IQ3_S as announced in the comment on #1274: the machine was switched to IQ2_XS on 2026-10-11.)

## 1. Agent coding: six local configurations, eight tasks

The tasks come from the author's own software projects: three bug fixes and five feature additions, four easy, three medium and one hard. The projects, the code and the task texts are private and are not described here. Each task has hidden tests (automatic, 80 of 100 points) and a name-masked quality review (20 points, four questions). A task passes when all hidden tests and the existing tests pass.

Every configuration ran each task once (n = 1). Re-running the same configuration (`m0` against `m3`) changed the total time by 0.6%; differences below 10% are treated as noise.

| Run | Quantization | Thinking | Passed | Mean score (of 100) | Total time, 8 tasks | Time per pass | Decode tok/s | Prompt tok/s | Tool-call errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| m0 | IQ3_S | default (high) | 8/8 | 100.0 | 101 min 40 s | 12 min 43 s | 30.8 | n/a | 10/183 (5.5%) |
| m1 | IQ2_XS | default (high) | 8/8 | 99.4 | 72 min 19 s | 9 min 02 s | 46.5 | 563.1 | 14/252 (5.6%) |
| m2 | IQ3_XXS | default (high) | 8/8 | 100.0 | 107 min 54 s | 13 min 29 s | 36.9 | 429.3 | 12/241 (5.0%) |
| m3 | IQ3_S, repeat of m0 | default (high) | 8/8 | 100.0 | 101 min 04 s | 12 min 38 s | 31.5 | 301.8 | 6/192 (3.1%) |
| m4 | IQ3_S | medium | 8/8 | 100.0 | 68 min 32 s | 8 min 34 s | 32.3 | 294.1 | 10/158 (6.3%) |
| m5 | IQ3_S | low | 8/8 | 100.0 | 95 min 42 s | 11 min 58 s | 31.4 | 289.5 | 6/192 (3.1%) |

How to read it:

- Every configuration passed all eight tasks, so accuracy does not separate them; time, speed and tool-call errors do.
- IQ2_XS decoded about 1.5 times faster than IQ3_S (46.5 against 30.8 to 32.3 tok/s) and read prompts about twice as fast (563 against 290 to 302 tok/s), and finished the eight tasks 28 min 44 s sooner than `m3`. Mean quality was 99.4 against 100.0, below the 2-point threshold set before the runs.
- Thinking effort `medium` cut the total time by 32% against the default and the share of thinking tokens from 54% to 43%. `low` was within noise (-5%).
- "Prompt tok/s" and "Decode tok/s" are from the engine's log lines. `m0` was measured before engine timestamps were logged, so its prompt speed is n/a. Thinking tokens (not shown) are an estimate counted from progress lines.
- Peak GPU memory was 7,676 to 7,799 MiB; the lowest free system RAM was 21.1 to 32.0 GiB. No retries, timeouts, engine restarts or out-of-memory kills.

## 2. The same eight tasks with paid hosted agents (reference only)

Same tasks, same hidden tests and the same name-masked review; one run each. The agents ran in a sandbox with internet and without access to the local network. The comparison mixes tools, models and hardware, so use it only to place the local numbers.

| Agent | Model | Passed | Mean score | Median time per task | Tokens reported by the tool |
| --- | --- | --- | --- | --- | --- |
| Codex CLI 0.160.0 | gpt-5.6-terra, reasoning xhigh | 8/8 | 99.4 | 179.8 s | 6,984,690 |
| Antigravity CLI (`agy`) 1.3.2 | Gemini 3.8 Flash (High) | 8/8 | 100.0 | 560.5 s | 3,619,431 |
| Claude Code 2.1.296 | Claude Sonnet 5.5, effort max | 8/8 | 99.4 | 1,184.0 s | 79,393,206 |
| OpenCode 1.18.16 + this machine (IQ3_S, default thinking) | Flash-Next IQ3_S | 8/8 | 100.0 | 835.0 s | 11,940,936 |

Limits of this table:

- Each tool counts tokens its own way (cached input is included for some), so the token column is not comparable across rows.
- The reviewer that gives the 20 quality points is also from the Sonnet family, which is one of the candidates. Names are masked and 80 of the 100 points are automatic, but the conflict is not removed.
- Eight tasks, one run each: small score differences (99.4 against 100.0) are one task graded 95 instead of 100.
- The thinking-effort settings differ per tool.

## 3. Sampling: greedy against the "thinking" preset (for #1274)

The A/B ran on 2026-10-11, 10:05 to 10:52 (local time), with 19 prompts x 2 arms x 3 rounds = 114 requests, 0 failed, engine 0.1.41.

Ratios preset / greedy for the same prompt in the same round; median (min-max).

| workload | decode tok/s | generated tokens | generation time (tokens / decode tok/s) | wall time |
| --- | --- | --- | --- | --- |
| all | 0.94 (0.83-1.23), n=57 | 1.08 (0.39-2.63), n=57 | 1.13 (0.42-2.91), n=57 | 1.25 (0.32-2.73), n=57 |
| chat | 0.92 (0.88-1.05), n=6 | 1.20 (0.99-1.57), n=6 | 1.30 (0.95-1.73), n=6 | 1.36 (0.91-1.78), n=6 |
| code | 0.96 (0.92-1.01), n=6 | 1.21 (0.92-1.72), n=6 | 1.23 (0.92-1.81), n=6 | 1.21 (0.98-1.68), n=6 |
| math | 0.93 (0.84-1.13), n=30 | 1.12 (0.39-2.63), n=30 | 1.21 (0.42-2.91), n=30 | 1.20 (0.41-2.73), n=30 |
| tools | 1.01 (0.83-1.23), n=15 | 0.96 (0.63-1.36), n=15 | 1.03 (0.51-1.40), n=15 | 1.61 (0.32-2.00), n=15 |

- Totals: greedy generated 60,470 tokens, 1213.3 s generation time, 1244.4 s wall time; preset generated 63,014 tokens, 1363.6 s generation time, 1396.2 s wall time (+4.2% tokens, +12.2% wall time).
- Draft acceptance: greedy 81.2% of 47,837 offered drafts, preset 72.7% of 49,600.
- Quality checks: math exact answers 26/30 greedy against 25/30 preset; tool calls valid against their schema 15/15 and 15/15; looped answers 0/57 and 0/57; answers that hit max_tokens 4/57 and 4/57, all of them on two of the four harder counting prompts; the server's own loop recovery fired 0 times in both arms.

What the numbers show: The preset wrote a little more text (tokens 1.08 median, +4.2% in total) at a slightly lower decode speed (0.94). Generation time was 1.13 median and total wall time +12.2%. Draft acceptance fell from 81.2% to 72.7% while decode speed fell by 6%, so tokens/second alone understates the cost in time. Accuracy and tool-call validity did not differ in this test (one answer apart). No answer looped in either arm, so there was no greedy loop to remove here. The test uses short prompts with at most 6,000 generated tokens, so it says nothing about long agent sessions.

Read with care: The wall-time ratio of the tool-call prompts (1.61) is an order effect, not a sampling effect: the first request of a pair reads the prompt in about 1,514 ms, the second finds it in the cache and needs about 72 ms, and the ratio is 0.50 when greedy went first (7 pairs) and 1.85 when the preset went first (8 pairs) (split by the `time` of each row in `data/ab_results.jsonl`). Greedy was not exactly repeatable on this server: the final answer text was identical in all three rounds for only 6 of 19 prompts (preset: 4 of 19), so the rounds are separate samples.

Method, so others can repeat it (`data/ab1274.py`; prompts are synthetic or public, no private content):

- Both arms are sent explicitly on every request: greedy = `temperature=0`; thinking = `temperature=1.0, top_p=0.95, top_k=20` and a fixed seed per prompt and round. Everything else is the same: streaming, `reasoning_effort=medium`, the same prompt and `max_tokens`, one request at a time, arm order alternating from prompt to prompt and from round to round.
- 19 prompts: 10 arithmetic and counting problems with one integer answer (six easier, four harder counting problems; the answers are computed by the script), 5 tool-calling prompts with 5 tool schemas, 2 code-writing prompts, 2 short chat prompts. 3 rounds, 114 requests, none failed.
- Speed and draft numbers come from the engine's own `timings` (`predicted_per_second`, `draft_n`, `draft_n_accepted`). A looped answer is one that hit `max_tokens` and whose last 4,000 characters are mostly repeats (a compression test and a repeated-window test). The server's own repeated-reasoning recovery is counted from `/status` when the request's `max_tokens` matches.
- Before every request the script waits until the queue is empty, so nobody else's work is queued behind a whole round.
- This is not `tools/ab_engine.py`: that script A/Bs engine builds with greedy requests and thinking off, restarting the server per arm. This one A/Bs the request's sampling on one running server, with thinking on, and also counts exact answers, tool-call validity and loops, because the author of #1274 pointed out in review that speculative decoding may accept more tokens, that greedy favors repetitions, and that tokens per second can therefore be misleading.
- To repeat it against your own server: `AB_BASE=http://127.0.0.1:8080 python3 data/ab1274.py run --out my-run` (add `AB_KEY_FILE=<file with the key>` only if the server needs an API key), then `python3 data/ab1274.py report --out my-run`. `python3 data/ab1274.py selftest` checks the script against a fake server first.

## Limits

- Section 1: one run per configuration; private tasks, so others cannot repeat them (settings and numbers are given instead); one agent (OpenCode); the proxy in front of the engine was in every run; sampling was the server default plus whatever OpenCode sends.
- Section 3: synthetic prompts, few of them; exact-answer and schema checks only (no judgement of writing or code quality); one card, one quantization; greedy is not exactly repeatable here (see Section 3).
- Section 3 waits for an empty queue before every request, so other users' long requests were never queued behind a whole round (the script waits on our queue proxy's `/queue`); here all three rounds finished.
- No claim about other GPUs, quantizations or engine versions.

## Data

- `data/agent_runs_summary.json`: the numbers of sections 1 and 2 (no task text, no project names).
- `data/ab1274.py`, `data/ab_results.jsonl`, `data/ab_RESULTS.md`: the section 3 script, every request, and the generated report.
- `data/ab1274.py` is the script that ran, with three changes made afterwards: no private defaults (the server comes from `AB_BASE`, the key file from `AB_KEY_FILE`), no waiting when the server has no `/queue` endpoint, and a report that adds generation time, totals and a repeatability count. The code that sends the requests and scores the answers is unchanged.

_Prepared with the help of Claude Code. The numbers come from the harness outputs listed above; nothing here is an estimate unless it says so._
