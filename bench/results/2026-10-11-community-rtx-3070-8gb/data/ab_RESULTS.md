# Greedy against the thinking sampling preset (for PR #1274)

Started 2026-10-11T10:05:06+09:00. Model `qwen3.8-flash-next-iq2_xs`, engine 0.1.41, GPU NVIDIA GeForce RTX 3070. Rounds with data: [1, 2, 3]. Requests: 114 (0 failed).

Both arms are sent explicitly: greedy = `temperature=0`; thinking = `temperature=1.0, top_p=0.95, top_k=20` (a fixed seed per prompt and round). Same prompt, same `max_tokens`, `reasoning_effort=medium`, streaming, one request at a time; the arm order alternates.

## Paired result (thinking / greedy, same prompt in the same round)

| workload | decode tok/s | generated tokens | generation time | wall time |
| --- | --- | --- | --- | --- |
| all | 0.94 (0.83-1.23), n=57 | 1.08 (0.39-2.63), n=57 | 1.13 (0.42-2.91), n=57 | 1.25 (0.32-2.73), n=57 |
| chat | 0.92 (0.88-1.05), n=6 | 1.20 (0.99-1.57), n=6 | 1.30 (0.95-1.73), n=6 | 1.36 (0.91-1.78), n=6 |
| code | 0.96 (0.92-1.01), n=6 | 1.21 (0.92-1.72), n=6 | 1.23 (0.92-1.81), n=6 | 1.21 (0.98-1.68), n=6 |
| math | 0.93 (0.84-1.13), n=30 | 1.12 (0.39-2.63), n=30 | 1.21 (0.42-2.91), n=30 | 1.20 (0.41-2.73), n=30 |
| tools | 1.01 (0.83-1.23), n=15 | 0.96 (0.63-1.36), n=15 | 1.03 (0.51-1.40), n=15 | 1.61 (0.32-2.00), n=15 |

Generation time = generated tokens / decode tok/s, so it leaves out reading the prompt. Wall time includes the prompt read: the second request of each pair finds the prompt in the engine's cache (`cache_n`) and skips most of it, and with an odd number of pairs (the five tool-call prompts per round) one arm goes first more often. For the short tool-call prompts read decode speed and generation time, not wall time.

## Totals over all requests of an arm

| arm | requests | generated tokens | generation time s | wall time s |
| --- | --- | --- | --- | --- |
| greedy | 57 | 60470 | 1213.3 | 1244.4 |
| thinking | 57 | 63014 | 1363.6 | 1396.2 |

## Per workload and arm (median, range in brackets)

| workload | arm | n | failed | decode tok/s | generated tokens | draft acceptance | wall s | first token s | hit max_tokens | looped | server recovered a loop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| chat | greedy | 6 | 0 | 44.9 (41.8-46.7) | 391.5 (226-508) | 68.2% of 1961 | 9.1 (5.9-11.6) | 0.3 (0.1-0.7) | 0 | 0 | 0 |
| chat | thinking | 6 | 0 | 41.4 (38.8-44.1) | 433.0 (253-770) | 61.0% of 2638 | 10.9 (6.6-18.8) | 0.3 (0.1-0.8) | 0 | 0 | 0 |
| code | greedy | 6 | 0 | 51.4 (49.6-52.6) | 411.0 (251-574) | 84.0% of 2042 | 8.6 (4.9-11.9) | 0.5 (0.1-0.9) | 0 | 0 | 0 |
| code | thinking | 6 | 0 | 50.2 (47.2-51.3) | 417.0 (252-985) | 81.6% of 2588 | 9.2 (6.1-19.9) | 0.5 (0.1-0.9) | 0 | 0 | 0 |
| math | greedy | 30 | 0 | 49.4 (44.6-58.9) | 809.5 (340-6000) | 81.4% of 42551 | 15.8 (7.1-124.9) | 0.4 (0.1-1.2) | 4 | 0 | 0 |
| math | thinking | 30 | 0 | 46.0 (39.7-54.8) | 1259.5 (286-6000) | 72.5% of 43090 | 28.4 (6.2-136.9) | 0.4 (0.1-1.2) | 4 | 0 | 0 |
| tools | greedy | 15 | 0 | 48.6 (45.5-52.7) | 106 (82-129) | 87.2% of 1283 | 2.6 (1.7-4.2) | 0.1 (0.1-1.6) | 0 | 0 | 0 |
| tools | thinking | 15 | 0 | 49.0 (42.6-56.0) | 105 (57-161) | 86.3% of 1284 | 3.1 (1.1-5.0) | 1.5 (0.1-1.6) | 0 | 0 | 0 |
| all | greedy | 57 | 0 | 49.3 (41.8-58.9) | 455 (82-6000) | 81.2% of 47837 | 10.5 (1.7-124.9) | 0.2 (0.1-1.6) | 4 | 0 | 0 |
| all | thinking | 57 | 0 | 47.0 (38.8-56.0) | 660 (57-6000) | 72.7% of 49600 | 13.6 (1.1-136.9) | 0.5 (0.1-1.6) | 4 | 0 | 0 |

## Quality checks

| arm | math correct | tool calls valid | tool task complete | tool task with no call | looped | hit max_tokens |
| --- | --- | --- | --- | --- | --- | --- |
| greedy | 26/30 | 15/15 | 15/15 | 0 | 0/57 | 4/57 |
| thinking | 25/30 | 15/15 | 15/15 | 0 | 0/57 | 4/57 |

## Limits

- Synthetic prompts, few of them, a handful of rounds: read the ranges, not only the medians. At `temperature=0` the final answer text was identical in every round for only 6 of 19 prompts (preset: 4 of 19), so the rounds are separate samples, not exact repeats.
- One server (single RTX 3070 8 GB), one model file and one engine version; draft acceptance comes from the engine's own counters.
- `looped` = the answer hit `max_tokens` and its last 4000 characters are mostly repeats (a compression test); `server recovered a loop` = the server's own repeated-reasoning recovery fired (read from /status after the request, only counts that match the request).
- The quality columns count what a script can check (an exact integer answer, tool-call JSON against its schema); they do not judge writing or code quality.
