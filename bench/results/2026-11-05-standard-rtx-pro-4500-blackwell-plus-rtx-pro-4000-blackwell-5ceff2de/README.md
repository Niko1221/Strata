This run is the comparison half of the report in the sibling folder ending `-1521c4a1`.

# Strata standard benchmark: NVIDIA RTX PRO 4500 Blackwell, NVIDIA RTX PRO 4000 Blackwell

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261105T191730Z-5ceff2de`, started 2026-11-05T19:17:30Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA RTX PRO 4500 Blackwell (31.9 GiB dedicated); NVIDIA RTX PRO 4000 Blackwell (23.9 GiB dedicated)
- CPU: Intel(R) Xeon(R) CPU E5-2697 v2 @ 2.70GHz; RAM: 251.8 GiB
- Storage of the model files: nvme WD PC SN5000S SDEPNSJ-1T00-1006 (nvme)
- OS: <host> Linux 43 (Workstation Edition) 7.0.14-101.fc43.x86_64; drivers: gpu-driver 580.159.03; runtime: cuda 13.0
- Strata commit: `c3b3f5035506f3e966f1b4b7b188aeaabae92f1d`; engine 0.1.57; source build
- Background work: None: qwen38-nvfp4.service and apex-35b.service were stopped and disabled before the run; no other program used either GPU

## Model and configuration

- Model: qwen3.8-flash-next-unsloth-ud-iq4_xs; quantization UD-IQ4_XS; revision 38bb39ee97821de2c9009abb7e93950eec396e66; source revision not measured
- Files: `Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf` `5ce89370720f8bf9...`; `Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf` `577a38a2392b40ca...`; `Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf` `d4634e6d84f0ebb0...`; `expert-profile.bin` `8f59b4aa8873209d...`
- Track: **custom** (STRATA_ISA_FLOOR=avx source build; 2 GPUs (RTX PRO 4500 32GB + RTX PRO 4000 24GB); layer split left to the engine split search; engine default pool workers; vision on; resident-budget-gib 55; split engine off, it requires identical cards with all routed experts resident)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 1 min 47 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 106.9 (99.2-107.4) n=3 | 0.92 (0.65-0.97) n=3 | 1.64 (1.58-2.54) n=3 | 4.3 (2.9-4.3) n=3 | 0/3 |
| prompt-4k | 1490.4 (1471.1-1514.2) n=3 | 101.7 (90.3-115.0) n=3 | 2.77 (2.73-2.81) n=3 | 4.99 (4.99-4.99) n=1 | 7.7 (7.0-8.5) n=3 | 2/3 |
| prompt-16k | 2572.8 (2566.4-2597.0) n=3 | 108.3 (103.5-119.9) n=3 | 6.41 (6.35-6.43) n=3 | 9.43 (8.43-10.43) n=2 | 11.1 (10.7-11.3) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 4 | 5 | 0 | 0 |
| sanity | 12 | 12 | 0 | 0 | 0 |
| concurrency | 0 | 0 | 0 | 0 | 0 |

Sanity checks: **12 of 12**.

| Case | Result | Reason |
| --- | --- | --- |
| I1 | pass | the expected JSON value |
| I2 | pass | the expected JSON value |
| I3 | pass | the expected JSON value |
| I4 | pass | the exact expected text |
| R1 | pass | the exact expected text |
| R2 | pass | the exact expected text |
| R3 | pass | the exact expected text |
| R4 | pass | the expected JSON value |
| T1 | pass | the expected tool call |
| T2 | pass | the expected tool call |
| T3 | pass | the expected tool call |
| T4 | pass | the expected tool call |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: performance-prompt-16k-r3: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
