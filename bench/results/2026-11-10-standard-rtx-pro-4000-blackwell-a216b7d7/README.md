# Strata standard benchmark: NVIDIA RTX PRO 4000 Blackwell

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261110T221039Z-a216b7d7`, started 2026-11-10T22:10:39Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA RTX PRO 4000 Blackwell (23.9 GiB dedicated)
- CPU: AMD Ryzen Threadripper 7970X 32-Cores; RAM: 125.3 GiB
- Storage of the model files: nvme Corsair MP600 PRO NH (nvme)
- OS: Ubuntu 24.04.4 LTS 7.0.0-38-generic; drivers: gpu-driver 595.91.07; runtime: cuda 13.2
- Strata commit: `628f74955a56fcba092495b800dd02af75373a0f`; engine 0.1.59.1; source build
- Background work: none (desktop idle; second GPU runs an unrelated Ollama)

## Model and configuration

- Model: qwen3.8-flash-next; quantization NVFP4 GPTQ (Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata); revision 9b70ad3f640b2546449bd05387407a64113acf8b; source revision not measured
- Files: `qwen-nvfp4-gptq-dense.gguf` `abdf74336b53d8f7...`
- Track: **custom** (NVFP4 GPTQ pack (Maximilian228) on one RTX PRO 4000 24 GB; v0.2.13.1 source build (CUDA 13); 131072 ctx, int8 KV, MTP spec 4; uncapped, 128 GB DDR5-4800 (baseline for the 64 GB run))
- Context 131072 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 1 min 40 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 101.4 (93.8-106.0) n=3 | 0.29 (0.26-0.29) n=3 | 0.96 (0.86-1.39) n=3 | 4.6 (3.5-4.9) n=3 | 0/3 |
| prompt-4k | 2193.3 (2192.8-2230.4) n=3 | 97.3 (85.5-101.1) n=3 | 1.89 (1.86-1.90) n=3 | 4.37 (3.93-4.82) n=2 | 7.1 (6.9-7.8) n=3 | 3/3 |
| prompt-16k | 2908.2 (2905.3-2944.2) n=3 | 93.5 (80.5-94.9) n=3 | 5.66 (5.59-5.67) n=3 | 8.71 (8.22-9.19) n=2 | 11.1 (11.0-12.0) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 3 | 6 | 0 | 0 |
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
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
