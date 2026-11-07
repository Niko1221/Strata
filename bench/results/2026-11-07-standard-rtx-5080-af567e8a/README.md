# Strata standard benchmark: NVIDIA GeForce RTX 5080

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261107T074427Z-af567e8a`, started 2026-11-07T07:44:27Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 5080 (15.9 GiB dedicated)
- CPU: Intel(R) Core(TM) Ultra 7 265KF; RAM: 62.4 GiB
- Storage of the model files: ssd CT1000MX500SSD1 (sata)
- OS: Arch Linux 7.2.14-arch1-1; drivers: gpu-driver 615.78.08; runtime: cuda 13.4
- Strata commit: `c3b3f5035506f3e966f1b4b7b188aeaabae92f1d`; engine 0.1.57; release build
- Background work: not measured

## Model and configuration

- Model: qwen3.8-flash-next-iq3_s; quantization IQ3_S; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` `4c1eb2ceb4915e11...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` `316b46f3a2dbd68c...`; `expert-profile.bin` `8f59b4aa8873209d...`
- Track: **custom** (no notes given by the operator)
- Context 65536 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 1 min 15 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 124.9 (120.9-126.3) n=3 | 0.42 (0.33-0.43) n=3 | 0.98 (0.92-2.64) n=3 | 3.3 (3.3-4.7) n=3 | 1/3 |
| prompt-4k | 3577.5 (3436.9-3610.9) n=3 | 115.9 (108.8-119.8) n=3 | 1.16 (1.15-1.21) n=3 | 3.60 (3.52-3.68) n=2 | 5.6 (5.4-5.9) n=3 | 3/3 |
| prompt-16k | 4290.1 (4277.2-4325.0) n=3 | 111.5 (106.1-123.0) n=3 | 3.84 (3.81-3.85) n=3 | 7.30 (6.00-8.59) n=2 | 8.4 (8.0-8.7) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 2 | 7 | 0 | 0 |
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
- Excluded from speed summaries: performance-prompt-16k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r1: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).

## Operator notes (custom track and model provenance)

- The configuration is the setup-wizard profile `iq3_s-64k` (see `config.json`), not the stock release defaults, hence `track: custom`. It includes: context 65536, KV cache int8, `--expert-profile expert-profile.bin`, expert-cache and prefill auto, MTP speculation (`--spec 4 --spec-min-p 0.5 --mtp rt`, rt draft pack), pool-workers 13, vision enabled. Sampling and template kwargs are in results.json; the effective server defaults are in config.json.
- Model provenance: the Strata setup wizard downloaded the IQ3_S subset of the Hugging Face repository `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` at the revision pinned by `setup.py` of Strata v0.2.11, `ed59f92082b1e93c0e96d60a8b11aab089b52f09`. The local shards match that revision's catalogue entries (`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf`, 54,817,524,224 bytes, sha256 `4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3`), identical to the hashes in `artifacts`. Base model: `Qwen/Qwen3.8-Flash-Next` (Qwen Community License 1.0); GSQ-RCO quantization by ISTA-DASLab. The null `model.revision` and `tokenizer.revision` values with reason "not pinned (draft)" mean the draft protocol records no pin and the runner cannot read the wizard's HF pin; this section records it.
- Software: release components. Engine binary 0.1.57 (`linux-x86_64-cuda-cuda13`); `strata` CLI v0.2.11 at the tag commit `c3b3f5035506f3e966f1b4b7b188aeaabae92f1d`, which is the `runner_commit` in results.json. `build_options` were not readable; engine.log carries the full launch line.
- Stages: the required warmup, performance and sanity stages ran; the optional concurrency stage was not selected (0 planned). 7 of 9 performance requests ended `truncated` at the 512-token answer cap, as expected for the long thinking cases; they are kept and summarized separately, not dropped. `required_duration_s` (75.6 s) is shorter than the 2–4 minute range the protocol doc reports for the author's older machines; the doc states that range descriptively, and hashing before the run (3 min 27 s) is excluded per protocol.
