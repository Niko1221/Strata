# Strata standard benchmark: AMD Radeon

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261021T101648Z-0e083d20`, started 2026-10-21T10:16:48Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: AMD Radeon
- CPU: Intel(R) Xeon(R) CPU E5-2676 v4 @ 2.40GHz; RAM: 62.7 GiB
- Storage of the model files: not measured
- OS: <user> 24.04.5 LTS 6.12.111+deb13-amd64; drivers: not measured; runtime: not measured
- Strata commit: `9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4`; engine 0.1.46.3; source build
- Background work: KDE desktop, Firefox and the DeepSeek Harness were running; nothing else

## Model and configuration

- Model: qwen3.8-flash-next-q2_0; quantization Q2_0 (GSQ-RCO, two GGUF shards); revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf` `69820c02ec7d0b45...`; `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf` `316b46f3a2dbd68c...`
- Track: **custom** (Radeon VII 16 GB (gfx906); expert-cache auto; --kv int8 --kv-resident 32768; --spec 4 --spec-min-p 0.5; MTP dir; --pool-workers 6; --vram-reserve-mib 1024; source build of release v0.2.1.4 (commit 9de04cd4), engine 0.1.46.3, no local patches)
- Context 65536 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 5 min 37 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s |
| --- | --- | --- | --- | --- | --- |
| short | 65.1 (65.0-67.9) n=3 | 50.0 (43.9-53.0) n=3 | 1.22 (1.16-1.22) n=3 | 3.18 (2.39-9.91) n=3 | 9.6 (8.9-12.8) n=3 |
| prompt-4k | 484.5 (483.6-485.8) n=3 | 45.7 (44.5-51.0) n=3 | 8.49 (8.47-8.51) n=3 | 15.15 (12.19-18.12) n=2 | 19.6 (17.8-20.0) n=3 |
| prompt-16k | 524.2 (514.2-527.1) n=3 | 47.8 (45.5-49.9) n=3 | 31.32 (31.15-31.93) n=3 | 38.78 (34.89-41.98) n=3 | 42.5 (41.4-42.6) n=3 |

Cells are median (min-max) n=samples.

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
- Excluded from speed summaries: performance-prompt-4k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The runner could not tell whether the server was freshly loaded.

## Additional data

`pool-workers-sweep/` holds ten runs of the community guide's quick suite (`tools/strata_bench.py`) on the same machine
and configuration, made to isolate `--pool-workers`: on this 16-core CPU the engine's default is 15 workers, and 10 or 6
decode 11-23 % faster, with three repeats of the default arm at each context to separate the effect from noise. They are
not standard-protocol results (that suite sends `temperature: 0` with thinking off and has no sanity checks); see
`pool-workers-sweep/README.md` for the tables and the repeat analysis.
