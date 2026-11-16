# Strata standard benchmark: NVIDIA GeForce RTX 4060 Ti, NVIDIA GeForce RTX 4060 Ti

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261116T101335Z-adea42ae`, started 2026-11-16T10:13:35Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 4060 Ti; NVIDIA GeForce RTX 4060 Ti
- CPU: Intel(R) Xeon(R) W-2145 CPU @ 3.70GHz; RAM: 110.0 GiB
- Storage of the model files: hdd Virtual Disk (unknown)
- OS: Ubuntu 26.04.1 LTS 6.18.58.1-microsoft-standard-WSL2; drivers: not measured; runtime: not measured
- Strata commit: `628f74955a56fcba092495b800dd02af75373a0f`; engine 0.1.59; release build
- Background work: not measured

## Model and configuration

- Model: qwen3.8-flash-next-q2_0; quantization Q2_0; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf` `69820c02ec7d0b45...`; `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf` `316b46f3a2dbd68c...`
- Track: **default** (re-run of #5817 baseline (2026-11-11, app 0.2.13.1 / engine 0.1.59) on the same box; warm server; same protocol and sampling)
- Context 262144 tokens; server capacity 2 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 3 min 21 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 69.1 (65.5-71.6) n=3 | 1.25 (1.23-1.37) n=3 | 2.40 (2.39-2.41) n=3 | 6.3 (5.7-6.6) n=3 | 0/3 |
| prompt-4k | 692.7 (674.5-698.8) n=3 | 62.0 (57.2-65.2) n=3 | 5.94 (5.89-6.10) n=3 | 11.50 (11.34-13.86) n=3 | 14.1 (13.9-14.9) n=3 | 3/3 |
| prompt-16k | 1233.2 (1226.8-1238.5) n=3 | 57.2 (55.2-65.7) n=3 | 13.32 (13.27-13.40) n=3 | 20.68 (19.73-21.64) n=2 | 22.2 (21.2-22.6) n=3 | 3/3 |

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
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered 45 request(s) before this run.

## Settings in effect during this run

`strata bench run` could not read them from a launch profile — the server runs from a hand-written systemd unit
(`/etc/systemd/system/strata-q20.service` plus a drop-in for the queue valves), so `configuration.kv_cache`,
`expert_cache`, `prefill` and `mtp` are `null` in `results.json`. The values below were read from the engine's
`/metrics` endpoint and from the unit on the same machine, minutes after the run.

| Setting | Value |
| --- | --- |
| drafting depth (`--spec`) | **8** |
| expert cache | **15,056 MiB / 11,421 slots** (213 MiB VRAM free at idle) |
| `--parallel` (server capacity) | 2 |
| VRAM reserve (`--vram-reserve-mib`) | 256 |
| KV cache | q4_0 |
| Context | 262,144 tokens |
| Queue valves (drop-in) | `STRATA_ADMIT_SHORT_FIRST=1500`, `STRATA_ADMIT_MAX_WAIT_S=5`, `STRATA_COALESCE_PREFILL=3`, `STRATA_COALESCE_MIN_REQS=3` |
| Engine | 0.1.59 release (cuda13), started 2026-11-16T06:35:35+03, warm — 45 requests served before the run |
| GPU driver | 580.97 · 2x RTX 4060 Ti 16 GB (sm_89, no NVLink) |

## Comparison with the 2026-11-11 baseline in this PR

Identical protocol, identical model files (sha256 match), identical engine 0.1.59 and Strata commit `628f749`;
sanity 12/12 in both runs. The difference between the two runs is machine configuration, not code: the baseline
ran with drafting depth 6 and a smaller expert cache (14,758 MiB / 11,195 slots), this run with depth 8 and
15,056 MiB (the VRAM-reserve change made on 2026-11-15).

| Case | Metric | 2026-11-11 | 2026-11-16 | change |
| --- | --- | --- | --- | --- |
| short | prefill tok/s | 58.5 | 62.7 | +7% |
| short | decode tok/s | 44.1 | 69.1 | +56% |
| short | TTFT s | 1.3 | 1.3 | -7% |
| short | total s | 9.9 | 6.3 | -36% |
| prompt-4k | prefill tok/s | 647.6 | 692.7 | +7% |
| prompt-4k | decode tok/s | 44.7 | 62.0 | +39% |
| prompt-4k | TTFT s | 6.4 | 5.9 | -6% |
| prompt-4k | total s | 17.8 | 14.1 | -21% |
| prompt-16k | prefill tok/s | 1,126.5 | 1,233.2 | +9% |
| prompt-16k | decode tok/s | 47.8 | 57.2 | +20% |
| prompt-16k | TTFT s | 14.6 | 13.3 | -9% |
| prompt-16k | total s | 25.4 | 22.2 | -13% |

Medians, n=3 per case in both runs.

## Variance (second run, same day)

`variance-run2-results.json` — identical second run (run `20261116T102531Z-6d74a693`), nine minutes later, sanity 12/12.
Decode: short 68.9, prompt-4k 57.7,
prompt-16k 64.5 tok/s. Between-run spread on 4k/16k is about ±7%, well below the
gap to the baseline.

## Limits

Same protocol caveats as the baseline: 6 of 9 performance requests are truncated at the 512-token cap (thinking at
`reasoning_effort=xhigh` fills it before the answer starts); the engine was already serving (`started_engine=false`),
so there is no cold-load run; the hardware probe reported both GPUs this time (the baseline reported one — a probe
artefact in WSL2) and `software.drivers` / `software.runtime` stay `null` because the server is not managed by a
launch profile.

Evidence: `results.json`, `outputs.jsonl`, `runner-README.md` (raw runner output), `config.json`,
`chat_template.jinja`, `corpus-build.json`, `corpus-manifest.sha256`, `engine-log.txt` (server journal for the run
window; no credentials), `variance-run2-results.json`, `ARTIFACTS.sha256`.
