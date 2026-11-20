# Strata standard benchmark: NVIDIA GeForce RTX 3090, NVIDIA GeForce RTX 3090

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261120T060959Z-e2f62039`, started 2026-11-20T06:09:59Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 3090 (24.0 GiB dedicated); NVIDIA GeForce RTX 3090 (24.0 GiB dedicated)
- CPU: AMD Ryzen 7 9800X3D 8-Core Processor; RAM: 127.6 GiB
- Storage of the model files: ssd Samsung SSD 990 EVO Plus 4TB (nvme)
- OS: Windows 10.0.26300; drivers: gpu-driver 616.92; runtime: cuda 13.0, cuda-driver-supports 13.4
- Strata commit: `55a557bb6d77facc73d291c3f9894289496c770c`; engine 0.1.62; release build
- Background work: Windows 11 Hyper-V host; Ubuntu Server + Talos VMs stopped for the run; otherwise idle desktop

## Model and configuration

- Model: qwen3.8-flash-next-iq3_s; quantization IQ3_S; revision ed59f92082b1e93c0e96d60a8b11aab089b52f09; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` `4c1eb2ceb4915e11...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` `316b46f3a2dbd68c...`; `expert-profile.bin` `8f59b4aa8873209d...`
- Track: **custom** (custom track: machine-calibrated setup (expert-profile.bin + --prefill-pipe 1536); experimental speed projection ON (--control-vector-scaled ...speed-projection.gguf:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer); vision ON (mmproj BF16); 2x RTX 3090 layer_split auto; --kv int8 --kv-resident 32768; --spec 4 --spec-min-p 0.5 MTP; server --parallel 3; 200K context; release binary 0.1.62-windows-x86_64-cuda-cuda13)
- Context 204800 tokens; server capacity 3 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 2 min 34 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 81.8 (81.3-82.3) n=3 | 0.69 (0.55-0.70) n=3 | 1.77 (1.56-2.05) n=3 | 6.8 (6.5-6.9) n=3 | 2/3 |
| prompt-4k | 982.8 (978.4-984.6) n=3 | 68.7 (68.1-75.3) n=3 | 4.20 (4.20-4.22) n=3 | 9.55 (8.06-9.61) n=3 | 11.7 (11.0-11.7) n=3 | 3/3 |
| prompt-16k | 1959.5 (1948.4-1961.0) n=3 | 69.1 (63.9-69.6) n=3 | 8.41 (8.39-8.44) n=3 | 14.19 (14.19-14.19) n=1 | 15.8 (15.8-16.4) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 1 | 8 | 0 | 0 |
| sanity | 12 | 12 | 0 | 0 | 0 |
| concurrency | 12 | 2 | 10 | 0 | 0 |

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

Concurrency rounds (four requests released together):

| Round | Wall s | Output tokens | Aggregate tok/s | Failed |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 25.469 | 2048 | 80.411481 | 0 |
| 2 | 24.078 | 1966 | 81.6513 | 0 |
| 3 | 23.844 | 2026 | 84.968965 | 0 |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: concurrency-r1-c2: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r1-c3: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r3-c2: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r3-c3: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r2: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).

## Recall extras (separate engine instances)

Two follow-up recall passes on the restarted production server (same launch profile, same model files and context/KV settings; light operator agent traffic; **not** the protocol engine instance, and drafting differs from this run — see the notes):

- engine **0.1.62-windows-x86_64-cuda-cuda13** (same build as this report), DFlash2 drafter, one concurrent slot: `tools/needle_bench.py --lengths 32k,128k --depths 10,50,90` found **6 of 6** needles (actual prompts 27,870 and 106,321–106,323 tokens; 12–48 s each). Evidence: [needle-2026-11-20-2x3090-engine-0.1.62.json](needle-2026-11-20-2x3090-engine-0.1.62.json).
- engine **0.1.59-windows-x86_64-cuda-cuda13** (older build, MTP drafting), taken earlier the same day before the engine update: **6 of 6** found, same command, 16–80 s each. Evidence: [needle-2026-11-20-2x3090.json](needle-2026-11-20-2x3090.json).
