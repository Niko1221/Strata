# Strata standard benchmark: NVIDIA RTX PRO 5000 72GB Blackwell

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261108T084141Z-fd907ea2`, started 2026-11-08T08:41:41Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA RTX PRO 5000 72GB Blackwell (71.7 GiB dedicated)
- CPU: AMD Ryzen 7 9800X3D 8-Core Processor; RAM: 89.7 GiB
- Storage of the model files: nvme Samsung SSD 9100 PRO 4TB (nvme)
- OS: Ubuntu 24.04.5 LTS 6.8.0-147-generic; drivers: gpu-driver 595.99.02; runtime: cuda 13.2
- Strata commit: `9780a8c27d84f408240ed4b07062ab7275f02f09`; engine 0.1.58; release build
- Background work: system services only (docker gateway, monitor, metrics); GPU embedder/reranker stopped for the run

## Model and configuration

- Model: qwen3.8-flash-next-unsloth-q8_0; quantization q8_0; revision 084c631ac221016019d6e0a863afe92d90edd3b0; source revision 812d96e10a0dd82b69d749e68693698cf6c36daa
- Files: `Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf` `2dabcbb53ca537a7...`; `Qwen3.8-Flash-Next-Q8_0-00002-of-00006.gguf` `494ca4ed3dbf97bc...`; `Qwen3.8-Flash-Next-Q8_0-00003-of-00006.gguf` `34efd79a80a1ce54...`; `Qwen3.8-Flash-Next-Q8_0-00004-of-00006.gguf` `bfa634025fabbd26...`; `Qwen3.8-Flash-Next-Q8_0-00005-of-00006.gguf` `232a8f14cc0fa426...`; `Qwen3.8-Flash-Next-Q8_0-00006-of-00006.gguf` `538a93bca9180649...`; `expert-profile.bin` `8f59b4aa8873209d...`
- Track: **custom** (engine 0.1.58 release (CUDA 13 component); saved profile unsloth-q8_0-bench from the setup-written run config: 2 slots, 262144 ctx, int8 KV, expert-cache auto, prefill auto, MTP --spec 4 --spec-min-p 0.5, resident-budget 66 GiB (low-RAM: 188 GB Q8_0, part of the experts streamed from NVMe), no vision)
- Context 262144 tokens; server capacity 2 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 2 min 33 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 66.5 (64.0-69.2) n=3 | 0.84 (0.64-0.91) n=3 | 1.96 (1.62-3.53) n=3 | 5.8 (3.7-8.9) n=3 | 1/3 |
| prompt-4k | 1226.8 (1178.8-1232.3) n=3 | 69.9 (60.0-71.4) n=3 | 3.35 (3.34-3.49) n=3 | 7.98 (6.97-9.43) n=3 | 10.5 (10.4-10.8) n=3 | 2/3 |
| prompt-16k | 2400.7 (2370.4-2405.9) n=3 | 70.7 (67.9-72.6) n=3 | 6.85 (6.83-6.94) n=3 | 11.96 (10.51-12.93) n=3 | 14.1 (13.9-14.5) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 3 | 6 | 0 | 0 |
| sanity | 12 | 12 | 0 | 0 | 0 |
| concurrency | 12 | 5 | 7 | 0 | 0 |

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
| 1 | 31.394167 | 1982 | 63.132747 | 0 |
| 2 | 19.916649 | 1472 | 73.908015 | 0 |
| 3 | 28.804516 | 2048 | 71.099962 | 0 |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: concurrency-r1-c1: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).

## Operator notes (machine context)

- The Q8_0 pack is 188 GB and the machine has 89.7 GiB of RAM: with `--resident-budget-gib 66` the resident experts stay in RAM and the rest is streamed from the NVMe (Samsung 9100 PRO 4TB) while answering. These numbers therefore include NVMe expert traffic; they are not an all-in-RAM configuration.
- `model.revision` is the HEAD of `unsloth/Qwen3.8-Flash-Next-GGUF` at run time (2026-11-08); the local files were downloaded around 2026-11-07. File provenance is carried by the SHA-256 hashes above.
- The optional concurrency stage ran on a server with capacity 2: four simultaneous requests queue (two decode, two wait). The round aggregates are wall-time throughput of the round, not four-way parallel decoding.
- Release engine 0.1.58, CUDA 13 component, driver 595.99.02. The machine also hosts a docker gateway and monitor services, idle during the run; the GPU embedding/reranking servers were stopped for the run.
