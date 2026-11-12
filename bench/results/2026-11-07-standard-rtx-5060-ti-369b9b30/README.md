# Strata standard benchmark: NVIDIA GeForce RTX 5060 Ti

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261107T183735Z-369b9b30`, started 2026-11-07T18:37:35Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 5060 Ti (15.9 GiB dedicated, 180 W power limit); PCIe x8, engine link calibration 14.4 GB/s
- CPU: AMD Ryzen 9 7900X 12-Core Processor; RAM: 64 GB (4 x 16 GB DDR5-6000), 61.8 GiB usable by the OS
- Storage of the model files: nvme Corsair MP700 PRO XT (nvme, PCIe 5.0 x4)
- OS: Fedora Linux 43 (Workstation Edition) 7.1.5-100.fc43.x86_64; drivers: gpu-driver 580.173.02; runtime: cuda 13.0
- Strata commit: `146df3c468739147f520ac3816de806f71232571`; engine 0.1.58; source build
- Background work: Enabled timers on the host: systemd --user tmpfiles-clean (00:21), eval-freshness (03:18), data-sync (04:36) - none fired during the run; system egress watchdog every 30 s; Hermes cron interval jobs: dongle watchdog every 1 min, network self-heal every 2 min, CPU guard and two cluster watchdogs every 5 min. All are shell or network checks; no other process held the GPU.

## Model and configuration

- Model: qwen3.8-flash-next-unsloth-ud-iq4_xs-v31; quantization UD-IQ4_XS-v31; revision d96201194f103c906994f283f504209dc8d2a10c; source revision not measured
- Files: `Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf` `93ff23a08e5cd750...`; `Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00004.gguf` `13d5dc8b93d039e4...`; `Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00004.gguf` `f6b9b5934908a204...`; `Qwen3.8-Flash-Next-UD-IQ4_XS-00004-of-00004.gguf` `5de8b5dbf89409ac...`; `expert-profile.bin` `8f59b4aa8873209d...`
- Track: **custom** (engine 0.1.58 native source build (STRATA_ISA_FLOOR empty, -march=native, CUDA 13.1.115, sm_120); pool-workers 11; resident-budget-gib 48; UD-IQ4_XS-v31 4-shard GGUF; expert cache auto; MTP spec 4; kv int8 kv-resident 32768; 262144 context)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 3 min 30 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 56.9 (51.3-62.9) n=3 | 0.90 (0.88-0.91) n=3 | 2.68 (2.04-3.21) n=3 | 8.9 (6.9-9.0) n=3 | 1/3 |
| prompt-4k | 783.2 (782.0-783.3) n=3 | 52.0 (49.8-55.9) n=3 | 5.25 (5.25-5.26) n=3 | 11.71 (10.22-12.87) n=3 | 15.1 (14.4-15.5) n=3 | 3/3 |
| prompt-16k | 1236.7 (1235.6-1240.7) n=3 | 48.3 (45.1-48.8) n=3 | 13.28 (13.24-13.29) n=3 | 23.21 (22.20-24.22) n=2 | 23.8 (23.7-24.6) n=3 | 3/3 |

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
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Expert memory: the requested `--resident-budget-gib 48` was clamped by the engine to 46.84 GiB, because 51.09 GiB of RAM was free at load (4 GiB headroom, 0.25 GiB margin). The expert set is 56.2 GiB, so about 9.4 GiB of it has no RAM copy and is read from the NVMe when those experts are routed: 27,342 blob reads and 148.7 GB read from the drive over the whole run, at a per-request VRAM expert-cache hit rate of 61-81%. These are not all-resident numbers.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine or the repeat-stop and loop-recovery switches of the server, and it cannot read the quantization repository's revision by itself; this run's revision was passed explicitly and is recorded above. See `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
