# Strata standard benchmark: Intel Arc Pro B70

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261023T054843Z-f329cb7f`, started 2026-10-23T05:48:43Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: Intel Arc Pro B70 (32 GB VRAM per vendor spec; `memory_bytes` and the power limit were not readable from inside the VM)
- CPU: AMD EPYC-Milan Processor (this is the QEMU VM label; the physical CPU is an AMD Ryzen 7 5700G); RAM: 89.3 GiB
- Storage of the model files: hdd QEMU HARDDISK (sata) — a virtual disk; the physical backing store is not visible from inside the guest
- OS: <host> 26.04.1 LTS 7.0.0-38-generic; drivers: not measured (the guest uses the Intel `xe` kernel driver; the userspace Level Zero / SYCL stack was not read by the runner); runtime: not measured
- Strata commit: `011c9e600a93bbc9fd451c563501b5831c32bbca`; engine 0.1.47; source build
- Background work: benchmark driver agent shares the same engine via loopback chat (Hermes agent runs on this model); GPU passthrough in QEMU VM, xe driver, PCIe Gen3 at x16 width (host CPU limited), no clocks pinned

## Model and configuration

- Model: qwen3.8-flash-next-iq3_s; quantization IQ3_S; revision ed59f92082b1e93c0e96d60a8b11aab089b52f09; source revision de4b8e4d43b917e7706784d8bb445c9af86a3540
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` `4c1eb2ceb4915e11...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` `316b46f3a2dbd68c...`; `expert-profile-sycl.bin` `ddddf9b361dd1cb1...`
- Track: **custom** (B70 SYCL AOT bmg-g31, engine 0.1.47 (v0.2.2 source build); setup recommendations plus manual: context 131072, kv int8, kv-resident 32768, ple-io ram, stream-experts, expert-cache auto, spec 4 (spec-min-p 0.5) with MTP draft mtp/rt, vram-reserve 2048 MiB, prefill 4096, vision cpu encoder, idle_unload_s 2700; drop-in STRATA_PCIE_FRAC auto (PcieAuto, measured ~0.34 at x16 Gen3))
- Context 131072 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 12 min 39 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 44.4 (42.0-45.4) n=3 | 0.75 (0.75-106.67) n=3 | 3.25 (2.58-108.82) n=3 | 8.2 (7.7-112.9) n=3 | 0/3 |
| prompt-4k | 1080.9 (1069.1-1081.1) n=3 | 37.9 (37.2-39.9) n=3 | 3.83 (3.83-133.00) n=3 | 75.39 (9.31-141.48) n=2 | 17.3 (16.6-146.7) n=3 | 3/3 |
| prompt-16k | 1117.9 (1114.0-1118.3) n=3 | 34.8 (33.5-37.2) n=3 | 14.70 (14.70-146.53) n=3 | 90.32 (25.05-155.59) n=2 | 30.0 (29.4-160.3) n=3 | 3/3 |

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
- Excluded from speed summaries: performance-prompt-16k-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The engine had been reloaded about one minute before the first request (freshly loaded by the submitter on purpose; the single request answered before the run was the runner's own template probe). Round 1 of each case therefore includes expert-cache and paging warm-up: the TTFT / first-answer / total maxima of round 1 (106 s to 156 s) are warm-up artifacts, not steady-state numbers. Medians and round 2/3 values are unaffected; the same run reproduced yesterday's medians within one percent.
- Model-file provenance: the two SHA-256 hashes of the GGUF shards equal the LFS OIDs published by the Hugging Face repository `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` at revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09` (checked against the HF API on 2026-10-23), so the hashed files are verifiably the repository's files.

## Submitter addendum (manual notes, 2026-10-23)

- The four "not measured / not read" gaps in `missing` (GPU `memory_bytes` and `power_limit_w`, driver and runtime versions) are tooling gaps of the VM guest (no standard sysfs entries for the xe passthrough), not configuration gaps; `status: complete` is therefore honest but readers should keep the four values in mind.
- `expert-profile-sycl.bin` (SHA-256 `ddddf9b361dd1cb1...`) is the expert-ranking profile that ships with the Strata repository for the xe cards (`data/expert-profile-sycl.bin`, selected by `sycl/setup_intel.py`); it is hashed here for completeness. The pack under `packs/iq3_s` was built from the two GGUF shards with the repo tool `tools/iq_pack.py` (schema 1).
- PCIe note for readers comparing rows: the runner only sees the QEMU stub link. This run's engine used its own auto-probe at start: 6.5 GB/s host to device -> `pcie_frac 0.18` (visible in the engine log excerpt). The physical link is PCIe Gen3 at x16 width (the host CPU limits it to Gen3), roughly half of an x16 Gen4 desktop box. The "~0.34" figure in the track note above is a standalone SYCL copy measurement from earlier days, not the value this engine used.

