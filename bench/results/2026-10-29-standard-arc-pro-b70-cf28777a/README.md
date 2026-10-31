# Strata standard benchmark: Intel Arc Pro B70

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261029T135209Z-cf28777a`, started 2026-10-29T13:52:09Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: Intel Arc Pro B70 (32 GB; PCIe passthrough to a QEMU/KVM VM, physical slot electrically x16 at Gen3 - host CPU platform limit; Strata runtime measured the effective link at 12.7 GB/s on 2026-10-28)
- CPU: AMD EPYC-Milan Processor; RAM: 94.3 GiB (the CPU string is a QEMU/KVM CPUID artifact; the physical CPU is an AMD Ryzen 7 5700G, 8C/16T, the VM is pinned to 10 vCPU)
- Storage of the model files: hdd QEMU HARDDISK (sata)
- OS: <host> 26.04.1 LTS 7.0.0-42-generic; drivers: not measured; runtime: not measured (runner cannot read Intel stacks; submitter: guest kernel 7.0.0-42 with the in-kernel Intel xe driver, Intel compute-runtime 26.35.39758.10-0 and level-zero loader 1.32.0 inside the SYCL container, oneAPI 2026.1, engine AOT target bmg-g31)
- Strata commit: `5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d`; engine 0.1.52; source build
- Background work: Hermes-Agent laeuft auf derselben Engine, waehrend des Laufs bewusst idle (kein Submitting-Traffic); QEMU/KVM-VM auf Ryzen 7 5700G, B70 per passthrough (x16 Gen3)

## Model and configuration

- Model: qwen3.8-flash-next-unsloth-ud-iq4_xs; quantization UD-IQ4_XS; revision 38bb39ee97821de2c9009abb7e93950eec396e66; source revision de4b8e4d43b917e7706784d8bb445c9af86a3540
- Files: `Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf` `5ce89370720f8bf9...`; `Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf` `577a38a2392b40ca...`; `Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf` `d4634e6d84f0ebb0...`; `expert-profile-sycl.bin` `ddddf9b361dd1cb1...`
- Track: **custom** (engine 0.1.52 (Strata v0.2.7, AOT bmg-g31); UD-IQ4_XS; ctx 262144; NO_HOST=0; PcieAuto (pcie_frac 0.25); kv int8; kv-resident 32768; spec 4 + MTP rt; vram-reserve 2048; prefill 8192; stream-experts; vision on)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 10 min 15 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 40.2 (37.4-40.3) n=3 | 1.45 (1.38-168.01) n=3 | 4.11 (3.60-170.38) n=3 | 8.1 (7.9-178.8) n=3 | 0/3 |
| prompt-4k | 841.9 (836.8-842.7) n=3 | 46.6 (37.3-47.5) n=3 | 4.90 (4.89-146.76) n=3 | 9.68 (9.68-9.68) n=1 | 15.8 (15.7-160.5) n=3 | 3/3 |
| prompt-16k | 1186.4 (1182.9-1187.7) n=3 | 44.1 (40.8-46.4) n=3 | 13.86 (13.84-13.91) n=3 | 20.08 (19.19-20.98) n=2 | 25.4 (24.9-26.4) n=3 | 3/3 |

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
- Excluded from speed summaries: performance-prompt-16k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
- Because of that fresh load, the first round of each case absorbs the expert-cache filling: the short and 4k maxima of TTFT/first-answer/total (up to 168 s) come from this filling, not from steady state; read the medians and rounds 2/3 for the steady state.
- The submitter's agent on the same engine stayed idle during the run; the only background traffic was the runner itself.
- `expert-profile-sycl.bin` is the repository's shipped xe expert ranking (selected by `sycl/setup_intel.py`), not a submitter artifact.
- Provenance: the SHA-256 of all three GGUF shards equals the HuggingFace LFS-OIDs of `unsloth/Qwen3.8-Flash-Next-GGUF` (`UD-IQ4_XS/` directory, verified against the tree API on 2026-10-29).
