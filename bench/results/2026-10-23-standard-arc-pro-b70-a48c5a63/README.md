# Strata standard benchmark: Intel Arc Pro B70

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261023T123010Z-a48c5a63`, started 2026-10-23T12:30:10Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: Intel Arc Pro B70 (32 GB VRAM; the guest does not expose memory/driver to the probe)
- CPU: reported "AMD EPYC-Milan Processor" is a QEMU/Fedora VM label; physical CPU is an AMD Ryzen 7 5700G (4 physical cores × SMT; VM sees 8 vCPU); RAM: 89.3 GiB
- Storage of the model files: HDD behind a QEMU SATA disk (libvirt VM; SMART not available in the guest)
- OS: <host> 26.04.1 LTS 7.0.0-38-generic; drivers: not measured; runtime: not measured
- Strata commit: `5c8dd322528cc84f68f461ad54a506b9da8fa8d4`; engine 0.1.48; source build
- Background work: no parallel load expected: submitter ends own agent turn during the run (engine is the submitter's LLM endpoint)

## Model and configuration

- Model: qwen3.8-flash-next-iq3_s; quantization IQ3_S; revision ed59f92082b1e93c0e96d60a8b11aab089b52f09; source revision de4b8e4d43b917e7706784d8bb445c9af86a3540
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` `4c1eb2ceb4915e11...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` `316b46f3a2dbd68c...`; `expert-profile-sycl.bin` `ddddf9b361dd1cb1...`
- Track: **custom** (engine 0.1.48 (v0.2.3+max_tokens-hotfix, commit 5c8dd322), AOT bmg-g31; CPU-experts ENABLED (drop-in STRATA_VERIFY_NO_HOST=0 + STRATA_VERIFY_DEVICE_PLAN=0); pcieauto auto (engine probe 12.8 GB/s host->device -> pcie_frac 0.35; physical PCIe x16 Gen3, host CPU limit); ctx 131072, kv int8 resident 32768, stream-experts, spec auto (mtp_max 4, lookup 3), idle_unload 2700s; VM 4 physical cores x HT; GPU Intel Arc Pro B70 32GB, xe driver)
- Context 131072 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 4 min 23 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 50.4 (39.1-53.3) n=3 | 0.90 (0.85-0.94) n=3 | 3.28 (2.84-3.84) n=3 | 9.8 (8.8-11.1) n=3 | 1/3 |
| prompt-4k | 981.7 (865.7-984.0) n=3 | 47.8 (40.0-47.9) n=3 | 4.20 (4.19-4.76) n=3 | 15.23 (15.23-15.23) n=1 | 14.9 (14.9-17.5) n=3 | 3/3 |
| prompt-16k | 1057.8 (1042.0-1064.4) n=3 | 50.9 (50.4-51.5) n=3 | 15.53 (15.43-15.76) n=3 | 23.44 (23.44-23.44) n=1 | 25.6 (25.3-25.9) n=3 | 3/3 |

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
- Excluded from speed summaries: performance-prompt-16k-r3: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered 1 request(s) before this run.

## Submitter addendum (context the probes could not read)

- **Setup diff vs. the earlier same-day run** (`20261023T054843Z-f329cb7f`, engine 0.1.47, GPU-only experts, submitted as PR #3799): this run uses engine **0.1.48** (commit `5c8dd322`, v0.2.3 + max_tokens hotfix) with **CPU experts enabled** (`STRATA_VERIFY_NO_HOST=0` + `STRATA_VERIFY_DEVICE_PLAN=0` via systemd drop-in). The host-arena path passed the submitter's 16k correctness gate (2× fresh 67k-char prompt, UUID echoed, no token-0 output) — on 0.1.44/0.1.45 the same switches produced token-0 garbage at ≥16k, see the run log. Visible effect: decode +26 % at 4k / +46 % at 16k, prompt rate −5…9 % (the three pool workers share the 4 physical cores with the host thread).
- **PCIe**: the guest cannot see the physical link (QEMU stub). Host side: Gen4 card in a Gen3 CPU slot, x16 electrical (Gen3 ×16 ceiling on a Ryzen 7 5700G host). The engine's own probe in this run: `PCIe probe: 12.8 GB/s host->device (best of 11.1 12.8 12.7 10.2) -> pcie_frac 0.35 (default 0.55)` — autotuned (`STRATA_PCIE_BANDWIDTH=auto`).
- **Warm-up artifact in round 1**: the expert/copy pools warm across repetitions — the min values of the short (39.1) and 4k (40.0 decode) ranges come from round 1; rounds 2–3 are homogeneous. Medians are the comparable figures.
- **Not measured** (probe blind spots in a VM, see `missing`): GPU driver version, GPU memory/power, CPU model string (QEMU label), storage SMART, model-card revision (protocol draft), `oq`/`mmr`/`mocks` not applicable on SYCL/XPU.
- Full engine log excerpt for the run window (reload at 12:28:24Z through suite end): [engine-log-excerpt.txt](engine-log-excerpt.txt).
