# Strata standard benchmark: Intel Arc Pro B70 (32 GB)

*(The runner's original heading named the CPU, for the reason in the correction below; changed by the operator.)*

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261108T034927Z-ae1a5657`, started 2026-11-08T03:49:27Z. **Status: complete.**

**Operator correction — hardware identity.** The runner's probe on this host reported the *CPU* as the GPU and
ROCm 6.2.1 as the runtime. This machine also has an AMD iGPU with a ROCm install, and `strata bench run` identified
that instead of the card the engine ran on. The engine ran on an **Intel Arc Pro B70 (Battlemage G31), 32 GB, PCI
`0000:03:00.0`**, through **Intel oneAPI 2026.2 SYCL**; there is no AMD discrete GPU, and the AMD iGPU drove the
display only. Evidence: the engine's own start-up line in
[provenance/engine-log-excerpt.txt](provenance/engine-log-excerpt.txt) —
`strata generate: GPU 0: Intel(R) Arc(TM) Pro B70 Graphics, compute capability 20.2` — and
[provenance/launch-command.txt](provenance/launch-command.txt). Nothing measured was edited: `results.json` is
byte-identical to the runner's output (SHA-256 `5b38eeca3742c5ba3b9d12307634698f…`) and still contains the wrong
`hardware.gpus[0]`. The correct values are listed in [operator-corrections.json](operator-corrections.json) instead
of being rewritten inside a file the runner wrote. This folder was also renamed from the runner's CPU-derived name to
one that carries the card, per the protocol's `bench/results/<UTC-date>-standard-<gpu>-<run-id>/` convention.

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: AMD Ryzen 7 7700X 8-Core Processor
- CPU: AMD Ryzen 7 7700X 8-Core Processor; RAM: 61.9 GiB
- Storage of the model files: nvme SAMSUNG MZVL4512HBLU-00BL7 (nvme)
- OS: Ubuntu 24.04.5 LTS 7.0.0-42-generic; drivers: gpu-driver 7.0.0-42-generic; runtime: rocm 6.2.1-112
- Strata commit: `146df3c468739147f520ac3816de806f71232571`; engine 0.1.58; source build
- Background work: the machine's desktop only; no other inference client

## Model and configuration

- Model: qwen3.8-flash-next-coder-iq1_m; quantization IQ1_M (Qwen3.8-Flash-Next GSQ-RCO Coder); revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf` `e11083ba855e7666...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf` `316b46f3a2dbd68c...`
- Track: **custom** (pinned v0.2.12 source build (146df3c46), this host's tuned config: --spec 6 --spec-min-p 0.5, prompt cache 6 with prompt-cache-every 4096, --kv int8 --kv-resident 32768, MTP draft layer rt with --mtp-window 8192, --expert-cache auto with --stream-experts, one slot (--batch 0), 131072 context)
- Context 131072 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 2 min 22 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 84.3 (82.5-91.4) n=3 | 0.65 (0.52-0.65) n=3 | 2.02 (1.40-3.22) n=3 | 5.4 (4.9-6.8) n=3 | 1/3 |
| prompt-4k | 1404.4 (1368.4-1426.5) n=3 | 80.8 (64.1-81.0) n=3 | 2.94 (2.89-3.01) n=3 | 6.98 (6.26-7.70) n=2 | 9.3 (9.2-10.9) n=3 | 3/3 |
| prompt-16k | 1773.0 (1752.2-1776.5) n=3 | 68.1 (66.0-70.6) n=3 | 9.27 (9.25-9.38) n=3 | 15.20 (14.01-16.39) n=2 | 16.9 (16.5-17.0) n=3 | 3/3 |

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
- Excluded from speed summaries: performance-prompt-4k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).

## Operator notes

Everything above this section was written by `strata bench run`; this section, the correction block at the top and
the files in `provenance/` are the operator's. Nothing measured was changed.

### What the server was launched with

The run used `--url`, so the runner could not read the engine's options — `config.json` says so, and
`results.json`'s `configuration.kv_cache`, `expert_cache`, `offloading`, `prefill` and `mtp` are `null` for that
reason. The launch command and the sanitized config are attached as
[provenance/launch-command.txt](provenance/launch-command.txt) and
[provenance/engine-config.json](provenance/engine-config.json):

- context `131072`; `--kv int8`; `--kv-resident 32768` of 131,072 cells per layer, K/V in 1.55 GiB of pinned RAM;
- `--prompt-cache 6 --prompt-cache-every 4096`;
- `--spec 6 --spec-min-p 0.5`; MTP draft layer `rt`, `--mtp-window 8192`;
- `--expert-cache auto --stream-experts --prefill auto`, `--vram-reserve-mib 700`, `--batch 0` (one slot);
- the engine probe measured 7.0 GB/s host→device and derived `pcie_frac 0.19`.

`--track custom` records that this is this host's tuned configuration, not a setup-generated default.

### Reading the speed numbers

- **Prompt:** 1,404 tok/s at 4K and 1,773 tok/s at 16K on this card. The 16K case is faster per token than the 4K
  case; the prompt is read by the AOT kernels while the expert cache fills, so the fixed per-request cost is
  amortised over more tokens rather than the rate improving with length.
- **Capped:** 7 of the 9 performance requests reached the 512-token cap while still reasoning (`reasoning_effort
  xhigh` is the protocol's setting, and this model thinks by default). Their totals are time-to-cap, and a capped
  request has no first-answer time — that is the `Capped` column, not a failure. Only the two `short` repetitions
  and one `prompt-4k`/`16k` pair produced an answer inside the cap.
- **Sanity:** 12 of 12, all four retrieval cases at 16K included, all four tool-call cases with exactly one
  structured call. Those run with a 2,048-token cap and did answer.
- **TTFT** is the client's clock from dispatch to the first nonempty delta of any kind, which here is reasoning text.

### Limitations

- **One card, one quantization, one configuration.** No second GPU, no other quant, no concurrency: the optional
  concurrency stage was not run, and the server had one slot (`--batch 0`), so it could not have been.
- **The profile does not describe the model.** `results.json` records `protocol.profile`
  `qwen3.8-flash-next-thinking` because that string is the runner's constant, not because the weights are the
  original Qwen/Qwen3.8-Flash-Next. The model here is a **Coder derivative** (GSQ-RCO IQ1_M); the protocol asks for
  a separate profile for derivatives, and there is no way to declare one from the CLI. Please treat this as a pilot
  on a card the standard list does not have rather than as a conforming entry for that profile.
- **Model identities:** both shard SHA-256 are in `results.json`; the quantization repository revision, the source
  weights revision and the tokenizer revision are `null` because I do not know them, and `protocol
  .model_card_revision` is `null` because the protocol is still a draft and pins none.
- **Hardware facts the runner could not see:** GPU power limit, clocks and VRAM high-water mark were not measured;
  the driver is the in-kernel `xe` module with no separate version.
- **No engine change rides on this report.** The build is the pinned `v0.2.12` tag (`146df3c46`), engine 0.1.58,
  source build, oneAPI 2026.2, AOT target `bmg-g31`.

## Operator artifacts

| File | Contents |
| --- | --- |
| [`operator-corrections.json`](operator-corrections.json) | the fields the runner got wrong, their correct values and the evidence |
| [`provenance/engine-log-excerpt.txt`](provenance/engine-log-excerpt.txt) | the engine's start-up block for this run |
| [`provenance/launch-command.txt`](provenance/launch-command.txt) | the exact server command |
| [`provenance/engine-config.json`](provenance/engine-config.json) | the sanitized launch configuration |
