# Strata standard benchmark: NVIDIA TITAN RTX

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261020T181949Z-015453d3`, started 2026-10-20T18:19:49Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA TITAN RTX (24.0 GiB dedicated)
- CPU: Intel(R) Xeon(R) CPU E5-2696 v4 @ 2.20GHz; RAM: 125.2 GiB
- Storage of the model files: nvme SAMSUNG MZVLW1T0HMLH-000H1 (nvme)
- OS: AlmaLinux 9.8 (Olive Jaguar) 5.14.0-687.51.1.el9_8.x86_64; drivers: gpu-driver 615.71.09; runtime: not measured
- Strata commit: `dea6920b818424c8972ae2c0daf5676bbddde214`; engine 0.1.46.1; release build
- Background work: none. The two GPUs carried only this server: the vision encoder
  (1,266 MiB) and the engine. No other inference client was connected, and the
  machine was otherwise idle during the run.

## Model and configuration

- Model: qwen3.8-flash-next-iq3_s; quantization IQ3_S; revision ed59f92082b1e93c0e96d60a8b11aab089b52f09; source revision de4b8e4d43b917e7706784d8bb445c9af86a3540
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` `4c1eb2ceb4915e11...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` `316b46f3a2dbd68c...`
- Track: **custom** (Strata v0.2.1 (engine 0.1.46.1, commit dea6920b) built from source for sm_75; original Flash-Next IQ3_S; model native 262,144-token window; --kv-resident 65536, --ple-io ram, --spec 4, --vision with an sm_75 encoder built locally, STRATA_STAGE_TRIM=1; server-side reasoning_budget_tokens 12000)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json` (what the runner could read from the server) and
  `server-config.json` (the full run config, API key redacted): the engine ran with
  `--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --max-context 262144
  --kv int8 --kv-resident 65536 --vision --vram-reserve-mib 700 --ple-io ram
  --layer-split auto`, an MTP draft layer, and the server-side
  `reasoning_budget_tokens 12000`. `STRATA_STAGE_TRIM=1` was set in the environment.
  Chat template: `provenance/chat_template.jinja`.

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 2 min 35 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s |
| --- | --- | --- | --- | --- | --- |
| short | 106.1 (104.5-106.9) n=3 | 86.3 (82.2-86.8) n=3 | 0.75 (0.74-0.76) n=3 | 1.99 (1.65-2.00) n=3 | 5.8 (4.4-6.1) n=3 |
| prompt-4k | 955.3 (952.8-962.5) n=3 | 76.4 (76.1-84.2) n=3 | 4.31 (4.28-4.32) n=3 | 8.89 (6.49-9.22) n=3 | 11.0 (10.3-11.0) n=3 |
| prompt-16k | 1474.6 (1473.1-1480.5) n=3 | 79.5 (68.8-81.1) n=3 | 11.15 (11.11-11.16) n=3 | 13.46 (13.15-13.77) n=2 | 17.6 (17.4-18.6) n=3 |

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
- Excluded from speed summaries: performance-prompt-16k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The runner could not tell whether the server was freshly loaded. It was: the
  engine was started for this run alone and had answered no requests before the
  warm-up. `engine.log` in this folder is that start-up and the whole run.

## Submission notes

- **What was measured.** The draft standard suite on this machine: one warm-up
  (excluded), three repetitions each of the short, 4K and 16K prompts, and the
  twelve sanity checks. The optional concurrency stage was not run, so the
  concurrency rows in the phase table are zero by plan, not by omission.
- **Track: `custom`.** The configuration is not what `setup.sh` generates for
  this box: the engine was built from source for `sm_75` (the release ships no
  Linux engine for Turing), the vision encoder was built locally for `sm_75`
  as well, and `--kv-resident 65536`, `--ple-io ram`, `--spec 4` and
  `STRATA_STAGE_TRIM=1` are deliberate choices. `STRATA_STAGE_TRIM=1` (PR #639)
  lets each card load only its own layers' dense weights and returns the freed
  VRAM to the expert cache; on this box it raised the resident expert count from
  8,649 to 9,387 and the prompt throughput at 128K from 1,627 to 1,854 tok/s in a
  separate non-protocol comparison.
- **Two cards, one slot.** The report shows a single GPU because the run was
  against a server with a layer split across the two cards; `nvidia-smi topo -m`
  reports `NV2` between them, but the engine does not use NVLink or peer-to-peer
  access (see `docs/MULTI_GPU.md`), so the second card adds expert cache and
  pipeline depth rather than bandwidth.
- **Reasoning.** The protocol's `xhigh` effort and preserved thinking were in
  force for every request; the server-side `reasoning_budget_tokens 12000`
  applies to the thinking phase, and no request failed because of it.
- **Raw evidence.** `results.json` (machine-readable result), `outputs.jsonl`
  (every request with its raw usage and timings), `engine.log` (the server start
  and the whole run), `server-config.json` (the run config, key redacted),
  `config.json` (the runner's view of the server), `provenance/` (the chat
  template and the corpus manifest).
