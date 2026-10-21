# Strata standard benchmark: NVIDIA GeForce RTX 3090

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261020T211303Z-ca1ea1e8`, started 2026-10-20T21:13:03Z. **Status: nonconforming.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

This run is not a complete standard result:

- the Strata commit is unknown (not a git checkout and no COMMIT file)

## Hardware and software

- GPU: NVIDIA GeForce RTX 3090 (24.0 GiB dedicated)
- CPU: Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz; RAM: 123.4 GiB
- Storage of the model files: nvme Micron_7300_MTFDHBA960TDF (nvme)
- OS: Ubuntu 24.04.4 LTS 7.0.0-38-generic; drivers: gpu-driver 580.173.02; runtime: cuda 13.0
- Strata commit: `not measured`; engine 0.1.46; release build
- Background work: not measured

## Model and configuration

- Model: swift-1.5-q6_k; quantization Q6_K; revision not measured; source revision not measured
- Files: `Swift-1.5-Qwen3.8-Flash-Next-Q6_K-00001-of-00005.gguf` `da4cedfee7e5298c...`; `Swift-1.5-Qwen3.8-Flash-Next-Q6_K-00002-of-00005.gguf` `8aeb4c8e478809a9...`; `Swift-1.5-Qwen3.8-Flash-Next-Q6_K-00003-of-00005.gguf` `c93bf8b3d818fa22...`; `Swift-1.5-Qwen3.8-Flash-Next-Q6_K-00004-of-00005.gguf` `8dc4f1ed14d0187e...`; `Swift-1.5-Qwen3.8-Flash-Next-Q6_K-00005-of-00005.gguf` `44d0706c626f8bb0...`
- Track: **custom** (Strata 2.1 stack, engine binary from the 2.1 release (0.1.46 per release notes; the server self-reports its version string as 0.1.45, known labeling bug). Launch: ~/bin/strata-2-q6-bmo with strata-swift-q6-final7.json: resident-budget 100 GiB, expert-cache auto + expert profile, spec 4 + MTP, kv int8, conversation-cache 12288 MiB / 2 slots / min 4096 tokens, pipeline-windows 2, vram-reserve 896/1024, prefill auto:16384, spec-min-p 0.70, adapt every 1 / swaps 80 / decay 0.97, pcie-frac 0; env STRATA_MMQ_KQ, STRATA_PREFILL_HELP, STRATA_EXCHANGE_ROTATE, STRATA_PROMPT_ATTN_IMMA, STRATA_ATTN_DAILY, STRATA_QSA_SPLIT_ATTN all 1; pack packs/swift-q6; 2x RTX 3090 layer split)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 3 min 25 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s |
| --- | --- | --- | --- | --- | --- |
| short | 33.2 (25.8-41.2) n=3 | 78.2 (75.4-82.5) n=3 | 2.35 (1.90-3.04) n=3 | 3.51 (2.79-3.86) n=3 | 5.7 (5.2-6.5) n=3 |
| prompt-4k | 500.2 (476.9-500.4) n=3 | 81.5 (77.3-89.6) n=3 | 8.23 (8.23-8.64) n=3 | 11.02 (10.85-12.10) n=3 | 14.5 (13.1-15.2) n=3 |
| prompt-16k | 1316.0 (1163.0-1326.1) n=3 | 84.7 (83.1-93.0) n=3 | 12.52 (12.42-14.17) n=3 | 15.89 (15.06-17.53) n=3 | 18.6 (18.0-20.2) n=3 |

Cells are median (min-max) n=samples.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 4 | 5 | 0 | 0 |
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
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- TThe runner could not independently verify server-load freshness; the operator's notes record that the engine was freshly loaded before this run. the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered 61 request(s) before this run.

## Supplementary long-context recall test

Run 2026-10-21 on the same server, after the benchmark above. This is supplementary evidence; it does not change the
run's nonconforming status and is not part of the standard protocol.

- Model: swift-1.5-q6_k
- Server-reported context capacity: 262,144 tokens (`/health`: status ok, loaded true, max_context 262144, model
  swift-1.5-q6_k, api_key true)
- Test script: `tools/needle_bench.py` (single needle, matched on the exact needle word)
- Test lengths: 32K and 128K
- Test depths: 10%, 50% and 90%
- API authentication: successful (key not disclosed)
- Overall result: **6 of 6 retrieval checks passed**

| Context | Depth | Result | Prompt tokens | Time |
|---|---:|---|---:|---:|
| 32K | 10% | FOUND | 31,722 | 22 s |
| 32K | 50% | FOUND | 31,722 | 20 s |
| 32K | 90% | FOUND | 31,723 | 19 s |
| 128K | 10% | FOUND | 125,056 | 57 s |
| 128K | 50% | FOUND | 125,054 | 52 s |
| 128K | 90% | FOUND | 125,054 | 33 s |

These six checks show successful needle retrieval for these six cases. They are not a comprehensive correctness test
and say nothing general about model quality. Results: [long-context/needles.json](long-context/needles.json), log:
[long-context/needles.txt](long-context/needles.txt).

## GPU inventory note (root cause found 2026-10-21)

The follow-up inventory taken alongside the recall test (`nvidia-smi -L`, 2026-10-21) saw two cards:

- GPU 0: NVIDIA GeForce RTX 3090
- GPU 1: NVIDIA GeForce RTX 3090

Both GPUs were visible, and the serving configuration selects both (`"gpu": [0, 1]` in the run config; the server
passes `CUDA_VISIBLE_DEVICES="0,1"` to the engine, and `/v1/status` and `/metrics` on the running server report
`gpu_count: 2` with both card names — verified live 2026-10-21). The generated results.json above reports one GPU.

The cause is in the benchmark runner, not in the server. The original run used `strata bench run --url`: with `--url`
the runner carries no run config, so the per-config GPU filter is skipped, and the server's combined served-name
string ("…3090 + …3090") matches no single card, so `bench_facts.py:262-266` falls back to the first card only
(identical code in 0.1.46.2 and 0.1.46.3). Running with `--profile <config>` instead keeps both indices: the
follow-up run below records both cards. results.json here is left exactly as generated, and the follow-up inventory
alone does not retroactively prove how this run's engine scheduled the two devices.

## Follow-up run (2026-10-21, shared-load intermediate)

The same machine, same model, same custom track and tuning, run with the fixed command above while agent sessions
shared the server. Kept as evidence that the fixes work under load; for the primary speed numbers see the final
isolated run below. Full report:
[../2026-10-21-standard-rtx-3090-strata-2.1-124948/README.md](../2026-10-21-standard-rtx-3090-strata-2.1-124948/README.md),
run `20261021T120047Z-66e200f7`, **status: complete** (the protocol is still draft; `conformance` is null). The
original run above keeps its nonconforming status; the follow-up is a separate run, not an edit of it.

- Software commit: now populated and verified. `software.strata_commit`, `software.runner_commit` and
  `protocol.git_commit` all say `90767252f420e25223c739d604145ad15911d48b` — tag `v0.2.1`, the release the server
  runs. The run used `STRATA_NO_HANDOVER=1 strata bench run`, whose install tree carries a `COMMIT` file with that
  hash (the file's mtime is 2026-10-20 22:24, after the original 21:13 run — which is why the original report has
  null commits). The runner read it; it was not typed in.
- Model revision: `0142554c963a8653912c0c8283f05f13f63653f3`, established from the local Hugging Face cache
  (the LM Studio trees file named with that revision lists the five Q6_K shards, sizes matching the loaded files).
  `source_revision`, `tokenizer_revision` and `model_card_revision` are not recoverable from this machine and are
  recorded as missing; no value was guessed.
- Hardware: `hardware.gpus` records both RTX 3090 cards (device_index 0 and 1).
- Quality: sanity **12 of 12**; 0 failed, 0 skipped. Truncated (output cap reached, cap includes reasoning): the
  warm-up request and 6 of the 9 performance requests (all repetitions of the long prompts reached their cap);
  3 completed within cap; no sanity request was truncated.
- Speed (median, 3 repetitions): short 22.8 prompt / 80.9 decode tok/s, prompt-4k 517.1 / 79.2,
  prompt-16k 1352.7 / 85.9; TTFT 3.40 / 7.95 / 12.17 s. Strata agent sessions shared the server during the window
  (declared with `--background`), so these numbers carry the same warm-cache caveat as the original.
- Engine instance: the running instance restarted 2026-10-21T11:34:32Z and had served no standard-v1 payload, so
  the protocol's "restart the engine between runs" cache rule is met.
- Terminal: the installed Terminal is update 0.2.1.4 (Control 0.1.46.3, self-update via `strata update`);
  `strata --version` prints the base label "strata 0.2.1 (build 0.1.46.3)". The commit-bearing run path above is
  the 0.2.1 launcher tree (build 0.1.46.2) with its `COMMIT` file. The engine binary is unchanged: 0.1.46.
- The 6/6 long-context recall result above was not rerun and stays as supplementary evidence; it does not prove
  general model correctness, and a custom-track Swift derivative run cannot be a standard-profile conforming result.

## Final isolated run (2026-10-21) — primary performance result

The primary performance result of this PR is the isolated run
[../2026-10-21-standard-rtx-3090-strata-2.1-135001/README.md](../2026-10-21-standard-rtx-3090-strata-2.1-135001/README.md),
run `20261021T130039Z-36b71c5e`, **status: complete** (started 2026-10-21T13:00:39Z). Same machine, model, config
and tuning; `--profile` with the run config (so both GPUs are recorded); `STRATA_NO_HANDOVER=1` runner path (so the
commits are recorded). The operator restarted the server so the engine instance was freshly loaded, closed all agent
sessions and clients, and the preflight saw no GPU compute apps outside the strata engine (`--background` says none).
The protocol is draft and defines no Swift profile, so this is a custom-track standard-v1 workload, not a
standard-profile conformance result; the 2026-10-20 run keeps its measurements and nonconforming status.

- Speed (median, 3 reps, min-max): short 30.9 (30.7-44.3) prompt / 82.2 (81.8-87.4) decode tok/s, TTFT 2.5 s;
  prompt-4k 515.3 (515.0-515.8) / 80.6 (80.5-90.0), TTFT 8.0 s; prompt-16k 1354.6 (1354.0-1356.9) /
  83.0 (82.6-88.9), TTFT 12.1 s.
- Quality: sanity **12 of 12** (instruction 4/4, retrieval 4/4, tool_call 4/4); 0 failed, 0 skipped. Truncated
  (`finish_reason=length`; the output cap includes reasoning): the warm-up request and 6 of the 9 performance
  requests — 7 of 22; no sanity request was truncated. Per protocol this is expected behaviour at the caps, not
  a failure.
- Verified metadata in results.json: all three commit fields =
  `90767252f420e25223c739d604145ad15911d48b` (tag `v0.2.1`); engine 0.1.46 release, binary sha256
  `e54498e95ee5974c5a2b92b430aa867861baf884d217e7761dd68e2d905a1e2e` (BUILD.json source `f-021-rel a7bdbfdabc9a`,
  CUDA 13.0, portable, archs 75/86/89/120); `hardware.gpus` = both RTX 3090 cards at 300 W;
  model revision `0142554c963a8653912c0c8283f05f13f63653f3`, quantization Q6_K, custom pack `packs/swift-q6`.
  Full lineage, pinned upstream links and one traceable post-run amendment (`tokenizer_revision` =
  `0bd4fe22431372cdad1979267d3ab45aa7e6150a`, set from byte-verified vocab/merges/chat-template matches; no
  measurement touched, report revalidated) are in
  [../2026-10-21-standard-rtx-3090-strata-2.1-135001/provenance/model-lineage.md](../2026-10-21-standard-rtx-3090-strata-2.1-135001/provenance/model-lineage.md).
- Unknown and recorded as missing, not guessed: `source_revision` (upstream pins no source-checkpoint commit;
  the weights path has no published hash chain; the Swift repo's single published revision is a strong inference
  — 40 s timeline caveat in the lineage doc), `model_card_revision` (protocol draft, no Swift profile),
  `build_options` (the release publishes signed build identity, not compile flags).
- Correctness evidence in this PR: the standard 12/12 sanity checks in every run plus the 6/6 long-context recall
  test above. Neither is a comprehensive correctness test.
