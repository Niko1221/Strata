# Strata standard benchmark: 2x Tesla V100-SXM2-16GB (SXM2 modules on PCIe carriers)

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261105T153821Z-b1ecbd4f`, started 2026-11-05T15:38:21Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: 2x Tesla V100-SXM2-16GB (16.0 GiB dedicated each, sm_70), SXM2 modules seated on PCIe adapter carriers, linked at Gen3 x16, 300 W power limit each; the engine splits layers across both cards
- CPU: Intel(R) Xeon(R) CPU E5-1660 v3 @ 3.00GHz (no AVX-512); RAM: 121.4 GiB
- Storage of the model files: nvme WDC WDS100T2B0C-00PXH0 (nvme)
- OS: Ubuntu 26.04 LTS 7.0.0-38-generic; drivers: gpu-driver 580.178.04; runtime: cuda 13.0
- Strata commit: `5c8dd322528cc84f68f461ad54a506b9da8fa8d4`; engine 0.1.53; source build
- Background work: openclaw gateway (idle agent session)

## Model and configuration

- Model: qwen3.8-flash-q4_k_xl-ablit; quantization UD-Q4_K_XL; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf` `4448186216b3af4c...`; `Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf` `6f3e21a5ae10c912...`; `Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf` `bb24c1d9c42c129a...`; `Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf` `3d6a9aee4ade5740...`
- Track: **custom** (Huihui-Qwen3.8-Flash-Next-abliterated-v2 UD-Q4_K_XL subquant + custom expert-profile + MTP drafts + q4 KV, 2x V100-16G PCIe, strata v0.2.7.1 CUDA12, Ubuntu 26.04 LTS)
- Context 204800 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 3 min 23 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 56.5 (52.0-57.7) n=3 | 1.76 (1.61-1.80) n=3 | 3.40 (3.05-3.50) n=3 | 7.3 (5.5-7.5) n=3 | 0/3 |
| prompt-4k | 677.6 (677.4-678.4) n=3 | 58.8 (53.2-60.3) n=3 | 6.07 (6.07-6.07) n=3 | 10.63 (9.31-13.78) n=3 | 14.7 (14.2-15.7) n=3 | 2/3 |
| prompt-16k | 1517.6 (1517.2-1518.0) n=3 | 56.8 (54.9-58.6) n=3 | 10.84 (10.84-10.84) n=3 | 14.45 (14.45-14.45) n=1 | 19.8 (19.5-20.1) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

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
- Excluded from speed summaries: performance-prompt-16k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered 1648 request(s) before this run.

## Operator notes (added before submission)

- Form factor: these are SXM2 modules seated on PCIe adapter carriers; nvidia-smi reports the module identity ("Tesla V100-SXM2-16GB"), the physical link is PCIe Gen3 x16 at the 300 W SXM2 power limit.
- Opt-in Volta decode kernels active ([NVIDIA_V100.md](../../NVIDIA_V100.md)): `STRATA_SM70_TABLE=1`, `STRATA_ADAPT_LAG=2`, as in daily service.
- Production serving settings around the run config: KV `q4_0` with flash attention, MTP speculative drafts (`draft-mtp`, n_max 2), context 204800, kv-resident expert budget 12 GiB, 1 slot. Nothing else used the GPUs while the suite ran (an idle agent gateway served as background load).
- Wall time for the whole command, including hashing the four model shards, was about 10 min; the protocol phase itself reports 3 min 23 s after model load. Optional concurrency stage skipped by design.
- Cross-check with our own multi-day hand measurements on this box: decode 52-61 tok/s and cold prefill 1300-1650 tok/s bracket the official figures above. The 4K prompt rate (677.6 tok/s) is dominated by fixed per-request overhead at this size on this platform; the 16K case is the honest prefill number.
- The host is a self-hosted inference gateway; the same card pair serves its own chat traffic day-to-day, so caches were warm rather than freshly loaded (see limitations above).
