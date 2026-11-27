# Notes on this run (added by the submitter)

This folder is one half of a two-run engine A/B on the same machine, the same
model and the same configuration. The partner folder is
[`2026-11-20-standard-rtx-5090-laptop-gpu-ef253641`](../2026-11-20-standard-rtx-5090-laptop-gpu-ef253641/README.md).

| run id | engine | control | what else differs |
|---|---|---|---|
| `20261120T195352Z-8eb18178` (this folder) | 0.1.62 | 0.1.63 (`12d491da4a6904eb49aaa6f2f63ec4b1a17582e3`) | nothing |
| `20261120T194747Z-ef253641` | 0.1.63 | 0.1.63 | nothing |

The two run configs (`config-0162.json`, `config-0163.json` in this folder)
differ only in the engine directory, the log path and the label — the model
files, the pack, the expert profile, the KV and prefill settings, the MTP
drafter and the sampling parameters are identical. Both runs are the same
`strata bench run` standard-v1 protocol on the same host.

## Why the run is `nonconforming` (as the runner recorded it)

Two independent reasons, both stated by the runner and not worked around:

1. **No model file could be hashed.** The run was started with
   `--skip-model-hash`, because the model files are a locally built pack
   outside the install tree. The file hashes were verified separately in the
   preflight of the same session; the runner simply had no file it was allowed
   to hash, so it marked the run `nonconforming` rather than silently
   reporting a hash it never computed.
2. **6 of the 9 performance requests were truncated.** The protocol keeps the
   model card's sampling, which on this model means `reasoning_effort: xhigh`.
   With thinking on and no explicit reasoning budget, six requests spent the
   whole 2048-token cap inside the thinking phase and never reached an answer.
   The quality section was unaffected: **12 of 12 sanity checks passed**
   (instruction 4/4, retrieval 4/4, tool call 4/4).

Neither reason is an engine failure: no request failed, and every reported
metric comes from the official runner.

## Reading the two runs

The medians of the two runs land within each other's spread on every case, so
the two engines are indistinguishable at this sample size:

| case | 0.1.62 | 0.1.63 |
|---|---|---|
| short decode tok/s | 31.0 (29.7–33.1) | 29.6 (25.2–34.1) |
| prompt-4k prompt tok/s | 1641.7 | 1630.4 |
| prompt-4k decode tok/s | 35.8 (33.9–40.3) | 37.0 (33.7–37.3) |
| prompt-16k prompt tok/s | 2617.4 | 2765.6 |
| prompt-16k decode tok/s | 36.8 (32.4–37.0) | 33.7 (32.6–35.3) |
| sanity checks | 12 / 12 | 12 / 12 |

The `official-bench` folder of the same session holds the two other official
suites (`serving-smoke`, `parallel-mixed-v1`) for both engines; both report
`ok: true`.

## Note on the reasoning budget

The protocol asks for the model card's defaults, so the run configs drop the
live arm's `reasoning_budget_tokens`. That is what leaves thinking unbounded
inside the 2048-token output cap and truncates those six requests. A protocol
revision that either fixes the cap or reports thinking and answering rates
separately would make these runs `complete` instead of `nonconforming`.

## Provenance

- Model: `qwen3.8-flash-next-unsloth-ud-q4_k_xl` (UD-Q4_K_XL), a locally built
  pack; the GGUF shards are not redistributed here.
- Prompt corpus: `bench/prompts/standard-v1`, corpus hash `e75eff8b9feea04f…`
  (recorded in `provenance/corpus-build.json`).
- Per-request records: `outputs.jsonl`; machine-readable result:
  `results.json`.
