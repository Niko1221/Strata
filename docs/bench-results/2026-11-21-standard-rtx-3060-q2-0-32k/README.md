# Standard Strata benchmark — RTX 3060 Q2_0, 32K configured context

Measured **2026-11-21 UTC**. Result status **complete**; `strata-qwen38-thinking-v1-draft.1` is the exact upstream protocol identifier. This is a completed measured submission, **not a Draft pull request**. The protocol itself was in the upstream draft phase.

- Hardware: NVIDIA GeForce RTX 3060 12 GiB, 170 W power cap; AMD Ryzen 7 255, ~60 GiB usable RAM; Minisforum DEG1 OCuLink PCIe Gen4 x4.
- Ubuntu 26.04.1, kernel 7.0.0-43-generic, NVIDIA 595.99.02, CUDA 13.2; Strata engine 0.1.63 CUDA.
- Model: Qwen3.8-Flash-Next Q2_0, source GGUF revision recorded in `results.json` as `ed59f92082b1e93c0e96d60a8b11aab089b52f09`. GGUF file hashes and runner commit are preserved unchanged in the original result.
- Model source revision: `de4b8e4d43b917e7706784d8bb445c9af86a3540`. Do not infer a missing value from another run.
- Context configured: **32,768**; vision off; parallel 1; drafting OFF; GPU Unlock ON.
- Thinking ON (`reasoning_effort=xhigh`), Qwen model-card sampling. Complete official 21 required requests, zero failed or skipped; 12/12 sanity checks passed; optional concurrency executed.
- 4K median prefill 780.3 tok/s; 16K median prefill 1378.0 tok/s; 16K median decode 46.8 tok/s.
- Native standard prompts reach only approximately 16K input even when context is configured to 262K. See the separate [actual long-context results](../2026-11-21-community-rtx-3060-strata-0.1.63-260k/).

## Exact original evidence

- [results.json](results.json) — original output, **unmodified**; every listed artifact SHA256 was verified against these files.
- [outputs.jsonl](outputs.jsonl) — full per-request standard evidence, **unmodified**.
- [config.json](config.json), [template](provenance/chat_template.jinja), [corpus build](provenance/corpus-build.json), [manifest SHA256](provenance/corpus-manifest.sha256).
- [Runner report](runner-report.md), [start-of-run engine metrics](metrics-start.json), [sampled GPU telemetry](gpu-telemetry.csv), [sanitized launch output](launch-output-sanitized.txt).
- [SHA256SUMS](SHA256SUMS) checks the exact committed bytes in this folder.

## Limitations and provenance

This was a user-launched Strata server (`strata bench run --url ...`), so the standard runner did not automatically capture all runtime build/offload/cache options, full engine runtime logs, or tokenizer revision; the fields remain unknown where the original JSON says unknown. Metrics from the server are supporting observations, not substitutes for missing provenance. The launch output is **not** a full engine log. Output caps include thinking and are reported as capped, not quality failures. The `off` substring in the private run name denotes **drafting off**, not thinking or GPU Unlock.

- Original run ID: `20261121T005301Z-2660aa1d`.
- Source archive run directory: `20261121T005133Z-32k-off-unlock-on` (archive not uploaded publicly).
- Model quant and all per-case details must be read from the original `results.json`, not reconstructed from this README.
