# Standard Strata benchmark — RTX 3060 Q2_0, 32K configured context, 128 GB RAM

Measured 2026-11-25 UTC. **Complete**. Upstream pilot protocol `strata-qwen38-thinking-v1-draft.1`; custom track. 21/21 required requests finished or capped, 0 failed or skipped; 12/12 sanity checks. Optional concurrency: 12 requests, 0 failed (6 capped). The performance phase had 8/9 capped requests; these are not complete answers.

- Minisforum AI X1-255 / Ryzen 7 255 / 128 GB installed (operator report), 121.3 GiB OS-visible (runner); RTX 3060 12 GiB, 170 W, OCuLink Gen4 x4; Ubuntu 26.04.1, kernel 7.0.0-44-generic, NVIDIA 595.99.02, CUDA 13.2; Strata engine 0.1.63.
- Q2_0, quant-source revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`; both GGUF SHA256 values match the November 21 standard 32K run. Base source and tokenizer revisions are not given in this run; no backfill.
- Configured context 32768; thinking xhigh; drafting OFF as requested, GPU Unlock ON, vision OFF, parallel 1. Raw runtime drafting counters are not proof that all speculative mechanisms are disabled.
- Median 4K prefill 779.9 tok/s; 16K prefill 1382.5 tok/s; 16K decode 47.6 tok/s. Actual upstream prompts only reach ~16K, not 32K.

[Original runner report](runner-report.md) · [results.json](results.json) · [outputs.jsonl](outputs.jsonl) · [config.json](config.json) · [provenance](provenance/) · [start metrics](metrics-start.json) · [GPU telemetry](gpu-telemetry.csv) · [sanitized launch plan](launch-plan-sanitized.txt) · [unlock evidence](gpu-unlock-success.txt) · [SHA256SUMS](SHA256SUMS).

All five artifact hashes referenced by results.json were independently checked against the exact packaged bytes. The original official JSON/JSONL/config/provenance files are unchanged. The generated report was renamed only. The launcher plan was redacted for the local username; this is not a full engine log. Missing runtime fields in the original JSON remain unknown.

[Follow-up comparison and actual long-context runs](../2026-11-25-community-rtx-3060-strata-0.1.63-128gb/) · [Original 64 GB standard run](../2026-11-21-standard-rtx-3060-q2-0-32k/).
