# Supplemental notes — Strata 0.2.2 / IQ3_S

Run: `20261022T094550Z-f75ef121`, 2026-10-22. These notes supplement the unmodified official files in this directory; they do not replace the recorded provenance or scores.

## Result and scope

- Official draft protocol: `strata-qwen38-thinking-v1-draft.1`; custom track, thinking enabled.
- Fresh engine, no previous requests; no concurrent local inference clients. Runner started and stopped the engine.
- Status `complete`; 12/12 sanity checks passed (instruction 4/4, retrieval 4/4, tool call 4/4); zero failed requests. Windows boot time was unchanged.
- Required suite took approximately 157 seconds after model loading. Model loading and file hashing are additional time.
- Median short decode: 56.7 tok/s. 4K: prefill 1529.1 tok/s, decode 56.8 tok/s, TTFT 2.72 s. 16K: prefill 1836.2 tok/s, decode 56.0 tok/s, TTFT 8.95 s.
- All six 4K/16K performance requests reached the protocol's 512-token cap (reasoning plus answer). They are performance samples, not completed-answer accuracy checks. Three had no answer delta before the cap; their first-answer latency is unavailable. TTFT includes reasoning deltas.
- This suite is not directly comparable with the previous thinking-disabled benchmark, nor a replacement for long-context or real-agent evaluation. No optional concurrency tests were run.

## Provenance clarification

App 0.2.2, engine 0.1.47, commit `76f90e58436d2f596e76a052277503fd4dcddde1`. Existing official setup-generated IQ3_S configuration was used unchanged: context 131072, INT8 KV, KV resident 32768, pool workers 4, speculative window 4 with MTP min-p 0.5, CPU vision reader, VRAM reserve 700 MiB, automatic expert cache/prefill.

The original results record null upstream revision fields; the official runner nevertheless validates this run as complete. Post-run comparison found both measured GGUF SHA-256 hashes and sizes exactly match the bundled `control/setup.py` IQ3_S manifest. That manifest pins repository `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` to snapshot `ed59f92082b1e93c0e96d60a8b11aab089b52f09`:

- Shard 1: 54817524224 bytes, `4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3`.
- Shard 2: 28800138432 bytes, `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113`.

This establishes consistency with the packaged manifest, not an independently fetched Hugging Face snapshot. Original Qwen source and tokenizer upstream commits remain unverified. Tokenizer/template provenance retained by the runner remains in the original artifacts.

## Runtime metadata discrepancy

The generated README says `cuda 13.4`. The actual engine component's `engine-component.json` identifies build `76f90e58436d+cuda13.0`, and bundled `BUILD.json` also specifies CUDA 13.0. The 13.4 value is driver-supported CUDA, not the engine build runtime. The component alias passed to runtime discovery appears to trigger its driver-version fallback; this explanation is inferred from `bench_facts.py` and the launch path. Raw official files were preserved.

## Stability and sharing limits

The engine logged only 366 MiB free VRAM after loading and warned about low headroom. No failure or reboot occurred in this short run; it does not establish long-session stability. Configuration was not adjusted during this measurement.

