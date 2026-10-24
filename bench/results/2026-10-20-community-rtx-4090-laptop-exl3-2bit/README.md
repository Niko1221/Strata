# RTX 4090 Laptop: EXL3 2.05 bpw worker comparison

Measured 2026-10-20 by @delirious6423. Seven CPU workers versus ten: fixed-prompt median decode increased **3.3–11.4%**; built-in agent follow-ups increased **11.2%**, but short chat decreased **4.0%**. Prefill changed little. These are workload-specific observations, not a general speedup claim.

**This is a modified EXL3 fork, not upstream Strata.** It uses Q8 dense projections, INT8 KV and MTP—not an all-EXL3 runtime. The EXL3 proposal [#1764](https://github.com/Niko1221/Strata/pull/1764) was closed without merging. This report proposes no engine or default-setting changes.

## Hardware and build

- HP OMEN 17t-ck200; RTX 4090 Laptop, 16,376 MiB VRAM; Core i7-13700HX (8P + 8E, 24 threads); 61.18 GiB usable RAM, 32 GiB swap.
- Linux `7.0.0-38-generic`; NVIDIA driver `595.99.02`; CUDA compiler `13.1.115`, GCC `15.2.0`; Release build for SM89. The suite's CUDA `13.2` field is driver-reported, not the compiler version.
- Storage: ext4 on a 3.7 TB drive; drive model not recorded. PCIe: up to Gen4 x16. Background applications, power limits and sustained clocks were not controlled; no compiler processes were detected during the fixed confirmation runs.
- Source: [sociolog/Strata at fa21db8](https://github.com/sociolog/Strata/tree/fa21db8bb1a9017e875717acb2b7c546f109b941), engine `0.1.45`, plus local compatibility, telemetry and opt-in experimental patches. Both configurations used the **same binary**. Hashes and source patches: [provenance.json](provenance.json), [reproduce/](reproduce/).

## Model and settings

[scorpoon/Qwen3.8-Flash-Next-exl3-SC](https://huggingface.co/scorpoon/Qwen3.8-Flash-Next-exl3-SC/tree/3b646f450626c4143c92ed76bf60bdfa7f74ed75), revision `3b646f4`, `2.05bpw_h8`; routed EXL3 weights unchanged. Both configurations use:

```text
--exl3 MODEL --ple-exl3 MODEL/ngram_embedding.safetensors
--exl3-lane --exl3-lane-profile expert-profile.bin
--exl3-vram-tier 6757 --prefill auto:1024 --max-context 65536
--kv int8 --spec 4 --spec-min-p 0.5 --mtp MTP_RUNTIME
```

`STRATA_EXL3_DENSE=q8`; tail skipping, resident-routing bias and experimental GPU miss-sharing off. Fixed tier: 5,613 resident experts plus 64 adaptation spares; adaptation retained. Prefill resolves to 8,192-token chunks. One serialized client, no vision.

- **Control:** `--pool-workers 10`, default affinity.
- **Selected:** `--pool-workers 7 --pool-affinity p-cores`.

Seven workers already used the P-cores under default affinity in screening; pinning is not a separately established gain. Full configurations: [control](reproduce/control.json), [selected](reproduce/winner.json).

## Fixed-prompt results

[Harness and frozen corpus](reproduce/): direct `StrataEngine.generate` calls, **not HTTP**. Identical token-ID hashes across configurations; temperature 0, thinking off, 128 generated tokens in every measured request, zero reused prompt tokens. One discarded 32-token warm-up per configuration; engine and adaptive expert-cache state retained across its requests. Three repeats per length, control then selected—not an order-randomized experiment. Loading excluded. Throughput uses engine phase timings; TTFT runs from the generate call to its first token.

Values are medians; brackets are minimum–maximum, not confidence intervals. Prefill and decode are tok/s; TTFT is seconds. Arrows mean control → selected.

| Prompt tokens | Prefill | TTFT | Control decode [range] | Selected decode [range] | Decode change |
|---:|---:|---:|---:|---:|---:|
| 512 | 176.24 → 171.63 | 2.95 → 3.02 | 46.01 [45.76–47.95] | 51.12 [47.96–53.86] | +11.1% |
| 2,048 | 623.61 → 610.60 | 3.32 → 3.39 | 60.85 [59.45–62.33] | 67.73 [65.29–69.44] | +11.3% |
| 8,192 | 1410.71 → 1431.62 | 5.84 → 5.75 | 57.78 [53.78–59.05] | 59.71 [59.29–61.10] | +3.3% |
| 32,768 | 1757.77 → 1780.41 | 18.68 → 18.44 | 46.01 [43.03–54.33] | 51.03 [49.01–54.67] | +10.9% |
| 65,392 | 1852.21 → 1880.27 | 35.37 → 34.84 | 50.37 [48.70–55.12] | 56.12 [55.44–59.15] | +11.4% |

All 30 measured requests completed. [Control data](raw/control-confirmation.json) and [selected data](raw/winner-confirmation.json) include every request, all timing ranges, output IDs, draft metrics and two-second memory samples; matching engine logs are in [raw/](raw/).

Both runs peaked at **15,273 MiB whole-GPU use**, with minimum observed free VRAM **673 MiB** (CUDA log: 672 MiB). Minimum available host RAM: **15.35 → 15.23 GiB**. Swap held about **16.75–17.02 GiB**; activity counters are in the samples. These sampled whole-machine values include desktop/background use and reclaimable cache; they do not show a model-memory reduction or exact allocation peaks.

## Built-in HTTP suite

`tools/strata_bench.py --runs 3 --long`, same two configurations; discarded chat/1K warm-ups, fresh tags for uncached prompts. Three short questions × three repeats, 100 outputs each; three six-turn agent sessions, up to 64 outputs per turn, with prefix reuse. The local CLI has a one-line port fix; workloads and timing calculations are unchanged.

| Metric | Control median [range] | Selected median [range] |
|---|---:|---:|
| Short-chat decode tok/s | 30.10 [24.90–42.10] | 28.90 [24.80–42.20] |
| Agent follow-up decode tok/s | 48.00 [34.50–59.40] | 53.40 [39.10–61.00] |
| Agent follow-up TTFT s | 3.72 [3.29–4.03] | 3.71 [3.28–4.02] |

HTTP prefill medians at 4,019 / 16,058 / 32,049 actual tokens were 1012 / 1620 / 1765 → 1006 / 1629 / 1765 tok/s (24 outputs each). Follow-up prefix reuse was about 68.8% → 68.7%. [Control](raw/stock-control.json) and [selected](raw/stock-winner.json) contain per-request timings, token counts and failures; agent continuations differ because they use the model's own replies.

## Quality and limits

Against the **original 2-bit fork runtime, not BF16**, the selected worker profile matched all 4,995 teacher-forced positions: top-1 agreement 100%, approximate reference-top-64 KL 0, PPL 3.9720507. All 20 full-vocabulary prompt-endpoint comparisons also had KL 0. [Quality data](raw/quality-approved-worker.json); teacher-forced TSVs are in [raw/](raw/). The adapter retains the tested KV, GPU placement and dense path.

This does **not** establish identical generation: only **2/15** complete 128-token greedy traces matched control. Repeated control-only 512-token traces matched 3/3. Different continuations change expert workload and draft acceptance, so these are observed prompt-level speeds, not isolated kernel ratios. [Repeatability data](raw/generation-repeatability.json).

Larger-cache and GPU miss-sharing candidates failed the baseline-equivalence gate and were excluded. No task-accuracy, long-context recall, concurrency, other hardware or current-upstream comparison is claimed. Three repeats and uncontrolled background state limit generalization; no statistical significance is asserted. Reproduction inputs and remaining gaps are documented in [reproduce/README.md](reproduce/README.md).
