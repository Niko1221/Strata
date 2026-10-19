# Single RX 7900 GRE (gfx1100) IQ2_XS production serving measurements at 262K context

Measured 2026-10-13 → 2026-10-19 by @lsunay on one AMD Radeon RX 7900 GRE (gfx1100, 16 GB), a Ryzen 5 5600X (6 cores / 12 threads, no AVX-512) and 64 GB of RAM, Debian 13. Original ISTA GSQ-RCO Qwen3.8-Flash-Next IQ2_XS, 512-expert MoE, 262,144-token context.

This is a **production serving report, not a controlled benchmark**: the numbers come from the engine's own per-request log over six days of real agent traffic with reasoning enabled. It adds a long-context gfx1100 data point where the 0.2.0 release notes list RX 7000 cards as untested, and confirms two settings that were decided on this machine: `--vram-reserve-mib 832` and `fit_max_tokens`.

1,004 request lines were recorded, 168 of them with `reused = 0` (fully fresh prompt, no cache reuse). Only the fresh lines are aggregated, because a reused prompt measures cache hits rather than ingestion.

## Hardware, source and model

| Item | Recorded environment |
| --- | --- |
| GPU | Radeon RX 7900 GRE, 16 GB, gfx1100 |
| Host | Ryzen 5 5600X, 6 cores / 12 threads, no AVX-512 (expert kernels on AVX-2), 64 GB RAM |
| OS | Debian 13, kernel 6.12.111+deb13-amd64 |
| Engine | Strata 0.1.43.1, source build, commit `a0116b2`, HIP/gfx1100 |
| hipBLASLt | gfx1100 tuning table `100200`, 26 rows, active |
| Model | GSQ-RCO Qwen3.8-Flash-Next IQ2_XS, native + PLE shards (64 GB), MTP draft `rt` (6.5 GB) |
| Hugepages | `vm.nr_hugepages = 0` — the host expert arena runs on 4 KB pages |
| PCIe | startup probe 27.8 GB/s host→device, `pcie_frac 0.55` (default) |

Exact engine config from the startup log:

```
--expert-profile data/expert-profile.bin --expert-cache auto --prefill auto
--spec 4 --spec-min-p 0.5 --mtp Strata-data/mtp/rt
--vram-reserve-mib 832 --max-context 262144 --kv int8 --kv-resident 32768
```

Expert cache: 7,310 → 7,654 slots (9.79 – 10.25 GiB VRAM) across restarts. KV is int8 with 32,768 tokens resident in VRAM and the rest streamed from RAM (~3.09 GiB pinned). Model and pack SHA-256 sums are in [model-manifest.json](model-manifest.json).

## Method

The engine logs one line per request: prompt tokens split into reused and read, the read time, and the generated tokens with their time. Reasoning was on, so `generated` includes thinking tokens and output lengths vary per request. There is no fixed greedy arm, no needle test and no time-to-first-token (the serve path logs per-request totals, not TTFT).

Median over the 168 fresh-prompt lines, bucketed by prompt size:

| Prompt size | n | Prompt tok/s (median) | Decode tok/s (median) |
| --- | ---: | ---: | ---: |
| < 1,000 tokens | 61 | 198.2 | 52.1 |
| 1,000 – 8,000 tokens | 25 | 777.8 | 53.5 |
| ≥ 8,000 tokens | 82 | 965.9 | 52.5 |

Long-context fresh prompts, the ten largest:

| Prompt tokens | Prompt ms | Prompt tok/s | Generated | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 114,495 | 121,377 | 943.3 | 390 | 54.8 |
| 113,920 | 123,326 | 923.7 | 412 | 63.2 |
| 103,134 | 105,930 | 973.6 | 819 | 57.1 |
| 98,893 | 101,849 | 971.0 | 999 | 49.6 |
| 98,502 | 104,728 | 940.5 | 174 | 53.2 |
| 94,708 | 95,768 | 988.9 | 941 | 49.5 |
| 92,505 | 93,880 | 985.4 | 962 | 48.1 |
| 92,427 | 99,179 | 931.9 | 155 | 51.8 |
| 80,388 | 81,758 | 983.2 | 810 | 59.7 |
| 76,483 | 78,138 | 978.8 | 662 | 52.1 |

Per-request data for all 168 fresh lines: [fresh-prompt-requests.csv](fresh-prompt-requests.csv). Computed aggregates: [aggregates.json](aggregates.json). The sanitized engine log (private paths removed, no credentials): [engine-log-sanitized.log](engine-log-sanitized.log).

## What this adds

1. **262K context on a 16 GB gfx1100 card holds up in sustained production use.** The 0.2.0 release notes say no RX 7000 card has run the 0.2.0 builds; the earlier RX 7900 GRE report ([#740](https://github.com/Niko1221/Strata/issues/740)) measured at 65,536 context. Here prefill stays ~950–990 tok/s up to 114K fresh prompt tokens on the same card family.
2. **`--vram-reserve-mib 832` prevented decode crashes.** Before it was added on 2026-10-13, decode crashed when VRAM filled at long context; after adding it there was no crash in six days of continuous serving.
3. **`fit_max_tokens` removed the 400 "request exceeds the context window"** error that appeared when a client sent `max_output_tokens: 32768` at long context.

## Limitations

- One machine, one card, no cross-engine or cross-GPU baseline.
- Production traffic, not a controlled bench: reasoning on, variable output length, no greedy arm, no recall check, no TTFT. Decode medians include thinking tokens, so they are not comparable to #740's greedy 128-token arm (IQ2_XS there: 1,154.7 prompt / 65.3 decode tok/s at 65K).
- The ROCm runtime version is not printed in the engine log and could not be confirmed on this host; only the hipBLASLt tuning table version (100200) is recorded.
- The 75,523-token row (564.6 tok/s prompt, 29.3 tok/s decode) is a long reasoning request, not a prefill regression — reported as observed.
- The desktop shares the measured GPU. RAM speed and GPU power cap were not recorded.

Reproduction status: all numbers come from the engine log produced on this machine. The log is a snapshot taken on 2026-10-19; the engine was still serving at the time, so the file may have grown after the snapshot. Model and pack hashes are the values on this machine on the same date.
