# Community-format benchmark: RTX 4080 SUPER, Ryzen 9 7950X3D, Swift IQ3_XXS at 256K

Measured on 2026-11-18 by XX (Eich) with `tools/strata_bench.py` against a running user-started
instance (thinking off, temperature 0), three full runs per model. Reproduces the community method
(`docs/COMMUNITY_BENCHMARKS.md`) on engine **0.1.60**. Main limitation: TTFT is the client's time to
the first token on a FIFO server, so it includes queue time.

## Hardware and software

- NVIDIA GeForce RTX 4080 SUPER, 16 GiB VRAM (power limit 320 W); driver 621.14; Windows 11 (WDDM).
- AMD Ryzen 9 7950X3D (16 cores / 32 threads), 63.7 GiB RAM.
- Engine 0.1.60 (CUDA 13, sm_89; `BUILD.json` of the verified v0.2.14 package, SHA256 + signed
  manifest checked on the 5 shipped files); serve wrapper reports project 0.1.61. 0.1.59 kept as the
  rollback basis.

## Model and configuration

swift-1.5-iq3_xxs, **Swift IQ3_XXS** (256K context), pack `Strata-data/packs/swift-iq3_xxs`, MTP
`Strata-data/mtp/rt`, expert-profile `Strata/data/expert-profile.bin`, PLE
`Strata-data/ple/ple-fp8.gguf`, KV int8.

```text
--pack D:/Strata-data/packs/swift-iq3_xxs --native D:/Strata-data/models/SWIFT_IQ3_XXS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf D:/Strata-data/ple/ple-fp8.gguf --expert-profile D:/Strata/data/expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp D:/Strata-data/mtp/rt --max-context 262144 --kv int8 --kv-resident 32768 --vision --vram-reserve-mib 1000 --pcie-frac 0.00 --spec-min-p 0.70 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97
```

## Method

`tools/strata_bench.py` against the live server: a warm-up chat and a 1K prompt are discarded; every
prompt starts with a unique tag so the conversation cache cannot help the fresh-prompt tests
(reused=0). Throughput from the engine's own timings. Recall:
[`tools/needle_bench.py`](../../../tools/needle_bench.py) `--lengths 32k,256k --depths 10,50,90`.
Per-run data: [`results.json`](results.json), [`needles.json`](needles.json).

## Results (median of 3 runs, range in brackets)

| Test | Prompt tok/s | Decode tok/s | TTFT s (client) |
| --- | --- | --- | ---: |
| short chat | n/a | **103.5** (103.1-106.0) | 0.25 |
| 4,096 fresh | 2308.7 (2302-2310) | n/a | 1.78 |
| 16,384 fresh | **3789.1** (3789-3795) | n/a | 4.28 |
| agent session | n/a | **122.7** (111-123) | 1.62 |

### Agent session (follow-up turns, representative arm)

Cache hit over the follow-up turns: **70.4%** (median over the 3 runs); first turn (nothing
cached) first token: 1.01s.

| Turn | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 667.4 | 0.94 | 102.5 |
| 1 | 4489 | 636 | 3853 | 2150.5 | 1.83 | 104.1 |
| 2 | 6022 | 4552 | 1470 | 1273.7 | 1.19 | 116.1 |
| 3 | 9250 | 6085 | 3165 | 1833.8 | 1.77 | 95.5 |
| 4 | 11333 | 9302 | 2031 | 1550.9 | 1.34 | 123.3 |
| 5 | 14242 | 11384 | 2858 | 1725.1 | 1.68 | 126.6 |

## Recall checks

One code word in a haystack of project text, lengths 32k,256k at depths 10/50/90%:

| Length | Depth | Result | Prompt tokens | Seconds |
| ---: | ---: | --- | ---: | ---: |
| 32k | 10% | found | 31669 | 8.8 |
| 32k | 50% | found | 31669 | 8.5 |
| 32k | 90% | found | 31670 | 8.5 |
| 256k | 10% | found | 248409 | 66.1 |
| 256k | 50% | found | 248407 | 66.1 |
| 256k | 90% | found | 248407 | 66.0 |

**6 / 6 found.** Raw runs: `needles.json`.

## Observations

- The 16K fresh-prompt read (**3789 tok/s**) beats every earlier engine on this box,
  including the 0.1.56 reference day (3687) that previously set the ceiling - consistent with the
  0.1.60 prefill now running on Strata's own kernels (NVIDIA/AMD) instead of the borrowed llama.cpp
  path.
- Agent follow-up decode **122.7 tok/s**, cache hit 70.4% (the hit band is a
  property of the harness prompts).

## Correctness and limitations

Needle recall is the only quality check. One machine, single stream, greedy, thinking off. TTFT
includes HTTP/first-token time; the engine was not restarted during the three runs. Engine version is
recorded honestly as 0.1.60 (the served wrapper reports its own project version separately).
