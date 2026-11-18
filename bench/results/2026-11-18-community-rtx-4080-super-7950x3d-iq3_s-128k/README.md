# Community-format benchmark: RTX 4080 SUPER, Ryzen 9 7950X3D, IQ3_S at 128K

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

qwen3.8-flash-next-iq3_s, **IQ3_S** (128K context), pack `Strata-data/packs/iq3_s`, MTP
`Strata-data/mtp/rt`, expert-profile `Strata/data/expert-profile.bin`, PLE
`Strata-data/ple/ple-fp8.gguf`, KV int8.

```text
--pack D:/Strata-data/packs/iq3_s --native D:/Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf D:/Strata-data/ple/ple-fp8.gguf --expert-profile D:/Strata/data/expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp D:/Strata-data/mtp/rt --max-context 131072 --kv int8 --vision --vram-reserve-mib 700 --pcie-frac 0.20 --spec-min-p 0.70 --adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97
```

## Method

`tools/strata_bench.py` against the live server: a warm-up chat and a 1K prompt are discarded; every
prompt starts with a unique tag so the conversation cache cannot help the fresh-prompt tests
(reused=0). Throughput from the engine's own timings. Recall:
[`tools/needle_bench.py`](../../../tools/needle_bench.py) `--lengths 32k,128k --depths 10,50,90`.
Per-run data: [`results.json`](results.json), [`needles.json`](needles.json).

## Results (median of 3 runs, range in brackets)

| Test | Prompt tok/s | Decode tok/s | TTFT s (client) |
| --- | --- | --- | ---: |
| short chat | n/a | **90.2** (87.8-93.8) | 0.28 |
| 4,096 fresh | 2048.9 (2046-2059) | n/a | 1.99 |
| 16,384 fresh | **3805.6** (3738-3812) | n/a | 4.27 |
| agent session | n/a | **108.6** (108-110) | 1.84 |

### Agent session (follow-up turns, representative arm)

Cache hit over the follow-up turns: **70.5%** (median over the 3 runs); first turn (nothing
cached) first token: 0.96s.

| Turn | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 666.9 | 0.93 | 75.4 |
| 1 | 4485 | 632 | 3853 | 1977.2 | 1.98 | 104.4 |
| 2 | 6018 | 4548 | 1470 | 1170.6 | 1.27 | 98.5 |
| 3 | 9246 | 6081 | 3165 | 1646.7 | 1.96 | 97.3 |
| 4 | 11341 | 9309 | 2032 | 1414.3 | 1.48 | 117.9 |
| 5 | 14251 | 11393 | 2858 | 1581.0 | 1.84 | 115.9 |

## Recall checks

One code word in a haystack of project text, lengths 32k,128k at depths 10/50/90%:

| Length | Depth | Result | Prompt tokens | Seconds |
| ---: | ---: | --- | ---: | ---: |
| 32k | 10% | found | 31669 | 16.0 |
| 32k | 50% | found | 31669 | 15.6 |
| 32k | 90% | found | 31670 | 15.7 |
| 128k | 10% | found | 123189 | 62.0 |
| 128k | 50% | found | 123187 | 62.3 |
| 128k | 90% | found | 123187 | 39.6 |

**6 / 6 found.** Raw runs: `needles.json`.

## Observations

- The 16K fresh-prompt read (**3806 tok/s**) beats every earlier engine on this box,
  including the 0.1.56 reference day (3687) that previously set the ceiling - consistent with the
  0.1.60 prefill now running on Strata's own kernels (NVIDIA/AMD) instead of the borrowed llama.cpp
  path.
- Agent follow-up decode **108.6 tok/s**, cache hit 70.5% (the hit band is a
  property of the harness prompts).

## Correctness and limitations

Needle recall is the only quality check. One machine, single stream, greedy, thinking off. TTFT
includes HTTP/first-token time; the engine was not restarted during the three runs. Engine version is
recorded honestly as 0.1.60 (the served wrapper reports its own project version separately).
