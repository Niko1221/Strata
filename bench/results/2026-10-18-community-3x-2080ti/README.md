# Community benchmark: 3× RTX 2080 Ti 22 GB (sm_75), i9-7980XE, 125.5 GiB

Three consumer Turing cards on one X299 board, a 125B MoE at its native
262,144-token window. Measured with `tools/strata_bench.py --long` (suite 1,
3 runs per test, medians with ranges) against a running server, plus a one-hour
soak and two A/Bs on the same rig.

## Hardware and software

| | |
| --- | --- |
| GPU | 3× NVIDIA GeForce RTX 2080 Ti, 22 GiB each (Turing, sm_75) |
| GPU power limits | **210 W / 167 W / 200 W** (default 250 W each; tuned for tokens-per-watt, not peak) |
| GPU links | GPU1–GPU2: `NV2` (two bonded NVLink bridges, unused by the engine); GPU0–GPU1/2: PCIe. Under load GPU0 x16 Gen3, GPU1 x16, GPU2 x8 |
| CPU | Intel Core i9-7980XE, 18 cores / 36 threads (Skylake-X); the engine's expert kernels ran on **AVX-2** (its own log line) |
| RAM | 125.5 GiB DDR4-3600 (8×16 GB, 4 channels, 2 DIMMs per channel) |
| Storage | model on NVMe (md1 RAID0), expert arena read through the file cache |
| OS | Linux 7.0.0-38-generic |
| Driver / CUDA | 580.178.04 / CUDA 13.0 |
| Strata | **v0.1.45** (`477ac13c`), source build, `-DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75`; `cuobjdump` shows 62 `sm_75` cubins, no `sm_86` |

## Model and configuration

Qwen3.8-Flash-Next **UD-Q4_K_XL** (pack `ud-q4_k_xl`, four GGUF shards):

```
--pack .../pack/ud-q4_k_xl --native ...UD-Q4_K_XL-00001-of-00004.gguf
--ple-gguf ...UD-Q4_K_XL-00002-of-00004.gguf --expert-profile data/expert-profile.bin
--expert-cache auto --prefill auto --spec 3 --spec-min-p 0.54 --mtp mtp/rt
--max-context 262144 --kv int8 --batch 2 --conversation-cache-mib 0
--prompt-cache 4 --trim-stage-weights
```

`layer_split: "16,32"` (CUDA0 layers 0–15, CUDA1 16–31, CUDA2 32–47),
`STRATA_NO_LARGEPAGES=1`, `STRATA_BF16_TC=1`. **`STRATA_SM75_PERF` unset** — the
release default, so the Turing kernel series is off in this report. The expert
cache sized to 14,734 slots across the three cards (5,064 primary, 15.0 GiB);
engine-reported `vram_free_mib` at start: 456.

## Method

`python tools/strata_bench.py --url http://127.0.0.1:8097 --long` against a
server already running (model load not timed). One warm-up chat and a 1K prompt
are discarded first; every prompt carries a unique tag so the conversation cache
cannot help the fresh-prompt tests. The third card was idle and its two usual
servers (an embedding model and an Ornith-1.5-35B llama-server) were stopped for
the run.

**Tool note (#2887):** `strata_bench.py` recorded this layer-split run as one
card (`"gpu": ["NVIDIA GeForce RTX 2080 Ti"]`); `results.json` here was corrected
to the real three-card set and the power limits, and the fields the tool left
empty (`quant`, `config`) were filled from the server's own start line.

## Results

| Configuration | Actual prompt tokens | Reused | Generated | Runs | Prompt tok/s median (range) | Decode tok/s median (range) | TTFT s median (range) |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43–51) | 0 | 100 | 9 | not measured | 64.1 (58.4–81.2) | 0.53 (0.49–0.59) |
| 4K prompt, fresh | 4,019 | 0 | 24 cap | 3 | 581.3 (580.5–581.5) | not measured | 6.94 (6.94–6.96) |
| 16K prompt, fresh | 16,058 | 0 | 24 cap | 3 | 1,342.7 (1,281.8–1,353.9) | not measured | 12.01 (11.91–12.58) |
| 32K prompt, fresh | 32,049 | 0 | 24 cap | 3 | 1,761.1 (1,755.3–1,763.0) | not measured | 18.28 (18.26–18.34) |
| agent session, follow-up turns | 14,577 at the last turn | 68.8% cache hit | up to 64 | 3 | see below | 71.7 (62.2–78.3) | 6.63 (6.12–6.91) |

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8%; earlier-turn tokens read again per
session: 3; the 13,719 new file tokens are read once. First turn (nothing
cached) first token: 3.89 s.

| Turn | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 157.8 | 3.891 | 56.4 |
| 1 | 3,759 | 638 | 3,121 | 454.6 | 6.902 | 68.7 |
| 2 | 5,292 | 3,823 | 1,469 | 240.8 | 6.134 | 76.7 |
| 3 | 8,725 | 5,355 | 3,370 | 490.5 | 6.907 | 66.9 |
| 4 | 11,634 | 8,772 | 2,862 | 427.5 | 6.732 | 76.0 |
| 5 | 14,593 | 11,694 | 2,899 | 439.7 | 6.63 | 63.3 |

Per-request data: `results.json`. Total suite time: 266 s.

## Addendum A — one-hour soak (the `TEST_REQUESTS` item)

Separate run on **two** of the three cards (GPUs 0–1, layer split 15,
`STRATA_SM75_PERF=1`), same model/config at 262,144 tokens, one engine start,
505 back-to-back requests over 60 minutes (short chats, ~1.6K-token prompts, a
~12.6K-token prompt every seventh request), 128 new tokens each:

| 10-min bucket | requests | decode median (tok/s) | p10 / p90 |
| --- | ---: | ---: | --- |
| 0–10 | 83 | 39.6 | 36.9 / 43.5 |
| 10–20 | 82 | 40.2 | 37.3 / 44.3 |
| 20–30 | 86 | 41.8 | 38.7 / 45.8 |
| 30–40 | 84 | 41.6 | 38.4 / 47.3 |
| 40–50 | 84 | 41.8 | 38.9 / 46.7 |
| 50–60 | 84 | 41.8 | 38.1 / 46.1 |

**0 errors, 0 restarts**, overall median 41.3 tok/s; max GPU temperatures
43 °C / 49 °C. No drift: the last bucket is slightly faster than the first
(expert cache warm-up).

## Addendum B — `STRATA_SM75_PERF=0` vs `1` (2 cards, 3 rounds)

Same binary, environment only, alternating order, `tools/ab_engine.py`:

| Request | PERF=0 | PERF=1 | Δ |
| --- | ---: | ---: | ---: |
| short chat decode (3 prompts) | 40.2 / 40.8 / 40.4 | 41.4 / 41.1 / 40.2 | +3.0 / +0.7 / −0.5% |
| 10,558-token decode | 39.1 | 39.7 | +1.5% |
| 10,558-token prefill | 598.6 | 603.1 | +0.8% |
| 25,298-token decode | 38.9 | 39.5 | +1.5% |
| 25,298-token prefill | 884.4 | 898.0 | +1.5% |

## Addendum C — 0.1.43.1 vs 0.1.45 (3 cards, default-vs-default, 3 rounds)

| Request | 0.1.43.1 | 0.1.45 | Δ |
| --- | ---: | ---: | ---: |
| chat0 decode | 62.2 | 63.5 | +2.1% |
| chat1 decode | 67.4 | 63.2 | −6.2% |
| chat2 decode | 61.4 | 63.3 | +3.1% |
| 10,558-token decode | 65.8 | 64.3 | −2.3% |
| 10,558-token prefill | 883.8 | 884.4 | +0.1% |

## Limitations

- The cards run at **210 / 167 / 200 W**, not their 250 W default: absolute
  numbers are below what the same rig does at full power.
- Speed only: no recall, needle or token-identity check in this suite.
- The engine reported its expert kernels on the **AVX-2** path (Q4_K_XL uses the
  K-quant rows; an IQ3_S pack would exercise the i-quant path).
- `strata_bench` did not record the card list, quant or config for this
  layer-split run (#2887); those fields were filled in by hand from the server's
  start line and `nvidia-smi`.
