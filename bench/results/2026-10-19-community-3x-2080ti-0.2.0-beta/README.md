# Community benchmark: 3× RTX 2080 Ti 22 GB (sm_75), Strata 0.2.0-beta, UD-Q4_K_XL at 262K

Three consumer Turing cards on one X299 board, a 125B MoE at its native 262,144-token
window, measured with `tools/strata_bench.py --long` against a running server. This is the
0.2.0-beta follow-up of `2026-10-18-community-3x-2080ti` (the same rig and config on
0.1.45), so the two folders compare directly.

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
| Strata | **0.2.0-beta**, engine 0.1.45.3 (`3d1fb480`), source build, `-DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75`; `cuobjdump --list-elf` shows sm_75 cubins only |

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
`STRATA_NO_LARGEPAGES=1`, `STRATA_BF16_TC=1`. The expert cache sized to 14,747 slots
across the three cards (5,060 primary). Sampling: greedy for the suite (temperature 0,
`top_k 1`, `presence_penalty 0`).

## Method

`python tools/strata_bench.py --url http://127.0.0.1:8097 --long` against a server already
running (model load not timed). One warm-up chat and a 1K prompt are discarded first; every
prompt carries a unique tag so the conversation cache cannot help the fresh-prompt tests.
Three runs per test, medians with ranges. Speed only: no correctness or recall check.

## Results

| Configuration | Actual prompt tokens | Generated | Runs | Prompt tok/s median (range) | Decode tok/s median (range) | TTFT s median (range) |
| --- | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43–51) | 100 | 9 | not measured | 65.2 (55.1–84.1) | 0.56 (0.49–0.60) |
| 4K prompt, fresh | 4,019 | 24 cap | 3 | 586.8 (584.2–587.0) | not measured | 6.88 (6.88–6.91) |
| 16K prompt, fresh | 16,058 | 24 cap | 3 | 1,345.2 (1,341.7–1,352.1) | not measured | 11.98 (11.92–12.01) |
| 32K prompt, fresh | 32,049 | 24 cap | 3 | 1,758.7 (1,757.9–1,762.0) | not measured | 18.28 (18.24–18.30) |
| agent session, follow-up turns | 14,619 at the last turn | up to 64 | 3 | see below | 69.7 (53.5–81.0) | 6.52 (6.07–6.90) |

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.3% (69.3–69.4); earlier-turn tokens read again per
session: 73 (2–74); the 13,745 new file tokens are read once. First turn (nothing cached)
first token: 3.87 s (3.83–3.88).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 158.3 | 3.876 | 50.5 |
| 1 | 4,039 | 619 | 3,420 | 498.4 | 6.900 | 69.1 |
| 2 | 5,572 | 4,032 | 1,540 | 253.4 | 6.111 | 65.4 |
| 3 | 9,004 | 5,635 | 3,369 | 495.0 | 6.844 | 53.5 |
| 4 | 11,750 | 9,069 | 2,681 | 408.7 | 6.603 | 69.4 |
| 5 | 14,621 | 11,813 | 2,808 | 433.3 | 6.522 | 71.0 |

Per-request data: `results.json`. Total suite time: 263 s.

## 0.1.45 → 0.2.0-beta on the same rig, same config

| | 0.1.45 (`2026-10-18-community-3x-2080ti`) | 0.2.0-beta (this folder) | Δ |
| --- | ---: | ---: | ---: |
| short chat decode | 64.1 (58.4–81.2) | 65.2 (55.1–84.1) | +1.7% |
| 4K prompt | 581.3 (580.5–581.5) | 586.8 (584.2–587.0) | +0.9% |
| 16K prompt | 1,342.7 (1,281.8–1,353.9) | 1,345.2 (1,341.7–1,352.1) | +0.2% |
| 32K prompt | 1,761.1 (1,755.3–1,763.0) | 1,758.7 (1,757.9–1,762.0) | −0.1% |
| agent decode | 71.7 (62.2–78.3) | 69.7 (53.5–81.0) | −2.8% (ranges overlap) |
| agent cache hit | 68.8% | 69.3% | — |

Everything is inside the run-to-run spread of this box; the engine's speed on this
configuration did not change between the two releases.

## Tool note (#2887)

`strata_bench.py` records a layer-split run as one card (`"gpu": ["NVIDIA GeForce RTX 2080 Ti"]`)
and leaves `quant` and `config` empty. In `results.json` here those fields were corrected to
the real three-card set, the power limits, the quant (`UD-Q4_K_XL`), the layer split and the
exact engine arguments from the server's start line. The measured numbers are untouched.

## Limitations

- Speed only: no recall, needle or token-identity check in this suite.
- The cards run at **210 / 167 / 200 W**, not their 250 W default: absolute numbers are below
  what the same rig does at full power.
- The engine reported its expert kernels on the **AVX-2** path (Q4_K_XL uses the K-quant rows;
  an IQ3_S pack would exercise the i-quant path).
- Three runs per prompt size; the agent rows are 15 requests.
