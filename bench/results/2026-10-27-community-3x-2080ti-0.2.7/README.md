# Community benchmark: 3× RTX 2080 Ti 22 GB (sm_75), Strata 0.2.7, UD-Q4_K_XL at 262K

Same rig, same config, third release in the series: 0.1.45 → 0.2.0-beta → **0.2.7**.
Measured with `tools/strata_bench.py --long` against a running server.

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
| Strata | **0.2.7**, engine 0.1.52 (`5e8a4cca`), source build, `-DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75` |

## Model and configuration

Qwen3.8-Flash-Next **UD-Q4_K_XL** (pack `ud-q4_k_xl`, four GGUF shards):

```
--pack .../pack/ud-q4_k_xl --native ...UD-Q4_K_XL-00001-of-00004.gguf
--ple-gguf ...UD-Q4_K_XL-00002-of-00004.gguf --expert-profile data/expert-profile.bin
--expert-cache auto --prefill auto --spec 3 --spec-min-p 0.54 --mtp mtp/rt
--max-context 262144 --kv int8 --batch 2 --conversation-cache-mib 0
--prompt-cache 4 --trim-stage-weights
```

`layer_split: "16,32"`, `STRATA_NO_LARGEPAGES=1`, `STRATA_BF16_TC=1`. Sampling: greedy
(temperature 0, `top_k 1`, `presence_penalty 0`).

One setup note specific to 0.2.6 and later: the engine's pre-load arena check reads the
process's cgroup `memory.max`, and under a 4 GiB-limited scope the load is refused with
"the expert arena needs 71.73 GiB of RAM but this process's memory limit ... is 4.00 GiB".
The server here ran under a transient unit with `MemoryMax=110G`.

## Method

`python tools/strata_bench.py --url http://127.0.0.1:8097 --long` against a server already
running (model load not timed). One warm-up chat and a 1K prompt are discarded first; every
prompt carries a unique tag so the conversation cache cannot help the fresh-prompt tests.
Three runs per test, medians with ranges. Speed only: no correctness or recall check.

## Results (0.2.7)

| Configuration | Actual prompt tokens | Generated | Runs | Prompt tok/s median (range) | Decode tok/s median (range) | TTFT s median (range) |
| --- | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43–51) | 100 | 9 | not measured | 64.7 (55.1–84.0) | 0.54 (0.49–0.60) |
| 4K prompt, fresh | 4,019 | 24 cap | 3 | 586.6 (578.9–586.7) | not measured | 6.88 (6.88–6.97) |
| 16K prompt, fresh | 16,058 | 24 cap | 3 | 1,342.3 (1,332.9–1,344.5) | not measured | 12.00 (11.98–12.09) |
| 32K prompt, fresh | 32,049 | 24 cap | 3 | 1,753.9 (1,753.2–1,758.2) | not measured | 18.33 (18.28–18.33) |
| agent session, follow-up turns | 12,817 at the last turn | up to 64 | 3 | see below | 69.4 (62.6–81.7) | 6.46 (6.06–6.72) |

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.1% (69.1–69.4); earlier-turn tokens read again per
session: 143 (3–144); the 11,952 new file tokens are read once. First turn (nothing cached)
first token: 3.88 s (3.83–3.93).

Per-request data: `results.json`.

## 0.1.45 → 0.2.0-beta → 0.2.7 on the same rig, same config

| | 0.1.45 | 0.2.0-beta | **0.2.7** | 0.2.7 vs 0.1.45 |
| --- | ---: | ---: | ---: | ---: |
| short chat decode | 64.1 (58.4–81.2) | 65.2 (55.1–84.1) | 64.7 (55.1–84.0) | +0.9% |
| 4K prompt | 581.3 (580.5–581.5) | 586.8 (584.2–587.0) | 586.6 (578.9–586.7) | +0.9% |
| 16K prompt | 1,342.7 (1,281.8–1,353.9) | 1,345.2 (1,341.7–1,352.1) | 1,342.3 (1,332.9–1,344.5) | 0.0% |
| 32K prompt | 1,761.1 (1,755.3–1,763.0) | 1,758.7 (1,757.9–1,762.0) | 1,753.9 (1,753.2–1,758.2) | −0.4% |
| agent decode | 71.7 (62.2–78.3) | 69.7 (53.5–81.0) | 69.4 (62.6–81.7) | −3.2% (ranges overlap) |
| agent cache hit | 68.8% | 69.3% | 69.1% | — |

Everything is inside the run-to-run spread of this box: on this configuration the engine's
speed did not move across the three releases. That is consistent with 0.2.7's headline
change being aimed at cards with 12 GB or less (the router prefers experts already on the
card); with 98–99% expert-cache hit rate on 22 GB cards there is little left to save here.

## Tool note (#2887)

`strata_bench.py` records a layer-split run as one card and leaves `quant` / `config` empty.
In `results.json` here those fields were corrected to the real three-card set, the power
limits, the quant (`UD-Q4_K_XL`), the layer split and the exact engine arguments from the
server's start line. The measured numbers are untouched.

## Limitations

- Speed only: no recall, needle or token-identity check in this suite.
- The cards run at **210 / 167 / 200 W**, not their 250 W default: absolute numbers are below
  what the same rig does at full power.
- The engine reported its expert kernels on the **AVX-2** path (Q4_K_XL uses the K-quant rows;
  an IQ3_S pack would exercise the i-quant path).
- Three runs per prompt size; the agent rows are 15 requests.
