# Fresh MTP versus all supported DFlash2 weight formats — 2026-10-19

Metadata correction (2026-10-20): the shared target head is Q8_0 (GGML type 8, 2560 × 248320, 675430400 bytes). The older engine hard-coded Q5_K in its log. This corrects the format label; the original timings and raw logs are unchanged. See the [follow-up](../2026-10-20-dflash2-long-tuned/README.md).

Engine source: `af85f5ce3b4411833b5529361ae18ffdadbdce3f`. This evidence adds scripts/results only; engine input hashes remain those in the preceding validation package. Executable hashes, exact commands and environment are retained per arm.

Fresh same-source runs on RTX 4070 Ti SUPER 16 GiB, Ryzen 9 5900X, 125.7 GiB RAM, Linux, CUDA 13.4 / SM89. Target: local Qwen3.8-Flash-Next NVFP4 pack, BF16 embeddings, FP8 PLE, shared Q8_0 output head, INT8 KV. Context 8192, prefill 256, auto expert cache, allocation reserve 512 MiB. All arms fill available VRAM; smaller draft allocations permit more target experts.

Greedy seed 1234, 96 new tokens, four fixed prompts (27/54/33/2425 input tokens), one warmup per prompt and three measured repeats per speculative arm. MTP at floor 0 is run before and after all four DFlash2 variants. An additional MTP run at setup's floor 0.5 follows those series and is the table reference; this sensitivity check was added after the initial series to avoid comparing only against an untuned zero-floor baseline. Both MTP settings remain in the raw evidence.

DFlash2 uses three proposals (verifier capacity 8, fixed block 3); MTP uses up to three proposals (verifier capacity 4), probability floor 0 in the fixed-budget series, plus the separate MTP setup-floor 0.5 control. The DFlash2 block length is fixed; the study does not independently tune each drafter.

CPU prompt sharing, PCIe sharing, adaptive swaps, suffix/lookup drafting and prompt/conversation snapshots are off. Engines run sequentially, with no build or other benchmark during measurements. Values are medians of native decode throughput; prefill/startup are excluded. Geomean weights all four task medians equally.

| Drafter | Code tok/s | Math tok/s | Prose tok/s | Long tok/s | Geomean tok/s (vs MTP) |
| --- | ---: | ---: | ---: | ---: | ---: |
| MTP (setup floor 0.5) | 38.05 | 44.84 | 39.94 | 55.19 | 44.04 (reference) |
| MTP (floor 0, pooled) | 36.81 | 43.68 | 36.26 | 48.61 | 41.03 (-6.8%) |
| DFlash2 BF16 | **43.25 (+13.7%)** | 43.16 (-3.7%) | 36.40 (-8.9%) | 41.77 (-24.3%) | 41.05 (-6.8%) |
| DFlash2 Q8_0 | **45.08 (+18.5%)** | 44.45 (-0.9%) | 37.05 (-7.2%) | 44.58 (-19.2%) | 42.65 (-3.1%) |
| DFlash2 Q5_0 | **45.19 (+18.8%)** | 44.87 (+0.1%) | 33.71 (-15.6%) | 43.89 (-20.5%) | 41.62 (-5.5%) |
| DFlash2 Q4_0 | **47.11 (+23.8%)** | **45.36 (+1.2%)** | **40.66 (+1.8%)** | 43.52 (-21.1%) | 44.10 (+0.1%) — tie |

Q4_0 has the best observed DFlash2 aggregate, 44.10 versus MTP's 44.04 tok/s (+0.1%). Their per-repeat geometric-mean ranges overlap: treat this as an overall tie. Its clear code advantage is +23.8%; math/prose improve by 1.2%/1.8% in this sample, while the long prompt regresses by 21.1%. Q8_0, Q5_0 and BF16 are slower in aggregate.

Per-repeat geometric means span 43.00–44.10 tok/s for MTP floor 0.5 and 44.07–44.20 tok/s for DFlash2 Q4_0. Bold task cells mean every measured DFlash2 sample exceeded every MTP floor-0.5 sample for that task; percentages are relative to MTP for that column. The bracketing zero-floor MTP's end/start geomean changed by -0.49%. Three samples per DFlash2/default-floor MTP task and six pooled zero-floor MTP samples are too few to establish a general speedup. Non-overlapping observed throughput ranges: bf16/code, Q8_0/code, Q5_0/code, Q4_0/code, Q4_0/math, Q4_0/prose. These are descriptive ranges, not confidence intervals.

## Memory and placement

| Drafter | Initial runtime MiB | Extra draft-head MiB | GPU expert slots | Minimum GPU free MiB |
| --- | ---: | ---: | ---: | ---: |
| MTP floor 0, start | 801.0 | 275.7 | 3140 | 182 |
| DFlash2 BF16 | 1354.6 | 0.0 | 3038 | 130 |
| DFlash2 Q8_0 | 945.0 | 0.0 | 3193 | 132 |
| DFlash2 Q5_0 | 780.9 | 0.0 | 3256 | 133 |
| DFlash2 Q4_0 | 726.2 | 0.0 | 3277 | 129 |
| MTP floor 0, end | 801.0 | 275.7 | 3141 | 181 |
| MTP floor 0.5 | 801.0 | 275.7 | 3143 | 166 |

The initial runtime column is the engine's load counter, not total process VRAM. MTP subsequently allocates another 275.7 MiB of selected output-head rows, plus index/sampling buffers. DFlash2's counter includes its weights, private KV pools and scratch, and it shares the full target output head without another head-weight copy. Expert slots and GPU-free snapshots show the practical final placement.

Reserve is withheld before later verifier/graph allocations; it does not promise that much memory free after graph capture. Auto sizing is intentional, so this compares practical weight/cache effects rather than isolated kernels. Model identities, formats and hashes are in `artifacts.json`; no weights are included. BF16 is the original drafter; Q8_0/Q5_0/Q4_0 quantize only the 36 backbone matrices and keep 45 other tensors BF16. The target output head is shared.

## Output controls and raw data

All 16 DFlash2 task/format output controls: **PASS**. `output-controls.json` records the individual outcomes. `summary.json` retains every mismatch against the separate auto-cache target-only arm. If a speculative sequence differs from that arm, a same-process target-only control (`temperature=1 top_k=1`) retains the drafter allocation and expert placement, disables proposal generation and checks the complete 96-token sequence against all three repeats. Those controls are outside measured samples. This identifies placement-dependent target rounding instead of silently hiding differing answers.

The per-arm JSON retains full output IDs, warmups, repeat timings, native INFO, accepted/offered proposals, VRAM snapshots and quality controls. `summary.json` includes median/min/max throughput and request-wall throughput. MTP's first/last runs and all individual measurements remain available. These short controlled prompts do not predict arbitrary conversations or other hardware.

## Reproduction and attempts excluded

Run `.venv/bin/python bench/results/2026-10-19-dflash2-pr-performance/suite.py`, then `analyze.py` in that directory. Existing local model files and the built engine are required. The suite skips complete arms; use a fresh evidence directory for a genuinely new run. Logs replace private checkout paths with `.`.

The first prefill-512 MTP attempt was stopped after warmup/partial measurements to use prefill 256 consistently; it is retained under `prefill512-interrupted/` and excluded. A first restart overlapped termination of that old process and failed the host memory guard; `startup-overlap-failed/` retains its traceback/log/exit status. The old process tree was then fully stopped before restarting the final complete series. No incomplete attempts enter the table. This is supporting engine-validation evidence, not an approved standard benchmark submission.
