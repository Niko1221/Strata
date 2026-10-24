# Community benchmark: 2x RX 6900 XT (gfx1030), layer split, 0.2.0-beta main, Flash-Next IQ3_S

Measured on 2026-10-20 by Vop3p with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges), plus real-text prompt and decode runs (below). Stock main e88ee39d (v0.2.0-beta + 12 docs/packaging commits), source build with setup.py's HIP options; no local patches. Main limitation: one machine, one model, speed only. Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: 2x AMD Radeon RX 6900 XT (gfx1030, 16 GB each), PCIe 4.0 x8 each (the engine's probe: 14.1 GB/s host to device); stock power limits
- CPU: AMD Ryzen 5 5600X (6 cores, 12 threads); RAM: 128 GB DDR4-3200 (121.5 GiB visible to the OS); storage: NVMe SSD
- OS: Ubuntu 26.04, Linux 7.0.0-38-generic (in-tree amdgpu); ROCm 10.0.0 (core-10.0 clang)
- Strata: main e88ee39d (engine 0.1.45.3), source build: `-DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_PREFILL_MMQ=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON -DCMAKE_HIP_ARCHITECTURES=gfx1030`, the pinned ggml
- Background workloads: none (the box's other model services were stopped for the run); GFXOFF at the distribution default

## Model and configuration

- Model: ISTA-DASLab Qwen3.8-Flash-Next GSQ-RCO, IQ3_S (both GGUF shards), the pack and MTP layer prepared by setup; no vision encoder
- Layer split over both cards (auto: layers 0-23 / 24-47), expert cache auto: 9,807 slots (5,534 + 4,273), int8 KV with 32,768 resident cells, context 262,144, `"parallel": 2` (2 batch slots, batch groups 2)
- MTP `--spec 3 --spec-min-p 0.7`, asked for with `--pipeline-windows 2 --batch-mtp`; with batch slots on a layer split 0.2.0 turns both off ("--pipeline-windows 2 is off: not with --batch slots"; "--batch-mtp is off: a layer split with --batch pipelines the slots in groups", `--batch-groups auto`: 2 groups of 1 slot), so a lone request decodes without the window overlap; thinking off for the suite; temperature 0

```text
args: --expert-cache auto --prefill auto --spec 3 --spec-min-p 0.7 --mtp <mtp dir> --max-context 262144 --kv int8
      --kv-resident 32768 --ple-io ram --vram-reserve-mib 700 --pipeline-windows 2 --batch-mtp   ("parallel": 2, layer_split auto)
env:  STRATA_IO_THREADS=128 STRATA_SH_STREAM=1 STRATA_HIP_PROMPT_F16=1 STRATA_HIP_ADAPT_KERNEL_COPY=1
      STRATA_PREFILL_CPU_SHARE=auto STRATA_SPLIT_MTP_BATCH=0 STRATA_HC_PERSIST=1 STRATA_PIPED_CONCURRENT=1
      STRATA_PREFILL_STREAM_MIN=2048 STRATA_ROCBLAS_TUNING=<a gfx1030 rocBLAS solution table>
```

## Method

Run by `python tools/strata_bench.py --url http://127.0.0.1:8089 --clients 2 --long --runs 3` against a freshly started server (cold start, then the suite); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 61.6 (56.8-69.0) | 0.54 (0.49-0.60) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 979.0 (978.8-979.8) | not measured | 4.13 (4.13-4.14) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1415.6 (1414.3-1417.3) | not measured | 11.38 (11.36-11.39) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1661.5 (1660.9-1663.8) | not measured | 19.34 (19.30-19.34) |
| agent session, follow-up turns | 14677 (14671-14707) at the last turn | see below | up to 64 | 3 | see below | 66.7 (58.8-78.4) | 3.54 (3.01-3.94) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.5 % (69.5-69.7); earlier-turn tokens read again per session: 74 (3-75) (earlier turns that the cache did not hold; the 13813 (13813-13813) new tokens of the files are read once); first turn (nothing cached) first token: 2.73 s (2.68-2.73).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 661 | 0 | 661 | 249.5 | 2.676 | 52.9 |
| 1 | 4161 | 674 | 3487 | 904.3 | 3.888 | 60.5 |
| 2 | 5694 | 4224 | 1470 | 479.2 | 3.096 | 60.9 |
| 3 | 9127 | 5758 | 3369 | 872.4 | 3.892 | 66.2 |
| 4 | 11869 | 9186 | 2683 | 775.7 | 3.49 | 66.7 |
| 5 | 14707 | 11900 | 2807 | 798.4 | 3.546 | 68.6 |

### Concurrency: 2 clients

Total 42.8 tok/s (41.6-46.3); first token p50 1.261 s, p95 1.748 s; request latency p50 5.967 s, p95 6.15 s; errors 0. Server parallel slots: 2 (requests beyond that wait their turn).

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

### Real text (same server, after the suite)

The suite's prompts are generated word lists; routing-sensitive paths read real text differently (on this box a word-list prompt overstated one prompt switch's gain, so I also measure real text). Three runs each on the same server, every prompt tagged, 0 tokens reused; the text is a private mixed English / Chinese / code corpus (not attached):

| Test | Prompt tokens | Generated | Runs | Prompt tok/s median (range) | Decode tok/s median (range) |
| --- | ---: | ---: | ---: | --- | --- |
| long read, max_tokens 1 | 31,039 | 1 | 3 | 1,621 (1,620-1,628) | - |
| long read, max_tokens 1 | 119,290 | 1 | 3 | 1,893 (1,892-1,896) | - |
| one request, summarize ~400 words, greedy | 4,123 | 495-512 | 3 | 934 (933-935) | 66.5 (64.4-67.7) |
| one request, summarize ~400 words, greedy | 30,322 | 512 | 3 | 1,608 (1,607-1,611) | 66.0 (65.4-67.2) |

The same two long reads on 20 more cold starts of builds whose default path is byte-identical to this one (two PR branches with their variables unset, 2026-10-20): 31,039 tokens 19,105-19,382 ms, 119,290 tokens 62,583-62,882 ms - the same as here (19,143-19,157 / 63,027-63,054 ms).

## Correctness and limitations

Not measured here: this is a speed suite (the same build passed an 8-chat log-probability comparison against itself as the baseline of the PR quality gates, not repeated in this folder). The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 224 s.

Developed with an AI coding assistant; every number above was measured on the machine described here.
