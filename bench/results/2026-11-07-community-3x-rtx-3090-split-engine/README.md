# Community benchmark: all-cards split engine on 3x RTX 3090 24 GB (+ a 4th RTX 3090 for the desktop and the picture reader)

Measured on 2026-11-07. The all-cards split engine (`strata-split-3`, HOSTRING) on three GeForce RTX 3090 24 GB, a setup
`strata_control/split_engine.py` lists as untested (NVIDIA sm_86 is tested on 2 cards; 3 cards and GeForce cards are not).
It runs and answers correctly, 30-35 % faster decode than the layer split on two of the same cards (169-174 vs 129 tok/s in our suite),
and serves up to 8 requests at once (we use `--batch 5`). Main limitation: on v0.2.11 it
starts only with `FNX_PF_ROWS=4096 FNX_PF_PASS=4` set by hand, or with the one-line fix proposed in #5496; without them
the start fails (log `engine-log-T1-stock.txt`). One machine, one model.

## Hardware and software

- GPUs: 4x NVIDIA GeForce RTX 3090 24 GB (driver 595.99.02). GPU 0-2 run the engine: PCIe 3.0 x8 / x8 / x16, all on NUMA
  node 0, power limits 350 / 370 / 350 W, no NVLink, no peer access (HOSTRING). GPU 3 (PCIe 3.0 x16, NUMA node 1) drives the
  desktop and runs the picture reader (`"vision": {"cuda_device": 3}`); about 4.2 GB of it is used by the desktop.
- CPU: 2x Xeon E5-2696 v4 (2x 22 cores, 88 threads, no AVX-512); RAM 503 GiB DDR4; model files on NVMe (Samsung 990 PRO +
  WD SN850X, ZFS mirror, PCIe 3.0 x4)
- OS: Ubuntu 26.04.1, kernel 6.17.0-29; CUDA 13.2
- Strata v0.2.11 (c3b3f503), engine 0.1.57, source build: `-DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_SPLIT=ON
  -DSTRATA_SPLIT_CARDS="2;3" -DSTRATA_SPLIT_ARCH=86 -DSTRATA_NATIVE_EXPERTS=ON`; picture reader built with
  `-DSTRATA_VISION_CUDA=ON -DSTRATA_PORTABLE=OFF`
- `RLIMIT_MEMLOCK` raised for the server: the engine `mlock`s the 29.5 GB embedding tables, and with the stock 8 MiB limit the
  CUDA build stops at the lock. Measured `VmLck` of `strata-split-3`: 27.45 GiB after loading and after requests (the CUDA
  page-locked buffers do not count against the limit, `VmPin` 0). We run it with `LimitMEMLOCK=64G` on the user's systemd
  manager (`/etc/systemd/system/user@1000.service.d/memlock.conf`); the bench runs below used `infinity`.
- Background: desktop session on GPU 3, nothing else on GPU 0-2

## Model and configuration

- Model: SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF, IQ3_S (`...-IQ3_S-00001-of-00002.gguf`, `-00002-of-00002.gguf`);
  MTP layer from Strata's `mtp/rt`; picture reader `mmproj-Qwen3.8-Flash-Next-BF16.gguf`
- Context 65536, 1 slot in the bench runs below (`--batch 1..8` in the next section), K/V int8, conversation cache 4096 MiB, arena on (default), coldest 3.0 GiB of experts in RAM
  (1,460 of 24,576, default; no further experts moved), MTP `--spec 6 --lookup 15`, reasoning effort medium in the config
  (the bench suite turns thinking off), no calibration, no speed projection

```text
strata-split-3 --serve --native Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf --mtp <data>/mtp/rt \
  --draft-vocab data/draft_vocab_en.bin --max-context 65536 --spec 6 --lookup 15 --conversation-cache-mib 4096 --batch 1 \
  --vision --layer-split auto
env: STRATA_NO_ARENA_THP=1, plus for "pf4096x4": FNX_PF_ROWS=4096 FNX_PF_PASS=4
```

Two configurations, same machine, each from a fresh engine start:

- **pf4096x4**: v0.2.11 as released, `FNX_PF_ROWS=4096 FNX_PF_PASS=4` (the 0.2.10 prompt-path size).
- **patched-auto**: v0.2.11 plus the 7-line fix from #5496, no `FNX_PF_*`; the engine picks 4096 x 2 itself.

Note on the tool's output: `tools/strata_bench.py` lists only `STRATA_NO_ARENA_THP=1` under "Environment of the config" and the
server prints "layer split across GPUs [0, 1, 2] (auto)", although `FNX_PF_*` were set and the split engine ran (see
`engine-log-*.txt`: "sfnx: loaded ... 3 cards"). The GPU line of the tool shows the three engine cards only.

## Method

`python tools/strata_bench.py --config <config> --runs 3 --long` (suite 1, the tool starts the server; loading not timed; one
warm-up chat and a 1K prompt discarded; unique tags on every fresh prompt; thinking off; temperature 0). Per-request data:
`results-pf4096x4.json`, `results-patched-auto.json`. Engine start-up lines: `engine-log-T1-stock.txt` (failed start without
the fix), `engine-log-T2-workaround.txt`, `engine-log-T3-patched.txt`.

## Results

Medians (lowest-highest over the runs):

| Configuration | Test | Prompt tokens | Reused | Generated | Runs | Prompt tok/s | Decode tok/s | TTFT s |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| pf4096x4 | short chat | 46 (43-51) | 0 | 100 | 9 | not measured | 186.4 (161.3-238.2) | 0.58 (0.39-0.79) |
| pf4096x4 | 4K prompt, fresh | 4019 | 0 | 24 cap | 3 | 1290.6 (1271.6-1324.2) | not measured | 3.14 (3.06-3.18) |
| pf4096x4 | 16K prompt, fresh | 16058 | 0 | 24 cap | 3 | 1832.7 (1823.6-1870.0) | not measured | 8.80 (8.62-8.85) |
| pf4096x4 | 32K prompt, fresh | 32049 | 0 | 24 cap | 3 | 1865.1 (1853.5-1867.3) | not measured | 17.25 (17.24-17.37) |
| pf4096x4 | agent session, follow-up turns | 13156 at turn 5 | 69.3 % | up to 64 | 15 | see json | 175.5 (145.4-202.0) | 2.22 (1.78-3.57) |
| patched-auto | short chat | 46 (43-51) | 0 | 100 | 9 | not measured | 192.9 (164.1-239.6) | 0.56 (0.38-0.77) |
| patched-auto | 4K prompt, fresh | 4019 | 0 | 24 cap | 3 | 1305.1 (1276.4-1308.9) | not measured | 3.10 (3.09-3.17) |
| patched-auto | 16K prompt, fresh | 16058 | 0 | 24 cap | 3 | 1794.9 (1789.8-1814.3) | not measured | 8.99 (8.89-9.02) |
| patched-auto | 32K prompt, fresh | 32049 | 0 | 24 cap | 3 | 1814.7 (1740.5-1818.1) | not measured | 17.73 (17.70-18.49) |
| patched-auto | agent session, follow-up turns | 13150 at turn 5 | 69.3 % | up to 64 | 15 | see json | 171.6 (152.0-217.8) | 2.28 (1.75-3.57) |

Memory (start-up snapshot from the engine log): free VRAM after the arena 548 / 601 / 3067 MiB (pf4096x4) and
1028 / 1081 / 3547 MiB (patched-auto) on GPU 0 / 1 / 2; after loading 318 / 371 / 2837 MiB and 798 / 851 / 3317 MiB.
Load time 133-141 s (weights from NVMe at ~3.5 GB/s, 49.1 GB). PCIe traffic during decode (nvidia-smi dmon, 5 samples, one
moment of a longer run, pf4096x4): GPU 0 rx 1.2-1.4 GB/s / tx 0.6, GPU 1 rx 0.75-0.94 / tx 0.6, GPU 2 rx 2.8-3.2 / tx 0.7-1.0.

Failures: none in the two bench runs. The start without `FNX_PF_*` on unpatched v0.2.11 fails (see #5496).

### Parallel requests: `--batch 1, 3, 5, 8`

Same configuration (pf4096x4, picture reader on GPU 3), only `--batch` changed, each from a fresh start:
`tools/strata_bench.py --runs 3 --clients N` with N = 3 for `--batch 1` and `--batch 3`, N = 5 and 8 for `--batch 5` and 8
(files `results-batch{1,3,5,8}.json`). All four start without moving more experts to RAM (3.0 GiB as above).

| `--batch` | Free VRAM GPU 0 after load | K/V pool positions | Chat decode tok/s (1 request) | Agent decode tok/s | 4K / 16K prompt tok/s | N clients at once: total tok/s | First token median / max |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 318 MiB | 1,132,224 | 187.7 | 168.1 | 1260 / 1815 | 3 clients: 103.2 (queued) | 1.62 / 3.03 s |
| 3 | 276 MiB | 1,114,176 | 181.9 | 167.4 | 1211 / 1744 | 3 clients: 126.6 | 0.87 / 1.72 s |
| 5 | 254 MiB | 1,089,216 | 182.3 | 162.6 | 1250 / 1822 | 5 clients: 137.8 | 1.31 / 2.75 s |
| 8 | 252 MiB | 1,051,776 | 194.6 | 172.8 | 1173 / 1714 | 8 clients: 131.9 | 2.51 / 4.49 s |

One request is not measurably slower with more lanes (chat and agent decode within run-to-run noise; prompt reading up to ~6 %
slower at 8). The total over parallel requests levels off at about 130-140 tok/s from 5 requests on (each client asks for 128
tokens, so the first-token time weighs in; from the per-request latencies the summed decode rate stays near the single-request
rate). With `--batch 1` the extra requests wait in the server's queue (keep-alives every 10 s, no error).

## Correctness and limitations

The bench suite measures speed only. Our own test suite (not part of Strata; private prompts and images, so only totals),
same model and machine, temperature 1.0 / top-p 0.95 / top-k 20, reasoning effort medium, `--num-predict 48000`:

Full run, one pass each (coding 10 tasks with unit tests, bug fixing 5, agent 3 with tools, 714 image questions, 43 text
prompts):

| Engine and split | Coding | Bug fix | Agent | Images (incl. anatomy) | Text prompts | Mean decode tok/s | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v0.2.11 split engine, 3 cards, `--batch 5`, reader on GPU 3 | 9/10 | 5/5 | 3/3 | 509/714 | 39/43 | 173 | 88 min |
| v0.2.11 layer split, 2 cards (GPU 0+1), reader on those cards | 7/10 | 5/5 | 3/3 | 482/714 | 41/43 | 129 | 119 min |

Repeated text tasks, three passes each:

| Engine and split | Coding (10 x 3) | Text prompts (38 x 3) | Mean decode tok/s |
| --- | ---: | ---: | ---: |
| v0.2.10 split engine, 3 cards, `--batch 1` | 26/30 | 106/114 | 165 |
| v0.2.11 split engine, 3 cards, `--batch 1` (pf4096x4) | 24/30 | 103/114 | 169 |

Paired on the same 714 image questions: split engine `--batch 1` 497 vs layer split 481 (55 vs 39 answered by only one,
sign test p = 0.12); split engine `--batch 5` 508 vs 481 (72 vs 45, p = 0.02). With temperature 1.0 and one pass each we read
this as "no loss from the split engine", not as a gain. Mean time per image question: 5.4-5.5 s (split engine, reader on
GPU 3) vs 7.5 s (layer split). The coding differences (2 of 30, 2 of 10) are within this model's run-to-run variation.

Not tested: more than 8 lanes, context above 65536, long multi-user sessions over hours, the HIP build, other models, a config
written by the launcher (this one was made by hand from `split_engine.split_config` without `split_fallback`).
