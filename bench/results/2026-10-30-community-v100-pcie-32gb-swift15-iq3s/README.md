# Community benchmark on one Tesla V100-PCIE-32GB (sm_70) with 2x Xeon Gold 6138: Swift 1.5 IQ3_S at 262K context

Measured 2026-10-29/30 by **koldfrontier**. One V100, one host, Strata 0.1.52 (`5e8a4cca`), Swift 1.5 IQ3_S, with
the same levers checked once more on Qwen3.8-Flash-Next IQ3_S. Greedy decoding throughout; every arm is a fresh engine
start. Medians of 2-3 starts per cell. Raw rows, configs, logs and the harness are in this folder.

## Findings

1. **PR #3411's 13 Volta environment switches** (0.1.52 turns none of them on, and `main`'s `nv_defaults.cpp` still has no sm_70 row):
   - **decode +8.6% to +10.5%** on a 32K x 1-slot config;
   - generated text byte-identical to the default path in 12 of 12 requests (3 starts x 4 prompts);
   - under the reproducible recipe, 7 of 8 outputs fall within the reference's own variants. The eighth is a near-tie
     docstring wording flip, of the kind the reference itself shows across restarts.
   - The three extra switches of the sm_60/sm_86 tables (`STRATA_FUSE_WAIT=2`, `STRATA_FUSE_DECODE=1`,
     `STRATA_CPU_IQ_ALL_EXACT1=1`) add nothing on top.
   - This is a single V100-PCIe data point for #3411, which measured +2.8% on one card.
2. **At 262K context, keep `--kv-resident 32768`.**
   - Setup writes it from 64K up when the RAM allows. Our 262K x 2 config had been made by editing setup's 32K
     config (a context and slot bump) instead of re-running setup, and so it lacked the flag.
   - Restoring it: the slot sessions shrink from 3.65 to 0.95 GiB each, and the expert cache grows from 7,478 to
     11,877 slots. Decode +20% to +36% (128K included), prompts unchanged or faster.
   - Raising the context by hand on a 32 GB card silently costs this much.
3. **On this machine a second slot adds no throughput.**
   - With V1 and KV streaming on both, one slot (S1) beats two slots (S2) on single requests, on DGX-style ladder
     aggregate throughput at 1, 2 and 4 concurrent clients, and on two concurrent 128K requests.
   - With two slots, a request decodes at **~7 tok/s while the other request's 128K prompt is read**.
   - The trade is head-of-line blocking: with one slot, a second request waits for the first to finish.

| Swift 1.5 IQ3_S, 262K context, decode tok/s | 4K code | 4K prose | 16K code | 16K prose | 128K code |
|---|---:|---:|---:|---:|---:|
| 2 slots, edited config, no KV streaming (reference) | 56.8 | 41.8 | 51.2 | 44.2 | 51.8 |
| **S2**: 2 slots + V1 + `--kv-resident 32768` + `--vram-reserve-mib 768` | 76.6 (+35%) | 52.3 (+25%) | 74.9 (+46%) | 60.0 (+36%) | 73.5 (+42%) |
| **S1**: S2 with 1 slot | 88.2 (+55%) | 62.7 (+50%) | 93.1 (+82%) | 66.1 (+50%) | 86.6 (+67%) |

Strata's twelve standard-v1 sanity checks pass 12/12 in all three arms (our own runner; see Deviations). The same S1
change on **Qwen3.8-Flash-Next IQ3_S** with the setup-built binary:
- decode 57.7 -> 92.9 (4K code), 44.2 -> 62.4 (4K prose), 54.2 -> 92.0 (16K code), 45.4 -> 65.3 (16K prose);
- sanity 12/12 (`data/production-check-qwen-iq3s-s1*.jsonl`).

## Hardware and software

- **GPU:**
  - Tesla V100-PCIE-32GB (sm_70), VBIOS 88.00.7E.00.03, PCIe Gen3 x16; engine probe 12.3 GB/s host->device;
  - driver 580.178.04.
  - Passive card with its own blower. Max 75-77 C in these runs, with no thermal-violation samples (nvidia-smi dmon
    `tviol`).
- **CPU:**
  - 2x Xeon Gold 6138 (Skylake-SP, 20 cores each, AVX-512F/BW, no VNNI or VBMI, so the expert kernels run AVX2).
  - NUMA on; the V100 sits on node 0.
  - **Capped at 2.7 GHz** (`scaling_max_freq`, turbo 3.7) as a stability workaround for this workstation, which froze
    under sustained full load; fan floors are raised too. Every number here is under the cap.
- **RAM:**
  - 192 GB DDR4-2666, 3 of 6 channels populated per socket (2+1 across the two controllers).
  - STREAM triad: socket 0 local 36.0 GB/s, socket 1 local 27.9, cross 21.7 / 28.5, both sockets interleaved 49.0
    (`data/profile/stream.txt`).
- **OS:** Ubuntu 26.04.1, kernel 7.0.0-42, CUDA 12.8.93 (toolkit from the runfile), g++-14.3.
- **Strata:** `Niko1221/Strata` at **`5e8a4cca`** (0.1.52), source build for sm_70, with setup's own configure line
  (`tools/dev-build.sh`):

  ```text
  cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
    -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=70 \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=g++-14 \
    -DSTRATA_GGML_DIR=third_party/llama.cpp -DSTRATA_EXPERIMENTAL_SM60=ON      # llama.cpp at 3cf03257
  ```

  - The setup-built engine has sha256 `33980ea8...` (BUILD.json 0.1.52, src `b7ff2efbe0819d6b`). The trial arms ran
    a second build of the same source and flags (`5cbede22...`).
  - **Parity of the two builds:** within the noise floor in all 4 cells, and the same set of outputs
    (`data/g13-summary.md`).

## Models

- **Swift 1.5 IQ3_S**: `ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, 2 shards (44,900,816,736 and
  38,842,920,384 bytes; its n-gram table sits at the front of shard 1 and the experts span both shards).
  - Pack from `tools/iq_pack.py`, shared MTP runtime, the shipped `data/expert-profile.bin`.
- **Qwen3.8-Flash-Next GSQ-RCO IQ3_S** for the production check only.
- Configs: `config/`. The reference is `serve-262k-2slot-setup.json`; S2 and S1 are `serve-s2.json` and
  `serve-s1.json`; `bench-32k-1slot.json` is setup's own 32K config.
  - All have `--kv int8 --prefill auto --spec 4 --spec-min-p 0.5 --ple-io ram`, `--expert-cache auto`, and the serve
    configs' reasoning controller (budget 28672, close-retry, loop recovery).
  - `config/g2-arms/` has every one-lever arm.
  - Paths are placeholders.

## Method

- **Decode microbench** (`tools/decode-bench.py`):
  - two fixed prompts (code, prose) at 4K and 16K, plus one 128K code prompt on the serve configs;
  - temperature 0, thinking off, 512 output tokens, streaming, one request at a time;
  - client-side decode tok/s, the engine's own per-request lines, and the output sha256 per request.
  - Every repeat is a fresh engine start. Arms are interleaved, the second repeat in reverse order. A throwaway start
    runs first (the first starts after a phase change ran 2-4% slow).
  - Noise floor: the max-min range over repeats. 0.1-3.5% per cell for the stack arms; up to 7% for some one-lever
    serve arms, whose first repeat ran slow.
- **Sanity** (`tools/sanity12.py`):
  - the twelve `bench/prompts/standard-v1/requests/sanity-*.json` fixtures, scored as `docs/BENCHMARK_PROTOCOL.md`
    describes (exact text, strict JSON, one structured tool call);
  - thinking on with the bundle's template kwargs, cap 2048, seed 1.
- **Ladder** (`tools/v41bench-qwen.py`): a fixed 8-category prompt set at concurrency 1, 2 and 4, thinking off,
  aggregate and per-stream tok/s, TTFT. The reference ladder was run on the setup-built binary
  (`data/ladder/bench-v100host-swift-iq3_s.*`).
- **Two concurrent 128K requests** (`tools/conc2.py`).
- **Profile**:
  - the prof build (`-lineinfo`, frame pointers) with `STRATA_VERIFY_PROFILE=1 STRATA_DECODE_TIMING=1`;
  - perf at 499 Hz on the engine, Intel PCM memory bandwidth, one nsys capture with CUDA-graph node tracing
    (`data/profile/`).
- **Runners:** `tools/g13-run.sh`, `g2-run.sh`, `g4-run.sh`, as run, with paths replaced by placeholders: they will
  not run unedited.

## Results in detail

**Volta stack (V1), 32K x 1, 3 starts each** (`data/g2-flags.jsonl`, `data/summary-vs-bench-ref.md`):

| | 4K code | 4K prose | 16K code | 16K prose |
|---|---:|---:|---:|---:|
| default | 81.0 | 56.5 | 84.8 | 59.4 |
| V1 (13 switches) | 88.8 | 61.4 | 93.7 | 65.2 |
| range (MDE) | 1.3% | 0.2% | 0.4% | 0.4% |

Prompt +2.3% to +3.1%. On the 262K x 2 config without KV streaming, V1 gives +4.6% to +6.5% (medians), under that
shape's 2.4-4.7% noise. There the CPU's experts dominate, and only 1 of 4 cells clears the noise.

**One lever at a time on the 262K x 2 config, 3 starts each** (`data/summary-vs-serve-ref.md`):

| Arm | Decode change | Prompt change | Outputs vs reference |
|---|---|---|---|
| `--kv-resident 32768` | +20% to +36% | -0.5% to +13% | differ (more experts on the GPU, which rounds differently) |
| one slot (`parallel: 1`) | +32% to +57% | +2.6% to +12.5% | differ |
| V1 | +4.6% to +6.5% | +0.6% to +2.3% | identical |
| `STRATA_ADAPT_NOWAIT=1 STRATA_ADAPT_LAG=2` | +1.7% to +6.8% (1 of 4 cells above noise) | 0 | differ |
| `--pool-cores 1-19` (socket 0 only, 19 workers instead of 32 on both) | +0.5% to +1.9% (noise) | 0 | identical |
| `--prefill auto:32768` | decode -1.6% to +2.6% | +4% to +9% at 16K and 128K | differ past 8K (not bit-exact on CUDA) |

The 32K-chunk arm is the only one that hit the card's 250 W power limit (156 one-second samples with `pviol`, up to
322 W, all during its prompts).

**Ladder, aggregate / per-stream tok/s** (`data/ladder/`):

| Concurrency | reference (2 slots, setup-built binary) | S2 (2 slots) | S1 (1 slot) |
|---|---|---|---|
| 1 | 30.2 / 40.8 | 42.5 / 58.1 | 58.1 / 81.6 |
| 2 | 31.0 / 20.8 | 38.8 / 26.0 | 60.1 / 86.8 |
| 4 | 34.0 / 22.3 | 42.0 / 26.7 | 63.0 / 92.1 |

Mean TTFT at 2 / 4 clients: reference 2.6 / 7.7 s, S2 2.0 / 6.0 s, S1 2.4 / 5.1 s.

**Two 128K requests sent together** (`data/stacks-conc2.jsonl`, one run per arm):

| | wall | first request | second request |
|---|---:|---|---|
| reference | 244.6 s | TTFT 99 s, decode 7.2 tok/s | TTFT 224 s, decode 25.6 |
| S2 | 238.9 s | TTFT 98 s, decode 7.3 tok/s | TTFT 222 s, decode 31.2 |
| S1 | 210.5 s | TTFT 96 s, decode 88.4 | queued; TTFT 202 s, decode 58.8 |

## Where a decode window goes (profile arms, `data/profile/decode-timing.txt`)

| ms per window | 32K x 1, 4K code | 262K x 2 (no KV streaming), 16K code |
|---|---:|---:|
| window (tokens per window) | 40.4 (3.07) | 60.6 (3.16) |
| GPU work | 24.8 | 25.2 |
| host waiting on CPU experts | 6.0 | 23.4 |
| MTP draft | 3.1 | 3.7 |
| adaptive-tier wait | 2.0 | 3.0 |

- On the 262K x 2 config, the extra time is the CPU experts, and the GPU stalls on them for 20 ms of the window.
- The CPU pool is not memory-bound:
  - perf puts 67-83% of its samples in `ExpertPool::worker`'s wait loop and 8-22% in llama.cpp's AVX2 dot
    products;
  - PCM shows 3.5-7.5 GB/s on socket 0 during decode, against STREAM's 36.
- In the nsys capture, `wait_flag_ge_kernel` (the GPU spinning on host flags) takes 4.4% of kernel time over 24.6K
  launches; the hyper-connection kernels take 7.7%.

## Deviations from the standard protocol, and limits

- This is **not** `strata bench run`, which is not in 0.1.52. The sanity runner is our own, over the frozen
  standard-v1 fixtures, at **temperature 0** instead of the model card's sampling, with the serve configs' reasoning
  controller on.
- **2.7 GHz CPU cap** (see Hardware); the clock effect was not measured. One machine, greedy decoding, synthetic
  filler prompts at 4K / 16K / 128K.
- Medians of 2 starts for the stack arms and 3 for the rest; one start for each production check; one concurrency run
  per arm.
- The reproducible recipe (`STRATA_IQ_MT_MIN=1 --prompt-cache 0 --adapt-swaps 0 --pcie-frac 0`) was byte-exact in 11
  of 12 outputs. The twelfth flipped one near-tie phrase at character 1,181 of 2,711, then ran on unchanged.
- With `--vram-reserve-mib 768` the engine starts with 554 MiB free (490 without). Its "below 512 MiB" warning still
  appears at some request starts.
- nsys 2024.6 rejects `--sample` / `--cpuctxsw` on `nsys launch` (they belong on `nsys start`). The wrapper in
  `tools/nsys-wrap.sh` has the working form.
- Measured on 0.1.52; `main` has moved on (no sm_70 row in `nv_defaults.cpp` there either).

## Files

- `config/`: the configs above and every one-lever arm.
- `data/`: raw rows (`*.jsonl`), per-cell summaries, ladder JSON and markdown, sanity results, the production checks,
  the profile (perf tops, PCM CSV, STREAM, nsys kernel summary, decode-timing and GPU-stage lines).
- `logs/`: start facts per arm and one engine-log excerpt per main arm.
- `tools/`: the harness as run (paths are placeholders).
