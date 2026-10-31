# Community benchmark on 3x Tesla V100-SXM2-16GB (sm_70, no NVLink)

Measured on 2026-10-29 by reaganovo. Three 16 GB Volta cards, PCIe 3.0, no NVLink, on an AVX1-only
host, running one model with `--layer-split 18,35`.

Main limitation: one machine, one model, one ~20K-token prompt, greedy decoding, and no standard
protocol run (`strata bench run` was not used; the machine serves from a source build and the
protocol is still a draft). The window breakdown, CUPTI kernel census and the two negative A/B
results are extra evidence beyond the throughput table.

## Hardware and software

- **GPU:** 3x Tesla V100-SXM2-16GB (`sm_70`), 16,384 MiB each, PCIe Gen3 **x16**. **No NVLink**:
  `nvidia-smi topo -m` gives PHB/PHB/SYS and every link reports inactive; GPU2 sits on the other
  NUMA node (across QPI), GPU0/GPU1 share node 0.
- **CPU:** 2x Xeon E5-2697 v2 (12 cores each), **AVX1 only** - no AVX2, no AVX-512, so the AVX1
  IQ kernels are the only path.
- **RAM:** 503 GiB DDR3. **Storage:** OS and model on one SATA SSD (Colorful SL500 1.8 TB); no
  network or HDD in the model path.
- **OS / driver:** Linux, NVIDIA driver 580.178.04, CUDA 12.8.
- **Strata:** engine `0.1.53`, source build, sha256 `329eb0fe...` (upstream v0.2.7.1).
- **Power:** 180 W per card cap (default 300 W), unchanged in this run. No other GPU workload:
  the machine serves one model through its service unit.
- Static environment captured in `data/env.txt`.

## Model and configuration

- **Model:** Qwen3.8-Flash-Next, GSQ-RCO **IQ3_S** (46.84 GiB of experts), two GGUF shards
  (`00001-of-00002`, `00002-of-00002`), native expert pack, MTP drafter.
- **Vision:** on (loaded, not exercised). **Conversation cache:** 98,304 MiB, 12 slots.
- **Expert profile:** shipped profile for load; learned profile saved every 10 turns.
- **Context:** 262,144. **KV:** int8, `--kv-resident 32768`. **Prefill:** 8,192.
  **Speculative:** `--spec 4 --spec-min-p 0.5`. **PLE:** RAM (the 26.8 GiB table does not fit mlock).
- **Sampling:** temperature 0 (greedy), no reasoning-effort override.
- **Calibration / experimental speed projection:** not used.

```text
strata --serve --pack <packs/iq3s> \
  --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --mtp <mtp/rt> \
  --spec 4 --spec-min-p 0.5 --max-context 262144 --vision --kv int8 --kv-resident 32768 \
  --trim-stage-weights --expert-profile expert-profile.bin \
  --vram-reserve-mib 1024 --vram-reserve-later-mib 1024 \
  --gpu 0,1,2 --layer-split 18,35 --prefill 8192 \
  --conversation-cache-mib 98304 --conversation-cache-slots 12
```

## Method

- **Prompts.** The tuning harness's synthetic ~20K-token prompt (five coding-task topics, each
  repeated 1,250 times). Sent three times back to back through `/v1/chat/completions` with a
  256-token cap. Run 1 processes the prompt fresh (20,040 prompt tokens); runs 2 and 3 reuse the
  conversation prefix and report 5 new prompt tokens. Prompt reuse is reported, not hidden.
- **Timing.** The engine's own `STRATA_DECODE_TIMING=1` lines. Model loading is not included; the
  engine was already serving. No `nvidia-smi` was run during inference (it perturbs the layer-split
  path on this machine).
- **TTFT and total latency:** not measured (no streaming client on this box).
- **Kernel census.** Separate decode-only run (104-token prompt, 512-token cap) under CUPTI with
  `CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL` only; durations are aggregated per kernel over that run's
  156 windows.
- **A/B.** Two engine restarts differing in one env-gated grid width for `gr_up`, same prompt, with
  the engine's own bit-identical output check between arms.
- **Failures.** No failed requests in the three measured runs. One A/B arm failed to start because
  port 8095 was still held by the previous arm's engine; the arm was restarted and re-run. The log
  is `data/ab/ab-gr-up.log`.

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s (median, range) | Decode tok/s (median, range) | TTFT s (median, range) |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 20K prompt, fresh | 20,040 | 0 | ~256 | 1 | not measured | 100.1 (1 run) | not measured |
| 20K prompt, prefix reused | 5 | 20,035 | ~256 | 2 | not measured | 109.5 / 108.0 (median 108.8) | not measured |

Per-run rows are in `data/warm-runs.jsonl`; the verbatim engine lines are in
`data/probes/warm-3-rounds.log`. (The upstream issue summary quotes 3.18 tokens/window; the per-run
logs in this folder give 2.94-3.16 depending on the run.)

### Window breakdown (warm runs)

| Run | Window ms | verify ms (GPU-reach wait) | per-layer host ms | stage ms | draft ms | commit/emit ms | Tokens/window |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 (fresh after load) | 31.57 | 27.79 (7.41) | 2.34 | 0.13 | 2.44 | 0.46 | 3.16 |
| 2 | 27.50 | 24.55 (7.99) | 0.73 | 0.12 | 2.40 | 0.42 | 3.01 |
| 3 | 27.24 | 24.42 (7.95) | 0.78 | 0.12 | 2.35 | 0.42 | 2.94 |

The window is verify-dominated. The three cards run their stage one after another, so **each card is
idle about two thirds of the window**; that is the structural limit this hardware hits, not a kernel
inefficiency.

Expert routing is almost entirely resident: **99.7% expert-cache hit (299,954 hits / 300,722
lookups)** on a 501-token decode, and **99.81% of 12,038,452 KV block reads hit VRAM** (91.3 MiB
read from RAM). Raw engine-log block in `data/cache-hit.txt`.

Per-card utilization sampled with nvidia-smi (200 ms) during decode was **34-39%**; that number is
approximate and was not archived (see Correctness and limitations).

### Kernel census (CUPTI, `CONCURRENT_KERNEL` only)

263 kernels, **5,362.59 ms of GPU time over 156 windows = 34.38 ms per window** summed across the
three cards; the top 15 kernels are **65.77%** of that.

| # | Kernel | Calls/window | ms/window | Share |
| ---: | --- | ---: | ---: | ---: |
| 1 | `wait_flag_ge_kernel` | 144.9 | 3.538 | 10.29% |
| 2 | `native_mmvq_il_kernel` (Q6K, 4x2) | 53.1 | 2.718 | 7.91% |
| 3 | `gr_down_staged_kernel` | 106.6 | 2.286 | 6.65% |
| 4 | `gr_up_fast_kernel` | 107.2 | 2.145 | 6.24% |
| 5 | `native_down_lds_kernel` | 39.3 | 2.010 | 5.85% |
| 6 | `fetch_blobs_kernel` | 48.3 | 1.495 | 4.35% |
| 7 | `native_gu_fused_kernel` (22) | 20.1 | 1.363 | 3.97% |
| 8 | `native_gu_fused_kernel` (18) | 17.1 | 1.165 | 3.39% |
| 9 | `bf16_f32_mmvf_multi_kernel` | 107.9 | 1.128 | 3.28% |
| 10 | `native_mmvq_il_kernel` (Q6K, 4x4) | 19.5 | 1.073 | 3.12% |

Full top-15 table: `data/cupti/kernel-top.csv`; the unfiltered 263-kernel table is
`data/cupti/C1.tsv`.

**Measurement caveat:** enabling `CUPTI_ACTIVITY_KIND_KERNEL` next to
`CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL` serializes kernels and inflates durations. In the discarded
run a `<<<1,128>>>` copy kernel was recorded as 193 us and looked like the top hotspot; the
concurrent-only number is 6.7 us. Only `CONCURRENT_KERNEL` gives usable per-kernel times here.

### Negative results

- **Widened `gr_up` grid** (320 or 1280 blocks instead of 160, env-gated, bit-identical output):
  old 93.10 / new default 91.90 / wide-8 93.40 / wide-2 93.40 tok/s. A same-code rebuild alone
  differed 1.3% between runs, so **the noise floor is about +-1.5%** and the wide grid is inside it.
  (This A/B ran on a separate ~93 tok/s arm set, not the 108-110 set above; it compares arms only.)
- **Batching the `copy_i32_from_mapped` calls:** 0.40 ms/window in total, below the same floor.

## Correctness and limitations

- The `gr_up` A/B checked that both arms produce the same tokens with the engine's own
  bit-identical check; token hashes were not recorded.
- No quality, needle, tool-call or concurrency checks were run.
- One prompt length (about 20K), none of the 4K / 32K / 128K cells; one model.
- VRAM use and per-card utilization were sampled live during tuning but are **not archived**; the
  CUPTI census is the precise per-card view in this folder. Expect the 34-39% figure to move a few
  points with sampling method.
- This is **not a standard protocol run**; treat the numbers as a report made with today's tools, as
  the guide defines it.

## Data in this folder

| File | What it backs |
| --- | --- |
| `data/env.txt` | GPU, CPU, RAM, driver, topology, engine build |
| `data/warm-runs.jsonl` | the three warm decode rows |
| `data/probes/warm-3-rounds.log` | verbatim probe output: 3 decode rates and 3 `decode timing` lines |
| `data/probes/probe-command.log` | the probe shell line as issued |
| `data/probes/cupti-run-1714.log`, `data/probes/census-run-*.log` | run boundaries, restore, health checks |
| `data/cupti/C1.tsv` | full 263-kernel CUPTI table |
| `data/cupti/kernel-top.csv` | top-15 derived from it (per-window, share) |
| `data/cupti/engine-timing.txt`, `data/cupti/response.json` | the instrumented run's window lines and response |
| `data/census/summary.txt` | CPU-side launch census (291.4 launches/window) |
| `data/cache-hit.txt` | expert-cache and KV-streaming lines for the 300,722-lookup request |
| `data/ab/ab-gr-up*.log` | the widened-grid A/B, including the failed start and the bit-identical check |

Large model files, the engine binary and the mixed engine log are not included; the extracted blocks
above are everything used for the numbers. Local paths in the logs were replaced with placeholders.

## Reproduce

```text
STRATA_DECODE_TIMING=1  (already set for the service that produced the logs)
send the same ~20K-token prompt three times, greedy, through /v1/chat/completions
grep 'decode timing' <engine log> | tail -3
```

CUPTI: run the engine under CUPTI with `CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL` only, decode 512
tokens, then aggregate `gpu_ms` per `name` over the run's window count.

## Provenance

One machine, one model, greedy decoding, three warm runs plus one instrumented run. Discussed
upstream in [Niko1221/Strata#4593](https://github.com/Niko1221/Strata/issues/4593); this folder is
the archived form, laid out so it can be dropped into `bench/results/` unchanged.
