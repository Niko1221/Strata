# Community benchmark: 2x Tesla V100-PCIE-32GB (layer split), a context ladder to 1M, concurrency

Measured on 2026-10-08 by caolonghao. The **first measured layer split over two V100**
([NVIDIA_V100.md](../../../docs/NVIDIA_V100.md) lists it as unmeasured; a V100 + RTX 4070 split
exists in #850 - this is the first with two V100), on **PCIe cards with no NVLink**: the whole run
rides PCIe Gen3 x16 and pinned host RAM. Single machine, one operator; no needle or correctness
checks; the shared NVMe holding the model had other load (warm restarts took 81-275 s, ~3.4x apart).

## Hardware and software

- **GPUs:** 2x Tesla V100-PCIE-32GB (GV100, sm_70), PCIe Gen3 x16 each, **PCI device ID
  `10DE:1DB5`**, **no NVLink** (`nvidia-smi nvlink -s`: all links inactive). The driver names them
  "Tesla V100-SXM2-32GB" - a VBIOS label quirk also seen in NVIDIA forum reports of PCIe V100s;
  the device ID and the inactive NVLink are the ground truth. Driver power limit 300 W as reported
  (the PCIe SKU ships at 250 W); no clock pinning.
- Background GPU load: none - the PC's 2x RTX 2080 Ti, the subject of the sibling report, were idle.
- **CPU and RAM:** 2x Xeon Silver 4316 (2.30 GHz, 80 threads, AVX-512), 125 GB DDR4 ECC.
- **Storage:** a shared NVMe volume (other load present; sequential reads measured 50 MB/s to
  1.5 GB/s during the session).
- **OS, driver, toolkit:** Ubuntu 22.04.5 (kernel 6.8.0-138), NVIDIA 580.178.04, CUDA 12.8 toolkit
  (`/usr/local/cuda-12.8`, gcc 11.4).
- **Strata:** upstream `main` at `e8ca9af`, engine 0.1.40.2, **source build** (no ready-made Linux
  engine) into `engine-cuda12/` with `-DCMAKE_CUDA_ARCHITECTURES=70
  -DSTRATA_EXPERIMENTAL_SM60=ON` - the NVIDIA_V100.md community build. The engine's PCIe probe
  measured 13.1 GB/s host->device on both cards.

## Model and configuration

- **Model:** `unsloth/Qwen3.8-Flash-Next-GGUF` at setup's pinned revision `38bb39ee`, **UD-IQ4_XS**;
  shards `Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf` .. `-00003-of-00003.gguf` (93.7 GB);
  pack built by setup (`--compat-bf16`); setup's own MTP draft layer and expert profile; no custom
  draft vocabulary; vision off.
- **Settings:** context 131,072; KV int8 with `--kv-resident 32768`; `--expert-cache auto`;
  `--prefill auto`; MTP `--spec 4 --spec-min-p 0.5`; `--remote-expert-opt`; `--layer-split auto`
  (the engine chose **K=25**; the two caches hold 22,473 of 24,576 profiled pairs, ~99.8% of the
  routed mass, best of 46 placements); **no low-RAM mode**; CPU expert pool at its **default**
  worker count (no `--pool-workers`); no calibration, no speed projection; greedy, reasoning
  effort none.

```text
./setup.sh --yes --family unsloth --model UD-IQ4_XS --gpus 0,1 --context 131072 --data-dir <nvme> --no-start
engine-cuda12/strata --serve --pack <data>/packs/unsloth-ud-iq4_xs \
  --native <data>/models/unsloth-UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
  --expert-profile data/expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 \
  --mtp <data>/mtp/rt --max-context 131072 --kv int8 --kv-resident 32768 --remote-expert-opt --layer-split auto
```

## Method

The community `benchmark.py` from [2026-09-30-community-rtx-5090](../2026-09-30-community-rtx-5090/),
unmodified: deterministic synthetic Python source, a distinct nonce per filler so **every prompt is
fresh** (the engine logged `reused: 0` on all measured requests), 256-token output cap, streaming;
client-side TTFT (the first streamed token is answer text - reasoning effort is none); the engine's
own `prompt_ms` / `decode_ms` for throughput; loading excluded; one warm-up excluded; one loaded
server per configuration; three measured runs per cell.

```text
python benchmark.py --url http://127.0.0.1:8080 --pack <data>/packs/unsloth-ud-iq4_xs \
  --targets 4096,32768 --runs 3 --out <dir>          # main sweep
... --targets 260000|390000|450000 --runs 3 ...      # the ladder rungs (below)
```

File sizes are decimal GB; the engine's cache and KV figures are GiB/MiB as it logs them.

## Results: the main sweep (4K / 32K)

| Prompt tokens | Reused | Generated | Runs | Prompt tok/s median (range) | Decode tok/s median (range) | TTFT s median (range) |
| ---: | ---: | ---: | ---: | --- | --- | --- |
| 4,096 | 0 | 256 | 3 | 1,277.8 (1,214.6-1,279.0) | 77.3 (74.9-80.3) | 3.24 (3.24-3.41) |
| 32,768 | 0 | 256 | 3 | 2,355.3 (2,351.8-2,357.4) | 80.3 (79.8-86.6) | 14.01 (14.00-14.03) |

- Decode expert-cache hit rate 0.992-0.999; MTP draft acceptance 0.59-0.79 per run (cell medians
  0.63 / 0.68); ~0 routed experts went over PCIe during decode.
- Against the **single** V100-SXM2 UD-IQ4_XS point in #902 (decode 26.3, prefill 665 at 8K; a
  different host, OS, engine version and KV type, one run per config there): the split's 64 GB
  holds ~2.2x the experts of one card, and decode measured ~3x.
- Warm restarts: 81-275 s. **A first cold start exceeded the server's 900 s READY watchdog** while
  the expert arena streamed from the busy shared NVMe; `STRATA_ENGINE_READY_S=3600` is the
  workaround. Worth knowing on slow storage.
- System RAM after load: ~62 GB used (a `free -g` startup snapshot; the peak during inference was
  not sampled); every request's client elapsed time is in its `summary.json`.

## The context ladder (auto layer split, original config otherwise)

Three measured runs per rung (per-run medians and ranges in each `data/context-*/results.json/
summary.json`); each rung is a restarted server, with the engine's own rope line confirming `yarn`
and the factor. The 131K row's 128,000-token point was measured under `--prefill auto:32768` (see
below); the main sweep's 4K/32K cells at the same 131,072 context are the original config. The
harness cannot synthesize beyond ~455,846 prompt tokens, so the 512K and 1M rungs both use a
450,000-token prompt - that pair isolates the pure context cost.

| Context | Rope | Prompt tokens | Prompt tok/s | Decode tok/s | TTFT s | Cache coverage | KV in pinned RAM |
| ---: | --- | ---: | ---: | ---: | ---: | --- | ---: |
| 131,072 | - | 128,000 | 2,191.0 | 81.2 | 58.8 | 99.8% (22,473 pairs) | 0.77 GiB |
| 262,144 | - | 260,000 | 1,979.1 | 77.5 | 131.9 | 99.0% (17,097) | 1.55 GiB |
| 393,216 | yarn 1.5 | 390,000 | 1,705.7 | 73.9 | 229.8 | 99.7% (22,249) | 2.32 GiB |
| 524,288 | yarn 2.0 | 450,000 | 1,627.1 | 74.8 | 277.7 | 99.7% (22,140) | 3.09 GiB |
| 1,048,576 | yarn 4.0 | 450,000 | 1,638.6 | 70.6 | 276.0 | 99.7% (21,690) | 6.19 GiB |

- **No VRAM wall up to 1M**: KV streaming keeps 32K cells per QSA layer in VRAM and the rest in
  pinned RAM; at 1M the expert caches still held 99.7% of the profiled pairs and both cards ran full
  (32 GB used each). The cost of 512K -> 1M at the same 450K prompt was **decode -5.6%**
  (74.8 -> 70.6) with prompt speed unchanged.
- The 262K rung cached ~5,000 fewer pairs (17,097) than the 384K rung above it (22,249); unexplained,
  not re-run.
- KV streaming block hit rate at depth: 92.3-93.9% of reads hit VRAM (243 MiB read from RAM per
  450K-prompt request).
- Rope quality past the trained 262,144 was **not** checked (no needle runs at 384K+).

## Concurrency (131,072 context; a separate, lighter workload)

The script ships in `data/concurrency/` (`concurrency_bench.py <url> <clients> <rounds>`): C
threads, each a short (~60-token) essay instruction, 256-token answers, greedy, streaming, medians
of 3 rounds (per-round data not retained). Its per-request tok/s divides tokens by the **whole
client elapsed time** - prompt, queueing and decode together - so it is a latency-flavoured
aggregate, not the engine's decode rate, and not comparable with the tables above or with the
engine decode of ~77-80. C=8 was not run on the default and `parallel 2` rows.

| Setting | C=1 tok/s | C=2 per / total | C=4 per / total | C=8 per / total |
| --- | ---: | ---: | ---: | ---: |
| one at a time (default) | 65.2 | 50.3 / 66.8 (TTFT 2.4 s) | 28.0 / 66.0 (6.2 s) | not run |
| `"parallel": 2` | 62.1 | 32.8 / 65.3 (1.3 s) | 23.5 / 65.7 (4.8 s) | not run |
| `--batch 8 --batch-groups 2 --trim-stage-weights`, split 25 | 51.5 | 19.1 / 38.2 (1.9 s) | 17.6 / 70.2 (3.1 s) | 14.9 / **117.7** (7.8 s) |

- Queueing keeps total throughput flat (~66); its cost is TTFT growth.
- `parallel 2` did not add throughput here (batch windows carry no MTP drafts; the solo path's
  ~2.6 tokens/window beat a 2-row window), it only improved TTFT.
- The 8-slot two-GPU pipeline reached **117.7 tok/s total (+78%)** with every request at 14.9;
  a lone request on that config likely paid 51.5 (the slots' sessions come out of the caches).
- The C=2 dip of the batched config (38.2 total) is reported as measured; not re-run to explain.

## `--prefill auto:32768` was slower here (tried, then reverted)

The +21-35% tip from #433 / #440 (single-card PCs) measured **-32% at 32K on this two-card split**
(2,355 -> 1,606 tok/s): the 32K chunk made a 32K prompt one block and lost the cross-card chunk
pipelining. #834 measured the same reversal on a 32 GB RTX 4090. 4K and decode were unchanged here;
128K read 2,191 tok/s. Full aggregates in `data/prefill-auto32768/`. `--pool-workers 19` (the
2-socket tip) showed no clear decode change inside the run-to-run noise.

## Correctness and limitations

- Every measured request completed; no engine errors or restarts during the runs. Answers were
  coherent Python commentary, not independently graded; **no needle / recall / HumanEval runs**.
- One machine, one operator, shared-storage variance (the same config restarted in 81 s and 275 s).
  The 131K/128K ladder row was measured with the prefill flag above; flagged wherever cited.
- Not measured: vision, sampled decoding, other quants, per-card PCIe saturation, peak RAM/VRAM,
  longer-than-one-hour stability.

## Files

- `data/main/` - the 4K/32K sweep: the 4K cell's per-run request records plus `summary.json` /
  `results.json` covering all six runs.
- `data/context-262k|384k|512k|1m/` - one summary + aggregate per rung.
- `data/prefill-auto32768/` - aggregates of the counter-example runs.
- `data/concurrency/` - the script; medians in the table above.

Not shipped, per the repository's trimming practice: the harness's raw per-run dumps and its
progress log (`*.log` is gitignored; the 32K cell's ~63 KB per-request files and the ladder rungs'
~0.5 MB ones were dropped for size - the aggregates carry every number above), and the engine log
(`strata-unsloth-ud-iq4_xs.log`; its startup, split-search and KV lines are quoted above).

Prepared with an AI assistant (Claude Code) from the raw measurements; the numbers are the engine's
and the client's own output, unedited.
