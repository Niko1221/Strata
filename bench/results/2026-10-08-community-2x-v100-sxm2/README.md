# Community benchmark: 2x Tesla V100-SXM2-32GB (layer split), plus a context ladder to 1M and concurrency

Measured on 2026-10-08 by caolonghao. **First measurement of a layer split over two V100**
([NVIDIA_V100.md](../../../docs/NVIDIA_V100.md) lists it as unmeasured), on one machine, one operator.
The same PC's 2x RTX 2080 Ti 22 GB pair is a separate report next to this one. Main limitations:
single machine, no needle / HumanEval checks, and the shared NVMe the model lives on had other
load (load times varied 4x between starts).

## Hardware and software

- **GPUs:** 2x Tesla V100-SXM2-32GB (sm_70), 300 W each, PCIe Gen3 x16 (engine probe: 13.1 GB/s
  host->device on both cards). The PC also holds the 2080 Ti pair of the other report; they were
  idle here.
- **CPU and RAM:** 2x Xeon Silver 4316 (2.30 GHz, 80 threads, AVX-512), 125 GB DDR4 ECC.
- **Storage:** a shared NVMe volume (other load present; sequential reads measured anywhere from
  50 MB/s to 1.5 GB/s during the session).
- **OS, driver, toolkit:** Ubuntu 22.04.5 (kernel 6.8.0-138), NVIDIA driver 580.178.04, CUDA 12.8
  toolkit (`/usr/local/cuda-12.8`, gcc 11.4).
- **Strata:** upstream `main` at `e8ca9af`, engine 0.1.40.2, **source build** (no ready-made Linux
  engine): `engine-cuda12/` with `-DCMAKE_CUDA_ARCHITECTURES=70 -DSTRATA_EXPERIMENTAL_SM60=ON`
  and the CUDA 12.8 toolkit - the NVIDIA_V100.md community build.
- **Power limits:** defaults (300 W); no clock pinning. Background workloads: none on the GPUs.

## Model and configuration

- **Model:** Unsloth `UD-IQ4_XS` installed by setup (`--family unsloth`), pinned revision
  `38bb39ee`, three GGUF shards (93.7 GB), pack built by setup with `--compat-bf16`.
- **Settings:** context 131,072; KV int8 with `--kv-resident 32768`; `--expert-cache auto`;
  `--prefill auto`; MTP `--spec 4 --spec-min-p 0.5`; `--remote-expert-opt`; `--layer-split auto`
  (the engine chose **K=25**, caches holding 22,473 of 24,576 profiled pairs, ~99.8% of the routed
  mass, best of 46 placements); greedy, reasoning effort none, vision off. No calibration, no
  speed projection.

```text
./setup.sh --yes --family unsloth --model UD-IQ4_XS --gpus 0,1 --context 131072 --data-dir <nvme> --no-start
./run-unsloth-ud-iq4_xs.sh    # engine args as logged in strata-unsloth-ud-iq4_xs.log
```

## Method

The community `benchmark.py` from [2026-09-30-community-rtx-5090](../2026-09-30-community-rtx-5090/),
unmodified: deterministic synthetic Python source, a distinct nonce per filler so **every prompt is
fresh** (the engine logged `reused: 0` on all measured requests), 256-token output cap, streaming,
client-side TTFT plus the engine's own `prompt_ms` / `decode_ms`. One warm-up excluded; three
measured runs per cell; one loaded server per configuration. Loading is excluded from all throughputs.

## Results: the main sweep (4K / 32K)

| Prompt tokens | Reused | Generated | Runs | Prompt tok/s median (range) | Decode tok/s median (range) | TTFT s median (range) |
| ---: | ---: | ---: | ---: | --- | --- | --- |
| 4,096 | 0 | 256 | 3 | 1,277.8 (1,214.6-1,279.0) | 77.3 (74.9-80.3) | 3.24 (3.24-3.41) |
| 32,768 | 0 | 256 | 3 | 2,355.3 (2,351.8-2,357.4) | 80.3 (79.8-86.6) | 14.01 (14.01-14.03) |

- Decode expert-cache hit rate 0.992-0.999; MTP draft acceptance 0.63-0.67; ~0 routed experts went
  over PCIe during decode (the two caches hold ~99.8% of the profiled pairs).
- Against the **single** V100-SXM2 UD-IQ4_XS point in #902 (decode 26.3, prefill 665 at 8K): the
  split's 64 GB holds roughly 2.2x the experts of one card, and decode measured ~3x.
- Warm restart of the server: 81-275 s. **First cold start exceeded the server's 900 s READY
  watchdog** while the expert arena streamed from the busy shared NVMe;
  `STRATA_ENGINE_READY_S=3600` is the workaround we used. Worth knowing on slow storage.

## The context ladder (auto layer split, original config otherwise)

Fresh near-full prompts per rung. 262K/384K/512K/1M rungs were restarted servers, each with the
engine's own rope line confirming `yarn` and the factor. The 131K row's 128K-point was measured
under `--prefill auto:32768` (below); its 4K/32K cells are the original config and match the main
sweep. The harness cannot synthesize beyond ~455,846 prompt tokens, so the 512K and 1M rungs both
use a 450,000-token prompt - the pair isolates the pure context cost.

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
- KV streaming block hit rate at depth: 92.3-93.9% of reads hit VRAM (243 MiB read from RAM per
  450K-prompt request).
- Rope quality past the trained 262,144 was **not** checked (no needle runs at 384K+).

## Concurrency (131,072 context; a separate, lighter prompt set)

A small script (`data/concurrency/concurrency_bench.py`): C threads, each an ~180-token essay
prompt, 256-token answer, greedy, streaming, client-side timing; median of 3 rounds. **These
per-request numbers are not comparable with the Python-source rows above** (different prompts).
Only medians were retained.

| Setting | C=1 tok/s | C=2 per / total | C=4 per / total | C=8 per / total |
| --- | ---: | ---: | ---: | ---: |
| one at a time (default) | 65.2 | 50.3 / 66.8 (TTFT 2.4 s) | 28.0 / 66.0 (6.2 s) | - |
| `"parallel": 2` | 62.1 | 32.8 / 65.3 (1.3 s) | 23.5 / 65.7 (4.8 s) | - |
| `--batch 8 --batch-groups 2 --trim-stage-weights`, split 25 | 51.5 | 19.1 / 38.2 (1.9 s) | 17.6 / 70.2 (3.1 s) | 14.9 / **117.7** (7.8 s) |

- Queueing keeps total throughput flat (~66); its cost is TTFT growth.
- `parallel 2` did not add throughput here (batch windows carry no MTP drafts; the solo path's
  ~2.6 tokens/window beat a 2-row window), it only improved TTFT.
- The 8-slot two-GPU pipeline reached **117.7 tok/s total (+78%)** with every request at 14.9;
  a lone request on that config paid 51.5 (the slots' sessions come out of the caches).
- The C=2 dip of the batched config (38.2 total) is reported as measured; not re-run to explain.

## `--prefill auto:32768` was slower here (tried, then reverted)

The +21-35% tip from #433/#440/#834 (single-card, 96 GB PCs) measured **-32% at 32K on this
two-card split** (2,355 -> 1,606 tok/s): the 32K chunk made a 32K prompt one block and lost the
cross-card chunk pipelining. 4K and decode were unchanged; 128K read 2,191 tok/s. Full data in
`data/prefill-auto32768/`. `--pool-workers 19` (the 2-socket tip) showed no clear decode change
inside the run-to-run noise.

## Correctness and limitations

- Every measured request completed; no engine errors or restarts during the runs. Answers were
  coherent Python commentary, not independently graded; **no needle / recall / HumanEval runs**.
- One machine, one operator, shared-storage variance (the same config loaded in 81 s and 275 s on
  different starts). The 131K/128K ladder row was measured with the prefill flag above; flagged
  wherever cited.
- Not measured: vision, sampled decoding, UD-Q4_K_XL, other quants, per-card PCIe saturation,
  longer-than-one-hour stability.

## Files

- `data/main/` - the 4K/32K sweep (per-run JSON + the harness log)
- `data/context-262k|384k|512k|1m/` - one summary + aggregate per rung (per-request files with the
  ~0.5 MB prompts were dropped for size)
- `data/prefill-auto32768/` - the prefill-tip counter-example
- `data/concurrency/` - the script; medians in the table above

Engine log: `strata-unsloth-ud-iq4_xs.log` next to the server (not shipped; the startup, split
search and KV lines are quoted above). Prepared with an AI assistant (Claude Code) from the raw
measurements; the numbers are the engine's and the client's own output, unedited.
