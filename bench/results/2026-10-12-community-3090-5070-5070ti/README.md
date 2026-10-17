# Community benchmark: RTX 3090 + RTX 5070 + RTX 5070 Ti (layer split, no 5090)

Measured on 2026-10-12. This tests Strata 0.1.42
with a custom Q2_0 native pack on a three-GPU layer split, 260,000-token context,
on a machine whose only large GPU is a 24 GB RTX 3090. The main limitation is
expert residency: the three cards together can hold 99% of the profiled expert
pairs, and the 12 GB RTX 5070 stage is the binding constraint.

Median prompt throughput was **3,275 tok/s at 44,736 prompt tokens** and
**1,175 tok/s at 756 prompt tokens**. Median decode throughput was **109.2 tok/s**
at 44,736 tokens and **129.2 tok/s** at 756 tokens. These are synthetic
code-explanation prompts with greedy decoding and a 256-token output cap. They do
not establish answer quality or performance on other workloads.

## Hardware and software

- GPUs: NVIDIA GeForce RTX 3090 (24,576 MiB), RTX 5070 (12,227 MiB),
  RTX 5070 Ti (16,303 MiB). All three are used in one layer split. Power limits
  as reported by the driver: 270 W, 250 W, 252 W. GPU clocks were not fixed.
- PCIe: the engine's own host-to-device probe at startup reported 6.5 GB/s for
  stage 0 (3090), 6.9 GB/s for stage 1 (5070), and 3.0–3.1 GB/s for stage 2
  (5070 Ti). Link width was not read from `lspci` (not permitted on this host);
  `not measured`.
- CPU: Intel Core i7-8700K, 6 physical cores / 12 logical. The engine selected
  AVX-2 expert kernels (no AVX-512 on this CPU) and 18 pool tasks per phase with
  6 participating threads (5 pool workers plus the host thread).
- RAM: 62 GiB installed; Linux reported 62.7 GiB usable.
- Storage: old nvme
  system on `/dev/sdd2` (ext4). The engine reported "the SSD is kept awake while
  rows are read" — the PLE table is read directly from the HDD, unbuffered.
- OS: KDE neon User Edition 24.04, kernel 7.0.0-38-generic, NVIDIA driver 595.91.07.
- CUDA 13.0 (`/usr/local/cuda-13.0`). Engine 0.1.42, source build
  (`engine/BUILD.json`: `"source": "local"`, `"src": "1e0c97588848ea0f"`,
  CUDA archs 86 and 120, no vision helper).
- Build note: `setup.py` carries one local patch that accepts KDE neon
  (`osr.get("ID") not in ("ubuntu", "neon")`) so the automatic CUDA Toolkit
  install runs on a neon 24.04 base. No other engine changes.
- Background workloads: LM Studio runs with its embedding worker on CPU only;
  Plasma desktop and Hermes are running. The GPUs are not isolated — Strata holds
  45,344–45,383 MiB of the 53,106 MiB total VRAM while idle.

## Model and configuration

Model: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, quant **Q2_0**, two GGUF shards:

- `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf` — 37,623,740,192 bytes
- `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf` — 28,800,138,432 bytes (PLE table)

Custom pack built with `tools/iq_pack.py` (no `--base`), recorded in
`packs/q2_0/conversions.json`. The Hugging Face revision of the Q2_0 shards was
not recorded at download time, and the shards are on a shared NTFS volume rather
than in LM Studio's model directory — `revision: not measured`. Byte sizes are
given above so the files can be identified. Expert profile: `data/expert-profile.bin`,
196,632 bytes, 24,576 ranked pairs built for 24,576 slots. MTP draft from
`Strata-data/mtp/rt` (dense.bin, experts.bin, draft_vocab.bin), with a per-tensor
`mtp-manifest.json` carrying SHA-256 hashes for each downloaded tensor range.

```text
strata --serve \
  --pack packs/q2_0 \
  --native /media/linux/.../Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --ple-gguf /media/linux/.../Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
  --expert-profile data/expert-profile.bin \
  --expert-cache auto --prefill auto:16384 \
  --spec 4 --spec-min-p 0.5 --mtp Strata-data/mtp/rt \
  --max-context 260000 --kv int8 --kv-resident 16384 \
  --pipeline-windows 2 --trim-stage-weights --layer-split 28,40
# gpu: [0, 1, 2]  (3090, then 5070, then 5070 Ti)
# env: STRATA_SPLIT_OWN=1, STRATA_PREFILL_HELP=1
```

- Context 260,000; INT8 KV; 20,480 KV cells per QSA layer resident on GPU, the
  rest in 1.79 GiB of pinned RAM.
- Expert cache `auto`: 14,336 slots on stage 0 (18.46 GiB), 5,922 of its 6,144
  profiled pairs on stage 1 (7.62 GiB), 4,096 of 4,096 on stage 2 (5.27 GiB).
  Engine reported **99% of the experts resident**.
- Prefill: `--prefill auto:16384`; with `STRATA_SPLIT_OWN=1` each stage keeps its
  own prompt buffers (~1,904 MiB each, 2,048-token chunks) and the prompt path
  streams a 96-slot ring.
- MTP spec 4, minimum draft acceptance probability 0.5. No calibration, no
  experimental speed projection. Greedy sampling (`temperature 0`).
- Low-RAM mode: not enabled. Conversation cache: not enabled.

## Method

Script: [`bench/community_bench.py`](bench/community_bench.py). Three runs each of
a short prompt (~756 tokens) and a long prompt (~44,736 tokens), greedy, 256-token
output cap, streaming enabled so TTFT is measured at the first non-empty delta.
Each prompt carries a random salt so no conversation prefix is reused; every run
reported `cached_tokens: 0`. Model loading was not included in any timing — the
server was already resident and had served 120 requests before the benchmark.
Expert cache state was whatever the running server had; it was not reset between
runs. TTFT and total latency are client-side. VRAM and RAM are snapshots taken
immediately before and after each request.

Prompt and decode throughput are taken from the engine's own timing lines, not
computed from client latency.

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT s median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short | 756 | 0 | 256 | 3 | 1,175 (1,058–1,176) | 129.2 (117.1–133.8) | 0.66 (0.66–0.73) |
| long | 44,736 | 0 | 256 | 3 | 3,275 (3,274–3,276) | 109.2 (108.0–114.0) | 13.71 (13.70–13.72) |

Engine timing lines for the six runs:

```text
prompt 756 tokens = 0 reused + 756 read in 714 ms (1058.2 tok/s), 256 generated in 2185 ms (117.1 tok/s)
prompt 756 tokens = 0 reused + 756 read in 643 ms (1175.7 tok/s), 256 generated in 1914 ms (133.8 tok/s)
prompt 756 tokens = 0 reused + 756 read in 643 ms (1175.1 tok/s), 256 generated in 1981 ms (129.2 tok/s)
prompt 44736 tokens = 0 reused + 44736 read in 13654 ms (3276.3 tok/s), 256 generated in 2371 ms (108.0 tok/s)
prompt 44736 tokens = 0 reused + 44736 read in 13666 ms (3273.6 tok/s), 256 generated in 2246 ms (114.0 tok/s)
prompt 44736 tokens = 0 reused + 44736 read in 13659 ms (3275.3 tok/s), 256 generated in 2344 ms (109.2 tok/s)
```

Total latency (client-side, whole request): short 2.90 / 2.56 / 2.63 s; long
16.06 / 15.95 / 16.04 s.

Draft acceptance (MTP spec 4): short runs 114/184, 143/203, 126/175; long runs
109/174, 121/171, 118/173. Acceptance varied run to run and is the main source of
the decode-rate spread.

**Memory.** VRAM in use across the three cards during the benchmark: 45,344–45,383
MiB of 53,106 MiB total (startup snapshot, essentially flat across runs). RAM:
52,644–53,023 MiB used of 64,231 MiB (`MemTotal`). No paging or OOM failures
observed.

**Long-context recall** (`tools/needle_bench.py --lengths 32k,128k --depths 10,50,90`):
6 of 6 found.

| Length | Depth | Prompt tokens | Reused | Found | Seconds | Fresh prompt tok/s |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| 32k | 10% | 32,804 | 0 | yes | 11.9 | 2,790 |
| 32k | 50% | 32,804 | 0 | yes | 11.5 | 2,871 |
| 32k | 90% | 32,805 | 0 | yes | 11.5 | 2,867 |
| 128k | 10% | 125,537 | 0 | yes | 47.6 | 2,651 |
| 128k | 50% | 125,535 | 0 | yes | 47.9 | 2,637 |
| 128k | 90% | 125,535 | 16,384 | yes | 42.6 | 2,576 |

The 128k/90% case reused a 16,384-token prefix from the previous run, so its
throughput is computed over the 109,151 freshly read tokens.

Per-run data: [runs.json](runs.json), [needles.json](needles.json).

## Correctness and limitations

The needle test is the only correctness check; it establishes recall on those six
inputs, not general quality. No coding task, tool call, or image question was run.

Limitations worth stating plainly:

- **The 12 GB RTX 5070 is the ceiling.** It holds 5,922 of its 6,144 profiled
  expert pairs. The 222 non-resident pairs stream from RAM on every prompt chunk.
  For comparison, the same Q2_0 pack on a two-card 3090 + 5090 split measured a
  median of 2,174 tok/s in the 40k–50k prompt band (3,150 tok/s best in that band,
  3,336 tok/s overall; decode up to 258 tok/s), so adding the 5070 as a third
  stage is a net gain in prompt throughput even at 99% residency.
- **Split position matters more than split ratio.** Moving layers onto the 5070 Ti
  (the slowest PCIe link, 3.0 GB/s) degrades prefill sharply. A `28,36` split
  (12 layers on the Ti) was observed by the operator to drop prefill to roughly
  120 tok/s on this machine — reported as an observation, not a controlled
  measurement, and not reproduced here.
- **Decode is lower than the published single-GPU 5090 numbers** (179–165 tok/s
  there vs 109–129 tok/s here) because two hand-offs per verify window are added
  across three cards.
- **Storage is a spinning HDD.** The PLE table (28.8 GB) is read directly from it
  on the token path. A machine with NVMe would likely measure higher prompt
  throughput at equal residency.
- **Not an isolated system.** LM Studio, Plasma, and Hermes are running; LM Studio's
  Electron process holds ~1.1 GiB of VRAM on the 5070.
- The Q2_0 pack is a **custom pack** built with `tools/iq_pack.py`, not the
  installer's default pack.
