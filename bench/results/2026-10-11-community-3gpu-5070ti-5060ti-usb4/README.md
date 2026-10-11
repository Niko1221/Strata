# Community benchmark: RTX 5070 Ti + 2x RTX 5060 Ti (one on a USB4 eGPU), Ryzen 7 7700X, 46 GiB RAM in an LXC

Measured on 2026-10-11 by **chiangww** (GitHub). Unsloth UD-IQ4_XS on Strata v0.1.42,
3-GPU layer split `32,41` with `--pipeline-windows 2`, resident RAM mode
(32 GiB page-locked expert complement) on a machine whose third GPU hangs off a
USB4/Thunderbolt dock.

Headline: **prompt 918–1,369 tok/s at 30K–120K fresh tokens** (single stream,
thinking off), **decode median 66.8 tok/s** (range 60.8–83.2 across both suites),
**needle 6/6** at 32k/128k. The USB4 card (PCIe probe 3.8 GB/s vs 28.9/14.4
GB/s internal) is a real but surmountable bottleneck: `--pipeline-windows 2`
overlapping the three stages recovered decode to the same ~67 tok/s the machine
gets on two internal cards at IQ3_S.

Main limitation: one engine session, one repetition per long-context length;
the negative result included (STRATA_SPLIT_RING=384, −22% prefill median here)
was measured on the same session.

## Hardware and software

- GPUs (all three selected, layer split):
  - GPU0 NVIDIA GeForce RTX 5070 Ti 16 GB — PCIe 5.0 x8, direct to CPU, probe 28.9 GB/s host→device
  - GPU1 NVIDIA GeForce RTX 5060 Ti 16 GB — PCIe 4.0 x8, direct to CPU, probe 14.4 GB/s
  - GPU2 NVIDIA GeForce RTX 5060 Ti 16 GB — **USB4/Thunderbolt eGPU** behind a TBT4 dock
    (`00:02.2-[12-79]` AMD USB4 host → 3 downstream bridges → card, link reports
    PCIe 4.0 x4 after the dock), probe **3.8 GB/s**
- CPU: AMD Ryzen 7 7700X (8C/16T, AVX-512 capable, max 5.58 GHz); Strata pool:
  7 workers + host thread (`--host-core first`), 24 tasks/phase automatic
- RAM: 46 GiB usable by the container (host is a Proxmox node; Strata runs in an LXC)
- Storage: KLEVV CRAS C930 1 TB NVMe (models and packs on local NVMe)
- GPU power limits: set to **90% of spec** — 270 W on the 5070 Ti (of 300 W),
  162 W on each 5060 Ti (of 180 W)
- OS: Ubuntu 26.04.1 LTS inside LXC (Proxmox host kernel 7.0.14-15-pve)
- Driver 595.91.07; CUDA 13.3 toolkit (`/usr/local/cuda-13.3`)
- Strata **v0.1.42** (`61b3fb5`), engine binary installed by setup/update
  (`engine/BUILD.json`: version 0.1.42, arch 120, src `1e0c97588848ea0f`;
  vision engine src `1fb2b3fb797ec402`) — not a manual source build
- Background load: the Proxmox host ran its usual other VMs/LXCs during the
  runs (~5% host CPU load); the LXC itself had no other Strata instances.
  No clock capture.

## Model and configuration

- Model: `unsloth/Qwen3.8-Flash-Next-GGUF` @ revision `766911a6b7369840a91dbcd95f9f997acaab6cd6`,
  `UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-0000{1,2,3}-of-00003.gguf` (~93.7 GB),
  sha256:
  - 00001 `5ce89370720f8bf90890f439361282104c1aa1482d4013bb9a50923e758e71a4`
  - 00002 `577a38a2392b40ca2193cea502e1d92f60b8cd370675d308e0ec21885d9daaa7`
  - 00003 `d4634e6d84f0ebb0940be15c90d3790bf6464e3dea3a1cddc567dc0e83ad8833`
- Vision encoder `mmproj-Qwen3.8-Flash-Next-BF16.gguf` loaded (`--vision`) but not
  exercised — all runs are text-only
- Native-experts pack `unsloth-ud-iq4_xs` (setup-built, experts served natively
  from the GGUF, largest blob 3.48 MB); shipped `expert-profile.bin`
  (sha256 `8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf`)
- MTP draft head from setup (`Strata-data/mtp/rt`), `--spec 5 --spec-min-p 0.5`
  (verify window grows to 7 with the default suffix drafter; web panel: "MTP
  drafts up to 4 + extensions")
- Context 204,800; KV `k8v4`; expert cache auto (GPU0 2,960 / GPU1 4,370 /
  GPU2 3,348 slots); low-RAM mode `--resident-experts --resident-budget-gib 32`
  (30.9 GiB expert complement, 32 GiB page-locked); conversation cache
  5,120 MiB / 8 slots; `--vram-reserve-mib 750`; `--pipeline-windows 2`
- Abliterated behavior via control vector:
  `Huihui-Qwen3.8-Flash-Next-refusal-direction-r.gguf:1.0`, project, per-layer,
  layers 1..47 (47 steered)
- Sampling defaults (server config): temp 0.7, top_p 1.0, top_k 64, min_p 0.08,
  repetition_penalty 1.03; `reasoning_budget_tokens` 5120 (unused here, thinking off)
- Environment: `STRATA_PREFILL_HELP=1`, `STRATA_SSD_KEEPALIVE=0`,
  `STRATA_IO_PREFETCH=1`, `STRATA_PREFILL_CPU_SHARE=auto`
- No `--calibrate`, no experimental speed projection, no `--batch`.

```text
strata --serve \
  --pack Strata-data/packs/unsloth-ud-iq4_xs \
  --native models/unsloth-UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
  --expert-profile data/expert-profile.bin \
  --expert-cache auto --prefill auto \
  --spec 5 --spec-min-p 0.5 \
  --mtp Strata-data/mtp/rt \
  --max-context 204800 --kv k8v4 \
  --resident-experts --resident-budget-gib 32 \
  --vision --vram-reserve-mib 750 \
  --pipeline-windows 2 \
  --conversation-cache-mib 5120 --conversation-cache-slots 8 \
  --control-vector-scaled models/IQ3_S/Huihui-Qwen3.8-Flash-Next-refusal-direction-r.gguf:1.0 \
  --control-vector-layer-range 1 47 --cvec-mode project --cvec-dir per-layer \
  --layer-split 32,41        # GPUs 0,1,2; CUDA0 layers 0-31, CUDA1 32-40, CUDA2 41-47 + head
```

Full server config (paths abbreviated, no credentials): [config.json](config.json).
Startup log (engine 0.1.42, one session, everything below measured in it):
[engine-log.txt](engine-log.txt).

## Method

Two Python 3 stdlib clients against the OpenAI-compatible streaming API over
LAN (scripts not attached; the workloads are fully described here):

- **Short suite** (`results-short.txt`): 10 general questions in Traditional
  Chinese (TCP handshake steps, birthday probability for 30 people, quicksort
  without built-in sort, REST vs GraphQL + N+1, a two-pipe reservoir equation,
  the uncertainty principle, a monthly top-3 SQL window-function query,
  self-attention math, a knights-and-knaves logic puzzle, PostgreSQL MVCC).
  One user message each, thinking off (`chat_template_kwargs.enable_thinking:
  false`), `max_tokens` 2,048, sequential single stream.
- **Long fresh-prompt ramp** (`results-long.txt`): targets 15,000→120,000
  tokens in 15K steps. Context = a long system-design document in Traditional
  Chinese built by cycling 12 fixed technical paragraphs (order rotated per run
  by a seeded RNG) until the target length, with one needle fact planted at
  ~50% depth; the question asks to extract the needle and summarize the
  document. `max_tokens` 512. **Every request carries a random 32-hex salt at
  the front**, so no run reuses a prefix — engine logs show `reused = 0` on all
  eight runs. Character counts are resized to hit targets using measured
  `prompt_tokens` from earlier runs.

Both suites send one warm-up request first (absorbs CUDA graph capture / KV
allocation; model loading is not in any timed run). Timing is client-side:
TTFT = first non-empty streamed content delta; `prefill tok/s = prompt_tokens / TTFT`;
`decode tok/s = completion_tokens / (total − TTFT)` using the usage counters the
server reports (`stream_options.include_usage`). LAN latency is sub-millisecond.
The engine's own log lines agree with the client numbers to within ~1%
(e.g. 120,081 tokens in 87.69 s = 1,370 tok/s both sides). Expert cache state:
one warm session — the adaptive tier and pinned complement were trained by
prior production traffic (long agent conversations), not reset for these runs;
the ramp's own fresh prompts kept the page cache hot as it went.

## Results

Short suite (`results-short.txt`, 10 questions, prompt 36–69 tokens, output 435–2,048):

| Configuration | Prompt tok/s median / range | Decode tok/s median / range | TTFT s median / range |
| --- | ---: | --- | --- |
| UD-IQ4_XS 3-GPU split, thinking off, ≤2,048 out | 66.7 (5.8–105.3, noise-dominated at 36–69 tokens) | **71.6** (65.1–83.2) | 0.74 (0.41–7.20) |

Long fresh-prompt ramp (`results-long.txt`, one run per length, `reused = 0` on all):

| Actual prompt tokens | Reused | Generated | Prefill tok/s | Decode tok/s | TTFT s | Total s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 13,777 | 0 | 246 | 529.0 | 66.3 | 26.0 | 29.8 |
| 30,041 | 0 | 424 | 918.2 | 69.1 | 32.7 | 38.9 |
| 45,096 | 0 | 174 | 1,023.7 | 65.3 | 44.1 | 46.7 |
| 60,127 | 0 | 206 | 1,087.7 | 64.2 | 55.3 | 58.5 |
| 75,068 | 0 | 131 | 962.3 | 71.5 | 78.0 | 79.8 |
| 90,112 | 0 | 157 | 1,138.0 | 60.8 | 79.2 | 81.8 |
| 105,071 | 0 | 397 | 1,202.9 | 64.4 | 87.4 | 93.5 |
| 120,081 | 0 | 395 | **1,369.4** | 67.3 | 87.7 | 93.6 |

Prefill keeps rising with length (amortizing expert streaming); decode is flat
60.8–71.5 tok/s across 13.8K→120K context. Draft acceptance over the whole
session ran 64–70%.

### Negative result: STRATA_SPLIT_RING=384 (same session, same ramp)

`results-long-split-ring-384.txt` — only change: added `STRATA_SPLIT_RING=384`
(lend ring grows 96→384 slots; borrowed slots 4,624→5,864). Same targets, fresh
salts. Prefill median **887 vs 1,056 tok/s (−16%)**, per-length gaps +9% to
+44% against it; decode unchanged (66.5 vs 66.8). On this machine the extra
borrowed slots come out of a 42%-resident expert tier, and re-reading them
during chunks costs more than the ring saves. The author's R9700 numbers
(+2.3%, file reads −65%) do not transfer to a USB4-eGPU split with tight RAM.
Reverted after this A/B; baseline is the table above.

### Correctness checks

- `tools/needle_bench.py --lengths 32k,128k --depths 10,50,90`:
  **6/6 found** (`needles.json`); 32.8K prompts 38–46 s, 125.5K prompts 103–209 s
  (the 209 s outlier is the depth-90 run right after a fresh 120K ramp —
  page-cache pressure, not a recall failure).
- The short suite doubles as qualitative spot-checks (reasoning, code, SQL);
  answers complete and on-topic at temp 0.7; no stalls, all runs finished with
  `stop`.

## Memory observations

- VRAM, idle snapshot with everything loaded: GPU0 15,593 / 16,303 MiB,
  GPU1 15,549 / 16,311, GPU2 15,725 / 16,311; engine reports 339 MiB free on
  GPU0 after all captures (`--pipeline-windows 2` holds 309 MiB back on CUDA0
  itself for the second verifier + GDN snapshots). No stalls, no OOM.
- RAM: 32 GiB page-locked expert complement built at startup in 33 s (engine
  released the model's cached pages first); ~10.8 GiB left for the OS page
  cache against a 23.4 GiB file-tier expert footprint — the startup WARNING
  that borrowed prompt-path slots have no RAM copy is expected in this regime.
- No paging, no engine restarts during measurement.

## Limitations

- One engine session; one repetition per long-context length (the short suite
  is 10 distinct questions, not repeats of one).
- Thinking/reasoning off; vision loaded but untested; TTFT includes tokenizer
  time and LAN, excludes queueing (single client).
- The expert cache and page cache were warmed by real prior traffic — a cold
  first-run-after-boot number will be lower (observed ~1,690 tok/s on 165K for
  a cold-cache 0.1.41 first prompt vs 2,000+ warm).
- Decode was not compared against `--pipeline-windows` off within this session;
  the two-card IQ3_S sibling config of this machine measures ~75 tok/s median,
  and without pipeline overlap this three-card IQ4_XS split ran clearly slower.
