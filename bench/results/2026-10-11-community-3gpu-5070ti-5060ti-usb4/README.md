# Community benchmark: RTX 5070 Ti + 2x RTX 5060 Ti (one on a USB4 eGPU), Ryzen 7 7700X, 46 GiB RAM in an LXC

Measured on 2026-10-11 by **chiangww** (GitHub). Unsloth UD-IQ4_XS on Strata v0.1.42,
3-GPU layer split `32,41` with `--pipeline-windows 2`, resident RAM mode
(32 GiB page-locked expert complement) on a machine whose third GPU hangs off a
USB4/Thunderbolt dock.

Headline (pipeline active, no repetition penalty): **prompt 852–1,281 tok/s at
30K–120K fresh tokens** (single stream, thinking off), **decode median 75.9
tok/s** on the short suite (67.1–86.7), **needle 6/6** at 32k/128k.

The transferable finding: **`--pipeline-windows` silently decodes serially
whenever a request carries `penalty_last_n > 0`** — so a llama.cpp-style
default sampling set (`repeat_penalty 1.03`) leaves the flag a no-op with only
a once-per-reason log line. We measured both arms on this rig: pipeline ON is
worth **+3.4–6% decode median** here (the decode bottleneck is expert file
fetches, not hand-off, so far from the +33% measured on 4x R9700); what the
overlap does hide is the per-window hand-off across the USB4 stage.

Main limitations: two engine sessions (one per arm), one repetition per
long-context length; the arms also differ in generated text (no penalty vs
1.03), so the A/B is regime-level, not strictly single-variable.

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
  (30.9 GiB expert complement, 32 GiB page-locked); `--vram-reserve-mib 750`;
  `--pipeline-windows 2`
- Conversation cache 5,120 MiB / 8 slots in the serial arm, 6,144 MiB in the
  pipeline arm (raised between sessions; not expected to affect decode)
- Abliterated behavior via control vector:
  `Huihui-Qwen3.8-Flash-Next-refusal-direction-r.gguf:1.0`, project, per-layer,
  layers 1..47 (47 steered)
- Sampling:
  - **pipeline arm (headline)**: server defaults temp 0.7, top_p 1.0, top_k 64,
    min_p 0.08 — **no `repeat_penalty`** ([config-rep-off.json](config-rep-off.json));
    clients sending `repeat_penalty: 1.0` did not trigger the serial fallback
  - **serial arm**: same plus `repetition_penalty 1.03` server-side
    ([config.json](config.json))
  - `reasoning_budget_tokens` 5120 (unused here, thinking off)
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
  --conversation-cache-mib 6144 --conversation-cache-slots 8 \
  --control-vector-scaled models/IQ3_S/Huihui-Qwen3.8-Flash-Next-refusal-direction-r.gguf:1.0 \
  --control-vector-layer-range 1 47 --cvec-mode project --cvec-dir per-layer \
  --layer-split 32,41        # GPUs 0,1,2; CUDA0 layers 0-31, CUDA1 32-40, CUDA2 41-47 + head
```

Startup logs: [engine-log-rep-off.txt](engine-log-rep-off.txt) (pipeline arm,
headline; contains **no** "decodes serially" line across 20 requests) and
[engine-log.txt](engine-log.txt) (serial arm, `repeat_penalty 1.03`, warning
present).

## Method

Two Python 3 stdlib clients against the OpenAI-compatible streaming API over
LAN (scripts not attached; the workloads are fully described here):

- **Short suite** (`results-short*.txt`): 10 general questions in Traditional
  Chinese (TCP handshake steps, birthday probability for 30 people, quicksort
  without built-in sort, REST vs GraphQL + N+1, a two-pipe reservoir equation,
  the uncertainty principle, a monthly top-3 SQL window-function query,
  self-attention math, a knights-and-knaves logic puzzle, PostgreSQL MVCC).
  One user message each, thinking off (`chat_template_kwargs.enable_thinking:
  false`), `max_tokens` 2,048, sequential single stream.
- **Long fresh-prompt ramp** (`results-long*.txt`): targets 15,000→120,000
  tokens in 15K steps. Context = a long system-design document in Traditional
  Chinese built by cycling 12 fixed technical paragraphs (order rotated per run
  by a seeded RNG) until the target length, with one needle fact planted at
  ~50% depth; the question asks to extract the needle and summarize the
  document. `max_tokens` 512. **Every request carries a random 32-hex salt at
  the front**, so no run reuses a prefix — engine logs show `reused = 0` on all
  runs. Character counts are resized to hit targets using measured
  `prompt_tokens` from earlier runs.

Both suites send one warm-up request first (absorbs CUDA graph capture / KV
allocation; model loading is not in any timed run). Timing is client-side:
TTFT = first non-empty streamed content delta; `prefill tok/s = prompt_tokens / TTFT`;
`decode tok/s = completion_tokens / (total − TTFT)` using the usage counters the
server reports (`stream_options.include_usage`). LAN latency is sub-millisecond.
The engine's own log lines agree with the client numbers to within ~1%.
Expert cache state: warm sessions in both arms — the adaptive tier and pinned
complement were trained by prior production traffic (long agent conversations),
not reset for these runs; the ramp's own fresh prompts kept the page cache hot
as it went.

## Results

### Pipeline arm (headline; no repetition penalty, `--pipeline-windows 2` active)

Short suite (`results-short-rep-off.txt`, 10 questions, prompt 36–69 tokens,
output 514–2,048):

| Configuration | Decode tok/s median / range | TTFT s median / range |
| --- | ---: | --- |
| UD-IQ4_XS 3-GPU split, pipeline ON, thinking off, ≤2,048 out | **75.9** (67.1–86.7) | 2.03 (0.57–2.49) |

TTFT on 36–69-token prompts was erratic (0.6–2.5 s; first eight requests ~2 s,
last two ~0.6 s) — prefill at these sizes is overhead- and noise-dominated;
listed for completeness, not as a throughput figure.

Long fresh-prompt ramp (`results-long-rep-off.txt`, one run per length,
`reused = 0` on all):

| Actual prompt tokens | Reused | Generated | Prefill tok/s | Decode tok/s | TTFT s | Total s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 13,777 | 0 | 283 | 545.4 | 69.1 | 25.3 | 29.4 |
| 30,038 | 0 | 229 | 851.8 | 69.1 | 35.3 | 38.6 |
| 45,097 | 0 | 347 | 1,020.1 | 67.5 | 44.2 | 49.4 |
| 60,134 | 0 | 174 | 1,048.1 | 71.0 | 57.4 | 59.8 |
| 75,062 | 0 | 241 | 982.4 | 68.6 | 76.4 | 79.9 |
| 90,110 | 0 | 123 | 1,262.1 | 61.4 | 71.4 | 73.4 |
| 105,076 | 0 | 141 | 1,281.3 | 64.9 | 82.0 | 84.2 |
| 120,080 | 0 | 219 | 1,262.4 | 66.6 | 95.1 | 98.4 |

Decode stays 61–71 tok/s across 13.8K→120K context; draft acceptance 62%.

### A/B: the `penalty_last_n` serial fallback

With `--pipeline-windows 2`, the engine decodes a request **serially** if it
carries `penalty_last_n > 0` (the once-printed line: `a request with repetition
penalties (penalty_last_n) decodes serially`). Sending `repeat_penalty: 1.0`
does **not** trigger it — the fallback keys on the history window, not on the
penalty value. Both arms are full 20-request sessions on the same machine and
engine config, one session each:

| Arm | Short decode median | Ramp decode median | Engine-side decode median | Acceptance |
| --- | ---: | ---: | ---: | ---: |
| serial (`repeat_penalty 1.03`) | 71.6 (65.1–83.2) | 65.8 (60.8–71.5) | 67.5 (60.3–83.1) | 0.65 |
| pipeline (no penalty) | **75.9** (67.1–86.7) | **68.0** (61.4–71.0) | **69.8** (60.5–86.6) | 0.62 |

Pipeline overlap is worth **+3.4% to +6% decode median** on this rig. Modest
because the decode bottleneck here is expert file fetches (engine-side stats
show decode waited 10–20% of wall time in fetches), which overlap does not
remove; what it does hide is the per-window hand-off across the USB4 stage.
Caveats: dropping the 1.03 penalty also changes the sampled text (acceptance
0.65 → 0.62), and the conversation cache differs between the sessions
(5,120 vs 6,144 MiB), so the delta is a regime-level number.
Suggested consideration: a request whose penalty is exactly 1.0 is a no-op —
skipping the penalty history for it (instead of only checking
`penalty_last_n > 0`) would keep pipeline mode on for llama.cpp-style default
sampling sets, which is what most clients send.

### Serial arm detail (`repeat_penalty 1.03`; also the baseline for the SPLIT_RING A/B)

Ramp (`results-long.txt`): prefill 529.0 / 918.2 / 1,023.7 / 1,087.7 / 962.3 /
1,138.0 / 1,202.9 / **1,369.4** tok/s at 13.8K/30K/45K/60K/75K/90K/105K/120K;
decode 60.8–71.5 tok/s; acceptance 64–70%. Short suite (`results-short.txt`):
decode median 71.6 (65.1–83.2).

### Negative result: STRATA_SPLIT_RING=384 (same serial session, same ramp)

`results-long-split-ring-384.txt` — only change: added `STRATA_SPLIT_RING=384`
(lend ring grows 96→384 slots; borrowed slots 4,624→5,864). Same targets, fresh
salts. Prefill median **887 vs 1,056 tok/s (−16%)**, per-length gaps +9% to
+44% against it; decode unchanged (66.5 vs 66.8). On this machine the extra
borrowed slots come out of a 42%-resident expert tier, and re-reading them
during chunks costs more than the ring saves. The author's R9700 numbers
(+2.3%, file reads −65%) do not transfer to a USB4-eGPU split with tight RAM.
Reverted after this A/B.

### Correctness checks

- `tools/needle_bench.py --lengths 32k,128k --depths 10,50,90` (serial session):
  **6/6 found** (`needles.json`); 32.8K prompts 38–46 s, 125.5K prompts 103–209 s
  (the 209 s outlier is the depth-90 run right after a fresh 120K ramp —
  page-cache pressure, not a recall failure).
- The short suite doubles as qualitative spot-checks (reasoning, code, SQL);
  answers complete and on-topic at temp 0.7; no stalls, all runs finished with
  `stop`, in both arms.

## Memory observations

- VRAM, idle snapshot with everything loaded: GPU0 15,593 / 16,303 MiB,
  GPU1 15,549 / 16,311, GPU2 15,725 / 16,311; engine reports 339 MiB free on
  GPU0 after all captures (`--pipeline-windows 2` holds 309 MiB back on CUDA0
  itself for the second verifier + GDN snapshots). No stalls, no OOM, both arms.
- RAM: 32 GiB page-locked expert complement built at startup in 33 s (engine
  released the model's cached pages first); ~10.8 GiB left for the OS page
  cache against a 23.4 GiB file-tier expert footprint — the startup WARNING
  that borrowed prompt-path slots have no RAM copy is expected in this regime.
- No paging, no engine restarts during measurement.

## Limitations

- Two engine sessions (one per A/B arm); one repetition per long-context length
  (the short suite is 10 distinct questions, not repeats of one).
- Thinking/reasoning off; vision loaded but untested; TTFT includes tokenizer
  time and LAN, excludes queueing (single client).
- The expert cache and page cache were warmed by real prior traffic — a cold
  first-run-after-boot number will be lower (observed ~1,690 tok/s on 165K for
  a cold-cache 0.1.41 first prompt vs 2,000+ warm).
- The pipeline-vs-serial A/B changes generated text between arms (penalty
  removed; acceptance 0.65 → 0.62) and conversation-cache size (5,120 vs
  6,144 MiB); the decode delta is a regime-level number, not a strictly
  single-variable one.
