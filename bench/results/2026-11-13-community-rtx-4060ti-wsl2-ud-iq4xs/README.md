# Community benchmark on RTX 4060 Ti 16 GB (WSL2)

Measured on 2026-11-13 by loopy321. What was tested: the community suite
(`tools/strata_bench.py --long`, three runs per test) against Unsloth
**UD-IQ4_XS** Flash-Next at a 131,072-token context on a 16 GB card under WSL2, with
the machine's normal launch configuration unchanged, **plus the rows the suite leaves
out** — a 128K arm, a 4-client burst and a needle-in-a-haystack recall check, all run in
the same engine session (see [Method](#method)). Main limitations: the long-prompt arms
are capped at 24 generated tokens (so their decode column is a 24-token sample, not a
256-token decode run), the concurrency row measures **queueing, not batching** (this
config has one slot), recall is 6 checks and nothing else, and the engine reports
**0 MiB of free VRAM** with everything loaded.

## Hardware and software

- GPU and VRAM; CPU; installed RAM; storage; PCIe link if known:
  - GPU: NVIDIA GeForce RTX 4060 Ti, 16,380 MiB VRAM, one GPU (no layer split).
  - CPU: 13th Gen Intel Core i7-13700F, 8 cores / 16 threads, 1 socket, 1 NUMA node.
  - Installed RAM: 78.5 GiB as seen by the Linux guest (WSL2 allocation, not the
    Windows physical total).
  - Storage: Crucial CT1000BX500SSD1 (BX500 1 TB SATA SSD) 931.5 GB, exposed to WSL2 as
    `/dev/sdd` with `rotational=1` (WSL2's virtual block device reports no transport and
    sets the rotational flag); `/data` is mounted from `/dev/sdd1`, and the model shards,
    pack and draft model all live on it. The engine's
    prefill plan therefore prints `drive 0.15 GB/s (sdd is a rotational disk)`, which
    is a WSL2 artefact of the virtual block device, not the real device speed.
  - PCIe: gen 4, width x8 (current = max gen 4, from the engine's own hardware
    snapshot and from `nvidia-smi`). Engine PCIe probe: `12.0 GB/s host->device (best
    of 12.0 12.0 12.0 11.9) -> pcie_frac 0.33 (default 0.55)`, later lowered to
    `pcie_frac 0.20` for the serve path with 7 pool workers.
- OS; driver; CUDA or ROCm:
  - Windows 11 (`ver` reports `Microsoft Windows [Version 10.0.26200.9711]`) with WSL2
    Ubuntu; guest kernel `Linux 6.18.26.1-microsoft-standard-WSL2`.
  - NVIDIA driver 610.88 (driver supports CUDA up to 13.3); local CUDA toolkit 12.8
    is installed but was not used to build anything for this run.
- Strata commit; engine version; release binary or source build:
  - Strata **v0.2.12**, release tag `v0.2.12`, source commit
    `146df3c468739147f520ac3816de806f71232571` (from the release's `verification.json`;
    the local source tree is an extracted archive, so `strata_commit` in `results.json`
    is `null` — the tool cannot read it from a non-git folder).
  - Engine **0.1.58**, official release binary
    `strata-engine-0.1.58-linux-x86_64-cuda-cuda12.tar.gz` (archive sha256
    `a0e068d6c79b30a0126ae015e9d61e39ad3b2b2b48d61e6b543bd2fc2dee5d2f`), `strata`
    binary sha256 `3b7f3ad358b4833f07b5f1e8e03a039d7746c6977347935610ca29ac44e9834f`
    (529,289,088 bytes). `BUILD.json`: `source f-0212-rc 0fe4ae5090bd`, CUDA 12.9,
    portable build for sm_60/61/70/75/80/86/89 + compute_89 PTX. No build options
    changed; nothing was compiled for this run.
- Background workloads and any power limits:
  - The machine's normal inference service (`llama-server.service`, port 8080) was
    drained (18.1 s to idle) and stopped before the run, freeing 11,910 MiB of VRAM
    (1,206 → 13,116 MiB free); the benchmark engine was the only inference process
    (telemetry `foreign_inference_pids` empty for all 437 samples). The run stops only
    `llama-server.service`; the machine's unrelated Docker containers were not touched
    by the profile.
  - GPU power limit 165.0 W (default). Power draw **was** sampled during the run (2 s
    `nvidia-smi` samples, 654 rows): median **116.9 W**, p90 127.9 W, max 139.1 W — the
    card never reached its 165 W limit. SM clock median 2,670 MHz (range 2,625–2,685),
    memory clock 8,751 MHz, temperature median 72.5 °C (max 84 °C). GPU utilisation from
    the harness sampler (437 samples every 3 s): median 91 %, p10 79 %, p90 97 %, max
    100 %; 409 of 437 samples are ≥ 70 %, and the low tail is between-request samples.

## Model and configuration

- Model repository and revision; quantization; GGUF filenames:
  - `unsloth/Qwen3.8-Flash-Next-GGUF`, revision
    `d96201194f103c906994f283f504209dc8d2a10c`, quantization **UD-IQ4_XS** (dynamic).
  - Shards (size, Hugging Face LFS etag = SHA-256):
    | File | Bytes | SHA-256 |
    | --- | ---: | --- |
    | `UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf` | 10,946,618 | `93ff23a08e5cd7501f24fa16f5b415d5807d0175e04edd3ed6a7a41b601a4eef` |
    | `UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00004.gguf` | 49,604,491,008 | `13d5dc8b93d039e4cfade5a9af7037f10c97f7ec4e53f75bd2e9ae34aae54aeb` |
    | `UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00004.gguf` | 15,256,783,680 | `f6b9b5934908a2043732269dd740155d0f921d0763a1dcdd3452246311651f87` |
    | `UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00004-of-00004.gguf` | 28,800,138,432 | `5de8b5dbf89409acc39468decafd4a4a95b13c29d88a14b0356107897829ec36` |
  - Shard 1 is `--native` (attention/dense + metadata), shard 4 is `--ple-gguf`.
- Vision encoder; custom packs or profiles:
  - Vision: **off** (`images: false`).
  - Custom pack: a **native-experts pack** built locally with `tools/iq_pack.py`
    (index v3, `align 64 pool 1485606400 tensors 1079`) from shard 1. Experts are read
    **in place** from the shards — there is no separate `experts.bin`.
    `native_experts.txt` sha256 `b1cf8679736b3452e59259a336b738b4880c0d406c61cf9f64cdb352e8ad74e1`
    (also recorded as `native_experts_sha256` in `experts.bin.src.json`, experts total
    60,384,870,400 bytes); `index.txt` sha256 `585dd55b9de46bb65223ff096e6262359d3b0d5be8293e89e35515b409ec66d1`;
    `dense.bin` sha256 `cdd71e91132cb69b955490877d4496ef9d990febd2535ba3c44c085bc836efde`
    (1,485,688,320 bytes); `compat-bf16.json` records the bf16-compat re-encode
    (rounding `nearest-even`) of `output_hc_down.weight` and the other `Q8_0` tensors.
  - Expert profile: `expert-profile.bin` sha256
    `8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf` — 24,576 ranked
    pairs, built for 24,576 slots; the engine pre-filled 2,294 of 2,294 slots from it
    in 1.5 s (3,702 MB/s) at load.
  - Draft vocabulary: `draft_vocab.bin` sha256
    `b1e1d3a7a9e4bf862dcd5923ce661fb59bbd07907e594df5cf86a62ac235cb91`.
  - Draft model: `qwen3.8-flash-next-dflash2.q4_0-selq4.pack` sha256
    `4a42b98220ea7880664002c8edda1563aecbae754b762a2e2f31b2b91ee93ecd` (339,694,592
    bytes) — a locally selected Q4_0 DFlash2 pack, not an official asset.
- Context; KV type and streaming; cache; prefill; low-RAM mode:
  - Context 131,072. KV `k8v4`, `--kv-resident 0` with `--kv-grow`: elastic, starting
    at 16,384 of 131,072 cells in VRAM (0.19 of 1.20 GiB) and growing on demand — to
    32,768 cells (0.33 GiB) during the 32K arm, taking 61 slots from the expert cache for
    it, and to 131,072 cells (1.20 GiB) during the 128K arm, taking 228 slots (2,066 of
    2,294 left holding experts). Both times the engine trimmed K/V back to 8,192 cells
    afterwards and returned the slots to the expert cache.
  - Expert cache: `--expert-cache 2700` **requested, clamped to 2,294 slots /
    5,413 MiB** by free VRAM (`expert cache 2294 slots, 5.29 GiB of VRAM`). The prompt
    path additionally borrows 1,938 cache slots (4.46 GiB). R4 hit path on, routing-aware
    prefetch on, `--mmap-experts`.
  - Prompt cache 6 entries; conversation cache 4,096 MiB / 2 slots (see limitations —
    it never paid back in this workload).
  - Prefill: `--prefill auto` → chunk 9,472, 1 stage, `prefill_pipe 0`,
    `stream_all_min 3072`. Low-RAM mode: not set. `--vram-reserve-mib 4096`.
  - CPU workers: 7 pool workers by default (8 physical cores, 16 logical; one worker
    per physical core but the host's), 24 tasks/phase, host thread also draining; aux
    CPUs off at start.
- MTP; reasoning; sampling; calibration; experimental speed projection:
  - MTP: none (`mtp_max 0`); drafting is DFlash2 (`--draft dflash2 --spec 6
    --draft-conf 0.7 --draft-fallback none`, `--suffix-draft 0`, `--lookup-max-t 0`,
    `spec_min_p 0.00`). Draft head 106,299 tokens (212.9 MiB), drafter 625 MiB of VRAM.
  - Reasoning: **thinking off** for the whole suite (`reasoning_effort: none` plus
    `chat_template_kwargs.enable_thinking: false`), so answers are short and comparable.
  - Sampling: `temperature 0` (the suite's own fixed value), no other sampling fields
    sent.
  - Calibration: not used. Experimental speed projection: **not enabled**
    (`projection: null` on every request record).

```text
<WORKSPACE>/engines/strata-v0.2.12-isolated/engine/strata --serve \
  --pack   <MODELS>/packs/unsloth-flash-next-ud-iq4-xs-v31 \
  --native <MODELS>/models/unsloth-flash-next-dynamic-v31/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf \
  --ple-gguf <MODELS>/models/unsloth-flash-next-dynamic-v31/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00004-of-00004.gguf \
  --expert-profile <WORKSPACE>/engines/strata-v0.2.10-source/Strata-0.2.10/data/expert-profile.bin \
  --expert-cache 2700 --max-context 131072 --kv k8v4 --kv-resident 0 --kv-grow \
  --prefill auto --suffix-draft 0 --lookup-max-t 0 \
  --conversation-cache-mib 4096 --conversation-cache-slots 2 --prompt-cache 6 \
  --vram-reserve-mib 4096 --mmap-experts \
  --spec 6 --draft dflash2 \
  --draft-model <MODELS>/dflash2/qwen3.8-flash-next-dflash2.q4_0-selq4.pack \
  --draft-fallback none \
  --mtp-draft-vocab <WORKSPACE>/engines/strata-v0.2.10-source/Strata-0.2.10/data/draft_vocab.bin \
  --draft-conf 0.7
```

`<WORKSPACE>` and `<MODELS>` are placeholders for local directories; no credentials are
involved. One slot only (`total_slots 1`, no `--batch`). No `--model`/preset file: the
arguments above are the whole configuration.

## Method

Suite: `python tools/strata_bench.py --url http://127.0.0.1:<port> --runs 3 --long`
(the port is a local loopback port on the same machine) from the extracted `v0.2.12`
source tree (the tool's own fixed prompts, tags and thresholds, unmodified). The server
was already running, so the tool reported `started_server: false`; the engine log
excerpt is from that server's own start-up.

**Two sessions on the same day, same config, same binary.** The suite was run twice in
two separate engine sessions. The numbers in the table below are all from the **second**
session (`…-171546`), which ran the published suite *and* the extra arms in one engine
session, so every row shares one model load, one power capture and one set of host
telemetry. The first session (`…-145725`) ran the published suite alone; its
[`results-session-145725.json`](results-session-145725.json) is included as a
reproducibility check. It reproduces the **prefill and latency** rows closely — prompt
tok/s 496.8 → 502.8 (+1.2 %), 795.5 → 814.2 (+2.4 %), 815.4 → 824.8 (+1.2 %); TTFT within
2.7 % on every arm; agent cache hit 69.84 % → 69.81 % — and it does **not** reproduce the
short-sample **decode** rows: 16K 30.1 → 33.6 (+11.6 %), 32K 26.0 → 23.4 (−10.0 %), agent
30.6 → 29.1 (−4.9 %), chat 27.2 → 27.9 (+2.6 %). That spread is the 24-token decode
sample talking, not the machine: the two sessions differ in decode by more than the arms
differ from each other, which is why the decode column here should be read as an
indication, not a measurement.

The extra arms were added **without changing a suite parameter**. They call the pinned
checkout's own code, so the prompts, the per-request unique tags, the words-per-token
calibration and the speed source are still the tool's:

| Phase | What ran | Requests |
| --- | --- | ---: |
| 1 | `tools/strata_bench.py --runs 3 --long` (the published suite, unchanged) | 40 |
| 2 | `Suite.run_prompt(130000, "128K")` — the tool's own prompt generator and its own context guard, after its own 2-request calibration | 5 |
| 3 | `Suite.run_clients(4)` — the tool's own burst path (`tools/serve_load.py`) | 12 |
| 4 | `tools/needle_bench.py --lengths 32k,128k --depths 10,50,90` — the repo's own recall checker | 6 |

- **Repetitions:** 3 per test. chat = 3 prompts x 3 runs = 9 requests; each prompt arm
  = 3 requests; agent = 3 sessions x 6 turns (1 initial + 5 turns) = 18 requests; 128K arm
  = 3 requests + 2 one-token calibration probes; burst = 4 clients x 3 rounds = 12
  requests; recall = 6 requests. That is **57 measured requests** in the table, plus the
  suite's own 4 warm-up requests and the 2 calibration probes = **63 requests issued by
  the driver**.
- **Warm-up and loading:** the harness issued one further 8-token generation before the
  suite (2.11 s wall), so the engine's own cumulative counter reads 64 requests — all
  figures in the table are the 57 measured requests, not that counter. Model loading is
  not included in any timing. One engine session for the whole run (single load, no
  restarts between arms). The expert cache was warm from load (2,294/2,294 slots
  pre-filled from the profile in 1.5 s), and the GGUF experts were fully page-cached
  (`prefill: the GGUF experts are 100% in the page cache`).
- **Prompt reuse:** every fresh-prompt arm carries a unique tag and reports
  `reused = 0` (chat, 4K, 16K, 32K, 128K), and so does the burst (each request has its
  own nonce). Only the agent-session turns (618 → 10,867 per turn) and the later recall
  rows reuse tokens. So prompt throughput and cache reuse are reported as separate
  conditions.
  **State between runs was not reset**: one engine session, no restart, no cache flush
  between the three runs of an arm — which is why the fresh-prompt arms are tagged
  uniquely (each run is a different prompt, so the prompt cache cannot help it), while
  the agent arm deliberately keeps its prefix. **One exception, disclosed:** the three
  recall rows share one haystack and differ only in where the code word is inserted, so
  the later rows get a genuine prefix-cache hit — `0`, `18,944` and `56,832` reused
  tokens at depths 10 %, 50 % and 90 %. Their *verdicts* are still independent (a unique
  code word per row, generated fresh), but their *seconds* are not comparable to the
  fresh-prompt rows; see the recall table.
- **Output caps:** chat 100 tokens, 4K/16K/32K/128K prompt arms 24 tokens, agent turns
  64 tokens, burst 128 tokens, recall 40 tokens. The chat arm generated exactly 100
  tokens in all 9 requests, every prompt arm exactly 24 in all 3, and every burst request
  exactly 128 in all 12 — those all ran to the cap. 8 of the 15 measured agent turns hit
  the 64-token cap; the other 7 stopped short on their own answer (18–54 tokens) and the
  three cold first turns at 10–13, so the agent decode figure is a short-sample figure
  too. The recall answers were 9–10 tokens (the code word) well inside the 40-token cap.
  The suite's request records do not carry a finish reason, so "ran to the cap" is read
  off the generated-token count matching the cap exactly.
- **Timing boundaries:** prompt and decode tok/s are the engine's own
  `prompt_per_second` / `predicted_per_second`; TTFT and total latency are measured by
  the client over the streaming HTTP API (keep-alives and empty deltas ignored). Because
  thinking is off, the first token counted for TTFT is always **answer text**, never a
  reasoning token. Total latency is client-side and includes prompt processing; model
  loading and queueing are not included (one slot, requests sent sequentially) — **the
  one exception is the burst row**, whose whole subject is the queueing delay four
  simultaneous requests see on a one-slot server. No decode figure was derived by
  dividing tokens by total request time.
- **Prompts and scripts:** no custom prompts. The suite's fixed prompts come from the
  repository itself — `tools/strata_bench.py`, which imports the three short chats and
  the word-list generator from `tools/ab_engine.py` (`PROMPTS`, `long_prompt`) and the
  agent system prompt and four tools from `tools/agent_turns_bench.py` (`SYSTEM`,
  `TOOLS`). The 128K arm reuses `long_prompt` through `Suite.run_prompt`; the burst uses
  `tools/serve_load.py`'s fixed essay topics through `Suite.run_clients`; the recall rows
  use `tools/needle_bench.py`, whose haystack is built from the repository's own docs and
  source files. Suite arguments: `--url`, `--runs 3`, `--long`, `--out`. Extra-arm
  arguments: a 130,000-token target (the largest the tool's own
  `tokens + 24 + 256 > context` guard allows at 131,072; the prompts land at 126,936 real
  tokens), `4` clients, and `--lengths 32k,128k --depths 10,50,90`.
- **Memory and power:** sampled every 3 s from the host (`nvidia-smi`/NVML: utilisation,
  used/free VRAM, RAM, swap, load, engine state) and every 2 s by a second sampler
  (power draw, SM/memory clocks, temperature — 654 samples, 0 failed); plus the engine's
  own `/metrics` snapshot before and after. See the memory lines under Results.
- **Not measured:** `--clients` with `--batch` (real batched decoding — this config has
  one slot, so the burst row is queueing only); a cold page cache (the experts were 100 %
  page-cached the whole run, so the SSD path is untested); a second engine session for
  the recall rows (each haystack is built once and shared across its three depths);
  Windows-side memory pressure; and any task-level quality measure beyond the 6 recall
  checks.

## Results

All rows are from one engine session (three runs per test, thinking off, temperature 0).
The 128K, burst and recall rows are the arms the published suite does not run; they were
produced by the tool's own code, as described under Method.

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| chat (short, cap 100) | 46 (43–51) | 0 | 100 | 9 | 41.0 (38.6–44.5) | 27.9 (26.7–37.7) | 1.21 (1.06–1.31) |
| 4K prompt (cap 24) | 4,019 | 0 | 24 | 3 | 502.8 (455.5–511.4) | 25.0 (24.9–28.9) | 8.04 (7.90–8.87) |
| 16K prompt (cap 24) | 16,058 | 0 | 24 | 3 | 814.2 (796.3–817.7) | 33.6 (30.8–35.8) | 19.77 (19.69–20.21) |
| 32K prompt (cap 24) | 32,049 | 0 | 24 | 3 | 824.8 (800.3–827.0) | 23.4 (20.0–27.0) | 38.92 (38.81–40.11) |
| 128K prompt (cap 24) | 126,936 | 0 | 24 | 3 | 875.3 (862.5–882.2) | 31.3 (29.0–32.3) | 145.12 (143.98–147.27) |
| agent turn (cap 64), turns 2–6 only | 8,600 (3,823–13,577; last turns 13,554–13,577) | 5,420 (618–10,867) | 64 (18–64) | 15 (3 sessions x 5) | 366.5 (228.7–418.2) | 29.1 (26.6–36.5) | 7.20 (5.87–8.92) |
| agent turn, all 18 turns | 6,980 (609–13,577) | 4,620 (0–10,867) | 54 (10–64) | 18 | 347.0 (96.8–418.2) | 28.9 (20.2–36.5) | 7.13 (5.55–8.92) |
| 4 clients at once, one slot (cap 128) | 67 (62–68) | 0 | 128 | 12 (4 x 3 rounds) | 39.5 (32.2–40.8) | 26.9 (21.2–30.1) | 12.03 p50 / 22.14 p95 |

The two agent rows differ only in which turns are counted: the suite's own per-turn
statistics (`turn_decode_tps`, `turn_ttft_s`) cover **15** turns, because each session's
first turn is excluded — it is cold (0 reused tokens) and only 10–13 tokens long, so its
rate is dominated by the first windows. The "all 18 turns" row is the same data including
those three turns, computed by hand from `results.json`. Use the 15-turn row when
comparing against another `strata_bench.py` report, since that is what the tool prints.

The 4K/16K/32K/128K decode columns are medians over three 24-token generations and are
noisy for that reason (the 32K spread, 20.0–27.0 tok/s, is wider than the difference
between any two arms). The 128K prefill rate is the **highest** in the table — 875.3
against 824.8 tok/s at 32K — which is what `--prefill auto` chunking plus a warm expert
cache looks like at this size, not a mistake: the three runs land within 2 % of each
other (862.5–882.2).

**Concurrency (4 clients, one slot).** This is a **queueing** measurement, not a batching
one: the config has no `--batch`, `/props` reports `total_slots: 1`, so the four requests
cannot decode together and the burst answers "what does a second request cost a user who
is already generating?". Per round, 4 x 128 = 512 tokens in 26.0 / 27.2 / 26.2 s of wall
time = **19.55 tok/s total (18.82–19.66)**, 0 errors in 12 requests. The per-request
first-token times in a round step like a queue — 1.71 / 8.15 / 15.14 / 22.75 s in the
slowest round — while each request's own decode is barely touched (26.9 tok/s median,
21.2–30.1, against 27.9 for a lone chat request). The 19.55 tok/s is arithmetic, not a
throughput ceiling: each request spends 1.5–2.1 s on its own 67-token prefill and 4.3–6.0
s generating 128 tokens, and the four of those sum to the 26–27 s of wall time the
aggregate divides 512 tokens by. The cost of concurrency here is the **wait** (p50 16.8 s,
p95 26.6 s per request), not slower generation. Full per-round data:
[`extra-arms.json`](extra-arms.json).

**Recall (needle in a haystack, `tools/needle_bench.py`, thinking off, greedy).** 6 of 6
found. The haystack is the repository's own docs and source files; a unique code word is
inserted at a given depth and the model is asked for it.

| Length | Depth | Prompt tokens | Reused tokens | Found | Code word | Seconds |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| 32k | 10 % | 31,634 | 0 | yes | `meadow-copper-504` | 43.6 |
| 32k | 50 % | 31,634 | 0 | yes | `falcon-quartz-940` | 39.5 |
| 32k | 90 % | 31,635 | 0 | yes | `willow-cobalt-696` | 39.4 |
| 128k | 10 % | 123,372 | 0 | yes | `falcon-saffron-138` | 146.1 |
| 128k | 50 % | 123,370 | 18,944 | yes | `quartz-tundra-528` | 126.5 |
| 128k | 90 % | 123,370 | 56,832 | yes | `quartz-glacier-192` | 80.7 |

The three rows of a length share one haystack and differ only in where the code word
sits, so the deeper rows get a real prefix-cache hit (18,944 and 56,832 reused tokens)
and their seconds drop accordingly — 146.1 → 126.5 → 80.7. The first row of each length
is the fresh-prompt figure (0 reused; the engine's own 847.5 tok/s over 145.6 s of prompt
processing at 123,372 tokens, in line with the 128K arm's 875.3). The verdicts are
independent of that: each row has its own code word, generated fresh. Raw rows:
[`needles.json`](needles.json).

Per-session agent cache accounting (3 sessions, near-identical): 42,769–42,779 prompt
tokens over the six turns, **29,361–29,442 reused** — the tool reports a **69.81 %** hit
rate, computed over the five follow-up turns only; recomputed over all six turns
including the cold first turn it is 68.65–68.82 %. Over those five follow-up turns the
engine did not reuse 12,728–12,799 of those 42,160–42,170 follow-up prompt tokens, and
the tool's own accounting says 12,725 of them were genuinely new file content — so only
**3–74 tokens per session were re-read** (`reread_tokens`: min 3, median 4, worst session
74). The first turn of each session is cold: 609 prompt tokens, 0 reused, TTFT 5.55–6.33 s.

Total latency (client-side, whole request, includes prompt processing):
chat 4.6 s (3.8–4.9), 4K 8.8 s (8.8–9.8), 16K 20.4 s (20.4–21.0), 32K 40.1 s (39.7–41.1),
128K 145.8 s (144.8–148.0), agent turn 9.1 s (7.7–11.1) over the five follow-up turns,
8.7 s (6.0–11.1) over all 18 turns, burst request 16.8 s p50 / 26.6 s p95.

Draft acceptance over the 57 measured requests: **1,896 of 2,594 drafts accepted
(73.1 %)** — chat 539/633 (85.2 %), 4K 31/34 (91.2 %), 16K 30/34 (88.2 %), 32K 23/39
(59.0 %), 128K 33/36 (91.7 %), agent 508/760 (66.8 %), burst 696/999 (69.7 %), recall
36/59 (61.0 %). The engine's cumulative counter (64 requests, including the seven
warm-up and calibration requests) reads 1,927/2,627 (73.4 %).

Memory and power observations:

- VRAM, host view (`nvidia-smi`, 437 samples every 3 s): **14,806–15,670 MiB used,
  438–1,302 MiB free** of 16,380 MiB, median 15,375 MiB used. GPU utilisation: median
  91 %, min 1 % (between requests), max 100 %.
- VRAM, engine's own accounting after everything is loaded: **`vram_free_mib: 0`**, and
  the engine printed its own warning (`only 0 MiB of VRAM free at the start of this
  request (below 512 MiB) … #2065`) during the run. The ~1.5 GiB difference is driver /
  CUDA-context overhead the engine does not count as usable.
- **Power and clocks** (2 s samples, 654 rows, 0 failed): power draw median **116.9 W**
  (min 39.7, p90 127.9, max 139.1) against a **165 W** limit — the card never hit its
  power cap. SM clock median 2,670 MHz (2,625–2,685, no sustained throttling), memory
  clock 8,751 MHz, temperature median 72.5 °C (max 84 °C). Summary:
  [`power-summary.json`](power-summary.json); the 654-row file is in the run record.
- **What 128K costs the expert cache:** the engine grows K/V on demand, and at 131,072
  cells it took **228 slots away from the expert cache** to make room —
  `K/V grown to 131072 cells (1.20 GiB); the expert cache gave 228 slots for it, 123
  hotter experts moved to colder ones' slots (2066 of 2294 slots hold experts) - and new
  VRAM, the cache being at its floor`. So the 128K arm ran with **2,066 of 2,294 expert
  slots**, not 2,294; after the arm the engine trimmed K/V back and returned all 228.
  The 32K arm cost 61 slots by the same mechanism.
- Host RAM: engine RSS 24.9 GiB in the first telemetry sample (taken during load) →
  **63.8 GiB peak** (mostly the page-cache mapping of the 88 GiB shard set); guest
  `MemAvailable` never fell below **65.3 GiB** (read from the telemetry field labelled
  `mem_available_mib`, which in this harness carries a kB value — converted here);
  **swap-out 0 MiB/s in all 436 samples**; load1 5.03–9.38 (median 6.5) on 16 threads.
- Storage: the run did **not** exercise the SSD path — across all 64 requests the engine
  logged its own I/O line, the OS read **0.1–2,906.2 MB** from storage with **0 major
  faults in every one of the 64 lines**, while serving 20,979–859,023 MB of expert reads
  out of the page cache (`page cache held 100 % of the decode's reads`). The 2.9 GB peak is
  one 128K recall request, right after K/V had grown to 131,072 cells — i.e. the experts
  that K/V growth displaced had to be re-read; a cold page cache (a first request after
  boot) is a different, unmeasured condition here.
- No paging, no OOM, no engine restart, no failed or cancelled requests (0 failures,
  0 skipped, 0 burst errors, 6/6 recall).

## Correctness and limitations

- **Recall is 6 checks, not a capability score.** `tools/needle_bench.py` found the code
  word in 6 of 6 cases (32k and 128k haystacks, depths 10/50/90 %, greedy, thinking off),
  so retrieval and attribute-binding held at 123,372 tokens in this sample. That is a
  pass-rate over six single-word lookups in one haystack type (the repo's own text); it
  is not a coding benchmark, not a multi-needle test, and not a long-context quality
  score. No task-level correctness was measured beyond it.
- **The 128K arm is 3 requests at one length.** 126,936 tokens is the largest prompt the
  tool's own context guard allows at 131,072, so the last ~4k tokens of the window are
  not covered by any row here, and the 128K figures are n=3.
- **Decode numbers are not comparable across reports with different output caps — and
  they do not even reproduce between two runs here.** The 4K/16K/32K/128K arms generate
  only 24 tokens each, so their decode column is a short sample dominated by the first
  windows; reports that cap at 256 output tokens measure a longer, steadier decode. The
  two sessions of this same config differ by +11.6 % (16K) and −10.0 % (32K) on that
  column while agreeing within 2.4 % on prefill, so treat the 24-token decode cells as an
  indication. The chat arm (100 tokens), the burst (128 tokens) and agent turns (64
  tokens) are the more representative decode figures here.
- **The burst row is queueing, not batching.** With `total_slots: 1` and no `--batch` the
  four clients cannot decode together, so 19.55 tok/s is what four sequential slots look
  like, not what a batched server would produce. Do not compare it against a batched
  server's aggregate throughput.
- **The card is completely full.** `--expert-cache 2700` is clamped to 2,294 slots
  (5,413 MiB) and the engine ends up with 0 MiB free by its own accounting (the host sees
  438 MiB free at the tightest point), which it flags as a stall risk. Nothing stalled in
  this run: decode stayed 18.1–37.7 tok/s across the 57 measured requests, swap-out was
  0 MiB/s in every sample and there were 0 major faults — but a longer context or a
  second slot on this card is not covered by these numbers. Anyone repeating this should
  note that the effective expert cache is 2,294 slots, not 2,700, and that at 131,072
  cells K/V takes 228 of them (2,066 left).
- **The conversation cache never paid back in this workload.** Over the session it parked
  49 times (58–126,959 tokens per park; median 92.1 ms, p90 393.2 ms, worst 1,292.8 ms;
  largest per-slot snapshot 2.11 GiB of the 4,096 MiB budget) and every one of the 49
  lines reports `reused_kv_bytes: 0`, with `restores: 0` in the post-run `/metrics`
  snapshot; the reuse visible in the agent and recall rows came from the prompt cache
  instead. The 4,096 MiB budget is therefore not justified by this suite, and this run
  does not show a benefit for it.
- **WSL2 specifics:** the guest sees the SATA SSD as a rotational virtual disk (so the
  engine's prefill plan underestimates the drive), and the reported 78.5 GiB RAM is the
  WSL2 allocation rather than the Windows physical total. Windows-side memory pressure
  was not sampled in this run.
- **Single engine session, one machine, one model, thinking off, temperature 0.** No
  version or setting comparison is implied by this report; nothing was re-tuned for it.

## How to reproduce

Engine started with the arguments above (the machine's normal configuration, nothing
added for the benchmark), then from the `v0.2.12` source tree:

```bash
# phase 1 — the published suite, unchanged
python tools/strata_bench.py --url http://127.0.0.1:<port> --runs 3 --long --out <dir>

# phases 2-3 — the tool's own methods, called from python
#   Suite(base, runs=3, think=False).calibrate(); Suite.run_prompt(130000, "128K");
#   Suite.run_clients(4)          # tools/serve_load.py, 400-word prompts, 128 new tokens

# phase 4 — the repo's own recall checker
python tools/needle_bench.py --url http://127.0.0.1:<port> --lengths 32k,128k --depths 10,50,90 --out <dir>/needles.json
```

Wall time: 419 s for phase 1, 524 s for phases 2-3 (three ~127k-token prefills dominate
it), 476 s for phase 4 — 1,419 s (23.7 min) of measurement in one engine session. The
run record spans 24 min 18 s from supervisor start to production restore, which includes
the quiet-mode drain, model load and warm-up.

## Files in this folder

| File | What it is |
| --- | --- |
| `results.json` | the suite's own machine-readable output for the session these numbers come from (per-request records) |
| `extra-arms.json` | the 128K arm and the 4-client burst, per-request and per-round |
| `needles.json` | the 6 recall rows, including the code word and answer text |
| `power-summary.json` | 654 power/clock/temperature samples reduced to median/min/max/p90 |
| `engine-log-excerpt.txt` | the engine's own log for the whole session (paths sanitised) |
| `results-session-145725.json` | the first session's suite output, as a reproducibility check |
| `engine-log-excerpt-session-145725.txt` | that first session's engine log |
