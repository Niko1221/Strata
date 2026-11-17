# Community benchmark on an RTX 5070 12 GB over OcuLink (PCIe 4.0 x4), 32 GB RAM

Measured on 2026-11-12 and 2026-11-13 by Amost-. Qwen3.8-Flash-Next Gyro-S at a 64K context on a mini PC with an
external graphics card. Main limitation: 31 GB of RAM, so only part of the CPU-side experts fit in RAM and the PC has
0.2 to 0.4 GB free under load. Sample sizes are small (2 runs per configuration) and run-to-run spread on this PC is
about ±15%.

This folder holds the measurements that are outside the standard protocol. The eight standard-protocol runs
(`strata-qwen38-thinking-v1-draft.1`) are in their own folders, written by `strata bench run`. The last four include
the optional concurrency stage:

| Folder | Track | Config |
| --- | --- | --- |
| [`2026-11-13-standard-rtx-5070-52faa0c2`](../2026-11-13-standard-rtx-5070-52faa0c2/README.md) | default | setup's config, first sample |
| [`2026-11-13-standard-rtx-5070-d00e99dc`](../2026-11-13-standard-rtx-5070-d00e99dc/README.md) | default | setup's config, second sample |
| [`2026-11-13-standard-rtx-5070-6ab06385`](../2026-11-13-standard-rtx-5070-6ab06385/README.md) | custom | plus `STRATA_RESIDENT_HEADROOM_GIB=12`, first sample |
| [`2026-11-13-standard-rtx-5070-6212acad`](../2026-11-13-standard-rtx-5070-6212acad/README.md) | custom | plus `STRATA_RESIDENT_HEADROOM_GIB=12`, second sample |
| [`2026-11-13-standard-rtx-5070-f6feca5e`](../2026-11-13-standard-rtx-5070-f6feca5e/README.md) | custom | headroom 12, `"drafting": "mtp"`, arm a (section 5) |
| [`2026-11-13-standard-rtx-5070-0d879070`](../2026-11-13-standard-rtx-5070-0d879070/README.md) | custom | headroom 12, `"drafting": "dflash2"`, arm a |
| [`2026-11-13-standard-rtx-5070-6f87b8c9`](../2026-11-13-standard-rtx-5070-6f87b8c9/README.md) | custom | headroom 12, `"drafting": "mtp"`, arm b |
| [`2026-11-13-standard-rtx-5070-01af8900`](../2026-11-13-standard-rtx-5070-01af8900/README.md) | custom | headroom 12, `"drafting": "dflash2"`, arm b |

## Hardware and software

- GPU and VRAM; CPU; installed RAM; storage; PCIe link: NVIDIA GeForce RTX 5070, 12 GB, as an external card on OcuLink,
  PCIe 4.0 x4 (the engine measured 7.2 GB/s); AMD Ryzen 7 8845HS (8 cores, 16 threads) with Radeon 780M; 32 GB
  DDR5-5600 (30.8 GiB usable); 1 TB NVMe SSD (PCIe 4.0). Windows page file 32 to 48 GB.
- OS; driver; CUDA: Windows 11 Pro 10.0.26200; NVIDIA driver 617.42; CUDA 13.4 runtime.
- Strata commit; engine version; release binary or source build: Terminal 0.2.13 (Control 0.1.59, commit
  `d2f096475817d4666c364c820b0ca830f751cfc5`), engine 0.1.59 `windows-x86_64-cuda-cuda13`, both release binaries
  (`install.ps1` and the signed feed). Section 3 was measured one day earlier with the Terminal 0.2.6 server
  (build 0.1.51) and the same engine 0.1.59.
- Background workloads and any power limits: the Windows desktop was driven by the integrated Radeon 780M, so the RTX
  5070 reported no display (except where section 3 says otherwise); Claude desktop app open and idle; LM Studio
  background service with no model loaded. GPU power limit 275 W (default).

## Model and configuration

- Model repository and revision; quantization; GGUF filenames: `agentionai/Qwen3.8-Flash-Next-Gyro-GGUF` at
  `1ada08422e404aca51ab9df665f089f34c419c6b`; Gyro-S (TQ1_0); `Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf`, SHA-256
  `c6ea1b08ca312f99095b27588ba35976b53f948268e7f91e7761926aa01d9c87`.
- Vision encoder; custom packs or profiles: pictures off; the pack and the expert profile are the ones setup prepares.
- Context; KV type and streaming; cache; prefill; low-RAM mode: 65,536 tokens; `--kv int8`, `--kv-resident 32768`;
  `--expert-cache auto` (4,503 of 24,576 experts in VRAM, 4.40 GiB); `--prefill auto`; `--resident-experts`.
- MTP; reasoning; sampling; calibration; experimental speed projection: MTP drafting (`--spec 4 --spec-min-p 0.5`;
  setup's `auto` would pick DFlash2; sections 1 to 4 use the MTP, section 5 compares the two); thinking on; temperature 0.6 for sections 1 to 3 (the standard runs
  use the protocol's sampling); no calibration; no speed projection.

```text
strata install -- --family gyro --model Gyro-S --context 65536 --vision no --drafter mtp --port 8091 --no-start

"default" arm:    the run config setup wrote
"headroom12" arm: the same file plus  "env": {"STRATA_RESIDENT_HEADROOM_GIB": "12"}

python serve/server.py --engine strata --config <run config> --port 8091
```

## Method

Script: [`long-context/bench_stream.py`](long-context/bench_stream.py), `python bench_stream.py http://127.0.0.1:8091
qwen3.8-flash-next-gyro-gyro-s 30000,60000`, over loopback, streaming, one request at a time. For each arm the server
was started fresh (model loading excluded), then in order: three short prompts with a 600-token cap; one 11K-token
calibration prompt; a fresh prompt of about 30K tokens of generated word lines with two facts hidden at the start and
the middle, and a question about them; a follow-up message on that conversation with a 600-token cap; the same at
about 60K. Each long prompt carries random content, so nothing is reused except by the follow-up. Arms ran in the
order default, headroom12, default, headroom12, each from a new process. There is no separate warm-up: the first
short request of each arm is the first request after loading.

Prompt and decode tok/s are the engine's own timing lines
([`long-context/*.engine.log`](long-context/)); first-token time is measured by the client and its first token is
reasoning. Reasoning tokens count as generated tokens. Memory figures are Windows' free physical memory and
`nvidia-smi` after the last request of an arm. Client-side output:
[`long-context/bench_stream-output.txt`](long-context/bench_stream-output.txt).

## Results

### 1. Long prompts, default against `STRATA_RESIDENT_HEADROOM_GIB=12`

Two arms of each; the two values are run a and run b.

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s (a, b) | Decode tok/s (a, b) | TTFT seconds (a, b) |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| default, 30K fresh | 30,553 | 0 | 2,252 / 2,253 | 2 | 1,348 / 1,238 | 51.8 / 50.4 | 22.7 / 24.8 |
| headroom12, 30K fresh | 30,553 | 0 | 2,251 / 2,252 | 2 | 1,404 / 1,434 | 45.2 / 50.1 | 21.8 / 21.4 |
| default, 60K fresh | 62,184 | 0 | 2,251 / 2,252 | 2 | 1,382 / 1,109 | 56.4 / 57.4 | 45.1 / 56.2 |
| headroom12, 60K fresh | 62,184 | 0 | 2,253 / 2,250 | 2 | 1,006 / 1,440 | 41.0 / 57.7 | 61.9 / 43.3 |
| default, follow-up at 30K | 30,901 / 30,878 | 30,548 | 600 | 2 | not meaningful (about 350 new tokens) | 45.7 / 48.0 | 1.5 / 3.0 |
| headroom12, follow-up at 30K | 30,885 / 31,117 | 30,548 | 600 | 2 | not meaningful | 44.3 / 50.7 | 3.6 / 1.7 |
| default, follow-up at 60K | 62,619 / 62,594 | 62,179 | 600 | 2 | not meaningful | 41.6 / 47.5 | 1.8 / 2.0 |
| headroom12, follow-up at 60K | 62,569 / 62,555 | 62,179 | 600 | 2 | not meaningful | 50.3 / 51.0 | 1.8 / 2.1 |

Short prompts (74 prompt tokens, 600 generated, three requests per arm, decode tok/s in request order):

| Arm | Run a | Run b |
| --- | --- | --- |
| default | 36.1, 47.1, 44.9 | 21.2, 21.7, 24.0 |
| headroom12 | 31.6, 43.3, 44.9 | 26.6, 48.0, 47.4 |

Memory:

| Arm | Experts locked in RAM | Free RAM after the arm | VRAM used | Conversation cache |
| --- | --- | --- | --- | --- |
| default | 20.21 GiB (a), 19.12 GiB (b) of 19.62 GiB | 0.4 GB, 0.3 GB | 10.6 GB | off: "not enough free RAM to park conversations" |
| headroom12 | 12.13 GiB (a), 11.54 GiB (b) | 0.2 GB, 0.3 GB | 10.6 GB | on: 3,328 MiB for up to 4 parked conversations (but see section 6: it rarely parks) |

What this shows, and what it does not:

- Neither arm is consistently faster. The standard runs agree: decode medians of 43.3 / 44.0 / 42.5 and
  42.8 / 42.8 / 43.0 tok/s (short / 4K / 16K) on the default track against 40.8 / 41.2 / 36.5 and 41.1 / 39.8 / 42.2 with
  headroom 12.
- A run can start slowly and recover. In default run b the three short requests and the 11K calibration prompt ran at
  21 to 24 tok/s decode and 414 tok/s prompt, then the 30K and 60K prompts ran at normal speed. The first request
  after a load was the slowest in every arm. I did not find the cause.
- The headroom setting changes what the engine does with RAM, not the speed: with the default the conversation cache
  turns itself off on this PC; with headroom 12 it is on. In practice that made little difference: with 0.1 to 0.6 GB
  of RAM free under load, the cache's own admission check ("need N MiB plus 2560 MiB floor") refused to park in 748 of
  776 attempts over the whole engine log (section 6).
- One day earlier, with the Terminal 0.2.6 server, I measured the default at 38 to 43 tok/s (short) and 48 (long)
  against 46 to 49 and 57 with headroom 12, and concluded the default was slower. The measurements above do not
  reproduce that difference, so I withdraw that conclusion.

### 2. Recall and tool calls

- The two hidden facts were found in all eight long prompts (30K and 60K, four arms).
- Separately, prompts of 40,753, 57,253 and 63,853 tokens with three codes hidden at the start, middle and end:
  3 of 3 found each time.
- Standard protocol: 12 of 12 sanity checks in all eight runs.
- A tool call through `/v1/chat/completions` and a two-step tool loop through `/v1/messages` both worked.

### 3. Desktop on the graphics card against on the integrated GPU (2026-11-12, Terminal 0.2.6 server, engine 0.1.59)

Same config (headroom 12), same day, same script at 30K.

| Desktop driven by | Runs | Decode tok/s, short | Decode tok/s, 30K | Prompt tok/s, 30K |
| --- | ---: | --- | --- | --- |
| RTX 5070 (it reports a display) | 4 | 33 to 41 | 42 to 48 | about 1,170 |
| Radeon 780M (the RTX reports no display) | 2 | 46 to 49 | 57 | about 1,370 |

Per-request data for this section was not kept, only the ranges.

### 4. Settings tried on 2026-11-12, none faster (Terminal 0.2.6 server, engine 0.1.59)

Fresh server per arm, two passes at 30K, decode tok/s (short / 30K) for pass 1 and pass 2. Prompt speed was 1,380
to 1,440 tok/s in every arm.

| Arm | Short | 30K |
| --- | --- | --- |
| Baseline, start of session | 41.1, 47.2 | 56.1, 52.4 |
| `STRATA_RAM_ADAPT=1` | 37.5, 48.7 | 55.9, 57.0 |
| `--host-core sibling` | 34.7, 47.9 | 51.3, 52.0 |
| `--host-core sibling --pool-cores 0,2,4,6,8,10,12,14` | 46.9, 51.1 | 57.7, 60.1 |
| `STRATA_ADAPT_OVERLAP=1` + `STRATA_ADAPT_OVERLAP_RESIDENT=1` | 47.3, 50.6 | 57.1, 57.5 |
| Baseline, end of session | 48.2, 51.0 | 58.4, 59.1 |

No arm beats the closing baseline by more than 3%, and the two baselines differ by more than that.

### 5. DFlash2 against the MTP on Gyro-S (2026-11-13 afternoon)

Drafter fetched with `strata drafting fetch` (`qwen3.8-flash-next-dflash2.q4_0-selq4.pack`). Same run config for
both arms (headroom 12) with the drafter chosen by name: `"drafting": "mtp"` or `"drafting": "dflash2"`. Arms ran in
the order MTP, DFlash2, MTP, DFlash2, each from a new process: the standard benchmark with the optional concurrency
stage, then two passes of `bench_stream.py` at 30K and 60K on a fresh server. Engine logs per arm:
[`drafter-compare/`](drafter-compare/).

| | MTP (arm a, arm b) | DFlash2 (arm a, arm b) |
| --- | --- | --- |
| Decode tok/s, all requests of 150+ generated tokens, pooled from the engine log (40 requests per arm) | 54.3, 51.8 | 57.2, 57.9 |
| Both arms pooled (80 requests) | 53.0 | 57.5 |
| Standard benchmark, decode medians short / 4K / 16K | 50.9 / 54.9 / 53.5 and 48.1 / 44.6 / 47.9 | 52.7 / 58.6 / 51.5 and 53.4 / 60.8 / 50.4 |
| Four requests at once (server capacity 1), tok/s in total per round | 45.1, 44.9, 49.1 and 44.7, 43.5, 45.1 | 47.2, 49.7, 52.0 and 48.1, 49.5, 51.7 |
| Decode tok/s after a fresh 30K prompt (client, 4 runs) | 58.8, 57.3, 58.2, 60.2 | 56.3, 67.1, 62.2, 65.0 |
| Decode tok/s after a fresh 60K prompt (client, 4 runs) | 61.8, 57.1, 56.3, 60.3 | 59.5, 62.0, 58.3, 59.3 |
| Prompt tok/s, prompts of 10K+ tokens (engine log, 11 per arm) | 1,330, 1,364 | 1,327, 1,383 |
| Drafts accepted | 15,202 of 21,209 and 15,593 of 21,527 (72%) | 14,291 of 16,026 and 14,338 of 16,056 (89%) |
| Expert cache slots; VRAM free with everything loaded | 4,503; 671 MiB | 4,757; 765 MiB |
| Sanity checks | 12 of 12, 12 of 12 | 12 of 12, 12 of 12 |
| My 127-item check, thinking on | 126 and 123 (2026-11-12) | 126 |

DFlash2 was ahead in both pairs: about 8% on pooled decode speed, about 9% on the four-at-once totals, with prompt
speed, recall and correctness unchanged. With 2 arms each and a spread of about ±15% between identical single
requests, this is "somewhat faster", not a precise figure.

The MTP standard runs of this section (48 to 55 tok/s) are faster than the morning's custom-track runs with the same
config (40 to 42). The PC had been serving for hours by the afternoon; I only compare arms alternated in one session.

### 6. Soak, a second client, power (2026-11-13 afternoon, MTP, headroom 12, one server process)

Scripts and raw logs: [`soak/`](soak/).

- **Soak, 2 hours** (`soak.py 7200`): 319 requests, 0 failed, on a server that had already been up for 3 hours.
  Recall prompts of 20,137, 41,065, 56,828, 62,093 and 64,736 tokens (medians): 20 of 20 at each size, three hidden
  codes each. Prompt speed on the 62K prompt was 1,440 to 1,464 tok/s in every one of 20 cycles; 6,000-token answers
  decoded at 48 to 53 tok/s in every cycle. 40 client disconnects (20 during the prompt, 20 during decoding), each
  followed by a normal request. A prompt of about 74,000 tokens was refused with HTTP 400, 19 of 19. Free RAM was 0.1
  to 0.3 GB at the end of every cycle.
- **A second client** (`concurrency.py`): requests are served one after another, and the total stays at 43 to 47
  tok/s for one, two or four simultaneous short requests. A follow-up on a 30K conversation reused 30,415 of 30,452
  tokens (first token 0.5 s); after one unrelated 400-token request it reused 0 and took 21 to 32 s. The engine log
  shows why: "conversation cache: skip parking (physical RAM admission; need 232 MiB plus 2560 MiB floor, 298 MiB
  available…)". Over the whole log: parked 28 times, skipped 748 times.
- **Power** (`monitor.py`, `nvidia-smi` every 5 s, the card's own sensor, not wall power): 144 W while reading a long
  prompt (540 samples), 108 W while decoding a long answer (453 samples), 3 to 6 W idle with the model loaded, 160 W
  peak, 123 W mean over the soak. 62 °C at most. About 0.6 Wh per 1,000 generated tokens on the card.

## Correctness and limitations

- Two runs per configuration. Differences under about 15% are within what this PC shows between identical runs.
- Sections 3 and 4 were measured with an older Terminal server (0.2.6) and only their summaries were kept.
- The Gyro-S pack is the smallest Flash-Next option; a separate 127-item check of my own (arithmetic, strings, Python
  output, 20 coding tasks run against tests, 10 tool tasks) scored 126 and 123 of 127. That check is near its ceiling
  and is not a ranking.
- Not measured: pictures, the serving suites (`parallel-*`), Gyro-M, power at the wall.
