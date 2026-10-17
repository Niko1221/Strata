# Community benchmark on RTX 3080 Ti 12 GB (KVM guest with GPU passthrough, Docker)

Measured on 2026-10-11 by moneymikeMD. One RTX 3080 Ti serving IQ3_S, UD-IQ4_XS and Swift 1.5 IQ3_S on engine
0.1.42, built and run from the repository's Dockerfile inside a Linux KVM guest that gets the card by PCI
passthrough. Main limitations: there is no bare-metal run of the same machine, so nothing here measures what the
virtual machine or Docker cost; three runs per cell; the context ladder and the batch-slot section were measured
earlier the same day with a different script and different prompt sizes, and say so.

## Hardware and software

| | |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3080 Ti, 12 GB (12,288 MiB), power limit 350 W, the only card given to the container |
| PCIe | Gen 4 x16 under load (reads Gen 1 at idle); the engine's probe: 24.9 GB/s host->device |
| Other GPU | a second NVIDIA card (12 GB) in the same guest, not visible to the container; it serves a small embedding model for an unrelated workload (2.4 GB of its VRAM, occasional requests) throughout |
| CPU | Intel Xeon w7-2475X, 20 cores / 40 threads, AVX-512; the guest has 32 vCPUs (`cpu: host`, one socket of 32 cores, not pinned) |
| CPU workers | engine default here: 31 pool workers plus the host thread |
| RAM | host 128 GB, 4 x 32 GB DDR5-4800 RDIMM (quad channel); guest 96 GiB (94 GiB visible), ballooning off, no swap, no huge pages (`MAP_HUGETLB unavailable`, transparent huge pages requested) |
| Storage | NVMe SSD (4 TB) -> LVM-thin -> virtio-scsi virtual disk (`cache=none`, `aio=io_uring`); experts load at 4.8 to 5.4 GiB/s |
| Machine | workstation-class PC |
| Host | Proxmox VE 9.2.2, kernel 7.0.2-6-pve; guest is q35 / OVMF with both GPUs passed through |
| Guest | Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic, NVIDIA driver 615.71.09 |
| Container | Docker 29.9.0, NVIDIA Container Toolkit 1.20.1-1; image from this repository's `Dockerfile` at v0.1.42 (commit 61b3fb5) with `--build-arg CUDA_ARCHITECTURES=86`, base `nvidia/cuda:13.0.0-devel-ubuntu24.04` (CUDA 13.0) |
| Engine | 0.1.42, source build in the image (`BUILD.json`: src `1e0c97588848ea0f`) |
| Background | the embedding model on the other card; a few idle shell sessions in the guest |

## Model and configuration

- IQ3_S: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, revision `ed59f920` (the one setup pins), shards
  54,817,524,224 B and 28,800,138,432 B.
- UD-IQ4_XS: `unsloth/Qwen3.8-Flash-Next-GGUF`, revision `38bb39ee`, shards 10,946,624 B, 49,835,229,856 B and
  43,836,407,744 B.
- Swift 1.5 IQ3_S: `ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, shards 44,900,816,736 B and
  38,842,920,384 B. Setup reported both shards as "not at the pinned revision any more" and took the repository's
  current files, so the revision is not the pinned one and was not recorded.
- Packs and the MTP layer were prepared by setup through the container entry point. Vision off. Experimental speed
  projection off.
- Context 262,144 for the standard sweep. KV int8, KV streaming on (`--kv-resident 32768`, 3.6 GB of KV in RAM).
  `--expert-cache auto`: 2,337 slots, 4.49 GiB of VRAM for IQ3_S at this context. `--prefill auto`, `--spec 4`,
  `--spec-min-p 0.5`. The engine moved its own PCIe share from the start-up 0.55 to 0.15-0.20 while serving.
- Requests: `reasoning_effort` none, `temperature` 0, `seed` 1, streamed.
- `"fit_max_tokens": true` was added to each config by hand. Nothing else differs from what setup wrote in the
  "stock" rows. The "calibrated" rows use what `setup.py --calibrate` chose (below).

```text
docker build -t strata:v0.1.42 --build-arg CUDA_ARCHITECTURES=86 .
docker run -d --name strata --gpus device=<GPU UUID> -p 127.0.0.1:8080:8080 --ulimit memlock=-1 \
  -v strata-data:/data --env-file strata.env \
  -e FAMILY=qwen -e MODEL=IQ3_S -e CONTEXT=262144 -e VISION=no strata:v0.1.42
```

`strata.env` holds `API_KEY=<redacted>`. The sanitized run configs are the `config-*.json` files.

## Method

- `sweep.py` (in this folder) cuts each prompt from this repository's `docs/*.md` and `README*.md` in sorted path
  order, sizes it with `/v1/messages/count_tokens`, and asks for a 200-word summary with a 256-token cap. "short" is a
  one-sentence question. Every prompt starts with a fresh nonce, so reused tokens are 0 in every row and each run
  reads the whole prompt.
- Three runs per cell, all published in the `sweep-*.jsonl` files, with the median and range below.
- Prompt and decode tok/s are the engine's own, from the response's `timings` block. Time to first token is measured
  at the client and includes the prompt read. The model was loaded and had served requests before each sweep; load
  time is not in any figure.
- Memory figures are snapshots after load, not peaks.

## Results

### Standard sweep, IQ3_S, context 262,144

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| IQ3_S stock, short | 64 | 0 | 88 | 3 | 119 (83-123) | 76.7 (53.6-78.0) | 0.56 (0.52-0.82) |
| IQ3_S stock, 4k | 4,084 | 0 | 256 | 3 | 1807 (1561-1827) | 93.9 (84.4-96.6) | 2.29 (2.26-2.64) |
| IQ3_S stock, 32k | 32,841 | 0 | 256 | 3 | 2274 (2273-2291) | 81.7 (81.1-84.4) | 14.48 (14.39-14.49) |
| IQ3_S stock, 128k | 131,081 | 0 | 256 | 3 | 2180 (2172-2190) | 75.2 (74.2-76.1) | 60.25 (59.95-60.47) |
| IQ3_S calibrated, short | 61 | 0 | 88 | 3 | 112 (95-124) | 76.4 (63.9-80.8) | 0.57 (0.50-0.69) |
| IQ3_S calibrated, 4k | 4,084 | 0 | 256 | 3 | 1808 (1601-1814) | 93.6 (85.1-94.4) | 2.28 (2.28-2.58) |
| IQ3_S calibrated, 32k | 32,843 | 0 | 256 | 3 | 2255 (2255-2263) | 83.0 (81.9-85.4) | 14.61 (14.55-14.61) |
| IQ3_S calibrated, 128k | 131,077 | 0 | 256 | 3 | 2182 (2180-2185) | 74.6 (74.4-75.8) | 60.20 (60.11-60.26) |

The short answer stops by itself at 88 tokens, so its decode figure includes the slow first windows. On 400-token
answers to the same kind of question the same card decodes at 96 to 105 tok/s (earlier runs, below).

Memory after load, IQ3_S: 11,506 MiB of VRAM; guest `free -g` 57 used, 37 available of 94. Experts: 46.84 GiB into
RAM in 19 s on a warm file cache, 62 s on the first load.

### Standard sweep, UD-IQ4_XS and Swift 1.5 IQ3_S, context 262,144

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| UD-IQ4_XS stock, short | 63 | 0 | 82 | 3 | 75 (45-75) | 37.0 (27.9-38.3) | 0.88 (0.87-1.40) |
| UD-IQ4_XS stock, 4k | 4,080 | 0 | 256 | 3 | 866 (763-1005) | 43.7 (38.6-46.2) | 4.76 (4.09-5.38) |
| UD-IQ4_XS stock, 32k | 32,841 | 0 | 256 | 3 | 1676 (1665-1695) | 39.8 (39.5-51.1) | 19.64 (19.42-19.78) |
| UD-IQ4_XS stock, 128k | 131,078 | 0 | 256 | 3 | 1687 (1681-1688) | 37.6 (37.3-44.7) | 77.81 (77.79-78.10) |
| UD-IQ4_XS calibrated, short | 61 | 0 | 88 | 3 | 110 (87-110) | 48.0 (32.0-49.2) | 0.59 (0.58-0.80) |
| UD-IQ4_XS calibrated, 4k | 4,081 | 0 | 256 | 3 | 817 (720-1056) | 54.9 (48.8-56.8) | 5.02 (3.89-5.70) |
| UD-IQ4_XS calibrated, 32k | 32,842 | 0 | 256 | 3 | 1691 (1689-1706) | 53.0 (51.6-53.9) | 19.46 (19.29-19.49) |
| UD-IQ4_XS calibrated, 128k | 131,081 | 0 | 256 | 3 | 1690 (1690-1691) | 55.5 (51.9-57.4) | 77.68 (77.63-77.69) |
| Swift 1.5 IQ3_S stock, short | 59 | 0 | 87 | 3 | 80 (67-88) | 59.7 (45.3-81.9) | 0.87 (0.69-0.95) |
| Swift 1.5 IQ3_S stock, 4k | 4,083 | 0 | 256 | 3 | 1815 (1590-1817) | 86.4 (83.6-94.3) | 2.27 (2.27-2.59) |
| Swift 1.5 IQ3_S stock, 32k | 32,841 | 0 | 256 | 3 | 2272 (2262-2274) | 82.8 (82.1-86.5) | 14.50 (14.48-14.57) |
| Swift 1.5 IQ3_S stock, 128k | 131,079 | 0 | 256 | 3 | 2182 (2180-2187) | 79.1 (77.2-80.0) | 60.20 (60.04-60.23) |

UD-IQ4_XS keeps 55 GiB of experts in RAM (`--resident-budget-gib 55`, setup's choice; its 59.5 GB of experts are 55.4 GiB). Calibration raised its
decode by 26% at 4K, 33% at 32K and 48% at 128K; the worker count is the setting that moved most (below). Swift 1.5 runs at the original's speed.

### Calibration (`setup.py --calibrate`)

Run in a one-off container with the entry point overridden (see the notes). Decode tok/s on calibrate's own workload,
median of 3 and range:

| Setting | IQ3_S | UD-IQ4_XS |
| --- | --- | --- |
| `--pcie-frac` 0 | 85.7 (81.7-85.8) | 45.2 (39.5-46.6) |
| `--pcie-frac` 0.2 | 84.2 (82.8-84.8) | 44.5 (43.7-49.1) |
| `--pcie-frac` 0.35 | 81.0 (79.2-85.4) | 46.2 (43.1-49.3) |
| `--pcie-frac` 0.55 | 69.1 (66.6-69.7) | 43.7 (39.3-43.9) |
| `--pcie-frac` 0.75 | 55.6 (53.6-56.0) | 35.3 (32.5-36.3) |
| `--pcie-frac` 1 | 48.0 (46.3-48.9) | 32.4 (31.5-32.7) |
| `--spec-min-p` 0.3 / 0.5 / 0.7 | 78.9 / 81.1 / 84.0 | 47.0 / 44.6 / 50.1 |
| 31 pool workers (default) | 82.8 | 35.8 |
| 21 pool workers | 82.8 | 54.7 |
| 16 pool workers | 81.4 | 46.9 |
| 8 pool workers | 64.0 | 44.3 |
| Chosen | `--pcie-frac 0.00 --spec-min-p 0.70 --adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97` (96.4 tok/s) | `--pcie-frac 0.35 --spec-min-p 0.70 --pool-workers 21 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97` (57.5 tok/s) |

For IQ3_S the calibrated settings change nothing measurable in the sweep above: the engine had already lowered its
PCIe share by itself. Worker count: on this 32-vCPU guest IQ3_S is flat from 31 down to 16 workers; UD-IQ4_XS is not
(35.8 tok/s with the default 31, 54.7 with 21). An earlier run of IQ3_S with `--pool-workers 19` held the same decode speed (99 to 103 tok/s on
400-token answers) with the guest's CPU at 63% busy instead of 100%.

### Context ladder, IQ3_S (earlier script, see "Other measurements")

| Context | Expert cache slots (GiB) | KV in RAM | Decode, 400-token answer | Prompt tok/s at ~30K | Longest prompt tried: tokens, prompt tok/s, decode tok/s, TTFT |
| ---: | --- | ---: | --- | --- | --- |
| 131,072 | 2,454 (4.71) | 1.7 GB | 101.0, 102.5 | 2,264 to 2,387 | 96,450: 2,253 to 2,275, 73.6 to 80.5, 42.5 s |
| 204,800 | 2,387 (4.59) | 2.8 GB | 97.0, 102.2 | 2,073 to 2,142 | 174,002: 2,164 to 2,172, 78.1 to 78.5, 80.3 s |
| 262,144 | 2,337 (4.49) | 3.6 GB | 95.7, 105.4 | 2,123 to 2,205 | 220,801: 2,107 to 2,114, 72.4 to 76.8, 104.7 s |
| 524,288 (yarn x2) | 2,104 (4.04) | 7.2 GB | 91.7, 95.0 | 1,996 to 2,050 | 397,800: 1,750 to 1,751, 39.5 to 46.6 on a 7-token answer, 227.6 s |

### Batch slots on a 12 GB card, IQ3_S, context 204,800

Two 2,000-token decodes sent at the same moment, short prompts, one run each:

| Configuration | Expert cache slots | One stream alone, tok/s | Two streams, tok/s each | Both done after | Tokens per second overall |
| --- | ---: | --- | --- | ---: | ---: |
| one slot (requests queue) | 2,387 | 97.0, 102.2 | 79.8 and 86.4, one after the other | 49.4 s | 81.0 |
| `"parallel": 2` | 1,488 | 54.5, 60.4 | 25.9 and 25.5 | 79.6 s | 50.2 |
| `"parallel": 2` with `--batch-mtp` | 1,430 | 55.5, 59.2 | 27.3 and 27.5 | 75.0 s | 53.4 |

On this card the second slot costs more expert cache than batching gives back: queueing through one slot finishes
both jobs sooner.

### Agent loop and the conversation cache, IQ3_S

One small real ticket (an infrastructure-as-code change of four to five text files in a private repository) was
given to Claude Code 2.1.296 in headless mode, pointed at this server through `ANTHROPIC_BASE_URL`, 23 tools, no MCP
servers, on a fresh snapshot of the repository each time. The agent may start subagents, whose requests alternate
with the main conversation's. Wall time is the whole session; token counts are Claude Code's own accounting.

| Engine arguments | `--effort` | Wall time | Turns | Uncached input tokens | Cached input tokens | Output tokens | Subagents started |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| default (no conversation cache), context 204,800 | high | 1,397 s | 39 | 1,048,377 | 600,516 | 23,849 | 3 |
| default (no conversation cache), context 204,800 | low | 570 s | 31 | 102,520 | 947,256 | 13,582 | 1 |
| `--conversation-cache-mib 8192 --conversation-cache-slots 4`, context 262,144 | high | 690 s | 38 | 69,463 | 1,321,408 | 19,274 | 2 |
| same | high | 296 s | 41 | 79,650 | 2,116,097 | 20,274 | 0 |
| same | low | 337 s | 53 | 62,197 | 2,113,749 | 24,552 | 0 |
| same | low | 448 s | 29 | 83,738 | 1,174,057 | 13,722 | 1 |
| same | low | 26 s | 2 | 48,597 | 18,448 | 32 | 0, failed (below) |

- The two groups differ in more than the cache: the cache-on sessions also ran at context 262,144 with the
  calibrated settings, which did not change IQ3_S's speed in the sweep.
- With the conversation cache off, a session that starts subagents re-reads its prompt each time the conversations
  alternate: about a million uncached input tokens in the first row. With it on, every completed session stayed
  under 84,000. The engine log shows the parking (for example "parked 68622 tokens in 755.8 ms").
- The effort level made no consistent difference here: the two `high` sessions took 690 s and 296 s, the completed
  `low` ones 337 s and 448 s. The model did not think less at `low` (55,590 and 57,407 characters of thinking against
  48,728 and 49,839 at `high`).
- One `low` session failed: the model wrote in its thinking that no user task had been given and ended the turn with
  an empty reply, after reading a 33,513-token prompt that ended with the task.
- The sessions could not run the repository's own checks (the harness denied those commands), so none of this says
  whether the edits were right. It is a timing and caching observation, not an agent benchmark.

## Correctness and limitations

- `tools/needle_bench.py --lengths 32k,128k,262k --depths 10,50,90` on calibrated IQ3_S at context 262,144: 9 of 9
  found (`needles-iq3_s.json`; 32,804, 125,535 and 258,539 prompt tokens).
- Context 524,288 with setup's `--rope-scaling yarn --rope-scale 2`: a six-digit code planted a tenth of the way into
  the prompt was returned in 1 of 1 runs at 220,828 tokens and 2 of 2 at 397,800 tokens (earlier script, not
  `needle_bench.py`). One planted fact; no other quality check past the trained length.
- Eight arithmetic and reasoning questions with known answers, one pass each at thinking none, low and high: 8 of 8
  for IQ3_S and for Swift 1.5 IQ3_S at every level. Output tokens including thinking, none / low / high: IQ3_S 3,188 /
  2,955 / 2,330; Swift 1.5 2,718 / 2,146 / 1,475. The gap at high comes from one question (1,319 against 463 tokens).
- Not measured: the same machine without the virtual machine or without Docker; vision; experimental speed
  projection; a soak; more than two simultaneous requests; a layer split or helper cache with the second card;
  thinking-on throughput; any coding or agent quality score.
- The guest shares its host with nothing else, but its 32 vCPUs sit on 20 physical cores and are not pinned.
- Figures in "Context ladder", "Batch slots" and the 400-token decode numbers come from an earlier script with the
  same request settings (thinking off, greedy, seed 1, a fresh nonce) but its own prompt sizes and output caps (64 to
  400 tokens), mostly two or three runs per cell. Its per-run output is in `other-measurements/`.

## Notes for this kind of setup

- **A virtual disk can report itself as rotational.** The guest's virtio-scsi disk sits on an NVMe SSD but
  `/sys/block/sda/queue/rotational` read 1. Setup then treats the model as being on a hard disk and can add
  `--ple-io ram`. Writing 0 to that file before the first start avoided it (it resets on reboot).
- **One card out of two in Docker:** `--gpus device=<GPU UUID>` gave the container the RTX 3080 Ti alone; the engine
  saw it as GPU 0.
- **Health check during the first start:** the image's 600 s start period is shorter than an 84 GB download, so the
  container shows "unhealthy" until the model has loaded. Harmless without a restart policy.
- **`-e REINSTALL=1` belongs on a one-off container.** Left on the serving container it reruns setup, and rewrites
  the config, on every restart.
- **Calibrating in Docker:** `docker run ... --entrypoint sh strata:v0.1.42 -c "cd /opt/strata && ln -sfn
  /data/config/strata-iq3_s.json strata-iq3_s.json && exec .venv/bin/python setup.py --calibrate --port 8080"`. The
  tuned config is written inside the container (the link is replaced by a file), so it has to be copied to
  `/data/config/` to survive.

## Files

- `sweep.py`: the sweep script. `sweep-*.jsonl`: one line per run and one summary line per size; fields are
  described at the top of the script.
- `needles-iq3_s.json`: output of `tools/needle_bench.py`.
- `calibration-iq3_s.log`, `calibration-ud-iq4_xs.log`: output of `setup.py --calibrate`.
- `engine-iq3_s-stock.log`, `engine-iq3_s-conversation-cache.log`, `engine-ud-iq4_xs-stock.log`: the engine logs of
  those runs. The conversation-cache log covers the agent sessions; its prompts are not in it, only token counts.
- `config-*.json`: the run configs with the key removed (stock, calibrated, and calibrated with the conversation
  cache). `BUILD.json`: the image's engine build record.
- `other-measurements/`: per-run JSON lines from the earlier script (context ladder, batch slots, thinking levels,
  Swift 1.5), with a note on their fields.
