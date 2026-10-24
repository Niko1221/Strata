# Community benchmark on RTX 3080 Ti 12 GB, 0.2.0-beta against 0.1.43 (KVM guest with GPU passthrough, Docker)

Measured on 2026-10-19 by moneymikeMD. One RTX 3080 Ti serving IQ3_S, Swift 1.5 IQ3_S and UD-IQ4_XS at context 262,144,
with the 0.2.0-beta image (engine package 0.1.45.3) and, the same day on the same model files, the 0.1.43 image. The two
were run in alternation, a fresh container for each sweep. Main results: IQ3_S and Swift 1.5 decode 20 to 35% faster on
0.2.0 at the same prompt speed; 0.2.0's own settings for UD-IQ4_XS were killed for lack of RAM on this 94 GiB guest in 4
of 4 starts, and with a RAM budget given by hand it read prompts slower than 0.1.43. Main limitations: three runs per
cell and two sweeps per arm; a virtual machine with Docker, and no bare-metal run of the same machine; the 0.1.42 check
is one sweep, run after the others.

## Hardware and software

| | |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3080 Ti, 12 GB (12,288 MiB), power limit 350 W, the only card given to the container |
| PCIe | Gen 4 x16 under load; the engine's probe: 25.1 to 25.2 GB/s host->device |
| Other GPU | a second NVIDIA card (12 GB) in the same guest, not visible to the container; it serves a small embedding model for an unrelated workload (2,397 MiB of its VRAM, occasional requests) throughout |
| CPU | Intel Xeon w7-2475X, 20 cores / 40 threads, AVX-512; the guest has 32 vCPUs shown as one socket of 32 cores, not pinned |
| RAM | host 128 GB, 4 x 32 GB DDR5-4800 RDIMM; guest 94 GiB visible, ballooning off, no swap, no huge pages |
| Storage | NVMe SSD (4 TB) -> LVM-thin -> virtio-scsi virtual disks (rotational reads 0). Model files on the guest's root disk; the 0.1.43 production volume on a second virtual disk of the same SSD |
| Machine | workstation-class PC |
| Host | Proxmox VE 9.2.2; guest q35 / OVMF with both GPUs passed through |
| Guest | Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic, NVIDIA driver 615.71.09 |
| Container | Docker 29.9.0, NVIDIA Container Toolkit 1.20.1-1 |
| 0.2.0 image | `docker/build.sh --flavor cuda` at tag v0.2.0-beta (3d1fb480) with the released engine package `strata-engine-0.1.45.3-linux-x86_64-cuda-cuda13.tar.gz` (SHA-256 7af23df6...0a18, matches the release's SHA256SUMS.txt; its BUILD.json: version 0.1.45.3, archs 75/86/89/120, CUDA 13.0) and base `public.ecr.aws/docker/library/ubuntu:24.04` (same digest as Docker Hub's). One local change, below |
| 0.1.43 image | the repository's Dockerfile at v0.1.43 with `--build-arg CUDA_ARCHITECTURES=86`, base `nvidia/cuda:13.0.0-devel-ubuntu24.04`; engine 0.1.43, source build (BUILD.json src `e568e70371a3d2e0`) |
| 0.1.42 image | the same at v0.1.42 (the image of the [2026-10-11 run](https://github.com/moneymikeMD/Strata/tree/4a57055c/bench/results/2026-10-11-community-rtx-3080-ti-kvm-docker), BUILD.json src `1e0c97588848ea0f`) |
| Background | the embedding model on the other card; a few idle terminal sessions in the guest |

**Local change to build the 0.2.0 image.** At v0.2.0-beta, `docker/build.sh` stops with `missing payload item:
docs/TERMINAL_LAUNCH.md`: its list of files for the build context has `docs/MANAGEMENT_PROTOCOL.md` but not
`docs/TERMINAL_LAUNCH.md`, which `packaging/package_terminal.py` requires. Adding that one path to the `paths=` line was the
only change; the image then built and passed the script's own checks. The same line is unchanged on `main` at f57c0adc.

## Model and configuration

- IQ3_S: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` at `ed59f920` (both versions pin it), the files already on the
  machine, given to both images read-only (`MODEL_PATH` for 0.2.0, `GGUF_DIR` for 0.1.43).
- Swift 1.5 IQ3_S: `ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF` at `99bb8f7f` (0.2.0's pin), downloaded by 0.2.0
  and given to 0.1.43 the same way. 0.1.43 pins an older revision (`b22d729e`) that has no IQ3_S files.
- UD-IQ4_XS: `unsloth/Qwen3.8-Flash-Next-GGUF` at `38bb39ee` (both pin it), downloaded by 0.2.0, shared the same way.
- Each image prepared its own pack and MTP layer in its own volume. Vision off on both (`VISION=off` / `VISION=no`; 0.2.0's
  default on a 12 GB card is the CPU picture reader, turned off here to match 0.1.43). Context 262,144 on both. No other
  setting given unless the row says so. The run configs both versions wrote are in `config/` (key removed); they are the
  same apart from paths: `--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --kv int8 --kv-resident 32768`,
  plus `--resident-budget-gib 55` from 0.1.43's setup for UD-IQ4_XS and `--ple-io ram` from 0.2.0's for UD-IQ4_XS.
- Settings given to 0.2.0 by `STRATA_SET` do not appear in its run config; the engine log of each such arm shows them in
  the command line (`--pool-workers 16`, `--resident-budget-gib 55`) or shows them taking effect (no sm_86 switch line
  with `STRATA_SM86_PERF=0`).
- CPU workers: both versions' defaults; 0.2.0 logged "30 workers by default (32 physical cores, 32 logical processors;
  bounded by physical cores)". In this guest the 32 vCPUs are presented as 32 physical cores, so 0.2.0's new rule gives
  about the same count as before.
- Requests: `reasoning_effort` none, `temperature` 0, `seed` 1, streamed.

```text
# 0.2.0 (per arm a fresh container and its own volume)
docker run -d --name strata-bench --gpus device=<GPU UUID> -p 127.0.0.1:8090:8080 --ulimit memlock=-1:-1 \
  --shm-size 2g --stop-timeout 120 -v <volume>:/data -v <gguf dir>:/models:ro -e MODEL_PATH=/models \
  -e FAMILY=qwen -e MODEL=IQ3_S -e CONTEXT=262144 -e VISION=off -e API_KEY=<key> [-e STRATA_SET=...] strata:0.2.0-beta-cuda13
# 0.1.43
docker run -d --name strata-bench --gpus device=<GPU UUID> -p 127.0.0.1:8090:8080 --ulimit memlock=-1 \
  --shm-size 2g --stop-timeout 120 -v <volume>:/data -v <gguf dir>:/gguf:ro -e GGUF_DIR=/gguf \
  -e FAMILY=qwen -e MODEL=IQ3_S -e CONTEXT=262144 -e VISION=no -e HOST=0.0.0.0 -e API_KEY=<key> strata:v0.1.43
```

The scripts that ran everything are in this folder: `lib.sh` (the two commands above, readiness wait, sweep),
`run.sh` (the main sequence), `extra*.sh` (the arms added during the day, in the order they ran).

## Method

- `sweep.py` (unchanged from the 2026-10-11 report) cuts each prompt from the v0.2.0-beta checkout's `docs/*.md` and
  `README*.md`, sizes it with `/v1/messages/count_tokens`, and asks for a 200-word summary with a 256-token cap; "short"
  is a one-sentence question. Every prompt starts with a fresh nonce: reused tokens are 0 in every sweep row.
- One sweep is short, 4k, 32k and 128k, three runs each, in one container after it answered a first request. Load time is
  not in any figure. Prompt and decode tok/s are the engine's own (`timings`); time to first token is the client's.
- Order for IQ3_S: 0.2.0, 0.1.43, 0.2.0 with `STRATA_SM86_PERF=0`, then the same three again; then needles, then
  `TUNE=quick`. Swift 1.5: 0.2.0, 0.1.43, twice. UD-IQ4_XS: 0.2.0 stock (killed, below), 0.1.43, twice; later 0.2.0 with
  the RAM budget, 0.1.43, 0.2.0 with the budget. The other rows ran once, after the pairs, in the order of the `extra*.sh`
  files. All runs are in the `sweep-*.jsonl` files.
- "Decode, median of each sweep" shows the per-sweep medians, so the two (or three) interleaved sweeps can be compared.

## Results

### IQ3_S, context 262,144: 0.1.43 and 0.2.0, interleaved

| Configuration | Prompt tokens | Generated tokens | Runs | Prompt tok/s median (range) | Decode tok/s median (range) | Decode, median of each sweep | TTFT s median (range) |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| 0.1.43, short | 62 | 88 | 6 | 113 (86-117) | 64.7 (50.8-70.2) | 63.5, 65.9 | 0.57 (0.56-0.74) |
| 0.1.43, 4k | 4,135 | 256 | 6 | 1,795 (1,626-1,826) | 76.4 (66.6-77.8) | 76.8, 70.8 | 2.33 (2.29-2.56) |
| 0.1.43, 32k | 33,000 | 256 | 6 | 2,276 (2,239-2,295) | 77.1 (72.2-81.7) | 79.6, 76.7 | 14.54 (14.43-14.79) |
| 0.1.43, 128k | 130,257 | 252 | 6 | 2,176 (2,162-2,184) | 68.4 (65.3-70.9) | 67.0, 69.9 | 59.95 (59.75-60.33) |
| 0.2.0, short | 62 | 90 | 6 | 121 (89-124) | 89.6 (58.7-91.5) | 90.9, 88.2 | 0.54 (0.50-0.72) |
| 0.2.0, 4k | 4,133 | 256 | 6 | 1,821 (1,613-1,836) | 97.0 (88.3-98.1) | 96.8, 97.2 | 2.29 (2.27-2.62) |
| 0.2.0, 32k | 32,998 | 256 | 6 | 2,349 (2,315-2,357) | 92.2 (90.9-99.8) | 91.8, 92.6 | 14.09 (14.04-14.30) |
| 0.2.0, 128k | 130,258 | 250 | 6 | 2,236 (2,231-2,249) | 81.8 (79.0-84.4) | 80.3, 82.9 | 58.36 (58.02-58.49) |

Decode is 27% faster at 4k, 20% at 32k and 20% at 128k; every 0.2.0 sweep beat every 0.1.43 sweep at every length.
Prompt speed is within 4%. The short answer stops by itself at about 88 tokens, so its decode includes the slow first
windows.

### IQ3_S, other 0.2.0 settings and 0.1.42

| Configuration | Prompt tokens | Generated tokens | Runs | Prompt tok/s median (range) | Decode tok/s median (range) | Decode, median of each sweep | TTFT s median (range) |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| 0.2.0, STRATA_SM86_PERF=0, short | 62 | 86 | 6 | 105 (84-114) | 73.9 (51.7-82.0) | 72.6, 75.3 | 0.61 (0.57-0.75) |
| 0.2.0, STRATA_SM86_PERF=0, 4k | 4,134 | 256 | 6 | 1,817 (1,613-1,838) | 83.5 (79.2-89.3) | 83.2, 83.7 | 2.30 (2.27-2.61) |
| 0.2.0, STRATA_SM86_PERF=0, 32k | 32,998 | 256 | 6 | 2,334 (2,313-2,364) | 79.6 (76.0-91.1) | 80.4, 78.7 | 14.19 (14.00-14.31) |
| 0.2.0, STRATA_SM86_PERF=0, 128k | 130,257 | 246 | 6 | 2,240 (2,228-2,252) | 73.8 (72.0-74.7) | 73.8, 73.7 | 58.25 (57.94-58.57) |
| 0.2.0, TUNE=quick, short | 64 | 87 | 3 | 121 (88-128) | 85.8 (60.3-89.3) | 85.8 | 0.56 (0.49-0.75) |
| 0.2.0, TUNE=quick, 4k | 4,135 | 256 | 3 | 1,803 (1,609-1,833) | 96.3 (90.2-97.2) | 96.3 | 2.32 (2.28-2.62) |
| 0.2.0, TUNE=quick, 32k | 32,998 | 256 | 3 | 2,335 (2,306-2,348) | 92.1 (88.0-95.1) | 92.1 | 14.18 (14.10-14.35) |
| 0.2.0, TUNE=quick, 128k | 130,256 | 253 | 3 | 2,235 (2,230-2,242) | 85.6 (83.7-85.8) | 85.6 | 58.38 (58.19-58.50) |
| 0.2.0, 16 workers + conversation cache, short | 62 | 87 | 3 | 99 (80-100) | 86.0 (61.2-88.6) | 86.0 | 0.66 (0.60-0.80) |
| 0.2.0, 16 workers + conversation cache, 4k | 4,133 | 256 | 3 | 1,730 (1,554-1,737) | 90.7 (85.6-95.6) | 90.7 | 2.41 (2.40-2.71) |
| 0.2.0, 16 workers + conversation cache, 32k | 33,001 | 256 | 3 | 2,309 (2,300-2,309) | 86.5 (85.6-88.3) | 86.5 | 14.33 (14.33-14.39) |
| 0.2.0, 16 workers + conversation cache, 128k | 130,259 | 228 | 3 | 2,189 (2,186-2,234) | 80.1 (78.6-84.7) | 80.1 | 59.59 (58.40-59.69) |
| 0.1.42, short | 62 | 88 | 3 | 118 (86-122) | 76.1 (55.1-78.6) | 76.1 | 0.55 (0.53-0.76) |
| 0.1.42, 4k | 4,137 | 256 | 3 | 1,804 (1,616-1,828) | 86.2 (83.9-87.0) | 86.2 | 2.32 (2.29-2.58) |
| 0.1.42, 32k | 32,998 | 256 | 3 | 2,290 (2,257-2,293) | 87.0 (68.3-89.5) | 87.0 | 14.46 (14.45-14.66) |
| 0.1.42, 128k | 130,258 | 256 | 3 | 2,173 (2,162-2,187) | 78.1 (78.0-80.2) | 78.1 | 60.05 (59.66-60.34) |

- **The sm_86 switch table** (interleaved with the two rows above): with `STRATA_SM86_PERF=0`, 0.2.0 decodes 14% slower at
  4k, 14% at 32k and 10% at 128k than with the table on. That is two thirds of 0.2.0's lead over 0.1.43 at 4k and four
  fifths at 32k. The release notes measured +10.2% on an RTX 3060, +13.0% on an RTX A4000. Tokens were not compared between the arms.
- **`TUNE=quick`** reported "Tuning result: already fastest ... Nothing needed changing", and its sweep matches the default.
- **16 workers and the conversation cache** (`STRATA_SET="pool-workers=16 conversation-cache-mib=8192
  conversation-cache-slots=4"`, the settings of this machine's own endpoint): 6% slower decode at 4k and 32k than the
  default 30 workers, 2% at 128k. One sweep, run after the pairs; the two settings were not separated.
- **0.1.42**, one sweep after all the others: faster than today's 0.1.43 at every length (4k 86.2 against 70.8 to 76.8;
  128k 78.1 against 67.0 to 69.9), and within 8% of its own figures of 2026-10-11 (4k 93.9, 32k 81.7, 128k 75.2): lower at 4k, higher at 32k and
  128k. This points
  to a decode change between 0.1.42 and 0.1.43 on this card rather than a change of the machine, but it is one sweep,
  not interleaved; it was not looked into further.

### Swift 1.5 IQ3_S, context 262,144: 0.1.43 and 0.2.0, interleaved

| Configuration | Prompt tokens | Generated tokens | Runs | Prompt tok/s median (range) | Decode tok/s median (range) | Decode, median of each sweep | TTFT s median (range) |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| 0.1.43, short | 62 | 88 | 6 | 114 (81-122) | 71.4 (52.4-73.7) | 70.2, 72.6 | 0.56 (0.52-0.80) |
| 0.1.43, 4k | 4,136 | 256 | 6 | 1,798 (1,602-1,821) | 77.2 (71.5-86.3) | 80.7, 74.2 | 2.33 (2.29-2.61) |
| 0.1.43, 32k | 33,000 | 256 | 6 | 2,269 (2,237-2,295) | 77.4 (69.4-80.2) | 79.5, 75.9 | 14.59 (14.43-14.80) |
| 0.1.43, 128k | 130,257 | 254 | 6 | 2,170 (2,168-2,185) | 63.2 (60.2-68.3) | 61.7, 63.4 | 60.12 (59.70-60.20) |
| 0.2.0, short | 60 | 85 | 6 | 112 (91-126) | 91.0 (55.7-96.7) | 89.1, 92.9 | 0.58 (0.49-0.70) |
| 0.2.0, 4k | 4,132 | 256 | 6 | 1,827 (1,625-1,844) | 96.0 (88.2-99.3) | 96.6, 95.1 | 2.29 (2.27-2.57) |
| 0.2.0, 32k | 32,998 | 256 | 6 | 2,337 (2,306-2,362) | 97.1 (92.2-99.8) | 96.4, 98.1 | 14.16 (14.02-14.35) |
| 0.2.0, 128k | 130,257 | 238 | 6 | 2,239 (2,229-2,254) | 85.6 (83.5-87.8) | 85.1, 86.3 | 58.27 (57.88-58.54) |

Decode 24% faster at 4k, 25% at 32k and 35% at 128k, prompt speed within 4%. Same files for both versions (0.2.0's pin).

### UD-IQ4_XS, context 262,144

**0.2.0 with its own settings did not start.** The kernel killed the engine for lack of memory while it loaded the
experts into RAM, in 4 of 4 starts (two of them during `TUNE=quick`, which then reported "Tuning result: failed"); the
guest's used memory reached 91 of 94 GiB. `engine-logs/kernel-oom-ud-iq4_xs-0.2.0.txt` has the kernel's lines (anon-rss 50 to
60 GB, file-rss 24 to 28 GB at the kill), `engine-logs/engine-0.2.0-ud.txt` the engine's. The likely reason, from reading
`setup.py` at v0.2.0-beta (not tested by changing it): `unsloth_budget()` now writes the RAM-budget mode only when the RAM
is less than the experts plus 16 GiB (`UNSLOTH_RESIDENT_BESIDE_GB`). For UD-IQ4_XS that is about 55.4 + 16 = 71 GiB, so
on 94 GiB every expert is kept in RAM, beside the PLE table that 0.2.0's setup locks in RAM (`--ple-io ram`) and 3.09 GiB
of KV streamed to RAM at this context. 0.1.43's setup wrote `--resident-budget-gib 55` for the same file, and the rows
below give 0.2.0 the same budget by hand.

| Configuration | Prompt tokens | Generated tokens | Runs | Prompt tok/s median (range) | Decode tok/s median (range) | Decode, median of each sweep | TTFT s median (range) |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| 0.1.43, short | 62 | 85 | 9 | 71 (56-87) | 31.9 (29.2-46.6) | 31.6, 31.9, 39.5 | 0.92 (0.75-1.11) |
| 0.1.43, 4k | 4,134 | 256 | 9 | 1,019 (962-1,138) | 48.2 (34.7-51.5) | 35.1, 48.2, 49.9 | 4.09 (3.68-4.33) |
| 0.1.43, 32k | 32,998 | 256 | 9 | 1,722 (1,635-1,733) | 42.2 (39.1-44.8) | 40.3, 42.2, 44.2 | 19.23 (19.10-20.23) |
| 0.1.43, 128k | 130,256 | 254 | 9 | 1,685 (1,681-1,691) | 38.7 (35.8-43.5) | 37.7, 38.7, 39.4 | 77.41 (77.12-77.59) |
| 0.2.0, resident-budget-gib=55, short | 61 | 98 | 6 | 63 (46-74) | 35.2 (27.7-44.3) | 33.2, 42.6 | 1.00 (0.85-1.35) |
| 0.2.0, resident-budget-gib=55, 4k | 4,134 | 256 | 6 | 549 (525-560) | 38.0 (37.7-40.8) | 38.3, 37.7 | 7.60 (7.45-7.96) |
| 0.2.0, resident-budget-gib=55, 32k | 32,996 | 256 | 6 | 1,194 (1,187-1,202) | 39.4 (34.8-40.9) | 40.1, 38.1 | 27.69 (27.53-27.86) |
| 0.2.0, resident-budget-gib=55, 128k | 130,256 | 255 | 6 | 1,462 (1,458-1,465) | 34.6 (32.8-36.3) | 34.9, 34.4 | 89.25 (89.01-89.46) |
| 0.2.0, budget 55, TUNE=quick, short | 62 | 96 | 3 | 66 (47-70) | 42.2 (29.3-42.5) | 42.2 | 0.97 (0.92-1.27) |
| 0.2.0, budget 55, TUNE=quick, 4k | 4,133 | 256 | 3 | 545 (530-556) | 38.7 (37.8-40.4) | 38.7 | 7.64 (7.49-7.89) |
| 0.2.0, budget 55, TUNE=quick, 32k | 32,999 | 256 | 3 | 1,201 (1,161-1,202) | 38.6 (36.9-39.6) | 38.6 | 27.55 (27.52-28.50) |
| 0.2.0, budget 55, TUNE=quick, 128k | 130,258 | 253 | 3 | 1,467 (1,456-1,470) | 35.4 (34.8-35.5) | 35.4 | 88.93 (88.72-89.60) |
| 0.2.0, budget 55, STRATA_PREFILL_HOST_PLAN=0, short | 59 | 93 | 3 | 58 (44-69) | 36.9 (27.0-40.0) | 36.9 | 1.05 (0.88-1.44) |
| 0.2.0, budget 55, STRATA_PREFILL_HOST_PLAN=0, 4k | 4,132 | 256 | 3 | 552 (545-556) | 39.7 (37.6-41.2) | 39.7 | 7.54 (7.49-7.69) |
| 0.2.0, budget 55, STRATA_PREFILL_HOST_PLAN=0, 32k | 32,996 | 256 | 3 | 1,201 (1,198-1,203) | 39.7 (36.9-40.5) | 39.7 | 27.54 (27.50-27.60) |
| 0.2.0, budget 55, STRATA_PREFILL_HOST_PLAN=0, 128k | 130,256 | 256 | 3 | 1,467 (1,456-1,469) | 34.9 (34.0-36.1) | 34.9 | 88.94 (88.79-89.55) |

- With the budget, 0.2.0 reads prompts 46% slower than 0.1.43 at 4k, 31% at 32k and 13% at 128k, and decodes 21% slower
  at 4k, about the same at 32k (-7%) and 11% slower at 128k. The first 0.1.43 sweep's 4k decode (35.1) is low; without it
  the 0.1.43 4k median is 49.9.
- The engine log of the budget runs has "prefill plan: chunk 7424, ring 96 kept: file set 8.0 GiB does not fit the page
  cache 3.2 GiB" and "a prompt chunk of 8192 tokens lends 1543 cache slots to the prompt path, but only 0 of their ...".
  Turning the new file-tier chunk plan off (`STRATA_PREFILL_HOST_PLAN=0`) kept the same 7,424-token chunk and changed
  nothing, so that plan is not the reason. The reason was not found. The two versions' run configs differ by
  `--ple-io ram`, which 0.2.0's setup adds for this file and 0.1.43's does not; it was not tested on its own.
- `TUNE=quick` with the budget reported "already fastest", and its sweep matches the untuned one.

## Correctness

- `tools/needle_bench.py --lengths 32k,128k,262k --depths 10,50,90` on 0.2.0 IQ3_S (default settings): 9 of 9 found
  (`needles-iq3_s-0.2.0.json`; 32,045, 123,966 and 253,375 prompt tokens). Two of the requests reused part of the
  previous needle prompt (64,512 tokens at 128k depth 90, 21,504 at each 262k depth; engine log), so those prompt times
  are not full reads. Not run on 0.1.43 or on the other models.
- Generated tokens were not compared between versions or arms; no quality check beyond the needles.

## Limitations

- No bare-metal run of the same machine, so nothing here measures what the virtual machine or Docker cost.
- Three runs per cell; two sweeps per arm for the interleaved rows (three for 0.1.43 UD-IQ4_XS), one for the others.
- The 0.2.0 image comes from the released engine package; the 0.1.43 and 0.1.42 images are source builds for sm_86 only.
- The 0.1.43 UD-IQ4_XS sweeps of the main sequence ran next to failed 0.2.0 starts (killed in under two minutes each),
  not next to working 0.2.0 sweeps; the third 0.1.43 sweep is the one interleaved with the budget runs.
- Neither run config has a `parallel` entry, so both versions served one request at a time; nothing here measures
  concurrent requests.
- The 0.2.0 engine logs warn that only 326 to 458 MiB of VRAM was free at the start of some requests ("the driver can
  page GPU memory to system RAM"); no request in the sweeps fell to the few tok/s that warning describes.
- Not measured: vision; a longer soak; agent sessions; any coding or answer-quality score; 0.2.0's own default worker count against
  fewer workers apart from the one 16-worker sweep.

## Files

- `sweep.py`: the sweep script (fields at its top). `sweep-<model>-<arm>.jsonl`: one line per run and one summary line per
  sweep. Arms: `old` 0.1.43, `new` 0.2.0, `0142` 0.1.42, `sm86off`, `tuned`, `prodcand` (16 workers + conversation
  cache), `budget` (`resident-budget-gib=55`), `budget-tuned`, `budget-nohostplan`.
- `needles-iq3_s-0.2.0.json`: output of `tools/needle_bench.py`.
- `config/`: the run configs each image's setup wrote, key removed (`0.2.0-*` from `/data/install`, `0.1.4x-*` from
  `/data/config`).
- `engine-logs/engine-0.2.0-*.txt`: the 0.2.0 engine logs (`/data/state/strata/launch/docker.engine.log`) of the arms, including
  the killed UD-IQ4_XS starts (`ud`). `engine-logs/kernel-oom-ud-iq4_xs-0.2.0.txt`: the kernel's out-of-memory lines. The 0.1.43
  engine logs were inside the containers and were removed with them.
- `lib.sh`, `run.sh`, `extra.sh` to `extra4.sh`: the scripts as run (the GPU UUID replaced by a placeholder).
