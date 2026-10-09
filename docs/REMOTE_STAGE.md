# Strata across two or more PCs (remote stage)

> **Experimental and opt-in.** The Qwen3.8-Flash-Next family only. Measured on three PCs (Linux and Windows) with two
> models (RVN IQ3_S, Huihui UD-Q4_K_XL). Nothing changes unless the engine is started with `--remote-stage` or
> `--stage-worker`. How it works inside, every measurement and what is still open:
> [remote-stage/ENGINEERING.md](remote-stage/ENGINEERING.md). Raw numbers:
> [bench/results/2026-10-10-remote-stage](../bench/results/2026-10-10-remote-stage/README.md).

A layer split ([MULTI_GPU.md](MULTI_GPU.md)) runs layers 0 to K-1 on one card and K onward on the next, with one
hand-off per verify window. The remote stage puts the later layers in a **worker process on another PC** and does
that hand-off over TCP:

- the **main PC** runs layers 0 to K-1, the output head, the draft (MTP) layer, sampling and the API server;
- a **worker PC** runs layers K to 47 without the head (`strata --serve --stage-worker PORT --stage-begin K`), or a
  middle range that it hands on to the next worker (a relay);
- each PC keeps an expert cache for **its own layers only**, its own CPU expert pool, its own part of the session
  (the K/V of its attention layers, the state of its GDN layers), and loads only its own layers' experts into RAM.

What it buys: more VRAM for the expert cache and more RAM for the experts. What it costs: one network round trip per
verify window per worker, and every prompt chunk's rows over the network. Decode does not run on two PCs at once: a
window goes through the stages one after the other.

## When it helps (measured)

Three PCs, 2026-10-09 and 2026-10-10. Every PC ran a build of this feature (CUDA 13.0).

| | PC A (main) | PC B | PC C |
|---|---|---|---|
| GPU | RTX 3080 10 GB, PCIe 4.0 slot at **x8** (13.0 GB/s probe) | RTX 2080 Ti 11 GB, PCIe 3.0 x16 (13.2 GB/s) | RTX 3070 8 GB, PCIe 4.0 x16 (26.3 GB/s); it also drives the desktop, so its worker was kept to ~6.5 GB of VRAM |
| CPU, RAM | Ryzen 9 5900XT 16 cores, 128 GB DDR4-3200 (96 GB for the container) | Ryzen 9 5900X 12 cores, 64 GB DDR4-2400 | Ryzen 7 5700X3D 8 cores, 64 GB DDR4-3600 |
| OS | Linux (Ubuntu 24.04, LXC container) | Linux host, kernel 6.14 | Windows 11 (native build: MSVC 2022, driver 591.86); also WSL2 |

Links: A-B 1 GbE (117 MB/s measured) or 10 GbE (390-400 MB/s measured: PC B's 10 GbE card sits in a slot that
negotiates PCIe 2.5 GT/s x2); B-C and A-C 1 GbE (114 MB/s). PC A's host also ran CI jobs during some runs.

Settings: `--kv int8 --kv-resident 32768 --max-context 131072 --spec 4` with the draft layer, no conversation cache,
greedy, one run per row. Prompts of 7,521 / 30,167 / 57,383 / 90,921 tokens with a 320-token answer (columns 8K, 32K,
64K, 100K: prefill / decode tok/s); "short" is the mean decode of nine 200-256-token answers to short prompts.

### RVN IQ3_S, two PCs (A + B over 1 GbE)

K is the number of layers on the 3080; the chunk is `--prefill` on both PCs.

| configuration | short | 8K | 32K | 64K | 100K |
|---|---:|---:|---:|---:|---:|
| 3080 alone, `--prefill auto` (5888) | 39.9 | 791 / 38.1 | 1,118 / 37.2 | 1,146 / 39.0 | 1,120 / 38.0 |
| K=20, chunk 2048 | 51.7 | 491 / 53.2 | 604 / 50.0 | 609 / 52.1 | 600 / 52.0 |
| K=24, chunk 2048 | 51.3 | 593 / 54.9 | 771 / 50.7 | 764 / 48.1 | 750 / 47.7 |
| K=26, chunk 2048 | 50.7 | 629 / 51.8 | 816 / 48.7 | 865 / 47.9 | 846 / 47.7 |
| K=28, chunk 2048 | 50.2 | 607 / 48.0 | 761 / 46.0 | 798 / 46.1 | 812 / 45.3 |
| K=26, chunk 4096 | 52.3 | 740 / 50.1 | 1,211 / 47.8 | 1,330 / 49.2 | 1,328 / 45.6 |
| K=24, chunk 5888 | 54.0 | 652 / 51.1 | 1,226 / 50.2 | 1,381 / 50.6 | 1,374 / 47.8 |
| **K=26, chunk 5888** | 53.5 | 655 / 48.1 | 1,231 / 49.7 | 1,474 / 47.7 | 1,521 / 45.6 |
| K=28, chunk 5888 | 49.4 | 690 / 47.7 | 1,217 / 45.9 | 1,503 / 45.9 | 1,591 / 45.7 |
| K=26, chunk 8192, `--vram-reserve-mib 1200` | 52.5 | 589 / 46.0 | 1,225 / 47.5 | 1,485 / 47.2 | 1,633 / 45.4 |
| K=26, chunk 8192, no reserve | - | out of VRAM at the first long prompt | | | |

- **Decode is 18-44% faster than the 3080 alone** in every row (100K: 45.3-52.0 vs 38.0; short prompts 49.4-54.0 vs
  39.9): the two caches hold more of the experts (the 3080's decode hit rate, from the engines' logs at chunk 2048 during the sweep: 77-87% with the split, 59-60% alone; those logs are not kept with the results).
- **Long prompts need a large chunk.** At 2048 every split read prompts slower than the 3080 alone (100K: 600-846 vs
  1,120); at 5888-8192 the split is faster from 32K on (1,374-1,633).
- **Short prompts lose a little** (8K: 589-740 vs 791); the faster decode about makes up for it over a request.

### RVN IQ3_S, three PCs

| configuration | short | 8K | 32K | 64K | 100K |
|---|---:|---:|---:|---:|---:|
| A + C (C under WSL2, clocks not locked), 1 GbE, K=40 | 36.2 | 685 / 33.7 | 1,018 / 31.8 | 1,138 / 32.4 | 1,189 / 33.0 |
| A + B + C, layers 24 / 16 / 8, all 1 GbE, C under WSL2 | 40.9 | 497 / 41.3 | 706 / 37.3 | 974 / 36.8 | 1,248 / 36.0 |
| the same, A-B on 10 GbE | 42.1 | 635 / 39.9 | 1,035 / 35.3 | 1,418 / 34.8 | 1,576 / 34.1 |
| the same, C native Windows with its clocks locked | 51.3 | 658 / 47.8 | 1,110 / 46.5 | 1,451 / 44.1 | **1,640** / 44.7 |

- With the 3070's clocks left to the Windows driver, adding it made things **slower** (decode 34-41 against 45-54 for
  A + B). The cause was its clocks, not WSL2 or the network: see [Windows: lock the worker GPU's clocks](#windows-lock-the-worker-gpus-clocks).
- With them locked, three PCs read 100K prompts the fastest of all (1,640 tok/s) and decode about as fast as two PCs
  (44.7 vs 45.6 at 100K, 51.3 vs 53.5 short). A third stage adds a hop to every window; on IQ3_S the 3080 and the
  2080 Ti already cache most of what the 3070 would add.
- On 1 GbE everywhere the middle PC carried every prompt chunk twice (in and out, each way): its link ran at line rate
  for one long prompt: 2.6 GB each way in the ~38 s it crossed, at up to 123 MB/s (sampled every second: [data/nic-pc-b-1gbe-3pc.txt](../bench/results/2026-10-10-remote-stage/data/nic-pc-b-1gbe-3pc.txt)). 10 GbE between A and B removed most of that.

### Huihui UD-Q4_K_XL (a larger model: 71.7 GiB of experts, 1.47-1.90 GiB a layer)

| configuration | short | 8K | 32K | 64K | 100K | 100K request |
|---|---:|---:|---:|---:|---:|---:|
| 3080 alone, `--resident-budget-gib 70` (auto chunk 2048, 620 cache slots) | 23.9 | 302 / 22.2 | 324 / 21.8 | 328 / 20.9 | 326 / 21.8 | 293 s |
| 3080 alone, all experts pinned (auto chunk 1280, 571 slots) | 27.3 | 206 / 25.4 | 212 / 24.5 | 209 / 25.3 | 209 / 23.7 | 449 s |
| A + B (10 GbE), K=26, chunk 5888 | **38.8** | **659** / 34.4 | **1,147** / 36.9 | 1,192 / 37.5 | 1,164 / 33.6 | 88 s |
| A + B + C, layers 24 / 16 / 8, chunk 5888, C locked | 38.4 | 549 / 35.3 | 945 / 36.6 | **1,210** / 33.2 | **1,334** / 35.2 | **77 s** |
| the tuner's pick: 24 / 16 / 8 again, C with `--expert-cache 1000` (1,248 slots) | 36.8 | 552 / 33.1 | 975 / 35.2 | - | 1,383 / 33.6 | - |

"100K request" is the measured wall time of the 90,921-token prompt with its 320-token answer. The tuner's row is
from its own prompts (built from the same corpus; its 8K / 32K / 100K are nominal lengths), one run each.

Over a workload of 30% 8K, 50% 32K and 20% 100K prompts with 500-token answers (prompt / prefill + 500 / decode from
the rows above; computed, not timed): the 3080 alone ~133 s a request (with the budget; ~190 s all pinned), A + B 46.3 s, A + B + C 47.6 s (the 24 / 16 / 8
row) and 49.1 s (the tuner's own profile run).

- **One 10 GB card is too small for this model**: the dense weights leave room for 571-620 cache slots, and the auto
  prompt chunk falls to 1280-2048 tokens, so every prompt streams all 71.7 GiB of experts over PCIe many times.
- **Two PCs make it usable**: decode +42% (short 38.8 vs 27.3), prompts 2.2-3.6x faster than the better single-card
  run, a 100K prompt in 88 s instead of 293.
- **The third PC pays for long prompts only** (100K: 1,334 vs 1,164 tok/s, 77 vs 88 s); at 8K and 32K it is 17-18%
  slower, decode is the same. With 24 / 16 / 8 layers the 3080 was the slow stage of a prompt: the tuner's timing
  lines at a ~30K prompt gave 3,333 ms a chunk for the 3080's 24 layers, 2,611 for the 2080 Ti's 16 and 1,246 for
  the 3070's 8.
- **Moving layers did not help.** The tuner tried eight splits, from 18 / 16 / 14 (stages of 2.4 / 2.4 / 2.1 s a
  chunk) to 26 / 14 / 8: every one took 45.7-48.0 s for a 32K request, within the run-to-run spread, and it kept
  24 / 16 / 8. A ~30K prompt is about five chunks, and each chunk crosses three stages and two links (241 MB each way
  per chunk on the 1 GbE hop to C) before its reply, so the pipeline's fill time, not its slowest stage, sets the
  speed; decode is the sum of the stages, ~40 ms of it the 3080's.

## Windows: lock the worker GPU's clocks

A worker's GPU is idle between its decode windows (the other stages run meanwhile: ~25-45 ms each window). The
Windows driver (WDDM; WSL2 runs through it too) lowers the clocks in that time; the Linux driver did not. Sampled every
100 ms through the same decode benchmark (RVN IQ3_S, layers 40-47 on the worker):

| worker GPU | P2 (full clocks) | memory clock | GPU wait per window (8 layers) | short decode |
|---|---:|---|---:|---:|
| RTX 3070, Windows, driver's choice | 6% of samples (P3 34%, P5 51%, P8 8%) | 810 MHz in 51% of samples, 5,001 in 35%, 6,801 in 6% | 11.9-22.2 ms | 33.8 |
| RTX 3070, Windows, `nvidia-smi -lgc 1500,1905 -lmc 7001,7001` | 100% | 6,801 MHz | 3.7-4.1 ms | 41.8 |
| RTX 2080 Ti, Linux | 100% | 6,800 MHz | 4.6-5.9 ms | 41.6-41.8 |

Prompt chunks (long, steady work) were barely affected: the 3070 on Windows read its 8 layers of a 5888-token chunk in
855-877 ms unlocked and 709-711 ms locked, the 2080 Ti in 1,044-1,143 ms. Locked, the 3070 drew 57 W on average
(151 W peak, 59 °C) during the run.

Locking needs an administrator. Either run `nvidia-smi -lgc MIN,MAX -lmc MIN,MAX` yourself before starting the worker
and `nvidia-smi -rgc -rmc` afterwards (a reboot resets them too), or give the node agent `"gpu_clocks"` and run it as
administrator: it locks them while a worker runs and resets them when none does. Under WSL2, lock them from Windows.
Pick values the card supports (`nvidia-smi -q -d SUPPORTED_CLOCKS`). This may also matter for cards in one Windows PC
with an in-process layer split; not measured.

## What you need

- **Linux or Windows on every PC.** Measured: Linux (a container and a host), Windows 11 native, and Windows 11
  through WSL2. A Windows build needs MSVC 2022 and the CUDA 13 toolkit; when installing only some of the toolkit's
  parts, include `crt` and `nvvm` (CUDA 13 ships them separately, and nvcc fails without them). Put `cudart64_13.dll`,
  `cublas64_13.dll` and `cublasLt64_13.dll` (in CUDA 13's `bin\x64`) next to `strata.exe`.
- **An NVIDIA card the engine supports on every PC** (RTX 20 or newer); each PC's build covers its own card.
- **The same model on every PC:** the same pack directory (both sides compare a fingerprint of its `index.txt` and
  `native_experts.txt` and refuse a mismatch), the GGUF shards (a worker reads every shard's header and its own
  layers' tensors; `tools/stage_ship.py` sends just that), and the expert profile. The fingerprint covers the pack's
  index files, not the weights.
- **A native pack that takes its experts from the GGUF** (no `experts.bin`, no `--shared-expert-arena`): only that
  loader can hold a range of layers. Anything else stops at start with "a layer range needs a native pack read from
  its GGUF".
- **RAM for its own layers' experts** on each PC, pinned: 1.11 GiB a layer for RVN IQ3_S, 1.47-1.90 GiB for
  UD-Q4_K_XL. On Windows, Task Manager shows this as "shared GPU memory" (below).
- **A network the PCs share**, and the ports open: the worker's (7841 here) and, with the tools, the agent's (7840).
  - Windows: an inbound firewall rule for those TCP ports, limited to the local subnet.
  - WSL2: mirrored networking (`networkingMode=mirrored` in `.wslconfig`), and a Hyper-V firewall rule for the ports
    (`New-NetFirewallHyperVRule` with WSL's VM creator id). The WSL VM keeps the page cache of every file it read or
    wrote and does not give it back to Windows; cap its memory in `.wslconfig` and let the node agent drop the
    cache (its default).

## Two PCs

**1. A shared secret.** Put the same string (up to 63 characters) in `STRATA_STAGE_TOKEN` on both PCs. A worker
started without one prints a warning and takes orders from any host that reaches its port.

**2. Start the worker first** (the worker PC, here 192.0.2.11, running layers 26-47):

```
export STRATA_STAGE_TOKEN="$(cat /opt/strata/stage-token)"
export LD_LIBRARY_PATH=/opt/strata/bin       # where the engine's CUDA libraries are, if not installed system-wide
/opt/strata/bin/strata --serve --stage-worker 7841 --stage-begin 26 --stage-bind 192.0.2.11 \
  --pack /data/packs/<pack> \
  --native /data/models/<model>/<model>-00001-of-0000N.gguf \
  --ple-gguf /data/models/<model>/<model>-00002-of-0000N.gguf \
  --expert-profile /opt/strata/data/expert-profile.bin --expert-cache auto --prefill 5888 \
  --spec 4 --max-context 131072 --kv int8 --kv-resident 32768
```

On Windows the same arguments go to `strata.exe` (`set STRATA_STAGE_TOKEN=...` first). It is ready when the log says
`strata stage worker: listening on 192.0.2.11:7841 (a token is required)`. It runs without stdin and serves one main
process at a time; when the main process goes away it waits for the next one. The worker does not read the PLE table
(layer 1 always runs on the main PC), but the argument checks still want `--ple-gguf` or a shard that holds it.

**3. Then the main PC.** Add these to the model's config (`strata-*.json`, `"args"`), and the token to its `"env"`:

```json
"args": [ "...the model's usual args...",
          "--prefill", "5888",
          "--layer-split", "26", "--split-device", "0", "--remote-stage", "192.0.2.11:7841" ],
"env": { "STRATA_STAGE_TOKEN": "the same secret" }
```

Then start the server as usual. `--layer-split K --split-device 0` keeps the head on this card and sends layers K
onward to the worker. Keep the split in `"args"` (the config's `"layer_split"` key is for cards in one PC), and drop
`--conversation-cache-mib`: the remote stage turns the conversation cache off anyway. The main process connects once,
at start, and stops if the worker is not listening. The log then says:

```
strata generate: remote stage: this process holds layers 0-25, the rest on 192.0.2.11:7841
strata serve: layer split: layers 0-25 (CUDA0), 26-47 (192.0.2.11:7841), the head (CUDA0), one hand-off per window
strata serve: remote stage 192.0.2.11:7841: layers 26-47 there (prompt chunk 5888, context 131072)
```

**What must agree** (each side checks the other's hello and says which number differs): the model pack, `--kv`, and
K (`--layer-split K` on the main PC, `--stage-begin K` on the worker). The worker's `--prefill`, `--max-context` and
`--spec` must be at least the main process's. Everything else is each PC's own choice: `--expert-cache`,
`--kv-resident`, `--vram-reserve-mib`, the CPU threads.

## More than two PCs (relay workers)

A middle worker runs a range of layers and hands its rows to the next worker; the next worker's reply comes back
through it: main -> B -> C -> B -> main. The main process is configured exactly as for two PCs (it sees one worker).
Start the **last** worker first; a relay worker connects to the next one when it starts (it retries for a minute).

```
# PC C, layers 40-47:
strata --serve --stage-worker 7841 --stage-begin 40 ...model and context args as above...
# PC B, layers 24-39, relaying to C:
strata --serve --stage-worker 7841 --stage-begin 24 --stage-end 40 --stage-next 192.0.2.12:7841 ...
# PC A (main), layers 0-23: --layer-split 24 --split-device 0 --remote-stage 192.0.2.11:7841
```

Measured with three PCs (the tables above). A relay carries every prompt chunk and every window twice, so give it the
fastest link: on 1 GbE the relay's link was the limit.

## Options

| flag or variable | where | what it does |
|---|---|---|
| `--remote-stage HOST:PORT` | main | the (first) worker's address. Needs `--serve` and `--layer-split K --split-device 0` |
| `--layer-split K --split-device 0` | main | K is the first layer the worker runs (2 or more): this PC runs 0 to K-1 and the head |
| `--stage-worker PORT` | worker | serve a main process on this TCP port instead of stdin. Needs `--serve`, not with `--layer-split` |
| `--stage-begin K` | worker | the first layer this worker runs: the main process's K (or the previous relay's `--stage-end`) |
| `--stage-end K2`, `--stage-next HOST:PORT` | relay worker | run layers K to K2-1 and hand the rest to the worker at HOST:PORT. Both or neither |
| `--stage-bind ADDR` | worker | the address to listen on (default: every interface, IPv4 and IPv6) |
| `STRATA_STAGE_TOKEN` | every PC | the shared secret in the hello (up to 63 characters). A worker without one warns and accepts anyone |
| `STRATA_REMOTE_TIMING=1` | any | per-stage timing lines: each prompt chunk, and every 64 windows (see below) |
| `STRATA_REMOTE_TIMEOUT_S` | any | seconds a send or receive may block before the link counts as dead (default 300). After the hello, a worker waits for the next message without a limit |

Fixed: a peer that connects but sends no hello within 10 s is dropped; TCP keepalive notices a peer that vanished
without closing (power, cable) in about a minute. On Windows a worker's port is bound exclusively
(`SO_EXCLUSIVEADDRUSE`), so a second worker on the same port fails as it does on Linux.

**Timing lines** (`STRATA_REMOTE_TIMING=1`), what the tuner reads:

```
strata prefill: layers 0-26, chunk T=5888 at 5888: own 2862 ms, waited 0 ms for the previous send     (main, relay)
strata remote: chunk T=5888: 7005 ms from its send to its rows = worker 2704 + link and queue 4301 (5888 rows back)
strata remote: 64 windows: mean 23.04 ms a round trip = worker 21.87 + link 1.17 (last: ...)             (main)
strata remote: main process between windows 25.32 ms each (64 windows: head, drafter, layers 0-K)       (main)
strata stage worker: chunk T=<tokens> at <position>: read in <ms> ms                                   (worker)
strata stage worker: windows: own layers 11.08 ms, next worker 8.59 ms each (64 windows; wait for the GPU 10.42,
  pool + plan 0.41, host staging 0.00)                                                                  (worker)
```

"link and queue" of a chunk includes the time it waited behind earlier chunks (the chunks are pipelined). A worker's
own window time splits into waiting for its GPU, the CPU expert pool and the plan, and host staging: a GPU wait far
above another card's for the same layers points at clocks (Windows, above).

## Picking K and the chunk

What the measurements say (re-measure on your PCs; `tools/stage_tune.py` does it):

- **Use a large prompt chunk on every PC** (5888 or more), not the 2048 default of a plain split. Each chunk streams
  the experts it routes to over PCIe, layer by layer, so a larger chunk spreads that cost over more tokens.
- **A larger chunk takes VRAM from the main card's decode cache.** At 8192 the 3080 had 65 MiB free and ran out at the
  first long prompt; `--vram-reserve-mib 1200` fixed it. The start-up line `N MiB of VRAM free with everything loaded
  - LOW ... add --vram-reserve-mib M` gives the number.
- **Give the faster stage more layers, and re-balance after changing the chunk.** Balance by each stage's measured
  time per layer, not by VRAM: with clocks locked the 3070 was faster per layer than the 2080 Ti (decode 3.7-4.1 vs
  4.6-5.9 ms for the same 8 layers; prompt chunks 709 vs 1,044-1,143 ms). On Q4_K_XL with 24 / 16 / 8 layers the 3080
  was the slow stage of a prompt and the 3070 the fast one.
- **For packs whose layers differ in size** (the Unsloth UD packs: layers 2, 4, 30, 46 and 47 hold up to 1.3-1.5x the
  expert bytes of the others), balance by bytes; the tuner does.
- **On a desktop card, size the cache by VRAM.** `--expert-cache N` on a native pack is a budget of N times the
  model's largest expert blob, filled with this worker's (smaller) blobs: on UD-Q4_K_XL layers 40-47, N=1000 gave
  1,248 slots and N=1800 gave 1,766 (VRAM-capped, 183 MiB left).

## Tools

Three Python scripts (standard library only) in `tools/`. The agent and the shipper run on Linux and Windows; `stage_tune.py tune` runs on a Linux main PC (it starts and stops the server with `pgrep` / `kill`), `stage_tune.py apply` anywhere. Their docstrings are the
full usage; `python -m unittest tools.test_stage_node` tests the agent's file store and the shipper's paths.

**`tools/stage_node.py`: a node agent** on each worker PC. It starts and stops that PC's stage workers on request
and reports the PC (GPU, RAM, CPU) and the link speed toward other nodes. Every request carries the token in an
`X-Stage-Token` header.

```
python3 tools/stage_node.py node.json
```

```json
{ "exe": "/opt/strata/bin/strata", "lib_dirs": ["/opt/strata/bin"],
  "bind": "192.0.2.11", "agent_bind": "0.0.0.0", "agent_port": 7840, "log_dir": "/opt/strata/stage",
  "args": ["--spec", "4", "--max-context", "131072", "--kv", "int8", "--kv-resident", "32768"],
  "token_file": "/opt/strata/stage-token",
  "data_dir": "/data/stage",
  "gpu_clocks": {"graphics": [1500, 1905], "memory": [7001, 7001]} }
```

- `POST /start` (`{"begin": K, "end": K2, "next": "host:port", "port": 7841, "prefill": 5888, "cache": "auto",
  "extra": [...], "model": {"dir": ..., "native": ..., "pack": ..., "profile": ..., "ple": ...}}`) answers when the
  worker listens, or with the log's tail when it exits. `"model"` names a model shipped under `data_dir`:
  `data_dir/models/<dir>/<native>`, `data_dir/packs/<pack>`, optionally `data_dir/data/<profile>` (else node.json's
  `expert_profile`, else `data_dir/data/expert-profile.bin`) and `--ple-gguf data_dir/models/<dir>/<ple>` (a worker
  never runs the PLE layer; the engine finds the shard by name). Names of letters, digits, `.`, `_` and `-` only, not
  dots alone and not a Windows device name, so every path stays inside `data_dir`. So **one agent serves every model
  shipped to it**; without `"model"` the worker runs the model node.json names (`pack`, `native`, `ple_gguf`,
  `expert_profile`, optional with `data_dir`).
- `"gpu_clocks": {"graphics": [MIN, MAX], "memory": [MIN, MAX], "gpu": INDEX}` locks the GPU's clocks (on GPU INDEX,
  else every GPU) while a worker runs. It is meant for Windows workers, and works on any OS where the agent runs as
  administrator (Windows) or root; otherwise the `/start` reply and the tuner say "NOT locked". The clocks are reset
  when no worker runs: after a `/stop`, a worker that exits (noticed at the next `/info`), or the agent stopped with
  Ctrl+C or Ctrl+Break (SIGTERM on Linux). A killed agent or a closed console window leaves them locked until
  `nvidia-smi -rgc -rmc` or a reboot. `"agent_bind"` lets the agent listen on every
  interface, so the tuner's link test runs over the stage link when the workers use a second NIC.
- Other endpoints: `GET /ping`, `GET /info` (GPU, RAM, the clock setting), `POST /stop`, `GET /log`, `POST /sink`
  and `POST /send` (a link test), `POST /put`, `GET /ranges`, `GET /sha256` (for `stage_ship.py`). A stop waits up
  to 180 s for the worker to exit (a worker with ~30 GB of pinned experts takes more than 30 s to unpin).
- Shipped files are sparse: on Windows they are marked sparse and grown with `SetEndOfFile` (NTFS and ReFS would
  otherwise write the gaps as zeros), so 84.77 GiB of shards holding 8 layers took 11.94 GiB of disk. A drive that
  cannot hold sparse files (FAT32, exFAT) is refused with that reason.

**`tools/stage_ship.py`: send a node only its layers.** From the PC that holds the whole model it sends, to a node
agent's `data_dir`: every GGUF shard's header and the tensors of the node's layers at their own offsets, in files of
the shards' full sizes (sparse on the node, so the loader and the pack's index work unchanged); the non-layer tensors
except the PLE table (`per_layer_token_embd`, ~26.8 GB for RVN IQ3_S), which only a node running layer 1 would need;
the pack directory, the expert profile and optionally the engine's directory. What the node already has is skipped
(a source file whose header or size changed voids what was written of it), and every 64 MiB piece's sha256 is
compared on arrival.

```
python3 tools/stage_ship.py --agent http://192.0.2.11:7840 --token-file /opt/strata/stage-token \
    --model-dir /data/models/<model> --pack-dir /data/packs/<pack> \
    --profile /opt/strata/data/expert-profile.bin --layers 26-47
```

Measured over 1 GbE: RVN IQ3_S layers 36-47 to a WSL2 node, 16.64 GiB in 217 s; RVN layers 40-47 and UD-Q4_K_XL
layers 40-47 to a Windows node, 11.39 GiB in 211 s and 15.51 GiB in 184 s; UD-Q4_K_XL layers 24-39 to a Linux node,
27.63 GiB in 342 s, and layers 40-47 more later (12.85 GiB in 162 s, the rest skipped). The UD-Q4_K_XL workers and
PC C's RVN workers in the measurements above ran from these sparse files (PC B's RVN worker had a full copy).

**`tools/stage_tune.py`: the layer-split tuner**, run on the main PC. It measures for real: for every configuration
it restarts the workers (through their agents) and the main server, reads one long prompt and three short ones, and
takes each stage's own times from the timing lines.

```
python3 tools/stage_tune.py tune  cluster.json OUTDIR    # measure and search; writes OUTDIR/main.json, nodes.json, report.md
python3 tools/stage_tune.py apply cluster.json OUTDIR    # start the workers as nodes.json says, then serve OUTDIR/main.json
```

```json
{ "main": {"config": "/opt/strata/strata-q4_k_xl.json", "exe": "/opt/strata/build/strata",
           "cwd": "/opt/strata", "python": "/opt/strata/.venv/bin/python", "port": 8080,
           "api_key_file": "/opt/strata/api-key", "extra": []},
  "nodes": [{"name": "pc-b", "agent": "http://192.0.2.11:7840", "host": "192.0.2.11", "port": 7841, "ship": true,
             "cache": "auto", "max_layers": 24},
            {"name": "pc-c", "agent": "http://192.0.2.12:7840", "host": "192.0.2.12", "port": 7841, "ship": true,
             "cache": 1000, "max_layers": 14}],
  "ship": {"model_dir": "/data/models/<model>", "pack_dir": "/data/packs/<pack>",
           "profile": "/opt/strata/data/expert-profile.bin"},
  "token_file": "/opt/strata/stage-token", "corpus": "/opt/strata/corpus.txt",
  "goal": {"measure_tokens": 32000, "answer_tokens": 500,
           "profile": [[8000, 0.3], [32000, 0.5], [100000, 0.2]], "max_context": 131072},
  "chunks": [5888], "max_runs": 8, "init_splits": [24, 40] }
```

- The nodes run in the order listed (every node but the last is a relay).
- The model's layer count and each layer's bytes come from the GGUF next to the main config's `--native`, so a split is
  balanced by bytes; `max_layers` caps a node by its RAM.
- The search: a first split by VRAM (or `init_splits`); the split that gives every stage the same measured prompt time
  per byte; the larger chunks at the better of the two (a chunk that runs out of VRAM is tried once more with the
  reserve the engine asks for); then each split point two and then one layer either way while that wins.
- The best few then read every prompt length of `goal.profile` (the score: the weighted mean of prompt / prefill +
  answer / decode), and the winner reads one prompt near `max_context` (131072, or the model's 262144).
- With `"ship": true` the tuner sends each node the layers it lacks before starting it and names the model in its
  `/start`. The report adds a note for a Windows (or WSL2) node whose clocks are not locked.
- On A + B with RVN IQ3_S (one node, 9 configurations, 12 minutes; [its report](../bench/results/2026-10-10-remote-stage/data/tuner-rvn-2pc-report.md)) it picked K=26 at chunk 5888, the same as the
  table. On A + B + C with UD-Q4_K_XL (8 searched configurations, two profile runs and the max-context check: 33
  minutes, including shipping 18.5 GiB of layers the nodes lacked) it kept 24 / 16 / 8 at chunk 5888: a weighted
  request of 49.1 s, and a 125,230-token prompt read at 1,410 tok/s with decode at 33.8. Its report:
  [bench/results/2026-10-10-remote-stage/data/tuner-q4xl-3pc-report.md](../bench/results/2026-10-10-remote-stage/data/tuner-q4xl-3pc-report.md).

`main.json` is a server config with the token in its `"env"`; keep it private.

## What is turned off or refused

- **Refused at start** with a remote stage: `--batch`, `--pipeline-windows`, `--vision`, `--peer-device`,
  `--control-vector` (and the speed projection, which is one), `--expert-cache-remote`.
- **Turned off without an error:** the prompt cache, the conversation cache and mid-prompt checkpoints (each PC
  holds only its layers' state, so every prompt is read from token 0), and `--kv-grow`.
- **Not checked** with a remote stage: `STRATA_PREFILL_HELP`, `--adapt-async 1`, `--resident-experts`,
  `--mmap-experts` / `--resident-budget-gib` (the mapped expert source gets no layer range; it maps the whole pack),
  `STRATA_PF_FUSED=1`.

## Security

- **The link is plain TCP, not encrypted.** Every verify window and prompt chunk carries the token ids and the
  layers' hidden states: whoever can read the traffic can read the conversation. Use it on a LAN you trust; across
  anything else put it inside a tunnel (not measured).
- **The token is a shared secret in the hello, also in clear.** It keeps other hosts on the LAN from driving a
  worker; it is no protection against someone who sees the traffic. Bind the worker to the LAN address
  (`--stage-bind`) and firewall the port.
- **A node agent's token is a full credential:** with it a request can start the engine with the flags the agent
  allows, write files under `data_dir`, and (with `"allow_put_bin"`) the engine itself. Treat it as a password and
  keep the agent's port on the LAN. An agent run as administrator for `"gpu_clocks"` starts its workers as
  administrator too.
- A worker serves one main process at a time; a connection that sends no hello within 10 s is dropped.

## What it works with

| | status |
|---|---|
| Model family | Qwen3.8-Flash-Next only (48 layers, 512 experts top-10, 4 hyper-connection streams, n_embd 2560). The hello checks the geometry |
| RVN IQ3_S, native pack read from its GGUF | **measured** (two and three PCs) |
| Huihui UD-Q4_K_XL (Unsloth UD layout), native pack read from its GGUF | **measured** (one, two and three PCs) |
| Other quantizations and models of the family (the official IQ3_S, IQ3_XXS, IQ2_XS, Coder, Swift 1.5, UD-IQ4_XS) | not tested. The hand-off is fp32 rows whatever the quantization, the arena range is cut by layer from the pack's expert table and the shipper works by tensor name |
| A pack with `experts.bin`, or `--shared-expert-arena` | refused at start (the layer range needs the GGUF loader) |
| Linux, NVIDIA (CUDA 13.0) | measured: Ubuntu 24.04 in an LXC container (main), a Linux host with kernel 6.14 (worker) |
| Windows 11 native, NVIDIA (MSVC 2022, CUDA 13.0) | measured as a worker (and its node agent); as the main PC not tested |
| WSL2 (Windows 11, mirrored networking) | measured as a worker |
| AMD (HIP) | builds (ROCm 7.0, gfx1100, in a container); not run: no AMD card was available |
| Intel (SYCL) | the SYCL port builds (oneAPI 2026.1.1, in a container) but has no remote stage: it keeps its own copies of the engine files this feature changes |
| Links | measured on 1 GbE and on 10 GbE (limited to ~400 MB/s by its slot) |

## Troubleshooting

| message or symptom | what to do |
|---|---|
| decode slower on a Windows or WSL2 worker than its card should be; the worker's `wait for the GPU` far above another card's for the same layers | lock its clocks ([above](#windows-lock-the-worker-gpus-clocks)) |
| `N MiB of VRAM free with everything loaded - LOW` on a card that also drives a desktop | a smaller `--expert-cache N` (on a native pack N counts the largest blob, see [Picking K](#picking-k-and-the-chunk)); the desktop's own VRAM use moves |
| Task Manager shows many GB of "shared GPU memory" on a Windows worker | its pinned experts (12.2 GiB for 8 UD-Q4_K_XL layers), not a spill: it stayed at 14.6 GB while the worker's VRAM went from 6.8 to 5.3 GB |
| `verify: instantiate: out of memory` at the first long prompt; at start `... add --vram-reserve-mib M` | add `--vram-reserve-mib M` (the tuner adds 100 more) or use a smaller `--prefill` |
| `remote stage: cannot connect to HOST:PORT` | start the worker first and wait for `listening on`; check `--stage-bind`, the address and the firewall (Windows: an inbound rule; WSL2: a Hyper-V firewall rule too) |
| `cannot connect ...: No route to host` over a second NIC that shows its link up | check that it receives at all (`ip -s link`); a 10 GbE card here received nothing after its host rebooted until the link was taken down and up |
| a worker restarted right after a stop fails (little VRAM free) | the previous worker is still unpinning (more than 30 s with ~30 GB of experts); wait until its process is gone |
| `remote stage: the worker refused: wrong token (STRATA_STAGE_TOKEN)` | the same `STRATA_STAGE_TOKEN` on every PC |
| `remote stage: the two sides differ (protocol ..., first remote layer ..., K/V type ..., model pack ...)` | the same build of the engine on every PC, the same pack, the same `--kv`, and `--stage-begin` = the previous stage's K |
| `the main process reads prompts in chunks of N tokens, this worker's chunk is M` | the worker's `--prefill` must be at least the main process's chunk |
| `the main process's context (N) or window (T) is larger than this worker's` | the worker's `--max-context` and `--spec` must be at least the main process's |
| `ArenaExpertSource: a layer range needs a native pack read from its GGUF` | a pack directory with `experts.bin`: use a copy without it; drop `--shared-expert-arena` |
| `strata generate: remote stage: it does not support ...` | remove that flag (see above) |
| a request fails with a link error | the next request's prompt stops the main engine ("the link failed earlier") and the server starts it again, which connects anew; the worker waits for the new connection. Read from the code, not tested |
| decode slower than expected on Linux | look at `decode expert cache hit rate` on every PC; keep the adaptive tier on everywhere (do not set `--adapt-every 0`) |
| numbers that move between runs | record the other load on every PC and repeat runs: the same configuration decoded a ~30K prompt's answer at 50.2 and 44.0 tok/s twenty minutes apart |
