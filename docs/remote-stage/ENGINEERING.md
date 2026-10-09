# Remote stage: design, process and measurements

How the layer split across PCs works inside, how it got to its present speed, every number measured on it, and what
is still open. Using it: [../REMOTE_STAGE.md](../REMOTE_STAGE.md). Raw numbers:
[bench/results/2026-10-10-remote-stage](../../bench/results/2026-10-10-remote-stage/README.md). Code:
`include/strata/net/stage_link.hpp` (the protocol), `src/net/stage_link.cpp` (the sockets, POSIX and Winsock), the
`remote_main` / `stage_worker` / `stage_relay` parts of `src/program/generate.cpp`, `Verifier::set_remote` /
`set_no_head` (`src/core/verify.cpp`), `Prefill::remote_send` / `remote_recv` (`src/prefill/prefill.cpp`),
`ArenaExpertSource::set_layer_range` (`src/core/expert_source.cpp`), and `tools/stage_node.py`,
`tools/stage_tune.py`, `tools/stage_ship.py` (tests: `tools/test_stage_node.py`).

## Design

### Where the network sits

An in-process layer split already hands a token from one stage to the next once per verify window (a pinned,
mapped buffer of T rows that the next stage reads) and hands each prompt chunk's rows to the next stage on a
`std::async` task while the first stage reads the next chunk. The remote stage puts the network exactly there. No
CUDA graph, kernel or per-layer path changes; only the object at the end of the hand-off is a TCP link instead of the
next stage.

The split chosen: the **main process** runs layers [0, K), the output head, sampling and the draft (MTP) layer; the
**worker** runs [K, 48) and sends back the residual after its last layer instead of running the head. A window
therefore goes main -> worker -> main. This keeps everything a request touches besides plain layers on the main PC
(the API, sampling, the drafter, the PLE table that layer 1 reads, penalties, logprobs); the worker has no notion of a
request. The price is the rows coming back every window (as large as the rows going out). The alternative of putting
the head and the drafter on the worker would have saved that return trip and moved most of the request state.

### The protocol

One TCP connection per main process. Every message is a 40-byte header (`magic` "STRM", `type`, three int64 fields
`a b c`, payload `bytes`) and a payload. Protocol version 3.

| message | header fields | payload | reply |
|---|---|---|---|
| Hello | | `StageHello` (below) | HelloOk with the worker's hello, or Error |
| Run (a verify window) | a = T, b = first position | int32 tokens[T], fp32 rows[T x 12,804] | RunOk: a = the worker's time (us), rows[T x 12,804] |
| Commit | a = accepted count | none | none (an error comes back on the next reply) |
| Prefill (a prompt chunk) | a = T, b = first position, c = flags (bit 0: the prompt is one chunk) and `skip` << 8 | int64 tokens[T], fp32 rows[T x 10,240] | PrefillOk: a = the worker's time (us), rows[(T - skip) x 10,240] |
| Reset (a fresh prompt) | | none | ResetOk |
| Error | | the message | |

**The hello** carries what both sides must agree on, and each side checks the other's: the protocol, `n_embd`, `hc`,
`n_layers`, `n_expert`, the first remote layer (K), the floats per window row, the K/V element type, and a
fingerprint of the pack (FNV-1a over the pack's `index.txt` and `native_experts.txt`). The worker also refuses a main
process whose prompt chunk, context or window is larger than its own, and a wrong `STRATA_STAGE_TOKEN` (the token is
the hello's last field; the worker's own hello carries none). A peer that sends no hello within 10 s is dropped; after
the hello the worker waits for messages without a time limit (the next prompt may be hours away), while sends and the
main process's receives time out after `STRATA_REMOTE_TIMEOUT_S` (300 s). Sockets: `TCP_NODELAY`, 8 MiB buffers,
keepalive after 30 s idle, 3 probes 10 s apart.

**Bytes, Qwen3.8-Flash-Next** (n_embd 2560, 4 hyper-connection streams, everything fp32):

- a window row is `hc * n_embd + n_embd + hc` = 12,804 floats = **51,216 B**: the four residual streams, the pending
  write of the last layer and the inject. A window of T tokens sends T rows and T token ids and gets T rows back.
  With `--spec 4`, windows of 1 to 6 tokens were captured: up to ~307 KB each way.
- a prompt row is `hc * n_embd` = 10,240 floats = **40,960 B** plus an 8-byte token id. A 5888-token chunk is 241 MB
  out. Only the rows the drafter reads come back: the drafter attends to the last `--mtp-window` (32,768) cells, so
  the main process asks for rows from position n - 32,768 - 128 on (`Prefill::set_remote_rows_from`). For the 90,921-
  token prompt that is 3.72 GB out and 1.35 GB back; for a prompt shorter than the window, every row comes back.
  Without a drafter no prompt rows come back.
- a commit is the header alone.

### Pipelining the prompt

- **main process:** a chunk's rows are copied into one of two pinned hand-off buffers and sent on a task, in order,
  one send at a time; the replies are taken on another task chain in the same order (receive into a pinned buffer,
  upload, then the usual per-chunk callback that builds the drafter's K/V). The main process reads chunk c + 1 while
  chunk c is on the wire or in the worker.
- **worker:** a receiving thread reads whole messages and puts a chunk's rows straight into one of two pinned input
  buffers (it waits when both are in use); the serving thread handles one message at a time in arrival order; a
  chunk's reply goes out on its own thread from one of two output buffers while the serving thread reads the next
  chunk.

### The window path

- **main:** stage 0 (`ver`, layers [0, K)) writes its hand-off as for an in-process split. The next "stage" is a
  head-only verifier (`set_stage(n_layers, -1)`) whose `set_remote(run, commit)` hooks send the window to the worker
  and wait for its rows before running the head; its commit sends the accepted count (one way) before committing
  locally.
- **worker:** a verifier over [K, 48) with `set_no_head(true)`: after the last layer it folds that layer's pending
  write into the residual (as the unsplit window does before the head) and writes the hand-off instead of running the
  head. Before a window it gives back the prompt path's loan from its cache (below) and applies the adaptive tier's
  pending swaps; after every `--adapt-every` commits it runs the adaptive tier on its own layers.

Decode is sequential: a window's time is the main process's layers, head and drafter, plus every worker's layers,
plus every hop's round trip. More PCs add VRAM for the expert cache and RAM pools; they do not run at the same time on
one conversation. Prompt chunks do overlap across stages, so the prompt rate is set by the slowest stage.

### More than two stages (relay workers)

A relay worker (`--stage-end K2 --stage-next HOST:PORT`) runs [K, K2) and is a client of the next worker as the main
process is its client: windows go on synchronously and the next worker's rows are this worker's reply; prompt chunks
go on through a `StageRelay` (one send in flight, in order), and the next worker's reply to a chunk is read and
forwarded on the replying thread (`StageHandlers::prefill_reply` defers the answer until it arrives). A reset and a
commit go on to the next worker first; a reset waits for a chunk reply still going out. The main process sees one
worker. A relay connects to the next worker when it starts (30 tries, 2 s apart; a refused hello - a wrong token or
a context mismatch - is not retried) and again at a reset if that link broke; when its main process goes away it drops
the link to the next worker, so replies still owed on it cannot reach the next main process.

### Windows

`stage_link.cpp` has one socket layer for both systems: a handle type (`SOCKET` on Windows, kept as `intptr_t` in the
header), close / shutdown / poll / errno wrappers (`WSAPoll`, `WSAGetLastError`), timeouts in the unit each system
takes (milliseconds as a `DWORD` on Windows, a `timeval` on Linux), `WSAStartup` once per process, error texts in
English (`FormatMessageA` with an English language id: the node agent reads the worker's log back as UTF-8), and
`SO_EXCLUSIVEADDRUSE` instead of `SO_REUSEADDR` for the listening socket (Winsock's `SO_REUSEADDR` binds over a live
listener). `ws2_32` is linked on Windows. The worker loads its experts with the Windows unbuffered GGUF reader
(`FILE_FLAG_NO_BUFFERING`); its byte count now covers the layer range (it reported the whole model's bytes, and a
Windows worker refused to start).

### Each process holds only its own layers

| what | how | log line |
|---|---|---|
| experts in RAM | `ArenaExpertSource::set_layer_range(lo, hi)`: one contiguous range of the arena layout is allocated, pinned and read from the GGUF; the base pointer is shifted back by the range's first byte, so every `blob_offset()` of a layer in range is unchanged; `blob()` is null outside it | `expert arena: layers 40-47 only (12.21 of 71.73 GiB; remote stage)` |
| dense weights | the tensors of other layers are skipped, as `STRATA_STAGE_TRIM=1` does for an in-process split | `the dense weights of layers 40-last only` |
| expert cache | the profile is cut to the process's layers before the cache is filled | `4096 of the 24576 pairs are this process's layers` |
| session | `session_init(lo, hi)`: only its own layers' K/V and GDN state | |
| PLE | the worker has no PLE run (layer 1 always runs in the main process, since K >= 2) | `no PLE here` |
| head | not loaded on the worker | |

**Layer sizes by pack** (the experts only, from each pack's `native_experts.txt`: blob bytes x 512; each layer's
dense tensors come on top):

| pack | experts per layer | all 48 layers | heavier layers |
|---|---:|---:|---|
| RVN IQ3_S | 1.111 GiB | 53.32 GiB (the engine's log agrees) | none |
| Unsloth UD-IQ4_XS | 1.111-1.660 GiB | 55.43 GiB | 2, 4, 30, 46, 47 |
| Unsloth UD-Q4_K_XL | 1.465-1.904 GiB | 71.73 GiB (the engine's log agrees) | 2, 4, 30, 46, 47 |

The range needs the loader that reads a native pack's experts from its GGUF: a pack with `experts.bin`, a shared
arena or `STRATA_ARENA_MMAP=1` would need offsets the range does not have, so the first two refuse at start and the
third is ignored for a range. `--mmap-experts` (and `--resident-budget-gib`, which uses it) is another expert source
that gets no range; not tested with a remote stage.

### Lending and refilling on the worker

On one card the prompt path borrows expert-cache slots for its buffers and refills them before decode. The worker does
the same: at a prompt's first chunk it lends slots for the main process's chunk size, and the first window after the
prompt refills them. The lend and refill code was moved out of the request loop so the worker's handlers can call it;
the main process uses it as before.

### Timing lines

With `STRATA_REMOTE_TIMING=1`: per chunk, a sending stage's own time (`strata prefill: layers A-B, chunk T=... own N
ms, waited M ms for the previous send`), the main process's view of each reply (`from its send to its rows = worker
+ link and queue`) and the worker's own time per chunk; every 64 windows, the round trip split into worker and link,
the main process's own share between windows, and on a worker its own layers' share (split into waiting for the GPU,
the CPU expert pool and the plan, and host staging) and, on a relay, the next worker's share. The worker puts its own
time in every reply's header, so the main process can split a round trip without synchronized clocks. The GPU wait is
what showed the Windows clock problem below.

## How it got here

All on PCs A and B below, RVN IQ3_S, K=24, the 32K prompt, chunk 2048 (PC A's host ran heavy CI load during these
runs, not recorded per run):

| version | prefill tok/s | decode tok/s | 3080 decode hit rate |
|---|---:|---:|---:|
| first: chunks one after the other, adaptive tier off in both processes | 448 | 38.4 | 55% |
| send and receive pipelined | 554 | 38.8 | - |
| + the worker replies on its own thread, + the adaptive tier on in both processes | 754 | 49.3 | 77-79% |
| the 3080 alone, the same chunk | 444 | 37.6 | 59-60% |

What went wrong on the way, and why:

- **The adaptive tier was off.** The first versions set `--adapt-every 0` for a remote stage. The caches then kept only
  what the profile put there: 55% hits on the 3080. With the tier on in both processes, 77-79%.
- **Chunks in series.** Send, worker, reply and the main process's next chunk did not overlap: 448 tok/s. Pipelining
  them gave 554, and the worker replying on its own thread 754 (with the tier).
- **2048-token chunks.** With every split point the split read long prompts slower than the 3080 alone at its auto
  chunk (best 846 vs 1,120 tok/s at 100K). The time per layer of a chunk grows much more slowly than the chunk, so
  small chunks pay it more often (table below). 5888 and 8192 fixed it.
- **A restart that did not wait.** A worker restarted right after a stop found only 496 MiB of VRAM: the previous
  worker, with ~30 GB of pinned experts, was still unpinning (more than 30 s). The node agent now waits for the process
  to exit (up to 180 s).
- **8192-token chunks without a reserve.** The 3080 had 65 MiB of VRAM free after loading, and the long prompts failed
  with `verify: instantiate: out of memory`. With `--vram-reserve-mib 1200` it ran. The tuner retries a chunk with the
  reserve the engine's start-up line asks for.
- **A third PC made it slower.** With an RTX 3070 as a third stage (under WSL2, then native Windows) decode fell from
  45-54 to 34-41 tok/s. Splitting each worker's window time into GPU wait, pool and staging showed the 3070 waiting
  11-26 ms for 8 layers where the 2080 Ti waited 4.6-5.9 ms; sampling its P-state showed why (below). Locking its
  clocks brought it to 3.7-4.1 ms.
- **The relay's link.** On 1 GbE the middle PC carried every prompt chunk in and out, each way: its link ran at line
  rate for one long prompt: 2.6 GB each way in the ~38 s it crossed, at up to 123 MB/s ([data/nic-pc-b-1gbe-3pc.txt](../../bench/results/2026-10-10-remote-stage/data/nic-pc-b-1gbe-3pc.txt)). 10 GbE between PCs A and B took most of it away (100K prompt:
  1,248 -> 1,576 tok/s).
- **Shipping into WSL2 and Windows.** Writing 16.6 GiB into WSL2 left 17.9 GB of page cache in the VM, which Windows
  did not get back; the agent now syncs and drops what it wrote and what a worker read. On Windows the shipped files
  were written dense (Python's `os.truncate` grows a file by writing zeros there, and NTFS/ReFS write the gap before
  an offset as zeros); they are now marked sparse and grown with `SetEndOfFile`.
- **Reviews.** Review passes added the token, `--stage-bind`, the pack fingerprint, the 10 s hello timeout, keepalive,
  a clean shutdown of a broken link, refusals of control vectors and helper-GPU caches, `--kv-grow` off, the relay's
  recovery after a main process goes away, the source fingerprint of shipped files (a model changed at the same path
  voids the old byte ranges) and the agent's limits (one `/start` at a time, a flag whitelist, no executables unless
  allowed, a constant-time token check).

## Measurements

### The rig

| | PC A (main) | PC B | PC C |
|---|---|---|---|
| GPU | RTX 3080 10 GB (sm_86), PCIe 4.0 slot negotiating **x8**, probe 13.0 GB/s | RTX 2080 Ti 11 GB (sm_75), PCIe 3.0 x16, probe 13.2 GB/s | RTX 3070 8 GB (sm_86), PCIe 4.0 x16, probe 26.3 GB/s; also drives the desktop (~1 GB of its VRAM), so its worker was kept to ~6.5 GB |
| CPU | Ryzen 9 5900XT, 16 cores / 32 threads (AVX2, no AVX-512) | Ryzen 9 5900X, 12 cores / 24 threads (AVX2) | Ryzen 7 5700X3D, 8 cores / 16 threads (AVX2) |
| RAM | 4 x 32 GB DDR4-3200 (96 GB for the container) | 4 x 16 GB DDR4-2400 | 64 GB DDR4-3600 |
| OS | Ubuntu 24.04 in an LXC container, host kernel 7.0.2 | Linux host, kernel 6.14.8 | Windows 11 (build 26200), driver 591.86; the same PC's WSL2 for the WSL2 runs |
| build | CUDA 13.0, driver 580.178.04 | the same Linux build | MSVC 2022 + CUDA 13.0 (native); the Linux build under WSL2 |
| other load | CI jobs on the host during some runs ("other CPU %") | none recorded | a desktop session |

Links: PC A's 2.5 GbE and PC B's 1 GbE through a switch, 117 MB/s measured (a 64 MiB HTTP POST to the node agent),
ping 0.13-0.56 ms; a 10 GbE pair (ConnectX-3) between A and B, 390-400 MB/s measured over TCP because PC B's card sits
in a slot that negotiates PCIe 2.5 GT/s x2; PC C on 1 GbE, 114 MB/s from PC B.

Models and settings: RVN-Qwen3.8-Flash-Next IQ3_S (8 GGUF shards) and Huihui-Qwen3.8-Flash-Next UD-Q4_K_XL (4
shards), each with a native pack from `tools/iq_pack.py` reading the experts from the GGUF; `--expert-cache auto
--spec 4 --spec-min-p 0.5` with the stock draft layer, `--max-context 131072 --kv int8 --kv-resident 32768`, no
conversation cache. Requests greedy, reasoning off. "Short decode": the mean of nine short prompts (code, prose,
Chinese; 200-256-token answers). The long prompts: 7,521 / 30,167 / 57,383 / 90,921 tokens of English text with a
320-token answer; speeds from the engine's own `timings`. One run per row.

### Two PCs, RVN IQ3_S, 1 GbE (2026-10-09)

| configuration | short decode | 8K prefill / decode | 32K | 64K | 100K | other CPU % |
|---|---:|---:|---:|---:|---:|---:|
| 3080 alone, `--prefill auto` (5888) | 39.9 | 791 / 38.1 | 1,118 / 37.2 | 1,146 / 39.0 | 1,120 / 38.0 | 34 |
| K=20, chunk 2048 | 51.7 | 491 / 53.2 | 604 / 50.0 | 609 / 52.1 | 600 / 52.0 | 29 |
| K=24, chunk 2048 | 51.3 | 593 / 54.9 | 771 / 50.7 | 764 / 48.1 | 750 / 47.7 | 103 |
| K=26, chunk 2048 | 50.7 | 629 / 51.8 | 816 / 48.7 | 865 / 47.9 | 846 / 47.7 | 109 |
| K=28, chunk 2048 | 50.2 | 607 / 48.0 | 761 / 46.0 | 798 / 46.1 | 812 / 45.3 | 19 |
| K=26, chunk 4096 | 52.3 | 740 / 50.1 | 1,211 / 47.8 | 1,330 / 49.2 | 1,328 / 45.6 | 30 |
| K=24, chunk 5888 | 54.0 | 652 / 51.1 | 1,226 / 50.2 | 1,381 / 50.6 | 1,374 / 47.8 | 26 |
| K=26, chunk 5888 | 53.5 | 655 / 48.1 | 1,231 / 49.7 | 1,474 / 47.7 | 1,521 / 45.6 | 18 |
| K=28, chunk 5888 | 49.4 | 690 / 47.7 | 1,217 / 45.9 | 1,503 / 45.9 | 1,591 / 45.7 | 12 |
| K=26, chunk 8192, `--vram-reserve-mib 1200` | 52.5 | 589 / 46.0 | 1,225 / 47.5 | 1,485 / 47.2 | 1,633 / 45.4 | 44 |
| K=26, chunk 8192, no reserve | 58.4* | out of VRAM | | | | 10 |

\* three code prompts only (the run failed at the next one); code decodes fastest, so it is not comparable.

The 3080's decode hit rates at chunk 2048: K=20 86.7%, K=24 81.3%, K=26 77.5-80.6%.

### The time per layer of a prompt chunk

From the timing lines (each stage's own time for a full chunk, divided by its layers; RVN IQ3_S):

| chunk | 3080, ms per layer | per token per layer | 2080 Ti, ms per layer | per token per layer | source |
|---:|---:|---:|---:|---:|---|
| 2048 | ~96 | ~47 us | ~112 | ~55 us | the sweep, K=24 |
| 4096 | 89 | 22 us | 120 | 29 us | tuner, K=22 |
| 5888 | 106 | 18 us | 130-134 | 22-23 us | tuner, K=26 and K=27 |
| 8192 | 125 | 15 us | 160 | 20 us | tuner, K=28 with the reserve |

The tuner rows divide each stage's chunk time in [the first tuner run's report](../../bench/results/2026-10-10-remote-stage/data/tuner-rvn-2pc-report.md)
by its layers; the 2048 row is from the sweep's timing lines (not kept).

A layer's experts are 1.11 GiB; streaming that over a ~13 GB/s link takes ~85-92 ms (an estimate from the probe's
bandwidth), close to the 2048-token time: the prompt path streams most of each layer's experts once per chunk whatever
its size, which is why the chunk matters more than the split point. The 3070 (PCIe 4.0 x16, 26.3 GB/s) read 8
layers of a 5888-token chunk in 709-877 ms on Windows (89-110 ms a layer), the 2080 Ti in 1,044-1,143 ms.

### Where the decode time goes (RVN IQ3_S)

- Two PCs, K=27, chunk 5888, ~30K (tuner): the round trip 18-21 ms = worker 15.5-17.9 + link 2.8-3.2 (1 GbE); the
  main process between windows (head, drafter, layers 0-26) 26-46 ms (the main process's timing lines during the
  first tuner run; not kept, the run's per-stage window times are in its report).
- Three PCs, 24 / 16 / 8 layers, A-B 10 GbE, C locked, ~91K prompt: the main process between windows 25.3 ms; the
  round trip 23.0 ms = worker 21.9 + link 1.2; the 2080 Ti's own 16 layers 10.7-13.6 ms (GPU wait 9.5-11.5) and the
  next worker 7.2-11.6 ms, of which the 3070's own 8 layers 4.3-6.2 ms (GPU wait 3.9-4.5). The B-C hop on 1 GbE is
  the rest, ~3-5 ms (estimated from those two, taken over different windows).

### Three PCs, RVN IQ3_S (2026-10-09)

Layers 24 / 16 / 8 on the 3080 / 2080 Ti / 3070, chunk 5888, the 3070 with 2,400 cache slots (~6.3 GiB):

| configuration | short | 8K | 32K | 64K | 100K |
|---|---:|---:|---:|---:|---:|
| A + C only (C under WSL2, clocks not locked), 1 GbE, K=40 | 36.2 | 685 / 33.7 | 1,018 / 31.8 | 1,138 / 32.4 | 1,189 / 33.0 |
| all on 1 GbE, C under WSL2 | 40.9 | 497 / 41.3 | 706 / 37.3 | 974 / 36.8 | 1,248 / 36.0 |
| A-B on 10 GbE, C under WSL2 | 42.1 | 635 / 39.9 | 1,035 / 35.3 | 1,418 / 34.8 | 1,576 / 34.1 |
| A-B on 10 GbE, C native Windows, clocks locked | 51.3 | 658 / 47.8 | 1,110 / 46.5 | 1,451 / 44.1 | 1,640 / 44.7 |

Against the best two-PC rows (K=26 at 5888: 53.5 short, 100K 1,521 / 45.6; at 8192 with the reserve 1,633), the
locked three-PC chain reads 100K prompts as fast or a little faster and decodes 1-8% slower (short 51.3 vs 53.5; 8K 47.8 vs 48.1; 32K 46.5 vs 49.7; 64K 44.1 vs 47.7; 100K 44.7 vs 45.6).

### The 3070: WSL2, native Windows, clocks (2026-10-09)

The same last stage on each worker: layers 40-47, 2,400 cache slots, chunk 5888, 1 GbE; the 3080 runs 0-39 and the
head. One ~30K prompt and the nine short ones each:

| worker | short decode | 32K prefill / decode | round trip (worker + link) | GPU wait per window | prompt chunk (8 layers) |
|---|---:|---:|---:|---:|---:|
| RTX 3070, WSL2 | 33.0 | 1,033 / 30.6 | 25.7 ms (21.6 + 4.0) | 10.6-25.7 ms | 887-1,130 ms |
| RTX 3070, native Windows | 33.8 | 995 / 31.7 | 23.8 ms (19.4 + 4.4) | 11.9-22.2 ms | 855-877 ms |
| RTX 3070, native Windows, clocks locked | 41.8 | 1,000 / 35.7 | 9.4 ms (5.3 + 4.1) | 3.7-4.1 ms | 709-711 ms |
| RTX 2080 Ti, Linux (two runs) | 41.8 / 41.6 | 1,042 / 37.7, 1,041 / 39.0 | 9.6 / 9.8 ms | 4.6-5.9 ms | 1,044-1,143 ms |

WSL2 and native Windows were alike, so the GPU's virtualization under WSL2 was not the cause. `nvidia-smi` sampled
every 100 ms through the short-prompt part of each run:

| | samples | P-states | memory clock | SM clock | mean GPU use |
|---|---:|---|---|---|---:|
| 3070, Windows, driver's choice | 1,056 | P2 6%, P3 34%, P5 51%, P8 8% | 6,801 MHz 6%, 5,001 MHz 35%, 810 MHz 51%, 405 MHz 8% | mostly 300-900 MHz | 26% |
| 3070, Windows, `-lgc 1500,1905 -lmc 7001,7001` | 781 | P2 100% | 6,801 MHz | 1,500-1,905 MHz | 10% |
| 2080 Ti, Linux | 808 | P2 100% | 6,800 MHz | 1,350 MHz and up | 10% |

A worker idles between its windows while the other stages run (25-45 ms here). The Windows driver takes that as a
light load and lowers the clocks, and a decode window, which streams a few MB of weights per layer, then runs at a
fraction of the memory bandwidth (P5's 810 MHz memory clock). The Linux driver held P2 for the CUDA context with the
same 10% mean load. Prompt chunks are long enough to bring the clocks up, so they were barely affected. Locked, the
3070 drew 57 W on average (p95 94 W, peak 151 W) and reached 59 °C.

**Shared GPU memory is not a spill.** Task Manager showed ~14.6 GB of "shared GPU memory" for a Q4_K_XL worker
(layers 40-47). Its pinned arena is 12,504 MiB and its pinned K/V ~266 MiB; the rest is other pinned buffers. The
number did not move when the worker's VRAM did: with `--expert-cache` 1800 / 1400 / 1000 (1,766 / 1,747 / 1,248
slots) the dedicated VRAM was 6,774 / 6,784 / 5,262 MiB and the shared 14,662 / 14,598 / 14,598 MiB. A driver spill
of VRAM into system memory would have grown and shrunk with it. (The cache count is a budget of N times the model's
largest expert blob, filled with this worker's smaller blobs, and capped by free VRAM.)

### Huihui UD-Q4_K_XL (2026-10-10)

| configuration | short | 8K | 32K | 64K | 100K | 100K request |
|---|---:|---:|---:|---:|---:|---:|
| 3080 alone, served config: `--resident-budget-gib 70` | 23.9 | 302 / 22.2 | 324 / 21.8 | 328 / 20.9 | 326 / 21.8 | 293 s |
| 3080 alone, no budget (all 71.7 GiB pinned) | 27.3 | 206 / 25.4 | 212 / 24.5 | 209 / 25.3 | 209 / 23.7 | 449 s |
| A + B (10 GbE), K=26, chunk 5888 | 38.8 | 659 / 34.4 | 1,147 / 36.9 | 1,192 / 37.5 | 1,164 / 33.6 | 88 s |
| A + B + C, 24 / 16 / 8, chunk 5888, C locked, 1,766 slots | 38.4 | 549 / 35.3 | 945 / 36.6 | 1,210 / 33.2 | 1,334 / 35.2 | 77 s |

- **One 10 GB card.** With the RAM budget, 69.93 GiB of experts were page-locked and 1.7-1.8 GiB read from the files;
  the cache got 620 slots and the auto prompt chunk 2048. All pinned, the cache got 571 slots and the auto chunk
  1280: decode +14%, prompts 32-36% slower. The card's VRAM, not the RAM, is what limits this model on one 10 GB card.
- **Two PCs:** the 2080 Ti's 22 layers were the slow stage of a long prompt (4.7-5.2 s per chunk at ~80K, against
  2.8-2.9 s for its 16 layers in the three-PC run), and its CPU pool worked harder in decode (pool + plan 4.2-6.5 ms a
  window with 2,456 cache slots for 22 layers, 1.1-2.4 ms with 2,717 slots for 16). The main process: 41.6-43.4 ms
  between windows; round trip 22.5-22.7 ms (link 1.4).
- **Three PCs:** the main process 37.3-38.7 ms between windows; round trip 26.8-27.1 ms (link 1.0); the 2080 Ti's own
  11.3-15.2 ms, the 3070's 6.2-8.1 ms (GPU wait 5.3-5.9: 1.32x IQ3_S's expert bytes a layer, ~1.3-1.4x its wait). The 3070's
  prompt chunk: 1,463-1,712 ms; its VRAM was full (183 MiB free).

### The tuner on three PCs (UD-Q4_K_XL, 2026-10-10)

`tools/stage_tune.py tune` with the three PCs (two nodes: B a relay, C last), both `"ship": true`; C's agent ran
as administrator with `gpu_clocks`, so every start of C's worker locked its clocks. Goal: 32,000-token prompts, a
500-token answer, the profile 30% 8K / 50% 32K / 20% 100K, max context 131072; chunk 5888 only, at most 8 searched
runs, `init_splits` [24, 40]; C with `--expert-cache 1000` (1,248 slots, ~5.3 GB of VRAM used) and at most 14 layers,
B at most 24. Its whole report: [data/tuner-q4xl-3pc-report.md](../../bench/results/2026-10-10-remote-stage/data/tuner-q4xl-3pc-report.md).

| stages | prefill | decode | short | request s | per stage: chunk ms / window ms |
|---|---:|---:|---:|---:|---|
| **0-23, 24-39, 40-47** | 950 | 35.9 | 37.1 | **45.7** | 3,333 / 41.4; 2,611 / 14.0; 1,246 / 6.5 |
| 0-17, 18-33, 34-47 (balanced by cost) | 926 | 36.9 | 40.4 | 48.0 | 2,424 / 29.2; 2,418 / 14.5; 2,122 / 14.7 |
| 0-21, 22-39, 40-47 | 960 | 35.7 | 38.6 | 47.4 | 3,000 / 38.5; 2,867 / 16.3; 1,158 / 6.9 |
| 0-25, 26-39, 40-47 | 1,007 | 32.4 | 36.9 | 47.3 | 3,622 / 46.3; 2,259 / 11.8; 1,139 / 6.5 |
| 0-23, 24-37, 38-47 | 984 | 33.9 | 37.1 | 47.3 | 3,327 / 42.1; 2,132 / 11.8; 1,598 / 8.8 |
| 0-23, 24-41, 42-47 | 979 | 33.6 | 36.8 | 47.6 | 3,330 / 40.1; 2,841 / 15.4; 980 / 4.7 |
| 0-22, 23-39, 40-47 | 959 | 34.3 | 35.4 | 48.0 | 3,163 / 37.0; 2,697 / 15.2; 1,187 / 6.5 |
| 0-24, 25-39, 40-47 | 988 | 33.5 | 37.3 | 47.4 | 3,484 / 42.2; 2,413 / 12.8; 1,197 / 6.5 |

Profile (the best two): 0-23 / 24-39 / 40-47 8K 552 / 33.1, 32K 975 / 35.2, 100K 1,383 / 33.6, weighted 49.1 s;
0-25 / 26-39 / 40-47 8K 592 / 34.7, 32K 1,016 / 31.6, 100K 1,332 / 34.5, weighted 49.3 s. Near the max context: a
125,230-token prompt at 1,410 tok/s, decode 33.8. 33 minutes in all; the balance step's split (18 / 16 / 14) needed
9.26 GiB more on each node, shipped in ~110 s per node.

What it says: with the stages' chunk times equal (18 / 16 / 14) the 32K request was no faster than with the 3080
taking 3.3 s a chunk and the 3070 1.2 s. A ~30K prompt is five chunks; each chunk passes three stages and two links
before its reply, and on the 1 GbE hop to C 241 MB goes each way per chunk (~2.1 s at 114 MB/s), so the time to fill
the pipeline sets a mid-length prompt's speed more than the slowest stage does. Decode is the sum of the stages and
links in every split; the 3080's share (head, drafter, its layers) was 29-46 ms of it. The searched requests spread
over 45.7-48.0 s, about what one configuration moves between runs, so the eight splits are a tie and the tuner kept
its start.

### The tuner's first run (RVN IQ3_S, two PCs, 2026-10-09)

`tools/stage_tune.py tune`, one node (B), goal: a 32,000-token prompt (29,802 tokens as tokenized) and a 500-token
answer, chunks 4096 / 5888 / 8192, at most 9 runs:

| stages | chunk | prefill | decode | short decode | request s | per stage: chunk ms / window ms | other CPU % |
|---|---:|---:|---:|---:|---:|---|---:|
| 0-21, 22-47 (by VRAM) | 4096 | 1,035 | 49.9 | 51.6 | 40.9 | 1,964 / 23.3; 3,132 / 20.1 | 25 |
| 0-27, 28-47 (balanced) | 4096 | 1,174 | 41.5 | 46.1 | 39.3 | 2,686 / 32.0; 2,123 / 15.1 | 58 |
| 0-27, 28-47 | 5888 | 1,221 | 43.7 | 49.1 | 37.6 | 3,031 / 31.7; 2,630 / 15.3 | 35 |
| 0-27, 28-47 | 8192 | out of VRAM (65 MiB free; the engine asked for 1,147 MiB) | | | | | |
| 0-27, 28-47, reserve 1,247 | 8192 | 1,235 | 42.2 | 46.2 | 37.8 | 3,495 / 36.5; 3,195 / 16.4 | 47 |
| **0-25, 26-47** | 5888 | 1,230 | 48.7 | 49.3 | **36.3** | 2,753 / 28.6; 2,857 / 15.3 | 14 |
| 0-23, 24-47 | 5888 | 1,230 | 44.0 | 49.7 | 37.4 | 2,533 / 26.4; 3,200 / 17.1 | 26 |
| 0-24, 25-47 | 5888 | 1,199 | 47.8 | 50.4 | 37.2 | 2,657 / 28.5; 3,235 / 16.1 | 20 |
| 0-26, 27-47 | 5888 | 1,239 | 45.9 | 50.2 | 36.7 | 2,862 / 31.2; 2,813 / 15.1 | 15 |

The balance step used the first run's cost per layer and moved K from 22 to 28; at 5888 the walk came back to K=26,
the sweep's choice too (36.1 s there, 36.3 s here).

### The default path

Without `--remote-stage` or `--stage-worker` nothing of this should run, and the code moved out of the request loop
(lend and refill) should behave as before. Checked on PC A (2026-10-10): one single-GPU configuration (RVN IQ3_S, the
3080 alone, `--prefill auto` (5888), the draft layer on, 131072 context, int8 K/V) served by the feature's build twice
and by upstream main's build (the merge base, the same build options) once, between them; greedy answers (temperature
0, top-k 1, at most 200 tokens, reasoning off) to five prompts: code, prose, Chinese, arithmetic, and a 9,364-token
document (two prompt chunks). The three result files are byte-identical (the same MD5): text, token counts and finish
reasons. The script and the files: [default_path_check.py](../../bench/results/2026-10-10-remote-stage/default_path_check.py),
[data/default-path-*.json](../../bench/results/2026-10-10-remote-stage/data/).

## Lessons

- **The prompt chunk decides the split's prompt speed.** At 2048 no split beat one card; at 5888-8192 every split did
  from 32K on. A layer of a chunk cost the 3080 96 ms at 2048 tokens and 125 ms at 8192: four times the tokens for a
  third more time.
- **Keep the adaptive tier on in every stage.** Off in both processes: 55% hits on the 3080 and 38.4 tok/s decode;
  on: 77-79% and 49.3.
- **On Windows, lock a worker's GPU clocks.** The driver lowers them between decode windows; a 3070's 8 layers took
  12-22 ms a window instead of 3.7-4.1. The per-stage GPU wait in the timing lines is what shows it, and the tools now
  lock the clocks or say that they should be.
- **A third PC adds a hop to every window.** Decode runs the stages one after the other, so another stage costs its
  round trip and its fixed per-window work. It paid for long prompts (prompt chunks overlap across stages) and for a
  model whose experts the first two cards could not cache well (UD-Q4_K_XL); for decode on IQ3_S it was 1-8% behind
  two PCs.
- **The slow stage moves; give the faster card more layers.** With the 3070's clocks locked it was the fastest stage
  per layer, yet held the fewest layers; on UD-Q4_K_XL the 3080's 24 layers were the slow stage of a prompt. Balance by
  measured time per byte, and again when the chunk or the model changes.
- **A big chunk takes VRAM from the decode cache.** At 8192 the 3080 had 65 MiB free and the first long prompt ran
  out; a reserve fixed it.
- **A relay's link carries everything twice.** On 1 GbE the middle PC's link was the limit of a three-PC prompt; put
  the fastest link there.
- **Wait for pinned memory to be released between restarts** (more than 30 s for ~30 GB).
- **Plain TCP on a LAN is enough for decode.** 1 GbE carried 117 MB/s; a hop was ~3-5 ms of a 45-60 ms window.
- **Measure on a quiet host, or record the load.** The other CPU on PC A's host ranged from 10% to 109% between runs.
- **One split serves both phases.** The best split for long prompts is not the best for decode; different splits per
  phase would need the K/V and GDN state of the moved layers to cross the network.

## Estimates (not measured)

- **Over the internet**, decode would need about 1 Gbps and a round trip under ~17 ms per hop to stay ahead of one
  card on this rig. Prefill needs bandwidth: 40,960 B per token each way.
- **Moving prompt state instead of rows:** reading the whole prompt on the main PC and sending the worker's layers'
  K/V and GDN state would be ~7.8 KB per token, about a fifth of the residual hand-off; the main PC would then need
  every layer's experts and stream all of them for the prompt. It would also allow a different split for prefill and
  for decode.
- **A ring** (main -> B -> C -> main) would carry each chunk and window once per hop instead of twice through the
  relay, and save the relay's forwarding (~3-5 ms of a window on 1 GbE here).

## Open work

- **The default path beyond one configuration:** checked byte-identical for one single-GPU configuration ([The
  default path](#the-default-path)); an in-process multi-GPU split, other models and a Windows build against
  upstream's Windows build were not compared.
- **Running on AMD and Intel:** the branch builds with CUDA 13.0 on Linux (GCC, sm_75 + sm_86) and Windows (MSVC
  2022, sm_86), with HIP (ROCm 7.0, gfx1100) and the SYCL port (oneAPI 2026.1.1, which also compiles the shared
  `include/strata/prefill/prefill.hpp` this feature changed); the last two in containers, on 2026-10-10. The remote
  stage ran on NVIDIA cards only. The SYCL port keeps its own copies of `generate.cpp`, `verify.cpp`,
  `expert_source.cpp` and `prefill.cpp`, so the Intel engine has no remote stage.
- **Correctness check** against the in-process split (`--layer-split K --split-device 0`) with the caches fixed: not
  done. So far the start of every benchmark answer (the 100-160 characters the benchmark keeps) was read: coherent and
  on topic in every configuration.
- **Reconnecting** without restarting the main engine (a broken link now ends the main engine at the next prompt's
  reset, and the server starts it again; read from the code, not tested); conversation checkpoints across stages.
- **A ring topology** for three or more PCs.
- **A bf16 hand-off** (half the traffic); its effect on the answers would need measuring.
- **Windows as the main PC** (only workers were run on Windows), and the clock behaviour of an in-process split on
  Windows.
- **Other quantizations** (official IQ3_S, IQ3_XXS, IQ2_XS, UD-IQ4_XS): not tested.
