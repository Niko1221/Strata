# Remote stage: design, process and measurements

How the layer split across PCs works inside, how it got to its present speed, every number measured on it, and what
is still open. Using it: [../REMOTE_STAGE.md](../REMOTE_STAGE.md). Raw numbers:
[lask3802/Strata, branch remote-stage-data](https://github.com/lask3802/Strata/blob/remote-stage-data/bench/results/2026-10-10-remote-stage/README.md). Code:
`include/strata/net/stage_link.hpp` (the protocol), `src/net/stage_link.cpp` (the sockets, POSIX and Winsock), the
`remote_main` / `stage_worker` / `stage_relay` parts of `src/program/generate.cpp`, `Verifier::set_remote` /
`set_no_head` (`src/core/verify.cpp`), `Prefill::remote_send` / `remote_recv` (`src/prefill/prefill.cpp`),
`ArenaExpertSource::set_layer_range` (`src/core/expert_source.cpp`), and `tools/stage_node.py`,
`tools/stage_tune.py`, `tools/stage_ship.py` (tests: `tools/test_stage_node.py`, and `src/net/stage_link_test.cpp`
for the link over loopback).

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
`a b c`, payload `bytes`) and a payload. Protocol version 5 (4 before images: the Positions message and the hello's
`vision`; 3 before the hello carried the tail-skip rank).

| message | header fields | payload | reply |
|---|---|---|---|
| Hello | | `StageHello` (below) | HelloOk with the worker's hello, or Error |
| Run (a verify window) | a = T, b = first position | int32 tokens[T], fp32 rows[T x 12,804] | RunOk: a = the worker's time (us), rows[T x 12,804] |
| Commit | a = accepted count | none | none (an error comes back on the next reply) |
| Prefill (a prompt chunk) | a = T, b = first position, c = flags (bit 0: the prompt is one chunk) and `skip` << 8 | int64 tokens[T], fp32 rows[T x 10,240] | PrefillOk: a = the worker's time (us), rows[(T - skip) x 10,240] |
| Reset (a fresh prompt) | | none | ResetOk |
| Positions (`--vision`: the next request's) | a = cells; b = 1: plain positions (no table) | int32 (t, h, w)[cells] | none (an error comes back on the next reply, a Reset's included) |
| Error | | the message | |

**The hello** carries what both sides must agree on, and each side checks the other's: the protocol, `n_embd`, `hc`,
`n_layers`, `n_expert`, the first remote layer (K), the floats per window row, the K/V element type, and a
fingerprint of the pack (FNV-1a over the pack's `index.txt` and `native_experts.txt`). The worker also refuses a main
process whose prompt chunk, context or window is larger than its own, and a wrong `STRATA_STAGE_TOKEN` (the token is
the hello's last field; the worker's own hello carries none). The main process's `STRATA_ROUTE_TAIL_SKIP` rank rides
along and the worker takes it (a relay passes it on), so every layer skips as in one process. So does whether the
main process reads images (`vision`): a worker started without `--vision` refuses such a main process, and the main
process checks the worker's hello for it too. The hello is 192 bytes in
every protocol, so an older peer is refused for its protocol number. A peer that sends no hello within 10 s is dropped; after
the hello the worker waits for messages without a time limit (the next prompt may be hours away), while sends and the
main process's receives time out after `STRATA_REMOTE_TIMEOUT_S` (300 s). Sockets: `TCP_NODELAY`, 8 MiB buffers,
keepalive after 30 s idle, 3 probes 10 s apart.

**Recovery.** The worker serves one connection at a time. A new one stays in the listen backlog, unread, until the
served one has had no message read, being read, or a reply going out for `STRATA_REMOTE_TAKEOVER_S` (30 s; 0:
never) - the time counts from the last of those, so a relay's reply that waited long on its next worker starts it
anew - and only then does the worker watch the listen socket, accept the new one and read its hello on a thread
of its own (a shorter pause is no sign the main process went away: inside one request it can leave the worker
waiting while it runs its own layers of a prompt chunk). A hello it accepts (the token,
the hello's checks against the hello it gave the served main process, the worker's own checks) takes the worker over:
the old connection ends as a closed one does (`disconnected`: the positions back to the identity, a relay's next link
closed), and the new one is served; the log says how long the old one had been silent. A refused hello leaves the
old connection alone; when the old connection closes, the new one is accepted at once as before. When the served
one sends again before the new one's hello is decided, the new one is closed without an error message (one line on
the worker's stderr, `connected while ... was served`): its main process sees the connection closed, a lost link,
and connects again every 2 s within `STRATA_REMOTE_CONNECT_S`, so a stale hello never takes over later. So a main
process that started again after its PC lost power, while its old connection still looks open on the worker, is
served within `STRATA_REMOTE_TAKEOVER_S`, and two live main processes pointed at one worker take turns (each takes it
over after the other's 30 s without a message; every request then starts by connecting again). After its
hello the receiving thread polls for the next message (250 ms) instead of blocking in `recv`: on Windows a socket's
`shutdown` does not wake a `recv` blocked on it, and a connection taken over is still open (the wait ended only when
Windows dropped the half-closed connection, 120 s later). A new connection's hello is read the same way, every read
polled (50 ms) within 10 s in all: with a blocking read, closing a silent one held the served main process's next
message for more than 1 s in `stage_link_test` on Windows (by the code, until the hello's 10 s read timeout, and a
peer sending a byte at a time could hold it longer). The main process
connects at its start, and again when a
request's positions or reset finds the link lost (closed, reset, timed out, a send or receive failed, the name did
not resolve: `stage_link_lost`), with the same hello; nothing else is per connection on either side (the reset zeroes the worker's
session, the hello carries the tail-skip rank, `--vision` and the chunk), and the request's positions are sent again.
While the name does not resolve, nothing listens or the network does not reach the worker, it tries every 2 s for
`STRATA_REMOTE_CONNECT_S` (180 s), each try with its own watchdog allowance; a refusal is not tried again. That is
the whole wait: each try's connect and its hello's reply are polled against what is left of it (a hello waiting in a
worker's backlog for its reply used to add up to `STRATA_REMOTE_TIMEOUT_S`), and within a request a STOP ends it
([The recovery's open points](#the-recoverys-open-points-2026-10-13)). Within a request a helper thread prints `INFO remote_stage_wait=S` (the
seconds so far) on stdout every 10 s for the whole wait, including a try that blocks, and is joined before the
connect returns: the server's frozen-engine check (`ENGINE_STALL_S`, 90 s without a line, CPU or I/O) would end the
engine first, and the request would get "frozen" instead of the link's error. Not at the engine's start: the server
then waits for READY and reads INFO lines as settings. A worker
down for longer is still found: the engine ends with the error, and the server's next request starts it again,
which tries anew. A link lost in the middle of a request still ends the main engine; the server answers an
`ERR remote stage: ...`, and the worker's own `ERR remote stage (worker): ...`, with 503 `server_error` and
`Retry-After: 30` (`RemoteStageLost` in `serve/server.py`; an error event once a stream has started), not a 400. A
Commit's error that the next Reset drops is written on the worker's stderr first. `stage_link_test` covers the
take-over over loopback with `STRATA_REMOTE_TAKEOVER_S=1` (a second main not served while the first sends a message
every 0.5 s, served once the first is silent past 1 s, served at once when the first closes; a wrong token or
another model refused while the old one stays; a peer accepted while the first was idle with 8 bytes of its hello
sent, closed as soon as the first sends again, the first answered at once, and the rest of that hello sent then not
taking over at the first's next idle second; a chunk's reply held 1.5 s by a relay's handler, then
0.4 s until the first's next message: not taken over), with `STRATA_REMOTE_TAKEOVER_S=0` (a second main not served
while the first is silent for 2.5 s), and which errors count as a lost link. The main process's reconnect, its
`INFO remote_stage_wait` lines and the agent's restarts were run on GPUs later, by injecting faults
([The recovery's open points](#the-recoverys-open-points-2026-10-13)). The node agent (`tools/stage_node.py`) starts a
worker that ended by itself again (5 s, doubling to 300 s; node.json `"restart": false` turns it off), keeping the
log of its last run that listened as `stage-worker-PORT.prev.log` (a restart that fails before it listens
overwrites only the current log); a refused `/start` leaves its port's worker and supervision as they were.

The engine's stall watchdog (issue #29, `STRATA_WATCHDOG_S`) watches
each message a worker serves (a window, a prompt chunk, a commit, a reset, a position table) and is idle between
them, so a worker stuck on its GPU ends with its stall report instead of holding every later main process at the
hello. A relay's waits for its next worker (a connect, its taking a prompt chunk, a window's or a reset's reply) get an allowance of
`STRATA_REMOTE_TIMEOUT_S` + 150 s instead: the link's timeouts end them, and a stuck next worker ends alone, with its
own report. `STRATA_REMOTE_STALL_AT=N` (a test hook) blocks the worker's Nth window. Checked on CUDA (the 3080 + 2080 Ti
pair, `STRATA_REMOTE_STALL_AT=60 STRATA_WATCHDOG_S=20` on the worker): the worker wrote its stall report 20 s into the
blocked window and exited; the request in flight failed 25-26 s after it was sent (`remote stage: the peer closed the
connection`), and the next request failed after 66-68 s (two runs), when the server's three starts had each ended with
`cannot connect to HOST:PORT: Connection refused`, since nothing started a worker again then (its agent does now:
"Recovery" above; run on GPUs after a `kill -9`, not after a stall report).

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
- a position table is 12 B a cell, for the cells a request can reach (its prompt, its answer and 64 more): about 0.36 MB
  for a 30,000-token request, sent once before it, and only when the request has a picture (or follows one).

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
one conversation. Prompt chunks do overlap across stages, so the prompt rate is set by the slowest stage. (Opt-in,
`--pipeline-windows 2` on the main process lets its layers run the next window, on a guess, while the worker and the
head run this one: [REMOTE_STAGE.md](../REMOTE_STAGE.md#overlapping-the-decode-windows---pipeline-windows-2).)

### More than two stages (relay workers)

A relay worker (`--stage-end K2 --stage-next HOST:PORT`) runs [K, K2) and is a client of the next worker as the main
process is its client: windows go on synchronously and the next worker's rows are this worker's reply; prompt chunks
go on through a `StageRelay` (one send in flight, in order), and the next worker's reply to a chunk is read and
forwarded on the replying thread (`StageHandlers::prefill_reply` defers the answer until it arrives). A reset and a
commit go on to the next worker first; a reset waits for a chunk reply still going out. The main process sees one
worker. A relay connects to the next worker when it starts (30 tries, 2 s apart; a refused hello - a wrong token or
a context mismatch - is not retried) and again at the first reset after each main process's hello (so the next worker
hears that main process's values, such as its tail-skip rank) or after that link broke; when its main process goes
away it drops the link to the next worker, so replies still owed on it cannot reach the next main process.

### Images

With `--vision`, a picture becomes image tokens whose input rows the encoder's embeddings replace at layer 0, on the
main PC; nothing about that crosses the link (the hand-off is the residual, whatever the input was). What every layer
needs is the position table: each rope kernel reads a cell's (t, h, w) from a device table
(`strata/kernels/mrope.hpp`). Text is the identity; a picture's cells sit on a grid at one t, and every cell after
the picture moves on by the picture's extent (the larger of its patch rows and columns), not by its token count. A
worker that kept the identity would rotate its layers' queries and keys differently from the main process's layers
for every cell from the picture on. The main process therefore sends its table before each request that has a
picture (`StageMsg::Positions`: the cells the request can reach, its prompt and answer plus 64), and "plain positions"
(no table) before the first text request after one; text after text sends nothing.

The worker keeps the table in the same device buffer for the whole process, because the rope kernels in its captured
CUDA graphs hold that pointer: it waits for the last request's commit and the device, then writes the buffer in place
(`stage_positions_fill`: the cells past the table sent continue from its last one, as the main process's own table
does past its prompt). The table outlives a connection while the next main process starts from the identity and
sends nothing for text, so a worker puts a picture's table back to the identity when its main process goes away. A
relay forwards the table to the next worker first, making its link to the next worker if a new main process's hello
dropped it, and a relay started with `--vision` checks at its start that the next worker has it too. Positions is one
way, like a commit, and its failure (or a malformed table) comes back as the next reply's error. A Reset drops a
pending commit error (it belongs to the request before; the worker writes it on its stderr), but not a Positions error, because the positions belong to
the request the Reset begins, which must not run on the old table. A later table replaces an earlier one's failure:
the main process sent no Reset for a request it then refused (a prompt longer than the context, say).
`STRATA_STAGE_POSITIONS=0` on the main process (a test switch) sends no table, to show what the positions change.
`src/net/stage_link_test.cpp` checks the table's fill, the hello, the table arriving whole, and the errors' order
over loopback.

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
  rate for one long prompt: 2.6 GB each way in the ~38 s it crossed, at up to 123 MB/s ([data/nic-pc-b-1gbe-3pc.txt](https://github.com/lask3802/Strata/blob/remote-stage-data/bench/results/2026-10-10-remote-stage/data/nic-pc-b-1gbe-3pc.txt)). 10 GbE between PCs A and B took most of it away (100K prompt:
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

The tuner rows divide each stage's chunk time in [the first tuner run's report](https://github.com/lask3802/Strata/blob/remote-stage-data/bench/results/2026-10-10-remote-stage/data/tuner-rvn-2pc-report.md)
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
B at most 24. Its whole report: [data/tuner-q4xl-3pc-report.md](https://github.com/lask3802/Strata/blob/remote-stage-data/bench/results/2026-10-10-remote-stage/data/tuner-q4xl-3pc-report.md).

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
and by upstream main's build (the merge base then, fb58e0d, engine 0.1.41; the same build options) once, between them; greedy answers (temperature
0, top-k 1, at most 200 tokens, reasoning off) to five prompts: code, prose, Chinese, arithmetic, and a 9,364-token
document (two prompt chunks). The three result files are byte-identical (the same MD5): text, token counts and finish
reasons. The script and the files: [default_path_check.py](https://github.com/lask3802/Strata/blob/remote-stage-data/bench/results/2026-10-10-remote-stage/default_path_check.py),
[data/default-path-*.json](https://github.com/lask3802/Strata/tree/remote-stage-data/bench/results/2026-10-10-remote-stage/data).

### After the rebase onto 0.1.42 (2026-10-11)

The branch was rebased onto upstream main 61b3fb5 (engine 0.1.42, 235 commits later). Two textual conflicts: the draft
layer's batched prompt K/V, now behind `draft_kv::plan` and still never taken by a remote main; and the #1237 line,
which 0.1.42 added to the refill code this branch moves out of the request loop. Two follow-ups to 0.1.42's changes:
the main process's tail-skip rank in the hello (protocol 4), and the CPU prefill share's check that every expert is in
the GPU cache, which counts the process's own layers. Checked on PC A and PC B (RVN IQ3_S):

- **The default path.** 0.1.42's stock one-GPU config does not repeat from run to run. The PCIe share is refined from
  the CPU pool's measured speed in the first decode windows. `default_path_check.py` gave two runs of this branch's
  build different answers on 2 of the 5 prompts (the arithmetic one and the 9,364-token one); the other three were
  the same in all three runs, upstream main's included. So the check ran at fixed expert sets instead
  (`--pcie-frac 0 --adapt-every 0 --adapt-swaps 0 --mtp-max-t 1 --suffix-draft 0`,
  `STRATA_IQ_MT_MIN=1 STRATA_PREFILL_CPU_SHARE=0`): a chat replay of two chats of four turns on a 16,000-token
  system prompt, with the conversation cache on and 160-token greedy answers. Each build ran twice: upstream main's,
  this branch's and the checkpoint branch's (#1887). The second round's three runs were byte-identical, 8 of 8 turns.
  Upstream main's first run was the outlier: it agreed with every other run, its own second run included, on at most
  1 of 8 turns. The other five runs agreed with one another on 5-8 of 8, and each differing pair was the same until
  one answer diverged mid-way. 0.1.42 does not repeat from run to run even at these flags.
- **The split** (A + B over 10 GbE, K=26, the same fixed expert sets on both PCs): the 16,000-token replay ran through,
  and three runs agreed pairwise on 5, 5 and 8 of 8 turns (the same until one answer diverged mid-way, as above). With the served flags
  and thinking on, the first token came after 14.6-15.1 s on every turn (every prompt is read from 0), and decode
  averaged 61.5 tok/s. Earlier the same day, in another session on the 0.1.41 base, it was 54.4 tok/s; the two are
  not a like-for-like comparison.
- **The tail-skip rank:** with `STRATA_ROUTE_TAIL_SKIP=0` on the main process only, the worker (default 7) logged
  `STRATA_ROUTE_TAIL_SKIP=0, the main process's (this worker's own was 7)`.
- **Builds:** CUDA 13.0 on Linux (sm_75 + sm_86) and Windows (MSVC 2022); HIP (ROCm 7.0, gfx1100); the SYCL port
  (oneAPI 2026.1.1).

The measurements above this section were taken on the 0.1.41 base. 0.1.42 turned `STRATA_ROUTE_TAIL_SKIP` on by
default for CUDA serve. Upstream measured 1.13-1.16x decode with it on single cards that miss experts, and nothing on a
split that holds nearly every expert (docs/DETAILS.md). The tables were not run again.

### After the rebase onto 0.1.43 (2026-10-12)

The branch was rebased onto upstream main 7695859a (engine 0.1.43, 72 commits after 61b3fb5) without a textual
conflict. Three follow-ups to 0.1.43's decode changes:
- A worker can let the adaptive tier's copies land while its next window runs (`STRATA_ADAPT_OVERLAP`), as the main
  process's decode loop does since 0.1.43. On a worker it is opt-in (`STRATA_ADAPT_OVERLAP=1`), because it was not
  measured faster there (below).
- The tail-skip default's "90% of the experts cached" counts only the process's own layers.
- A remote main keeps the probed PCIe share under `STRATA_PCIE_AUTO_SPLIT=1`.

Then images ([Images](#images)). Checked on PC A and PC B (RVN IQ3_S, 10 GbE, K=26):

- **The default path.** The 0.1.42 check ran again: fixed expert sets, the 16,000-token chat replay, 8 turns. This
  branch's build and upstream 0.1.43's ran in the order: branch, 0.1.43, 0.1.43, branch. All four pairs were
  byte-identical, 8 of 8 turns: each build's two runs, and the two builds in each half. No run was an outlier this
  time.
- **Images: the position table.** Six picture prompts and one text prompt after them, at the fixed expert sets and
  `STRATA_ROUTE_TAIL_SKIP=0` on every process. The split's first-token distribution was compared with the same model
  on one card (KL divergence):
  - Text: 1.6e-4, with or without the table.
  - The pictures: the geometric mean is 43% lower with the table than with `STRATA_STAGE_POSITIONS=0`.
    - It is better on 3 of the 6, the table picture the most (7.4e-4 against 1.7e-2).
    - It is slightly worse on the other 3. Why was not measured; a guess is that layers 26-47 react little to the
      positions, so other differences between the split and one card show through.
  - With the tail skip on (the default), the skip's own choices outweigh the table: an arithmetic mean of 0.146 with
    it, 0.149 without.
- **Images: the answers.** With the served flags they were right: the shapes, a table's largest count and its CSV, a
  chart, a screenshot, and the text after the pictures.
- **Images: a main restart.** A main process restarted after a picture got the same text reply from the worker as
  before the picture. Before the worker put its table back to the identity at a disconnect, it did not.
- **The overlap on the worker.** It was on in the runs above and in the speed runs below (the build then followed
  upstream's default on the worker too):
  - The 16,000-token replay: 133-137 windows took swaps still landing, and the worker's layers waited 0.03 ms in all.
    The main process's layers waited 1.4-1.8 s over about 150 windows.
  - A 40-prompt run: 583-600 windows, 23-30 ms of waiting.

  Whether it changes answers could not be told byte for byte: with the adaptive tier on, two runs of the same
  setting agreed on only 3 and 4 of 8 turns.
- **Speed** (the served flags, 40 prompts x 200 tokens, speculative decoding with the draft layer, two runs each):
  | Build | Decode tok/s |
  |---|---|
  | This branch on 0.1.43 | 62.7, 63.2 |
  | The same, with `STRATA_ADAPT_OVERLAP=0` on both PCs | 63.2, 63.8 |
  | This branch on 0.1.42 | 63.5, 64.0 |

  The overlap's runs were the slowest: about 0.9% below the runs without it and 1.3% below 0.1.42. Two runs of one
  setting differed by 0.8-0.95%, so two runs each cannot tell whether that difference is real.
- **The worker's overlap alone.** The same flags and prompts; the main process's overlap on in every run, the
  worker's on and off in 6 interleaved pairs (PC B's 10 GbE card had been replaced since the runs above):
  | | The worker's overlap on | Off |
  |---|---|---|
  | Decode, median of 6 runs | 64.19 tok/s | 64.33 tok/s |
  | The worker's own time per window, median | 15.85 ms | 15.85 ms |

  With it on, decode was faster in 2 of 6 pairs and the worker's time per window shorter in 3 of 6; two runs of one
  setting differed by 0.39% (median), up to 1.24%. Its layers waited 26-42 ms per run in all: the worker's copies
  land while the main process runs its own layers, so there is little left to hide. Not faster, so a worker turns it
  on only with `STRATA_ADAPT_OVERLAP=1`. The main process keeps upstream's default (on for Linux CUDA), as on one
  card: upstream's overlap needs a process whose layers are on one GPU (a layer split across GPUs in one process
  leaves it off), and each PC of a remote stage is one such process.
- **The link test.** `stage_link_test` (the images change's 22 checks) passes on Linux (GCC 13) and on Windows
  (MSVC 2022).
- **Builds.** CUDA 13.0 on Linux (sm_75 + sm_86) and on Windows (MSVC 2022, sm_86); HIP (ROCm 7.0, gfx1100) and
  the SYCL port (oneAPI 2026.1.1), both in containers.

### After the rebase onto 0.1.43.1 (2026-10-13)

The branch was rebased onto upstream main a0116b27 (engine 0.1.43.1, 33 commits after 7695859a). One textual
conflict: #2090 changed the prompt path's loan (`lend`) at its old place in `main()`, which this branch had moved up
so a worker can use it; the moved copy took the change, and it is the same as upstream's. Three follow-ups:
- #2090 halves a prompt chunk whose loan does not fit the cache, instead of ending the engine. A worker reads the main
  process's chunk whole (its rows go back as one reply, and a relay forwards it as one chunk), so a worker whose loan
  comes out smaller than the chunk it was sent answers an error that names the `--prefill` that fits (now only when
  it cannot lend for its own chunk either: [below](#the-recoverys-open-points-2026-10-13)). Before 0.1.43.1 that
  worker failed the loan's relayout with an error too; halved without the check, it would have answered wrong rows,
  and a relay would have lost step with its next worker.
- 0.1.43.1 made `STRATA_ADAPT_OVERLAP` opt-in on every backend (#2010). The worker's own opt-in clause is gone: the
  same switch now holds on both PCs, and neither overlaps by default. In the runs above the main process's overlap
  was on.
- The server's 503 for a remote stage's error sends its Retry-After through `_json`'s headers, which upstream added
  too; the merge had sent the field twice, and the server's test now checks it is sent once.

### After the rebase onto 0.2.0-beta (2026-10-20)

The branch was rebased onto upstream main e88ee39d (0.2.0-beta, engine 0.1.45.3; 1335 commits after a0116b27). The
textual conflicts, and what each side kept:
- **The expert arena** (`ArenaExpertSource`): pin by need (`STRATA_PIN_BY_NEED=1`, opt-in, 0.1.44) lays the arena
  out with a page-aligned blob stride, and this branch cuts a worker's arena to its layer range. A layer range keeps
  the packed layout (its `base_` is shifted back by the range's first byte, which a stride would not match), so pin by
  need is off with a remote stage: the log says so, and the prefix pinning stays. `load_experts_gguf` takes both the
  padded map and the range; `blob_at` answers null outside the range, as `blob` does.
- **The prompt path's loan** (`lend` and `refill`, which this branch moves up so a worker can use them): upstream
  changed them at their old place (`--prefill-head-alias`, `STRATA_REFILL_PARTS`, the `exact` chunk of a planned
  segment and `seg_lend`, and 0.2's host-aware chunk, `planned_chunk` with `pf_plan_chunk`, #2824). The moved copy
  is upstream's, whitespace aside. The worker calls it with `exact` 0, which leaves its chunk as before, and the
  worker's `lend_whole` (the hello's test of whether `lend` would lend the main process's chunk whole) takes the
  chunk through `planned_chunk` as `lend` does.
- **`Verifier::run`** calls `remote_run_` first, then upstream's prelaunch check of the overlapped split. A remote
  stage is a split on one GPU (`--split-device 0`), which `STRATA_SPLIT_OVERLAP` never prelaunches.
- **The prompt hand-off**: `STRATA_HANDOFF_PIPE`'s pieces are made for an in-process next stage only, so a remote
  send always reads a hand-off the host has waited for; the send's own wait comes before the pieces' code.
- The tail-skip default's 90% rule (upstream adds an NVFP4 clause next to it), the drafter's batched K/V plan
  (`mtp_first`), `STRATA_PRIMARY_DEVICE`'s refusal of a layer split (after the remote stage's checks), the
  pinned-memory reserve around the arena's `open` (`pin_reserve_take`), the hand-off's row count for pipelined slot
  groups, the overlapped split's line and the job thread's place in `verify.cpp`: both sides kept.

In the first commit (so every commit of the branch builds where upstream does):
- The Metal build compiles `generate.cpp`, so it links the link's code (and `send` asks for no SIGPIPE through
  `SO_NOSIGPIPE` where there is no `MSG_NOSIGNAL`). Not built here (no Mac).
- A worker's arena read could wait for most or all of its registration slices under `STRATA_DEFERRED_REGISTER=1`
  (Windows, opt-in): the readers counted the slices from layer 0, the arena from the range's first layer. They count
  from the range's first layer now. Read from the code, not run.

Follow-ups (one commit), each a check of this branch's own for what upstream added after 0.1.43.1:
- `--prefill-head-alias` is refused with a remote stage: a worker has no head, and the alias needs one GPU.
- A Qwen3.6-35B-A3B file is refused with the remote stage's flags (`--flag value` or `--flag=value`). Upstream runs
  that family from the same binary on its own engine path, which has no layer split and skips an option it does not
  know in serve mode: a main process would have run every layer itself, and a worker would never have listened.
- A remote stage is refused on the Metal build, since the Metal window runs init, run and commit itself, past the
  remote stage's hooks.
- `--adapt-async 1` is turned off with a line on a worker, in any mode (it runs the blocking adaptive tier between
  its windows), and on a remote main in the expert arena's mode (the arena's copy-in tier has not been measured with
  a remote stage).
- The automatic prompt chunk's 8192 cap of a layer split on gfx12 applies to a remote main too: it reads its chunks
  through the worker as the same pipeline. Without it, a gfx12 main on `--prefill auto` took 16384 chunks, and a
  worker on `auto` that is not gfx12 (8192) refused its hello. Not run: no AMD card here.

Read in the code, no change needed:
- The server's 503 for a remote stage's error sends its Retry-After through `_json`'s `retry_after` argument, which
  upstream added; the field is sent once, as the server's test checks.
- 0.2's host-aware prompt plan (#2824) and `STRATA_RAM_ADAPT2` act on the mapped file tier only (`srcp == &src`); a
  remote stage's PCs run the expert arena, so neither changes their chunks or their RAM tier.
- 0.2's sm_86 / sm_60 defaults set `STRATA_ADAPT_NOWAIT=1` and `STRATA_ADAPT_LAG=2` on a PC whose cards are all that
  architecture. They are each PC's own: a worker's window handler takes the swaps as the main process's decode loop
  does with them (`adapt_nowait()`), and the adaptive tier does not change the answers.

None of this has run on a GPU since the rebase: the builds and the CPU tests are listed in the pull request.

### The recovery's open points (2026-10-13)

Faults injected on A + B (RTX 3080 main, RTX 2080 Ti worker, 10 GbE, RVN IQ3_S, K=26) with the branch on 0.1.43
(371b400f), its agent and server, each followed by requests; none needed a hand:

| Fault | That request | The next request |
|---|---|---|
| the worker killed (`kill -9`) while idle | waited for the agent's restart (5 s, then 40 s to load), answered after 45.6 s | - |
| the worker killed in the middle of an answer | 503, `Retry-After: 30`, "remote stage: the peer closed the connection" (2.9 s) | answered after 50.2 s |
| the same, streamed | an error event of the `server_error` kind | answered after 44.8 s |
| the worker stopped and started through its agent | - | answered in 1.6 s |
| the main engine killed while idle | the server started it again, it took the worker over: answered after 20.5 s | 1.2 s |
| the main engine killed in the middle of an answer | the server's 503 for an engine that stopped | answered after 13.4 s |
| the worker stopped while a request waited, started again 70 s later | answered after 109.6 s, past the 90 s frozen-engine check | - |

Four cases were left waiting too long or failing; each now ends in no error or a clear one within a bound:
- **A reconnect's whole wait is `STRATA_REMOTE_CONNECT_S`.** `StageClient::connect` takes a deadline and a stop flag.
  The connect is non-blocking (`select` on Windows, where `WSAPoll` misses a failed connect before Windows 10 2004;
  `poll` elsewhere; then `SO_ERROR`), and the hello's reply is polled; a name's addresses but the last get 10 s each
  at most. Before, a try could block for the system's SYN retries, and a worker that took the connection into its
  backlog without answering (it served another main process) held the hello for `STRATA_REMOTE_TIMEOUT_S` (300 s)
  on top of the 180 s.
- **A STOP ends that wait.** The server sends STOP when the client goes away. The wait ends with
  `ERR remote stage: the wait was cancelled`, which `stage_link_lost` does not count, or with the lost link of a try
  the STOP came in during and could not cut short (the name's lookup, a connect refused at once); either way the
  engine goes on to the next request, and the link stays closed, so that request connects again. Before, the engine went on waiting for the
  worker until its window ran out, and then ended.
- **A worker lends for its own chunk** when it cannot lend the buffers for the main process's chunk whole: a smaller
  chunk can need more slots than a larger one (#2090), and a multi-GPU worker reads short requests in its
  `STRATA_SPLIT_SMALL_OWN` chunks (#340). `lend()` would then lend for fewer tokens than the main process sends, and
  the worker answered every long prompt with an error. The hello decides it (`lend_whole`, `lend()`'s own test) and
  the worker says so; the error stays for a worker that can lend neither.
- **A relay looks at its next worker while idle**, about once a second while its main process sends nothing
  (`StageHandlers::idle`, `StageClient::alive`: a poll and a one-byte `MSG_PEEK`), and when that link is gone it
  closes its main process's connection. Before, the main process's next request found its own link fine, its first
  message failed at the relay, and the main engine ended.

The first two on the same pair, the main process with `STRATA_REMOTE_CONNECT_S=60`. "This branch" is these fixes'
first build (their review's changes, run once more on the same faults, gave 58.2 s and 60.0 s again); "before" is
the build of both PRs on 0.1.43.1 before them (0936f027, the checkpoints PR on top: its request start already checks
the link with `alive()`, and it waits for the worker as this branch did):

| Fault | This branch | Before |
|---|---|---|
| the worker stopped; a streamed request whose client leaves after 20 s, then at once a second request | the engine ended the first at the STOP and waited for the second: 503, `Retry-After: 30` after 58.2 s (its own 60 s), `cannot connect` | the engine waited out the first one's 60 s and ended (exit code 1): the second got the server's 503 for an engine that stopped, after 37.9 s |
| the worker started again, a request | answered after 15.5 s (the second's failed reconnect had ended the engine, as before) | answered after 12.8 s |
| the worker's port taken by a listener that never answers, a request | 503, `Retry-After: 30` after 60.0 s, `receive failed: timed out (no reply to the hello)` | 503, `Retry-After: 30` after 310.5 s (`STRATA_REMOTE_TIMEOUT_S` for the one hello) |
| the listener gone, the worker started again, a request | answered after 12.7 s | answered after 12.7 s |

Not run on GPUs: a worker's own chunk (with the agent's explicit `--prefill` on one card the loan grows with the
chunk, so the case did not arise on this pair) and a relay's look (it needs three PCs; `stage_link_test` covers it
over loopback, with a stand-in for the relay's hook).

### A reader's report: what a peer can make a process do (2026-10-13)

A reader of the link's code (building a SYCL worker on this branch) found that a worker's receiving thread parsed any
message before the hello and multiplied a window's T by its row width before bounding T. A `Run` header with
T = 2^62 makes `T * handoff_floats` and `T * 4` wrap to 0 (both are multiples of 4), so a header claiming 0 bytes
passed the size check, and the token vector's resize threw on the receiving thread, which ends the process. Any host
that can reach the port could send it: no hello, no token. A `Prefill` header the same. Fixed:

- **A worker reads nothing but a hello first.** The first message is read as a waiting connection's hello is: whole
  within 10 s in all, and nothing else (before, each recv had its own 10 s, so a peer that sent a byte every few
  seconds held the worker for over half an hour). Another first message, or a hello of a wrong size, is answered
  (`hello first`, `bad hello`) and its connection closed, its body neither waited for nor drained.
- **After a hello, nothing is read until its token is taken** (the first hello, or another a served main process
  sends). A refused peer's next header, with a body that never comes, held the receiving thread in a recv; shutting
  the socket down does not wake that on Windows, and the worker waited for the thread for good. (A review of this
  fix found it; on Linux the shutdown wakes the recv.)
- **A window's and a chunk's T are bounded before any product with them** (T at most the buffer's floats over a
  row's), so a served main process that sends such a header gets the malformed-message answer, and the worker serves
  on.
- **The main process takes no size on trust either:** an error reply over 64 KiB is not read (at the hello an
  unexpected reply; later a link out of step, which connecting again mends), and a hello's reply is read whole
  within the connect's deadline and stop flag (before, only its first byte was: a peer that sent part of it held the
  main process for `STRATA_REMOTE_TIMEOUT_S`).
- Smaller ones from the same report: `--stage-begin` at or past the layer count stops at start (the report saw it
  reach the session's setup with a layer out of range); a worker started with `--mtp` stops at start saying the
  draft layer runs in the main process (it stopped later, at the draft layer's setup); `StageClient`'s failed-link
  flag is an atomic (written by the sending side, read by the receiving one under another lock).

`stage_link_test` sends these headers by hand over loopback (and a hello a byte a second), and each fix undone fails
it; the wait for the token only on Windows, where the recv it prevents is not woken.

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

- **The default path beyond one configuration:** checked byte-identical for one single-GPU configuration on 0.1.41
  ([The default path](#the-default-path)) and at fixed expert sets on 0.1.42 and 0.1.43
  ([0.1.42](#after-the-rebase-onto-0142-2026-10-11), [0.1.43](#after-the-rebase-onto-0143-2026-10-12)); an in-process
  multi-GPU split, other models and a Windows build against upstream's Windows build were not compared.
- **Images beyond two PCs:** a relay worker forwards the position table (covered by `stage_link_test` over loopback),
  but no picture went through three PCs; the image checks used RVN IQ3_S only.
- **Running on AMD and Intel:** the branch builds with CUDA 13.0 on Linux (GCC, sm_75 + sm_86) and Windows (MSVC
  2022, sm_86), with HIP (ROCm 7.0, gfx1100) and the SYCL port (oneAPI 2026.1.1, which also compiles the shared
  `include/strata/prefill/prefill.hpp` this feature changed); the last two in containers, on 2026-10-10 and again
  after the rebase onto 0.1.43. The remote stage ran on NVIDIA cards only. The SYCL port keeps its own copies of `generate.cpp`, `verify.cpp`,
  `expert_source.cpp` and `prefill.cpp`, so the Intel engine has no remote stage.
- **Correctness check** against the in-process split (`--layer-split K --split-device 0`) with the caches fixed: not
  done. So far the start of every benchmark answer (the 100-160 characters the benchmark keeps) was read: coherent and
  on topic in every configuration.
- **Reconnecting** in the middle of a request (a link lost between requests is made again at the next request's
  start; one lost mid-request ends the main engine, and the server starts it again). Faults run on GPUs: two PCs,
  Linux on both ([The recovery's open points](#the-recoverys-open-points-2026-10-13)); a relay's next worker, a
  network partition and a Windows PC on either side were not; conversation checkpoints across stages.
- **A ring topology** for three or more PCs.
- **A bf16 hand-off** (half the traffic); its effect on the answers would need measuring.
- **Windows as the main PC** (only workers were run on Windows), and the clock behaviour of an in-process split on
  Windows.
- **Other quantizations** (official IQ3_S, IQ3_XXS, IQ2_XS, UD-IQ4_XS): not tested.
