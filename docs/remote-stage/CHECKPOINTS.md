# Conversation checkpoints across a remote stage

Status: built, unit-tested and checked on two GPUs over 10 GbE ([GPU results](#gpu-results)). Opt-in:
checkpoints stay off without `--prompt-cache N` on the main process. The design comes first; [As built](#as-built)
says where it lives and the small deviations.

## The problem

With `--remote-stage`, every prompt is read from token 0: `generate.cpp` sets `prompt_cache`, `prompt_cache_every`
and `conversation_cache_mib` to 0 because a checkpoint holds only the running state of the process's own layers. A
chat client sends the whole conversation every turn, so every turn of a long conversation waits for the whole prompt
to be read again before its first token. On one GPU the prompt cache reads only the new part: measured on an RTX 3080
alone, `prompt 14468 tokens = 14453 reused + 15 read in 383 ms`, and a replay of 2 chats x 4 turns on a 16K-token
system prompt (first turn 15.5 s to the first token, every later turn 0.45-1.84 s).

## What a checkpoint is (unchanged)

`ConversationCheckpoint` (`include/strata/core/conversation_cache.hpp`) holds RUNNING state only: per GDN layer the
recurrence and conv history (~3.27 MB a layer), per QSA layer the indexer tail / dead / block_pos, the PLE history,
and the token ids. Positional state (QSA K/V, pooled indexer keys, the drafter's K/V) is not copied: it stays in place
and a checkpoint is valid only while the cells below it still hold its tokens. That is why the chain `checks` is
always one branch: at request start every checkpoint that is not a prefix of the new prompt is erased. Restore copies
the running state back and rewrites the pooled-index row at `L / idx_block` from the saved dead marker; nothing is
truncated (the next read overwrites from `resume`). The worker's streamed K/V needs nothing either: the host pool is
authoritative and the resident window's CLOCK state is residency, not position.

So the worker's half of a checkpoint is its own layers' running state, and it stays on the worker. Nothing big crosses
the link: the main sends "save" and "restore" with an id.

The drafter's K/V (main only) is built only near each prompt's end (`set_remote_rows_from`, `MtpDrafter::prefill`
skips earlier groups); a restore to a shorter branch may leave the drafter cells near the new end unbuilt. That costs
draft acceptance, not correctness (every draft is verified), and is the same on one GPU. Not changed here.

## Opt-in

A remote stage keeps its behaviour (no checkpoints, every prompt from 0) unless `--prompt-cache N` is given
EXPLICITLY on the main's command line (an `o.prompt_cache_explicit` flag set by the parser; a config that passes
`--prompt-cache 6`, the one-GPU default, turns it on too). Without it the messages stay the same (a Reset is the
header alone, as before) and the cut loop does not split the prompt. The hello says protocol 6, so the main and every
worker must run this version, with or without checkpoints (they always had to run the same protocol).

## Scope

In:

- `--prompt-cache N` (the checkpoint chain) with a remote stage, at the SYNCHRONOUS cut points only: `turn_at`,
  `root_at`, `message_at` (opt-in env) and `pin_at`. They come right after `read_part` / `read_windows` return for
  the part ending there. STRATA_CKPT_REREAD (a test switch that reads `[0, L)` again instead of restoring) is not a
  cut point: it Resets the worker keeping the chain (below).
- `from_live` (the next request extends what the session holds). The worker runs unverified draft rows in a window
  (K/V written, GDN untouched) and replays only the accepted prefix at Commit, so at a request boundary it holds the
  committed prefix, the same as the main's `live`.
- A relay worker (main -> w1 -> w2) forwards the new messages.

Out (forced off or refused in remote mode, each with a stderr line):

- Mid-prompt checkpoints (`--prompt-cache-every`, `--prompt-cache-tail`): forced 0 / off. They would need a one-way
  "save after this chunk" flag in Prefill and the main's part saved in `on_stage_chunk` (the main's `on_chunk` runs
  on the remote receive thread, past `done`). A later step if prompts that diverge mid-way matter.
- Parking (`--conversation-cache-mib`): ignored on a remote main (forced 0, stderr line), with or without
  checkpoints, as #1748 does, so a config with both still starts. A later step: the worker would keep its K/V images
  under a park id.
- Session files: refused with a remote stage (the existing guard checks `stages`/`multi_gpu`, which a remote main has
  neither of).

## Protocol (kStageProtocol 5 -> 6)

An old and a new side refuse each other at hello ("the two sides differ (protocol 5/6)"). Hello gains
`int32 ckpt_max` (the main: its prompt_cache + 1 when checkpoints are on, else 0); the worker refuses a main whose
ckpt_max exceeds its own limit (64). It takes the first 4 bytes of the informational `build` text (now 52 chars), so
the hello stays 192 bytes and an old side reads a new one whole and refuses it for the protocol, not as malformed.

| message | header | payload | reply |
|---|---|---|---|
| `Reset` (changed) | - | `int64 keep[]` | `ResetOk` |
| `CkptSave` (new, 12) | a = id, b = L | `int64 keep[]` | `CkptSaveOk` (13: a = 1 stored, 0 not stored) or `Error` |
| `CkptRestore` (new, 14) | a = id, b = L | `int64 keep[]` | `CkptRestoreOk` (15) or `Error` |

Ids are int64 from a per-process counter on the main (from 1, never reused; no wrap in practice).
`ConvCheckpoint` gains `remote_id` (0 = none; an `int32_t`, see [As built](#as-built)). `keep[]` is the set of ids the
main will hold if the operation succeeds (for CkptSave: the chain after its eviction, plus the new id).

Retention is decided by the main alone; the worker never evicts on its own:

- The worker prunes to `keep[]` only when the operation succeeds. CkptSave that cannot store (the 64-entry limit after
  pruning, or the host allocation failing) replies `CkptSaveOk a=0` and leaves the store as it was; the main then
  creates no checkpoint and evicts nothing (the chain is as before the cut). The request goes on.
- The worker clears its store on disconnect (a new main is a new id space). So after the main process connects again
  (a worker restarted, a take-over; ENGINEERING.md, "Recovery") the worker holds no part and no session: see
  [After a reconnect](#after-a-reconnect).
- Ids the main dropped without a message (the non-prefix erase at request start, a from_live request) stay on the
  worker until the next Reset / CkptSave / CkptRestore prunes them: at most one chain's worth.

Failure classes (not every save failure can be skipped):

- Not stored (admission): skip the checkpoint, go on.
- State or transport: a position mismatch, a pending deferred Commit error, a CUDA / copy failure, a relay's
  downstream error, a link broken in the middle of a request. The worker's session can no longer be trusted, so the
  main prints `ERR` and the engine exits, as it does for a failed Reset; the server starts it again, the worker sees
  a disconnect and clears its store, and the next request reads from 0. A link lost at a request's start (its
  CkptRestore included) is made again instead: [After a reconnect](#after-a-reconnect).
- CkptRestore of an id the worker does not have (should not happen; a store bug or a restarted relay): the main falls
  back to reading from 0 within the same request (below). A missing id is the only non-fatal restore error; the
  worker distinguishes it from state errors in the Error text (a fixed prefix).
- Images (`--vision`): the request's position table goes before its Reset or CkptRestore (ENGINEERING.md,
  "Images"). A failed table is that request's, so a CkptRestore reports it (a state error: the engine exits) where
  it drops a failed Commit's error, as a Reset does; a CkptSave or a window reports it too. A restored part's K/V were rotated by the table of the request
  that saved it; a request that resumes it sends the same prefix, pictures included, so its table agrees on those
  cells, as on one GPU.

Worker handling (the serving thread, strictly in order after any Commit before it):

- Position tracking: the worker keeps `pos` = the end of its last Prefill (`p0 + T`), the window's `pos0 + n_keep`
  after a Commit, 0 after Reset, L after CkptRestore. A CkptSave whose L differs from `pos` is a state error.
- `CkptSave(id, L)`: `replies_out()`; `wait_commit`; a pending deferred Commit error is reported (state error); save
  its carve's running state with an EXPLICIT length L (new overloads of `conversation_checkpoint_save/restore` taking
  the token count; the existing ones call them with `ids.size()`, unchanged); store; prune to keep.
- `CkptRestore(id, L)`: `replies_out()`; `wait_commit`; refill / apply_pending as `reset` does; missing id -> the
  missing-id Error; L differs from the stored one -> state error; restore with the explicit length (the pooled-index
  dead row at `L / idx_block`; `ple_prev` is not touched on the worker, which has no PLE); `pos = L`; prune to keep. A
  pending deferred Commit error is logged and cleared (the state is replaced).
- `Reset`: as before (it waits for `replies_out()` first), then prune to keep.
- Relay: after `replies_out()` and the relay's outstanding replies are drained (as Reset does: `relay.wait()`),
  forward synchronously and wait for the downstream reply; then the local step. Downstream error -> reply Error
  (state class). Downstream `CkptSaveOk a=0` -> do not store locally either, reply a=0.

Main side:

- `checkpoint_at(L, nullptr, ...)` in remote mode: choose the eviction victims without applying them, send
  `CkptSave(id, L, keep)`; a=1 -> save the main's part (a failure there is fatal, as on one GPU), apply the eviction,
  insert with `remote_id = id`; a=0 -> stderr line, nothing changes.
- Ordering: CkptSave follows the part's last window on the same ordered connection. The worker gets one window at a
  time: each Run waits for the worker's reply, and the window's one-way Commit goes out before the next message, so
  the worker has run that Commit when it reads the CkptSave, and a Commit that failed comes back as the CkptSave's
  error. With `--pipeline-windows` a Run waits on a job thread, so every use of the link from the request's own
  thread (CkptSave, CkptRestore, Reset, Positions, the `alive()` look at a request's start) comes after a
  `pl_drain`, which ends with no Run in flight (one that stops at an
  error first waits for a window still waiting on the worker, and gives it up without its Commit); that error is set
  aside (`pl_drain_err`): no CkptSave over it, and no live continuation after a decode it ended, so the worker's
  session is next replaced by a Reset or a CkptRestore.
- Request start, checkpoint path: after the main's restore, `CkptRestore(c.remote_id, resume, keep)`. On the
  missing-id Error: one shared helper zeroes the main's session, sends Reset (keep = {}), and sets `resume = 0`,
  `from_live = false`, `reread_to = -1`, `checks.clear()`; everything after (MTP `kv_restore`, `read_from`,
  `limit_reuse`, the cuts) is computed later from `resume`, so the request then reads from 0. A lost link is made
  again ([After a reconnect](#after-a-reconnect)); any other error: fatal.
- STRATA_CKPT_REREAD in remote mode zeroes and Resets the worker too (keep = the chain), so the comparison holds.
- `resume == 0`: Reset with keep = the surviving checkpoints' ids (normally empty).

Cost (estimated, not measured on its own): one round trip per checkpoint saved and per request resumed, plus the
worker's copy (~55 MB for ~17 GDN layers). The prompt is read in parts at the cut points as on one GPU.

Memory: the worker holds up to prompt_cache + 1 parts in RAM plus at most one stale chain (~55 MB each for layers
26-47 of a 48-layer model). The RAM check at hello (As built) counts the prompt_cache + 1 parts, not the stale chain.

### After a reconnect

A main process connects again when a request's start finds the link lost (ENGINEERING.md, "Recovery"). The worker it
reaches then is a new process, or the same one on a new connection: its session is zeroed and its store is empty. A
request that continued the live session or mounted a checkpoint there would run the worker's layers from `resume` on
an empty session, so:

- At each request's start, before its positions are sent and before it decides where to resume, the main process
  checks the link with nothing sent (`StageClient::alive`: a poll of 0 ms, then a 1-byte peek). The worker sends
  nothing unasked, so a socket with something to read means the worker's end closed or failed, or the link is out of
  step; the link is then made again. Without this check a request that continues the live session, which sends
  nothing before its first window, found the loss only there: the engine ended, the server started it again (the
  model loaded anew, the conversation cache lost) and the client got a 503.
- Whenever the link is made again in a request (that check, its positions, its Reset, its CkptRestore), the main
  process drops its live session and every checkpoint (on a remote main each has a part on the worker), and the next
  request that runs - normally this one; one that first ends with an error, such as a picture that cannot be read,
  leaves it to the next - reads its prompt from token 0 with a Reset that keeps nothing. After the line that says the
  link was made again, one more says so when there was something to drop (`the worker holds no session or checkpoint
  after connecting again: ...`; on a CkptRestore, `the worker holds no checkpoint after connecting again: ...`); a
  link made again at the Reset of a request that read from token 0 anyway has only the first.
- A CkptRestore that finds the link lost makes it again and falls back as for a missing id. If connecting again
  fails, or the worker reports an error of its own, the `ERR` line starts with that error (`remote stage: ...`,
  `remote stage (worker): ...`), so the server answers 503, not 400.
- A CkptSave that finds the link lost, or that the worker fails, is in the middle of a request (after the part's
  windows): the engine ends as for a lost window, the `ERR` line again starting with the link's or the worker's error.
- What the check cannot see: a worker PC that vanished without closing (power, cable) looks alive until TCP
  keepalive notices it (about a minute); a request in that time finds the loss at its first message, as before. A
  loss between the check and the request's first window is found at that window, and the engine ends. With a
  relay, the check sees only the main process's own link: a next worker that restarted while the main process was
  idle is found by the relay at the request's first message to it, an error that ends the engine (read from the
  code; relays were not run).

## Checks

- Unit (no GPU): the worker store (save / restore / keep pruning / admission limit / clear on disconnect), the
  explicit-length save/restore sizes (L % idx_block == 0 and != 0), the new messages' encode/decode, protocol 4 refused
  at hello, ckpt_max refused above the worker's limit, and a loopback link test.
- GPU exactness (a main PC + a worker PC; fixed expert sets: --adapt-every 0 / --adapt-swaps 0 both sides,
  STRATA_IQ_MT_MIN=1, CPU share off, greedy). The replay must take the CHECKPOINT path, not from_live: thinking on
  (the template drops the previous turn's thinking, so the prompt leaves `live` at the last turn boundary) or answers
  edited before re-sending, plus a second chat sharing the system prompt (a sibling of the root). Each turn is sent
  twice: the second request mounts the checkpoint the first saved at its turn boundary (both sides) and reads the
  same tail; the answers must match byte for byte, and the logs must show the restores (one per resumed turn); a run
  where they did not happen does not count. STRATA_CKPT_REREAD is not this check for a chain of turns: it reads
  `[0, L)` in one run and every short part batched, so it reads a checkpoint's tokens in the same parts only when the
  request that saved it read them from 0 in one part; a root cut before it, or a checkpoint saved by a resumed turn,
  changes the parts.
- Default path: without --prompt-cache, the remote main's answers and log match the base build (no Ckpt messages).
- Cancellation: a request cancelled during decode, then the next turn of that chat (from_live or checkpoint), repeated.
- Fallback: a worker debug env (STRATA_REMOTE_CKPT_DROP=1: empty the store before a restore) -> the request reads
  from 0 and finishes.
- Speed: the replay at the served flags; per-turn time to the first token against one GPU.
- After a reconnect (not run on GPUs yet): the worker restarted, or taken over by another main process, between
  two turns of a chat; the next turn must read from token 0 (the log lines above) and answer as a run without
  `--prompt-cache` does.
- Builds: CUDA Linux, MSVC (the Windows worker), HIP; SYCL untouched (it has no remote stage).

## As built

- Protocol and the worker's side of it: `include/strata/net/stage_link.hpp`, `src/net/stage_link.cpp`. The keep[]
  payload is read by the receiving thread (more than 64 ids, or a size not a multiple of 8: "bad checkpoint list").
  `serve_stage` keeps `pos` and checks it, reports a pending Commit error on CkptSave and logs and clears it on
  CkptRestore, and refuses a hello whose ckpt_max is above `kStageCkptMax`; the handlers (`StageHandlers::ckpt_save`
  / `ckpt_restore`, `reset` with keep) do the rest. The missing-id error starts with `kStageCkptMissing`
  ("checkpoint not held: "); the client sets `missing` only when the worker's own text starts with it, so a relay's
  forwarded error (prefixed "relay: the next worker: ") is the state class, as above.
- The store: `include/strata/net/stage_ckpt_store.hpp` (header-only; admission, prune, clear; a put never prunes
  its own id, so a part answered "stored" is held even if keep[] left it out). At the hello a worker also refuses a
  main process whose ckpt_max parts of this worker's size would not fit in half of the RAM available
  (`stage_ckpt_ram_ok`, with the engine's `available_host_bytes()`: MemAvailable and the cgroup limits on Linux,
  GlobalMemoryStatusEx on Windows, and `--memory-limit-mib` when set), and any ckpt_max when that RAM cannot be read
  (a cgroup limit that does not parse, as the engine's RAM guard refuses parking then): Linux overcommits, so a store
  that outgrows the RAM would meet the OOM killer, not std::bad_alloc. The worker's handlers
  in `generate.cpp` (`hs.ckpt_save`, `hs.ckpt_restore`, `hs.reset`, `hs.disconnected`) use it; a worker
  synchronizes the device before it copies. A relay whose next worker stored the part but which cannot store its
  own (the limit, host RAM) answers an Error, not a=0: the next worker has already pruned to keep[], so a=0 would
  leave the main process holding ids the next worker no longer has. A relay links its next worker again at the first
  reset after each main process's hello, so the next worker's RAM check sees that main process's ckpt_max, not the 0
  a relay sends at its start; a refusal there ends that first request, not the main process's hello (read from the
  code; relays were not run).
- A live continuation settles the worker first. A request that continues the live session sends no Reset or
  CkptRestore: its first Prefill comes straight after the last request's windows. The worker's `hs.prefill` now waits
  for the last Commit and lands the adaptive tier's swaps in flight (`wait_commit`, `apply_pending(true)`) before it
  lends cache slots to the prompt path, as `hs.reset` and `hs.ckpt_restore` do; otherwise the loan could hand the
  prompt path slots that a swap started by `adapt()` at the last Commit is still writing, some rows come out
  non-finite, and the reply degenerates into token 0 (`!`). Both calls return at once when nothing is pending, so
  every chunk asks. Without checkpoints every request starts with a Reset, which settles, so this never showed before.
  Measured on the pipelined-decode build (see [GPU results](#gpu-results)): 4 such failures in about 10 live
  continuations that read a batched part of about 1,000 tokens or more, 0 in 35 after the change.
- `ConvCheckpoint::remote_id` is an `int32_t` in the padding after `pinned`, not an int64: the struct's size (which
  the parking byte counts and their log lines use) stays the same on the default path. The wire and the worker's
  store keep int64 ids; the main's counter is 32-bit (2^31 checkpoints per process).
- Explicit length: `conversation_checkpoint_save/restore(..., size_t tokens, ...)` in `src/core/conversation_state.cpp`.
  The restore with ids is the shared copy-back plus ple_prev, as before; the explicit one does not touch ple_prev.
- Main side, `generate.cpp`: `checkpoint_at` works the eviction out on a list of pointers with the same victim
  function the real eviction then uses (`evict_victim`), so keep[] is exactly the chain the eviction leaves (plus
  the new id even when the policy itself would drop it: the worker then keeps one stale part until the next
  keep). The shared helper is `session_from0(keep)` (zero every session, Reset the worker keeping `keep`), used by
  `resume == 0`, STRATA_CKPT_REREAD and the missing-id fallback; the fallback then sets `resume`, `from_live`,
  `reread_to` and `checks` where it is. `STRATA_REMOTE_CKPT_DROP=1` is read by the worker's restore handler. One
  stderr line each when a checkpoint is saved on the worker, restored on both sides, or the live session continued.
- Unit checks: `stage_ckpt_store_test`, `stage_link_test` (loopback: the hello refusals, the messages' fields and
  keep[], positions, a=0, missing vs other restore errors and a relay's forwarded missing id, a failed Commit
  reported on CkptSave and dropped by CkptRestore, the position after a reset and a restore, a too long keep[], the
  store cleared on disconnect, a failed image position table reported on a CkptRestore and a CkptSave,
  `StageClient::alive` on an idle link, after a take-over, after the worker stopped and after bytes no message asked
  for, none of its checks over 100 ms; its handlers are the test's own: the worker's real ones in `generate.cpp` run
  only in the GPU checks) and the explicit-length cases in `conversation_transfer_test`. What `generate.cpp` does
  after a reconnect (the dropped session and checkpoints, the read from 0) has no unit test: read from the code.

## GPU results

2026-10-11, one session on engine 0.1.41 (the branch before its rebase): an RTX 3080 main (layers 0-25, the head, the
drafter; `--prefill 5888`) and an RTX 2080 Ti worker (layers 26-47) over 10 GbE, RVN IQ3_S, `--max-context 131072`.
The build measured is this branch plus one commit proposed on its own, a guard that checks each window's logits for
NaN or infinity (#879) and changes nothing while they are finite; the base build is #1748 as it was then, on both
PCs. The checks after the rebases onto 0.1.42 and 0.1.43 follow below
([0.1.42](#after-the-rebase-onto-0142-2026-10-11), [0.1.43](#after-the-rebase-onto-0143-2026-10-12)). The replay: two chats
of four turns on one shared system prompt of about 16,000 tokens, each turn sent with the earlier ones and their
answers as a chat client does, greedy. "Exact flags": `--pcie-frac 0 --adapt-every 0 --adapt-swaps 0` on both PCs,
`--mtp-max-t 1 --suffix-draft 0` on the main, `STRATA_IQ_MT_MIN=1 STRATA_PREFILL_CPU_SHARE=0` on both.

| Check | How | Result |
|---|---|---|
| Default path | no `--prompt-cache`, exact flags, 160 tokens | 8 of 8 answers byte-identical to the base build; no checkpoint line in the log |
| Repeated turns | `--prompt-cache 6`, exact flags, 160 tokens; each answer re-sent with a change at its front (the next turn resumes through a checkpoint); every request sent twice | 8 of 8 repeats byte-identical; 15 restores on both sides, 0 errors |
| Repeated turns, thinking on | exact flags, 300 tokens; answers re-sent as they were (the template drops each answer's thinking, so the next turn resumes through a checkpoint); every request sent twice | 8 of 8 repeats byte-identical; 15 restores |
| Cancellation | repeated turns, each request first sent and dropped after 20 streamed chunks | 8 of 8 repeats byte-identical, and all 8 answers equal the run without cancels; 23 restores |
| Fallback | the worker with `STRATA_REMOTE_CKPT_DROP=1`; answers re-sent with a change at their end (most turns continue the live session) | the one restore (the second chat's shared system prompt) missed; that request read its prompt from 0 and finished; 0 errors |
| Live continuation, adaptive tier on | served flags, thinking off; each answer re-sent whole plus 100-2,500 new tokens (about half of them real mis-decoded tool output), 300 tokens, 15 minutes | 163 requests, 157 continued the live session, 102 of them read 1,000 new tokens or more: 0 replies of `!`, 0 windows flagged by the guard |

With `--prompt-cache` the answers need not match a run without it byte for byte: a prompt read from 0 is read in
parts that end at its cut points, and a resumed or continued turn reads only its new tokens, where the default path
reads the whole prompt in chunks of 5,888 - a different order of arithmetic. In the fallback run 4 of the 8 answers
differ from the default path's, the first turn (read from 0) among them. Byte equality is what the repeated turns
check: both requests read the same parts.

Time to the first token at the served flags, thinking on, 400 tokens:

| | Base build (every prompt from 0) | This branch, `--prompt-cache 6` |
|---|---|---|
| First turn (16,079 tokens) | 14.61 s | 15.30 s |
| The 7 later turns | 14.20-14.92 s (mean 14.55) | 0.33-0.49 s (mean 0.41); 7 restores on both sides |
| Decode, mean | 54.4 tok/s | 56.8 tok/s |

A turn read from 0 was 0.7-1.3 s slower with `--prompt-cache` in this session (here 14.61 -> 15.30 s; at the exact
flags 14.54 -> 15.81 s from the base build to the repeated turns, and 14.65 / 14.88 -> 15.67 / 15.75 s from the
default path to the fallback run's two turns read from 0): the prompt is read in parts that end at
its cut points, and a checkpoint is saved on both PCs at each (not timed apart). The answers differ (the first chat's
later prompts already do), so the decode rates are not a like-for-like comparison.

### After the rebase onto 0.1.42 (2026-10-11)

Both branches were rebased onto upstream main 61b3fb5 (engine 0.1.42). The hello now also carries the main process's
tail-skip rank (#1748, protocol 4), so this branch's protocol was 5 (6 since #1748's images), and a worker that
cannot read its RAM refuses checkpoints at the hello. Same pair, same replay and flags; this branch's build without
the guard, and #1748's rebased build as the base:

| Check | Result |
|---|---|
| Default path on the split (no `--prompt-cache`, exact flags) | 8 of 8 answers byte-identical to the base build in 2 of the 6 pairs between two runs of this build and three of the base's; the other pairs agreed on 4-7 of 8 turns, each the same until one answer diverged mid-way, as 0.1.42 does on one card |
| One GPU, conversation cache (the restore code this branch reshaped), exact flags | in the second of two rounds, upstream main's, #1748's and this branch's builds 8 of 8 byte-identical (upstream main's own two runs: 1 of 8) |
| Repeated turns | 8 of 8 repeats byte-identical; 15 restores on both sides, 0 errors |
| Repeated turns, thinking on | 8 of 8; 15 restores |
| Cancellation | 8 of 8, and all 8 answers equal the run without cancels; 23 restores |
| Fallback (`STRATA_REMOTE_CKPT_DROP=1`) | the one restore missed; that request read its prompt from 0 and finished; 0 errors |
| Live continuation, adaptive tier on, 15 minutes | 161 requests, 154 continued the live session, 98 of them read 1,000 new tokens or more: 0 replies of `!` |

Time to the first token at the served flags, thinking on: the first turn 16.00 s (the base build: 14.68 s), the 7
later turns 0.28-0.54 s, mean 0.38 (the base build: 14.57-15.06 s); 7 restores on both sides. Decode 64.1 tok/s mean
(the base build 61.5; the answers differ, so not like for like).

Builds: CUDA 13.0 on Linux (GCC, sm_75 + sm_86; the unit checks above pass), MSVC 2022 + CUDA 13.0 (the touched files
compile; `stage_link_test`, `stage_ckpt_store_test`, `conv_cache_test`, `conversation_cache_test` and
`draft_kv_plan_test` pass), HIP (ROCm 7.0, gfx1100) and the SYCL port (oneAPI 2026.1.1; no remote stage there), both
compile only.

### After the rebase onto 0.1.43 (2026-10-12)

#1748 was rebased onto upstream main 7695859a (engine 0.1.43) and gained images: the main process sends the worker
its M-RoPE position table (protocol 5). This branch went on top without a textual conflict and is now protocol 6.
A failed position table outlives a CkptRestore as it outlives a Reset, because it belongs to the request being
started.

The setup was the same pair, the same replay and the same flags as before. This branch's build ran against #1748's
build on 0.1.43:

| Check | Result |
|---|---|
| Default path on the split (no `--prompt-cache`, exact flags) | one run of each build: 8 of 8 answers byte-identical to #1748's build |
| Repeated turns | 8 of 8 repeats byte-identical; 15 restores on both sides, 0 errors |
| Repeated turns, thinking on | 8 of 8; 15 restores |
| Cancellation | 8 of 8, and all 8 answers equal the run without cancels; 23 restores |
| Fallback (`STRATA_REMOTE_CKPT_DROP=1`) | the one restore missed; that request read its prompt from 0 and finished; 0 errors |
| Pictures (`--vision` on both PCs): two-turn picture chats, the second turn sent twice | the second turn resumed through a checkpoint or continued the live session (2 restores, 2 live continuations); the re-sent turn byte-identical; all 6 answers equal those of a run without `--prompt-cache` |
| Live continuation, adaptive tier on, 15 minutes | 165 requests, 158 continued the live session, 97 of them read 1,000 new tokens or more: 0 replies of `!` |

In these runs the worker's adaptive overlap was on (#1748's build then followed upstream's default on the worker
too). A worker now runs without it unless `STRATA_ADAPT_OVERLAP=1`: the worker's path of the 0.1.42 runs above,
where the same live-continuation check gave 161 requests and 0 replies of `!`.

Time to the first token at the served flags, thinking on, 400 tokens:
- **Without checkpoints (#1748's build):** 14.19-15.06 s on every turn (mean 14.62).
- **With checkpoints, the first turn:** 14.50 s.
- **With checkpoints, the 7 later turns:** 0.32-0.44 s (mean 0.37), with 7 restores on both sides.

Decode averaged 63.5 tok/s with checkpoints and 61.1 without. The answers differ, so these two are not like for like.

Builds:
- **CUDA 13.0 on Linux** (GCC 13, sm_75 + sm_86): `stage_link_test`, `stage_ckpt_store_test`, `conv_cache_test`,
  `conversation_cache_test`, `conversation_validation_test` and `conversation_transfer_test` pass.
- **MSVC 2022 + CUDA 13.0:** the touched files compile; `stage_link_test`, `stage_ckpt_store_test`,
  `conv_cache_test`, `conversation_cache_test`, `conversation_validation_test` and `draft_kv_plan_test` pass
  (`conversation_transfer_test` is built on Linux only).
- **HIP** (ROCm 7.0, gfx1100) and **the SYCL port** (oneAPI 2026.1.1; no remote stage there), in containers: both
  compile.

The feature was first validated on a build with the remote decode pipelined (`--pipeline-windows 2` on a remote main,
on an earlier base), on an RTX 3080 main + an RTX 2080 Ti worker over 10 GbE: the answers after a
restore matched the request that saved the checkpoint byte for byte, the default path matched its base build, and a
forced restore miss read from 0 and finished. Those numbers belong to that build and are not repeated here.
On this branch with `--pipeline-windows 2`, the same PCs and model, exact flags, `--expert-cache 1951` and
`STRATA_ROUTE_TAIL_SKIP=0` on the main process:

| Check | Result |
|---|---|
| Repeated turns (every request sent twice) | 8 of 8 repeats byte-identical; 15 restores on both sides, 0 errors |
| Repeated turns, thinking on (300 tokens) | 8 of 8; 15 restores |
| Cancellation | 8 of 8, and all 8 answers equal the run without cancels; 23 restores |
| Fallback (`STRATA_REMOTE_CKPT_DROP=1`) | 2 restores missed in 16 requests (two passes over the 8 turns, one miss in each: the second chat's first request); those requests read their prompt from 0 and finished; 0 errors |
| Default path (no `--prompt-cache`), the first replay on fresh servers, against the first `--pipeline-windows 0` server of the branch this one is built on | `0`: 4, 8 and 8 of 8; `2`: 8 and 7 of 8 |

The runs at 4 and 7 of 8 took another word partway into an answer. In the same session, #1887's head (no
`--pipeline-windows` on a remote main) gave 5 and 5 of 8 against the same run, and 2 of 8 against each other, so a
single fresh server does not settle the default path here; three of this branch's five matched it.

### After the rebase onto 0.2.0-beta (2026-10-20)

#1748 was rebased onto upstream main e88ee39d (0.2.0-beta, engine 0.1.45.3) and this branch on top of it. The
textual conflicts were in `generate.cpp`, around the main process's checkpoint chain, which upstream changed after
0.1.43:
- `--prompt-cache-spread` drops the checkpoint whose removal leaves the smallest gap for its distance from the
  newest (`spread_victim`) instead of the least recently used.
  The main process works out the chain a new checkpoint leaves before the worker stores its part (the ids it sends to
  keep), so that step and the eviction after it now ask the same function (`evict_victim`), which follows the flag:
  the worker drops what the main process drops.
- `STRATA_CKPT_POOL` and `STRATA_CKPT_THREADS` (opt-in) reuse a dropped checkpoint's buffers and save the stages'
  parts on threads. A remote main has no in-process stage, so its own part is the only one, and the worker's part is
  kept by id on the worker: both switches act on the main process's buffers alone. The ids and the eviction stay as
  the chain's.
- Upstream's lines next to this branch's (the restore's "continuing the live session" line, the pipelined drain's
  error cleared before the admission merge's check): both kept.

0.1.44's Metal backend hands `conversation_checkpoint_save` and `_restore` to its decode window; the range-taking
overloads this branch adds stay on the session's buffers, and a remote stage is refused on Metal (#1748). The
protocol is unchanged (6).

The GPU checks above were not run again on this base.
