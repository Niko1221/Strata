# Project Maya (maya-ds4), DeepSeek-V4.1 engine

The files under `src/dsv41/`, `include/strata/dsv41/`, `tests/dsv41/` and `tools/dsv41/` come from
https://github.com/mw00/project-maya (release v1.1.17, commit 8d9a9b5d), its DeepSeek engine `src/ds4`, `tests/ds4_*`,
`tools/ds4` and the kernel sources it builds on (`exl3`, `sampler`). Project Maya is itself built on Strata.

Copyright (c) 2026 Peasant Smith (mw00) and the Project Maya contributors, including needmorevram.
MIT License, the full text is in `LICENSE` beside this file. The notice stays with every copy.

Changes made while porting into Strata are listed in the git history of the `src/dsv41` directory
(the first commit of each file is the unmodified upstream copy).

Later changes taken from Project Maya releases v1.1.18 to v1.1.26 (the same approach re-written onto Strata's engine,
for example the long-prompt layer skip, the thinking budget, conversation slots) are covered by the same copyright
and permission notice, in `LICENSE` beside this file.

MAYA-B port (2026-10-31), staged merges onto the v1.1.17 base:
- v1.1.17 to v1.1.19: the decode PCIe lane (#312, Peasant Smith / mw00; ds4_pcie.hpp, ds4_pcie.cuh, the lane block of ds4_model.cu,
  tests/dsv41/ds4_lane_bench.cpp, the lane case of ds4_gguf_test), with the same tag step's Q8_K activation rounding, penalties,
  DISK_SKIP_AUTO, the stdin read(2) loop and the CPU-lane tuning. In Strata the lane (DS4_STREAM=1) and the Q8_K rounding
  (DS4_IQ_Q8K=1) are opt-in until their gates have run.
- #329 (mw00): the lane judged over whole decode steps.
- #344 (needmorevram; selective port): the expert cache's recent slots (rc_* fields, the fill thread, tab_set, batched table updates), opt-in
  DS4_RECENT=<n>. Not taken: its all-reduce over DMA, segmented top-k and the tensor-core index scores.
- #476 (RubenTorney), 1b60249 / 482e46c / 7c0ceb1 (mw00, #496 #551), 32f3c8c (handmade0octopus), #646 (handmade0octopus), #567 (TheSmokeDev),
  #563 (Cass, mw00): one-GPU decode with the CPU lane's copies pulled in place, verify windows streamed, streamed experts landing in recent
  slots, IQ1_S multi-token CPU rows. In Strata the lane is opt-in (DS4_STREAM=1); the helper-GPU condition (#415) is not ported.

Port of Project Maya's DeepSeek restore points into `src/dsv41` (0.2.9.x, branch f-041-maya-a), staged against Maya v1.1.42
(MIT, same copyright and permission notice as above):
- A0, restore points (`ds4_marks.hpp`, `Model::mark`, marks in `restore()` and in the conversation slots, the `marks=` /
  `shared=` / `tail=` GEN keys): Project Maya #639 (needmorevram, 50a3313; `ds4_marks.hpp` also holds #668's tail-cost
  logic by mw00, 8721412 and 66fb17b). Test: `tests/dsv41/marks_test.cpp`, from
  Maya's `tests/ds4_marks_test.cpp`.
- A1, the server side (`deepseek_restore_points`, `restore_point_settings`, `StrataEngine.restore_keys`,
  `serve/test_ds4_restore_points.py`): Project Maya #639 (needmorevram) and v1.1.37 / v1.1.39 (mw00).
- A3, shared start across conversations (`prefix_keep` / `prefix_find` / `prefix_take` in `ds4_main.cpp`): Project Maya #607
  (mw00, b2d4c30, v1.1.37).
- A4, restore points near a prompt's end kept in one read (`Model::feed_marks`, `marks_inside`, `feed(..., after)`, the
  layers' snapshot sets `snap_win[]`/`snap_kv[]`/`snap_sc[]`): Project Maya #691 (needmorevram, c7d08c5, v1.1.40). The
  "a chunk streams only the experts it picks" and EXL3 lend parts of #691 are not ported.
- A2, the tail restore point judged by its measured cost (`TailCost` in `ds4_marks.hpp` wired into the serve loop in
  `ds4_main.cpp`; `STRATA_DS4_RESTORE_TAIL_AUTO=0` keeps it always): Project Maya #668 (mw00, 8721412 and 66fb17b).
- A5, the prompt path's flow timing correction kept per prompt-size class (`ds4_path_policy.hpp`, `PathModel::flow`; only
  the opt-in `DS4_PICK=adaptive` timing pick uses it, the default pick is by length): Project Maya #612 (Will Hampson,
  6e0d27d). Test: `tests/dsv41/path_policy_test.cpp`.

Taken from Project Maya v1.1.35 to v1.1.42 on branch f-040-maya-ds:
- the thread pool's generation fence (#618, Cody Harrison).
- Small items (A2 slice): #619 sampler error ends the request and `logits()` copies on the head stream (Cody Harrison, 2040174), #620 CPU-lane layers that cannot compute stop the engine instead of answering zeros (Cody Harrison, e598716), #676 `bf16_rn` by arithmetic (rilken, 79628d7), #608 `DS4_ASYNC_T=16` as an opt-in value (mw00, b9c07e7).
- A4b, a chunk of few rows streams only the experts it picks, on the batches path (`DS4_SKIP_UNPICKED_ROWS`, `Streamer::skip_k`
  in `copy_ahead`; opt-in): Project Maya #691 (needmorevram, c7d08c5). The flow path (the default for in-place GGUF
  experts) already read only routed experts.

MAYA-PAR (2026-10-31, branch f-041-maya-par), DeepSeek `--parallel`:
- #332 (mw00, 5da555f): several conversations in one engine (`Options::seqs`, `Model::seq_bind` / `step_multi` / `sample_multi`, the
  per-conversation rings, partial groups, KV and index keys, `--parallel N`, the multi protocol of `--serve` (GEN rid=<id>, READY multi=<n>),
  `--multi-check`, `ds4_gguf_test --rows`). Strata parks the restore points (marks) per conversation too. The server side
  (`StrataEngine` reading the tagged lines, `parallel_args`, `serve/test_ds_parallel.py`) is written for Strata's server.
- #609 (mw00, 2e0d2f9 and 4b76907): the prompt ring sized by the GPU's measured host link (`ds4_ring_link.hpp`, `link_h2d`,
  `tests/dsv41/ring_link_test.cpp`); in Strata opt-in, `DS4_RING_MB=link`.

MAYA-PORT2 (2026-10-31, branch f-041-maya-port2), DeepSeek `--parallel` scheduling:
- #587 (mrsirg97-rgb, a73bdbc): the serve loop on an event queue (`ds4_evt.hpp`, `tests/dsv41/evt_test.cpp`), a prompt read in three steps (begin, piece, end), `Model::set_yield` / `fed` / `prompt_chunk` / `pipeline_stages` (a read alone ends at a chunk every stage reaches once a request waits). Prompts take turns a piece each with the answers' tokens between only with `STRATA_DS4_READ_PIECES=1`; off by default.
- #611 (needmorevram e3de4aa; mw00 1434096, 4b2378b, 0d0f261): the DSpark drafter in the steps of several requests (`Model::step_multi_spec`, `verify_multi`, `can_draft_multi` / `draft_multi`, per-conversation compressor snapshots, `--multi-check` passes with the drafter). The multi step's rows always on the async path; each run's replicated rows at its own place of the pinned stage. The prompt lookup, think budget and DS4_DECODE_TRIM parts of Maya's diff are not in this tree and were left out.
