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

Port of Project Maya's DeepSeek restore points into `src/dsv41` (0.2.9.x, branch f-041-maya-a), staged against Maya v1.1.42
(MIT, same copyright and permission notice as above):
- A0, restore points (`ds4_marks.hpp`, `Model::mark`, marks in `restore()` and in the conversation slots, the `marks=` /
  `shared=` / `tail=` GEN keys): Project Maya #639 (needmorevram, 50a3313; `ds4_marks.hpp` also holds #668's tail-cost
  logic by mw00, 8721412 and 66fb17b). Test: `tests/dsv41/marks_test.cpp`, from
  Maya's `tests/ds4_marks_test.cpp`.
- A1, the server side (`deepseek_restore_points`, `restore_point_settings`, `StrataEngine.restore_keys`,
  `serve/test_ds4_restore_points.py`): Project Maya #639 (needmorevram) and v1.1.37 / v1.1.39 (mw00).
- A2, the tail restore point judged by its measured cost (`TailCost` in `ds4_marks.hpp` wired into the serve loop in
  `ds4_main.cpp`; `STRATA_DS4_RESTORE_TAIL_AUTO=0` keeps it always): Project Maya #668 (mw00, 8721412 and 66fb17b).
- A5, the prompt path's flow timing correction kept per prompt-size class (`ds4_path_policy.hpp`, `PathModel::flow`; only
  the opt-in `DS4_PICK=adaptive` timing pick uses it, the default pick is by length): Project Maya #612 (Will Hampson,
  6e0d27d). Test: `tests/dsv41/path_policy_test.cpp`.

Taken from Project Maya v1.1.35 to v1.1.42 on branch f-040-maya-ds:
- the thread pool's generation fence (#618, Cody Harrison).
