# RAM32: Sync-Strata's pre-start squeeze and commit-limit advice, ported

The user asked to adopt, from Sync-Strata (`zapabob/sync-strata`, fork of
`hiruhiru2005/sync-strata`, MIT, itself a Strata fork), the parts that let a
125B-expert model run on a 32 GB PC. Their launcher adds three things this
fork's Terminal lacked: a pre-start **memory squeeze** (other programs' pages
moved to the page file, nothing closed), a **commit-limit-aware budget
estimate with advice**, and named **commit-heavy programs** in that advice.

The port keeps this fork's shapes: the engine already clamps the expert copy
to `min(physical, commit) - headroom - 0.25 GiB`
(`src/core/expert_source.cpp`, `clamp_resident_budget`; headroom default
4 GiB, `src/program/generate.cpp`), so nothing changes in the engine and no
start is refused - the Terminal only says what will happen and hands the copy
more free RAM. Constants: the engine's own pre-copy commit (2.0 GiB) is the
Sync-Strata launcher's measured `ENGINE_COMMIT_GIB`; the headroom follows
`STRATA_RESIDENT_HEADROOM_GIB` (4 GiB default) so the estimate matches this
engine, not Sync-Strata's.

New `strata_control/ram_fit.py` (stdlib only, every entry point best effort
and never raising): `mem_gib` (GlobalMemoryStatusEx / MemAvailable),
`commit_room_gib` (the engine clamp mirrored, floor 0.25 GiB steps),
`commit_cut_notes` (the two advisory lines), `top_commit_apps` (PrivateUsage
per program), `budget_from_args`, and `squeeze` (Sync-Strata's two stages:
EmptyWorkingSet on the session's other programs, then a short VirtualLock
flush; Windows critical processes, our own tree, the engine and python are
excluded; working-set limits read first and always restored; 4 GiB commit
kept free; Linux: drop_caches only when root-writable). Gates:
`STRATA_RAM_SQUEEZE` = `auto` (default: only a start that asks for a RAM
budget squeezes), `on`, `off`; `STRATA_FAKE_SQUEEZE=before:after` is the test
fake. A squeeze that frees less than 0.1 GiB prints nothing.

Wiring: `serve/server.py` squeezes once in the `engine == "strata"` branch
right before the model load (auto mode runs it only when the config asks for
`--resident-budget-gib`), and `warn_budget_over_ram` (#1080) now also prints
the commit-limit lines and works without psutil (the ctypes read takes over).
`setup.py` prints the same commit advice after its "RAM budget" line.
`docs/TROUBLESHOOTING.md` gains the 32 GB commit-limit paragraph. The register
keeps the audit truthful: `STRATA_RESIDENT_HEADROOM_GIB` records the new
launch read (`strata_control/ram_fit.py:93`) and the TROUBLESHOOTING doc
mention lines that my paragraph moved. `STRATA_RAM_SQUEEZE` and
`STRATA_FAKE_SQUEEZE` are launch-side only and the engine never reads them,
so the register (engine reads) does not list them - like
`STRATA_ENGINE_NICE`. The full regen was skipped because main's register is
already stale (three unregistered `STRATA_Q38_*` engine reads in
`src/qwen38/engine.cu`, unrelated work); `tools/settings_register.py --check`
still reports exactly those three, nothing more.

Verified: `py -3 -m pytest tests/terminal/test_ram_fit.py` 19 passed;
`BudgetOverRam` in `serve/test_server.py` 7 passed (the pre-existing
`test_a_budget_that_fits_is_quiet` extended: it now pins the commit-free
value, since a budget that fits the free RAM alone can still be cut);
full `serve/test_server.py` 387 passed, 3 skipped; the rest of `serve`
1346 passed, 24 skipped (16 min); the RAM32 suites
`tests/resources/test_model_memory.py`, `tests/models/test_model_memory_real.py`,
`serve/test_resource_warnings.py` 7 passed, 3 skipped;
`py -m ruff check` clean on both new files (the remaining findings in the
touched tracked files are pre-existing; the repo has no linter config and CI
gates with tests); `git diff --check` clean; `py_compile` clean.
Real-machine read on this 32 GB PC: commit free 26.49 GiB, room 20.0 GiB,
and the 40 GiB-budget advice named msedge 4.3 / 7z 3.0 / OpenCode 2.1 GiB.

The full `tests/terminal` suite on the changed tree ran 3514 passed, 16
failed, 115 skipped (75 min). One failure was this task's:
`test_no_debugger_children.py` lists every place product code starts a
process, and `ram_fit.py`'s two powershell probes were unlisted - fixed by
adding the file to `OTHER_PROGRAMS` (the file's 18 tests pass). The other 15
are the working tree's pre-existing state, none of them touching this task's
files: the uncommitted 0.6.5竊・.6.6 version bump (the two
`test_tui2_023_versions` and `test_tuiqa_version` identity tests read
VERSION 0.6.6 against a 0.6.5 manifest and the 0.6.3 AGENTS.md), WSL's
missing /bin/bash (`test_install_sh_syntax_and_guard`,
`test_docker_files` smoke test), and network/signing/build-script tests
(`test_net_launch_waits`, `test_dsv41_engine`, `test_net_install_sh`,
`test_update_notice`, `test_sycl_build_script`, `test_delivery_q38_current_proof`,
`test_release_keys`) that do not read this task's code. The VERSION/README/
package files in the diff are the previous session's bump work, left
uncommitted here and untouched.

Residuals: the real two-stage squeeze is not re-measured on this PC (only the
test fake ran here; the stages are Sync-Strata's working code with this
fork's names, and a real start will exercise them); the pre-start squeeze
runs only through `serve/server.py`, so an app-host start that spawns the
engine directly is out of scope; Sync-Strata's per-start budget recompute,
RAM-tiered pack choice and `STRATA_PARTIAL_PIN_GIB` pin rule were not ported
(the pin measured no faster here and changes numerics; the pack tiers do not
map to this fork's model list; the engine already clamps per start).
Recommended next: one real `strata launch` with a RAM-budget model on this PC
to see the squeeze and the commit lines in a live start, then the full
Harness regression group before any release.
