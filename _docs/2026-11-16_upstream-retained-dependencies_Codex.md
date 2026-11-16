# Retained dependency restoration, 2026-11-16

The final integration review found nine missing files still referenced by retained source or tests. This wave restores only those files and this record. Its starting all-index tree was `d15b17be85b8f6b50a75cf0a11889f2c98a442ff`. The immutable restoration source is fork commit `6f13c7555a7d50289f0e3587ab780627c65a8000`. No test assertion or resource threshold was changed.

The canonical PC prompt was read before edits. Applied controls were MILSPEC Codex Standard, common engineering, implementation start gate, skill selection, security, application development, Python and C++ SOPs. Stronger active instructions take precedence over incompatible identity, platform or tool claims in the prompt. This is a bounded restoration record, not a formal MILSPEC certification.

## Scope and provenance

`src/dsv41/ds4_model.cu` refers to the Project Maya notice. The Metal vision source refers to antirez/ds4. Their original MIT licences and notices preserve the named authors and port history. New gguf-py licences cover different authors and do not replace these files.

`tests/terminal/test_pack_gguf_engram.py` opens two 416-byte real-header fixtures. `tests/packaging/test_backend_packages.py::test_windows_packaging_scripts_present` unconditionally requires the Windows helper. `tests/dsv41/tqw_hip_test.cpp` includes `wmma_emu.hpp`, which includes `computed_codes.hpp`. Only those two lab headers are restored.

All nine files were extracted with binary `subprocess.check_output([git, show, COMMIT:PATH])` and written with `Path.write_bytes`; no binary passed through a text pipeline. Eight files match the frozen source byte for byte. The batch helper alone removes the obsolete mandatory external `STRATA_GGML_DIR` check: an empty override now permits `cmake/strata_ggml.cmake` to choose the vendored tree. Its default compiler jobs are five under the PC policy. Comments identify an optional developer helper and make no claim that the fork distributes or has built the DeepSeek executable.

| Restored path | Bytes | SHA-256 (working file; identical to source except helper) |
| --- | ---: | --- |
| `third_party/project-maya/LICENSE` | 1160 | `574cffb7c57769a9271043ac0c2b07c5e7e7621cfb81081b901d8981e791d121` |
| `third_party/project-maya/NOTICE.md` | 6544 | `17f937b6509d72a7deb92cd78a4de0e325ee8c83e4b358fe439bd43a6cc9a844` |
| `third_party/antirez-ds4/LICENSE` | 1143 | `4a8a1358f5b048627d71e4988de0bafd386f4988991929629e6cb9e29b7b9b2f` |
| `third_party/antirez-ds4/NOTICE.md` | 1013 | `6aa872c5503bcdaebf5587bfd99ff38de4025b9fb4b7b040b7387cc8af70a2cb` |
| `tests/fixtures/dsv41/engram-final-header.gguf` | 416 | `97a55e86b495fdc4ae2aea6d52a337578d2e514dc099676164cda48c635f40d1` |
| `tests/fixtures/dsv41/engram-converted-27oct-header.gguf` | 416 | `b2d8082cdfee3c7630adb23ff64593bd417c56533aa0c97e82088dc686db4efe` |
| `tools/dsv41_lab/wmma_emu.hpp` | 9655 | `b1dcce1398c795515290ad9810907a31836e50bf2219b791ca9b0c2069dbe666` |
| `tools/dsv41_lab/computed_codes.hpp` | 3693 | `cebb4b27a18de0a2fd85f3c2f0b885980dd74f4fa9619f94c3eedf9043880c1e` |
| `packaging/build_dsv41_windows.bat` (adapted) | 2508 | `c60529d8b909c72c694094a9acd87e056826c25de100a5192c94b4828c323296` |

The source helper SHA-256 is `c11fd8f0a593acbb78d869da6e8bfb6508a17c7a8c2b893f952c18bd45404c42`; its bounded adaptations are described above and visible in the diff.

## Fresh verification

Before restoration, the focused Engram and Windows script-presence suite produced **7 failed, 7 passed**, exit 1. The failures identify the missing fixture files and batch helper. After restoration, the unchanged suite produced **14 passed in 0.39 seconds**, exit 0. The original missing-header/licence findings were static source-reference findings, separate from the pytest failures.

The restored host header and its local dependency also passed C++17 syntax checking with the installed Clang, exit 0. The sole diagnostic is `#pragma once in main file`, because the header was deliberately compiled directly. This checks host C++ parsing and include resolution; it does not compile HIP kernels or prove GPU numerical behaviour.

Each command below used the existing production guard. It retains RAM 3 GiB, commit headroom 4 GiB, H: free space 10 GiB and C: free space 5 GiB, filtered environment, test concurrency one, compiler parallelism five, 300-second deadline and 8 MiB output limit. All three receipts report `FINISHED`, observed native exit codes, confirmed owned-process termination and `job_active_after=0`. No process cap, floor relaxation or inherited environment dump was introduced.

```powershell
uv run --no-project --offline --python C:/Users/downl/AppData/Local/Programs/Python/Python312/python.exe python _docs/upstream-sync-20261116/check.py --seconds 300 retained-dependencies-red-20261116 python -m pytest tests/terminal/test_pack_gguf_engram.py tests/packaging/test_backend_packages.py::test_windows_packaging_scripts_present -q
uv run --no-project --offline --python C:/Users/downl/AppData/Local/Programs/Python/Python312/python.exe python _docs/upstream-sync-20261116/check.py --seconds 300 retained-dependencies-green-20261116 python -m pytest tests/terminal/test_pack_gguf_engram.py tests/packaging/test_backend_packages.py::test_windows_packaging_scripts_present -q
uv run --no-project --offline --python C:/Users/downl/AppData/Local/Programs/Python/Python312/python.exe python _docs/upstream-sync-20261116/check.py --seconds 300 retained-dependencies-host-header-20261116 'C:/Program Files/LLVM/bin/clang++.exe' -std=c++17 -fsyntax-only -x c++ tools/dsv41_lab/wmma_emu.hpp
```

Private receipts and command logs are retained under `H:/Strata/evidence/upstream-sync-20261116/receipts/`, using the three command names above and `.json` / `.command.txt` suffixes. The red, green and host-header command-log SHA-256 values are respectively `124e589825f201903e11f0df910eeab145411cafa2c1be8ef6d37c661ae19f83`, `2728ef65d6a5df391855139a31136ee42ea98fc3db137e985b31b0ea5ddc4a08` and `db8a973af5c05f6d76eb7ed9d146ecf08e502d75117ad8868887b56764363b5a`.

## Limits and handoff

Actual HIP compilation and execution are **NOT_RUN**: `hipcc` was absent from the inspected command path, and no compiler installation or GPU use was authorised for this wave. The optional Windows CUDA helper was not run, configured or used to build a product. CUDA/Metal/HIP model inference, distributed packages, release qualification and the full packaging suite are unverified by this restoration. Existing `releaseQualified=false` remains unchanged. No commit, push, release, key change, model stop or GPU command occurred here.

Self-review-event: `qa_defect` and `ticket_completed`, review-only. The red/green evidence supports checking retained source include paths, fixture opens and licence references during deletion review. Broader governance edits and importing the whole developer lab were rejected as outside this wave. Rollback is an exact reversal of these ten owned paths; other staged work must remain intact. The next action is the parent's scoped final re-review before its separately authorised main merge and commit/push.
