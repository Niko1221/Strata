# What was left out of this folder, and what was changed in what stayed

Everything in this folder was produced on one measuring machine in one working folder
(`<HOME>/strata185/Strata+4xV100+NVlink/`, 771 files, 54 MB). This PR carries 615 files, 8.8 MB.
Nothing was re-measured to make the package tidier, nothing was overwritten, and no number was changed.

## 1. Renames

The documents were translated into English, and a few names were made less dependent on the internal
numbering of the work that produced them. The content is the same document.

| on the measuring machine | here |
|---|---|
| `stand-validation.md` | `ab-stand.md` |
| `strata-architecture-audit.md` | `sm70-compatibility.md` |
| `strata-data-path-analysis.md` | `data-path-analysis.md` |
| `patch-recommendations.md` | `patch-notes.md` |
| `tests/cuda/` + `source/*.cu` (probe sources) | `probes/`, `probes/cuda-tests/` |
| `README.md` (the working folder's index) | `README.md` (rewritten as the index of this package) |

Unchanged names: `p2p-results.md`, `test-plan.md`, `baseline-profile.md`, `correctness-report.md`,
`system-inventory.md`, `baseline-profile.csv`, `correctness-matrix.csv`, `p2p-matrix.csv`,
`p2p-benchmark.csv`, `p2p-correctness.json`, `system-inventory.json`, `comparisons/`, `configs/`,
`scripts/`, `raw/`.

## 2. Working documents that stayed on the measuring machine

They are the process around the measurements, not results: task statements, hand-off notes for the next
session, a progress log with decisions in the order they were made. Their factual content is already in the
documents that are here.

| file | size | why it is not in the PR |
|---|---:|---|
| `PROGRESS.md` | 52 KB | the running log of what was done and of the working rules (R1-R13); the rules that matter for reading the numbers are restated inside the reports |
| `run-log.md` | 101 KB | a timeline of commands with their output; every entry in it led to an artifact under `raw/` |
| `HANDOFF.md`, `HANDOFF-PROMPT.md` | 62 KB | instructions for the next working session on that machine |
| `research-task.md` | 44 KB | the task statement the work was planned against |
| `audit-existing-materials.md` | 29 KB | an inventory of what PR #2525 already contained, so it would not be measured twice; its conclusions are used in `baseline-profile.md` §0 and `system-inventory.md` §8 |

## 3. Large artifacts that stayed on the measuring machine

| path | size | why | how to regenerate it |
|---|---:|---|---|
| `build/kernels/` (62 cubins) | 38 MB | binary extracts of the engine binary; they are the engine's own code, not data | `cuobjdump -xelf all <engine binary>` (`correctness-report.md` §7) |
| `build/cubin-symbols.csv` | 288 KB | a subset of `raw/build-audit/2026-10-17T003034Z/kernel-inventory.csv`, which is included and carries more columns (translation unit, registers, shared, local) | the command in `correctness-report.md` §7 |
| `build/p2p-probe-ext`, `build/peer-read-probe`, `build/sm70-tests`, `build/kernel-load-audit`, `build/negctl/` | 4.3 MB | compiled binaries; the sources are in `probes/` and the build commands are in the reports | the "How to repeat" section of each report |
| `source/pr-2525/` | 2.5 MB | a local copy of PR #2525's files (`pr2525.diff`, `pr-files-list.json`, `runs.json`, `env.json`, `needles.json`, `interconnect-experiments/`) kept so the comparison could be made offline | it is PR #2525 itself, already in the repository |
| `scripts/__pycache__/` | 152 KB | compiled Python | - |

## 4. Anonymization

Five substitutions were applied across the carried files. They are the only edits made to any raw artifact,
script or config. The originals are written out so that the substitution can be reversed and checked; none of
them is a secret - they identify the machine, not an account.

| original | here |
|---|---|
| `/home/dgbox` | `<HOME>` |
| `dgbox-ai-server` (the host name) | `<HOSTNAME>` |
| `dgbox` (the user name) | `<USER>` |
| `192.168.2.185` (the machine's LAN address) | `<HOST>` |
| the value of `api_key` in the service config | `<REDACTED>` |

187 of the 577 files under `raw/`, `comparisons/` and `configs/` contained at least one of these strings and
were edited. This claim is checkable: applying the five substitutions in reverse to the copies in this folder
reproduces the SHA-256 of all 577 originals on the measuring machine, with no exceptions.

## 5. Tooling

`scripts/` (16 tools) and `probes/` (5 probe sources) are the programs that produced the numbers. They are
carried unchanged: 20 of the 21 source files are byte-identical to the ones on the measuring machine, and
`scripts/baseline-profile.py` differs only by the `<HOME>` substitution above. The SHA-256 values quoted in the
reports identify these files as they ran on the measuring machine.

Two notes for a reader who wants to run them:

- **The comments and the printed messages inside the tools are in Russian**, and so are the notes written by
  those tools into some raw artifacts (CSV `note` columns, probe stdout headers, `run-notes.txt`). The engine's
  own log lines, all numbers, all identifiers and all API traffic are in English. Each report states what its
  tool does, how to call it and what it writes, so the tools can be used without reading their text.
- The tools assume the layout of that machine (`<HOME>/strata185/...`, port 8080, the private CUDA 12.8 and
  g++-14 under `<HOME>/strata185/`). Paths are read from the top of each script; nothing is installed by them.

## 6. Secrets

No API key, token or password appears anywhere in this folder. The service config is identified by its SHA-256
(`f539e51b…`), and the only config file carried here (`configs/strata-185-iq3_s.split12.json`) has its
`api_key` value replaced by `<REDACTED>`.
