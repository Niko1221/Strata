# Trimmed files

Only the report, `REPRODUCE.md` and this note are in the pull request. Everything below exists on
the reporter's machine and is available on request; the hashes in `REPRODUCE.md` pin the speed and
correctness outputs the tables were built from.

| file | why |
|---|---|
| `speed-strata-v43-fast.{out,jsonl}`, `speed-strata-q4kxl-final.{out,jsonl}` | raw per-request rows and generated text; hashed in `REPRODUCE.md` rather than committed |
| `strata-v43-fast-{std,hard}.out`, `strata-q4kxl-final-{std,hard}.out` | ctxbench per-task grader output; the failed tasks are tabulated in `README.md` |
| `speed.py`, `ctxbench.py`, `iotrack.py` | submitter-local harness scripts, not published with this report and not part of Strata |
| engine logs, health dumps, per-stage chain logs | logs |
| engine binaries and the GGUF packs (~191 GB) | reproducible from the pinned HF revisions in `README.md` |
| `iotrack` disk-I/O samples | collected but **not used as a result** in this report |

## Not measured at all

Stated so the scope is not mistaken for the fuller gfx1151 reports in this directory, which also
publish needle recall, parallel/QFUSE behaviour, thermal telemetry and a `provenance/` build
manifest. None of those dimensions are covered here:

- **needle recall** — not run
- **parallel requests / QFUSE** — not run
- **thermal / power telemetry** — not collected
- **build manifest, `ctest` log, loaded-library provenance** — not captured; the engine versions and
  ROCm build strings in `README.md` were read from the running installs, not from a manifest
- **repeats of the long-context rows** — single run each (short rows were run 3x)
