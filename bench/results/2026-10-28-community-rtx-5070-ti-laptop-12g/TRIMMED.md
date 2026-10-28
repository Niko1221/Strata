# Trimmed from this submission

Kept out of the folder (2026-10-28 update):

- Full engine server logs from the restart/bench sessions (private paths and unrelated chat traffic; the engine timing lines that matter are in the per-run JSONs, verbatim in each `line` field).
- The runner script that restarted the engine and drove the three tiers (private paths; the protocol is described in README.md Method).
- The machine's full 2026-10-15..24 benchmark history (40+ A/B JSONs, flag sweeps) — quoted as summary numbers in the 2026-10-24 report README, which this folder updates.
- The superseded engine snapshots measured earlier on 2026-10-28 (the engine 0.1.50 data first pushed to this folder this morning and the 0.1.46 rows it was compared against; both stay in this branch's commit history, and their medians are re-quoted inside the 0.1.52 README).
- Model GGUF/pack files (~111 GB) and the machine's engine install tree (needle runs were executed from the installed 0.2.7 tree, unmodified; the needle script is included, hash-checked against `tools/needle_bench.py` in that tree).
- Daily-driver telemetry collected by the machine's own control panel (not part of this report's method).
