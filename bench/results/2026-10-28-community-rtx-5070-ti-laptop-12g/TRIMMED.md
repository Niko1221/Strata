# Trimmed from this submission

Kept out of the folder (2026-10-28 update):

- Full engine server logs from the restart/bench sessions (private paths and unrelated chat traffic; the engine timing lines that matter are in the per-run JSONs, verbatim in each `line` field).
- The runner script that restarted the engine and drove the three tiers (private paths; the protocol is described in README.md Method).
- The machine's full 2026-10-15..24 benchmark history (40+ A/B JSONs, flag sweeps) — quoted as summary numbers in the 2026-10-24 report README, which this folder updates.
- Model GGUF/pack files (~111 GB) and the machine's engine install tree (needle runs were executed from the installed 0.2.6-era tree, unmodified; the needle script is included).
- Daily-driver telemetry collected by the machine's own control panel (not part of this report's method).
