# Notes on these artifacts (what we changed and why)

1. **`machine` blocks in `results.json` were hand-corrected.** Run with
   `--url` (the bench client on a second machine), `collect_machine()` blends
   client and server fields: the originals recorded `ram_gib: 3.8` (the client
   VM), `gpu_driver: null`, and one combined `gpu` string. Corrected to the
   host's measured facts: `gpu` split into the two cards, `gpu_driver
   615.71.09`, `cuda 13.4`, `ram_gib 61.9` (visible to the nspawn container;
   64 GB installed). Reported upstream in #2062.
2. **Environment variables were not captured per run.** 0.1.43 does not log
   `STRATA_*` state, so the C run's `STRATA_ADAPT_OVERLAP=0` is attested by our
   run record, not provable from these artifacts. The D run's
   `STRATA_ROUTE_TAIL_SKIP=0` **is** auditable from the logs: A's log carries
   `route tail skip: N missed experts skipped` lines (last: 98,217), D's log
   carries none. PR #2011 adds the logging.
3. **One measurement day:** all six speed runs and the needle suite ran in one
   continuous session on one machine, 2026-10-11 22:39 to 2026-10-12 00:46
   local (A at 22:39, D and the needles after midnight). No reboots, no other
   workloads, same config tree, same pinned profile.
