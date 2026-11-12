# NCM feasibility study: recorded files, 2026-10-19

These are the raw files behind [docs/NCM.md](../../NCM.md). Machine: NVIDIA GeForce RTX 5070 (12 GB),
AMD Ryzen 9 7900X, Linux 7.0.0-38. Model: Swift 1.5 IQ2_XS with MTP.

| File | What it is |
| --- | --- |
| `standalone-probe.json` | Output of `tools/ncm_probe.py` on NCM's released TinyStories model |
| `target/results.json` | Output of `tools/ncm_target_bench.py`: ten engine runs, IDs, speeds, routing unions |
| `target/<prompt>.<mode>.command.json` | Exact argument list of each engine run |
| `target/<prompt>.<mode>.stdout`, `.stderr` | Engine output of each run |
| `target/<prompt>.prompt.txt`, `.prompt.ids` | Prompt text and its rendered token IDs |
| `target/<prompt>.oracle.ids` | Greedy reference output, used as the oracle draft |
| `target/routing-bin.sha256` | SHA-256 of the four `--dump-routing` traces (1.1 MB, not kept here) |
| `terminal-tests.log` | `python -m pytest tests/terminal -q` on the same day: 834 passed, 18 failed, 17 skipped |

Local paths in these files were rewritten after the run: `$STRATA` is the Strata checkout, `$DATA` the
Strata data folder (`../Strata-data`), `$OUT` the output folder (`/tmp/strata-ncm-target-v2` during the run).
Nothing else was changed. The per-window routing unions in `results.json` were computed from the traces;
`tools/ncm_target_bench.py` recomputes them from new traces.
