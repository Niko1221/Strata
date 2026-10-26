# The batch path's checks and windows (dsv41)

Each of these ran against the box (10.188.27.29) and their results are in `docs/DSV41_HANDOFF.md`.  The shell scripts stop
production, start the test server on 8099 with `build/strata-dsv41.batch` and `strata-batch1.json`, run their check,
and always restore production (a trap), so a run is ~4-6 minutes of downtime.

| file | what it does |
|---|---|
| `../batch_probe.py` | the engine's batch protocol with no HTTP: BGEN/BADM/BT/BDONE/BSTOP, N slots, GEN-vs-BGEN token equality |
| `batch_http_check.py` | the batched server over HTTP: the reference answer against production's, three clients interleaving, one-token requests, a client leaving mid-stream; `timing` mode prints wall/first tokens/handovers |
| `batch_yield_check.py` | a long prompt with a short one behind it: the short one's first token and wall time (BYIELD and the read's slices) |
| `batch_turn_check.py` | a two-turn conversation with another request between the turns (the slot's kept conversation) |
| `../batch_fuzz.py` | rounds of concurrent requests, a third abandoned mid-stream, a canary after each |
| `../batch_soak.py` | rounds of concurrent streaming requests, all answered |
| `../batch_seq_parity.py` | M2's per-sequence regions: the same prompts on sequence 0 and on sequence 1 (`STRATA_DS4_VSEQ`), two engine loads, token-for-token |
| `windows/batch_window.sh` | the probe (`--slots 2`) then the HTTP check |
| `windows/batch_fuzz_window.sh` | the fuzz |
| `windows/batch_soak_window.sh` | the soak |
| `windows/batch_turn_window.sh` | the two-turn check then the HTTP check |
| `windows/batch_nseq_window.sh` | the probe and the HTTP check with `STRATA_DS4_NSEQ=2` (two window rings) |
| `windows/batch_seq_window.sh` | the parity check: sequence 0 and sequence 1 on the same prompts (no HTTP) |

The engine binary the windows use (`build/strata-dsv41.batch`, built from this branch) and the config
(`strata-batch1.json`, the production config plus `--batch N --slot-burst K --read-slice 1024`, port 8099) are made by
`/data/works/mk_batch_cfg.py`; production keeps `build/strata-dsv41` (the stock binary) and port 8080.
