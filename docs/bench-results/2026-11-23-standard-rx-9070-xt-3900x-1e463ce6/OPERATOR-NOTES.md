# Operator notes for run `20261123T043305Z-1e463ce6`

The generated `README.md`, `results.json` and `outputs.jsonl` are unedited. Hardware, software and configuration are
the same as in `../2026-11-23-standard-rx-9070-xt-3900x-bbb73d60/OPERATOR-NOTES.md`.

- Command: `strata bench run --profile launch-iq3_s-256k --no-start --track default`, against the same running engine
  about five minutes after run `bbb73d60`, without restarting it.
- **Status nonconforming, as the runner reports.** The 4K and 16K prompts were still in the server's prompt cache
  from the previous run ("unexpected reuse of the user payload"), so their prompt and decode results are excluded.
  Included only to keep every outcome; it is not a standard result.
- Still valid as observations: short-prompt decode 60.4 (53.1-66.1) tok/s n=3, and the sanity checks passed 12/12.
- The server had answered 117 requests before this run (95 interactive + 22 from run `bbb73d60`).
- Engine log: not captured (`--no-start`).
