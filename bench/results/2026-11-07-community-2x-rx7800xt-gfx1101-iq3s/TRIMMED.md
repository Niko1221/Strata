# What was trimmed from this folder

- The engine log itself (`/home/dom/Strata/strata-iq3_s.log`, ~86 MB) - only the decode-timing lines are kept, in `decode-timing-excerpt.txt`, and the per-request accounting lines are quoted in the README.
- Per-arm raw JSON for the 26-arm screen (the machine-readable summary is `summary6.jsonl`, the human table is `screen-arm-table.md`).
- `house-r1-*` / `house-r2-*` files: the two interleaved rounds that were meant to compare two configurations but compared one configuration with itself (the deploy bug in `PROMOTED.json.incident`). Kept as variance data under `variance-*.json` and as raw rounds; they are not quoted as effects.
- The QA transcripts for the arms that changed the arithmetic path are kept as `qa-*.json`; the six-prompt fingerprint table is `qa-fingerprints.md`.
- No model shards, no tokenizer, no expert profile, no screenshots (they are hashed and sized in `model-provenance.json`).
