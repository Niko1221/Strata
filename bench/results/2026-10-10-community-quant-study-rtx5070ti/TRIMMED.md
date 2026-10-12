# Curated evidence and exclusions

## Included

- Canonical ALL-METRICS, 20 complete attempted-task HumanEval+ files (gzip JSON), all 18 pinned 92 x 5 tool arms, all 18 finite three-depth performance profiles, baseline knowledge/needle notes plus the four new notes, baseline debugging matrix and 25 new debugging records with their grading summary.
- Original Gyro-M done/manifest/closeout records and CTest/audit logs: invalid-slice status, missing audit-step names and failed parity test are retained, not repaired away.
- Ten shared charts, final PDF/HTML, generated README/PR draft, claim/recommendation registry, final scoreboard, table inventory, numerical/page proof and portable build/audit scripts.

## Excluded from scoring/package

- Gyro-S rc1 tool/proxy runs: no usable native forced tool support and contaminated transport; tool fields are n/a.
- All Gyro-S synthetic speed files, including v2: malformed million-token/s/null decode cells; no usable throughput baseline. Retired earlier-engine and superseded speed files are not included.
- Aborted first Gyro-M campaign, pilots, standalone effort probe, other-engine runs and the separate Swift v2.8.0 / 97-scenario dataset. The final rerun evidence, not the pilots, backs the scores.
- Old original rollup/summary/wall intermediates and obsolete chart/scripts: no stale second recommendation source is shipped.
- Model weights, expert/MTP packs, full build/server logs and rig-specific lifecycle/service scripts. The extractor requires the study rig and is not needed to rebuild from the packaged canonical store.

## Redaction and reproducibility boundary

Credentials are [REDACTED]; home/host/LAN paths are placeholders. JSON object structure, scenario points, task grades, completion status and timings are retained. Gzip compression is lossless after this explicit privacy sanitation; original unredacted measurement files were not modified. The extracted store retains measurement-source hashes; SOURCE-MANIFEST hashes the actual sanitized distributed bytes.

The portable compiler reads data/ALL-METRICS.json only, rebuilds ten charts, the PDF and the GitHub report, and generates one registry for the final board and recommendations. Auditors verify task-level denominators, all raw five-trial arrays, profile source values, all common-table rows, whole-table PDF pagination and package completeness. No engine or production settings are changed.
