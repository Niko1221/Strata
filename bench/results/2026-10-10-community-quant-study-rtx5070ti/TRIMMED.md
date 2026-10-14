# Curated evidence and exclusions

## Included

- Canonical ALL-METRICS, 20 local attempted-task HumanEval+ files (gzip JSON), all 20 pinned local 92 x 5 tool arms, and the recorded three-depth performance profiles. Gyro-S rc2 engine-log-audited profiles are included; Gyro-M speed observations remain excluded from comparison because their endpoint route is unverified. The separate sanitized GPT-6-luna MAX and GPT-6.1-SOL ultra OpenAI-hosted references are shipped as `data/hosted-api-reference.json` and `data/hosted-api-reference-sol.json`, respectively, and integrated into comparable tables/charts; neither is a local inventory entry or recommendation. SOL uses first-upload scores pinned to source commit `a48292f5766f66efcdb1e23396569b0907c44c04`; no prompts, outputs, or second-upload regrades are included.
- Gyro-S required HE+/TEB low/xhigh scoring slices and restoration closeout passed eligibility, but the overall battery is `aborted_or_invalid` because auxiliary `plugin-mmlu` is invalid and the controller exited. Only the validated required slices are used. Original Gyro-M done/manifest/closeout records and CTest/audit logs are also retained, including invalid-slice status, missing audit-step names and failed parity test.
- Ten shared charts, final PDF/HTML, generated README/PR draft, claim/recommendation registry, final scoreboard, table inventory, numerical/page proof and portable build/audit scripts.

## Excluded from scoring/package

- Historical Gyro-S rc1 tool/proxy runs: no usable native forced tool support and contaminated transport; they are not blended with the separate scored rc2 tool arms.
- Legacy Gyro-S rc1 synthetic speed files, including v2: malformed million-token/s/null decode cells; they are excluded. The distinct rc2 full-battery engine-log speed audit is included and kept separate from the retired files.
- GPT-6-luna MAX matched decode performance is unavailable: invalid partial rates, diagnostic bursts and exploratory tg=1024 pilot are not published or substituted.
- GPT-6.1-SOL ultra has no comparable GPU, throughput or latency measurements. Only the first-upload Five Bugs aggregate (25/25 visible, 20/25 hidden) is included; second-upload post-feedback corrections/regrades are excluded from the score and package.
- Aborted first Gyro-M campaign, pilots, standalone effort probe, other-engine runs and the separate Swift v2.8.0 / 97-scenario dataset. The final rerun evidence, not the pilots, backs the scores.
- Old original rollup/summary/wall intermediates and obsolete chart/scripts: no stale second recommendation source is shipped.
- Model weights, expert/MTP packs, full build/server logs and rig-specific lifecycle/service scripts. The extractor requires the study rig and is not needed to rebuild from the packaged canonical store.

## Redaction and reproducibility boundary

Credentials are [REDACTED]; home/host/LAN paths are placeholders. JSON object structure, scenario points, task grades, completion status and timings are retained. Gzip compression is lossless after this explicit privacy sanitation; original unredacted measurement files were not modified. The extracted store retains measurement-source hashes; SOURCE-MANIFEST hashes the actual sanitized distributed bytes.

The portable compiler reads data/ALL-METRICS.json only, rebuilds ten charts, the PDF and the GitHub report, and generates one registry for the final board and recommendations. Auditors verify task-level denominators, all raw five-trial arrays, profile source values, all common-table rows, whole-table PDF pagination and package completeness. No engine or production settings are changed.
