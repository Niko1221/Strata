# Scripts and configuration in this report

Everything here is a **verbatim copy** of a file that was already present in the measuring machine's working
tree (either a tracked repository file or a local untracked file from the measurement session). Nothing was
edited for publication. Paths in the JSON configs are the measuring machine's local paths and are kept as-is so
the arms can be reconstructed exactly; there are no credentials, API keys or tokens anywhere in this folder
(the server ran on loopback without `--api-key`).

| File here | Copied from | Why it is here |
| --- | --- | --- |
| `bench_decode.py` | `tools/opt/bench_decode.py` | The probe that produced every file in `../raw/`. **It is a local script, not an upstream file** — `tools/bench_decode.py` does not exist at this commit and `tools/opt/` is untracked. It is included verbatim so a reviewer can read exactly how each number was parsed, and it documents its own caveats (for example that `STRATA_VERIFY_PROFILE=1` changes execution and must not be mixed with production arms). |
| `strata-iq3_xxs.json` | `strata-iq3_xxs.json` in the repository root | The live engine config for the last arms. Git-ignored upstream (`/strata-*.json`), which is why a copy is needed to reproduce. |
| `cfg-ab/base.json` | `tools/opt/cfg-ab/base.json` | The **base** arm: `base.json` + the three shared env keys. |
| `cfg-ab/shstream0.json` | `tools/opt/cfg-ab/shstream0.json` | Adds `"STRATA_SH_STREAM": "0"` to the base env. |
| `cfg-ab/hcplain.json` | `tools/opt/cfg-ab/hcplain.json` | Adds `"STRATA_HC_SPLIT": "0"`. |
| `cfg-ab/dbstore.json` | `tools/opt/cfg-ab/dbstore.json` | Adds `"STRATA_DOORBELL_STORE": "1"`. |
| `cfg-ab/coherent.json` | `tools/opt/cfg-ab/coherent.json` | Adds `"STRATA_VERIFY_COHERENT": "1"`. |
| `cfg-ab/devplan.json` | `tools/opt/cfg-ab/devplan.json` | Adds `"STRATA_VERIFY_DEVICE_PLAN": "1"`. |
| `cfg-ab/shstream0-minp0.json` | `tools/opt/cfg-ab/shstream0-minp0.json` | `STRATA_SH_STREAM=0` **and** `--spec-min-p 0` instead of `0.70`. |
| `cfg-ab/base-prof.json` | `tools/opt/cfg-ab/base-prof.json` | Adds `STRATA_VERIFY_PROFILE=1`. **Not used in any reported arm** — kept only because `bench_decode.py` warns that the stage profiler changes execution. No raw file corresponds to it. |
| `cfg-ab/show_arm.py` | `tools/opt/cfg-ab/show_arm.py` | Prints an arm config with the base keys and the delta highlighted; the quickest way to confirm that two arms differ in exactly one variable. |
| `drivers/run-final-validation.ps1` | `tools/opt/run-final-validation.ps1` | Produced the `final-*` and `final-long-*` arms, including the trailing base anchors. |
| `drivers/run-w2-sweep.ps1` | `tools/opt/run-w2-sweep.ps1` | Produced the `w2-*-2` sweep A arms. |
| `drivers/run-ctx-det.ps1` | `tools/opt/run-ctx-det.ps1` | Produced the `ctx-short-*` arms and the determinism text runs. |
| `drivers/run-verify-minp0.ps1` | `tools/opt/run-verify-minp0.ps1` | Produced the `w2-shstream0-minp0` arm. |
| `drivers/probe-199k.ps1` | `tools/opt/probe-199k.ps1` | The **failed** long-context probe. Included so the failure in the report is reproducible rather than merely asserted. |

## Reading the arm configs

Every `cfg-ab/*.json` differs from `cfg-ab/base.json` in exactly one place — one extra key inside `env`, or one
changed `args` value — plus the `log` path, which only decides where the engine's timing lines are written.
That is the single-variable property the report's A/B rests on; `show_arm.py` prints the difference.

## What is deliberately not here

- **No model files, no native pack, no `experts.bin`, no MTP runtime.** They total well over 70 GB and the
  contribution guide asks for them to stay out of the PR. The pack was built with the repository's
  `tools/iq_pack.py`, but the exact invocation and the resulting hashes were not recorded, so the pack is *not*
  bit-reproducible from this report. See limitation 11 in the README.
- **No engine logs.** The drivers parse `tools/opt/logs/*.log`, and those files total about 550 KB of raw
  engine output including start-up banners with local paths. Every timing number the report quotes is already
  preserved per request in `../raw/*.json` as the engine's verbatim `raw_line`, so the logs add bulk rather
  than evidence. Request the engine logs separately if a reviewer wants them.
- **No `check-output.txt`, `rocm-install.err`, `docs/opt/` or the `*.bak-*` config backups** from the measuring
  machine. They are local scratch files, not part of this measurement.

## Reproducing one arm

From a repository root that has the model shards, pack and MTP runtime in place, with the local
`strata-iq3_xxs.json` style config pointing at them:

```powershell
# 1. an idle GPU: no strata.exe, nothing listening on 8080, no live tools/opt/gpu-window.json lock
Get-Process strata -ErrorAction SilentlyContinue
Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue

# 2. start the server on the arm's config and let the harness stop it again
python tools/opt/bench_decode.py `
  --start-server "python serve/server.py --engine strata --config tools\opt\cfg-ab\shstream0.json --port 8080" `
  --gpu-lock --gpu-lock-wait 180 --log-path tools\opt\logs\shstream0.log --arm-label my-shstream0 `
  --prompt-tokens 32768 --max-tokens 128 --repeats 3 --seed 1234 `
  --warm-cold "cold engine; rep1 fresh prompt+cold expert cache, rep2-3 warm" `
  --out-json out\my-shstream0.json --out-md out\my-shstream0.md
```

`--prompt-tokens 32768` yields an actual prompt of 132,886 tokens; the knob is a size selector, not a count
(see `../raw/FIELDS.md`). Swap `shstream0.json` for `base.json` to measure the anchor. Then repeat the base arm
**after** the variant: if the two base arms differ by more than 5%, the window is invalid and the comparison
should not be used.

The four drivers in `drivers/` do exactly this loop with the idle guards and retry policy already written in.
They are the shortest path to repeating this report; read them before running anything, because they contain
force-kill paths guarded by the card lock.