# Metal workflow finding correction, round 1

Scope: only `.github/workflows/metal-build.yml`; this file is staged for controller review. Reference review tree: `dad73653a2c8fe6f0a3568fad538ec5b91d56614`. No commit, build, test weakening, dependency installation, or publication was performed.

Cause: the Build step label had been concatenated to the Configure command, leaving two `run` keys in one step. The correction separates these steps and preserves vendored ggml. It aligns the existing backend workflow policy: `macos-26-intel`, `CMAKE_OSX_ARCHITECTURES=arm64`, `GGML_NATIVE=OFF`, all three `-mcpu=apple-m1` flags, genuine `strata` and `strata_ds4` targets, and `lipo -archs` checks. Build parallelism is explicitly five.

Commands executed from `H:/Strata/worktrees/upstream-sync-20261116`:

```powershell
actionlint -shellcheck= -pyflakes= .github/workflows/metal-build.yml
git diff dad73653a2c8fe6f0a3568fad538ec5b91d56614 -- .github/workflows/metal-build.yml
git add -- .github/workflows/metal-build.yml
git diff --check -- .github/workflows/metal-build.yml
Get-FileHash .github/workflows/metal-build.yml -Algorithm SHA256
```

Validation: actionlint passed with exit 0 and no diagnostics. Git staging and diff whitespace checks passed. The corrected workflow SHA-256 is `4f2120b46b41c0c4584a1f5bafdd75b2835922c5eec5222827586da24fcdf1e5`. The exact diff is retained as `metal-fix1.diff` beside this report; its SHA-256 is recorded in `metal-fix1.diff.sha256.txt`.

NOT_RUN: actual hosted macOS compilation and link; this Windows PC cannot establish those results. Actionlint ran with optional shellcheck and pyflakes disabled, because the scoped validation is workflow syntax and structure and no Python step was modified. Independent controller review remains required.

Self-review-event: qa_defect. This narrow workflow fix addresses the observed malformed step without changing SOP or governance. No broader automatic change is justified by this one finding.
