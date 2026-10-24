# Calibration validation, 2026-10-20

Same hardware/model/executable as the parent report. `validation.json` pins the
Python source hashes. No native engine source was changed by this calibration work.

## Passed

- `.venv/bin/python -m unittest tools.test_calibrate tools.test_calibrate_dflash2 tools.test_setup_dflash2 tests.terminal.test_calibration`: **97 tests passed**. See `calibration-targeted-release-tests.log`. Tests cover joint selection, residency drift, subthreshold/held-out rejection, EOS normalization, cache reuse, loaded block limits, OOM candidates, cancellation, original-profile restoration, setup persistence and Terminal report validation.
- Python compilation of `tools/calibrate.py`, `tools/calibrate_dflash2.py`, `setup.py`, `strata_control/calibration.py` and both report helpers; whitespace checks on source/docs/JSON. Raw logs and the discarded patch retain their original whitespace.
- Main native benchmark reanalysis: `python bench/results/2026-10-20-dflash2-fast-prompt/analyze.py bench/results/2026-10-20-dflash2-fast-prompt`. Output is `fast-prompt-analysis-validation.log`.
- `.venv/bin/python ../Strata-data/dflash2-long-next/calibration-verified-gpu-smoke.py`: exit **0**, four actual engine starts, no failed candidates or reused prompt tokens; source hashes verified unchanged from start to finish. See the retained script, JSON, stdout and four native logs. A local `strata-dflash2-fast-prompt.json` supplies the existing model paths and selected profile from `../profile.json`; the smoke reduces expert slots to 2700 for headroom. No config is written.

The GPU smoke bounds the run to 32 output tokens, a 768-token document body (793
including chat format), one allocation profile, block/floor search and one held-out
confirmation pair. It uses the production document factory. The candidate improved
the balanced confirmation score by only 1.03%, so settings correctly remained
unchanged. This validates the real protocol, search/restarts and rejection path;
it does not establish a calibration speedup. Its timing was collected alongside the
broad Python tests and is not used in the performance table.

## Failed attempts retained

- `calibration-gpu-smoke-failed.*`: an initial smoke cropped the old long fixture
  instead of constructing a valid chat document. It generated one EOS token, and
  the incomplete-sample guard rejected every profile. The fixture was corrected.
- `calibration-final-gpu-smoke.*` and six native logs: a subsequent two-allocation
  smoke completed measurement and confirmation without failed profiles, but its
  final source-integrity assertion failed because Python source was edited while
  it ran. Its source hashes remain in the JSON. It is superseded by the frozen-source
  successful smoke above and is not claimed as final-source validation.
- Broad command below: **785 tests, 9 failures, 16 errors, 48 skipped**. All 16 errors
  are missing `rich`/`textual`. Failures concern discovery, shell parsing, monitor
  lifecycle, import layout inside `.venv`, private-file checks and process lifetime.
  These failures were not resolved or attributed by a clean-parent rerun in this
  work. See the complete `calibration-frozen-broad-tests.log`; no full Terminal
  suite pass is claimed.

```sh
.venv/bin/python -c 'import unittest; loader=unittest.TestLoader(); suite=unittest.TestSuite([loader.loadTestsFromNames(["tools.test_calibrate", "tools.test_calibrate_dflash2", "tools.test_setup_dflash2"]), loader.discover("tests/terminal")]); result=unittest.TextTestRunner(verbosity=1).run(suite); raise SystemExit(not result.wasSuccessful())'
```

## Not run

Full production DFlash2 calibration (128 output tokens, 2048/4096-token document
bodies, three confirmation pairs), other hardware/backends/Windows, simultaneous
serving and long-context stress. The successful reduced smoke does not replace
these checks. The PR remains a draft.
