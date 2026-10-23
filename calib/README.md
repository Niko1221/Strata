# jev-port calibration (M3.2 + per-tuple, donor calib/ parity)

Donor reference: jevcpp/calib/ (fit.py from B1.1 log rows; b13_eval.py the eval
harness; calib.json the shipped fit, tuple `p-jevcpp-qwen38-27b|q4_k_xl|timber|v1|V|all`,
T_choice 1.5861 on the 27B).

This tree:
- bench_suite.py     the live calibration suite (text/image/real). Fits Platt (a,c)
                     for noul and temperature for choice/score on dev splits,
                     evaluates on held-out test with standard top-label ECE15 and
                     mean NLL; writes per-item logs (jsonl) for adversarial audit.
                     Guards: refuses Platt slope a<=0 (labels/prompt inverted) and
                     flags T at search boundaries.
- convert_benchmarks.py  POPE adversarial / CV-Bench / CountBenchQA -> item JSONs
                     (banned-field selftest: no `prompt`/`text`/`rationales` in
                     requests; sha256 image manifest; dev/test image-disjoint;
                     POPE sampled by image, exact 50/50).
- calib-L.json / calib-D.json   the active per-tuple calibrators, read by the
                     server at startup (calib-<mode>.json per readout letter;
                     the donor's global calib.json still works as the text tuple).

Refit procedure (the script IS the calibration):
  1. python calib/bench_suite.py <base> <key> <outdir> text 60 100    (or `real`)
  2. take the printed fits; write calib/calib-<L|D|S>.json
  3. POST /admin/calib (or restart the preset) to hot-swap
  4. verify per-answer `calib` ids + a held-out ECE check

Fits shipped (2026-10-15, QFN IQ3_XXS, corrected top-label harness):
- L (text): a_noul 0.6127 c_noul 0.6539 T_choice 1.0557 T_score 2.263
            (BoolQ/ARC/SST-5 dev 60 each; test: noul ECE15 .084->.033, score
            .301->.154, NLL down on all three)
- D (image direct): a_noul 1.0571 c_noul -0.172 T_choice 1.5456 T_score 0.6555
            (POPE-adv/CV-Bench/CountBenchQA dev; test: POPE 86.7->88.3%,
            ECE15 .122->.081)
- S (image summary): identity until its own dev split is run (per-tuple swap
  loads calib-S.json when present).

## Status 2026-10-23 (what re-measurement changed)

The three shipped files are still correct, but three claims in the notes above do not survive a
held-out re-run and are corrected here:

1. **The 2026-10-15 test numbers are calibration-saturated.** They were evaluated on the same dev
   split the fits came from. Re-running the same suites on the current binary: BoolQ .910 -> .910
   (raw ECE .049), ARC 1.000 -> 1.000 (ECE .007), SST-5 .520/.530 with the refit HURTING test ECE
   (.127 -> .140). The readout improved underneath the calibrator; the headroom on these three
   tasks is now ~0. Refits come back near identity (a=1.009 c=.0037 / a=1.032 c=.062).
2. **Image tuples are at their ceiling.** Refitting on top of calib-D asks for T=.985 / T=.984 and
   a=1.066, c=-.003. Paired against identity-served on the same 100 test items, the shipped
   calibrator changes no decision and improves Brier by .0046 (POPE) / .0315 (CV-Bench) / .0209
   (CountBench), ECE by -.0075 / +.0100 / -.0583.
3. **A fitted noul calibrator is NOT accuracy-neutral.** The residual monotonicity argument holds
   for temperature only. Platt's offset c is a decision threshold: a fit on banking77 moved
   held-out accuracy .670 -> .800 with the boundary at raw p ~.94, i.e. threshold tuning on a
   balanced prior, not better probabilities.

Refit rules that follow, and that the scripts enforce or warn about:

- Fit per (tuple, task, mode) and evaluate on items the fit never saw. If the dev split is
  saturated (acc 1.000, NLL ~.00x) T is unidentifiable - a fitter that reports 0.05 or 5.03 there
  is reporting its own search bound, not a temperature.
- Never fit on calibrated output. Move `calib-*.json` aside to serve identity for a raw->fit
  measurement, or invert the known transform exactly first.
- Never carry c across tasks. The slope is even approximately portable; the offset is a prior.
  A banking77 fit cost POPE 3.3 points and GQA 8.3 points; zeroing only c restored both.
- `STRATA_JEV_CALIB_STRICT=1` refuses a file whose keys the loader would silently drop (the
  `a`/`c` vs `a_noul`/`c_noul` trap) and refuses a degenerate T.
