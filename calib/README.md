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
