# scripts/ — measurement and arm-switching tools (jev M3.1)

Everything here measures the live preset, or fits something from a measurement. None of it is
imported by the server; `serve/` is the shipped surface.

## Switch the live preset (do this through the script, never by hand)

| tool | what it does |
|---|---|
| `jevport-arm.sh <c\|b\|h> [--calib <file>\|identity\|none] [--labels upper\|digit-lower]` | rewrites the preset env in `config.yaml` (anchored on the preset key — block-splitting clobbered a neighbour preset once), restarts llama-swap, then VERIFIES the live process env and smoke-tests. A config reload alone does not restart an already-running preset, so "just retry" silently measures the old arm. |
| `calib-reload-live.sh` | `POST /admin/calib` from *inside* the container (the published llama-swap route 401s: llama-swap wants its own key in `Authorization`), then a live smoke to prove `calib=platt_v1` rather than identity. |

Restore the box after every experiment: `jevport-arm.sh c` then `calib-reload-live.sh`. Confirm the
serve process has no `STRATA_JEV_ROWS`/`HYBRID`/`LABELS` env and that a smoke says `platt_v1`.

## Labelled evaluation (accuracy, Brier, ECE)

| tool | what it does |
|---|---|
| `build_mf_set.py` | banking77 → labelled multi-field jsonl (`is_card` + `group`), balanced 50/50, `--exclude` for a train/held-out split that shares no utterance |
| `eval_mf_set.py run\|compare` | runs the set against `/v1/systemone`, reports accuracy / Brier / both-right / mean confidence, McNemar between two runs |
| `ab_arms_mf.py` | paired A/B of two arms over the same labelled items |
| `build_gqa_set.py` | GQA → labelled image `noul` jsonl: balanced yes/no, stride-sampled (not the head of the split), sha256 manifest per image |
| `run_gqa.py` | image-direct GQA run; dev/test split, Platt fit, identity-vs-fit, incremental `.rows.jsonl` + resume by id (a crash must not cost a 30-minute run) |
| `ab_arms_gqa.py` | paired arm comparison that inverts a known served transform back to raw first (round-trip verified to 1e-16) |
| `order_test.py` | field-order reversal with `--alternate` (half the items forward-first) — without alternation the cold first prefill looks like a 3× order effect |
| `label_ab.py run\|compare` | label alphabet A/B; `compare` inverts the served T, then reports decision parity, mean \|Δp\|, and each alphabet's own bounded-optimum T |
| `ab_image_calib.py` | paired image-suite A/B computed from the *served* probabilities of two item logs |

## Calibrator fitting

| tool | what it does |
|---|---|
| `fit_calib_mf.py` | Platt (noul) + temperature (choice) on labelled rows, 5-fold CV, ECE/Brier/NLL before vs after; inverts the currently-served calibrator before fitting so the params are installable |
| `fit_tuple_from_log.py` | the same, from one `real-item-log.jsonl` (per-tuple: L/D/S) |
| `cross_domain_calib.py` | fit on one task, evaluate on others — the tool that showed a calibrator does not transfer; warns when Newton diverges instead of printing a runaway `a` as a fit |
| `check_calib.py` | validate a calibrator file without starting the server |

## Measurement hazards (each one cost a wrong number once)

1. **Served probabilities may already be calibrated.** Fitting a dev split on top of a calibrated
   output compounds the transform and the params are not installable. Serve identity, or invert.
2. **A saturated split cannot identify T.** acc 1.000 / NLL .004 makes the fit run to its search
   bound (0.05, 5.03). Bounded search + a `flat`/`boundary` flag; if it is flat, report identity.
3. **Platt's offset is a threshold, not a shape.** It can move accuracy. Quote an accuracy delta
   from `noul` as a threshold/prior decision, never as calibration quality.
4. **The offset does not transfer across tasks; the slope roughly does.**
5. **Alternate condition order.** The first request in a run pays the cold prefill (946 ms vs
   378 ms), which will masquerade as any effect you are testing if the conditions are blocked.
6. **Variance.** This box is ~±15% run to run. Anything under that, on n≈100, is not a result.
7. **Never quote the suite's own raw/cal columns when a calibrator is installed** — they re-apply a
   dev fit on top of served probabilities. Use the paired comparator on the item logs.
