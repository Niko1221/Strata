# Calibrating the jev decision surface — per-task refit

**Read this if** your answers come back more confident than they deserve, or if the
yes/no field is skewed toward one class. The short version: **calibration is only valid
on the task family it was fitted on**, so every distinct task gets its own fit.

---

## 0. TL;DR — the recipe

```bash
# 1. items with GOLD labels (see §2; 100-500 items, balanced, ~25 is the floor)
#    docs/examples/calib-items-example.jsonl shows the exact shape

# 2. collect readouts + fit + gate, in one command
scripts/refit_calib.py fit \
    --base http://127.0.0.1:8080/upstream/p-my-arm --key "$KEY" \
    --set my-items.jsonl \
    --task my-task \
    --criteria card top_up transfer cash other \
    --out-dir /cfg/strata/.../calib-my-task

# 3. serve with the matching task tag and calib dir
#    (llama-swap preset env, see §5)
STRATA_JEV_TASK=my-task
STRATA_JEV_CALIB_DIR=/cfg/strata/.../calib-my-task

# 4. hot-reload without restarting (needs STRATA_JEV_ADMIN_KEY set)
curl -X POST -H "Authorization: Bearer $STRATA_JEV_ADMIN_KEY" \
     http://127.0.0.1:$PORT/admin/calib
```

If the fit fails the gate, **do not ship it** — identity is better than a wrong fit
(measured, see §1).

---

## 1. Why per-task, in numbers

The server loads **one fit per readout letter**. `calib-L.json` is applied to *every*
text task. A fit is a domain statement: the intercept encodes "how often is the answer
yes in *my* data".

Measured on 500 labelled banking77 items, one GPU, same binary, only the calibrator
differing:

| arm | is_card acc | Brier | ECE |
|---|---|---|---|
| shipped `calib-L` (fitted on BoolQ60/ARC60/SST5-60) | 0.664 | 0.2325 | 0.275 |
| identity — **no calibration at all** | 0.760 | 0.1983 | 0.218 |
| refit on this task (intercept only) | 0.816 | 0.1494 | 0.148 |
| refit on this task (full Platt) | **0.816** | **0.1315** | **0.109** |

Three rules follow, and they are the whole point of this document:

1. **Doing nothing beat what shipped** (+9.6 points). A cross-task fit is worse than no fit.
   That is why the loader now fails closed: a fit whose recorded task does not match the
   runtime task is not applied, and the response says `"calib": "identity"`.
2. **The intercept is the decision lever.** Intercept-only ties full Platt on accuracy;
   the slope only buys Brier/ECE. A handful of labels gets you the decision win.
3. **It is cheap.** Held-out accuracy by dev-set size: 25 items → 0.824, 50 → 0.824,
   100 → 0.822, 250 → 0.809. Twenty-five balanced labels is the practical floor.

Why a "wrong" fit hurts so much here: the readout is close to **ternary**, not
probabilistic. Raw-score bands on that task:

| raw p | n | accuracy |
|---|---|---|
| < 0.1 | 50 | 0.940 |
| 0.1 – 0.9 | 187 | 0.663 (no signal) |
| 0.9 – 0.97 | 15 | **0.333 (anti-correlated)** |
| > 0.97 | 248 | 0.823 |

The good decision rule is "yes only when saturated". An intercept tuned elsewhere
admitted everything above raw p 0.256, so it swept 187 undecided items into "yes" and
manufactured false positives. **Never trust the 0.9–0.97 sliver.**

---

## 2. Building the item set

One JSON object per line, `gold` **required** (never model-derived):

```json
{"id": "ex-0",
 "state": {"utterance": "I lost my card at the airport, how do I get a new one?"},
 "questions": {
   "is_card": {"type": "noul", "instructions": "Does this customer request concern a card?"},
   "group":   {"type": "choice", "instructions": "Which group does this customer request belong to?",
               "criteria": {"card": null, "top_up": null, "transfer": null, "cash": null, "other": null}}},
 "gold": {"is_card": 1, "group": "card"}}
```

Working example: `docs/examples/calib-items-example.jsonl`.

Rules that bite:

- **You only need the fields you fit.** `gold.is_card` for a `noul` fit; `gold.group` for a `choice` fit.
- **Balance the binary field.** 50/50 by class. An unbalanced set makes the intercept meaningless.
- **Labels must be ONE token.** The server refuses a label that encodes to more than one
  token (it would be scored on a truncated path). Keep `criteria` keys short and lowercase.
- **`criteria` order is the choice index order.** It must match what the serving arm uses,
  because `T_choice` and the returned `choice` index are positional. Put the fallback last.
- **Build the gold with a real question.** `"card" in intent_name` is a crude proxy: a
  duplicate-charge query is card-related but its label has no "card", so a correct answer
  scores wrong. It adds noise to every arm equally, but it will drag your absolute numbers down.
- **Never fit on the same items you quote.** The tool splits dev/holdout for you (§3).

---

## 3. Fitting

Two ways to get the per-item readouts:

```bash
# (a) the tool collects them
scripts/refit_calib.py fit --base URL --key "$KEY" --set items.jsonl --task my-task --out-dir DIR

# (b) reuse an arm you already evaluated (eval_mf_set.py writes `rows` with p_yes)
scripts/refit_calib.py fit --eval-json arms/head-raw.json --task my-task --out-dir DIR
```

Both run the same pipeline: stratified dev/holdout split (`--dev-frac`, `--seed`) → fit
Platt `(a, c)` on **dev only** → score identity vs refit on the **holdout** → gate.

Observed output on the banking77 task:

```
dev=250 (125+125 balanced)  holdout=250  seed=7
fitted on dev: a_noul=0.4504 c_noul=-1.5607   (intercept-only c=-3.6074)

HOLDOUT (the fit never saw these):
  arm                      acc    brier     ECE
  identity (no fit)      0.764   0.1949  0.2033
  refit platt(a,c)       0.820   0.1293  0.0794
  refit intercept-only   0.828   0.1445  0.1347
  paired vs identity: fixed 31, broke 17, McNemar exact p=5.95e-02

GATE: refit must beat identity on accuracy AND ECE -> PASS
```

How to read it:

- **`a_noul` / `c_noul`** — `p' = sigmoid(a·logit(p) + c)`. `c` moves the decision
  boundary; `a` sharpens or softens the probabilities. A large **positive** `c` means the
  fit expects a lot of "yes" — the exact signature of a cross-task transplant.
- **GATE** — the fit is written only if it beats identity on **both** accuracy and ECE.
  On failure the tool writes nothing and exits non-zero. `--force` overrides; don't,
  unless you have a specific reason.
- **McNemar p vs identity** — the *incremental* value of calibrating on top of no
  calibration. It can be weak (p≈0.06 above) while the gain over a **wrong** fit is
  overwhelming (fixed 59 / broke 21, p=2.5e-05). Both are true and both matter.
- **`T_choice`** — temperature for the multi-way field. It is monotone, so it never
  changes which label wins; the tool only writes a non-1.0 value if argmax accuracy
  actually improves. Leave it at 1.0 otherwise.
- **`--min-items`** (default 40) — refuses to fit on too little data rather than shipping noise.

---

## 4. What gets written

```json
{
  "tuple": "qfn-iq3_xxs|jev-port|none|v1|L|banking77-mf",
  "task": "banking77-mf",
  "a_noul": 0.45035,
  "c_noul": -1.560668,
  "T_choice": 1.0,
  "T_score": 1.0,
  "provenance": "refit 2026-11-01 task=banking77-mf dev=250 holdout=250 identity(acc=0.764 ece=0.203) -> refit(acc=0.820 ece=0.079) mcnemar p=5.9e-02"
}
```

- The file **must** be named `calib-<mode>.json` for the readout letter it is for
  (`calib-L.json` for text). The loader refuses a file whose `tuple` 5th field disagrees
  with the filename — that guard already existed; do not work around it.
- **`task` is the contract.** At serve time `STRATA_JEV_TASK` must equal it, or the fit is
  refused and you get identity. An **untagged** file is refused too when a task tag is set.
- **`provenance` is required reading.** It carries the holdout numbers so nobody has to
  take the fit on trust.

---

## 5. Deploying

Calibration is selected by **directory**, so per task means per directory. With llama-swap,
give the arm its own calib dir and task tag in the preset env:

```yaml
  "p-my-arm-3090":
    name: "My arm (3090)"
    cmd: >
      sh -c 'cd /cfg/strata/MyArm && exec /cfg/strata/venv/bin/python -m serve.server \
             --engine strata --config /cfg/strata/MyArm/pool.json --host 127.0.0.1 \
             --port ${PORT} --gpu 1'
    env: ["LD_LIBRARY_PATH=/cfg/toolchain/cuda-13.0/lib64",
          "STRATA_JEV_MODES=chat,decision",
          "STRATA_JEV_TASK=my-task",
          "STRATA_JEV_CALIB_DIR=/cfg/strata/MyArm/calib-my-task"]
```

Then hot-reload the fit without dropping the model:

```bash
curl -s -X POST -H "Authorization: Bearer $STRATA_JEV_ADMIN_KEY" \
     http://127.0.0.1:$PORT/admin/calib
```

(With no `STRATA_JEV_ADMIN_KEY` configured the endpoint answers 404 by design — the admin
surface does not exist rather than advertising itself.)

---

## 6. Verifying in production

1. **Check what actually applied.** Every `noul` answer carries `"calib": "platt_v1"` or
   `"identity"`. If it says `identity` after you deployed a fit, the task tag or the
   directory is wrong — check the server log for
   `[jev-calib] not applying fit (fit task '...' != runtime task '...')`.
2. **Re-run the holdout through the served arm** and confirm the numbers you were promised:

   ```bash
   python3 scripts/eval_mf_set.py run --base URL --key "$KEY" \
       --set holdout.jsonl --out arms/verify.json --label verify
   ```

3. **Run it twice.** The engine is **not bit-exact run to run** (adaptive / expert-swap
   tiers): `p_yes` was observed to drift up to 5.7e-2 between identical passes, and
   decisions at the boundary flip. Measured decision agreement on 250 items: 247/250
   (98.8%), so treat any accuracy difference under ~1.2% as noise.
4. **Never edit llama-swap config while an eval is running.** The config watcher disturbs
   the upstream and turned a 25-item block of requests into HTTP 400s mid-run once.

---

## 7. Pitfalls

- **Don't reuse another task's numbers.** A fit is a domain claim; the same `a`/`c` is
  wrong somewhere else. That is the defect this document exists to prevent.
- **Don't "fix" a bad fit by forcing it.** `--force` exists for deliberate overrides.
- **Don't tune on the holdout.** Fit on dev, quote holdout. Re-splitting after seeing
  results is how you fool yourself.
- **Don't expect calibration to rescue a dead readout.** If accuracy by confidence band
  is flat (§1), the signal is not there; abstain on the ambiguous band or fix the prompt.
- **`STRATA_JEV_CALIB_STRICT=1`** refuses half-fitted files (unknown keys, absurd `a`/`T`).
  Worth turning on for any arm you care about.

---

## 8. Runtime knobs

| env | meaning | default |
|---|---|---|
| `STRATA_JEV_CALIB_DIR` | directory holding `calib-<mode>.json` | `calib` |
| `STRATA_JEV_TASK` | runtime task tag; a fit with a different `task` is refused | unset (legacy: apply whatever loads) |
| `STRATA_JEV_CALIB_STRICT` | refuse half-fitted / absurd files | `0` |
| `STRATA_JEV_ADMIN_KEY` | enables `POST /admin/calib` hot reload | unset (404) |
| `STRATA_JEV_MODES` | which surfaces exist; `decision` is required for `/v1/systemone` | `chat` |
| `STRATA_JEV_ALIAS` / `STRATA_JEV_QUANT` | recorded in the tuple string for traceability | `local` / auto |

---

## 9. End-to-end worked example

Real run, 2026-11-01, RTX 3090, banking77 multi-field task:

```bash
# items: 500, is_card balanced 250/250, built from the HF rows API with gold from intent labels
python3 scripts/eval_mf_set.py run \
    --base http://10.89.0.57:8080/upstream/p-jevstrata-head-raw-3090 --key "$KEY" \
    --set mf-500.jsonl --out arms/head-raw.json --label head-raw
#   -> is_card acc=0.760 brier=0.1983 group acc=0.740

scripts/refit_calib.py fit --eval-json arms/head-raw.json --task banking77-mf \
    --criteria card top_up transfer cash other \
    --out-dir /cfg/strata/Jevstrata-head/calib-b77
#   -> a_noul=0.4504 c_noul=-1.5607, GATE PASS, wrote calib-L.json
```

Deployed on a preset with `STRATA_JEV_TASK=banking77-mf` and served the untouched
holdout half: **acc 0.816, Brier 0.1285** (the fit predicted 0.820/0.1293). Repeat pass:
0.820, decisions 247/250 identical. Against the previously shipped fit on those same
items: fixed 59, broke 21, **McNemar p = 2.5e-05**.

That is the loop: **label ~100 items → fit on half → gate on the other half → deploy with
a task tag → verify twice.**

---

## 10. Dynamic calibration — adapt the boundary from live traffic, without labels

A static fit is right for the month it was fitted in. When the request mix shifts you need a
new boundary, and waiting for a fresh labelled set means being wrong for however long the
labelling takes. You do **not** need labels on the traffic to move the boundary.

### What makes it possible

Three things, two of which already exist:

1. **A labelled dev set, once** → the confusion matrix `C[pred][y]`.
2. **The live marginal of predictions** → the server already writes one row per answered
   question to `calib/data/log-<mode>.jsonl`, each with `raw_p` (pre-calibration).
3. **An assumption about the base rate** — that the traffic's true positive rate is stable
   enough to balance against (or that you know it from the business).

Then BBSE (black-box shift estimation) solves `C·π = q` for the traffic class prior, and you
place the threshold where the predicted positive rate equals it. No traffic labels anywhere.

### Measured (500 gold-labelled banking77 items; traffic labels used only for scoring)

| boundary | accuracy on unseen traffic |
|---|---|
| shipped static fit (wrong task) | 0.664 |
| identity — no calibration | 0.764 |
| **BBSE prior-matched, NO traffic labels** | **0.816** |
| oracle prior-matched (upper bound) | 0.816 |
| labelled offline refit (needs 250 labels) | 0.816 |

Estimated base rate from unlabelled marginals: **0.484** against a true 0.500. Streaming over
five 50-item batches with shrinkage + EMA: **dynamic 0.828 vs static 0.764**. Per-batch BBSE
is noisy (n=50: mean 0.488, sd 0.108) — shrink and smooth or the boundary jitters.

On the real serving log (984 rows, this rig): estimated prior **0.493** (true 0.500) and a
recommended `c_noul = -1.6186` against the hand-fitted `-1.5607` — the two agree to 0.06 from
completely different information.

### Running it

```bash
scripts/dynamic_calib.py estimate --log /cfg/.../calib/data/log-L.jsonl \
    --dev-eval arms/dev-only.json            # what would the boundary be?
scripts/dynamic_calib.py update --log ... --dev-eval ... \
    --calib /cfg/.../calib-L.json --out /cfg/.../calib-L.json
curl -X POST -H "Authorization: Bearer $STRATA_JEV_ADMIN_KEY" .../admin/calib   # hot reload
```

Deliberate design choices:

- **It is a scheduled job, not an in-process control loop.** A serving process that silently
  re-tunes its own boundary from live traffic, with no human-visible record, is a runaway
  waiting to happen. This writes a file the existing reload path picks up, and stamps every
  update into a `dynamic` block with `n`, the estimate, the threshold and `c_prev`→`c_new`.
- **Bounded**: `--max-step` (default 0.5) caps the movement per update, so one bad log cannot
  swing the boundary across the data.
- **Shrunk**: `λ = n/(n+n0)` toward the prior (default n0=200), plus optional `--ema` across
  successive updates.
- **Alarmed**: if the estimate lands more than `--alarm` (default 0.45) from the expected
  prior, it refuses to write — that is a distribution shift or broken logging, not drift.
- **Singularity check**: if the readout never separates the classes at 0.5, `C` is not
  invertible and BBSE refuses rather than inventing a prior.
- **Needs a per-task dev set.** The confusion matrix is task-specific, exactly like the fit.

### When NOT to use it

- No labelled dev set at all → you have no confusion matrix; use the labelled refit (§3).
- The readout is nearly binary (see the band table in §1) → the marginal carries little
  information and BBSE estimates get unstable; prefer a periodic labelled refit.
- Delayed feedback is available (a human eventually corrects answers) → prefer real labels
  over an inferred prior; they are strictly more informative.
