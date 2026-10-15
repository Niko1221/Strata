# Jev systemone — the decision engine for Strata (jev-port PR)

Port of the jevcpp decision engine (`/opt/data/llm-swap/jevcpp`, `src/server.cpp`) onto
Strata's serve layer. A model that answers typed questions with **calibrated
probabilities in one decode step per question** — label logits read off the final row
(the engine's `--jev-brow` BROW wire), readout math applied post-hoc. Never generates
prose to decide.

Donor reference: jevcpp on Qwen3.8-27B (q4_k_xl), shipped calibration tuple
`p-jevcpp-qwen38-27b|q4_k_xl|timber|v1|V|all` (T_choice 1.5861). This port runs on
QFN = Qwen3.8-Flash-Next IQ3_XXS MoE (RTX 3090, Strata engine v0.1.43-lineage, 32k ctx).

## 1. Features (the full jevcpp surface)

| Feature | Request | Status |
|---|---|---|
| Core decision endpoint | `POST /v1/systemone` `{state, questions}` — noul (yes/no), choice (letters), score (0-10 levels) | ✅ live |
| Frozen D3 prompt | byte-frozen question block (donor render, `enable_thinking=false`, verbatim string states) | ✅ golden-tested |
| BROW readout | `--jev-brow` engine flag; final-row logits, fail-closed | ✅ engine commit |
| Calibration (M3.2) | post-hoc, argmax-preserving: Platt (a,c) for noul, temperature for choice/score; per-tuple (L/V/D/S) files `calib-<mode>.json`; `POST /admin/calib` hot-swap; donor-schema log rows for `fit.py` | ✅ live |
| V-mode (M3.1) | header `X-Jev-Mode: V` — per-option yes/no branches, `p = softmax(s_yes − s_no)`, name-keyed | ✅ live |
| Image → decision (M3.8) | `image_b64` + `image_mode`: `summary` (vision → 160-tok summary → decide) or **`direct`** (raw image embeddings + question, ONE readout) | ✅ live |
| Decide-then-generate (M3.9) | `"then": {"generate": {...}}` — answers folded into the last user turn, one wire request | ✅ live |
| Chat surface (M3.3) | `POST /v1/chat/completions` with `"jev": {state|messages, questions}` | ✅ live |
| Concurrency | FIFO-serialized engine turns; 7 clients × 3 questions, zero cross-talk (tested) | ✅ tested |

Error contract: donor's `422` + `{"error": {"field": ...}}`. Streaming refused with the
jev extension, `state` XOR `messages`, `image_mode` validated, D+V refused with 422.

## 2. Usage

```bash
# text decision (one decode step)
curl -s $URL/v1/systemone -d '{
  "state": {"engine": "hot", "load": 0.9},
  "questions": {"q1": {"type": "noul", "instructions": "Throttle?"}}}'

# image → decision, direct mode (raw image tokens decide)
curl -s $URL/v1/systemone -d '{
  "image_b64": "'"$(<screen.b64)"'", "image_mode": "direct",
  "questions": {"threat": {"type": "noul", "instructions": "Enemy visible?"}}}'

# decide, then narrate
curl -s $URL/v1/systemone -d '{
  "state": {"hp": 12}, "questions": {...},
  "then": {"generate": {"messages": [{"role": "user", "content": "Narrate."}],
                        "max_tokens": 128}}}'
```

Full surface + examples: `docs/JEV-SYSTEMONE.md`. Refit calibration:
`calib/README.md` (the suite IS the procedure).

## 3. Test strategy

**Unit (mock engine) — 54/54 green:** readout math invariants, prompt rendering
(byte-pinned noul/choice blocks — the Yes/No-swap regression is now a golden test),
per-tuple calibration isolation, V-mode softmax, image two-call/direct shapes,
decide-then-generate, chat-surface validation arms, 7×3 concurrency + FIFO
serialization.

**Live suites** (`calib/bench_suite.py`): dev-split fit → held-out test eval,
standard top-label ECE15 + mean NLL, per-item JSONL logs. Guards: Platt slope
a≤0 refused (label/prompt inversion detector — it caught a real bug), temperature
at search boundaries flagged.

**Adversarial audit protocol** (Justin's rule, 2026-10-15): 1-in-10 of all test
items re-verified by hand — text golds re-read from source passages, image items
regenerated from committed seeds with sha256 match against the logged hash, sampled
error images independently re-counted with vision. Result across text+image+real
suites: 0 scoring hallucinations, 0 gold errors, 1 documented CV-Bench ambiguity.
Artifacts: `audit-*.json`, `audit-imgs/`, `*-item-log.jsonl` (920 rows).

**Real-benchmark converters** (`calib/convert_benchmarks.py`): POPE adversarial,
CV-Bench, CountBenchQA via HF datasets-server; banned-field self-test (answer-bearing
fields never enter requests), sha256 image manifest, dev/test image-disjoint, POPE
sampled by image at exactly 50/50.

**Vision strategy** (Opus-reviewed, `/opt/data/reviews/vision-decision-strategy-plan.md`):
6-arm design (D/S/S-QC/S-V/blind/shuffled — the last two are contamination
detectors), circular rotation for choice position bias, phased screen→full→frozen
PR gate.

## 4. Benchmarking results (all live, QFN IQ3_XXS, RTX 3090)

### 4a. Pre/post calibration (held-out test, offline replay of identical items)

| tuple (mode\|qtype\|set) | n | acc pre→post | NLL pre→post | ECE15 pre→post |
|---|---|---|---|---|
| L\|noul\|BoolQ | 100 | .890 → **.910** | .298 → **.262** | .084 → **.033** |
| L\|choice\|ARC | 100 | 1.000 → 1.000 | .005 → .007 | .005 → .006 |
| L\|score\|SST-5 | 100 | .520 → .520 | 1.471 → **1.049** | .301 → .154 |
| D\|noul\|POPE-adv | 60 | .867 → **.883** | .204 → **.195** | .122 → **.081** |
| D\|choice\|CV-Bench | 20 | .650 → .650 | .944 → **.863** | .145 → .260* |
| D\|score\|CountBench | 20 | 1.000 → 1.000 | .102 → **.036** | .091 → **.033** |

*CVB n=20, bin-noise territory; NLL improves, acc flat. Fitted: L a=.6127 c=.6539
Tc=1.0557 Ts=2.263 · D a=1.0571 c=−.172 Tc=1.5456 Ts=.6555.
Also shown in `calib-prepost-ab.json`: the OLD single global calibrator made image
tuples WORSE than raw (CountBench ECE .091→.451!) — the per-tuple fix is the story.

### 4b. Benchmark vs jevcpp's own benchmark (donor m2_baseline, same 3 tasks)

Donor: Qwen3.8-27B q4_k_xl, dense, its own server. Port: QFN IQ3_XXS MoE, Strata.
Same task sets (BoolQ / ARC / SST-5 slices from jevcpp's eval data), same metrics
code, post-calibration numbers:

| task | jevcpp 27B (donor) | jev-port QFN (this PR) |
|---|---|---|
| BoolQ noul acc | 0.905 raw / 0.890 cal | 0.910 cal |
| BoolQ ECE15 | 0.036 raw / 0.049 cal | **0.033** cal |
| ARC choice acc | 0.990 | 1.000 |
| SST-5 score acc | 0.569 | 0.520 |
| SST-5 ECE15 | 0.110 raw / **0.030** cal | 0.154 cal |
| SST-5 NLL | 1.103 raw / 1.031 cal | 1.049 cal |

The 26B-smaller IQ3 MoE matches the donor within noise on noul and choice; score is
−4.9pp (a genuinely harder readout at this quant) but calibration recovers most of
the confidence quality.

### 4c. Performance (warm, single client, n=10 per cell)

| workload | p50 | p95 | rps |
|---|---|---|---|
| text 1 question | 272 ms | 368 ms | 3.7 |
| text 3 questions | 1002 ms | 1016 ms | 1.0 |
| image direct, 1 question | 275 ms | 362 ms | 3.6 |
| image summary, 1 question | 1625 ms | 2731 ms | 0.6 |

vs donor (27B dense, its m1_perf matrix, 1q-1c-warm): **71 ms p50 / 14.0 rps**.
The port is ~3.8× slower per decision — expected: MoE expert routing + engine wire
vs the donor's dense-batched loop. In exchange it runs in the Strata serving stack
(KV cache, vision, MTP, one process for chat+decisions). Image direct ≈ text
latency (encoder cached); summary costs ~6× direct (full 160-token generation).

Determinism caveat (inherited): GPU expert routing rounds differently per run —
repeat-item drift is possible (documented in the golden battery).

## 5. Artifacts

- `calib-prepost-ab.json` — 4a's numbers + method (offline replay, identical items)
- `perf-matrix.json` — 4c (live measured)
- `text-suite.md`, `image-suite.md`, `real-suite.md` + `*-item-log.jsonl` — suite reports
- `convert-manifest.json` — real-benchmark provenance (sha256 per image)
- `audit-text-sample.json`, `audit-image-sample.json`, `audit-real-sample.json` — 1-in-10 audits
- Donor reference files: `jevcpp/reports/m2_baseline.json`, `m1_perf.json`, `calib/calib.json`
