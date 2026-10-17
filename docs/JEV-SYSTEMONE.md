# Jev systemone — the decision endpoint (jev-port M3.x)

`POST /v1/systemone` turns the model into a **decision engine**: it reads a state,
answers typed questions with calibrated probabilities, and never generates prose to do it.
One decode step per question, the label logits read straight off the final row (the
engine's BROW wire line, `--jev-brow`), the jev readout math applied post-hoc.

Ported from jevcpp (`src/server.cpp`) onto Strata's serve. Wire-compatible: the donor's
request/response shapes, error contract (`422` with `field`), and calibration math.

## Request

```json
POST /v1/systemone
{
  "state":   {"weather": "clear", "season": "summer"},
  "questions": {
    "q1": {"type": "noul",   "instructions": "Is the weather suitable for outdoor work?",
           "criteria": {"true": "…", "false": "…"}},
    "q2": {"type": "choice", "instructions": "Which season is it?",
           "criteria": {"spring": null, "summer": null, "autumn": null, "winter": null}},
    "q3": {"type": "score",  "instructions": "Rate the conditions.",
           "criteria": ["poor", "fair", "good", "excellent"]}
  }
}
```

- `state`: an object (rendered as pretty JSON) or a string (passed through verbatim).
- `criteria` values on `choice`/`noul` are optional per-option descriptions — they are
  shown to the model.
- Headers: `X-Jev-Mode: V` switches **choice** questions to the per-option yes/no
  readout (below). Default `L`.

## Response

```json
{"answers": {"q1": {"type": "noul", "noul": 0.88, "calib": "platt_v1"},
             "q2": {"type": "choice", "choice": 1, "probabilities": [0.0, 0.98, 0.01, 0.01],
                    "confidence": 0.97, "calib": "T_choice_v1"},
             "q3": {"type": "score", "score": 2.1, "probabilities": [...],
                    "confidence": 0.4, "calib": "identity"}},
 "usage": {"prompt_tokens": 250, "readout_tokens": 3}}
```

Errors: `422` with `{"error": {"field": "questions.q2.criteria", "message": …}}`.

## Extensions

| Feature | Request | What it does |
|---|---|---|
| **V mode** | header `X-Jev-Mode: V` | Choice readout becomes K per-option yes/no branches: `d_i = s_yes − s_no`, `p = softmax(d)`. Probabilities keyed by option **name**, `choice` = the name. More robust on weak readouts; ~K× the readouts. |
| **Image → decision** | add `image_b64` (+ optional `vision_prompt`) | One wire request: the image is encoded (Strata's vision path), a bounded summary is generated, and the **summary becomes the decision state**. Response adds `image_summary` and `stages_used: ["vision", "decision"]`. Fail-closed: an empty summary is a 502, the decision never runs. |
| **Decide-then-generate** | add `"then": {"generate": {"messages": [...], "max_tokens": 512}}` | Decide first, then generate with the answers folded into the last user turn as `Jev decisions (JSON): …`. Response adds `then: {text, finish_reason, usage}`. |
| **Chat surface** | `POST /v1/chat/completions` with `"jev": {"questions": …}` and either `jev.state` or `messages` | The decision readout from the chat endpoint; the conversation is the state (`v1-chat`). `stream` is refused with the extension. Response: OpenAI-style envelope with the decision body under `jev`. |

## Calibration (M3.2)

Post-hoc, monotone, argmax-preserving — it reshapes probabilities, never picks a different label.

- `noul`: Platt scaling `sigmoid(a·d + c)` on the yes-logit
- `choice`/`score`: temperature on label log-probs

`calib/calib.json` is read at startup (identity until present), hot-swapped via
`POST /admin/calib` (Bearer `STRATA_JEV_ADMIN_KEY`; empty key = open like the rest of
the control plane). Every answered question logs a row to `calib/data/log.jsonl`
(`ts, tuple, qid, type, raw_p[, v_d]`) — the same schema the donor's `fit.py` consumes.

**Measured on QFN (Qwen3.8-Flash-Next IQ3_XXS, RTX 3090), 100 items/task:**

| task | acc raw→cal | ECE15 raw→cal | ms/item |
|---|---|---|---|
| BoolQ (noul) | 88% → 90% | 0.073 → 0.057 | 1620 |
| ARC (choice) | 100% → 100% | 0.033 → 0.033 | 1201 |
| SST-5 (score) | 55% → 55% | 0.301 → **0.141** | 1222 |

Fitted: `a_noul=0.7446 c_noul=0.2449 T_choice=1.0459 T_score=2.2428`
(donor 27B baseline: 90.5 / 99 / 56.9 — parity). V-mode on the same ARC items:
100% acc, 100/100 argmax agreement with L, 4631 ms/item.

## Concurrency

7 concurrent clients × 3 questions: every client gets exactly its own answers
(qid-keyed), and the engine FIFO serializes the readout turns — one engine turn at a
time, the serve-side equivalent of the donor's per-decode-step `ENG.mu`.

## Examples

```bash
# a noul question
curl -s localhost:8095/v1/systemone -d '{
  "state": {"engine": "hot", "load": 0.9},
  "questions": {"q": {"type": "noul", "instructions": "Throttle the engine?"}}}'

# choice in V mode
curl -s localhost:8095/v1/systemone -H 'X-Jev-Mode: V' -d '{
  "state": "Route: bridge flooded, tunnel open.",
  "questions": {"route": {"type": "choice", "instructions": "Which route?",
    "criteria": {"bridge": null, "tunnel": null, "ferry": null}}}}'

# image → decision (one request)
curl -s localhost:8095/v1/systemone -d '{
  "image_b64": "'"$(base64 -w0 screen.png)"'",
  "questions": {"threat": {"type": "noul", "instructions": "Is an enemy visible?"}}}'

# decide, then generate
curl -s localhost:8095/v1/systemone -d '{
  "state": {"hp": 12, "potions": 1},
  "questions": {"act": {"type": "choice", "instructions": "Next action?",
    "criteria": {"fight": null, "drink": null, "flee": null}}},
  "then": {"generate": {"messages": [{"role": "user", "content": "Narrate the turn."}],
                          "max_tokens": 128}}}'
```
