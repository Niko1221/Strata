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

## Multi-field reads (M3.1)

N questions on one state used to cost N cold prefills of the same state (the M3.0 defect). Three
paths exist; the default is the one that ships.

| path | env | what it changes | measured (N fields, RTX 3090, same states) |
|---|---|---|---|
| **C — state parking (default, ON unless disabled)** | `STRATA_JEV_PIN=0` disables | prompts stay byte-identical to M3.0; the engine is told where the shared prefix ends, reads it once and pins its checkpoint, so each later field reads only its own question turn | 1/2/3/4/5 fields 210/549/1232/1405/1699 ms (measured 2026-10-23, engine `a2182599`, median of 3); field 1 readout byte-identical; order-insensitive (199/200 argmax agreement when the fields are reversed) |
| **B — one prompt, N rows (opt-in)** | `STRATA_JEV_ROWS=1` | one consolidated prompt; the request names every field's cue position (`jevpos=`), the engine copies those head rows out of whichever verify window covers them while the prompt is read, and `BROWAT` reads them back **by absolute position**, so one prefill answers N questions | **29–45% fewer prompt tokens from 2 fields up, but slower wall-clock at every N≥2** (N=2 134 vs 189 tokens, 1013 vs 549 ms; N=5 270 vs 490 tokens, 1727 vs 1699 ms — measured 2026-10-23, engine `a2182599`, median of 3; the earlier "1.33× faster at 2 fields" was measured while this path silently fell back to one prompt per field and is withdrawn); fields 2..N see the earlier questions, field order matters (147/200 agreement, mean \|Δp\| .1326, McNemar p=.0008), and at N=1 it disagrees with C on 17/600 GQA items (−1.8 points) — hence opt-in |
| **H — pin + row capture (opt-in, rejected)** | `STRATA_JEV_HYBRID=1` | per-field prompts, pinned prefix, cue row captured from the shared read | 1.15× at 2 fields but 20–28% *slower* than C at N≥3 (one serial read per field); kept in the tree as a measured negative |

Route by field count: N=1 either; **N≥2 C**, unless the caller needs fewer prompt tokens and can pin the
field order itself and accept the answer shift — B is a token-cost path, not a latency one, so it stays
opt-in and is never the default.

**Two changes are on by default, not behind a flag**, and both are deliberate: the single-token label
refusal (above) and **state parking** (`STRATA_JEV_PIN`, on unless set to 0), which is the M3.1 fix
itself. Pinning does not change the prompt — every field's prompt is byte-identical to M3.0, and the
readout position with it — but it does make the engine park a checkpoint at the shared prefix, so answer
identity rests on checkpoint-resume exactness rather than on a full re-read. Measured on the live box,
field 1's probabilities move by ≤.008 between pinned and unpinned runs (the same order as run-to-run
variation), and a 200-item field-order test agrees 199/200; `STRATA_JEV_PIN=0` restores the fully
unpinned read if a caller wants to rule the cache out entirely.

The engine ops behind these are `BROWAT <row> <tokens>` → `BROWATD <n> <tok>:<logit>…` and
`BROWROWS` → `BROWROWSD <n>`. `BROWAT` has **two addressings**, and the difference matters:

- **captured** — if the same request carried `jevpos=`, the named rows were kept during that read, and
  `BROWAT <absolute prompt position>` reads one of them. A position the request never ran is refused
  **by name**; it never falls back to a neighbouring row. `BROWCAP` → `BROWCAPD <n>` reports how many
  rows were kept, so the server verifies the count equals what it named **before** reading; if it does
  not (a dropped list, a position no window ran), the request falls back to one prompt per field rather
  than let an absolute position be read as a window index.
- **window** — with no capture, `BROWAT <row>` indexes the LAST verify window, whose rows start at
  that window's first token. `BROWROWS` reports how many rows it holds, so a caller can tell whether
  its own numbering is the engine's (it only is when the count equals the tokens it sent).

A `jevpos=` list names at most **32** positions, and every entry must be all digits. A longer list, a
malformed one, or an entry that is not a position (`12x`) is refused **whole** — never truncated, never
a guessed position — and the request then falls back to one prompt per field. The cap counts
*positions*, not characters: 32 three-digit positions are accepted.

The capability is advertised in the engine's `INFO` line as `jevpos=1`; an engine that does not
advertise it (older build, or started without `--jev-brow`) is never sent the row path at all. Both
ops are under `--jev-brow`, and the engine is never asked unless the server sends one. One limit of
that guarantee, measured on this build: the *per-field* readout still calls `BROW` (it is the same
readout M3.0 shipped), so an engine built without `--jev-brow` answers `ERR BROW:` and the server ends
and restarts it rather than serving a 503 — start the serve process with `--jev-brow` for any decision
workload.

## Label alphabet (`choice`)

`STRATA_JEV_LABELS` picks how choice options are rendered and which token each option's logit is
read from. The option list and the readout always use the same alphabet.

- `upper` (default): `A`…`Z`. Every calibrator on disk was fitted against this.
- `digit-lower`: `0`-`9`, then `a`-`z` to 26 options (the scheme requested on #3643, so a text
  multi-field path can share a token-path scorer). Unknown values fail closed at import.

Every rendered label must be **exactly one token**; a request whose label encodes to more than one
is refused with `422` on `questions.<qid>.criteria` rather than scored on a truncated path. This check
runs on **every** request, including the default `upper` scheme. It applies to `choice` **and** `score`
labels, which are rendered and read the same way. With `A`-`Z` and `0`-`9`/`a`-`z` every label is one
token in the shipped tokenizer, so no valid request changes status.

Changing the alphabet moves the readout slightly (`mean |Δp| = .0034` per option on ARC, 0/160
identical vectors) but it did not change any decision on 160 items and the two alphabets' own
optimal temperatures sit 8% apart (T = 1.00 vs .92), inside the shipped 1.0557. No refit was needed
for the scheme itself — but note the test task was 99%-accurate ARC, the weakest possible detector;
a 10-way uncertain task is where a refit would show if one is owed.

## Calibration (M3.2)

Post-hoc and per-tuple. **Not accuracy-neutral**, and it is worth being precise about why:

- `choice` / `score`: temperature on label log-probs — a monotone rescale of one softmax, so
  these genuinely cannot change the winning label. Accuracy is unchanged by construction.
- `noul`: Platt `sigmoid(a·logit(p_yes) + c)`. The map is monotone, so it cannot reorder options,
  but a non-zero offset **is** a decision threshold: it moves the probability that crosses 0.5
  away from the raw one. A fit can therefore change yes/no answers and accuracy.

Measured on banking77 (2-field, held out, 2026-10-23): fitting `noul` moved accuracy .670 → .800
and Brier .2208 → .1385, with the boundary pushed to raw p ≈ .94. That gain is threshold tuning
on a balanced 50/50 prior, **not** better probabilities — on traffic with a different base rate it
will over- or under-predict. The offset must be fitted per task on a representative sample and
must never be carried over from another task: a banking77 fit moved POPE .883 → .850 and GQA
.803 → .720, and zeroing only the offset (keeping the slope) restored both.

Two more rules that measurement forced:

- **Fit on the task you serve.** Where the readout is already honest, refuse to fit: GQA raw ECE
  is .054 and a dev fit made it worse (.083) while losing 2.3 points of accuracy. On the image
  tuples the shipped fits are at their ceiling — refitting on top of them asks for T = .985/.984
  and `a = 1.066, c = −.003`.
- **Never fit on calibrated output.** A dev fit applied to probabilities that were already
  calibrated compounds the transform, and the resulting parameters are not installable. Serve
  identity (move `calib-*.json` aside) for any raw→fit measurement, and compare arms only after
  inverting a known transform.

`calib/calib-L.json`, `calib-D.json`, `calib-S.json` are the per-tuple files (L = text,
D = image direct, S = image summary), read at startup, hot-swapped via `POST /admin/calib`
(Bearer `STRATA_JEV_ADMIN_KEY`; empty key = open like the rest of the control plane). A file whose
keys the loader does not read (e.g. `a`/`c`) loads as identity while still honouring its `T` — a
silent half-fit; `STRATA_JEV_CALIB_STRICT=1` refuses that file instead. Every answered question
logs a row to `calib/data/log.jsonl` (`ts, tuple, qid, type, raw_p[, v_d]`) — the schema the
donor's `fit.py` consumes.

**Measured on QFN (Qwen3.8-Flash-Next IQ3_XXS, RTX 3090), 100 items/task, 2026-10-15 readout:**

| task | acc raw→cal | ECE15 raw→cal | ms/item |
|---|---|---|---|
| BoolQ (noul) | 88% → 90% | 0.073 → 0.057 | 1620 |
| ARC (choice) | 100% → 100% | 0.033 → 0.033 | 1201 |
| SST-5 (score) | 55% → 55% | 0.301 → **0.141** | 1222 |

Fitted: `a_noul=0.7446 c_noul=0.2449 T_choice=1.0459 T_score=2.2428`
(donor 27B baseline: 90.5 / 99 / 56.9 — parity). V-mode on the same ARC items:
100% acc, 100/100 argmax agreement with L, 4631 ms/item.

**Read that table with its own caveat (added 2026-10-23).** Those fits were evaluated on the
same dev split they were fitted on, so they are calibration-saturated: re-running the same three
suites on the current binary gives BoolQ .910 → .910 (ECE .049 raw), ARC 1.000 → 1.000 (ECE
.007), SST-5 .520/.530 with the refit *hurting* test ECE (.127 → .140). The delivered calibrator
has not regressed — the readout improved underneath it, and the remaining headroom on these three
tasks is now ~zero. Where calibration still pays is a genuinely overconfident tuple: image summary
mode wanted T = 5.03 before its file was fitted (POPE .883 → .767 when it was missing).

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
