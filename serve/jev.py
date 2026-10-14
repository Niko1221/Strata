"""serve/jev.py - the jev decision endpoint (M3.0 of the jevstrata port): POST /v1/systemone.

Renders the frozen decision prompt (DECISIONS.md D3 - do not edit), appends each question's
suffix, decodes one step per branch through the engine turn, reads the final-row logits, and
answers with the jev readout math (jevref/jmath.py, M1 identity calibration).

Readout boundary: an engine method `final_row_logits(prompt_ids) -> {token_id: logit}` - the
final position's logits row (the real engine's Verifier::head_logits_ window, served by a BROW
wire line in the M3.0 engine commit).  Engines without it get 503, never a fabricated answer.
"""
from __future__ import annotations

import json
import math
import threading
import time

from serve import jev_calib

# jevcpp's frozen decision prompt (DECISIONS.md D3; byte-frozen - reopening forces recalibration).
SYSTEM_V1 = ("You are a precise decision engine. Read the state, then answer the question "
             "with a single label and nothing else.")

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ---- the jev readout math (a line-for-line port of jevref/jmath.py, M1 identity) ----

def log_softmax_masked(logits, usable=None):
    if usable is None:
        usable = [True] * len(logits)
    mx = -math.inf
    for z, u in zip(logits, usable):
        if u and z > mx:
            mx = z
    acc = 0.0
    for z, u in zip(logits, usable):
        if u:
            acc += math.exp(z - mx)
    lse = math.log(acc) + mx
    return [z - lse if u else -math.inf for z, u in zip(logits, usable)], lse


def softmax(scores):
    mx = max(scores)
    e = [math.exp(x - mx) for x in scores]
    t = sum(e)
    return [x / t for x in e]


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def confidence(p):
    k = len(p)
    if k < 2:
        return 0.0
    v = (k * max(p) - 1.0) / (k - 1)
    return min(1.0, max(0.0, v))


def readout_scores(row: dict, variants: list[list[int]]) -> list[float]:
    """Label log-scores from a logits row.  The LSE constant cancels in softmax(scores) and
    sigma(d) (jmath's verified invariance), so normalizing over the label set alone gives the
    same probabilities as the full masked vocab."""
    zs = []
    for vs in variants:
        z = row.get(vs[0])
        if z is None:
            raise ValueError(f"logits row has no token {vs[0]}")
        zs.append(z)
    logps, _ = log_softmax_masked(zs)
    return logps


def answers_from_scores(qtype: str, scores: list[float]) -> dict:
    """Identity (uncalibrated) answers - kept for the battery and offline replay."""
    if qtype == "noul":
        d = scores[1] - scores[0]                       # [No, Yes] order
        return {"type": "noul", "noul": sigmoid(d)}
    p = softmax(scores)
    if qtype == "choice":
        best = max(range(len(p)), key=lambda i: p[i])
        return {"type": "choice", "choice": best, "probabilities": p, "confidence": confidence(p)}
    if qtype == "score":
        sc = sum(i * pi for i, pi in enumerate(p))
        return {"type": "score", "score": sc, "probabilities": p, "confidence": confidence(p)}
    raise ValueError(qtype)


def answers_from_calibrated(qtype: str, pr: list[float], calib_id: str) -> dict:
    """Answers from calibrated label probabilities (donor order: rounded pr -> calib ->
    answers).  `pr` is already calibrated and renormalized; `scores` for noul is
    recovered as the pair's log-odds so the sigmoid path stays exact."""
    if qtype == "noul":
        p = min(max(pr[1] if len(pr) > 1 else 0.0, 1e-9), 1 - 1e-9)
        return {"type": "noul", "noul": p, "calib": calib_id}
    best = max(range(len(pr)), key=lambda i: pr[i])
    if qtype == "choice":
        return {"type": "choice", "choice": best, "probabilities": pr,
                "confidence": confidence(pr), "calib": calib_id}
    if qtype == "score":
        sc = sum(i * pi for i, pi in enumerate(pr))
        return {"type": "score", "score": sc, "probabilities": pr,
                "confidence": confidence(pr), "calib": calib_id}
    raise ValueError(qtype)



# ---- request validation (the donor's 422-with-field contract) ----

def _err(field: str, message: str) -> tuple[int, dict]:
    return 422, {"error": {"field": field, "message": message}}


def parse_questions(questions) -> tuple[list | None, dict]:
    """-> (qdefs, err_body): qdefs is None on error; [{id, type, instructions, labels}] on success."""
    if not isinstance(questions, dict) or not questions:
        return None, _err("questions", "must be a non-empty map")
    qdefs = []
    for qid, v in questions.items():
        if not isinstance(v, dict):
            return None, _err(f"questions.{qid}", "must be an object")
        qtype = v.get("type")
        if qtype not in ("noul", "choice", "score"):
            return None, _err(f"questions.{qid}.type", "must be noul|choice|score")
        ins = v.get("instructions")
        if not isinstance(ins, str) or not ins:
            ins_text = json.dumps(ins, indent=2) if not isinstance(ins, str) else ins
            if not isinstance(ins, (str, dict, list)) or ins_text.strip() == "":
                return None, _err(f"questions.{qid}.instructions", "required")
            ins = ins_text
        criteria = v.get("criteria")
        descriptions: list[str] = []
        if qtype == "choice":
            if criteria is None:
                return None, _err(f"questions.{qid}.criteria", "required for choice")
            if isinstance(criteria, dict):
                labels = list(criteria.keys())
                descriptions = [": " + str(criteria[k]) if criteria[k] is not None else ""
                                for k in labels]
            elif isinstance(criteria, list):
                labels = [str(x) for x in criteria]
                descriptions = [""] * len(labels)
            else:
                return None, _err(f"questions.{qid}.criteria", "must be object or array")
            if len(labels) < 2:
                return None, _err(f"questions.{qid}.criteria", "need >= 2 options")
            if len(labels) > 26:
                return None, _err(f"questions.{qid}.criteria",
                                            ">26 options need the numeric-label trie (not in this build)")
            render_labels = [LETTERS[i] for i in range(len(labels))]
        elif qtype == "score":
            if not isinstance(criteria, list) or len(criteria) < 2 or len(criteria) > 10:
                return None, _err(f"questions.{qid}.criteria", "must be an array of 2..10 levels")
            labels = [str(x) for x in criteria]
            descriptions = [""] * len(labels)
            render_labels = [str(i) for i in range(len(labels))]
        else:
            labels, render_labels = ["No", "Yes"], ["No", "Yes"]
            if isinstance(criteria, dict):
                true_d = criteria.get("true")
                false_d = criteria.get("false")
                descriptions = [": " + str(false_d) if false_d is not None else "",
                                ": " + str(true_d) if true_d is not None else ""]
            else:
                descriptions = ["", ""]
        qdefs.append({"id": qid, "type": qtype, "instructions": ins,
                      "labels": labels, "render_labels": render_labels,
                      "descriptions": descriptions})
    return qdefs, (200, {})


def render_state(state) -> str:
    """The state's serialized text - the DONOR'S render (server.cpp:266-268):
    string states pass through verbatim; objects pretty-print as sorted JSON."""
    if isinstance(state, str):
        return state
    return json.dumps(state, indent=2, sort_keys=True, ensure_ascii=False)


def question_block(qdef) -> str:
    """The question's rendered suffix - the DONOR'S render (server.cpp render_qblock),
    frozen D3 shape: the full option list is always shown to the model."""
    q = f"Question: {qdef['instructions']}\n"
    if qdef["type"] == "choice":
        q += "Options:\n"
        for letter, label, desc in zip(LETTERS, qdef["labels"], qdef["descriptions"]):
            q += f"{letter}. {label}{desc}\n"
        q += "Reply with the label of the best option.\n"
    elif qdef["type"] == "score":
        q += "Levels (0 is lowest):\n"
        for i, label in enumerate(qdef["labels"]):
            q += f"{i}. {label}\n"
        q += "Reply with the level number.\n"
    else:
        true_d, false_d = qdef["descriptions"]
        if true_d:
            q += f"Yes means: {true_d}\n"
        if false_d:
            q += f"No means: {false_d}\n"
        q += "Reply Yes or No.\n"
    q += "Answer:"
    return q


def handle(svc, req: dict) -> tuple[int, dict]:
    """POST /v1/systemone -> (HTTP status, body).  The whole inference holds the engine turn
    (svc.fifo), like every other engine-touching path."""
    if not isinstance(req, dict) or "state" not in req:
        return _err("state", "required")
    if "questions" not in req:
        return _err("questions", "required")
    qdefs, err_body = parse_questions(req["questions"])
    if qdefs is None:
        return err_body                            # already (422, {...})

    engine = svc.engine
    if not hasattr(engine, "final_row_logits"):
        return 503, {"error": {"message": "systemone needs an engine with a logits readout "
                                           "(the M3.0 engine commit adds BROW)"}}

    state_text = render_state(req["state"])
    answers, in_tokens, readouts = {}, 0, 0
    with svc.fifo:                                       # the engine turn: one request at a time
        for qdef in qdefs:
            prompt = svc.template.render(
                [{"role": "system", "content": SYSTEM_V1},
                 {"role": "user", "content": state_text + "\n\n" + question_block(qdef)}],
                add_generation_prompt=True,
                enable_thinking=False)   # D3 (frozen): the readout position must predict the label, not open <think>
            ids = svc.tok.encode(prompt, parse_special=True)
            if not ids:
                return _err("state", "tokenization failed")
            variants = [svc.tok.encode(lbl, parse_special=True)[:1] for lbl in qdef["render_labels"]]
            cancel = threading.Event()
            for _ in engine.generate(ids, 1, {}, cancel):
                pass                                     # the one decode step; its logits row is the readout
            tokens = sorted({t for v in variants for t in v})
            row = engine.final_row_logits(ids, tokens)   # BROW: only the label tokens asked for
            try:
                scores = readout_scores(row, variants)
            except ValueError as e:
                return _err("state", f"logits readout failed: {e}")
            # ---- donor order (server.cpp:1134-1142): round 4dp, renormalize, THEN calibrate ----
            pr_raw = softmax(scores)                     # unrounded, for the calib log
            pr = [round(x * 10000.0) / 10000.0 for x in pr_raw]
            t = sum(pr)
            if t > 0:
                pr = [x / t for x in pr]
            calib_id = None
            pr, calib_id = jev_calib.apply(qdef["type"], pr)
            answers[qdef["id"]] = answers_from_calibrated(qdef["type"], pr, calib_id)
            # M2 B1.1 log row (unrounded raw_p; donor schema minus label_mass, BROW is sparse)
            jev_calib.log_enqueue({
                "ts": int(time.time()),
                "tuple": jev_calib.tuple_of("L", qdef["type"]),
                "qid": qdef["id"],
                "type": qdef["type"],
                "raw_p": pr_raw,
            })
            in_tokens += len(ids)
            readouts += 1
    return 200, {"answers": answers,
                 "usage": {"prompt_tokens": in_tokens, "readout_tokens": readouts}}
