"""serve/jev.py - the jev decision endpoint (M3.0 of the jevstrata port): POST /v1/systemone.

Renders the frozen decision prompt (DECISIONS.md D3 - do not edit), appends each question's
suffix, decodes one step per branch through the engine turn, reads the final-row logits, and
answers with the jev readout math (jevref/jmath.py, M1 identity calibration).

Readout boundary: an engine method `final_row_logits(prompt_ids) -> {token_id: logit}` - the
final position's logits row (the real engine's Verifier::head_logits_ window, served by a BROW
wire line in the M3.0 engine commit).  Engines without it get 503, never a fabricated answer.
"""
from __future__ import annotations

import contextlib
import json
import math
import threading
import time

from pathlib import Path

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
                # labels are [No, Yes]: descriptions follow the SAME order.  The old
                # code also baked ': ' in here AND in question_block (doubled) and
                # question_block unpacked them as (true, false) - Yes showed No's
                # meaning (found by the image suite + Opus, 2026-10-15).
                descriptions = [false_d, true_d]
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
        no_d, yes_d = qdef["descriptions"]        # [No, Yes] order (was unpacked reversed)
        if yes_d:
            q += f"Yes means: {yes_d}\n"
        if no_d:
            q += f"No means: {no_d}\n"
        q += "Reply Yes or No.\n"
    q += "Answer:"
    return q


def question_head(qdef) -> str:
    """V-mode shared head (donor render_qhead, server.cpp:337-347): the option list as
    plain bullets - the branches below it each propose one option."""
    q = f"Question: {qdef['instructions']}\nOptions:\n"
    for label, desc in zip(qdef["labels"], qdef["descriptions"]):
        q += f"- {label}{desc}\n"
    return q


def v_branch(qdef, key: str) -> str:
    """V-mode per-option branch (donor render_vbranch, server.cpp:349-353)."""
    return (f"Proposed answer: {key}\n"
            f"Is the proposed answer the best option? Reply Yes or No.\n"
            f"Answer:")


VISION_PROMPT_DEFAULT = "Describe this game screen:"


def _handle_image_direct(svc, req: dict, mode: str, qdefs: list, b64: str) -> tuple[int, dict]:
    """image_mode: "direct" — the RAW image tokens decide, no summary in between
    (the donor's V2 idea, safe on Strata because the readout is BROW on the final
    row, never a seq fork): one prompt = [image] + question_block, one decode step,
    label logits off that row.  The image rides the user turn via Strata's normal
    vision path (svc.prepare encodes + expands the pads); the question text sits
    AFTER the image so the readout position attends the full image."""
    data_uri = "data:image/png;base64," + b64 if not b64.startswith("data:") else b64
    answers, in_tokens, readouts = {}, 0, 0
    # no outer fifo hold: svc.prepare takes its own engine turn for the encode,
    # and each generate() call is its own turn (same as the chat path)
    for qdef in qdefs:
        messages = [{"role": "system", "content": SYSTEM_V1},
                    {"role": "user",
                     "content": [{"type": "image", "source": data_uri},
                                 {"type": "text",
                                  "text": question_block(qdef)}]}]
        try:
            ids, _, max_new = svc.prepare(messages, tools=None,
                                          kwargs={"enable_thinking": False}, max_new=1)
        except (ValueError, OSError) as e:
            return 400, {"error": {"type": "invalid_request_error", "param": "image_b64",
                                   "message": f"vision: {e}"}}
        variants = [svc.tok.encode(lbl, parse_special=True)[:1]
                    for lbl in qdef["render_labels"]]
        cancel = threading.Event()
        emb = getattr(svc.embeddings, "path", None)   # this qdef's prepare() staged it
        svc.embeddings.path = None
        try:
            for _ in svc.engine.generate(ids, 1, {}, cancel, embeddings=emb):
                pass                                 # the one decode step past the image
        finally:
            if emb:
                with contextlib.suppress(OSError):
                    Path(emb).unlink(missing_ok=True)
        tokens = sorted({t for v in variants for t in v})
        row = svc.engine.final_row_logits(ids, tokens)
        try:
            scores = readout_scores(row, variants)
        except ValueError as e:
            return _err("image_b64", f"logits readout failed: {e}")
        pr_raw = softmax(scores)
        pr = [round(x * 10000.0) / 10000.0 for x in pr_raw]
        t = sum(pr)
        if t > 0:
            pr = [x / t for x in pr]
        pr, calib_id = jev_calib.apply(qdef["type"], pr, "D")   # P0-1: per-tuple apply
        answers[qdef["id"]] = answers_from_calibrated(qdef["type"], pr, calib_id)
        jev_calib.log_enqueue({
            "ts": int(time.time()),
            "tuple": jev_calib.tuple_of("D", qdef["type"]),   # D = direct image readout
            "qid": qdef["id"], "type": qdef["type"],
            "raw_p": pr_raw})
        in_tokens += len(ids)
        readouts += 1
    return 200, {"answers": answers,
                 "usage": {"prompt_tokens": in_tokens, "readout_tokens": readouts},
                 "stages_used": ["image-direct"]}


def handle_image(svc, req: dict, mode: str = "L") -> tuple[int, dict]:
    """POST /v1/systemone with image_b64 — M3.8 two-call pattern, ONE request on the wire:
    (1) the image is encoded and a bounded summary generated through Strata's existing
    per-preset vision path (svc.prepare does the encode + image-pad expansion), then
    (2) the summary becomes the decision STATE and the standard /v1/systemone readout
    (L or V mode) answers the questions.  Donor contract (server.cpp:2419-2603):
    image_b64 + optional vision_prompt (default 'Describe this game screen:') ->
    image_summary + answers + stages_used ['vision', 'decision']."""
    if not isinstance(req, dict) or "questions" not in req or not req.get("image_b64"):
        return _err("image_b64", "required for the image decision path")
    b64 = req["image_b64"]
    if not isinstance(b64, str) or len(b64) < 64:
        return _err("image_b64", "too short")
    if svc.vision is None:
        return 400, {"error": {"type": "invalid_request_error",
                               "param": "image_b64",
                               "message": "vision not loaded on this server (no vision entry in the config)"}}
    qdefs, err_body = parse_questions(req["questions"])
    if qdefs is None:
        return err_body
    vprompt = req.get("vision_prompt") or VISION_PROMPT_DEFAULT
    image_mode = req.get("image_mode", "summary")
    if image_mode not in ("summary", "direct"):
        return _err("image_mode", "must be 'summary' (two-call) or 'direct' (one readout)")

    if image_mode == "direct":
        if mode == "V":
            # P0-3 (Opus plan): D-V does not exist; refuse loudly, not silently
            return 422, {"error": {"type": "invalid_request_error", "field": "X-Jev-Mode",
                                   "message": "X-Jev-Mode: V is not supported with image_mode "
                                              "'direct' (the yes/no branch readout needs a text "
                                              "state); use image_mode 'summary' or omit the header"}}
        return _handle_image_direct(svc, req, mode, qdefs, b64)

    # ---- stage 1: vision summary (Strata's existing path, engine FIFO held) ----
    data_uri = "data:image/png;base64," + b64 if not b64.startswith("data:") else b64
    messages = [{"role": "user",
                 "content": [{"type": "image", "source": data_uri},
                             {"type": "text", "text": vprompt}]}]
    try:
        ids, _, max_new = svc.prepare(messages, tools=None,
                                      kwargs={"enable_thinking": False}, max_new=160)
    except (ValueError, OSError) as e:
        return 400, {"error": {"type": "invalid_request_error", "param": "image_b64",
                               "message": f"vision: {e}"}}
    cancel = threading.Event()
    out = []
    from serve.server import Detokenizer              # late import: server.py loads jev first
    detok = Detokenizer(svc.tok)
    parts = []
    emb = getattr(svc.embeddings, "path", None)       # prepare() staged it; run() passes the
    svc.embeddings.path = None                        # same to generate as GENI (else: black image)
    try:
        with svc.fifo:
            for tok in svc.engine.generate(ids, min(max_new, 160), {}, cancel, embeddings=emb):
                if tok is None or tok < 0:            # engine sentinel ids (verification artifacts)
                    continue
                out.append(tok)
                piece = detok.push(tok)
                if piece:
                    parts.append(piece)
                if len(out) >= 160:
                    break
    finally:
        if emb:
            with contextlib.suppress(OSError):
                Path(emb).unlink(missing_ok=True)     # run() deletes its own; we are the run here
    summary = "".join(parts).split("<|im_end|>")[0].strip()

    # ---- stage 2: decision on the summary state (the standard readout) ----
    sub = {k: v for k, v in req.items() if k not in ("image_b64", "vision_prompt")}
    sub["state"] = {"image_summary": summary} if summary else summary
    if not summary:
        return 502, {"error": {"type": "server_error",
                               "message": "vision stage produced an empty summary; "
                                          "the decision stage was not run (fail-closed)"}}
    code, body = handle(svc, sub, mode)
    if code != 200:
        return code, body
    body["image_summary"] = summary
    body["stages_used"] = ["vision", "decision"]
    return code, body


THEN_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}


def handle_then(svc, req: dict, mode: str = "L") -> tuple[int, dict]:
    """POST /v1/systemone with "then": {"generate": {...}} — amendment 7 mode 1:
    decide first (the standard readout on this request's state/questions), then
    generate with the answers folded into the LAST USER TURN as
    'Jev decisions (JSON): {...}' (a trailing system message throws in this chat
    template - donor folds it too).  Response = decision body + then {text,
    finish_reason, usage}.  Sampling mirrors the chat endpoint (donor defaults
    temp 0.6, top_p 0.95, top_k 20; max_tokens default 512)."""
    gen = ((req or {}).get("then") or {}).get("generate")
    if not isinstance(gen, dict) or not gen.get("messages"):
        return _err("then.generate.messages", "required non-empty array")
    sub = {k: v for k, v in req.items() if k != "then"}
    code, dec = handle(svc, sub, mode)
    if code != 200:
        return code, dec
    answers = dec.get("answers", {})

    messages = [m for m in gen["messages"] if isinstance(m, dict) and m.get("role")]
    injected = False
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            text = messages[i].get("content") if isinstance(messages[i].get("content"), str) else \
                "".join(p.get("text", "") for p in (messages[i].get("content") or [])
                        if isinstance(p, dict) and p.get("type") == "text")
            messages[i] = {**messages[i],
                           "content": text + "\n\nJev decisions (JSON): " + json.dumps(answers, indent=2)}
            injected = True
            break
    if not injected:
        messages.append({"role": "user",
                         "content": "Jev decisions (JSON): " + json.dumps(answers, indent=2)})

    sampling = {**THEN_SAMPLING}
    for k in ("temperature", "top_p", "top_k", "min_p", "seed"):
        if isinstance(gen.get(k), (int, float)) and not isinstance(gen[k], bool):
            sampling[k] = gen[k]
    max_tokens = gen.get("max_tokens") if isinstance(gen.get("max_tokens"), int) else 512
    max_tokens = max(1, max_tokens)
    stops = gen.get("stop")
    stops = ([stops] if isinstance(stops, str) else
             [s for s in stops if isinstance(s, str)] if isinstance(stops, list) else [])

    prompt = svc.template.render(messages, add_generation_prompt=True, enable_thinking=False)
    ids = svc.tok.encode(prompt, parse_special=True)
    if not ids:
        return 400, {"error": {"type": "invalid_request_error",
                               "message": "then.generate: prompt tokenization failed"}}
    cancel = threading.Event()
    out = []
    finish = "stop"
    from serve.server import Detokenizer              # late import: server.py loads jev first
    detok = Detokenizer(svc.tok)                      # the server's incremental decoder
    text_parts = []
    with svc.fifo:
        for tok in svc.engine.generate(ids, max_tokens, sampling, cancel):
            if tok is None or tok < 0:              # engine sentinel ids (verification artifacts)
                continue
            out.append(tok)
            piece = detok.push(tok)
            if piece:
                text_parts.append(piece)
                tail = "".join(text_parts[-8:])
                if any(st and st in tail for st in stops):
                    break
            if len(out) >= max_tokens:
                finish = "length"
    text = "".join(text_parts).strip()
    for st in stops:
        if st in text:
            text = text.split(st)[0]
    dec["then"] = {"text": text, "finish_reason": finish,
                   "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out)}}
    return 200, dec


def handle(svc, req: dict, mode: str = "L") -> tuple[int, dict]:
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
    v_mode = mode == "V"
    image_state = isinstance(req.get("state"), dict) and "image_summary" in req.get("state", {})
    answers, in_tokens, readouts = {}, 0, 0
    with svc.fifo:                                       # the engine turn: one request at a time
        for qdef in qdefs:
            use_v = v_mode and qdef["type"] == "choice"   # X-Jev-Mode scopes to choice (donor :767)
            if use_v:
                # ---- V mode (A2.6): shared head, one yes/no branch per option ----
                head = question_head(qdef)
                yes_no = [svc.tok.encode(lbl, parse_special=True)[:1] for lbl in ("No", "Yes")]
                d_scores = []
                for key in qdef["labels"]:
                    prompt = svc.template.render(
                        [{"role": "system", "content": SYSTEM_V1},
                         {"role": "user", "content": state_text + "\n\n" + head + v_branch(qdef, key)}],
                        add_generation_prompt=True,
                        enable_thinking=False)
                    ids = svc.tok.encode(prompt, parse_special=True)
                    if not ids:
                        return _err("state", "tokenization failed")
                    cancel = threading.Event()
                    for _ in engine.generate(ids, 1, {}, cancel):
                        pass
                    tokens = sorted({t for v in yes_no for t in v})
                    row = engine.final_row_logits(ids, tokens)
                    try:
                        s = readout_scores(row, yes_no)    # [logp(No), logp(Yes)]
                    except ValueError as e:
                        return _err("state", f"logits readout failed: {e}")
                    d_scores.append(s[1] - s[0])           # d_i = s_yes - s_no
                    in_tokens += len(ids)
                    readouts += 1
                # p = softmax(d) (donor :1150-1163: round 4dp, renormalize)
                mx = max(d_scores)
                pr_raw = [math.exp(x - mx) for x in d_scores]
                z = sum(pr_raw)
                pr_raw = [x / z for x in pr_raw]
                pr = [round(x * 10000.0) / 10000.0 for x in pr_raw]
                t = sum(pr)
                if t > 0:
                    pr = [x / t for x in pr]
                # donor V applies the choice temperature to softmax(d) (calib_apply :617)
                pr, calib_id = jev_calib.apply("choice", pr, "V")   # P0-1: per-tuple
                best = max(range(len(pr)), key=lambda i: pr[i])
                probs = {name: pr[i] for i, name in enumerate(qdef["labels"])}
                answers[qdef["id"]] = {
                    "type": "choice", "choice": qdef["labels"][best],
                    "probabilities": probs, "confidence": confidence(pr),
                    "calib": calib_id}
                jev_calib.log_enqueue({
                    "ts": int(time.time()),
                    "tuple": jev_calib.tuple_of("V", "choice"),
                    "qid": qdef["id"], "type": "choice",
                    "raw_p": pr_raw, "v_d": d_scores})
                continue
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
            pr, calib_id = jev_calib.apply(qdef["type"], pr, "S" if image_state else "L")
            answers[qdef["id"]] = answers_from_calibrated(qdef["type"], pr, calib_id)
            # M2 B1.1 log row (unrounded raw_p; donor schema minus label_mass, BROW is sparse)
            jev_calib.log_enqueue({
                "ts": int(time.time()),
                "tuple": jev_calib.tuple_of("S" if image_state else "L", qdef["type"]),
                "qid": qdef["id"],
                "type": qdef["type"],
                "raw_p": pr_raw,
            })
            in_tokens += len(ids)
            readouts += 1
    return 200, {"answers": answers,
                 "usage": {"prompt_tokens": in_tokens, "readout_tokens": readouts}}
