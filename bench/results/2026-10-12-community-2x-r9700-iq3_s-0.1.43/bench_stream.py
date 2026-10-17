#!/usr/bin/env python3
"""Shared client for tune_bench.py, decode_bench.py and conc_bench.py: one streamed OpenAI-compatible chat completion,
validated.

A request counts as "ok" only when the stream
  - carries no `error` object (Strata's serve/server.py sends one inside an already-open SSE stream when the engine fails),
  - ends with a finish_reason of "stop" or "length",
  - reports usage.completion_tokens (no fallback to counting chunks: chunks are not tokens),
  - ends with `data: [DONE]` (anything else is a premature end of the stream),
  - and has well-formed events: `usage`, `timings` and `prompt_tokens_details` objects when present, token counts
    non-negative integers, `choices` a list of objects, each `delta` an object, the content / reasoning pieces and the
    finish reason strings, timing figures numbers. An event that breaks this ends the read with status `bad_event`
    (`"usage": null` and a chunk without `choices` are allowed).
Every other outcome is returned as a record with status != "ok" and an `error` text; callers keep these records and
leave them out of every speed summary.

The record also carries structured hashes of the output (`hashes`): the reasoning text and the answer text hashed
separately, the completion token count and the finish reason, and one combined hash over all four. Token ids are not
hashed: Strata's OpenAI endpoint returns neither token ids nor logprobs.
"""
import argparse
import hashlib
import http.client
import json
import time
import urllib.error
import urllib.request

OK_FINISH = ("stop", "length")


def positive_int(text):
    """argparse type for counts (--runs, --rounds, --repeat, --max-tokens): a positive integer, else a usage error
    (exit 2) before any request is sent."""
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}")
    if v <= 0:
        raise argparse.ArgumentTypeError(f"must be positive: {v}")
    return v


def _sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def output_hashes(reasoning, answer, completion_tokens, finish_reason):
    """Structured exactness check: two runs are the same output only if all five fields match."""
    canon = json.dumps({"reasoning": reasoning, "answer": answer, "completion_tokens": completion_tokens,
                        "finish_reason": finish_reason}, sort_keys=True, ensure_ascii=False)
    return {"reasoning": _sha(reasoning), "answer": _sha(answer), "completion_tokens": completion_tokens,
            "finish_reason": finish_reason, "combined": _sha(canon), "token_ids": None}


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def schema_error(ev):
    """Return None if a parsed (dict) event has the field types parse_sse reads, else a short description."""
    usage = ev.get("usage")
    if usage is not None:
        if not isinstance(usage, dict):
            return f"usage is {type(usage).__name__}, not an object"
        for k in ("prompt_tokens", "completion_tokens"):
            v = usage.get(k)
            if v is not None and not (_is_int(v) and v >= 0):
                return f"usage.{k} is {v!r}, not a non-negative integer"
        det = usage.get("prompt_tokens_details")
        if det is not None:
            if not isinstance(det, dict):
                return f"usage.prompt_tokens_details is {type(det).__name__}, not an object"
            v = det.get("cached_tokens")
            if v is not None and not (_is_int(v) and v >= 0):
                return f"usage.prompt_tokens_details.cached_tokens is {v!r}, not a non-negative integer"
    timings = ev.get("timings")
    if timings is not None:
        if not isinstance(timings, dict):
            return f"timings is {type(timings).__name__}, not an object"
        for k in ("predicted_per_second", "prompt_per_second"):
            v = timings.get(k)
            if v is not None and not _is_num(v):
                return f"timings.{k} is {v!r}, not a number"
    choices = ev.get("choices")
    if choices is not None:
        if not isinstance(choices, list):
            return f"choices is {type(choices).__name__}, not a list"
        for i, ch in enumerate(choices):
            if not isinstance(ch, dict):
                return f"choices[{i}] is {type(ch).__name__}, not an object"
            d = ch.get("delta")
            if d is not None:
                if not isinstance(d, dict):
                    return f"choices[{i}].delta is {type(d).__name__}, not an object"
                for k in ("content", "reasoning_content", "reasoning"):
                    v = d.get(k)
                    if v is not None and not isinstance(v, str):
                        return f"choices[{i}].delta.{k} is {type(v).__name__}, not a string"
            f = ch.get("finish_reason")
            if f is not None and not isinstance(f, str):
                return f"choices[{i}].finish_reason is {type(f).__name__}, not a string"
    return None


def parse_sse(lines, t0, clock=time.time):
    """Read an SSE body (an iterable of raw lines, bytes or str) and return a validated record.

    t0 is the time the request was sent (same clock).  Pure apart from `clock`, so tests feed canned streams."""
    first = last = None
    text, reasoning = [], []
    events = 0
    usage = timings = finish = error = None
    status = None
    done = False
    try:
        for raw in lines:
            line = (raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw).strip()
            if not line.startswith("data:"):
                continue                                   # SSE comments (keep-alives), event: lines, blank lines
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            try:
                ev = json.loads(data)
            except ValueError:
                status, error = "bad_event", f"invalid JSON in the stream: {data[:120]!r}"
                break
            if not isinstance(ev, dict):
                status, error = "bad_event", f"unexpected event: {data[:120]!r}"
                break
            if ev.get("error") is not None:
                e = ev["error"]
                msg = e.get("message") if isinstance(e, dict) else str(e)
                status, error = "server_error", f"error event in the stream: {msg}"
                break
            why = schema_error(ev)
            if why is not None:
                status, error = "bad_event", f"malformed event ({why}): {data[:120]!r}"
                break
            if ev.get("usage"):
                usage = ev["usage"]
            if ev.get("timings"):
                timings = ev["timings"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                c = d.get("content")
                r = d.get("reasoning_content") or d.get("reasoning")
                if c or r:
                    now = clock()
                    if first is None:
                        first = now
                    last = now
                    events += 1
                    if c:
                        text.append(c)
                    if r:
                        reasoning.append(r)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    except (OSError, http.client.HTTPException, UnicodeDecodeError) as e:   # reset, timeout, IncompleteRead
        status, error = "read_error", f"the stream broke: {type(e).__name__}: {e}"

    ct = (usage or {}).get("completion_tokens")
    if status is None:
        if not done:
            status, error = "truncated", "the stream ended without [DONE]" + ("" if finish else " and without a finish_reason")
        elif finish is None:
            status, error = "truncated", "no finish_reason before [DONE]"
        elif finish not in OK_FINISH:
            status, error = "bad_finish", f"finish_reason {finish!r}"
        elif not isinstance(ct, int) or isinstance(ct, bool) or ct < 0:
            status, error = "no_usage", "usage.completion_tokens missing"
        else:
            status = "ok"
    answer, thought = "".join(text), "".join(reasoning)
    ok = status == "ok"
    span = (last - first) if (first is not None and last is not None and last > first) else None
    return {
        "status": status, "error": error, "finish_reason": finish,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "completion_tokens": ct if isinstance(ct, int) else None,
        "cached_tokens": ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
        "engine_tps": (timings or {}).get("predicted_per_second"),
        "engine_pp_tps": (timings or {}).get("prompt_per_second"),
        "ttft": (first - t0) if first is not None else None,
        # decode tok/s = (completion tokens - 1) / (last streamed token - first streamed token); only for ok streams
        "tps": (ct - 1) / span if ok and span and ct and ct > 1 else None,
        "content_events": events,
        "hashes": output_hashes(thought, answer, ct if isinstance(ct, int) else None, finish),
        "text": answer, "reasoning": thought,
    }


def stream(url, model, messages, temp, max_tokens, seed=None, timeout=3600, top_p=0.95, top_k=20):
    """POST one streamed chat completion and return parse_sse's record plus request facts and wall-clock times.
    Never raises: HTTP and network failures come back as records with status http_error / connect_error, and any other
    exception, while building the request (e.g. an invalid URL) or while reading the stream, as client_error."""
    facts = {"temperature": temp, "max_tokens": max_tokens, "seed": seed}
    t0 = time.time()
    try:
        body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temp, "top_p": top_p,
                "top_k": top_k, "stream": True, "stream_options": {"include_usage": True}}
        if seed is not None:
            body["seed"] = seed
        req = urllib.request.Request(str(url).rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        try:
            detail = e.read()[:300].decode("utf-8", "replace")
        except Exception:                                    # noqa: BLE001 - the detail is best effort
            detail = ""
        rec = failed("http_error", f"HTTP {e.code}: {detail}")
    except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
        rec = failed("connect_error", f"{type(e).__name__}: {e}")
    except Exception as e:                                   # noqa: BLE001 - bad URL or body: a record, not a crash
        rec = failed("client_error", f"request not sent: {type(e).__name__}: {e}")
    else:
        try:
            with resp:
                rec = parse_sse(resp, t0)
        except Exception as e:                               # noqa: BLE001 - a bug here must not end the run
            rec = failed("client_error", f"{type(e).__name__}: {e}")
    rec.update(facts)
    rec["t_start"], rec["t_end"] = t0, time.time()
    return rec


def failed(status, error):
    return {"status": status, "error": error, "finish_reason": None, "prompt_tokens": None, "completion_tokens": None,
            "cached_tokens": None, "engine_tps": None, "engine_pp_tps": None, "ttft": None, "tps": None,
            "content_events": 0, "hashes": None, "text": "", "reasoning": ""}


def public(rec, **extra):
    """The record without the full texts, for JSON output."""
    out = {k: v for k, v in rec.items() if k not in ("text", "reasoning")}
    for k in ("ttft", "tps", "t_start", "t_end", "engine_tps", "engine_pp_tps"):
        if isinstance(out.get(k), float):
            out[k] = round(out[k], 4 if k in ("t_start", "t_end", "ttft") else 2)
    out.update(extra)
    return out
