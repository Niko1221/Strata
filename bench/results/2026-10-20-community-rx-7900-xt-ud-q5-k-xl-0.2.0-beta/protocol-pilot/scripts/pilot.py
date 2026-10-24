#!/usr/bin/env python3
"""Hand-built pilot of docs/BENCHMARK_PROTOCOL.md (strata-qwen38-thinking-v1-draft.1).

Not an official runner (none exists) and NOT conforming: the fixture bundle is not the approved/frozen one,
the tokenizer/template revisions are not pinned. It follows the draft's schedule, settings, fixture construction
and scoring rules as written, so the numbers are comparable in kind. Stdlib only.

  pilot.py calibrate FIXTURES.json     choose the record counts for ~4096 / ~16384 prompt tokens
  pilot.py run FIXTURES.json OUTDIR [--concurrency]
"""
import json, os, sys, time, math, threading, hashlib, datetime, urllib.request, urllib.error

BASE = os.environ.get("STRATA_URL", "http://127.0.0.1:8080")
KEY = os.environ.get("STRATA_KEY", "")
SETTINGS = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0,
            "repetition_penalty": 1.0, "reasoning_effort": "xhigh",
            "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}}
SHORT = "Explain how a hash table finds a value, including collisions and one small example."
LONGQ = "Summarize the structure of these records and explain how to look up a record by its ID."
DESC = "Synthetic benchmark lookup; no external action is performed."
TOOLS = {  # id -> (prompt, name, {prop: json type}, expected args)
    "T1": ("Use lookup_record to look up record A-17.", "lookup_record", {"id": "string"}, {"id": "A-17"}),
    "T2": ("Use get_weather for Zagreb in celsius.", "get_weather", {"city": "string", "unit": "string"},
           {"city": "Zagreb", "unit": "celsius"}),
    "T3": ("Use convert_units to convert 3 km to m.", "convert_units",
           {"value": "number", "from_unit": "string", "to_unit": "string"},
           {"value": 3, "from_unit": "km", "to_unit": "m"}),
    "T4": ("Use get_build_status for project Orion, build 17.", "get_build_status",
           {"project": "string", "build": "integer"}, {"project": "Orion", "build": 17}),
}
TEXT = {
    "I1": ("Return only a JSON object with ok set to true and count set to the integer 3. No other keys.",
           "json", {"ok": True, "count": 3}),
    "I2": ("Sort these integers in ascending order and remove duplicates: 9, 2, -4, 2, 1. Return only a JSON array.",
           "json", [-4, 1, 2, 9]),
    "I3": ("Project: Orion. Build: 17. Return only a JSON object with keys project (string) and build (integer).",
           "json", {"project": "Orion", "build": 17}),
    "I4": ("Reply with exactly the word READY in uppercase and nothing else.", "text", "READY"),
}
RETR = {  # id -> (size, [(fraction, line)], question, kind, expected)
    "R1": (4096, [(0.10, "access_code = ORBIT-731")], "What is access_code? Reply only with its value.", "text", "ORBIT-731"),
    "R2": (4096, [(0.50, "storage_label = CEDAR-284")], "What is storage_label? Reply only with its value.", "text", "CEDAR-284"),
    "R3": (16384, [(0.90, "archive_code = DELTA-906")], "What is archive_code? Reply only with its value.", "text", "DELTA-906"),
    "R4": (16384, [(0.10, "north = N-124"), (0.90, "south = S-875")],
           "Return only a JSON object containing north and south and their values.", "json",
           {"north": "N-124", "south": "S-875"}),
}


def rec(i):
    return f"record{i:06d}: alpha beta gamma delta epsilon zeta eta theta."


def records_text(n, facts=()):
    lines = [rec(i) for i in range(1, n + 1)]
    for frac, line in sorted(facts, key=lambda x: -x[0]):   # insert from the back so indices stay valid
        lines.insert(math.floor(frac * n), line)
    return "\n".join(lines)


def user_msg(rid, body):
    return f"[case {rid}]\n{body}"


def post_stream(body, timeout=1800):
    """One streamed chat request. Returns a dict of everything observed."""
    body = dict(body, model="strata", stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": "Bearer " + KEY} if KEY else {})})
    t0 = time.perf_counter()
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    r = {"started_at": started, "reasoning": [], "content": [], "tool_calls": {}, "first_s": None, "first_kind": None,
         "answer_s": None, "finish_reason": None, "usage": None, "timings": None, "error": None, "chunks": 0}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                d = line[5:].strip()
                if d == "[DONE]":
                    break
                try:
                    j = json.loads(d)
                except ValueError:
                    continue
                r["chunks"] += 1
                if j.get("error"):
                    r["error"] = str(j["error"])[:300]
                if j.get("usage"):
                    r["usage"] = j["usage"]
                if j.get("timings"):
                    r["timings"] = j["timings"]
                for ch in j.get("choices") or []:
                    dl = ch.get("delta") or {}
                    now = time.perf_counter() - t0
                    rs = dl.get("reasoning_content") or dl.get("reasoning") or ""
                    ct = dl.get("content") or ""
                    tc = dl.get("tool_calls") or []
                    if rs:
                        r["reasoning"].append(rs)
                    if ct:
                        r["content"].append(ct)
                    for c in tc:
                        slot = r["tool_calls"].setdefault(c.get("index", 0), {"name": "", "arguments": ""})
                        f = c.get("function") or {}
                        slot["name"] += f.get("name") or ""
                        slot["arguments"] += f.get("arguments") or ""
                    kind = "reasoning" if rs else "answer" if ct else "tool_call" if tc else None
                    if kind and r["first_s"] is None:
                        r["first_s"], r["first_kind"] = now, kind
                    if (ct or tc) and r["answer_s"] is None:
                        r["answer_s"] = now
                    if ch.get("finish_reason"):
                        r["finish_reason"] = ch["finish_reason"]
    except urllib.error.HTTPError as e:
        r["error"] = f"HTTP {e.code}: {e.read()[:200]!r}"
    except Exception as e:  # noqa: BLE001
        r["error"] = repr(e)[:300]
    r["total_s"] = time.perf_counter() - t0
    r["reasoning"] = "".join(r["reasoning"])
    r["content"] = "".join(r["content"])
    return r


def req_body(msgs, cap, seed, tools=None):
    b = dict(SETTINGS, messages=msgs, max_tokens=cap, seed=seed)
    if tools:
        b["tools"] = tools
        b["tool_choice"] = "auto"
    return b


def calibrate(path):
    def ptoks(n):
        b = dict(SETTINGS, messages=[{"role": "user", "content": user_msg(f"calibrate-{n}", records_text(n) + "\n" + LONGQ)}],
                 max_tokens=1, seed=1)
        r = post_stream(b)
        return r["usage"]["prompt_tokens"] if r.get("usage") else None
    out = {}
    a, b = 200, 800
    pa, pb = ptoks(a), ptoks(b)
    per = (pb - pa) / (b - a)
    base = pa - per * a
    for tgt in (4096, 16384):
        n = round((tgt - base) / per)
        best = None
        for cand in range(n - 3, n + 4):
            p = ptoks(cand)
            if best is None or abs(p - tgt) < abs(best[1] - tgt):
                best = (cand, p)
        out[str(tgt)] = {"records": best[0], "prompt_tokens": best[1], "error_pct": round(100 * (best[1] - tgt) / tgt, 2)}
    out["method"] = "server prompt_tokens for the full rendered request incl. the chat template; closest record count"
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(out, indent=1))


def strict_json(text):
    def hook(pairs):
        keys = [k for k, _ in pairs]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate key")
        return dict(pairs)
    return json.loads(text, object_pairs_hook=hook)


def same(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    return type(a) is type(b) and a == b


def score_text(r, kind, expected):
    if r["error"] or r["finish_reason"] not in ("stop",):
        return False, f"not completed (error={r['error']}, finish={r['finish_reason']})"
    t = r["content"].strip(" \t\r\n\f\v")
    if kind == "text":
        return t == expected, "exact text compare"
    try:
        return same(strict_json(t), expected), "strict JSON compare"
    except ValueError as e:
        return False, f"invalid JSON: {e}"


def score_tool(r, name, expected):
    if r["error"] or r["finish_reason"] not in ("tool_calls", "stop"):
        return False, f"not completed (error={r['error']}, finish={r['finish_reason']})"
    calls = list(r["tool_calls"].values())
    if len(calls) != 1:
        return False, f"{len(calls)} tool calls"
    if calls[0]["name"] != name:
        return False, f"wrong function {calls[0]['name']!r}"
    try:
        args = strict_json(calls[0]["arguments"])
    except ValueError as e:
        return False, f"invalid arguments JSON: {e}"
    return same(args, expected), "strict arguments compare"


def tool_def(name, props):
    return [{"type": "function", "function": {"name": name, "description": DESC, "parameters": {
        "type": "object", "properties": {k: {"type": v} for k, v in props.items()},
        "required": list(props), "additionalProperties": False}}}]


def digest(r):
    u, t = r["usage"] or {}, r["timings"] or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    return {
        "prompt_tokens": u.get("prompt_tokens"), "api_output_tokens": u.get("completion_tokens"),
        "cached_tokens_usage": cached, "cache_n_timings": t.get("cache_n"),
        "read_tokens": t.get("prompt_n"), "prompt_ms": t.get("prompt_ms"),
        "prompt_tps_engine": t.get("prompt_per_second"),
        "decode_tokens_engine": t.get("predicted_n"), "decode_ms": t.get("predicted_ms"),
        "decode_tps_engine": t.get("predicted_per_second"),
        "ttft_s": r["first_s"], "ttft_kind": r["first_kind"], "first_answer_s": r["answer_s"], "total_s": r["total_s"],
        "finish_reason": r["finish_reason"], "error": r["error"]}


def run(fx_path, outdir, concurrency):
    fx = json.load(open(fx_path))
    n4, n16 = fx["4096"]["records"], fx["16384"]["records"]
    os.makedirs(outdir, exist_ok=True)
    outputs = open(os.path.join(outdir, "outputs.jsonl"), "w")
    results = []

    def record(rid, phase, case, rep, seed, cap, r, extra=None):
        d = digest(r)
        item = {"request_id": rid, "phase": phase, "case": case, "repetition": rep, "seed": seed, "cap": cap, **d,
                "started_at": r["started_at"], **(extra or {})}
        results.append(item)
        outputs.write(json.dumps({"request_id": rid, "reasoning": r["reasoning"], "content": r["content"],
                                  "tool_calls": list(r["tool_calls"].values()), "timings": r["timings"],
                                  "usage": r["usage"]}) + "\n")
        outputs.flush()
        print(f"{rid}: {d['finish_reason']} out={d['api_output_tokens']} ttft={d['ttft_s'] and round(d['ttft_s'],2)} "
              f"pp={d['prompt_tps_engine']} dec={d['decode_tps_engine']} err={d['error']}", flush=True)
        return item

    def msgs(rid, body):
        return [{"role": "user", "content": user_msg(rid, body)}]

    t_suite = time.perf_counter()
    r = post_stream(req_body(msgs("warmup", SHORT), 128, 1))
    record("warmup", "warmup", "warmup", 1, 1, 128, r)
    for rep in (1, 2, 3):
        for case, body in (("short", SHORT), ("prompt-4k", records_text(n4) + "\n" + LONGQ),
                           ("prompt-16k", records_text(n16) + "\n" + LONGQ)):
            rid = f"performance-{case}-r{rep}"
            r = post_stream(req_body(msgs(rid, body), 512, rep))
            record(rid, "performance", case, rep, rep, 512, r)
    for cid in ("I1", "I2", "I3", "I4"):
        prompt, kind, exp = TEXT[cid]
        rid = f"sanity-{cid}"
        r = post_stream(req_body(msgs(rid, prompt), 2048, 1))
        ok, why = score_text(r, kind, exp)
        record(rid, "sanity", cid, 1, 1, 2048, r, {"passed": ok, "scoring": why})
    for cid in ("R1", "R2", "R3", "R4"):
        size, facts, q, kind, exp = RETR[cid]
        n = n4 if size == 4096 else n16
        rid = f"sanity-{cid}"
        body = records_text(n, facts) + "\n" + q
        r = post_stream(req_body(msgs(rid, body), 2048, 1))
        ok, why = score_text(r, kind, exp)
        record(rid, "sanity", cid, 1, 1, 2048, r, {"passed": ok, "scoring": why})
    for cid in ("T1", "T2", "T3", "T4"):
        prompt, name, props, exp = TOOLS[cid]
        rid = f"sanity-{cid}"
        r = post_stream(req_body(msgs(rid, prompt), 2048, 1, tools=tool_def(name, props)))
        ok, why = score_tool(r, name, exp)
        record(rid, "sanity", cid, 1, 1, 2048, r, {"passed": ok, "scoring": why})
    required_s = time.perf_counter() - t_suite

    rounds = []
    optional_s = None
    if concurrency:
        t_opt = time.perf_counter()
        for rnd in (1, 2, 3):
            lock, got = threading.Lock(), []

            def one(j, rnd=rnd):
                rid = f"concurrency-r{rnd}-c{j}"
                seed = 4 * (rnd - 1) + j
                r = post_stream(req_body(msgs(rid, SHORT), 512, seed))
                with lock:
                    got.append((rid, seed, r, time.perf_counter()))
            t0 = time.perf_counter()
            ths = [threading.Thread(target=one, args=(j,)) for j in (1, 2, 3, 4)]
            [t.start() for t in ths]
            [t.join() for t in ths]
            wall = max(g[3] for g in got) - t0
            toks = 0
            for rid, seed, r, _ in sorted(got, key=lambda g: g[0]):
                item = record(rid, "concurrency", "short-burst", rnd, seed, 512, r, {"round": rnd})
                if not item["error"]:
                    toks += item["api_output_tokens"] or 0
            rounds.append({"round": rnd, "wall_s": wall, "output_tokens": toks, "aggregate_tps": toks / wall if wall else None})
        optional_s = time.perf_counter() - t_opt

    sanity = [x for x in results if x["phase"] == "sanity"]
    summary = {"required_duration_s": required_s, "optional_duration_s": optional_s,
               "sanity_passed": sum(1 for x in sanity if x["passed"]), "sanity_total": 12,
               "sanity_by_category": {c: sum(1 for x in sanity if x["case"].startswith(c) and x["passed"])
                                      for c in "IRT"},
               "concurrency_rounds": rounds}
    json.dump({"format": "hand-run pilot of strata-qwen38-thinking-v1-draft.1 (nonconforming; not the draft's schema)",
               "settings": SETTINGS, "fixtures": fx, "summary": summary, "requests": results},
              open(os.path.join(outdir, "results.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "calibrate":
        calibrate(sys.argv[2])
    else:
        run(sys.argv[2], sys.argv[3], "--concurrency" in sys.argv)
