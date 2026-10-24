"""GBNF constrained-decoding acceptance on the GPU rig (plan 2026-10-24, Task 12).

Phases, in order:
  0  protocol smoke: raw `strata.exe --serve` pipe - GRAMMAR OK/ERR, grammar= key,
     the grammar+spec refusal, the INFO capability field.
  1  JSON suite: n prompts through response_format json_schema on the grammar path;
     every reply is validated client-side (target: 100% schema pass).
  2  tool suite: n prompts with tool_choice "required"; every reply parses to tool
     calls (target: 0 failures); totals.tool_calls_from_reasoning must not move
     (grammar calls never need rescuing from the thinking).
  3  plain vs constrained tok/s on the same prompt (target: <=10% constrained cost).
  4  multi-slot probe: `slots` parallel constrained vs plain - engine process CPU
     seconds per output token (the per-step mask build cost, spec section 9).

The server runs this checkout's serve/server.py against this checkout's
build/strata.exe with the rig's IQ3_S model. Plain runs ride the shipped defaults
(suffix drafting on); constrained runs run drafted-free by design (spec and drafts
are refused with a grammar), so phase 3/4 compare the shipping default path
against the constrained path, not grammar-against-grammar.

Writes bench/results/<stamp>/grammar-acceptance.json; the console gets the summary.
Run:  python tools/grammar_acceptance.py            (full, model loads twice:
      smoke engine, then the server's engine)
"""
import argparse
import concurrent.futures as cf
import ctypes
import ctypes.wintypes as wt
import hashlib
import io
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
RIG = {
    "exe": str(ROOT / "build" / "strata.exe"),
    "pack": "F:/Strata-data/packs/swift-iq3_s",
    "native": "F:/Strata-data/models/swift-IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
    "profile": "F:/Strata/data/expert-profile.bin",
    "tokenizer": "F:/Strata-data/packs/swift-iq3_s/tokenizer",
    "model_name": "swift-qwen3.8-flash-next-iq3-s",
}


def engine_args(max_context):
    # the IQ pack's start contract: --native + --spec T (T >= 2) + --prefill. Grammar requests
    # turn drafting off per request (suffix/n-gram family); MTP would be a hard refusal.
    return ["--pack", RIG["pack"], "--native", RIG["native"],
            "--resident-budget-gib", "40", "--expert-profile", RIG["profile"],
            "--expert-cache", "auto", "--prefill", "auto", "--spec", "2",
            "--max-context", str(max_context), "--kv", "int8"]


# ---------------------------------------------------------------- schemas

SCHEMAS = {
    "person": {"type": "object", "additionalProperties": False,
               "properties": {"name": {"type": "string"}, "age": {"type": "integer", "minimum": 0},
                              "city": {"type": "string"},
                              "role": {"type": "string", "enum": ["engineer", "designer", "manager"]}},
               "required": ["name", "age", "city", "role"]},
    "weather": {"type": "object", "additionalProperties": False,
                "properties": {"city": {"type": "string"}, "temp_c": {"type": "number"},
                               "condition": {"type": "string", "enum": ["sunny", "rainy", "cloudy", "snowy"]},
                               "wind_kmh": {"type": "number", "minimum": 0}},
                "required": ["city", "temp_c", "condition", "wind_kmh"]},
    "books": {"type": "object", "additionalProperties": False,
              "properties": {"books": {"type": "array", "minItems": 3, "maxItems": 3,
                                       "items": {"type": "object", "additionalProperties": False,
                                                 "properties": {"title": {"type": "string"},
                                                                "year": {"type": "integer", "minimum": 1800}},
                                                 "required": ["title", "year"]}}},
              "required": ["books"]},
    "recipe": {"type": "object", "additionalProperties": False,
               "properties": {"dish": {"type": "string"},
                              "steps": {"type": "array", "minItems": 4, "maxItems": 12,
                                        "items": {"type": "string"}},
                              "vegetarian": {"type": "boolean"}},
              "required": ["dish", "steps", "vegetarian"]},
    "contact": {"type": "object", "additionalProperties": False,
                "properties": {"full_name": {"type": "string"}, "email": {"type": "string"},
                               "phone": {"type": "string"},
                               "tags": {"type": "array", "maxItems": 6, "items": {"type": "string"}}},
               "required": ["full_name", "email", "phone"]},
}

PROMPTS = {
    "person": [
        "Alice Nakamura is 34, lives in Osaka, and leads the platform team.",
        "Benoit Roux, 41, lives in Lyon; he manages the field engineers.",
        "Chidi Okafor, 28, lives in Lagos; he designs the onboarding flows.",
        "Dana Petrov is 37, lives in Sofia and manages the data team.",
        "Emil Lindqvist, 25, lives in Malmo; he is a designer.",
        "Fatima Al-Sayed, 45, lives in Dubai; she manages partnerships.",
        "Gustavo Souza, 31, lives in Sao Paulo; he engineers the billing service.",
        "Hana Kovac, 29, lives in Zagreb; she designs the mobile app.",
        "Ivan Horvat, 52, lives in Rijeka; he engineers the kernels.",
        "Junko Tanaka, 38, lives in Kyoto; she manages the QA guild.",
    ],
    "weather": [
        "Tokyo today: 24 C, light breeze at 8 km/h, overcast.",
        "Reykjavik is at 2 C with a 40 km/h wind and snow.",
        "Cairo: 36 C, dry and cloudless, wind 12 km/h.",
        "London: 11 C, rain all day, wind 22 km/h.",
        "Phoenix: 41 C, bright sun, wind 5 km/h.",
        "Seattle: 15 C, drizzle, wind 18 km/h.",
        "Nairobi: 21 C, partly cloudy, wind 9 km/h.",
        "Buenos Aires: 28 C, sunny, wind 14 km/h.",
        "Oslo: -1 C, snow, wind 30 km/h.",
        "Mumbai: 31 C, monsoon rain, wind 26 km/h.",
    ],
    "books": [
        "Three classic novels by Jane Austen.",
        "Three novels by Charles Dickens.",
        "Three science-fiction novels by Isaac Asimov.",
        "Three novels by the Bronte sisters (one each).",
        "Three dystopian novels from the 20th century.",
        "Three Russian novels from the 19th century.",
        "Three modern bestsellers by Andy Weir, Blake Crouch and Matt Haig (one each).",
        "Three Lord of the Rings volumes (titles and years).",
        "Three novels by Gabriel Garcia Marquez.",
        "Three German novels from the early 20th century.",
    ],
    "recipe": [
        "Garlic butter pasta for two.",
        "Shakshuka for four.",
        "Thai green curry with chicken.",
        "Neapolitan margherita pizza.",
        "Classic beef chili.",
        "Vegetable fried rice for one.",
        "French onion soup.",
        "Avocado toast with poached eggs.",
        "Chicken tikka masala.",
        "Lemon roast chicken with potatoes.",
    ],
    "contact": [
        "Wei Zhang, wei.zhang@example.com, +86-10-5555-0142.",
        "Priya Nair, priya.nair@example.in, +91-98200-11223.",
        "Lucas Meyer, lucas@meyer-dev.de, +49-30-1234-5678.",
        "Sofia Rossi, sofia.rossi@uniroma.it, +39-06-4455-1122.",
        "James O'Brien, jobrien@example.ie, +353-1-555-0321.",
        "Amara Diallo, amara.d@example.sn, +221-33-800-1122.",
        "Kenji Sato, kenji.sato@example.jp, +81-3-5555-7788.",
        "Elena Petrova, elena.p@example.bg, +359-2-555-4411.",
        "Omar Haddad, omar.haddad@example.ae, +971-4-555-9090.",
        "Nina Berg, nina.berg@example.se, +46-8-555-1234.",
    ],
}

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {"type": "object", "additionalProperties": False,
                       "properties": {"city": {"type": "string"},
                                      "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                       "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "search_flights",
        "description": "One-way flights between two airports.",
        "parameters": {"type": "object", "additionalProperties": False,
                       "properties": {"from_ap": {"type": "string"}, "to_ap": {"type": "string"},
                                      "date": {"type": "string"}},
                       "required": ["from_ap", "to_ap", "date"]}}},
]

TOOL_PROMPTS = [
    f"What is the weather in {c} right now?" for c in
    ["Tokyo", "Paris", "Cairo", "Oslo", "Lima", "Hanoi", "Madrid", "Seoul", "Cape Town", "Reykjavik"]
] + [
    f"Find a one-way flight from {a} to {b} on 2026-11-{d}."
    for a, b, d in [("SFO", "NRT", "03"), ("LHR", "CDG", "05"), ("JFK", "LAX", "07"),
                    ("DXB", "BOM", "09"), ("SYD", "AKL", "11"), ("FRA", "ATH", "13"),
                    ("GRU", "MEX", "15"), ("SIN", "DPS", "17"), ("YYZ", "YVR", "19"),
                    ("CPT", "NBO", "21")]
] + [
    f"What is the weather in {c} right now?" for c in
    ["Lagos", "Bangkok", "Buenos Aires", "Dublin", "Nairobi", "Tel Aviv", "Vienna", "Bogota", "Manila", "Zurich"]
]

# ---------------------------------------------------------------- validator (the venv has no jsonschema)

def validate(value, schema, path="$", errors=None):
    errors = [] if errors is None else errors
    t = schema.get("type")
    if t == "object":
        if not isinstance(value, dict):
            return errors.append(f"{path}: not an object") or errors
        for k in schema.get("required", []):
            if k not in value:
                errors.append(f"{path}: missing required '{k}'")
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for k in value:
                if k not in props:
                    errors.append(f"{path}: extra property '{k}'")
        for k, sub in props.items():
            if k in value:
                validate(value[k], sub, f"{path}.{k}", errors)
    elif t == "array":
        if not isinstance(value, list):
            return errors.append(f"{path}: not an array") or errors
        n = len(value)
        if "minItems" in schema and n < schema["minItems"]:
            errors.append(f"{path}: {n} items < minItems {schema['minItems']}")
        if "maxItems" in schema and n > schema["maxItems"]:
            errors.append(f"{path}: {n} items > maxItems {schema['maxItems']}")
        sub = schema.get("items")
        if sub:
            for i, v in enumerate(value):
                validate(v, sub, f"{path}[{i}]", errors)
    elif t == "string":
        if not isinstance(value, str):
            errors.append(f"{path}: not a string")
        elif "enum" in schema and value not in schema["enum"]:
            errors.append(f"{path}: {value!r} not in enum")
        elif "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(f"{path}: {value!r} fails pattern")
    elif t in ("integer", "number"):
        if t == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
            errors.append(f"{path}: not an integer")
        elif t == "number" and not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append(f"{path}: not a number")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in schema and value < schema["minimum"]:
                errors.append(f"{path}: {value} < minimum {schema['minimum']}")
            if "maximum" in schema and value > schema["maximum"]:
                errors.append(f"{path}: {value} > maximum {schema['maximum']}")
    elif t == "boolean":
        if not isinstance(value, bool):
            errors.append(f"{path}: not a boolean")
    return errors


def post(base, payload, timeout=1200):
    r = requests.post(base + "/v1/chat/completions", json=payload, timeout=timeout)
    return r


# ---------------------------------------------------------------- engine cpu (windows)

class PE32(ctypes.Structure):
    _fields_ = [("dwSize", wt.DWORD), ("cntUsage", wt.DWORD), ("th32ProcessID", wt.DWORD),
                ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)), ("th32ModuleID", wt.DWORD),
                ("cntThreads", wt.DWORD), ("th32ParentProcessID", wt.DWORD),
                ("pcPriClassBase", ctypes.c_long), ("dwFlags", wt.DWORD), ("szExeFile", wt.CHAR * 260)]


def descendants_named(parent, exe_name):
    """The pid of the process named `exe_name` under `parent` (the serve spawns the engine through a
    wrapper, so a direct-children scan can miss it): walk the parent->children map, breadth first."""
    k32 = ctypes.windll.kernel32
    snap = k32.CreateToolhelp32Snapshot(0x2, 0)
    rows, e = [], PE32()
    e.dwSize = ctypes.sizeof(PE32)
    if k32.Process32First(snap, ctypes.byref(e)):
        while True:
            rows.append((int(e.th32ProcessID), int(e.th32ParentProcessID), e.szExeFile))
            if not k32.Process32Next(snap, ctypes.byref(e)):
                break
    k32.CloseHandle(snap)
    kids = {}
    for pid, ppid, name in rows:
        kids.setdefault(ppid, []).append((pid, name))
    seen, queue = set(), [parent]
    while queue:
        for pid, name in kids.get(queue.pop(0), []):
            if pid in seen:
                continue
            seen.add(pid)
            if (name.decode(errors="replace") if isinstance(name, bytes) else name).lower() == exe_name.lower():
                return pid
            queue.append(pid)
    return None


def cpu_seconds(pid):
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, False, pid)          # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return 0.0
    c = wt.FILETIME(); x = wt.FILETIME(); k = wt.FILETIME(); u = wt.FILETIME()
    ok = k32.GetProcessTimes(h, ctypes.byref(c), ctypes.byref(x), ctypes.byref(k), ctypes.byref(u))
    k32.CloseHandle(h)
    if not ok:
        return 0.0
    def secs(ft):
        return ((ft.dwHighDateTime << 32) | ft.dwLowDateTime) / 1e7
    return secs(k) + secs(u)


# ---------------------------------------------------------------- phase 0: protocol smoke

def protocol_smoke(out):
    exe = RIG["exe"]
    proc = subprocess.Popen([exe, "--serve", *engine_args(8192)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, cwd=str(ROOT))
    proc.stdin = io.TextIOWrapper(proc.stdin, encoding="utf-8", newline="\n")
    proc.stdout = io.TextIOWrapper(proc.stdout, encoding="utf-8")
    lines = []

    def send(line):
        proc.stdin.write(line + "\n")
        proc.stdin.flush()

    def expect(pred, what, timeout=1200):
        t0 = time.time()
        while time.time() - t0 < timeout:
            line = proc.stdout.readline()
            if not line:
                out["protocol_smoke"] = {"ok": False, "why": f"engine died waiting for {what}"}
                return False
            lines.append(line.rstrip())
            if pred(line.rstrip()):
                return True
        out["protocol_smoke"] = {"ok": False, "why": f"timeout waiting for {what}: last {lines[-3:]}"}
        return False

    try:
        if not expect(lambda l: l.startswith("INFO"), "INFO line (model loaded)"):
            return False
        info = [l for l in lines if l.startswith("INFO")][-1]
        checks = {}
        checks["info_grammar"] = "grammar=1" in info
        sha = hashlib.sha256(b'root ::= "ab"').hexdigest()[:16]
        send(f'GRAMMAR {sha} 13 root ::= "ab"')
        checks["register_ok"] = expect(lambda l: l == f"GRAMMAR OK {sha}", "GRAMMAR OK")
        send("GRAMMAR 1111111111111111 19 root ::= notdefined")
        checks["register_err"] = expect(lambda l: l.startswith("GRAMMAR ERR 1111111111111111"), "GRAMMAR ERR")
        send(f"GEN 8 seed=7 temperature=0.8 grammar={sha} 1,2,3")
        toks = []

        def collect(line):
            if line.startswith("T "):
                toks.append(int(line.split()[1]))
            return line.startswith("DONE") or line.startswith("ERR")
        checks["gen_with_grammar"] = expect(collect, "DONE") and len(toks) >= 1
        send("GEN 4 grammar=deadbeefdeadbeef 1,2,3")
        checks["not_registered"] = expect(lambda l: l.startswith("ERR grammar deadbeefdeadbeef not registered"),
                                          "not-registered ERR")
        send(f"GEN 4 grammar={sha} spec_min_p=0.5 1,2,3")
        checks["spec_refused"] = expect(lambda l: "grammar + speculative decoding is not supported" in l,
                                        "grammar+spec ERR")
        out["protocol_smoke"] = {"ok": all(checks.values()), "checks": checks,
                                 "info": info, "tokens": toks[:8]}
        return all(checks.values())
    finally:
        proc.kill()


# ---------------------------------------------------------------- server

def start_server(port, slots, log_path):
    cfg = {"exe": RIG["exe"], "args": engine_args(16384), "cwd": str(ROOT),
           "tokenizer": RIG["tokenizer"], "model_name": RIG["model_name"],
           "log": str(log_path), "port": port, "gpu": 0, "host": "127.0.0.1"}
    cfg_path = Path(log_path).with_suffix(".cfg.json")
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(ROOT / "serve" / "server.py"),
                             "--engine", "strata", "--config", str(cfg_path),
                             "--port", str(port), "--no-api-key", "--parallel", str(slots)],
                            stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT))
    base = f"http://127.0.0.1:{port}"
    try:
        if requests.get(base + "/v1/status", timeout=3).status_code == 200:
            raise SystemExit(f"port {port} is already serving: kill the old server first "
                             f"(a leaked serve would take these requests and the pid below would lie)")
    except requests.RequestException:
        pass
    t0 = time.time()
    while time.time() - t0 < 420:   # the model load takes 1-3 minutes (page-locked cache build)
        try:
            if requests.get(base + "/v1/status", timeout=5).status_code == 200:
                return proc, base
        except Exception:
            pass
        if proc.poll() is not None:
            raise SystemExit(f"serve died; see {log_path}")
        time.sleep(2)
    raise SystemExit(f"serve never bound; see {log_path}")


def warm(base):
    r = post(base, {"messages": [{"role": "user", "content": "Reply with the single word: ok"}],
                    "max_tokens": 8}, timeout=2400)
    if r.status_code != 200:
        raise SystemExit(f"warm-up failed {r.status_code}: {r.text[:300]}")


# ---------------------------------------------------------------- phase 1: json suite

def parse_answer(text):
    """One JSON value out of the answer: a code fence around it is a legal grammar path (the think
    prologue's fenced alternative), so strip it the way the serve canary does before parsing."""
    s = (text or "").strip()
    if s.startswith("```"):
        first = s.find("\n")
        if first != -1:
            end = s.find("```", first + 1)
            if end != -1:
                s = s[first + 1:end].strip()
    return json.loads(s)


def json_suite(base, n, out_dir=None):
    jl = open(out_dir / "json-items.jsonl", "a", encoding="utf-8") if out_dir else None

    def rec(row):
        if jl:
            jl.write(json.dumps(row, ensure_ascii=False) + "\n")
            jl.flush()

    items = [(k, PROMPTS[k][i % len(PROMPTS[k])]) for k in SCHEMAS for i in range(n // len(SCHEMAS))]
    passed, canary, other = 0, 0, 0
    fails = []
    for j, (name, prompt) in enumerate(items):
        schema = SCHEMAS[name]
        print(f"  json {j + 1}/{len(items)} {name} ...", end=" ", flush=True)
        t0 = time.time()
        row = {"i": j, "name": name}
        try:
            r = post(base, {"messages": [{"role": "user", "content": prompt + "\nAnswer with one JSON object."}],
                            "response_format": {"type": "json_schema",
                                                "json_schema": {"name": name, "schema": schema, "strict": True}},
                            "max_tokens": 3200, "reasoning_effort": "low", "temperature": 0.6}, timeout=420)
            el = time.time() - t0
            row["s"] = round(el, 1)
            if r.status_code == 502:
                canary += 1
                detail = r.text[:500]
                row["verdict"] = "CANARY"
                row["detail"] = detail
                fails.append((j, name, "canary 502: " + detail[:200]))
                print(f"CANARY {el:.0f}s :: {detail[:120]}", flush=True)
            elif r.status_code != 200:
                other += 1
                detail = r.text[:300]
                row["verdict"] = f"HTTP {r.status_code}"
                row["detail"] = detail
                fails.append((j, name, f"http {r.status_code}: {detail[:200]}"))
                print(f"HTTP {r.status_code} {el:.0f}s", flush=True)
            else:
                body = r.json()
                msg = body["choices"][0]["message"]
                row["finish"] = body["choices"][0].get("finish_reason")
                row["reasoning_len"] = len(msg.get("reasoning_content") or "")
                row["content_len"] = len(msg.get("content") or "")
                row["content_tail"] = (msg.get("content") or "")[-200:]
                try:
                    value = parse_answer(msg.get("content"))
                except Exception as e:
                    other += 1
                    row["verdict"] = "NOT-JSON"
                    row["detail"] = str(e)[:200]
                    fails.append((j, name, f"not json: {e}: {str(msg.get('content'))[:160]}"))
                    print(f"FAIL {el:.0f}s", flush=True)
                else:
                    errs = validate(value, schema)
                    if errs:
                        other += 1
                        row["verdict"] = "SCHEMA-FAIL"
                        row["detail"] = "; ".join(errs[:3])
                        fails.append((j, name, row["detail"]))
                        print(f"FAIL {el:.0f}s :: {row['detail'][:110]}", flush=True)
                    else:
                        passed += 1
                        row["verdict"] = "PASS"
                        print(f"pass {el:.0f}s", flush=True)
        except Exception as e:
            el = time.time() - t0
            other += 1
            row["s"] = round(el, 1)
            row["verdict"] = "EXC"
            row["detail"] = repr(e)[:400]
            fails.append((j, name, row["detail"]))
            print(f"EXC {el:.0f}s", flush=True)
        rec(row)
    if jl:
        jl.close()
    return {"n": len(items), "passed": passed, "canary_502": canary, "other_failures": other,
            "fails": fails[:20]}


# ---------------------------------------------------------------- phase 2: tools suite

def tools_suite(base, n, base_totals, out_dir=None):
    jl = open(out_dir / "tool-items.jsonl", "a", encoding="utf-8") if out_dir else None

    def rec(row):
        if jl:
            jl.write(json.dumps(row, ensure_ascii=False) + "\n")
            jl.flush()

    ok, bad = 0, 0
    fails = []
    for j, prompt in enumerate(TOOL_PROMPTS[:n]):
        print(f"  tool {j + 1}/{len(TOOL_PROMPTS[:n])} ...", end=" ", flush=True)
        t0 = time.time()
        row = {"i": j}
        try:
            r = post(base, {"messages": [{"role": "user", "content": prompt}], "tools": TOOLS,
                            "tool_choice": "required", "max_tokens": 2400, "reasoning_effort": "low",
                            "temperature": 0.6}, timeout=420)
            el = time.time() - t0
            row["s"] = round(el, 1)
            if r.status_code != 200:
                bad += 1
                row["verdict"] = f"HTTP {r.status_code}"
                row["detail"] = r.text[:300]
                fails.append((j, f"http {r.status_code}: {r.text[:200]}"))
                print(f"HTTP {r.status_code} {el:.0f}s", flush=True)
            else:
                d = r.json()
                ch = d["choices"][0]
                calls = ch["message"].get("tool_calls") or []
                names = {t["function"]["name"] for t in TOOLS}
                good = ch.get("finish_reason") == "tool_calls" and len(calls) >= 1
                row["finish"] = ch.get("finish_reason")
                row["reasoning_len"] = len(ch["message"].get("reasoning_content") or "")
                row["calls"] = [(c.get("function") or {}).get("name") for c in calls]
                for c in calls:
                    fn = c.get("function") or {}
                    if fn.get("name") not in names:
                        good = False
                    try:
                        json.loads(fn.get("arguments") or "")
                    except Exception:
                        good = False
                if good:
                    ok += 1
                    row["verdict"] = "OK"
                    print(f"ok {el:.0f}s", flush=True)
                else:
                    bad += 1
                    row["verdict"] = "FAIL"
                    row["detail"] = f"finish={ch.get('finish_reason')} calls={str(calls)[:200]}"
                    fails.append((j, row["detail"]))
                    print(f"FAIL: {row['detail'][:140]} {el:.0f}s", flush=True)
        except Exception as e:
            el = time.time() - t0
            bad += 1
            row["s"] = round(el, 1)
            row["verdict"] = "EXC"
            row["detail"] = repr(e)[:400]
            fails.append((j, row["detail"]))
            print(f"EXC {el:.0f}s", flush=True)
        rec(row)
    if jl:
        jl.close()
    totals = requests.get(base + "/metrics", timeout=30).json()["totals"]
    rescued = totals.get("tool_calls_from_reasoning", 0) - base_totals.get("tool_calls_from_reasoning", 0)
    return {"n": len(TOOL_PROMPTS[:n]), "parsed": ok, "parse_failures": bad,
            "rescued_from_reasoning_delta": rescued, "fails": fails[:20]}


# ---------------------------------------------------------------- phase 3: perf

PERF_SCHEMA = {"type": "object", "additionalProperties": False,
               "properties": {"mountains": {"type": "array", "minItems": 5, "maxItems": 5,
                                            "items": {"type": "object", "additionalProperties": False,
                                                      "properties": {"name": {"type": "string"},
                                                                     "height_m": {"type": "integer"}},
                                                      "required": ["name", "height_m"]}}},
               "required": ["mountains"]}
PERF_PROMPT = ("List the five tallest mountains on Earth with their height in metres. "
               "Be precise; give exactly five.")


def run_once(base, constrained):
    payload = {"messages": [{"role": "user", "content": PERF_PROMPT}],
               "max_tokens": 400, "reasoning_effort": "none", "temperature": 0.8, "seed": 7}
    if constrained:
        payload["response_format"] = {"type": "json_schema",
                                      "json_schema": {"name": "mountains", "schema": PERF_SCHEMA, "strict": True}}
    t0 = time.time()
    r = post(base, payload)
    dt = time.time() - t0
    d = r.json()
    return d["usage"]["completion_tokens"] / dt, d["usage"]["completion_tokens"]


def perf_suite(base, runs):
    plain, constr = [], []
    for i in range(runs):
        tp, _ = run_once(base, False)
        plain.append(tp)
        print(f"  perf plain {i + 1}: {tp:.1f} tok/s", flush=True)
        tc, _ = run_once(base, True)
        constr.append(tc)
        print(f"  perf constrained {i + 1}: {tc:.1f} tok/s", flush=True)
    p, c = statistics.median(plain), statistics.median(constr)
    return {"runs": runs, "plain_med_tok_s": round(p, 2), "constrained_med_tok_s": round(c, 2),
            "constrained_cost_pct": round((p - c) / p * 100.0, 1),
            "plain_all": [round(x, 2) for x in plain], "constrained_all": [round(x, 2) for x in constr]}


# ---------------------------------------------------------------- phase 4: multi-slot probe

PROBE_PROMPT = "Name five rivers in Europe with their countries."


def probe_round(base_url, eng, slots, constrained):
    def one(i):
        payload = {"messages": [{"role": "user", "content": PROBE_PROMPT + f" (set {i})"}],
                   "max_tokens": 600, "reasoning_effort": "none", "temperature": 0.6, "seed": 100 + i}
        if constrained:
            payload["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "rivers", "schema": {"type": "object", "additionalProperties": False,
                                             "properties": {"rivers": {"type": "array", "minItems": 5, "maxItems": 5,
                                                                       "items": {"type": "object",
                                                                                  "additionalProperties": False,
                                                                                  "properties": {"name": {"type": "string"},
                                                                                                 "country": {"type": "string"}},
                                                                                  "required": ["name", "country"]}}},
                                             "required": ["rivers"]}, "strict": True}}
        r = post(base_url, payload)
        d = r.json()
        u = ((d.get("usage") or {}).get("completion_tokens")) or 0
        if r.status_code != 200 or u == 0:
            raise RuntimeError(f"http {r.status_code}: {r.text[:200]}")
        return u
    errs = []

    def safe(i):
        try:
            return one(i)
        except Exception as e:
            errs.append(repr(e)[:250])
            return 0
    cpu0, wall0 = cpu_seconds(eng), time.time()
    with cf.ThreadPoolExecutor(max_workers=slots) as ex:
        toks = sum(ex.map(safe, range(slots)))
    dt, dcpu = time.time() - wall0, cpu_seconds(eng) - cpu0
    if toks == 0:
        return {"error": "; ".join(errs)[:400], "tok_s": 0, "cpu_s": round(dcpu, 2), "cpu_ms_per_tok": 0,
                "tokens": 0}
    return {"tok_s": round(toks / dt, 1), "cpu_s": round(dcpu, 2), "cpu_ms_per_tok": round(1000 * dcpu / toks, 2),
            "tokens": toks, "slot_errors": errs}


def probe_suite(srv_pid, base_url, slots, rounds):
    eng = descendants_named(srv_pid, "strata.exe")
    if eng is None:
        k32 = ctypes.windll.kernel32
        snap = k32.CreateToolhelp32Snapshot(0x2, 0)
        e = PE32()
        e.dwSize = ctypes.sizeof(PE32)
        if k32.Process32First(snap, ctypes.byref(e)):
            while True:
                if (e.szExeFile.decode(errors="replace")
                        if isinstance(e.szExeFile, bytes) else e.szExeFile).lower() == "strata.exe":
                    eng = int(e.th32ProcessID)
                    break
                if not k32.Process32Next(snap, ctypes.byref(e)):
                    break
        k32.CloseHandle(snap)
    print(f"  probe engine pid: {eng}", flush=True)
    plain, constr = [], []
    for i in range(rounds):
        plain.append(probe_round(base_url, eng, slots, False))
        print(f"  probe plain {i + 1}: {plain[-1]}", flush=True)
        constr.append(probe_round(base_url, eng, slots, True))
        print(f"  probe constrained {i + 1}: {constr[-1]}", flush=True)
    med = lambda xs, k: statistics.median([x[k] for x in xs])
    return {"slots": slots, "rounds": rounds, "plain": plain, "constrained": constr,
            "plain_med_cpu_ms_per_tok": med(plain, "cpu_ms_per_tok"),
            "constrained_med_cpu_ms_per_tok": med(constr, "cpu_ms_per_tok"),
            "plain_med_tok_s": med(plain, "tok_s"), "constrained_med_tok_s": med(constr, "tok_s")}


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json-n", type=int, default=50)
    ap.add_argument("--tools-n", type=int, default=30)
    ap.add_argument("--perf-runs", type=int, default=3)
    ap.add_argument("--probe-slots", type=int, default=6)
    ap.add_argument("--probe-rounds", type=int, default=2)
    ap.add_argument("--port", type=int, default=8199)
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--skip-probe", action="store_true")
    ap.add_argument("--stamp", default=time.strftime("%Y-%m-%d-%H%M"))
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    out_dir = ROOT / "bench" / "results" / f"{a.stamp}-grammar-4090"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {"stamp": a.stamp, "rig": "RTX 4090 48GiB, 96GiB RAM", "model": RIG["model_name"],
           "exe": RIG["exe"], "engine_args": engine_args(16384)}
    ok = True

    if not a.skip_smoke:
        print("== phase 0: protocol smoke", flush=True)
        ok &= protocol_smoke(out)
        print("   " + json.dumps(out["protocol_smoke"], default=str)[:400], flush=True)

    print("== starting serve", flush=True)
    srv, base = start_server(a.port, a.probe_slots, out_dir / "serve.log")
    try:
        print("   warming (model load)...", flush=True)
        warm(base)
        status = requests.get(base + "/v1/status", timeout=30).json()
        so = status.get("structured_output") or {}
        out["capability"] = so
        if so.get("method") != "grammar":
            print(f"   NOT grammar-capable: {so}", flush=True)
            ok = False
        totals0 = requests.get(base + "/metrics", timeout=30).json()["totals"]

        print(f"== phase 1: json suite ({a.json_n})", flush=True)
        out["json_suite"] = json_suite(base, a.json_n, out_dir)
        ok &= out["json_suite"]["passed"] == out["json_suite"]["n"]

        print(f"== phase 2: tools suite ({a.tools_n})", flush=True)
        out["tools_suite"] = tools_suite(base, a.tools_n, totals0, out_dir)
        ok &= out["tools_suite"]["parse_failures"] == 0 and out["tools_suite"]["rescued_from_reasoning_delta"] == 0

        print(f"== phase 3: plain vs constrained ({a.perf_runs} runs each)", flush=True)
        out["perf"] = perf_suite(base, a.perf_runs)

        if not a.skip_probe:
            print(f"== phase 4: multi-slot probe ({a.probe_slots} slots)", flush=True)
            out["probe"] = probe_suite(srv.pid, base, a.probe_slots, a.probe_rounds)
    finally:
        srv.terminate()
        srv.wait(timeout=60)

    out["acceptance_ok"] = bool(ok)
    (out_dir / "grammar-acceptance.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in out if k in
                      ("protocol_smoke", "capability", "json_suite", "tools_suite", "perf", "probe", "acceptance_ok")},
                     indent=2)[:2400], flush=True)
    print(f"results: {out_dir / 'grammar-acceptance.json'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
