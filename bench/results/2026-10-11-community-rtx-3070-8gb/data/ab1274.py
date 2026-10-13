#!/usr/bin/env python3
"""A/B for Strata PR #1274 on one server: the default (greedy) against the "thinking" sampling preset.

Both arms are sent explicitly on every request, so the result does not depend on what the server's run config defaults to:
  greedy   : temperature=0
  thinking : temperature=1.0, top_p=0.95, top_k=20, seed=<fixed per prompt and round>    (the set `--thinking` writes in PR #1274)
Everything else is identical: the same server, streaming, reasoning_effort=medium, the same prompt and max_tokens.
Prompts are public or synthetic (no private project content). The order of the two arms alternates.
If a queue relay with a /queue endpoint sits in front of the server, the run is polite: before every request it waits until the
queue is empty (someone else's work is never queued behind a whole round), and it gives up when the model or engine changes
underneath it. A stock server has no /queue: nothing is waited for.

usage:
  ab1274.py run --out DIR [--start-at "YYYY-MM-DD HH:MM"] [--rounds 3] [--deadline-min 110]
  ab1274.py report --out DIR        rebuild RESULTS.md from results.jsonl (and meta.json, if there)
  ab1274.py selftest                the whole flow against an in-process fake server (no model, no key, about a minute)
Stop early: touch DIR/STOP - it finishes the request in flight, then writes the report. Ctrl-C does the same.
AB_BASE is the server address (default http://127.0.0.1:8080). AB_KEY_FILE names a file holding the API key, only if the
server needs one (--api-key); the key is never printed or stored.
"""
import argparse
import collections
import datetime as dt
import http.client
import http.server
import json
import os
import pathlib
import re
import shutil
import socketserver
import statistics
import sys
import tempfile
import threading
import time
import zlib
from itertools import permutations

KEY_FILE = os.environ.get("AB_KEY_FILE", "")      # optional: only a server started with --api-key needs a key
BASE = os.environ.get("AB_BASE", "http://127.0.0.1:8080")
EFFORT = "medium"
ARMS = {"greedy": {"temperature": 0}, "thinking": {"temperature": 1.0, "top_p": 0.95, "top_k": 20}}
HARD_CAP_S = 25 * 60          # one request, wall clock
MAX_FAILS_IN_A_ROW = 5        # then the run stops: something is wrong that waiting for an empty queue does not fix
IDLE_PATIENCE_S = 15 * 60     # how long to wait for an empty queue (or a reachable engine) before the run gives up
POLL_S, IDLE_POLLS = 2.0, 1   # the queue must be empty on this many polls in a row, this far apart (selftest shortens it)
STATUS_KEYS = ("reasoning_recoveries", "generated", "prompt_tokens", "max_tokens")   # numbers only: /status also holds text


class Abort(Exception):
    """Ends the run. What is already measured stays in results.jsonl and goes into the report.
    retry=True: the cause may pass (a busy queue, an engine that is not reachable yet), so the wrapper may start again."""
    def __init__(self, msg: str, retry: bool = False):
        super().__init__(msg)
        self.retry = retry


def log(msg: str) -> None:
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ the prompts
def math_problems() -> list:
    """(question, answer): the answers are computed here, not typed."""
    def change(total, coins):
        ways = [1] + [0] * total
        for c in coins:
            for n in range(c, total + 1):
                ways[n] += ways[n - c]
        return ways[total]

    def lattice(n):                      # right/up paths from (0,0) to (n,n) that never go above y = x
        grid = [[0] * (n + 1) for _ in range(n + 1)]
        grid[0][0] = 1
        for x in range(n + 1):
            for y in range(min(x, n) + 1):
                if (x, y) != (0, 0):
                    grid[x][y] = (grid[x - 1][y] if x else 0) + (grid[x][y - 1] if y else 0)
        return grid[n][n]

    seat = sum(1 for p in permutations(range(1, 8)) if p[0] != 1 and p[-1] != 1)     # person 1 is not next to person 0
    digit_sum_20 = sum(1 for n in range(1, 10 ** 6 + 1) if sum(map(int, str(n))) == 20)
    triples = sum(1 for x in range(16) for y in range(16) if 0 <= 30 - x - y <= 15)
    subsets = sum(1 for mask in range(1 << 14) if sum(i + 1 for i in range(14) if mask >> i & 1) % 5 == 0)
    strings = sum(1 for mask in range(1 << 14) if bin(mask).count("1") == 6 and "111" not in format(mask, "014b"))
    return [
        ("What is the remainder when 7^222 is divided by 13?", pow(7, 222, 13)),
        ("How many integers from 1 to 1000 inclusive are divisible by 3 or by 5, but not by 15?",
         sum(1 for n in range(1, 1001) if (n % 3 == 0 or n % 5 == 0) and n % 15)),
        ("What is the sum of the decimal digits of 2^100?", sum(map(int, str(2 ** 100)))),
        ("Eight people, including Ana and Ben, sit around a round table. Only the relative order around the table matters "
         "(rotations of one arrangement count as the same). In how many arrangements are Ana and Ben not next to each other?", seat),
        ("In how many ways can you make exactly 100 cents using any number of pennies (1), nickels (5), dimes (10) and quarters (25)? "
         "The order of coins does not matter.", change(100, (1, 5, 10, 25))),
        ("How many paths go from (0,0) to (6,6) using only unit steps right or up, and never go above the line y = x "
         "(every point (x, y) on the path has y <= x)?", lattice(6)),
        # four harder counting problems: the ones where a model can fail or loop
        ("How many positive integers up to 1,000,000 have a decimal digit sum of exactly 20?", digit_sum_20),
        ("How many ordered triples (x, y, z) of integers with 0 <= x, y, z <= 15 satisfy x + y + z = 30?", triples),
        ("How many subsets of {1, 2, ..., 14}, including the empty set, have a sum that is divisible by 5?", subsets),
        ("How many binary strings of length 14 have exactly six 1s and no three 1s in a row?", strings),
    ]


def _tool(name, description, props, required):
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": {"type": "object", "properties": props, "required": required}}}


TOOLS = [
    _tool("get_weather", "Current weather for a city.",
          {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, ["city"]),
    _tool("search_files", "Find files whose name matches a glob, optionally only those containing a text.",
          {"directory": {"type": "string"}, "glob": {"type": "string"}, "contains": {"type": "string"}}, ["directory", "glob"]),
    _tool("read_file", "Read lines of a text file.",
          {"path": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, ["path"]),
    _tool("run_command", "Run a shell command and return its output.",
          {"command": {"type": "string"}, "timeout_s": {"type": "integer"}}, ["command"]),
    _tool("create_event", "Create a calendar event.",
          {"title": {"type": "string"}, "date": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"},
           "time": {"type": "string", "pattern": r"^\d{2}:\d{2}$"}, "attendees": {"type": "array", "items": {"type": "string"}}},
          ["title", "date", "time"]),
]
TOOL_PROMPTS = [
    ("What is the weather in Seoul right now in celsius, and in Austin in fahrenheit?", {"get_weather": 2}),
    ("Search the directory /srv/app for files matching *.py that contain the text TODO.", {"search_files": 1}),
    ("Show me lines 10 to 25 of /srv/app/main.py.", {"read_file": 1}),
    ("Schedule a meeting titled 'Design review' on 2026-11-03 at 14:30 with ana@example.com and ben@example.com.",
     {"create_event": 1}),
    ("Run the unit tests with the command `pytest -q tests/`, and also show me the first 5 lines of /srv/app/README.md.",
     {"run_command": 1, "read_file": 1}),
]
CODE_PROMPTS = [
    "Write a Python function merge_intervals(intervals) that merges overlapping closed intervals given as (start, end) tuples "
    "and returns them sorted. Add five assert-based tests. Answer with one code block and nothing else.",
    "Write a JavaScript function debounce(fn, ms) that returns a debounced function with a .cancel() method, and a short usage "
    "example. Answer with one code block and nothing else.",
]
CHAT_PROMPTS = [
    "Explain in five sentences how a hash table handles collisions.",
    "Translate into Korean, then into French: 'The meeting has been moved to Thursday afternoon because the room is not available.'",
]


def build_items() -> list:
    items = []
    for i, (question, answer) in enumerate(math_problems(), 1):
        items.append({"id": f"math{i}", "workload": "math", "max_tokens": 6000, "answer": answer,
                      "user": question + "\n\nThink it through, then end with one final line exactly like: ANSWER: <integer>"})
    for i, (prompt, expect) in enumerate(TOOL_PROMPTS, 1):
        items.append({"id": f"tool{i}", "workload": "tools", "max_tokens": 2500, "expect": expect, "user": prompt, "tools": True,
                      "system": "You are a coding assistant with tools. Call the tools you need; do not guess results."})
    items += [{"id": f"code{i}", "workload": "code", "max_tokens": 3500, "user": p} for i, p in enumerate(CODE_PROMPTS, 1)]
    items += [{"id": f"chat{i}", "workload": "chat", "max_tokens": 1500, "user": p} for i, p in enumerate(CHAT_PROMPTS, 1)]
    return items


# ------------------------------------------------------------------------------------------------ the server
class Server:
    def __init__(self, base: str, key: str):
        host, _, port = base.split("//", 1)[1].partition(":")
        self.host, self.port = host, int(port or 80)
        self.auth = {"Authorization": "Bearer " + key} if key else {}

    def _conn(self, timeout: float) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(self.host, self.port, timeout=timeout)

    def get(self, path: str, timeout: float = 15.0):
        """(status, parsed JSON or None); (0, None) when the relay cannot be reached."""
        try:
            conn = self._conn(timeout)
            conn.request("GET", path, headers=self.auth)
            resp = conn.getresponse()
            raw = resp.read()
            conn.close()
        except (OSError, http.client.HTTPException):
            return 0, None                      # intentional: unreachable is a state the callers wait out, not a crash
        try:
            return resp.status, json.loads(raw)
        except ValueError:
            return resp.status, None            # intentional: a non-JSON body (an HTML error page) counts as "no answer"

    def chat_stream(self, body: dict, cap_s: float = HARD_CAP_S) -> dict:
        t0 = time.time()
        res = {"http_status": None, "error": None, "ttft_s": None, "first_data_s": None, "n_keepalive": 0, "content": [],
               "reasoning": [], "calls": {}, "finish_reason": None, "usage": None, "timings": None}
        conn = self._conn(300)
        try:
            payload = json.dumps(body).encode()
            conn.request("POST", "/v1/chat/completions", payload,
                         {**self.auth, "Content-Type": "application/json", "Accept": "text/event-stream"})
            resp = conn.getresponse()
            res["http_status"] = resp.status
            if resp.status == 200:
                read_sse(resp, res, t0, cap_s)
            else:
                res["error"] = resp.read(2000).decode("utf-8", "replace")[:500]
        except (OSError, http.client.HTTPException) as exc:
            res["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            conn.close()
        res["total_s"] = round(time.time() - t0, 2)
        return res


def read_sse(resp, res: dict, t0: float, cap_s: float) -> None:
    while True:
        if time.time() - t0 > cap_s:
            res["error"] = f"cap of {cap_s:.0f} s reached"
            return
        line = resp.readline()
        if not line:
            return
        text = line.decode("utf-8", "replace").rstrip("\r\n")
        if text.startswith(":"):
            res["n_keepalive"] += 1
        elif text.startswith("data:") and text[5:].strip() != "[DONE]":
            try:
                chunk = json.loads(text[5:])
            except ValueError:
                res["error"] = "an SSE data line was not JSON"
                return
            if res["first_data_s"] is None:
                res["first_data_s"] = round(time.time() - t0, 2)
            absorb_chunk(res, chunk, t0)


def absorb_chunk(res: dict, chunk: dict, t0: float) -> None:
    if chunk.get("error"):
        res["error"] = json.dumps(chunk["error"])[:500]
        return
    choice = (chunk.get("choices") or [{}])[0]
    delta, seen = choice.get("delta") or {}, False
    for field, bucket in (("content", "content"), ("reasoning_content", "reasoning")):
        if delta.get(field):
            res[bucket].append(delta[field])
            seen = True
    for piece in delta.get("tool_calls") or []:
        call = res["calls"].setdefault(piece.get("index", len(res["calls"])), {"name": "", "arguments": ""})
        fn = piece.get("function") or {}
        call["name"] += fn.get("name") or ""
        call["arguments"] += fn.get("arguments") or ""
        seen = True
    if seen and res["ttft_s"] is None:
        res["ttft_s"] = round(time.time() - t0, 2)
    for field in ("usage", "timings"):
        if chunk.get(field):
            res[field] = chunk[field]
    if choice.get("finish_reason"):
        res["finish_reason"] = choice["finish_reason"]


# ------------------------------------------------------------------------------------------------ waiting politely
def check_same_engine(up: dict, state: dict) -> None:
    sig = (up.get("model"), up.get("engine"))
    if state.setdefault("sig", sig) != sig:
        raise Abort(f"the model or engine changed during the run: {state['sig']} -> {sig}")
    uptime = up.get("uptime_s")
    if isinstance(uptime, (int, float)):
        if uptime < state.get("uptime", 0) - 2:
            raise Abort("the engine restarted during the run (its uptime went backwards)")
        state["uptime"] = uptime


def wait_idle(srv: Server, state: dict, stop_path: pathlib.Path) -> dict:
    """The relay's /queue once it has been empty on IDLE_POLLS polls in a row (and the engine is reachable)."""
    t0, empty = time.time(), 0
    while True:
        if stop_path.exists():
            raise Abort("the STOP file was found")
        status, q = srv.get("/queue")
        if status == 404:                       # a stock server has no queue relay in front: nothing is queued behind this script
            return {}
        up = (q or {}).get("upstream") or {}
        if status == 200 and up.get("reachable") and not q.get("running") and not q.get("waiting"):
            check_same_engine(up, state)
            empty += 1
            if empty >= IDLE_POLLS:
                return q
        else:
            empty = 0
        if time.time() - t0 > IDLE_PATIENCE_S:
            raise Abort(f"the queue was not empty (or the engine not reachable) for {IDLE_PATIENCE_S / 60:.0f} minutes", retry=True)
        time.sleep(POLL_S)


def wait_until(start_at: str, stop_path: pathlib.Path) -> None:
    if not start_at:
        return
    target = dt.datetime.strptime(start_at, "%Y-%m-%d %H:%M")
    log(f"waiting for the start time {target:%Y-%m-%d %H:%M} (local time); touch {stop_path} to cancel")
    last_note = 0.0
    while dt.datetime.now() < target:
        if stop_path.exists():
            raise Abort("the STOP file was found before the start time")
        if time.time() - last_note > 1800:
            log(f"still waiting, {(target - dt.datetime.now()).total_seconds() / 60:.0f} minutes to go")
            last_note = time.time()
        time.sleep(min(30.0, max(0.2, (target - dt.datetime.now()).total_seconds())))


# ------------------------------------------------------------------------------------------------ scoring
def schema_problems(value, schema: dict, path: str) -> list:
    kind = schema.get("type")
    py = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}.get(kind)
    if py is not None and (not isinstance(value, py) or (isinstance(value, bool) and kind in ("integer", "number"))):
        return [f"{path}: expected {kind}"]
    problems = []
    if "enum" in schema and value not in schema["enum"]:
        problems.append(f"{path}: {value!r} is not one of {schema['enum']}")
    if "pattern" in schema and isinstance(value, str) and not re.search(schema["pattern"], value):
        problems.append(f"{path}: {value!r} does not match {schema['pattern']}")
    if kind == "object":
        problems += [f"{path}.{name}: missing" for name in schema.get("required", []) if name not in value]
        for name, sub in schema.get("properties", {}).items():
            if name in value:
                problems += schema_problems(value[name], sub, f"{path}.{name}")
    if kind == "array":
        for i, v in enumerate(value):
            problems += schema_problems(v, schema.get("items", {}), f"{path}[{i}]")
    return problems


def score_tools(item: dict, res: dict) -> dict:
    calls = [res["calls"][i] for i in sorted(res["calls"])]
    specs = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}
    problems = []
    for call in calls:
        if call["name"] not in specs:
            problems.append(f"unknown tool {call['name']!r}")
            continue
        try:
            args = json.loads(call["arguments"] or "{}")
        except ValueError:
            problems.append(f"{call['name']}: the arguments are not JSON")
            continue
        problems += schema_problems(args, specs[call["name"]], call["name"])
    have = collections.Counter(c["name"] for c in calls)
    complete = bool(calls) and not problems and all(have[name] >= n for name, n in item["expect"].items())
    return {"n_calls": len(calls), "problems": problems, "valid": bool(calls) and not problems, "complete": complete}


def score(item: dict, res: dict) -> dict:
    if item["workload"] == "math":
        found = re.findall(r"ANSWER:\s*[*`$]*\s*(-?\d[\d,]*)", "".join(res["content"]))
        got = int(found[-1].replace(",", "")) if found else None
        return {"answer": got, "correct": got == item["answer"]}
    if item["workload"] == "tools":
        return score_tools(item, res)
    return {}


def repetition(text: str) -> dict:
    """Is the end of the text a loop? Compression ratio of the last 4000 characters, and the share of 40-character windows that repeat."""
    tail = text[-4000:]
    if len(tail) < 1500:
        return {"ratio": None, "dup": None, "looped": False}
    raw = tail.encode("utf-8")
    ratio = len(zlib.compress(raw, 6)) / len(raw)
    windows = [tail[i:i + 40] for i in range(0, len(tail) - 40, 20)]
    counts = collections.Counter(windows)
    dup = sum(c for c in counts.values() if c > 1) / len(windows)
    return {"ratio": round(ratio, 3), "dup": round(dup, 3), "looped": ratio < 0.18 or dup > 0.5}


# ------------------------------------------------------------------------------------------------ one request
def request_body(item: dict, arm: str, seed: int, model: str) -> dict:
    messages = ([{"role": "system", "content": item["system"]}] if item.get("system") else []) + [{"role": "user", "content": item["user"]}]
    body = {"model": model, "messages": messages, "stream": True, "max_tokens": item["max_tokens"], "reasoning_effort": EFFORT, **ARMS[arm]}
    if arm == "thinking":
        body["seed"] = seed
    if item.get("tools"):
        body.update(tools=TOOLS, tool_choice="auto")
    return body


def status_numbers(srv: Server) -> dict:
    status, st = srv.get("/status")
    return {k: st[k] for k in STATUS_KEYS if isinstance(st, dict) and isinstance(st.get(k), (int, float))} if status == 200 else {}


def one_request(srv, state, stop_path, item, arm, rnd, pos, model) -> dict:
    queue = wait_idle(srv, state, stop_path)
    gpu = (queue.get("upstream") or {}).get("gpu") or {}
    seed = 7000 + 100 * rnd + int(re.sub(r"\D", "", item["id"]) or 0) + (50 if item["workload"] == "tools" else 0)
    res = srv.chat_stream(request_body(item, arm, seed, model))
    text = "".join(res["reasoning"]) + "".join(res["content"])
    row = {"time": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "item": item["id"], "workload": item["workload"],
           "arm": arm, "round": rnd, "position": pos, "seed": seed if arm == "thinking" else None,
           "gpu_temp_c": gpu.get("temp_c"), "gpu_util_pct": gpu.get("util_pct"),
           **{k: res[k] for k in ("http_status", "error", "ttft_s", "first_data_s", "n_keepalive", "finish_reason", "usage", "timings", "total_s")},
           "content": "".join(res["content"])[:20000], "reasoning_tail": "".join(res["reasoning"])[-3000:],
           "calls": [res["calls"][i] for i in sorted(res["calls"])], "score": score(item, res) if not res["error"] else {},
           "repetition": repetition(text)}
    row["looped"] = bool(row["finish_reason"] == "length" and row["repetition"]["looped"])
    row["status_after"] = status_numbers(srv)
    row["status_matches"] = row["status_after"].get("max_tokens") == item["max_tokens"]     # another user's request would show its own cap
    return row


# ------------------------------------------------------------------------------------------------ the run
def preflight(srv: Server, out: pathlib.Path, state: dict, stop_path: pathlib.Path) -> str:
    queue = wait_idle(srv, state, stop_path)
    up = queue.get("upstream") or {}
    _, models = srv.get("/v1/models")
    model = ((models or {}).get("data") or [{}])[0]
    _, settings = srv.get("/settings")
    _, config = srv.get("/config")
    meta = {"started": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "base": BASE, "effort": EFFORT, "arms": ARMS,
            "upstream": {k: up.get(k) for k in ("model", "engine", "uptime_s", "gpu", "ram")}, "model_entry": model,
            "shared_settings": settings, "run_config": redact(config), "python": sys.version.split()[0]}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    name = model.get("id") or up.get("model") or "unknown"
    t0 = time.time()
    warm = srv.chat_stream({"model": name, "messages": [{"role": "user", "content": "Reply with the single word: ok"}], "stream": True,
                            "max_tokens": 64, "temperature": 0, "reasoning_effort": "low"}, 600)
    log(f"warm-up: model {name!r}, error={warm['error']!r}, {time.time() - t0:.0f} s (this loads the model if it was unloaded)")
    if warm["error"]:
        raise Abort(f"the warm-up request failed: {warm['error']}", retry=True)
    return name


def redact(value):
    if isinstance(value, dict):
        return {k: ("<redacted>" if re.search(r"key|token|secret|pass", k, re.I) else redact(v)) for k, v in value.items()}
    return [redact(v) for v in value] if isinstance(value, list) else value


def run(args) -> int:
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stop_path = out / "STOP"
    srv, state, rows, fails, retry = Server(BASE, pathlib.Path(KEY_FILE).read_text(encoding="utf-8").strip() if KEY_FILE else ""), {}, 0, 0, False
    deadline = time.time() + args.deadline_min * 60
    try:
        wait_until(args.start_at, stop_path)
        deadline = time.time() + args.deadline_min * 60          # the deadline counts from the start, not from the reservation
        model = preflight(srv, out, state, stop_path)
        items = build_items()
        log(f"{len(items)} prompts x 2 arms x up to {args.rounds} rounds; deadline in {args.deadline_min} minutes")
        for rnd in range(1, args.rounds + 1):
            for idx, item in enumerate(items):
                order = ("greedy", "thinking") if (idx + rnd) % 2 == 0 else ("thinking", "greedy")
                for pos, arm in enumerate(order):
                    if time.time() > deadline:
                        raise Abort("the deadline was reached")
                    row = one_request(srv, state, stop_path, item, arm, rnd, pos, model)
                    with open(out / "results.jsonl", "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows += 1
                    fails = fails + 1 if row["error"] else 0
                    log(f"round {rnd} {item['id']:6s} {arm:8s} {'ERROR ' + str(row['error'])[:80] if row['error'] else row['finish_reason']} "
                        f"{(row['timings'] or {}).get('predicted_per_second')} tok/s, {row['total_s']} s")
                    if fails >= MAX_FAILS_IN_A_ROW:
                        raise Abort(f"{fails} failed requests in a row")
    except (Abort, KeyboardInterrupt) as why:
        retry = bool(getattr(why, "retry", False))
        log(f"stopping: {why or 'Ctrl-C'}")
    finally:
        (out / "ended.txt").write_text(f"{dt.datetime.now().astimezone().isoformat(timespec='seconds')} rows={rows}\n", encoding="utf-8")
        if (out / "results.jsonl").exists():
            report(out)
    return 3 if retry and rows == 0 else 0          # 3 = nothing was measured and trying again later may work


# ------------------------------------------------------------------------------------------------ the report
def med(values: list):
    values = [v for v in values if v is not None]
    return round(statistics.median(values), 2) if values else None


def span(values: list) -> str:
    values = [v for v in values if v is not None]
    return f"{round(statistics.median(values), 1)} ({round(min(values), 1)}-{round(max(values), 1)})" if values else "n/a"


def table(header: list, body: list) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in row) + " |" for row in body])


def group_stats(rows: list) -> dict:
    ok = [r for r in rows if not r["error"]]
    timings = [r["timings"] or {} for r in ok]
    offered = sum(t.get("draft_n") or 0 for t in timings)
    accepted = sum(t.get("draft_n_accepted") or 0 for t in timings)
    return {"n": len(ok), "errors": len(rows) - len(ok), "decode": span([t.get("predicted_per_second") for t in timings]),
            "tokens": span([(r["usage"] or {}).get("completion_tokens") for r in ok]),
            "draft": f"{100 * accepted / offered:.1f}% of {offered}" if offered else "n/a",
            "total_s": span([r["total_s"] for r in ok]), "ttft": span([r["ttft_s"] for r in ok]),
            "length": sum(r["finish_reason"] == "length" for r in ok), "looped": sum(r["looped"] for r in ok),
            "recovered": sum(bool(r.get("status_matches")) and (r["status_after"] or {}).get("reasoning_recoveries", 0) > 0 for r in ok)}


def paired(rows: list) -> list:
    """(item, round) pairs where both arms succeeded: thinking / greedy for decode speed, tokens and wall time."""
    by = collections.defaultdict(dict)
    for r in rows:
        if not r["error"]:
            by[(r["item"], r["round"])][r["arm"]] = r
    out = []
    for pair in by.values():
        if len(pair) == 2:
            g, t = pair["greedy"], pair["thinking"]
            ratio = lambda a, b: (a / b) if a and b else None      # noqa: E731
            out.append({"workload": g["workload"],
                        "decode": ratio((t["timings"] or {}).get("predicted_per_second"), (g["timings"] or {}).get("predicted_per_second")),
                        "tokens": ratio((t["usage"] or {}).get("completion_tokens"), (g["usage"] or {}).get("completion_tokens")),
                        "gen": ratio(gen_time(t), gen_time(g)),
                        "time": ratio(t["total_s"], g["total_s"])})
    return out


def ratio_cell(pairs: list, field: str) -> str:
    values = [p[field] for p in pairs if p[field] is not None]
    return f"{statistics.median(values):.2f} ({min(values):.2f}-{max(values):.2f}), n={len(values)}" if values else "n/a"


def gen_time(r):
    """It leaves out reading the prompt."""
    rate = (r.get("timings") or {}).get("predicted_per_second")
    if not rate:
        return None
    tokens = (r.get("usage") or {}).get("completion_tokens")
    return (tokens / rate) if tokens is not None else None


def totals_rows(rows: list) -> list:
    out = []
    for arm in ARMS:
        mine = [r for r in rows if r["arm"] == arm and not r["error"]]
        out.append([arm, len(mine),
                    sum((r.get("usage") or {}).get("completion_tokens") or 0 for r in mine),
                    round(sum(gen_time(r) or 0 for r in mine), 1),
                    round(sum(r.get("total_s") or 0 for r in mine), 1)])
    return out


def repeatable(rows: list, arm: str) -> str:
    by_item = collections.defaultdict(list)
    for r in rows:
        if r["arm"] == arm and not r["error"]:
            by_item[r["item"]].append(r["content"])
    multi = [contents for contents in by_item.values() if len(contents) > 1]
    n = sum(len(set(contents)) == 1 for contents in multi)
    return f"{n} of {len(multi)}"


def quality_rows(rows: list) -> list:
    out = []
    for arm in ARMS:
        mine = [r for r in rows if r["arm"] == arm and not r["error"]]
        math = [r for r in mine if r["workload"] == "math"]
        tools = [r for r in mine if r["workload"] == "tools"]
        out.append([arm, f"{sum(r['score']['correct'] for r in math)}/{len(math)}",
                    f"{sum(r['score']['valid'] for r in tools)}/{len(tools)}", f"{sum(r['score']['complete'] for r in tools)}/{len(tools)}",
                    f"{sum(r['score']['n_calls'] == 0 for r in tools)}", f"{sum(r['looped'] for r in mine)}/{len(mine)}",
                    f"{sum(r['finish_reason'] == 'length' for r in mine)}/{len(mine)}"])
    return out


def report(out: pathlib.Path) -> None:
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8")) if (out / "meta.json").exists() else {}
    up = meta.get("upstream") or {}
    model = up.get('model') or (meta.get('model_entry') or {}).get('id') or 'not reported'
    engine = up.get('engine') or 'not reported'
    gpu = (up.get('gpu') or {}).get('name') or 'not reported'
    rounds = sorted({r["round"] for r in rows})
    parts = ["# Greedy against the thinking sampling preset (for PR #1274)\n",
             f"Started {meta.get('started')}. Model `{model}`, engine {engine}, GPU {gpu}. "
             f"Rounds with data: {rounds}. Requests: {len(rows)} ({sum(bool(r['error']) for r in rows)} failed).\n",
             "Both arms are sent explicitly: greedy = `temperature=0`; thinking = `temperature=1.0, top_p=0.95, top_k=20` (a fixed seed per prompt and round). "
             f"Same prompt, same `max_tokens`, `reasoning_effort={meta.get('effort')}`, streaming, one request at a time; the arm order alternates.\n"]
    pairs = paired(rows)
    parts += ["## Paired result (thinking / greedy, same prompt in the same round)\n",
              table(["workload", "decode tok/s", "generated tokens", "generation time", "wall time"],
                    [[w, ratio_cell([p for p in pairs if w in (p["workload"], "all")], "decode"),
                      ratio_cell([p for p in pairs if w in (p["workload"], "all")], "tokens"),
                      ratio_cell([p for p in pairs if w in (p["workload"], "all")], "gen"),
                      ratio_cell([p for p in pairs if w in (p["workload"], "all")], "time")]
                     for w in ["all"] + sorted({r["workload"] for r in rows})]), "",
              "Generation time = generated tokens / decode tok/s, so it leaves out reading the prompt. Wall time includes the prompt read: the second request of each pair finds the prompt in the engine's cache (`cache_n`) and skips most of it, and with an odd number of pairs (the five tool-call prompts per round) one arm goes first more often. For the short tool-call prompts read decode speed and generation time, not wall time.\n",
              "## Totals over all requests of an arm\n",
              table(["arm", "requests", "generated tokens", "generation time s", "wall time s"], totals_rows(rows)), ""]
    body = []
    for workload in sorted({r["workload"] for r in rows}) + ["all"]:
        for arm in ARMS:
            s = group_stats([r for r in rows if r["arm"] == arm and workload in (r["workload"], "all")])
            body.append([workload, arm, s["n"], s["errors"], s["decode"], s["tokens"], s["draft"], s["total_s"], s["ttft"],
                         s["length"], s["looped"], s["recovered"]])
    parts += ["## Per workload and arm (median, range in brackets)\n",
              table(["workload", "arm", "n", "failed", "decode tok/s", "generated tokens", "draft acceptance", "wall s", "first token s",
                     "hit max_tokens", "looped", "server recovered a loop"], body), "",
              "## Quality checks\n",
              table(["arm", "math correct", "tool calls valid", "tool task complete", "tool task with no call", "looped", "hit max_tokens"],
                    quality_rows(rows)), "",
              "## Limits\n",
              f"- Synthetic prompts, few of them, a handful of rounds: read the ranges, not only the medians. At `temperature=0` the final answer text was identical in every round for only {repeatable(rows, 'greedy')} prompts (preset: {repeatable(rows, 'thinking')}), so the rounds are separate samples, not exact repeats.\n"
              "- One server (single RTX 3070 8 GB), one model file and one engine version; draft acceptance comes from the engine's own counters.\n"
              "- `looped` = the answer hit `max_tokens` and its last 4000 characters are mostly repeats (a compression test); `server recovered a "
              "loop` = the server's own repeated-reasoning recovery fired (read from /status after the request, only counts that match the request).\n"
              "- The quality columns count what a script can check (an exact integer answer, tool-call JSON against its schema); they do not judge "
              "writing or code quality.\n"]
    (out / "RESULTS.md").write_text("\n".join(parts), encoding="utf-8")
    log(f"wrote {out / 'RESULTS.md'}")


# ------------------------------------------------------------------------------------------------ the fake server (selftest)
class Fake:
    """Enough of the relay + engine for the harness: /queue, /health, /v1/models, /settings, /config, /status and a streaming chat."""
    def __init__(self, answers: dict):
        self.answers, self.busy_polls, self.restart_after, self.served, self.started = answers, 0, None, 0, time.time()
        self.last, self.fail_posts, self.fail_from = {}, 0, None

    def uptime(self) -> float:
        up = time.time() - self.started
        return 1.0 if self.restart_after is not None and self.served > self.restart_after else up + 100


def make_handler(fake: Fake):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *a):
            pass

        def send_json(self, obj, code=200):
            raw = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if "Bearer" not in self.headers.get("Authorization", ""):
                return self.send_json({"error": {"type": "authentication_error"}}, 401)
            if self.path == "/queue":
                busy = fake.busy_polls > 0
                fake.busy_polls = max(0, fake.busy_polls - 1)
                return self.send_json({"running": [{"id": "x"}] if busy else [], "waiting": [], "upstream": {
                    "reachable": True, "model": "fake-model", "engine": "0.0.1", "uptime_s": fake.uptime(),
                    "gpu": {"name": "Fake GPU", "temp_c": 55, "util_pct": 0}}})
            routes = {"/v1/models": {"data": [{"id": "fake-model"}]}, "/settings": {"shared": False, "defaults": {}},
                      "/config": {"spec": 4, "api_key": "SHOULD-BE-REDACTED"}, "/status": {**fake.last, "tail": "private text"}}
            return self.send_json(routes[self.path]) if self.path in routes else self.send_json({"error": "not found"}, 404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            fake.served += 1
            if fake.fail_posts > 0 or (fake.fail_from is not None and fake.served > fake.fail_from):
                fake.fail_posts = max(0, fake.fail_posts - 1)
                return self.send_json({"error": {"type": "server_error", "message": "engine busy"}}, 500)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b": keep-alive\n\n")
            sampled = body.get("temperature", 0) > 0
            text, calls = self.compose(body, sampled)
            self.stream(body, sampled, text, calls)

        def compose(self, body, sampled):
            user = body["messages"][-1]["content"]
            if body.get("tools"):
                args = {"city": "Seoul", "unit": "kelvin" if sampled else "celsius"}
                return "", [("get_weather", json.dumps(args)), ("get_weather", json.dumps({"city": "Austin", "unit": "fahrenheit"}))] if "weather" in user \
                    else [("read_file", "{not json" if sampled else json.dumps({"path": "/srv/app/main.py"}))]
            answer = fake.answers.get(user.split("\n\n")[0])
            if answer is not None:
                return f"The result is below.\nANSWER: {answer + (1 if sampled and answer % 2 else 0)}", []
            return ("x" * 5000 if sampled and "hash" in user else "A short answer."), []

        def stream(self, body, sampled, text, calls):
            def send(chunk):
                self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                self.wfile.flush()
            base = {"id": "c1", "object": "chat.completion.chunk", "model": "fake-model"}
            send({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})
            send({**base, "choices": [{"index": 0, "delta": {"reasoning_content": "thinking... " * 5}, "finish_reason": None}]})
            for i, (name, args) in enumerate(calls):
                send({**base, "choices": [{"index": 0, "delta": {"tool_calls": [{"index": i, "id": f"t{i}", "type": "function", "function": {"name": name, "arguments": ""}}]}, "finish_reason": None}]})
                send({**base, "choices": [{"index": 0, "delta": {"tool_calls": [{"index": i, "function": {"arguments": args}}]}, "finish_reason": None}]})
            if text:
                send({**base, "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]})
            finish = "length" if len(text) >= 5000 else ("tool_calls" if calls else "stop")
            done = 200 if sampled else 300
            timings = {"predicted_per_second": 24.0 if sampled else 30.0, "predicted_n": done, "prompt_n": 50, "draft_n": 100, "draft_n_accepted": 55 if sampled else 70}
            fake.last = {"reasoning_recoveries": 1 if sampled and body["max_tokens"] == 6000 else 0, "generated": done, "prompt_tokens": 50, "max_tokens": body["max_tokens"]}
            send({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                  "usage": {"prompt_tokens": 50, "completion_tokens": done}, "timings": timings})
            self.wfile.write(b"data: [DONE]\n\n")
    return Handler


class Quiet(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def selftest(keep: str = "") -> int:
    global BASE, KEY_FILE, POLL_S
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="ab1274-selftest-"))
    key = tmp / "key"
    key.write_text("sk-fake-key-for-selftest-only", encoding="utf-8")
    fake = Fake({q: a for q, a in math_problems()})
    httpd = Quiet(("127.0.0.1", 0), make_handler(fake))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    BASE, KEY_FILE, POLL_S = f"http://127.0.0.1:{httpd.server_port}", key, 0.05
    failures = []

    def check(name, ok, detail=""):
        print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail and not ok else ""))
        if not ok:
            failures.append(name)

    items = build_items()
    small = [i for i in items if i["id"] in ("math1", "math3", "tool1", "tool3", "code1", "chat1")]
    globals()["build_items"] = lambda: small
    ns = lambda out, rounds=2, **kw: argparse.Namespace(out=str(tmp / out), start_at="", rounds=rounds, deadline_min=5, **kw)   # noqa: E731
    fake.busy_polls = 6
    t0 = time.time()
    run(ns("normal"))
    out = tmp / "normal"
    got = [json.loads(line) for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    check("12 requests per round x 2 rounds were written", len(got) == 24, str(len(got)))
    check("it waited while the queue was busy", time.time() - t0 > 6 * POLL_S)
    check("the order alternates", {r["position"] for r in got if r["arm"] == "greedy"} == {0, 1})
    check("greedy math answers are correct", all(r["score"]["correct"] for r in got if r["workload"] == "math" and r["arm"] == "greedy"))
    check("a wrong sampled math answer is caught", any(not r["score"]["correct"] for r in got if r["workload"] == "math" and r["arm"] == "thinking"))
    check("bad tool JSON and a bad enum are caught", sum(not r["score"]["valid"] for r in got if r["workload"] == "tools" and r["arm"] == "thinking") == 4)
    check("good tool calls pass", all(r["score"]["complete"] for r in got if r["item"] == "tool1" and r["arm"] == "greedy"))
    check("a repeated 5000-character answer counts as looped", any(r["looped"] for r in got if r["arm"] == "thinking" and r["item"] == "chat1"))
    check("only numbers are kept from /status", all(set(r["status_after"]) <= set(STATUS_KEYS) for r in got))
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    check("the run config's key-like fields are redacted", meta["run_config"]["api_key"] == "<redacted>")
    report_text = (out / "RESULTS.md").read_text(encoding="utf-8")
    check("the report shows the speed ratio 0.80", "0.80" in report_text, report_text[:300])
    check("the report shows draft acceptance 70.0% and 55.0%", "70.0%" in report_text and "55.0%" in report_text)
    check("a server loop recovery is counted for sampled math only (4 of them)",
          bool(re.search(r"\| math \| thinking \|.*\| 4 \|\s*$", report_text, re.M)) and bool(re.search(r"\| math \| greedy \|.*\| 0 \|\s*$", report_text, re.M)))
    check("no key in any output", "sk-fake-key" not in "".join(p.read_text(encoding="utf-8") for p in out.iterdir() if p.is_file()))

    fake.restart_after = fake.served + 5
    run(ns("restart", rounds=1))
    check("an engine restart ends the run and the report still exists", (tmp / "restart" / "RESULTS.md").exists()
          and len((tmp / "restart" / "results.jsonl").read_text(encoding="utf-8").splitlines()) <= 6)

    fake.restart_after = None
    fake.fail_posts = 3                                   # the warm-up and the first two requests of the run fail
    run(ns("flaky", rounds=1))
    flaky = [json.loads(line) for line in (tmp / "flaky" / "results.jsonl").read_text(encoding="utf-8").splitlines()] \
        if (tmp / "flaky" / "results.jsonl").exists() else []
    check("a failed warm-up ends the run before any request", not flaky and (tmp / "flaky" / "ended.txt").exists())
    fake.fail_posts = 0
    fake.busy_polls = 10 ** 6
    patience, ab = IDLE_PATIENCE_S, sys.modules[__name__]
    ab.IDLE_PATIENCE_S = 0.5
    busy_code = run(ns("busy", rounds=1))
    ab.IDLE_PATIENCE_S = patience
    check("a busy queue returns exit code 3 (the wrapper may retry)", busy_code == 3, str(busy_code))
    fake.busy_polls = 0
    check("a queue that never empties ends the run without any request", not (tmp / "busy" / "results.jsonl").exists()
          and (tmp / "busy" / "ended.txt").exists())
    fake.served, fake.fail_from = 0, 1                    # the warm-up is request 1 and passes; every request after it fails
    run(ns("failing", rounds=1))
    failing = [json.loads(line) for line in (tmp / "failing" / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    check("5 failed requests in a row end the run, each one recorded", len(failing) == MAX_FAILS_IN_A_ROW and all(r["error"] for r in failing),
          str(len(failing)))
    fake.served, fake.fail_from = 0, None
    (tmp / "stop").mkdir()
    (tmp / "stop" / "STOP").write_text("", encoding="utf-8")
    stop_code = run(ns("stop"))
    check("a STOP file returns exit code 0 (the wrapper must not retry)", stop_code == 0, str(stop_code))
    check("a STOP file before the first request writes no result rows", not (tmp / "stop" / "results.jsonl").exists())
    class NoQueue(Server):                                # a stock server answers 404 to /queue: nothing to wait for
        def get(self, path, timeout=15.0):
            return (404, None) if path == "/queue" else super().get(path, timeout)
    check("a server without /queue is not waited for", wait_idle(NoQueue(BASE, "x"), {}, tmp / "none") == {})
    httpd.shutdown()
    if keep:
        shutil.copy2(tmp / "normal" / "RESULTS.md", keep)
    shutil.rmtree(tmp, ignore_errors=True)
    print("SELFTEST", "FAILED: " + ", ".join(failures) if failures else "OK")
    return 1 if failures else 0


def serve_fake(port: int) -> int:
    httpd = Quiet(("127.0.0.1", port), make_handler(Fake({q: a for q, a in math_problems()})))
    log(f"fake relay + engine on 127.0.0.1:{port} (for testing the wrapper script; Ctrl-C ends it)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--out", required=True)
    p.add_argument("--start-at", default="")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--deadline-min", type=float, default=110)
    sub.add_parser("report").add_argument("--out", required=True)
    sub.add_parser("fake").add_argument("--port", type=int, default=18999)
    sub.add_parser("selftest").add_argument("--keep", default="", help="copy the selftest's RESULTS.md to this file")
    args = parser.parse_args()
    if args.cmd == "run":
        return run(args)
    if args.cmd == "report":
        report(pathlib.Path(args.out))
        return 0
    if args.cmd == "fake":
        return serve_fake(args.port)
    return selftest(args.keep)


if __name__ == "__main__":
    sys.exit(main())
