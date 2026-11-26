#!/usr/bin/env python3
"""usage_suite.py: what people really do with a Strata server, as a standalone runner against any OpenAI-compatible endpoint.

    python3 usage_suite.py --url http://127.0.0.1:8080 [--model ID] [--cases coding,agent,vision,longdoc,chat,parallel]
                           [--long 30000,60000] [--agent-turns 20] [--parallel-levels 1,2,4,8] [--out result.json] [--md table.md]
                           [--key-env STRATA_API_KEY] [--timeout 1800]

Every case goes through the server's HTTP API (`/v1/chat/completions`, streamed so the time to first token is real); nothing here calls an engine or `--bench`.
Standard library only (Python 3.8+); the pictures are drawn by this file (no personal or downloaded images); code a model writes is run with `python3 -I`
in a temporary folder with a time limit. The key is read from the environment variable named by --key-env, never printed.

Cases
  coding    write a function from a spec, fix a bug in a file, edit a 300-line file with a unified diff, explain code. Pass: the tests pass / the diff applies, no truncation, no empty reply.
  agent     15 to 25 turns with tools (read_file, write_file, run_tests, search) on a small repo with several bugs, the way coding agents loop. Pass: the tests end green,
            every tool call has valid JSON arguments, the loop ends, no stall or endless reply, the prompt cache is reused turn to turn.
  vision    describe a drawn scene, read text from a drawn screenshot, read a drawn bar chart, compare two drawings. Skipped when the server has no picture reader.
  longdoc   a document of about 32K and 64K tokens with facts in it, then questions about the middle.
  chat      short questions at thinking levels low / medium / high: thinking length per level, a visible answer every time.
  parallel  1, 2, 4 and 8 clients at once, each with its own secret word and a mix of the cases above: total throughput, per-request decode, stalls, cross-talk.

Per case the result records prefill tok/s, decode tok/s, time to first token, cached tokens, and the counts of empty / looping / truncated replies, with a pass or fail.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zlib

PASS, FAIL, SKIP = "pass", "fail", "skip"
SPEED_PROMPT_P = "Write a long, detailed essay about the history of the printing press, from Gutenberg to the modern day."


# ===================================================================================================== the client
class Client:
    def __init__(self, url, model=None, key=None, timeout=1800):
        self.base = url.rstrip("/")
        self.model = model
        self.key = key
        self.timeout = timeout

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = "Bearer " + self.key
        return h

    def get(self, path):
        req = urllib.request.Request(self.base + path, headers=self._headers())
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode("utf-8", "replace"))

    def find_model(self):
        if not self.model:
            self.model = self.get("/v1/models")["data"][0]["id"]
        return self.model

    def chat(self, messages, tools=None, timeout=None, **extra):
        """One streamed chat completion. Returns a dict with text, reasoning, tool_calls, finish, ttft, wall, usage, timings, error."""
        body = {"model": self.model, "messages": messages, "stream": True, "stream_options": {"include_usage": True}}
        if tools:
            body["tools"] = tools
        body.update(extra)
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(), headers=self._headers())
        out = {"text": "", "reasoning": "", "tool_calls": [], "finish": None, "ttft": None, "usage": {}, "timings": {}, "error": None}
        calls = {}
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                for raw in r:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        ev = json.loads(data)
                    except ValueError:
                        continue
                    if ev.get("usage"):
                        out["usage"] = ev["usage"]
                    if ev.get("timings"):
                        out["timings"] = ev["timings"]
                    for ch in ev.get("choices") or []:
                        d = ch.get("delta") or {}
                        got = False
                        if d.get("content"):
                            out["text"] += d["content"]
                            got = True
                        rc = d.get("reasoning_content") or d.get("reasoning")
                        if rc:
                            out["reasoning"] += rc
                            got = True
                        for tc in d.get("tool_calls") or []:
                            slot = calls.setdefault(tc.get("index", 0), {"id": None, "name": "", "arguments": ""})
                            if tc.get("id"):
                                slot["id"] = tc["id"]
                            fn = tc.get("function") or {}
                            slot["name"] += fn.get("name") or ""
                            slot["arguments"] += fn.get("arguments") or ""
                            got = True
                        if got and out["ttft"] is None:
                            out["ttft"] = time.time() - t0
                        if ch.get("finish_reason"):
                            out["finish"] = ch["finish_reason"]
        except urllib.error.HTTPError as e:
            out["error"] = "HTTP %s: %s" % (e.code, e.read().decode("utf-8", "replace")[:300])
        except Exception as e:  # noqa: BLE001 (a dead or stalled server shows up as many exceptions)
            out["error"] = "%s: %s" % (type(e).__name__, str(e)[:200])
        out["wall"] = time.time() - t0
        out["tool_calls"] = [{"id": c["id"] or "call_%d" % i, "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                             for i, c in sorted(calls.items())]
        u, t = out["usage"], out["timings"]
        out["prompt_tokens"] = u.get("prompt_tokens")
        out["completion_tokens"] = u.get("completion_tokens")
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
        out["cached"] = cached if cached is not None else t.get("cache_n")
        out["prefill_tps"] = t.get("prompt_per_second")
        out["decode_tps"] = t.get("predicted_per_second")
        if not out["decode_tps"] and out["completion_tokens"] and out["ttft"] is not None and out["wall"] > out["ttft"] and out["completion_tokens"] > 3:
            out["decode_tps"] = (out["completion_tokens"] - 1) / (out["wall"] - out["ttft"])
        return out


# ===================================================================================================== recording
class Recorder:
    """Collects the per-request numbers of one case."""

    def __init__(self):
        self.reqs = []
        self.lock = threading.Lock()
        self.empty = self.loops = self.trunc = self.errors = 0

    def add(self, r, expect_text=True, cap=None):
        with self.lock:
            self.reqs.append(r)
            if r.get("error"):
                self.errors += 1
                return
            if expect_text and not r["text"].strip() and not r["tool_calls"]:
                self.empty += 1
            if looping(r["text"]):
                self.loops += 1
            if r.get("finish") == "length":
                self.trunc += 1

    def summary(self):
        def med(key, cond=lambda r: True):
            v = [r[key] for r in self.reqs if r.get(key) and not r.get("error") and cond(r)]
            return round(statistics.median(v), 1) if v else None
        return {"requests": len(self.reqs), "prefill_tps": med("prefill_tps", lambda r: (r.get("prompt_tokens") or 0) >= 200 and (r.get("cached") or 0) < (r.get("prompt_tokens") or 0) * 0.5),
                "decode_tps": med("decode_tps"), "ttft_s": med("ttft"),
                "cached_tokens": sum(r.get("cached") or 0 for r in self.reqs if not r.get("error")),
                "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in self.reqs if not r.get("error")),
                "empty": self.empty, "looping": self.loops, "truncated": self.trunc, "errors": self.errors}


def looping(text, n=12, times=5):
    words = (text or "").split()
    if len(words) < n * times:
        return False
    seen = {}
    for i in range(len(words) - n):
        k = " ".join(words[i:i + n])
        seen[k] = seen.get(k, 0) + 1
        if seen[k] >= times:
            return True
    return False


def case_result(rec, ok, notes, extra=None):
    d = rec.summary()
    d["verdict"] = PASS if ok else FAIL
    d["notes"] = notes
    if extra:
        d.update(extra)
    if d["empty"] or d["looping"] or d["truncated"] or d["errors"]:
        if ok:
            d["verdict"] = FAIL
            d["notes"] = notes + ["%d empty, %d looping, %d truncated, %d errors" % (d["empty"], d["looping"], d["truncated"], d["errors"])]
    return d


# ===================================================================================================== running code from a model
def code_blocks(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return blocks or ([text] if "def " in text else [])


def run_python(code, timeout=20):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            p = subprocess.run([sys.executable, "-I", path], cwd=d, capture_output=True, text=True, timeout=timeout)
            return p.returncode == 0, (p.stdout + p.stderr)[-400:]
        except subprocess.TimeoutExpired:
            return False, "timeout"


# ===================================================================================================== case 1: coding
FIX_FILE = '''def moving_average(values, window):
    """Average of each run of `window` consecutive values."""
    out = []
    for i in range(len(values) - window):
        out.append(sum(values[i:i + window]) / window)
    return out
'''

EXPLAIN_FILE = '''def pack(s):
    out, i = [], 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        out.append(s[i] + str(j - i))
        i = j
    return "".join(out)
'''


def big_file(n=100):
    parts = ['"""Utility module with %d small functions."""\n' % n]
    for i in range(1, n + 1):
        parts.append("def f%03d(x):\n    \"\"\"Step %d of the pipeline.\"\"\"\n    return x + %d\n" % (i, i, i % 7 + 1))
    return "\n".join(parts)


def apply_unified_diff(original, diff):
    """Minimal unified-diff applier: each hunk's old lines must be found in the file once; returns (new text or None, message)."""
    lines = original.split("\n")
    hunks, cur = [], None
    for ln in diff.split("\n"):
        if ln.startswith("@@"):
            cur = []
            hunks.append(cur)
        elif cur is not None and (ln[:1] in (" ", "+", "-")) and not ln.startswith(("+++", "---")):
            cur.append(ln)
        elif cur is not None and ln == "":
            cur.append(" ")
    if not hunks:
        return None, "no hunk in the reply"
    for h in hunks:
        old = [x[1:] for x in h if x[0] in " -"]
        new = [x[1:] for x in h if x[0] in " +"]
        while old and old[-1] == "" and new and new[-1] == "":
            old.pop()
            new.pop()
        pos = [i for i in range(len(lines) - len(old) + 1) if lines[i:i + len(old)] == old]
        if len(pos) != 1:
            return None, "hunk does not apply (%d places found)" % len(pos)
        lines[pos[0]:pos[0] + len(old)] = new
    return "\n".join(lines), "ok"


def case_coding(c, a):
    rec, notes, ok = Recorder(), [], True

    def ask(prompt, cap=900):
        r = c.chat([{"role": "user", "content": prompt}], temperature=0, max_tokens=cap, reasoning_effort="none")
        rec.add(r)
        return r

    # a. write from a spec
    r = ask("Write a Python function merge_intervals(intervals) that takes a list of (start, end) tuples and returns the merged, sorted list of tuples "
            "(touching intervals merge). Reply with one python code block containing only the function.")
    tests = ("\nassert merge_intervals([(1,3),(2,6),(8,10),(15,18)]) == [(1,6),(8,10),(15,18)]\nassert merge_intervals([(1,4),(4,5)]) == [(1,5)]\n"
             "assert merge_intervals([]) == []\nassert merge_intervals([(5,6),(1,2)]) == [(1,2),(5,6)]\nprint('ok')\n")
    good = any(run_python(b + tests)[0] for b in code_blocks(r["text"]))
    notes.append("write-function: " + ("tests pass" if good else "tests FAIL"))
    ok &= good
    # b. fix a bug in a file
    r = ask("This function has a bug: moving_average([1,2,3,4], 2) should be [1.5, 2.5, 3.5] but returns fewer values.\n\n```python\n%s```\n\n"
            "Reply with the corrected function in one python code block." % FIX_FILE)
    tests = "\nassert moving_average([1,2,3,4], 2) == [1.5, 2.5, 3.5]\nassert moving_average([2,4,6], 3) == [4.0]\nassert moving_average([1], 2) == []\nprint('ok')\n"
    good = any(run_python(b + tests)[0] for b in code_blocks(r["text"]))
    notes.append("fix-bug: " + ("tests pass" if good else "tests FAIL"))
    ok &= good
    # c. edit a 300-line file with a diff
    src = big_file(100)
    r = ask("Here is a Python file of %d lines.\n\n```python\n%s\n```\n\nChange f057 so that it returns x * 3 instead of adding a number. "
            "Reply with ONLY a unified diff (a ```diff block with @@ hunks and 2 lines of context), nothing else." % (len(src.split("\n")), src), cap=700)
    diffs = re.findall(r"```(?:diff|patch)?\s*\n(.*?)```", r["text"], re.S) or [r["text"]]
    new, msg = None, "no diff"
    for dtext in diffs:
        new, msg = apply_unified_diff(src, dtext)
        if new:
            break
    good = bool(new) and "def f057(x):\n    \"\"\"Step 57 of the pipeline.\"\"\"\n    return x * 3" in new and new.count("x * 3") == 1
    notes.append("edit-300-lines: " + ("diff applies, only f057 changed" if good else "FAIL (%s)" % msg))
    ok &= good
    # d. explain code
    r = ask("What does pack('aaabccdddd') return? Think about the code, then answer with the exact string on the last line.\n\n```python\n%s```" % EXPLAIN_FILE, cap=500)
    good = "a3b1c2d4" in r["text"]
    notes.append("explain: " + ("right" if good else "FAIL, the reply ends: " + r["text"][-90:].replace("\n", " ")))
    ok &= good
    return case_result(rec, ok, notes)


# ===================================================================================================== case 2: agent loop
REPO_FILES = {
    "mathkit/__init__.py": "",
    "mathkit/stats.py": "def mean(xs):\n    return sum(xs) / (len(xs) + 1)\n\n\ndef maximum(xs):\n    best = xs[0]\n    for x in xs[1:]:\n        if x < best:\n            best = x\n    return best\n",
    "mathkit/text.py": "def words(s):\n    return s.split(',')\n\n\ndef title(s):\n    return s.upper()\n",
    "mathkit/geo.py": "import math\n\n\ndef area_circle(r):\n    return math.pi * r\n\n\ndef perimeter_rect(w, h):\n    return w + h\n",
    "tests/test_all.py": ("import sys\nsys.path.insert(0, '.')\nfrom mathkit import stats, text, geo\n\n"
                          "def test_mean():\n    assert stats.mean([2, 4, 6]) == 4\n\n"
                          "def test_maximum():\n    assert stats.maximum([3, 9, 2]) == 9\n\n"
                          "def test_words():\n    assert text.words('a b  c') == ['a', 'b', 'c']\n\n"
                          "def test_title():\n    assert text.title('hello world') == 'Hello World'\n\n"
                          "def test_area():\n    assert abs(geo.area_circle(2) - 12.566370614359172) < 1e-9\n\n"
                          "def test_perimeter():\n    assert geo.perimeter_rect(2, 3) == 10\n\n"
                          "def run():\n    import traceback\n    bad = 0\n    for n, f in sorted(globals().items()):\n        if n.startswith('test_'):\n            try:\n                f(); print('PASS', n)\n            except Exception:\n                bad += 1; print('FAIL', n, traceback.format_exc().splitlines()[-1]); break\n    print('%d failed' % bad)\n    sys.exit(1 if bad else 0)\n\n"
                          "if __name__ == '__main__':\n    run()\n"),
}

AGENT_TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a file of the repo.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Replace a file of the repo with new content.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "run_tests", "description": "Run the test suite; returns its output.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "search", "description": "Search the repo for a text; returns file:line matches.", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}}},
]


class Repo:
    def __init__(self):
        self.files = dict(REPO_FILES)

    def run_tests(self):
        with tempfile.TemporaryDirectory() as d:
            for p, t in self.files.items():
                full = os.path.join(d, p)
                os.makedirs(os.path.dirname(full), exist_ok=True)
                with open(full, "w", encoding="utf-8") as f:
                    f.write(t)
            try:
                pr = subprocess.run([sys.executable, "-I", "tests/test_all.py"], cwd=d, capture_output=True, text=True, timeout=30)
                return pr.returncode == 0, (pr.stdout + pr.stderr)[-1500:]
            except subprocess.TimeoutExpired:
                return False, "tests timed out"

    def call(self, name, args):
        if name == "read_file":
            return self.files.get(str(args.get("path", "")), "error: no such file")
        if name == "write_file":
            p = str(args.get("path", ""))
            if not p or p.startswith("/") or ".." in p:
                return "error: bad path"
            self.files[p] = str(args.get("content", ""))
            return "wrote %d bytes to %s" % (len(self.files[p]), p)
        if name == "run_tests":
            return self.run_tests()[1]
        if name == "search":
            pat = str(args.get("pattern", ""))
            hits = ["%s:%d: %s" % (p, i + 1, ln) for p, t in self.files.items() for i, ln in enumerate(t.split("\n")) if pat and pat in ln]
            return "\n".join(hits[:30]) or "no match"
        return "error: unknown tool"


def case_agent(c, a):
    rec, notes = Recorder(), []
    repo = Repo()
    max_req = int(a.agent_turns) + 12
    msgs = [{"role": "system", "content": "You are a coding agent working in a small Python repo. Use the tools. Run the tests, find the bugs, fix them with write_file, and run the tests again until they all pass. Then say DONE and summarise the fixes in two lines."},
            {"role": "user", "content": "The tests in tests/test_all.py fail. Fix the code in mathkit/ (not the tests)."}]
    bad_json, stalls, prev_prompt, reuse, ttfts, tools_used = 0, 0, None, [], [], 0
    finished = False
    for turn in range(1, max_req + 1):
        r = c.chat(msgs, tools=AGENT_TOOLS, temperature=0, max_tokens=1500, reasoning_effort="none", timeout=min(c.timeout, 900))
        rec.add(r)
        if r.get("error"):
            notes.append("turn %d: %s" % (turn, r["error"]))
            if "timed out" in r["error"].lower() or "timeout" in r["error"].lower():
                stalls += 1
            break
        if r["ttft"] is not None:
            ttfts.append(round(r["ttft"], 2))
        if prev_prompt and r.get("cached") is not None:
            reuse.append(min(1.0, r["cached"] / prev_prompt))
        prev_prompt = r.get("prompt_tokens")
        calls = r["tool_calls"]
        if not calls:
            msgs.append({"role": "assistant", "content": r["text"]})
            finished = True
            break
        msgs.append({"role": "assistant", "content": r["text"] or None, "tool_calls": calls})
        for tc in calls:
            tools_used += 1
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
                assert isinstance(args, dict)
            except (ValueError, AssertionError):
                bad_json += 1
                args = {}
                msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": "error: arguments are not valid JSON"})
                continue
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": repo.call(tc["function"]["name"], args)})
    green, _ = repo.run_tests()
    reuse_med = round(statistics.median(reuse), 3) if reuse else None
    ok = green and finished and bad_json == 0 and stalls == 0 and (reuse_med is None or reuse_med >= 0.5)
    notes.append("tests %s after %d requests, %d tool calls" % ("GREEN" if green else "still red", len(rec.reqs), tools_used))
    if not finished:
        notes.append("the loop did not end by itself in %d requests" % max_req)
    if bad_json:
        notes.append("%d tool calls with invalid JSON arguments" % bad_json)
    if reuse_med is not None and reuse_med < 0.5:
        notes.append("prompt reuse median %.0f%% of the previous prompt" % (reuse_med * 100))
    return case_result(rec, ok, notes, {"turns": len(rec.reqs), "tool_calls": tools_used, "bad_json": bad_json, "reuse_median": reuse_med,
                                       "ttft_per_turn_s": ttfts[:40], "tests_green": green})


# ===================================================================================================== case 3: vision (pictures drawn here)
_FONT_ROWS = {
    "A": ".###. #...# #...# ##### #...# #...# #...#", "B": "####. #...# #...# ####. #...# #...# ####.",
    "C": ".###. #...# #.... #.... #.... #...# .###.", "D": "####. #...# #...# #...# #...# #...# ####.",
    "E": "##### #.... #.... ####. #.... #.... #####", "F": "##### #.... #.... ####. #.... #.... #....",
    "G": ".###. #...# #.... #.### #...# #...# .###.", "H": "#...# #...# #...# ##### #...# #...# #...#",
    "I": ".###. ..#.. ..#.. ..#.. ..#.. ..#.. .###.", "J": "..### ...#. ...#. ...#. ...#. #..#. .##..",
    "K": "#...# #..#. #.#.. ##... #.#.. #..#. #...#", "L": "#.... #.... #.... #.... #.... #.... #####",
    "M": "#...# ##.## #.#.# #.#.# #...# #...# #...#", "N": "#...# ##..# #.#.# #..## #...# #...# #...#",
    "O": ".###. #...# #...# #...# #...# #...# .###.", "P": "####. #...# #...# ####. #.... #.... #....",
    "Q": ".###. #...# #...# #...# #.#.# #..#. .##.#", "R": "####. #...# #...# ####. #.#.. #..#. #...#",
    "S": ".#### #.... #.... .###. ....# ....# ####.", "T": "##### ..#.. ..#.. ..#.. ..#.. ..#.. ..#..",
    "U": "#...# #...# #...# #...# #...# #...# .###.", "V": "#...# #...# #...# #...# #...# .#.#. ..#..",
    "W": "#...# #...# #...# #.#.# #.#.# ##.## #...#", "X": "#...# #...# .#.#. ..#.. .#.#. #...# #...#",
    "Y": "#...# #...# .#.#. ..#.. ..#.. ..#.. ..#..", "Z": "##### ....# ...#. ..#.. .#... #.... #####",
    "0": ".###. #...# #..## #.#.# ##..# #...# .###.", "1": "..#.. .##.. ..#.. ..#.. ..#.. ..#.. .###.",
    "2": ".###. #...# ....# ...#. ..#.. .#... #####", "3": "####. ....# ....# .###. ....# ....# ####.",
    "4": "...#. ..##. .#.#. #..#. ##### ...#. ...#.", "5": "##### #.... ####. ....# ....# #...# .###.",
    "6": "..##. .#... #.... ####. #...# #...# .###.", "7": "##### ....# ...#. ..#.. .#... .#... .#...",
    "8": ".###. #...# #...# .###. #...# #...# .###.", "9": ".###. #...# #...# .#### ....# ...#. .##..",
    "-": "..... ..... ..... ##### ..... ..... .....", ":": "..... ..#.. ..#.. ..... ..#.. ..#.. .....",
    ".": "..... ..... ..... ..... ..... .##.. .##..", " ": "..... ..... ..... ..... ..... ..... .....",
}
FONT = {k: v.replace(" ", "") for k, v in _FONT_ROWS.items()}


class Img:
    def __init__(self, w, h, bg=(255, 255, 255)):
        self.w, self.h = w, h
        self.px = [list(bg) for _ in range(w * h)]

    def rect(self, x0, y0, x1, y1, col):
        for y in range(max(0, y0), min(self.h, y1)):
            row = y * self.w
            for x in range(max(0, x0), min(self.w, x1)):
                self.px[row + x] = list(col)

    def circle(self, cx, cy, r, col):
        for y in range(max(0, cy - r), min(self.h, cy + r + 1)):
            for x in range(max(0, cx - r), min(self.w, cx + r + 1)):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                    self.px[y * self.w + x] = list(col)

    def tri(self, x0, y0, x1, y1, x2, y2, col):
        for y in range(max(0, min(y0, y1, y2)), min(self.h, max(y0, y1, y2) + 1)):
            for x in range(max(0, min(x0, x1, x2)), min(self.w, max(x0, x1, x2) + 1)):
                d1 = (x - x1) * (y0 - y1) - (x0 - x1) * (y - y1)
                d2 = (x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)
                d3 = (x - x0) * (y2 - y0) - (x2 - x0) * (y - y0)
                if not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0)):
                    self.px[y * self.w + x] = list(col)

    def text(self, x, y, s, scale=4, col=(0, 0, 0)):
        for ch in s.upper():
            g = FONT.get(ch, FONT[" "])
            if len(g) != 35:
                g = g.ljust(35, ".")[:35]
            for r in range(7):
                for k in range(5):
                    if g[r * 5 + k] == "#":
                        self.rect(x + k * scale, y + r * scale, x + (k + 1) * scale, y + (r + 1) * scale, col)
            x += 6 * scale

    def png(self):
        raw = b"".join(b"\x00" + bytes(v for p in self.px[y * self.w:(y + 1) * self.w] for v in p) for y in range(self.h))

        def chunk(t, d):
            c = struct.pack(">I", len(d)) + t + d
            return c + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
        return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", self.w, self.h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")

    def data_url(self):
        return "data:image/png;base64," + base64.b64encode(self.png()).decode()


def img_scene():
    im = Img(480, 320, (135, 206, 235))
    im.rect(0, 220, 480, 320, (60, 160, 60))
    im.circle(400, 60, 36, (255, 220, 0))
    im.rect(150, 140, 300, 250, (200, 40, 40))
    im.tri(140, 140, 225, 80, 310, 140, (110, 60, 30))
    im.rect(210, 190, 245, 250, (90, 55, 20))
    return im


def img_screenshot():
    im = Img(480, 220, (245, 245, 245))
    im.rect(0, 0, 480, 36, (40, 70, 140))
    im.text(12, 8, "ORDER SYSTEM", 3, (255, 255, 255))
    im.text(20, 60, "ORDER 4417", 5)
    im.text(20, 108, "TOTAL 83 USD", 5)
    im.text(20, 156, "STATUS SHIPPED", 5, (20, 120, 20))
    return im


def img_chart():
    im = Img(480, 320, (255, 255, 255))
    im.rect(50, 20, 54, 280, (0, 0, 0))
    im.rect(50, 276, 460, 280, (0, 0, 0))
    for i, (lab, val, col) in enumerate([("A", 3, (220, 60, 60)), ("B", 9, (60, 90, 220)), ("C", 5, (60, 180, 80))]):
        x = 90 + i * 120
        h = val * 26
        im.rect(x, 276 - h, x + 80, 276, col)
        im.text(x + 28, 288, lab, 4)
        im.text(x + 24, 276 - h - 30, str(val), 4)
    return im


def img_circle():
    im = Img(320, 240, (255, 255, 255))
    im.circle(160, 120, 80, (220, 30, 30))
    return im


def img_square():
    im = Img(320, 240, (255, 255, 255))
    im.rect(80, 40, 240, 200, (30, 50, 220))
    return im


def case_vision(c, a):
    rec, notes, ok = Recorder(), [], True

    def ask(images, q):
        content = [{"type": "text", "text": q}] + [{"type": "image_url", "image_url": {"url": im.data_url()}} for im in images]
        return c.chat([{"role": "user", "content": content}], temperature=0, max_tokens=500, reasoning_effort="none")

    first = ask([img_scene()], "Describe this picture in two or three sentences.")
    if first.get("error") and re.search(r"image|vision|picture|multimodal|not support", first["error"], re.I):
        return {"verdict": SKIP, "notes": ["the server has no picture reader (start it with --vision on): " + first["error"][:120]], "requests": 1}
    rec.add(first)
    t = first["text"].lower()
    groups = [r"sun|yellow", r"house|building|home|cottage", r"green|hill|grass|field", r"sky|blue"]
    hit = sum(1 for g in groups if re.search(g, t))
    notes.append("describe: %d of 4 key facts (sun, house, green ground, sky)" % hit)
    ok &= hit >= 3
    r = ask([img_screenshot()], "Read the text in this screenshot. List the order number, the total and the status.")
    rec.add(r)
    t = r["text"].lower()
    good = "4417" in t and "83" in t and "shipped" in t
    notes.append("ocr: " + ("order, total and status read" if good else "FAIL: " + r["text"][:80].replace("\n", " ")))
    ok &= good
    r = ask([img_chart()], "This is a bar chart with three bars A, B and C. Which bar is the tallest and what value is written above it?")
    rec.add(r)
    t = r["text"]
    good = bool(re.search(r"\bB\b", t)) and "9" in t
    notes.append("chart: " + ("B, 9" if good else "FAIL: " + t[:80].replace("\n", " ")))
    ok &= good
    r = ask([img_circle(), img_square()], "Compare the two images. What shape and colour is in each?")
    rec.add(r)
    t = r["text"].lower()
    good = "circle" in t and "square" in t and "red" in t and "blue" in t
    notes.append("compare: " + ("circle/red vs square/blue" if good else "FAIL: " + t[:80].replace("\n", " ")))
    ok &= good
    return case_result(rec, ok, notes)


# ===================================================================================================== case 4: long document
SECTION_FACTS = [("harbour", "The harbour master is Ilse Moreau and the tide table is kept in cabinet 12."),
                 ("archive", "The archive was founded in 1873 by a retired surveyor named Teodor Brandt."),
                 ("orchard", "The orchard grows 412 pear trees and the oldest one is 94 years old."),
                 ("relay", "The relay station uses the call sign KELVIN-SEVEN on 14.2 megahertz."),
                 ("bridge", "The bridge was repainted in the colour called Verdigris Night in the spring of 2019."),
                 ("market", "The Thursday market has 63 stalls and the cheese seller is called Odalys."),
                 ("tunnel", "The tunnel is 2,840 metres long and closes for cleaning every second Sunday."),
                 ("glacier", "The glacier retreated 31 metres last year according to the survey of Mira Okoye.")]


def long_doc(tokens, salt, cpt=3.0):
    chars = int(tokens * cpt)
    per = max(1, chars // (len(SECTION_FACTS) * 74))
    parts = ["Report %s." % salt]
    for name, fact in SECTION_FACTS:
        parts.append("\n## Section: %s\n" % name)
        for i in range(per):
            parts.append("Note %d on the %s: reading %d logged on day %d at station %d." % (i, name, (i * 37) % 991, i % 28, (i * 11) % 97))
            if i == per // 2:
                parts.append(fact)
    return " ".join(parts)


def case_longdoc(c, a):
    rec, notes, ok, extra = Recorder(), [], True, {}
    cpt = 3.0                                              # characters per token of this text; measured from the first answer
    for tokens in [int(x) for x in a.long.split(",") if x]:
        doc = long_doc(tokens, "%d-%d" % (tokens, int(time.time())), cpt)
        q = "Answer from the report above in two short lines. 1) Who is the harbour master? 2) What is the call sign of the relay station?"
        r = c.chat([{"role": "user", "content": doc + "\n\n" + q}], temperature=0, max_tokens=120, reasoning_effort="none")
        rec.add(r)
        if r.get("prompt_tokens"):
            cpt = len(doc) / r["prompt_tokens"]
        t = r["text"].lower()
        good = "moreau" in t and "kelvin" in t and not r.get("error")
        notes.append("%dK-class: %s (%s tokens, prefill %s tok/s, first token %.0f s)" % (tokens // 1024, "right" if good else "FAIL: " + (r.get("error") or r["text"][:60]).replace("\n", " "),
                                                                                  r.get("prompt_tokens"), r.get("prefill_tps") and round(r["prefill_tps"]), r.get("ttft") or 0))
        extra["prefill_%dk_tps" % round(tokens / 1000)] = r.get("prefill_tps") and round(r["prefill_tps"], 1)
        ok &= good
    return case_result(rec, ok, notes, extra)


# ===================================================================================================== case 5: chat at thinking levels
CHAT_QS = ["A train leaves at 9:40 and the trip takes 2 hours 35 minutes. When does it arrive? Give the time.",
           "Give three reasons why the sky is blue at noon and red at sunset, one sentence each.",
           "What is 48 times 25? Show the answer as a number."]


def case_chat(c, a):
    rec, notes, ok, per = Recorder(), [], True, {}
    for level in ("low", "medium", "high"):
        chars, ans = [], 0
        for q in CHAT_QS:
            r = c.chat([{"role": "user", "content": q}], temperature=0.6, top_p=0.95, max_tokens=3000, reasoning_effort=level)
            rec.add(r)
            chars.append(len(r["reasoning"]))
            ans += 1 if r["text"].strip() else 0
        per[level] = {"thinking_chars_avg": round(sum(chars) / len(chars)), "visible_answers": ans}
        ok &= ans == len(CHAT_QS)
        notes.append("%s: thinking %d chars on average, %d of %d visible answers" % (level, per[level]["thinking_chars_avg"], ans, len(CHAT_QS)))
    return case_result(rec, ok, notes, {"levels": per})


# ===================================================================================================== case 6: parallel clients
def case_parallel(c, a):
    results, notes, ok = {}, [], True
    words = ["amber", "basalt", "cobalt", "dahlia", "ember", "fjord", "garnet", "heron"]
    for n in [int(x) for x in a.parallel_levels.split(",") if x]:
        recs = [Recorder() for _ in range(n)]
        cross, errors = [], []

        def client(i):
            w = words[i % len(words)] + str(i)
            jobs = [
                ("Your secret word is %s. Repeat the secret word once, then name the capital of France in one sentence." % w, 160),
                ("Your secret word is %s. Write a Python function that reverses a string, in a code block, and end the reply with the secret word." % w, 400),
                ("Your secret word is %s. %s\n\nReply with the secret word and then the first section name." % (w, long_doc(3500, "p%d-%d" % (i, int(time.time())))), 60),
            ]
            for k in range(2):
                prompt, cap = jobs[(i + k) % len(jobs)]
                r = c.chat([{"role": "user", "content": prompt}], temperature=0, max_tokens=cap, reasoning_effort="none", timeout=min(c.timeout, 1200))
                recs[i].add(r)
                if r.get("error"):
                    errors.append("client %d: %s" % (i, r["error"][:80]))
                    continue
                t = r["text"].lower()
                if w not in t:
                    cross.append("client %d (job %d): its secret word %s is missing; reply starts: %r" % (i, (i + k) % 3, w, r["text"][:80]))
                for other in words:
                    o = other + str((i + 1) % 8)
                    if o != w and re.search(r"\b%s\b" % re.escape(o), t):
                        cross.append("client %d: the reply names another client's word %s" % (i, o))
        t0 = time.time()
        th = [threading.Thread(target=client, args=(i,)) for i in range(n)]
        for t in th:
            t.start()
        for t in th:
            t.join()
        wall = time.time() - t0
        allr = [r for rc in recs for r in rc.reqs if not r.get("error")]
        toks = sum(r.get("completion_tokens") or 0 for r in allr)
        dec = [r["decode_tps"] for r in allr if r.get("decode_tps")]
        walls = [r["wall"] for r in allr]
        stall = bool(walls) and max(walls) > 6 * max(statistics.median(walls), 5.0)
        lvl = {"clients": n, "wall_s": round(wall, 1), "requests": sum(len(rc.reqs) for rc in recs), "completion_tokens": toks,
               "throughput_tps": round(toks / wall, 1) if wall else None, "decode_tps_median": round(statistics.median(dec), 1) if dec else None,
               "slowest_request_s": round(max(walls), 1) if walls else None, "errors": len(errors), "cross_talk": len(cross), "stall": stall,
               "empty": sum(rc.empty for rc in recs), "truncated": sum(rc.trunc for rc in recs), "looping": sum(rc.loops for rc in recs)}
        # decode burst: n clients, the same short prompt, 200 tokens each: total decode throughput of the engine at this concurrency
        bres = []

        def burst(i):
            bres.append(c.chat([{"role": "user", "content": SPEED_PROMPT_P}], temperature=0, max_tokens=200, reasoning_effort="none", timeout=min(c.timeout, 1200)))
        bt0 = time.time()
        bth = [threading.Thread(target=burst, args=(i,)) for i in range(n)]
        for t in bth:
            t.start()
        for t in bth:
            t.join()
        bwall = time.time() - bt0
        btoks = sum(r.get("completion_tokens") or 0 for r in bres if not r.get("error"))
        lvl["burst_tps"] = round(btoks / bwall, 1) if bwall else None
        lvl["burst_errors"] = sum(1 for r in bres if r.get("error"))
        lvl["cross_samples"] = cross[:3]
        results[str(n)] = lvl
        good = not errors and not cross and not stall and not lvl["empty"]
        ok &= good
        notes.append("%d clients: decode burst %s tok/s total; mixed work %s tok/s total, %s tok/s per request, slowest %s s%s" % (n, lvl["burst_tps"], lvl["throughput_tps"], lvl["decode_tps_median"], lvl["slowest_request_s"],
                                                                                          "" if good else "  FAIL: " + "; ".join((errors + cross)[:2] or ["stall" if stall else "empty reply"])))
    d = {"verdict": PASS if ok else FAIL, "notes": notes, "levels": results, "requests": sum(v["requests"] for v in results.values()),
         "empty": sum(v["empty"] for v in results.values()), "truncated": sum(v["truncated"] for v in results.values()),
         "looping": sum(v["looping"] for v in results.values()), "errors": sum(v["errors"] for v in results.values())}
    try:
        d["server_status"] = c.get("/v1/status")
    except Exception:  # noqa: BLE001
        pass
    return d


CASES = {"coding": case_coding, "agent": case_agent, "vision": case_vision, "longdoc": case_longdoc, "chat": case_chat, "parallel": case_parallel}


# ===================================================================================================== table
def fmt(v, d=1):
    return "-" if v is None else ("%.*f" % (d, v) if isinstance(v, float) else str(v))


def markdown(res, title="Usage suite"):
    lines = ["# %s" % title, "", "Server: %s, model %s" % (res.get("url"), res.get("model")), "",
             "| case | verdict | requests | prefill tok/s | decode tok/s | first token s | cached tokens | empty / loop / truncated | notes |", "|---|---|---|---|---|---|---|---|---|"]
    for name, r in res["cases"].items():
        notes = "; ".join(r.get("notes") or []).replace("|", "/")
        lines.append("| %s | **%s** | %s | %s | %s | %s | %s | %s / %s / %s | %s |" % (
            name, r["verdict"].upper(), r.get("requests", "-"), fmt(r.get("prefill_tps")), fmt(r.get("decode_tps")), fmt(r.get("ttft_s"), 2),
            r.get("cached_tokens", "-"), r.get("empty", 0), r.get("looping", 0), r.get("truncated", 0), notes[:420]))
    par = (res["cases"].get("parallel") or {}).get("levels")
    if par:
        lines += ["", "| clients | decode burst total tok/s | mixed work total tok/s | per-request decode tok/s | slowest request s | errors | cross-talk | stall |", "|---|---|---|---|---|---|---|---|"]
        for k, v in par.items():
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (k, fmt(v.get("burst_tps")), fmt(v["throughput_tps"]), fmt(v["decode_tps_median"]), fmt(v["slowest_request_s"]), v["errors"], v["cross_talk"], v["stall"]))
    return "\n".join(lines) + "\n"


def run_suite(url, model=None, key=None, cases=None, **kw):
    a = argparse.Namespace(long=kw.get("long", "30000,60000"), agent_turns=kw.get("agent_turns", 20), parallel_levels=kw.get("parallel_levels", "1,2,4,8"))
    c = Client(url, model, key, kw.get("timeout", 1800))
    c.find_model()
    res = {"url": url, "model": c.model, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "cases": {}}
    for name in cases or list(CASES):
        t0 = time.time()
        try:
            r = CASES[name](c, a)
        except Exception as e:  # noqa: BLE001 (one broken case must not end the suite)
            r = {"verdict": FAIL, "notes": ["suite error: %s: %s" % (type(e).__name__, e)]}
        r["seconds"] = round(time.time() - t0)
        res["cases"][name] = r
        print("[%s] %-9s %-5s %4ds  %s" % (time.strftime("%H:%M:%S"), name, r["verdict"].upper(), r["seconds"], "; ".join(r.get("notes") or [])[:200]), flush=True)
    res["verdict"] = FAIL if any(r["verdict"] == FAIL for r in res["cases"].values()) else PASS
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--model")
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--long", default="30000,60000", help="document sizes in tokens (the 32K and 64K classes; the server needs a context of 32K / 64K or more)")
    ap.add_argument("--agent-turns", type=int, default=20)
    ap.add_argument("--parallel-levels", default="1,2,4,8")
    ap.add_argument("--key-env", default="STRATA_API_KEY")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out")
    ap.add_argument("--md")
    a = ap.parse_args(argv)
    key = os.environ.get(a.key_env) or None
    res = run_suite(a.url, a.model, key, [x for x in a.cases.split(",") if x], long=a.long, agent_turns=a.agent_turns,
                    parallel_levels=a.parallel_levels, timeout=a.timeout)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1, sort_keys=True)
    md = markdown(res)
    if a.md:
        with open(a.md, "w", encoding="utf-8") as f:
            f.write(md)
    print(md)
    return 0 if res["verdict"] == PASS else 1


if __name__ == "__main__":
    sys.exit(main())
