#!/usr/bin/env python3
"""cell_runner.py: runs on a TEST BOX (never on the owner PC). One release-grid cell = one release x one box x one model x drafter on/off.

    python3 cell_runner.py --plan PLAN.json --out RESULTS_DIR

PLAN.json (written by grid.py, never by hand) is {"cells": [spec, ...]}; every spec carries everything the cell needs (see grid.py `plan`).
The cells run one after the other (one engine at a time); every cell writes RESULTS_DIR/<label>/<cell id>.json whatever happens.

What a cell does, through the USER path (the released installer and `strata launch`, in a separate test HOME):
  1. install    install.sh of the release folder (a URL, or a folder this runner serves on loopback) into HOME/.local/share/strata
  2. start      `strata launch --yes --no-tui --restart ...` with the model files, drafter on or off; the engine comes from the feed
  3. first chat six fixed questions at temperature 0: answers right (>= min_correct) and never empty
  4. 4K / 32K   needle-in-a-haystack prompts of about 4000 and about 30000 tokens: the needle is found, the reply is not empty
  5. agent      a scripted session of about 10 turns with tools: no empty reply, no endless reply, the prompt cache reused from turn to turn
  6. speed      decode and prefill tok/s (the server's own timings), several repeats, with the spread
  7. fingerprint a hash of the greedy replies, compared by grid.py with the previous release's
The server is stopped at the end (`strata launch --stop`); a memory guard stops it when the box runs short of RAM or starts to swap.
Standard library only, Python 3.8+. No keys or tokens are read or printed.
"""
from __future__ import annotations

import argparse
import ast
import glob
import http.server
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "usage"))      # in a checkout; on a box the scripts sit in one folder
import grid_lib as gl  # noqa: E402

PASS, FAIL, SKIP = gl.PASS, gl.FAIL, gl.SKIP

# --------------------------------------------------------------------------------------------------------------------
# the fixed question set (first chat; also the greedy fingerprint) and the tools of the agent session
# --------------------------------------------------------------------------------------------------------------------
QUESTIONS = [
    ("What is the capital of Australia? Answer in one short sentence.", [r"canberra"]),
    ("What is 17 times 23? Answer with just the number.", [r"\b391\b"]),
    ("What is the chemical symbol for gold? Answer with just the symbol.", [r"\bau\b"]),
    ("How many days are in a leap year? Answer with just the number.", [r"\b366\b"]),
    ("Which planet is the largest in our solar system? Answer with one word.", [r"jupiter"]),
    ("Spell the word 'strata' backwards. Answer with just the word.", [r"atarts"]),
]

NOTES = {
    "alpha": "The vault code is 4821.",
    "beta": "The meeting is at 14:30 in room 7.",
    "gamma": "Budget: 1200 for hardware and 800 for travel.",
    "delta": "The project codename is Kestrel.",
}

TOOLS = [
    {"type": "function", "function": {"name": "list_notes", "description": "List the names of all notes.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "read_note", "description": "Read one note by name.", "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}},
    {"type": "function", "function": {"name": "search_notes", "description": "Find the notes that contain a word.", "parameters": {"type": "object", "properties": {"word": {"type": "string"}}, "required": ["word"]}}},
    {"type": "function", "function": {"name": "calculator", "description": "Evaluate an arithmetic expression such as 12*(3+4).", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
]

# (user message, regexes that the final answer should match (all), is_content_check)
AGENT_SCRIPT = [
    ("List the notes you can see.", [r"alpha", r"beta", r"gamma", r"delta"]),
    ("Read the note called alpha and tell me the vault code.", [r"4821"]),
    ("What time is the meeting? Look in the right note.", [r"14:30|2:30"]),
    ("Add the two budget numbers in note gamma using the calculator.", [r"2000|2,000"]),
    ("What is the project codename? It is in note delta.", [r"kestrel"]),
    ("Now multiply the vault code by 3 with the calculator.", [r"14463|14,463"]),
    ("Search the notes for the word room and tell me which note has it.", [r"beta"]),
    ("Read the note called epsilon.", [r"(not|no).{0,40}(exist|found|such)|doesn't|does not|unknown|error"]),
    ("Summarise in two sentences what you learned in this conversation.", [r"4821|kestrel|14:30|2000"]),
    ("Which was the first note I asked you to read? Answer in a few words.", [r"alpha"]),
]

SPEED_PROMPT = "Write a long, detailed essay about the history of the printing press, from Gutenberg to the modern day."

ADJ = ["quiet", "bright", "heavy", "narrow", "ancient", "rapid", "hollow", "silver", "narrow", "distant", "warm", "sharp"]
NOUN = ["harbour", "relay", "orchard", "bridge", "signal", "archive", "glacier", "engine", "lantern", "market", "ledger", "tunnel"]

CAUSES = [
    (r"never rang|operation not supported on global/shared", "CUDA trap: a shared-memory read through __ldg (#6713, the 0.2.20 Q8_0 / Unsloth crash)"),
    (r"begin capture failed", "CUDA graph capture failed"),
    (r"out of memory|cudaErrorMemoryAllocation|hipErrorOutOfMemory|failed to allocate|alloc(ation)? failed|VRAM exhausted", "out of memory"),
    (r"missing (dspark )?tensor", "a tensor the engine asks for is missing in the file (drafter or model)"),
    (r"invalid \.strata magic", "Invalid .strata magic (a GGUF read as a Strata pack)"),
    (r"illegal instruction", "illegal instruction (CPU features)"),
    (r"segmentation fault|sigsegv|core dumped|signal 11", "segmentation fault"),
    (r"did not become ready", "the server did not become ready"),
    (r"no (supported|usable) (graphics|gpu|card)|no engine for this", "no engine for this computer"),
    (r"not enough (free )?(ram|memory)|memory guard", "not enough memory"),
    (r"tokenizer is missing|tokenizer.*not found", "the model's tokenizer files are missing (they are downloaded beside the model; a box without DNS cannot)"),
]


# --------------------------------------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------------------------------------
def now() -> float:
    return time.time()


def stamp(t=None) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t or now()))


def say(msg: str):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def clip(text, n=300):
    text = (text or "").replace("\r", "")
    return text if len(text) <= n else text[: n - 3] + "..."


def tail(path, n=40):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return ""


def find_cause(*texts) -> str:
    blob = "\n".join(t for t in texts if t)
    for pat, name in CAUSES:
        m = re.search(pat, blob, re.I)
        if m:
            line = ""
            for ln in blob.splitlines():
                if re.search(pat, ln, re.I):
                    line = ln.strip()
                    break
            return "%s [%s]" % (name, clip(line, 160))
    return ""


def free_port(preferred: int) -> int:
    for p in [preferred] + list(range(preferred + 1, preferred + 40)):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    return preferred


# --------------------------------------------------------------------------------------------------------------------
# HTTP to the server under test
# --------------------------------------------------------------------------------------------------------------------
class Api:
    def __init__(self, base: str):
        self.base = base.rstrip("/")
        self.model = None

    def get(self, path, timeout=30):
        with urllib.request.urlopen(self.base + path, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))

    def models(self):
        data = self.get("/v1/models")
        self.model = data["data"][0]["id"]
        return self.model

    def chat(self, messages, timeout=900, **extra):
        """One non-streaming chat completion. Returns {text, finish, usage, timings, message, wall, error}."""
        body = {"model": self.model, "messages": messages, "stream": False, "reasoning_effort": "none"}
        body.update(extra)
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = now()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            return {"error": "HTTP %s: %s" % (e.code, clip(e.read().decode("utf-8", "replace"), 200)), "wall": now() - t0, "text": ""}
        except Exception as e:  # noqa: BLE001 (a dead server shows up as many different exceptions)
            return {"error": "%s: %s" % (type(e).__name__, clip(str(e), 200)), "wall": now() - t0, "text": ""}
        wall = now() - t0
        try:
            ch = data["choices"][0]
        except (KeyError, IndexError, TypeError):
            return {"error": "no choices in the reply: " + clip(json.dumps(data), 200), "wall": wall, "text": ""}
        msg = ch.get("message") or {}
        usage = data.get("usage") or {}
        timings = data.get("timings") or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        if cached is None:
            cached = timings.get("cache_n")
        return {"text": (msg.get("content") or ""), "finish": ch.get("finish_reason"), "message": msg,
                "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                "cached_tokens": cached, "timings": timings, "wall": wall, "error": None}


# --------------------------------------------------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------------------------------------------------
def haystack(n_chars: int, needle: str, salt: str) -> str:
    """Deterministic filler text of about n_chars with one needle sentence in the middle; `salt` makes the start unique (no cache hit)."""
    out, size, i = ["Log of run %s." % salt], 0, 0
    sentences = []
    while size < n_chars:
        i += 1
        s = "Entry %d: the %s %s of station %d reported %d units on day %d." % (
            i, ADJ[i % len(ADJ)], NOUN[(i * 7) % len(NOUN)], (i * 13) % 997, (i * 31) % 4093, i % 31)
        sentences.append(s)
        size += len(s) + 1
    sentences.insert(len(sentences) // 2, needle)
    return " ".join(out + sentences)


def needle_prompt(n_chars: int, code: str, salt: str) -> str:
    text = haystack(n_chars, "Remember this: the secret access code of the vault is %s." % code, salt)
    return text + "\n\nWhat is the secret access code of the vault mentioned in the text above? Answer with just the code."


def has_repetition(text: str, n=12, times=5) -> bool:
    words = text.split()
    if len(words) < n * times:
        return False
    seen = {}
    for i in range(len(words) - n):
        key = " ".join(words[i:i + n])
        seen[key] = seen.get(key, 0) + 1
        if seen[key] >= times:
            return True
    return False


def run_tool(name, args):
    try:
        if name == "list_notes":
            return ", ".join(NOTES)
        if name == "read_note":
            n = str(args.get("name", "")).strip().lower()
            return NOTES.get(n) or "error: no such note '%s'" % n
        if name == "search_notes":
            w = str(args.get("word", "")).lower()
            hits = [k for k, v in NOTES.items() if w and w in v.lower()]
            return ", ".join(hits) or "no note contains '%s'" % w
        if name == "calculator":
            expr = str(args.get("expression", ""))
            if not re.fullmatch(r"[0-9+\-*/(). %]+", expr):
                return "error: only numbers and + - * / ( ) are allowed"
            tree = ast.parse(expr, mode="eval")
            return str(eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, {}))  # noqa: S307 (only digits and operators got through the regex)
    except Exception as e:  # noqa: BLE001
        return "error: %s" % e
    return "error: unknown tool %s" % name


# --------------------------------------------------------------------------------------------------------------------
# the checks (against a running server; also used by the unit tests against a fake one)
# --------------------------------------------------------------------------------------------------------------------
def step(status, reason=None, **kw):
    d = {"status": status}
    if reason:
        d["reason"] = reason
    d.update(kw)
    return d


def check_first_chat(api: Api, profile: dict):
    answers, texts, wrong, empty = [], [], 0, 0
    for q, pats in QUESTIONS:
        r = api.chat([{"role": "user", "content": q}], temperature=0, max_tokens=int(profile.get("answer_tokens", 200)),
                     timeout=float(profile.get("request_timeout_s", 900)))
        if r.get("error"):
            return step(FAIL, "request failed: " + r["error"], answers=answers), []
        text = r["text"]
        texts.append(text)
        ok = all(re.search(p, text.lower()) for p in pats)
        if not text.strip():
            empty += 1
        elif not ok:
            wrong += 1
        answers.append({"q": q, "a": clip(text, 160), "ok": ok, "finish": r.get("finish")})
    right = sum(1 for a in answers if a["ok"])
    need = int(profile.get("min_correct", 5))
    if empty:
        return step(FAIL, "%d empty reply(ies)" % empty, answers=answers, right=right), texts
    if right < need:
        return step(FAIL, "%d of %d right (need %d)" % (right, len(QUESTIONS), need), answers=answers, right=right), texts
    return step(PASS, answers=answers, right=right), texts


def check_needle(api: Api, target_tokens: int, profile: dict, label: str, chars_per_token: float, salt: str):
    """One needle prompt of about target_tokens. Returns (step, measured chars_per_token, prefill tok/s)."""
    code = "%s-%d" % ("OSPREY" if "4" in label else "KESTREL", 4000 + (target_tokens // 7) % 5000)
    prompt = needle_prompt(int(target_tokens * chars_per_token), code, salt)
    r = api.chat([{"role": "user", "content": prompt}], temperature=0, max_tokens=60, timeout=float(profile.get("big_request_timeout_s", 3600)))
    if r.get("error"):
        return step(FAIL, "request failed: " + r["error"], wall_s=round(r.get("wall", 0), 1)), chars_per_token, None
    text = r["text"]
    pt = r.get("prompt_tokens") or 0
    cpt = (len(prompt) / pt) if pt else chars_per_token
    rate = (r.get("timings") or {}).get("prompt_per_second")
    if not rate and pt and r.get("wall"):
        rate = pt / r["wall"]
    digits = code.split("-")[1]
    if not text.strip():
        return step(FAIL, "empty reply", prompt_tokens=pt, wall_s=round(r["wall"], 1)), cpt, rate
    if digits not in text:
        return step(FAIL, "the needle was not found (answer: %s)" % clip(text, 80), prompt_tokens=pt, wall_s=round(r["wall"], 1)), cpt, rate
    return step(PASS, prompt_tokens=pt, wall_s=round(r["wall"], 1), prefill_tps=round(rate, 1) if rate else None), cpt, rate


def check_agent(api: Api, profile: dict):
    turns = int(profile.get("agent_turns", 10))
    reply_cap = int(profile.get("agent_reply_tokens", 700))
    messages = [{"role": "system", "content": "You are a careful assistant with tools for a small notes app. Use the tools to answer; never guess what a note says. Keep answers short."}]
    log, reuse, problems, correct, requests, reqs = [], [], [], 0, 0, []
    prev_prompt = None
    for n, (user, pats) in enumerate(AGENT_SCRIPT[:turns], 1):
        messages.append({"role": "user", "content": user})
        final, tool_names, wall = "", [], 0.0
        for _round in range(6):
            r = api.chat(messages, tools=TOOLS, temperature=0, max_tokens=reply_cap, timeout=float(profile.get("request_timeout_s", 900)))
            requests += 1
            wall += r.get("wall", 0)
            if r.get("error"):
                return step(FAIL, "turn %d: request failed: %s" % (n, r["error"]), turns=log), None
            msg = r.get("message") or {}
            calls = msg.get("tool_calls") or []
            pt, cached = r.get("prompt_tokens"), r.get("cached_tokens")
            reqs.append([pt, cached, round(r.get("wall", 0), 1), (r.get("timings") or {}).get("prompt_n")])      # prompt tokens, cached tokens, seconds, tokens read now
            if prev_prompt and cached is not None and prev_prompt > 0:
                reuse.append(min(1.0, cached / prev_prompt))
            prev_prompt = pt
            if r.get("finish") == "length":
                problems.append("turn %d: the reply hit the %d-token limit (endless reply)" % (n, reply_cap))
            if has_repetition(r["text"]):
                problems.append("turn %d: the reply repeats itself" % n)
            if calls:
                messages.append({"role": "assistant", "content": r["text"] or None, "tool_calls": calls})
                for c in calls:
                    fn = (c.get("function") or {})
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except ValueError:
                        args = {}
                    tool_names.append(fn.get("name"))
                    messages.append({"role": "tool", "tool_call_id": c.get("id") or "call_%d" % len(messages),
                                     "content": run_tool(fn.get("name"), args if isinstance(args, dict) else {})})
                continue
            final = r["text"]
            messages.append({"role": "assistant", "content": final})
            break
        else:
            problems.append("turn %d: more than 6 tool rounds without an answer" % n)
        if not final.strip():
            problems.append("turn %d: empty reply" % n)
        ok = bool(final.strip()) and all(re.search(p, final.lower()) for p in pats)
        correct += 1 if ok else 0
        log.append({"turn": n, "tools": tool_names, "ok": ok, "answer": clip(final, 120), "wall_s": round(wall, 1)})
    med_reuse = gl.median(reuse)
    need = int(profile.get("min_agent_correct", 4))
    info = {"turns": log, "requests": requests, "reqs": reqs, "correct": correct, "reuse_median": round(med_reuse, 3) if med_reuse is not None else None}
    if problems:
        return step(FAIL, "; ".join(problems[:4]), **info), info
    if correct < need:
        return step(FAIL, "only %d of %d turns answered right (need %d)" % (correct, len(log), need), **info), info
    if med_reuse is not None and med_reuse < float(profile.get("min_reuse", 0.5)):
        return step(FAIL, "the prompt cache was reused for only %.0f%% of the previous turn's prompt (median)" % (med_reuse * 100), **info), info
    if med_reuse is None:
        info["note"] = "the server reports no cached-token count"
    return step(PASS, **info), info


def measure_speed(api: Api, profile: dict, chars_per_token: float):
    reps = int(profile.get("speed_reps", 4))
    dec, pre, texts = [], [], []
    for i in range(reps):
        r = api.chat([{"role": "user", "content": SPEED_PROMPT}], temperature=0, max_tokens=int(profile.get("decode_tokens", 160)),
                     timeout=float(profile.get("request_timeout_s", 900)))
        if r.get("error"):
            return step(FAIL, "decode run failed: " + r["error"]), [], [], []
        t = r.get("timings") or {}
        rate = t.get("predicted_per_second")
        n = r.get("completion_tokens") or t.get("predicted_n")
        if not rate and n and n > 4 and r.get("wall"):
            rate = n / r["wall"]
        if i > 0 or reps == 1:                          # the first run warms up the kernels and the expert cache
            dec.append(rate)
        texts.append(r["text"])
    pre_tokens = int(profile.get("prefill_tokens", 2000))
    for i in range(int(profile.get("prefill_reps", 3))):
        prompt = needle_prompt(int(pre_tokens * chars_per_token), "SPEED-%d" % i, "speed-%d-%d" % (i, int(now())))
        r = api.chat([{"role": "user", "content": prompt}], temperature=0, max_tokens=4, timeout=float(profile.get("big_request_timeout_s", 3600)))
        if r.get("error"):
            return step(FAIL, "prefill run failed: " + r["error"]), dec, pre, texts
        rate = (r.get("timings") or {}).get("prompt_per_second")
        if not rate and r.get("prompt_tokens") and r.get("wall"):
            rate = r["prompt_tokens"] / r["wall"]
        pre.append(rate)
    dec_ok = [d for d in dec if d]
    if not dec_ok:
        return step(FAIL, "the server gave no decode speed", decode=dec, prefill=pre), dec, pre, texts
    return step(PASS, decode=[round(d, 2) for d in dec_ok], prefill=[round(p, 1) for p in pre if p],
                decode_spread_pct=round(gl.spread(dec_ok) * 100, 1), prefill_spread_pct=round(gl.spread(pre) * 100, 1)), dec, pre, texts


def run_checks(api: Api, profile: dict, steps: dict, deadline: float):
    """All the post-start checks. Fills `steps`; returns the numbers dict."""
    numbers = {}
    cpt = 3.0

    def time_left():
        return deadline - now()

    try:
        api.models()
    except Exception as e:  # noqa: BLE001
        steps["first_chat"] = step(FAIL, "the server does not list a model: %s" % e)
        return numbers
    fc, texts = check_first_chat(api, profile)
    steps["first_chat"] = fc
    greedy_texts = list(texts)
    # 4K
    if time_left() < 60:
        steps["prompt_4k"] = step(SKIP, "cell time budget used up")
    else:
        s4, cpt, rate4 = check_needle(api, int(profile.get("small_tokens", 4000)), profile, "4K", cpt, "4k-%d" % int(now()))
        steps["prompt_4k"] = s4
        if rate4:
            numbers["prefill_4k_tps"] = round(rate4, 1)
        # 32K only when the projected time fits the step budget
        big = int(profile.get("big_tokens", 30000))
        budget = float(profile.get("step_budget_s", 1500))
        if profile.get("skip_big"):
            steps["prompt_32k"] = step(SKIP, str(profile["skip_big"]))
        elif s4["status"] != PASS:
            steps["prompt_32k"] = step(SKIP, "the 4K prompt did not pass")
        else:
            proj = (big / rate4) if rate4 else None
            if proj and proj > budget:
                steps["prompt_32k"] = step(SKIP, "projected %.0f s at %.0f tok/s is over the %.0f s step budget" % (proj, rate4, budget))
            elif time_left() < (proj or 600) + 60:
                steps["prompt_32k"] = step(SKIP, "cell time budget used up")
            else:
                s32, cpt, rate32 = check_needle(api, big, profile, "32K", cpt, "32k-%d" % int(now()))
                steps["prompt_32k"] = s32
                if rate32:
                    numbers["prefill_32k_tps"] = round(rate32, 1)
    # agent
    if time_left() < 120:
        steps["agent"] = step(SKIP, "cell time budget used up")
    else:
        sa, info = check_agent(api, profile)
        steps["agent"] = sa
        if info and info.get("reuse_median") is not None:
            numbers["agent_reuse"] = info["reuse_median"]
    # speed
    if time_left() < 120:
        steps["speed"] = step(SKIP, "cell time budget used up")
    else:
        sp, dec, pre, dtexts = measure_speed(api, profile, cpt)
        steps["speed"] = sp
        numbers["decode"] = [round(d, 2) for d in dec if d]
        numbers["prefill"] = [round(p, 1) for p in pre if p]
        numbers["decode_tps"] = round(gl.median(dec), 2) if gl.median(dec) else None
        numbers["prefill_tps"] = round(gl.median(pre), 1) if gl.median(pre) else None
        if dtexts:
            greedy_texts.append(dtexts[0])
    if greedy_texts:
        steps["fingerprint"] = step(PASS, value=gl.text_fingerprint(greedy_texts), items=len(greedy_texts),
                                    heads=[clip(t, 40) for t in greedy_texts[:3]])
    return numbers


# --------------------------------------------------------------------------------------------------------------------
# a loopback server for a local feed folder
# --------------------------------------------------------------------------------------------------------------------
class FeedServer:
    """Serves a folder on 127.0.0.1 from a thread of this process (no second process to clean up)."""

    def __init__(self, folder):
        handler = type("H", (http.server.SimpleHTTPRequestHandler,), {"log_message": lambda *a, **k: None})
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), lambda *a, **k: handler(*a, directory=str(folder), **k))
        self.url = "http://127.0.0.1:%d" % self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


# --------------------------------------------------------------------------------------------------------------------
# memory guard
# --------------------------------------------------------------------------------------------------------------------
def meminfo() -> dict:
    out = {}
    try:
        with open("/proc/meminfo") as f:
            for ln in f:
                k, _, v = ln.partition(":")
                out[k] = int(v.split()[0]) / 1048576.0       # GiB
    except OSError:
        pass
    return out


def swap_used_gib() -> float:
    mi = meminfo()
    if mi:
        return mi.get("SwapTotal", 0) - mi.get("SwapFree", 0)
    try:                                                      # macOS
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=10).stdout
        m = re.search(r"used = ([\d.]+)([MG])", out)
        if m:
            return float(m.group(1)) / (1024.0 if m.group(2) == "M" else 1.0)
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def our_anon_rss_gib(home) -> float:
    """Anonymous resident memory (RssAnon, not the file cache) of the processes started from this cell's HOME: the engine's own RAM."""
    total = 0.0
    for d in glob.glob("/proc/[0-9]*"):
        try:
            with open(d + "/cmdline", "rb") as f:
                cmd = f.read().decode("utf-8", "replace")
            if str(home) not in cmd:
                continue
            with open(d + "/status") as f:
                for ln in f:
                    if ln.startswith("RssAnon:"):
                        total += int(ln.split()[1]) / 1048576.0
        except OSError:
            continue
    return total


class Guard(threading.Thread):
    """Watches free RAM and swap while a model runs; when either leaves its limit it stops the server (strata launch --stop) and says why."""

    def __init__(self, min_free_gib, max_swap_growth_gib, stop_fn, rss_cap_gib=None, home=None):
        super().__init__(daemon=True)
        self.min_free, self.max_swap, self.stop_fn = min_free_gib, max_swap_growth_gib, stop_fn
        self.rss_cap, self.home, self.peak_rss = rss_cap_gib, home, 0.0
        self.base_swap = swap_used_gib()
        self.tripped = None
        self.min_seen = None
        self._halt = threading.Event()

    def run(self):
        while not self._halt.wait(3.0):
            avail = meminfo().get("MemAvailable")
            if avail is not None:
                self.min_seen = avail if self.min_seen is None else min(self.min_seen, avail)
                if self.min_free and avail < self.min_free:
                    self.trip("memory guard: only %.1f GiB RAM available (floor %.1f)" % (avail, self.min_free))
                    return
            if self.rss_cap and self.home:
                rss = our_anon_rss_gib(self.home)
                self.peak_rss = max(self.peak_rss, rss)
                if rss > self.rss_cap:
                    self.trip("memory cap: the engine holds %.1f GiB of anonymous RAM (cap %.1f GiB)" % (rss, self.rss_cap))
                    return
            grown = swap_used_gib() - self.base_swap
            if self.max_swap is not None and grown > self.max_swap:
                self.trip("memory guard: swap grew by %.2f GiB (limit %.2f)" % (grown, self.max_swap))
                return

    def trip(self, why):
        self.tripped = why
        say(why)
        try:
            self.stop_fn()
        except Exception as e:  # noqa: BLE001
            say("stop after guard failed: %s" % e)

    def halt(self):
        self._halt.set()


# --------------------------------------------------------------------------------------------------------------------
# one cell
# --------------------------------------------------------------------------------------------------------------------
def link_files(files, dest: Path):
    """Make dest a folder holding the model files by hard link (same disk), else symlink: no copy of a 100 GB model."""
    dest.mkdir(parents=True, exist_ok=True)
    for src in files:
        src = Path(os.path.expanduser(src))
        tgt = dest / src.name
        if tgt.exists() or tgt.is_symlink():
            continue
        try:
            os.link(src, tgt)
        except OSError:
            os.symlink(src, tgt)


def run_cmd(cmd, env, log_path, timeout, cwd=None):
    """Run a command to the end with output in log_path; returns (exit code or None on timeout, seconds)."""
    t0 = now()
    with open(log_path, "ab") as lf:
        p = subprocess.Popen(cmd, env=env, cwd=cwd, stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, 15)                          # our own child's group only
                p.wait(timeout=30)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(p.pid, 9)
                except Exception:  # noqa: BLE001
                    pass
            rc = None
    return rc, now() - t0


def newest_engine_log(home: Path, since: float) -> str:
    best, best_t = "", 0.0
    for base in (home / ".local" / "state", home / "Library" / "Application Support", home / ".strata"):
        for p in glob.glob(str(base / "**" / "*.engine.log"), recursive=True):
            try:
                t = os.path.getmtime(p)
            except OSError:
                continue
            if t >= since - 5 and t > best_t:
                best, best_t = p, t
    return best


def gpu_facts() -> str:
    out = []
    for cmd in (["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                ["rocm-smi", "--showproductname"], ["sh", "-c", "lspci 2>/dev/null | grep -i -E 'vga|display|3d' | head -4"]):
        if shutil.which(cmd[0]):
            try:
                o = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
                if o:
                    out.append(clip(o, 300))
            except Exception:  # noqa: BLE001
                pass
    return " | ".join(out)


def gpu_total_mib(dev=0):
    try:
        out = subprocess.run(["nvidia-smi", "-i", str(dev), "--query-gpu=memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
        return int(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


class Hog:
    """Holds GPU memory so a big card acts as a card of  GiB (the VRAM-class cells). Only its own process is ever stopped."""

    def __init__(self, grid_dir: Path, here: str, vram_class_gib: float, dev: int, seconds: int, log):
        self.proc = None
        total = gpu_total_mib(dev)
        if not total:
            raise RuntimeError("cannot read the card's memory size (nvidia-smi): VRAM-class cells need an NVIDIA card")
        hold = int(total - vram_class_gib * 1024)
        if hold <= 0:
            raise RuntimeError("the card (%d MiB) is not larger than the %.0f GiB class" % (total, vram_class_gib))
        src = os.path.join(here, "vram_hog.cu")
        exe = grid_dir / ("vram_hog-%s" % gl.sha256_file(src)[:8])
        if not exe.exists():
            r = subprocess.run(["nvcc", "-O1", src, "-o", str(exe)], capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                raise RuntimeError("nvcc could not build vram_hog.cu: %s" % clip(r.stderr, 200))
        self.proc = subprocess.Popen([str(exe), str(dev), str(hold), str(seconds)], stdout=open(log, "wb"), stderr=subprocess.STDOUT)
        for _ in range(60):
            if "hog holds" in tail(log, 3):
                break
            if self.proc.poll() is not None:
                raise RuntimeError("the VRAM hog stopped: " + clip(tail(log, 3), 200))
            time.sleep(1)
        self.held_mib = hold

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()                           # our own child
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()


FACT_RE = re.compile(r"experts?.*(gpu|vram|host|ram|resident)|hit.?rate|cache.?hit|hits?=|expert cache|offload", re.I)


def engine_facts(elog: str, api: "Api"):
    """The expert split and the cache hit rate the engine and the server report: lines of the engine log plus the server's status keys."""
    lines = []
    if elog:
        try:
            with open(elog, "r", encoding="utf-8", errors="replace") as f:
                allv = f.readlines()
        except OSError:
            allv = []
        hits = [ln.strip() for ln in allv if FACT_RE.search(ln)]
        lines = [clip(x, 220) for x in hits[:6] + hits[-6:]] if len(hits) > 12 else [clip(x, 220) for x in hits]
    status = {}
    for path in ("/v1/status", "/v1/telemetry"):
        try:
            data = api.get(path)
        except Exception:  # noqa: BLE001
            continue
        flat = {}

        def walk(prefix, v, depth=0):
            if depth > 4:
                return
            if isinstance(v, dict):
                for k, x in v.items():
                    walk(prefix + "." + str(k) if prefix else str(k), x, depth + 1)
            elif isinstance(v, (int, float, str)) and re.search(r"expert|hit|cache|vram|resident|offload|slots", prefix, re.I):
                flat[prefix] = v
        walk("", data)
        status[path] = dict(list(flat.items())[:30])
    return {"log_lines": lines, "status": status}


def run_cell(spec: dict, out_root: Path) -> dict:
    cid, label = spec["cell"], spec["label"]
    out_dir = out_root / label
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / (cid + ".log")
    res = {"cell": cid, "release": label, "parallel_flag": spec.get("parallel_flag"), "box": spec["box"], "model": spec["model"], "drafter": spec["drafter"], "steps": {},
           "numbers": {}, "started": stamp(), "gpu": gpu_facts(), "host": socket.gethostname()}
    profile = dict(spec.get("profile") or {})
    steps = res["steps"]
    t_start = now()
    deadline = t_start + float(profile.get("cell_budget_s", 7200))

    def finish(verdict=None, reason=None):
        res["ended"] = stamp()
        res["seconds"] = round(now() - t_start)
        if verdict:
            res["verdict"] = verdict
        if reason:
            res["skip_reason"] = reason
        (out_dir / (cid + ".json")).write_text(json.dumps(res, indent=1, sort_keys=True), encoding="utf-8")
        return res

    if spec.get("skip"):
        return finish(SKIP, spec["skip"])

    grid_dir = Path(os.path.expanduser(spec["grid_dir"]))
    home = grid_dir / ("home-" + label)
    home.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("STRATA_") and k not in ("LD_LIBRARY_PATH",)}
    env.update({"HOME": str(home), "XDG_STATE_HOME": str(home / ".local" / "state"), "XDG_DATA_HOME": str(home / ".local" / "share"),
                "XDG_CONFIG_HOME": str(home / ".config"), "XDG_CACHE_HOME": str(home / ".cache"),
                "STRATA_DATA_DIR": str(home / ".local" / "share" / "strata"), "STRATA_BIN_DIR": str(home / ".local" / "bin"),
                "PYTHONUNBUFFERED": "1", "NO_COLOR": "1"})
    env.update({k: str(v) for k, v in (spec.get("env") or {}).items()})
    feed_server = None
    guard = None
    hog = None
    strata = str(home / ".local" / "bin" / "strata")
    try:
        # ---- feed
        feed = spec["feed"]
        if not re.match(r"^https?://", feed):
            feed_server = FeedServer(os.path.expanduser(feed))
            feed = feed_server.url
        env["STRATA_UPDATE_FEED"] = feed
        if spec.get("trust"):
            env["STRATA_UPDATE_TRUSTED_KEYS"] = os.path.expanduser(spec["trust"])
            env["STRATA_INSTALL_TEST_TRUST"] = "1"
        # ---- install (once per release label and box; later cells reuse it)
        mark = home / ".grid-install-ok"
        if mark.exists() and os.path.exists(strata):
            steps["install"] = step(PASS, note="already installed by an earlier cell of this pass", version=mark.read_text().strip())
        else:
            t0 = now()
            inst = grid_dir / ("install-%s.sh" % label)
            try:
                with urllib.request.urlopen(feed + "/install.sh", timeout=120) as r:
                    inst.write_bytes(r.read())
            except Exception as e:  # noqa: BLE001
                steps["install"] = step(FAIL, "could not fetch install.sh from %s: %s" % (feed, e))
                raise StopCell()
            ienv = dict(env, STRATA_FEED=feed)
            rc, secs = run_cmd(["sh", str(inst)], ienv, str(log_path), 900)
            if rc != 0 or not os.path.exists(strata):
                steps["install"] = step(FAIL, "install.sh exited %s: %s" % (rc, clip(tail(log_path, 6), 300)), seconds=round(secs))
                raise StopCell()
            ver = subprocess.run([strata, "--version"], env=env, capture_output=True, text=True, timeout=60).stdout.strip()
            mark.write_text(ver or "installed")
            steps["install"] = step(PASS, seconds=round(secs), version=ver)
        # ---- model folder
        model = spec["model_spec"]
        if model.get("files"):
            for src in model["files"]:                      # a shared copy another lane holds may be gone or half-written: say so plainly
                sp = Path(os.path.expanduser(src))
                want = (model.get("sizes") or {}).get(sp.name)
                if not sp.exists():
                    steps["launch"] = step(FAIL, "model file %s is gone (a shared copy another lane held was removed); fetch it again" % src)
                    raise StopCell()
                if want and sp.stat().st_size != want:
                    steps["launch"] = step(FAIL, "model file %s is %d bytes, pinned %d (incomplete or another file)" % (src, sp.stat().st_size, want))
                    raise StopCell()
            mdir = grid_dir / "models" / spec["model"]
            link_files(model["files"], mdir)
            model_path = str(mdir)
        else:
            model_path = os.path.expanduser(model["path"])
        if profile.get("vision", "off") != "off" and model.get("vision_files"):
            # `strata launch --model-path` takes pictures from files BESIDE the model (Inkling-2-Mini-Vision.strata, or mmproj*.gguf for a GGUF) and
            # starts without the picture reader when there are none, whatever --vision says; so the grid puts them there.
            for vf in model["vision_files"]:
                if not os.path.exists(os.path.expanduser(vf)):
                    steps["launch"] = step(FAIL, "picture reader file %s is missing" % vf)
                    raise StopCell()
            link_files(model["vision_files"], Path(model_path))
        need = [Path(model_path)] if not model.get("files") else []
        for p in need:
            if not p.exists():
                steps["launch"] = step(FAIL, "model files not found: %s" % p)
                raise StopCell()
        # ---- RAM floor before the start
        mi = meminfo()
        floor = float(profile.get("min_free_gib", 0))
        if mi and floor and mi.get("MemAvailable", 1e9) < floor:
            steps["launch"] = step(SKIP, "only %.1f GiB RAM available, the box rule needs %.1f" % (mi["MemAvailable"], floor))
            return finish(SKIP, steps["launch"]["reason"])
        # ---- launch
        port = free_port(int(spec.get("port", 18770)))
        cmd = [strata, "launch", "--yes", "--no-tui", "--restart", "--no-api-key", "--port", str(port), "--feed", feed,
               "--data-dir", str(grid_dir / "data"), "--context", str(profile.get("context", "32k")), "--vision", str(profile.get("vision", "off")), "--tune", "skip", "--model-path", model_path]
        cmd += list(model.get("launch_args") or [])
        if spec["drafter"] == "off":
            cmd += ["--drafting", "off"]
        cmd += list(spec.get("launch_extra") or [])
        res["launch_cmd"] = " ".join(cmd).replace(str(home), "$HOME")
        subprocess.run([strata, "launch", "--stop"], env=env, capture_output=True, timeout=120)

        def stop_server():
            try:
                subprocess.run([strata, "launch", "--stop"], env=env, capture_output=True, timeout=180)
            except Exception as e:  # noqa: BLE001
                say("launch --stop: %s" % e)

        if spec.get("vram_class_gib"):
            try:
                hog = Hog(grid_dir, os.path.dirname(os.path.abspath(__file__)), float(spec["vram_class_gib"]), int((spec.get("env") or {}).get("CUDA_VISIBLE_DEVICES", 0) or 0),
                          int(float(profile.get("cell_budget_s", 7200)) + 600), str(out_dir / (cid + ".hog.txt")))
            except RuntimeError as e:
                steps["launch"] = step(FAIL, "VRAM class %s GiB: %s" % (spec["vram_class_gib"], e))
                raise StopCell()
            res["vram_class_gib"] = spec["vram_class_gib"]
            res["hog_held_mib"] = hog.held_mib
        guard = Guard(float(profile.get("guard_free_gib", 0)) or None, profile.get("guard_swap_growth_gib"), stop_server, spec.get("ram_cap_gib"), home)
        guard.start()
        t_launch = now()
        say("%s: launching" % cid)
        rc, secs = run_cmd(cmd, env, str(out_dir / (cid + ".launch.txt")), float(profile.get("ready_timeout_s", 2400)))
        elog = newest_engine_log(home, t_launch)
        launch_txt = tail(out_dir / (cid + ".launch.txt"), 30)
        if rc != 0:
            cause = find_cause(launch_txt, tail(elog, 80) if elog else "") or ("timed out" if rc is None else "")
            if not cause:                                   # no known pattern: the Terminal's own last "strata:" line is the best reason there is
                last = [ln for ln in launch_txt.splitlines() if ln.startswith("strata")]
                cause = clip(last[-1], 300) if last else ""
            if "out of memory" in cause:
                cause += " (box RAM available before the start: %.0f GiB%s)" % (mi.get("MemAvailable", 0), "; VRAM class %s GiB" % spec["vram_class_gib"] if spec.get("vram_class_gib") else "")
            steps["launch"] = step(FAIL, "strata launch exited %s%s" % (rc, (": " + cause) if cause else ""), seconds=round(secs),
                                   log_tail=clip(launch_txt, 600), engine_log_tail=clip(tail(elog, 8), 600) if elog else "")
            raise StopCell()
        if guard.tripped:
            steps["launch"] = step(FAIL, guard.tripped, seconds=round(secs))
            raise StopCell()
        steps["launch"] = step(PASS, seconds=round(secs), port=port)
        api = Api("http://127.0.0.1:%d" % port)
        res["numbers"] = {} if profile.get("skip_standard") else run_checks(api, profile, steps, deadline)
        if guard.tripped:
            steps["memory_guard"] = step(FAIL, guard.tripped)
        if guard.min_seen is not None:
            res["numbers"]["min_ram_available_gib"] = round(guard.min_seen, 1)
        if guard.peak_rss:
            res["numbers"]["peak_engine_anon_ram_gib"] = round(guard.peak_rss, 1)
        elog = newest_engine_log(home, t_launch)
        if elog:
            res["engine_log"] = clip(find_cause(tail(elog, 400)) or "", 200)
        try:
            res["engine_facts"] = engine_facts(elog, api)
        except Exception as e:  # noqa: BLE001
            res["engine_facts"] = {"error": str(e)}
        if profile.get("usage_suite"):
            import usage_suite
            ur = usage_suite.run_suite("http://127.0.0.1:%d" % port, api.model, None, profile.get("usage_cases"), long=profile.get("usage_long", "30000,60000"),
                                       agent_turns=profile.get("usage_agent_turns", 20), parallel_levels=profile.get("usage_parallel", "1,2,4,8"))
            res["usage"] = ur
            steps["usage"] = step(PASS if ur["verdict"] == PASS else FAIL, None if ur["verdict"] == PASS else "usage cases failed: " + ", ".join(k for k, v in ur["cases"].items() if v["verdict"] == FAIL))
        stop_server()
    except StopCell:
        pass
    except Exception as e:  # noqa: BLE001 (a bug in the cell must not end the whole pass)
        steps.setdefault("runner", step(FAIL, "runner error: %s: %s" % (type(e).__name__, e)))
    finally:
        if guard:
            guard.halt()
        if hog:
            hog.stop()
        try:
            subprocess.run([strata, "launch", "--stop"], env=env, capture_output=True, timeout=180)
        except Exception:  # noqa: BLE001
            pass
        if feed_server:
            feed_server.close()
    # steps after a failed start say so
    if steps.get("launch", {}).get("status") != PASS:
        for name in ("first_chat", "prompt_4k", "prompt_32k", "agent", "speed"):
            steps.setdefault(name, step(SKIP, "the server did not start"))
    res["verdict"] = FAIL if any(s.get("status") == FAIL for s in steps.values()) else PASS
    return finish()


class StopCell(Exception):
    pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", help="comma list of cell ids")
    a = ap.parse_args(argv)
    plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))
    out = Path(os.path.expanduser(a.out))
    only = set(a.only.split(",")) if a.only else None
    worst = 0
    for spec in plan["cells"]:
        if only and spec["cell"] not in only:
            continue
        say("== cell %s (%s)" % (spec["cell"], spec["label"]))
        r = run_cell(spec, out)
        say("== %s -> %s (%s s)%s" % (spec["cell"], r.get("verdict"), r.get("seconds"),
                                      "" if r.get("verdict") != FAIL else ": " + "; ".join(
                                          "%s: %s" % (k, v.get("reason")) for k, v in r["steps"].items() if v.get("status") == FAIL)[:300]))
        worst = max(worst, 1 if r.get("verdict") == FAIL else 0)
    say("PLAN-DONE")
    return 0                                   # a failed cell is a result, not a failed job


if __name__ == "__main__":
    sys.exit(main())
