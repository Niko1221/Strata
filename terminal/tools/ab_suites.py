"""Opt-in test suites for ab_engine: requests, checks and the frozen corpus.

The corpus is read from git tag v0.2.12, so every run and every version sees the same bytes.
"""
import ast
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import gguf_reader

SEED = 7
CORPUS_REV = "v0.2.12"
ANSWER_ROOM = 1024  # tokens left for question and answer
REPO = Path(__file__).resolve().parents[2]
CHARS_PER_TOKEN = 3.3  # pinned; Task 9 measures it on the real run
LADDER_QUESTION = "List every room mentioned in the document with its access code, one line per room, as 'room: code'."
WORDS = ("COBALT", "AMBER", "INDIGO", "SCARLET", "JADE", "IVORY", "COPPER", "VIOLET",
         "SILVER", "CRIMSON", "TEAL", "OCHRE", "MAROON", "AZURE", "SAFFRON", "EBONY")


@dataclass(frozen=True)
class Req:
    suite: str
    tag: str
    body: dict  # full JSON for POST /v1/chat/completions
    check: Callable[[dict], tuple[bool, str]]  # takes the server's response


def looping(text: str, n: int = 8, limit: int = 4) -> bool:
    """True when any run of n words occurs more than limit times."""
    w = text.split()
    return any(c > limit for c in Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1)).values())


def base_check(resp: dict) -> tuple[bool, str]:
    """Empty or looping answers fail. finish_reason "length" is not an error."""
    text = resp["choices"][0]["message"].get("content") or ""
    if not text.strip():
        return False, "leer"
    if looping(text):
        return False, "Schleife"
    return True, ""


def _git(args: list[str], repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, encoding="utf-8", errors="replace")


def need_corpus(repo: Path = REPO, rev: str = CORPUS_REV) -> None:
    """Exit with a clear message when git or the corpus tag is missing."""
    try:
        r = _git(["cat-file", "-e", f"{rev}^{{commit}}"], repo)
    except FileNotFoundError:
        raise SystemExit("error: git is needed for the suites (the corpus is read from git tag " + rev + ")")
    if r.returncode:
        raise SystemExit(f"error: git tag {rev} not found in {repo}; the suites read their corpus from it "
                         f"(try: git fetch --tags)")


def git_show(path: str, repo: Path = REPO, rev: str = CORPUS_REV) -> str:
    r = _git(["show", f"{rev}:{path}"], repo)
    if r.returncode:
        raise RuntimeError(r.stderr.strip())
    return r.stdout


def git_grep(pattern: str, repo: Path = REPO, rev: str = CORPUS_REV) -> str:
    """Like agent_turns_bench.tool_result("grep"), but at the frozen tag."""
    r = _git(["grep", "-n", "-I", "-E", "-e", pattern, rev], repo)
    if r.returncode > 1:  # 1 means no match
        raise RuntimeError(r.stderr.strip())
    lines = [ln.removeprefix(f"{rev}:") for ln in r.stdout.splitlines()]
    return "\n".join(lines[:200]) + (f"\n... ({len(lines) - 200} more lines)" if len(lines) > 200 else "")


def model_context(model: Path) -> int:
    """Context length from the GGUF metadata of a model file or folder."""
    model = Path(model)
    if model.is_dir():
        files = sorted(model.glob("*-00001-of-*.gguf")) or sorted(model.glob("*.gguf"))
        if not files:
            raise SystemExit(f"error: no .gguf file in {model}; pass --context N")
        model = files[0]
    elif not model.is_file():
        raise SystemExit(f"error: model path {model} does not exist; pass --context N")
    try:
        md = gguf_reader.GGUFFile(model).metadata
    except (ValueError, OSError) as e:
        raise SystemExit(f"error: cannot read {model}: {e}; pass --context N")
    n = md.get(f"{md.get('general.architecture')}.context_length")
    if not isinstance(n, int):
        raise SystemExit(f"error: no context_length in {model}; pass --context N")
    return n


def ladder_steps(n_ctx: int) -> list[int]:
    """8192 doubled while below the top (n_ctx minus ANSWER_ROOM), then the top itself."""
    top = n_ctx - ANSWER_ROOM
    if top < 8192:
        return []
    steps, s = [], 8192
    while s < top:
        steps.append(s)
        s *= 2
    return steps + [top]


def _feed(proc: subprocess.Popen, text: str) -> None:
    try:
        proc.stdin.write(text.encode())
    except OSError:  # the reader stopped early and killed git
        pass
    finally:
        try:
            proc.stdin.close()
        except OSError:
            pass


@lru_cache(maxsize=4)
def corpus_text(n_chars: int, repo: Path = REPO, rev: str = CORPUS_REV) -> str:
    """The first n_chars of the repo files at rev (docs first), repeated when shorter.

    One `git cat-file --batch` reads the blobs and stops once there are enough characters.
    """
    r = _git(["ls-tree", "-r", "--name-only", rev, "--", "docs", "src", "include", "serve"], repo)
    if r.returncode:
        raise RuntimeError(r.stderr.strip())
    paths = sorted((p for p in r.stdout.splitlines() if p.endswith((".md", ".cpp", ".cu", ".hpp", ".h", ".py"))),
                   key=lambda p: (not p.startswith("docs/"), p))
    parts, total = [], 0
    proc = subprocess.Popen(["git", "-C", str(repo), "cat-file", "--batch"], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        # a thread: git answers while it reads, so one big write could block both sides
        threading.Thread(target=_feed, args=(proc, "".join(f"{rev}:{p}\n" for p in paths))).start()
        for p in paths:
            head = proc.stdout.readline().split()
            if len(head) != 3:  # "<name> missing"
                continue
            data = proc.stdout.read(int(head[2]))
            proc.stdout.read(1)  # newline after the blob
            parts.append(f"\n\n# file: {p}\n\n" + data.decode("utf-8", "replace"))
            total += len(parts[-1])
            if total >= n_chars:
                break
    finally:
        proc.kill()
        proc.wait()
        proc.stdout.close()
    base = "".join(parts)
    return (base * (n_chars // len(base) + 1))[:n_chars]


def _with_needles(text: str, rooms: list[int], codes: list[str]) -> str:
    """Insert one sentence per room at the line break nearest 10, 35, 60 and 85 % of text."""
    cuts = []
    for frac in (0.10, 0.35, 0.60, 0.85):
        i = int(len(text) * frac)
        before, after = text.rfind("\n", 0, i), text.find("\n", i)
        cuts.append(after if after >= 0 and (before < 0 or after - i <= i - before) else max(before, 0))
    for at, room, code in sorted(zip(cuts, rooms, codes), reverse=True):  # from the end: earlier offsets stay valid
        text = text[:at] + f"\nThe access code for room {room} is {code}." + text[at:]
    return text


def ladder_check(codes: list[str], step: int) -> Callable[[dict], tuple[bool, str]]:
    def check(resp: dict) -> tuple[bool, str]:
        got = (resp["choices"][0]["message"].get("content") or "").lower()
        n = sum(c.lower() in got for c in codes)
        # timings.prompt_n is what was read (prompt minus cache hit); usage.prompt_tokens is the full prompt
        used = (resp.get("timings") or {}).get("prompt_n")
        full = (resp.get("usage") or {}).get("prompt_tokens")
        if used is None:
            used = full
        if used is None or abs(used - step) > 0.1 * step or (full is not None and used < 0.9 * full):
            return False, "Cache-Treffer oder falsche Länge"
        return n == len(codes), f"{n}/{len(codes)} Codes"
    return check


def ladder_reqs(n_ctx: int) -> list[Req]:
    steps = ladder_steps(n_ctx)
    if not steps:
        return []
    corpus = corpus_text(int(steps[-1] * CHARS_PER_TOKEN))  # every step is a prefix of this
    reqs = []
    for step in steps:
        rng = random.Random(SEED * 1_000_003 + step)
        rooms = rng.sample(range(1000, 10000), 4)
        codes = [f"{w}-{rng.randint(10, 99)}" for w in rng.sample(WORDS, 4)]
        margin = 0.97 if step == steps[-1] else 1  # the top step must not overshoot the context (with the needles and question)
        text = _with_needles(corpus[:int(step * margin * CHARS_PER_TOKEN)], rooms, codes)
        content = f"Document {step}\n{text}\n\n{LADDER_QUESTION}"
        body = {"messages": [{"role": "user", "content": content}], "max_tokens": 256, "temperature": 0,
                "seed": SEED, "chat_template_kwargs": {"enable_thinking": False}}
        reqs.append(Req("ladder", f"ladder.{step // 1024}k", body, ladder_check(codes, step)))
    return reqs


THINK_END = "End your reply with a line 'ANTWORT: <answer>'."
# (prompt, integer solution); every answer is above 100 and not in the prompt
MATH = [
    ("How many integers from 1 to 2000 are divisible by 6 or by 15, but not by 10?", 334),
    ("Let a(1) = 3 and a(n+1) = 2*a(n) + n for n >= 1. What is a(10)?", 2549),
    ("Form every three-digit number whose digits are three different digits from 1, 2, 3, 4, 5, with an odd first "
     "digit and an even last digit. What is the sum of all these numbers?", 5994),
    ("What is the remainder when 7^2026 is divided by 1000?", 649),
]
READ_SNIPPET = """def f(xs):
    out = []
    for i, x in enumerate(xs):
        if x % 2:
            out.append(x * i)
        else:
            out.insert(0, x + i)
    return out

print("-".join(map(str, f([3, 4, 5, 6, 7]))))"""
BUG_SNIPPET = """1  def bsearch(a, t):
2      lo, hi = 0, len(a) - 1
3      while lo < hi:
4          mid = (lo + hi) // 2
5          if a[mid] == t:
6              return mid
7          if a[mid] < t:
8              lo = mid + 1
9          else:
10             hi = mid - 1
11     return -1"""
# (prompt, text solution)
READ = [
    (f"```python\n{READ_SNIPPET}\n```\nWhat exactly does this program print? Give the one output line.", "9-5-0-10-28"),
    (f"This binary search of a sorted list returns the index of t, or -1. Exactly one line is wrong. Line numbers "
     f"are not part of the code.\n```\n{BUG_SNIPPET}\n```\nWhich line number is wrong? Give the number only.", "3"),
]
# (function name, prompt, hidden tests, reference solution)
CODE = [
    ("longest_run",
     "Write longest_run(s): the length of the longest run of one repeated character in the string s "
     "(for example 'aabbbc' gives 3). The empty string gives 0.",
     "from solution import longest_run\nassert longest_run('aabbbc') == 3\nassert longest_run('abc') == 1\n"
     "assert longest_run('zzzz') == 4\nassert longest_run('abbbaaaa') == 4\nassert longest_run('') == 0",
     "def longest_run(s):\n    best = cur = 0\n    for i, c in enumerate(s):\n"
     "        cur = cur + 1 if i and s[i - 1] == c else 1\n        best = max(best, cur)\n    return best"),
    ("merge_intervals",
     "Write merge_intervals(iv): iv is a list of (start, end) integer pairs in any order. Merge intervals that "
     "overlap or touch (3 touches 3) and return the merged intervals as a sorted list of tuples. "
     "An empty list gives [].",
     "from solution import merge_intervals\nassert merge_intervals([(1, 3), (2, 6), (8, 10)]) == [(1, 6), (8, 10)]\n"
     "assert merge_intervals([(5, 7), (1, 2), (2, 4)]) == [(1, 4), (5, 7)]\n"
     "assert merge_intervals([(1, 10), (2, 3)]) == [(1, 10)]\nassert merge_intervals([]) == []",
     "def merge_intervals(iv):\n    out = []\n    for a, b in sorted(iv):\n"
     "        if out and a <= out[-1][1]:\n            out[-1] = (out[-1][0], max(out[-1][1], b))\n"
     "        else:\n            out.append((a, b))\n    return out"),
]


def answer_of(resp: dict) -> str | None:
    """The last 'ANTWORT:' value in the answer text (never in the reasoning), or None."""
    found = re.findall(r"ANTWORT:[ \t]*((?:(?!ANTWORT:).)*)", resp["choices"][0]["message"].get("content") or "")
    return found[-1].strip() if found else None


def _norm(text: str) -> str:
    """Lower case, spaces collapsed, quotes, backticks, asterisks and a final period removed."""
    return " ".join(text.lower().split()).strip("\"'`* ").rstrip(".").strip("\"'`* ")


# Runs every assert of the hidden tests alone, so the report can say how many passed.
HARNESS = """import ast, sys
sys.path.insert(0, '.')  # -I leaves the script folder off the path
ns, ok = {}, 0
tree = ast.parse(%r)
try:
    for n in tree.body:
        try:
            exec(compile(ast.Module([n], []), 'tests', 'exec'), ns)
            ok += isinstance(n, ast.Assert)
        except Exception:
            if not isinstance(n, ast.Assert):
                raise
finally:
    print('PASSED', ok)
"""


def run_code(src: str, tests: str, timeout: float = 20) -> tuple[bool, str]:
    """Run tests against src in a fresh temp dir. This is not isolation: the code has the user's rights."""
    k = sum(isinstance(n, ast.Assert) for n in ast.parse(tests).body)
    d = Path(tempfile.mkdtemp())
    try:
        (d / "solution.py").write_text(src, encoding="utf-8")
        (d / "test_solution.py").write_text(HARNESS % tests, encoding="utf-8")
        env = {key: os.environ[key] for key in ("PATH", "SYSTEMROOT") if key in os.environ}
        try:
            r = subprocess.run([sys.executable, "-I", "test_solution.py"], cwd=d, env=env, timeout=timeout,
                               capture_output=True, encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:  # subprocess.run has killed the child
            return False, "Zeitlimit"
        m = re.findall(r"PASSED (\d+)", r.stdout)
        i = int(m[-1]) if m else 0
        return i == k, f"Tests: {i} von {k}"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _think_check(kind: str, want) -> Callable[[dict], tuple[bool, str]]:
    def check(resp: dict) -> tuple[bool, str]:
        choice = resp["choices"][0]
        content = choice["message"].get("content") or ""
        cut = choice.get("finish_reason") == "length"
        if not content.strip():
            return False, "Token-Grenze erreicht" if cut else "keine Antwort"
        if kind == "code":
            blocks = re.findall(r"```python[ \t]*\n(.*?)```", content, re.S)
            if not blocks:
                return False, "Token-Grenze erreicht" if cut else "kein Code"
            return run_code(blocks[-1], want)
        got = answer_of(resp)
        if got is None:
            return False, "Token-Grenze erreicht" if cut else "keine Antwort"
        if kind == "math":  # the last integer of the value: "x = 334", "2,549", "$\boxed{334}$"
            nums = re.findall(r"-?\d[\d,]*", got)
            ok = bool(nums) and nums[-1].replace(",", "") == str(want)
        elif want.isdigit():  # "Line 3" is right for "3"
            ok = re.findall(r"\d+", got) == [want]
        else:
            ok = _norm(re.sub(r"\\boxed\{(.*)\}", r"\1", got).replace("$", "")) == _norm(want)
        return ok, "" if ok else "falsche Antwort"
    return check


def think_reqs(think_tokens: int, run_code: bool) -> list[Req]:
    """Six thinking tasks, plus two code-writing tasks that run model code only with run_code."""
    tasks = [("math", f"math{i}", f"{p}\n{THINK_END}", w) for i, (p, w) in enumerate(MATH, 1)]
    tasks += [("read", f"read{i}", f"{p}\n{THINK_END}", w) for i, (p, w) in enumerate(READ, 1)]
    if run_code:
        tasks += [("code", f"code{i}", f"{p}\nReply with one ```python block that defines {name}.", tests)
                  for i, (name, p, tests, _) in enumerate(CODE, 1)]
    return [Req("think", f"think.{tag}",
                {"messages": [{"role": "user", "content": text}], "max_tokens": think_tokens, "temperature": 0,
                 "seed": SEED, "chat_template_kwargs": {"enable_thinking": True}, "reasoning_effort": "high",
                 "reasoning_budget_tokens": think_tokens * 3 // 4},
                _think_check(kind, want))
            for kind, tag, text, want in tasks]


FR_WORDS = {"le", "la", "les", "est", "pour", "dans", "une", "des"}


def agent_check(lang: str) -> Callable[[dict], tuple[bool, str]]:
    """ok: a valid tool call, or non-empty non-looping text (French for lang "fr")."""
    from agent_turns_bench import TOOLS  # lazy: it imports ab_engine, which will import this module

    required = {t["function"]["name"]: t["function"]["parameters"]["required"] for t in TOOLS}

    def check(resp: dict) -> tuple[bool, str]:
        msg = resp["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        if calls:
            fn = calls[0].get("function") or {}
            try:
                args = json.loads(fn.get("arguments"))
            except (TypeError, ValueError):
                args = None
            if fn.get("name") in required and isinstance(args, dict) and all(k in args for k in required[fn["name"]]):
                return True, f"Tool-Call {fn['name']}"
            return False, "Tool-Call ungültig"
        ok, why = base_check(resp)
        if not ok:
            return False, why
        if lang == "fr" and sum(w in FR_WORDS for w in re.findall(r"\w+", msg["content"].lower())) < 3:
            return False, "nicht Französisch"
        return True, "Text"

    return check


def agent_reqs(n_ctx: int, chars_per_token: float = CHARS_PER_TOKEN) -> tuple[list[Req], list[str]]:
    """One request per point right after a tool result, in session order, built as agent_turns_bench.turns()
    does but with tool results from the frozen tag. Also returns the tags left out as too long for n_ctx."""
    from agent_turns_bench import SESSIONS, SYSTEM, TOOLS

    reqs, left = [], []
    for name, task, steps in SESSIONS:
        check = agent_check("fr" if name.endswith("-fr") else "en")
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
        for i, (tool, arg, before) in enumerate(steps):
            cid = f"call_{name}_{i}"
            if tool == "read_file":
                lines = git_show(arg).splitlines()
                result = "\n".join(lines[:2000]) + (f"\n... (truncated: lines 1-2000 of {len(lines)})" if len(lines) > 2000 else "")
            else:
                result = git_grep(arg)
            args = {"path": arg} if tool == "read_file" else {"pattern": arg}
            msgs.append({"role": "assistant", "content": before,
                         "tool_calls": [{"id": cid, "type": "function",
                                         "function": {"name": tool, "arguments": json.dumps(args)}}]})
            msgs.append({"role": "tool", "tool_call_id": cid, "content": result})
            tag = f"agent.{name}.{i}"
            if len(json.dumps(msgs)) / chars_per_token > n_ctx - ANSWER_ROOM:
                left.append(tag)
                continue
            body = {"messages": [dict(m) for m in msgs], "tools": TOOLS, "max_tokens": 256, "temperature": 0,
                    "seed": SEED, "chat_template_kwargs": {"enable_thinking": False}}
            reqs.append(Req("agent", tag, body, check))
    return reqs, left
