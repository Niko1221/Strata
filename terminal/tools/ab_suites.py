"""Opt-in test suites for ab_engine: requests, checks and the frozen corpus.

The corpus is read from git tag v0.2.12, so every run and every version sees the same bytes.
"""
import random
import shutil
import subprocess
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
    r = _git(["grep", "-n", "-I", "-E", pattern, rev], repo)
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
    md = gguf_reader.GGUFFile(model).metadata
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
        used = (resp.get("usage") or {}).get("prompt_tokens") or (resp.get("timings") or {}).get("prompt_n")
        if not used or abs(used - step) > 0.1 * step:
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
        text = _with_needles(corpus[:int(step * CHARS_PER_TOKEN)], rooms, codes)
        content = f"Document {step}\n{text}\n\n{LADDER_QUESTION}"
        body = {"messages": [{"role": "user", "content": content}], "max_tokens": 256, "temperature": 0,
                "seed": SEED, "chat_template_kwargs": {"enable_thinking": False}}
        reqs.append(Req("ladder", f"ladder.{step // 1024}k", body, ladder_check(codes, step)))
    return reqs
