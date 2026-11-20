"""Opt-in test suites for ab_engine: requests, checks and the frozen corpus.

The corpus is read from git tag v0.2.12, so every run and every version sees the same bytes.
"""
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

SEED = 7
CORPUS_REV = "v0.2.12"
ANSWER_ROOM = 1024  # tokens left for question and answer
REPO = Path(__file__).resolve().parents[2]


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
