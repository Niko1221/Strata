#!/usr/bin/env python3
"""
coding_eval.py -- a small, dependency-light coding-capability harness for
comparing local LLM endpoints served over OpenAI-compatible HTTP APIs.

What it does
------------
1.  Loads HumanEval (164 problems) from a local cache (``HumanEval.jsonl.gz``),
    downloading it once from the official URL if the cache is missing.  If the
    download fails it falls back to a bundled set of self-written problems
    (``BUNDLED``, 22 problems, embedded below) and records that fact in the
    output JSON (``dataset`` / ``dataset_note``).
2.  For each problem it POSTs the HumanEval prompt to
    ``{base_url}/chat/completions`` and appends the completion to the prompt
    using the HumanEval convention:

        program = prompt + completion + "\\n" + test + "\\n" + "check(<entry_point>)"

3.  Extraction of the completion is deliberately forgiving: markdown code
    fences are stripped, leading prose is dropped, a re-defined function is
    cut back to its ``def``, and a trailing ``if __name__ == "__main__":``
    block is removed.  Only ``message.content`` is used --
    ``message.reasoning_content`` (and any other thinking field) is ignored.
4.  Every candidate program is compiled and executed in a **subprocess** with a
    hard timeout (``--exec-timeout``, default 10 s), so a hanging, crashing or
    infinite-looping solution cannot stall the run.  stdout/stderr are captured
    and the tail of stderr is stored in the JSON for diagnosis.
5.  Problems run sequentially by default; ``--parallel N`` runs them
    concurrently.

Output
------
A JSON file (``--out``) shaped like::

    {
      "model": "...", "base_url": "...", "dataset": "humaneval"|"bundled",
      "n": 164, "passed": 120, "pass_at_1": 0.7317,
      "total_completion_tokens": 12345, "wall_s": 123.4,
      "started_utc": "...", "temperature": 0.0, "max_tokens": 512,
      "reasoning_effort": null,
      "summary": { ... same summary fields, mirrored ... },
      "problems": [
        {"task_id": "HumanEval/0", "passed": true, "error": null,
         "duration_s": 1.23, "prompt_tokens": 120, "completion_tokens": 45,
         "completion": "...truncated to 2000 chars..."},
        ...
      ]
    }

The summary fields are present both at the top level and mirrored under
``"summary"`` so either consumer style works.

Safety
------
This harness never touches a GPU or a model process; it is pure HTTP + stdlib.
Port 8080 on the loopback interface is refused by default because that port is
reserved for GPU benchmarking on this machine.  Override only if you are sure:
set the environment variable ``CODING_EVAL_ALLOW_8080=1``.

Usage
-----
  python coding_eval.py --base-url http://127.0.0.1:8080/v1 --model qwen \\
      --out run.json --limit 20
  python coding_eval.py --mock --base-url http://127.0.0.1:8199/v1 --model stub \\
      --out mock-run.json --limit 5          # offline self-test vs stub_server.py
  python coding_eval.py --selftest           # validate the bundled problems
  python coding_eval.py --fetch-dataset      # just (re)download the dataset
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

try:  # requests is installed in the project venv; stdlib is the fallback.
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None

HERE = Path(__file__).resolve().parent
DATASET_URL = (
    "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
)
DATASET_PATH = HERE / "HumanEval.jsonl.gz"
COMPLETION_TRUNC = 2000
DEFAULT_EXEC_TIMEOUT = 10.0

VERSION = "1.0"


# --------------------------------------------------------------------------
# bundled fallback problem set (self-written; used only if HumanEval cannot be
# downloaded).  Spec tuples are:
#     (entry_point, signature_args, docstring, body_lines, checks)
# ``checks`` is either a list of assert lines (wrapped in ``def check``) or a
# verbatim test-code string.
# --------------------------------------------------------------------------
BUNDLED_SPECS = [
    (
        "add_two_numbers",
        "a, b",
        "Return the sum of a and b.",
        ["return a + b"],
        [
            "assert candidate(1, 2) == 3",
            "assert candidate(-1, 1) == 0",
            "assert candidate(0, 0) == 0",
        ],
    ),
    (
        "reverse_words",
        "s",
        "Return the words of s in reverse order, joined by single spaces.",
        ['return " ".join(s.split()[::-1])'],
        [
            'assert candidate("the sky is blue") == "blue is sky the"',
            'assert candidate("") == ""',
            'assert candidate("  hello   world  ") == "world hello"',
        ],
    ),
    (
        "first_duplicate",
        "nums",
        "Return the first value that occurs twice in nums (in the order the\n"
        "    second occurrence is met), or None if every value is unique.",
        [
            "seen = set()",
            "for x in nums:",
            "    if x in seen:",
            "        return x",
            "    seen.add(x)",
            "return None",
        ],
        [
            "assert candidate([2, 1, 3, 2, 1]) == 2",
            "assert candidate([1, 2, 3]) is None",
            "assert candidate([]) is None",
            "assert candidate([5, 5]) == 5",
        ],
    ),
    (
        "count_vowels",
        "s",
        "Return how many vowels (a, e, i, o, u) s contains, case-insensitive.",
        ["return sum(1 for ch in s.lower() if ch in \"aeiou\")"],
        [
            'assert candidate("Hello World") == 3',
            'assert candidate("AEIOU") == 5',
            'assert candidate("xyz") == 0',
            'assert candidate("") == 0',
        ],
    ),
    (
        "clamp",
        "value, low, high",
        "Return value limited to the inclusive range [low, high].",
        ["return max(low, min(high, value))"],
        [
            "assert candidate(5, 1, 10) == 5",
            "assert candidate(-3, 0, 10) == 0",
            "assert candidate(42, 0, 10) == 10",
            "assert candidate(1, 5, 5) == 5",
        ],
    ),
    (
        "is_palindrome",
        "s",
        "Return True if s reads the same forwards and backwards, ignoring\n"
        "    case and any character that is not alphanumeric.",
        [
            "cleaned = [ch.lower() for ch in s if ch.isalnum()]",
            "return cleaned == cleaned[::-1]",
        ],
        [
            'assert candidate("A man, a plan, a canal: Panama") is True',
            'assert candidate("race a car") is False',
            'assert candidate("") is True',
            'assert candidate("Ab1bA") is True',
        ],
    ),
    (
        "factorial",
        "n",
        "Return n! for n >= 0, with factorial(0) == 1.",
        ["result = 1", "for i in range(2, n + 1):", "    result *= i", "return result"],
        [
            "assert candidate(0) == 1",
            "assert candidate(1) == 1",
            "assert candidate(5) == 120",
            "assert candidate(10) == 3628800",
        ],
    ),
    (
        "sum_even",
        "numbers",
        "Return the sum of the even integers in numbers.",
        ["return sum(x for x in numbers if x % 2 == 0)"],
        [
            "assert candidate([1, 2, 3, 4]) == 6",
            "assert candidate([]) == 0",
            "assert candidate([1, 3]) == 0",
            "assert candidate([-2, 2]) == 0",
        ],
    ),
    (
        "min_max",
        "nums",
        "Return the pair (smallest, largest) of a non-empty list.",
        ["return (min(nums), max(nums))"],
        [
            "assert candidate([3, 1, 2]) == (1, 3)",
            "assert candidate([5]) == (5, 5)",
            "assert candidate([-1, -7]) == (-7, -1)",
        ],
    ),
    (
        "unique_sorted",
        "nums",
        "Return the distinct values of nums in ascending order.",
        ["return sorted(set(nums))"],
        [
            "assert candidate([3, 1, 3, 2]) == [1, 2, 3]",
            "assert candidate([]) == []",
            "assert candidate([1, 1, 1]) == [1]",
        ],
    ),
    (
        "word_lengths",
        "s",
        "Return a dict mapping each whitespace-separated word of s to its length.",
        ["return {w: len(w) for w in s.split()}"],
        (
            "def test_word_lengths(candidate):\n"
            '    assert candidate("a bb ccc") == {"a": 1, "bb": 2, "ccc": 3}\n'
            "\n"
            "def check(candidate):\n"
            "    test_word_lengths(candidate)\n"
            '    assert candidate("") == {}\n'
            '    assert candidate("one") == {"one": 3}\n'
        ),
    ),
    (
        "celsius_to_fahrenheit",
        "c",
        "Convert a Celsius temperature to Fahrenheit.",
        ["return c * 9 / 5 + 32"],
        [
            "assert candidate(0) == 32.0",
            "assert candidate(100) == 212.0",
            "assert candidate(-40) == -40.0",
        ],
    ),
    (
        "gcd",
        "a, b",
        "Return the greatest common divisor of the non-negative a and b.",
        ["while b:", "    a, b = b, a % b", "return a"],
        [
            "assert candidate(12, 18) == 6",
            "assert candidate(17, 5) == 1",
            "assert candidate(0, 5) == 5",
            "assert candidate(0, 0) == 0",
        ],
    ),
    (
        "flatten_one",
        "nested",
        "Flatten a list of lists by exactly one level.",
        ["return [item for sub in nested for item in sub]"],
        [
            "assert candidate([[1, 2], [3], []]) == [1, 2, 3]",
            "assert candidate([]) == []",
            "assert candidate([[], []]) == []",
        ],
    ),
    (
        "truncate_string",
        "s, n",
        "Return s shortened to at most n characters.",
        ["return s[:n]"],
        [
            'assert candidate("hello", 3) == "hel"',
            'assert candidate("hi", 5) == "hi"',
            'assert candidate("", 3) == ""',
        ],
    ),
    (
        "count_char",
        "s, ch",
        "Return how many times the character ch occurs in s.",
        ["return s.count(ch)"],
        [
            'assert candidate("banana", "a") == 3',
            'assert candidate("banana", "z") == 0',
            'assert candidate("", "a") == 0',
        ],
    ),
    (
        "average",
        "nums",
        "Return the arithmetic mean of nums as a float, or 0.0 if nums is empty.",
        ["if not nums:", "    return 0.0", "return sum(nums) / len(nums)"],
        [
            "assert candidate([1, 2, 3]) == 2.0",
            "assert candidate([2, 2]) == 2.0",
            "assert candidate([]) == 0.0",
        ],
    ),
    (
        "remove_duplicates_keep_order",
        "nums",
        "Remove duplicates from nums while preserving first-seen order.",
        [
            "seen = set()",
            "result = []",
            "for x in nums:",
            "    if x not in seen:",
            "        seen.add(x)",
            "        result.append(x)",
            "return result",
        ],
        [
            "assert candidate([1, 2, 1, 3, 2]) == [1, 2, 3]",
            "assert candidate([]) == []",
            "assert candidate([7, 7, 7]) == [7]",
        ],
    ),
    (
        "starts_with_vowel",
        "word",
        "Return True if word is non-empty and starts with a vowel.",
        ['return bool(word) and word[0].lower() in "aeiou"'],
        [
            'assert candidate("apple") is True',
            'assert candidate("banana") is False',
            'assert candidate("") is False',
            'assert candidate("Umbrella") is True',
        ],
    ),
    (
        "digits_sum",
        "n",
        "Return the sum of the decimal digits of the absolute value of n.",
        ["return sum(int(d) for d in str(abs(n)))"],
        [
            "assert candidate(123) == 6",
            "assert candidate(-450) == 9",
            "assert candidate(0) == 0",
        ],
    ),
    (
        "rotate_left",
        "nums, k",
        "Rotate nums left by k positions (k may exceed the length).",
        [
            "if not nums:",
            "    return []",
            "k = k % len(nums)",
            "return nums[k:] + nums[:k]",
        ],
        [
            "assert candidate([1, 2, 3, 4, 5], 2) == [3, 4, 5, 1, 2]",
            "assert candidate([1, 2], 0) == [1, 2]",
            "assert candidate([], 3) == []",
            "assert candidate([1, 2, 3], 4) == [2, 3, 1]",
        ],
    ),
    (
        "is_prime",
        "n",
        "Return True if n is a prime number, else False.",
        [
            "if n < 2:",
            "    return False",
            "for i in range(2, int(n ** 0.5) + 1):",
            "    if n % i == 0:",
            "        return False",
            "return True",
        ],
        [
            "assert candidate(2) is True",
            "assert candidate(1) is False",
            "assert candidate(0) is False",
            "assert candidate(17) is True",
            "assert candidate(25) is False",
            "assert candidate(-7) is False",
        ],
    ),
]


def _spec_to_problem(index: int, spec) -> dict:
    entry_point, sig, doc, body_lines, checks = spec
    prompt = 'def %s(%s):\n    """%s\n    """\n' % (entry_point, sig, doc)
    solution = "".join(
        ("" if line == "" else "    " + line) + "\n" for line in body_lines
    )
    if isinstance(checks, str):
        test = checks
    else:
        test = "def check(candidate):\n" + "".join("    %s\n" % c for c in checks)
    return {
        "task_id": "Bundled/%d" % index,
        "entry_point": entry_point,
        "prompt": prompt,
        "canonical_solution": solution,
        "test": test,
    }


BUNDLED = [_spec_to_problem(i, s) for i, s in enumerate(BUNDLED_SPECS)]


# --------------------------------------------------------------------------
# dataset handling
# --------------------------------------------------------------------------
class DatasetError(Exception):
    pass


def download_dataset(path: Path = DATASET_PATH, verbose: bool = True) -> Path:
    """Download HumanEval.jsonl.gz once into ``path``.  Raises DatasetError."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    if verbose:
        print("[dataset] downloading %s" % DATASET_URL, flush=True)
    try:
        if requests is not None:
            r = requests.get(DATASET_URL, timeout=60)
            if r.status_code != 200:
                raise DatasetError("HTTP %s from %s" % (r.status_code, DATASET_URL))
            blob = r.content
        else:
            with urllib.request.urlopen(DATASET_URL, timeout=60) as resp:
                blob = resp.read()
    except DatasetError:
        raise
    except Exception as exc:  # network, DNS, TLS, proxy ...
        raise DatasetError("%s: %s" % (type(exc).__name__, exc)) from exc

    if not blob:
        raise DatasetError("empty response body")
    try:
        text = gzip.decompress(blob).decode("utf-8")
    except Exception as exc:
        raise DatasetError("not a valid gzip payload: %s" % exc) from exc
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 100:
        raise DatasetError("expected >=100 problems, got %d" % len(lines))
    tmp.write_bytes(blob)
    os.replace(tmp, path)
    if verbose:
        print(
            "[dataset] cached %d problems -> %s (%d bytes)"
            % (len(lines), path, path.stat().st_size),
            flush=True,
        )
    return path


def load_humaneval(path: Path = DATASET_PATH) -> list:
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out.append(
                {
                    "task_id": rec["task_id"],
                    "entry_point": rec["entry_point"],
                    "prompt": rec["prompt"],
                    "canonical_solution": rec.get("canonical_solution", ""),
                    "test": rec["test"],
                }
            )

    def key(p):
        m = re.search(r"(\d+)$", p["task_id"])
        return (0, int(m.group(1))) if m else (1, p["task_id"])

    out.sort(key=key)
    if len(out) != 164:  # not fatal -- report it, keep going
        print("[dataset] warning: expected 164 problems, found %d" % len(out))
    return out


def select_dataset(args) -> tuple:
    """Return (problems, dataset_name, note)."""
    if args.dataset == "bundled":
        return list(BUNDLED), "bundled", "bundled problem set requested with --dataset bundled"
    if DATASET_PATH.exists():
        try:
            problems = load_humaneval()
            return problems, "humaneval", "loaded from cache %s" % DATASET_PATH
        except Exception as exc:
            note = "cache %s unreadable (%s)" % (DATASET_PATH, exc)
            if args.dataset == "humaneval":
                try:
                    download_dataset()
                    return load_humaneval(), "humaneval", note + "; re-downloaded"
                except DatasetError as exc2:
                    note += "; re-download failed (%s)" % exc2
            else:
                note += "; using bundled set"
            return list(BUNDLED), "bundled", note
    if args.mock:
        return (
            list(BUNDLED),
            "bundled",
            "no HumanEval cache and --mock is offline; using bundled set",
        )
    try:
        download_dataset()
        return load_humaneval(), "humaneval", "downloaded %s" % DATASET_URL
    except DatasetError as exc:
        return (
            list(BUNDLED),
            "bundled",
            "HumanEval download FAILED (%s); fell back to the bundled set of %d "
            "self-written problems" % (exc, len(BUNDLED)),
        )


# --------------------------------------------------------------------------
# completion extraction (robust, HumanEval convention)
# --------------------------------------------------------------------------
_FENCE_RE = re.compile(r"```[ \t]*([A-Za-z0-9_+\-.]*)[ \t]*\r?\n(.*?)```", re.S)
_CODE_PREFIXES = (
    "def ",
    "class ",
    "import ",
    "from ",
    "async ",
    "@",
    "return",
    "if ",
    "for ",
    "while ",
    "try",
    "except",
    "finally",
    "with ",
    "else",
    "elif ",
    "assert ",
    "raise ",
    "yield ",
    "print(",
    "#",
    '"""',
    "'''",
)


def _strip_main_block(text: str) -> str:
    """Drop a trailing ``if __name__ == "__main__":`` block (and everything after)."""
    out = []
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("if __name__") and "__main__" in stripped:
            break
        out.append(line)
    return "".join(out)


def _drop_leading_prose(text: str) -> str:
    """Drop leading natural-language lines (keep from the first line of code)."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.strip() == "":
            continue
        if line[:1].isspace() or line.lstrip().startswith(_CODE_PREFIXES):
            return "\n".join(lines[i:])
    return text


def _cut_before_def(text: str, entry_point) -> str:
    """If the model re-emitted the function, keep only from its ``def`` onwards."""
    pattern = r"^def\s+%s\s*\(" % re.escape(entry_point) if entry_point else r"^def\s+\w+\s*\("
    match = re.search(pattern, text, re.M)
    if match:
        return text[match.start():]
    return text


def _indent_rescue(text: str) -> str:
    """Last resort: the body was emitted flush left -- indent it."""
    lines = text.split("\n")
    if any(ln[:1].isspace() for ln in lines if ln.strip()):
        return text
    return "\n".join(("    " + ln) if ln.strip() else ln for ln in lines)


def _compiles(program: str) -> bool:
    try:
        compile(program, "<candidate>", "exec")
        return True
    except SyntaxError:
        return False


def extract_completion(completion: str, entry_point: str, prompt: str) -> tuple:
    """Return (code, description) -- the best completion variant to append."""
    text = (completion or "").replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        return "", "empty completion"

    sources = []
    blocks = _FENCE_RE.findall(text)
    if blocks:
        preferred = [b for tag, b in blocks if tag in ("", "python", "py", "python3")]
        pool = preferred or [b for _, b in blocks]
        sources.append(max(pool, key=len))
    sources.append(text)

    variants = []
    for si, src in enumerate(sources):
        base = _strip_main_block(src)
        tag = "fenced" if (si == 0 and blocks) else "raw"
        steps = (
            ("as-is", base),
            ("prose-stripped", _drop_leading_prose(base)),
            ("def-cut", _cut_before_def(base, entry_point)),
            ("generic-def-cut", _cut_before_def(base, None)),
            ("indent-rescue", _indent_rescue(_drop_leading_prose(base))),
        )
        for name, cand in steps:
            if cand.strip() == "":
                continue
            if any(cand == v[0] for v in variants):
                continue
            variants.append((cand, "%s+%s" % (tag, name)))
    if not variants:
        return "", "empty completion after extraction"

    prefix = prompt if prompt.endswith("\n") else prompt + "\n"
    for cand, name in variants:
        if _compiles(prefix + cand):
            return cand, name
    return variants[0][0], variants[0][1] + " (no variant compiled)"


def build_program(problem: dict, code: str) -> str:
    """The HumanEval convention: prompt + completion + test + check(...)."""
    prompt = problem["prompt"]
    if not prompt.endswith("\n"):
        prompt += "\n"
    test = problem["test"].rstrip("\n")
    return "%s%s\n%s\ncheck(%s)\n" % (prompt, code, test, problem["entry_point"])


# --------------------------------------------------------------------------
# sandboxed execution
# --------------------------------------------------------------------------
def run_program(program: str, timeout: float) -> dict:
    """Compile + run ``program`` in a throwaway subprocess.  Never raises."""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONSTARTUP", None)
    started = time.time()
    try:
        with tempfile.TemporaryDirectory(prefix="coding_eval_") as td:
            path = Path(td) / "candidate.py"
            path.write_text(program, encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(path)],
                cwd=td,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        elapsed = time.time() - started
        out, err = proc.stdout or "", proc.stderr or ""
        if proc.returncode == 0:
            return {"passed": True, "error": None, "stdout": out, "stderr": err,
                    "exec_s": elapsed}
        error = _summarize_error(err, proc.returncode)
        return {"passed": False, "error": error, "stdout": out, "stderr": err,
                "exec_s": elapsed}
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        err = exc.stderr or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        if isinstance(err, bytes):
            err = err.decode("utf-8", "replace")
        return {
            "passed": False,
            "error": "timeout after %.1fs" % timeout,
            "stdout": out,
            "stderr": err,
            "exec_s": time.time() - started,
        }
    except Exception as exc:  # pragma: no cover - harness bug guard
        return {
            "passed": False,
            "error": "harness error: %s: %s" % (type(exc).__name__, exc),
            "stdout": "",
            "stderr": "",
            "exec_s": time.time() - started,
        }


def _summarize_error(stderr: str, returncode: int) -> str:
    if stderr:
        for line in reversed(stderr.strip().splitlines()):
            line = line.strip()
            if line and not line.startswith(("File ", "  ", "Traceback")):
                return line[:300]
        tail = stderr.strip().splitlines()[-1].strip()
        if tail:
            return tail[:300]
    return "exit code %s" % returncode


# --------------------------------------------------------------------------
# HTTP chat call
# --------------------------------------------------------------------------
class HttpError(Exception):
    pass


def http_post_json(url: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if requests is not None:
        try:
            resp = requests.post(url, data=body, headers=headers, timeout=timeout)
        except Exception as exc:
            raise HttpError("%s: %s" % (type(exc).__name__, exc)) from exc
        if resp.status_code != 200:
            raise HttpError("HTTP %d: %s" % (resp.status_code, resp.text[:300]))
        try:
            return resp.json()
        except Exception as exc:
            raise HttpError("response is not JSON: %s" % exc) from exc
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise HttpError("HTTP %d: %s" % (exc.code, detail)) from exc
    except Exception as exc:
        raise HttpError("%s: %s" % (type(exc).__name__, exc)) from exc
    try:
        return json.loads(raw)
    except Exception as exc:
        raise HttpError("response is not JSON: %s" % exc) from exc


def chat_completion(args, prompt: str) -> dict:
    url = args.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    if args.reasoning_effort:
        payload["reasoning_effort"] = args.reasoning_effort
    data = http_post_json(url, payload, args.timeout)
    choices = data.get("choices") or []
    if not choices:
        raise HttpError("response has no choices: %s" % json.dumps(data)[:300])
    message = choices[0].get("message") or {}
    # Only message.content is used; reasoning_content / thinking fields are ignored.
    content = message.get("content")
    if content is None:
        content = choices[0].get("text") or ""
    usage = data.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    source = "usage"
    if not isinstance(prompt_tokens, int):
        prompt_tokens = max(1, len(prompt) // 4)
        source = "estimate"
    if not isinstance(completion_tokens, int):
        completion_tokens = max(0, len(content) // 4)
        source = "estimate"
    return {
        "content": content,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "tokens_source": source,
        "finish_reason": choices[0].get("finish_reason"),
    }


# --------------------------------------------------------------------------
# one problem
# --------------------------------------------------------------------------
def evaluate_one(problem: dict, args) -> dict:
    t0 = time.time()
    row = {
        "task_id": problem["task_id"],
        "entry_point": problem["entry_point"],
        "passed": False,
        "error": None,
        "duration_s": 0.0,
        "gen_s": 0.0,
        "exec_s": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "tokens_source": "usage",
        "finish_reason": None,
        "extraction": None,
        "completion": "",
        "stderr_tail": "",
    }
    try:
        gen_t0 = time.time()
        reply = chat_completion(args, problem["prompt"])
        row["gen_s"] = time.time() - gen_t0
        row["prompt_tokens"] = reply["prompt_tokens"]
        row["completion_tokens"] = reply["completion_tokens"]
        row["tokens_source"] = reply["tokens_source"]
        row["finish_reason"] = reply["finish_reason"]

        content = reply["content"] or ""
        row["completion"] = content[:COMPLETION_TRUNC]
        code, how = extract_completion(content, problem["entry_point"], problem["prompt"])
        row["extraction"] = how
        program = build_program(problem, code)
        outcome = run_program(program, args.exec_timeout)
        row["passed"] = bool(outcome["passed"])
        row["error"] = outcome["error"]
        row["exec_s"] = outcome["exec_s"]
        row["stderr_tail"] = (outcome["stderr"] or "")[-500:]
    except HttpError as exc:
        row["error"] = "request failed: %s" % exc
    except Exception as exc:  # pragma: no cover - harness bug guard
        row["error"] = "harness error: %s: %s" % (type(exc).__name__, exc)
    row["duration_s"] = time.time() - t0
    return row


class Printer:
    """Thread-safe incremental table + final summary."""

    def __init__(self, n: int):
        self.lock = threading.Lock()
        self.n = n
        self.done = 0
        with self.lock:
            print("%-16s %-6s %8s  %s" % ("task_id", "result", "seconds", "note"))
            print("-" * 72, flush=True)

    def row(self, r: dict) -> None:
        status = "PASS" if r["passed"] else "FAIL"
        note = r["error"] or ""
        if len(note) > 44:
            note = note[:41] + "..."
        with self.lock:
            self.done += 1
            print(
                "%-16s %-6s %7.2fs  %s"
                % (r["task_id"], status, r["duration_s"], note),
                flush=True,
            )


# --------------------------------------------------------------------------
# modes
# --------------------------------------------------------------------------
def guard_base_url(base_url: str) -> None:
    """Refuse the port that is reserved for GPU benchmarking on this machine."""
    if os.environ.get("CODING_EVAL_ALLOW_8080") == "1":
        return
    match = re.match(r"^https?://([^/:]+)(?::(\d+))?", base_url or "")
    if not match:
        return
    host, port = match.group(1).lower(), match.group(2)
    if host in ("127.0.0.1", "localhost", "::1", "[::1]", "0.0.0.0") and port == "8080":
        raise SystemExit(
            "refusing to send requests to %s: port 8080 is reserved for GPU "
            "benchmarking on this machine.\nUse another port, or set "
            "CODING_EVAL_ALLOW_8080=1 if you really mean it." % base_url
        )


def cmd_fetch_dataset() -> int:
    try:
        path = download_dataset()
    except DatasetError as exc:
        print("[dataset] download FAILED: %s" % exc)
        print("[dataset] the harness will use the bundled fallback (%d problems)"
              % len(BUNDLED))
        return 1
    problems = load_humaneval(path)
    print("[dataset] OK: %d problems, %d bytes at %s"
          % (len(problems), path.stat().st_size, path))
    return 0


def cmd_selftest() -> int:
    """Prove the bundled problems are correct: canonical passes, junk fails."""
    print("selftest: %d bundled problems" % len(BUNDLED))
    bad = []
    for problem in BUNDLED:
        good = run_program(
            build_program(problem, problem["canonical_solution"]), DEFAULT_EXEC_TIMEOUT
        )
        junk = run_program(build_program(problem, "    return None\n"), DEFAULT_EXEC_TIMEOUT)
        ok = good["passed"] and not junk["passed"]
        print(
            "  %-16s canonical=%s junk=%s %s"
            % (
                problem["task_id"],
                "PASS" if good["passed"] else "FAIL",
                "PASS" if junk["passed"] else "FAIL",
                "" if ok else "  <-- PROBLEM",
            )
        )
        if not ok:
            bad.append(problem["task_id"])
    if bad:
        print("selftest FAILED for: %s" % ", ".join(bad))
        return 1
    print("selftest OK: all %d bundled problems accept their canonical solution "
          "and reject a wrong one" % len(BUNDLED))
    return 0


def cmd_mock_selftest(args) -> int:
    """--mock: run the normal HTTP path, but offline against a stub server.
    Nothing here touches a GPU or downloads anything."""
    print("[mock] self-test mode: offline problem set, HTTP path unchanged, "
          "no dataset download")
    if args.dataset == "humaneval":
        if DATASET_PATH.exists():
            try:
                return run_suite(
                    args,
                    forced_dataset=(
                        "humaneval",
                        "loaded from cache %s (--mock, offline)" % DATASET_PATH,
                        load_humaneval(),
                    ),
                )
            except Exception as exc:
                print("[mock] cache unreadable (%s); using the bundled set" % exc)
        else:
            print("[mock] no HumanEval cache at %s; using the bundled set"
                  % DATASET_PATH)
    return run_suite(args, forced_dataset=("bundled", "bundled problem set (--mock)", None))


def run_suite(args, forced_dataset=None) -> int:
    started = datetime.now(timezone.utc)
    t0 = time.time()
    if forced_dataset:
        dataset, note = forced_dataset[0], forced_dataset[1]
        problems = list(forced_dataset[2]) if forced_dataset[2] else list(BUNDLED)
    else:
        problems, dataset, note = select_dataset(args)
    total = len(problems)
    offset = max(0, args.offset)
    if offset:
        problems = problems[offset:]
    if args.limit is not None:
        problems = problems[: max(0, args.limit)]
    if not problems:
        print("no problems selected (offset=%d limit=%s)" % (offset, args.limit))
        return 2

    if args.mock:
        note = (note + "; " if note else "") + "MOCK run (stub server / no GPU)"
    print(
        "[run] model=%s base_url=%s dataset=%s problems=%d/%d parallel=%d "
        "temperature=%s max_tokens=%s reasoning_effort=%s exec_timeout=%.1fs"
        % (
            args.model,
            args.base_url,
            dataset,
            len(problems),
            total,
            args.parallel,
            args.temperature,
            args.max_tokens,
            args.reasoning_effort or "default",
            args.exec_timeout,
        ),
        flush=True,
    )
    print("[run] dataset note: %s" % note, flush=True)

    printer = Printer(len(problems))
    results = [None] * len(problems)
    if args.parallel <= 1:
        for i, problem in enumerate(problems):
            results[i] = evaluate_one(problem, args)
            printer.row(results[i])
    else:
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = {pool.submit(evaluate_one, p, args): i for i, p in enumerate(problems)}
            for fut in as_completed(futures):
                i = futures[fut]
                results[i] = fut.result()
                printer.row(results[i])

    passed = sum(1 for r in results if r and r["passed"])
    completion_tokens = sum(r["completion_tokens"] for r in results if r)
    wall = time.time() - t0
    summary = {
        "model": args.model,
        "base_url": args.base_url,
        "dataset": dataset,
        "n": len(results),
        "passed": passed,
        "pass_at_1": (passed / len(results)) if results else 0.0,
        "total_completion_tokens": completion_tokens,
        "wall_s": round(wall, 3),
        "started_utc": started.isoformat(timespec="seconds"),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort,
        # extras (not required, useful for a 3-way comparison)
        "dataset_note": note,
        "mock": bool(args.mock),
        "harness_version": VERSION,
        "offset": offset,
        "limit": args.limit,
        "parallel": args.parallel,
        "exec_timeout_s": args.exec_timeout,
        "http_timeout_s": args.timeout,
        "total_prompt_tokens": sum(r["prompt_tokens"] for r in results if r),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    payload = dict(summary)
    payload["summary"] = summary
    payload["problems"] = results
    out_path = Path(args.out)
    if out_path.parent and not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("-" * 72)
    print(
        "RESULT: model=%s dataset=%s passed=%d/%d pass@1=%.4f wall=%.1fs "
        "completion_tokens=%d"
        % (args.model, dataset, passed, len(results), summary["pass_at_1"], wall,
           completion_tokens)
    )
    print("JSON written to %s" % out_path)
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="coding_eval.py",
        description="HumanEval-style coding-capability harness for OpenAI-compatible "
                    "local endpoints (pass@1).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
               "  coding_eval.py --mock --base-url http://127.0.0.1:8199/v1 --model stub \\\n"
               "      --out mock-run.json --limit 5\n"
               "  coding_eval.py --base-url http://127.0.0.1:8000/v1 --model qwen3 \\\n"
               "      --out run-qwen.json --limit 20 --temperature 0\n",
    )
    p.add_argument("--base-url", default="http://127.0.0.1:8199/v1",
                   help="OpenAI-compatible base URL; '/chat/completions' is appended")
    p.add_argument("--model", default="stub", help="model name sent in the request body")
    p.add_argument("--out", default="coding_eval-out.json", help="path of the output JSON")
    p.add_argument("--limit", type=int, default=None, help="evaluate at most N problems")
    p.add_argument("--offset", type=int, default=0, help="skip the first N problems")
    p.add_argument("--timeout", type=float, default=120.0,
                   help="per-request HTTP timeout in seconds (default 120)")
    p.add_argument("--temperature", type=float, default=0.0, help="sampling temperature")
    p.add_argument("--max-tokens", type=int, default=512, help="max_tokens per request")
    p.add_argument("--reasoning-effort", default=None,
                   choices=["none", "low", "medium", "high"],
                   help="sent as 'reasoning_effort' when provided")
    p.add_argument("--exec-timeout", type=float, default=DEFAULT_EXEC_TIMEOUT,
                   help="hard subprocess timeout per candidate in seconds (default 10)")
    p.add_argument("--parallel", type=int, default=1,
                   help="number of problems to evaluate concurrently (default 1)")
    p.add_argument("--dataset", default="auto", choices=["auto", "humaneval", "bundled"],
                   help="auto (default): cached/downloaded HumanEval, bundled on failure")
    p.add_argument("--mock", action="store_true",
                   help="self-test mode against a stub server: offline bundled problem "
                        "set, marks the run as mock, never downloads the dataset")
    p.add_argument("--fetch-dataset", action="store_true",
                   help="download/cache HumanEval.jsonl.gz and exit")
    p.add_argument("--selftest", action="store_true",
                   help="verify the bundled problems (canonical solution passes, wrong "
                        "solution fails) and exit")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        return cmd_selftest()
    if args.fetch_dataset:
        return cmd_fetch_dataset()
    guard_base_url(args.base_url)
    if args.mock:
        return cmd_mock_selftest(args)
    return run_suite(args)


if __name__ == "__main__":
    sys.exit(main())
