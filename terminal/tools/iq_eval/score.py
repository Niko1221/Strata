"""Score a run.py output against subsets.jsonl.

    python score.py RUN.jsonl --out SCORED.jsonl [--subsets D:\\exstrata-work\\evalsets\\subsets.jsonl]

Writes one line per item: id, set, correct, pred, gold, completion_tokens, finish_reason.  Prints per-set accuracy.
HumanEval EXECUTES MODEL-WRITTEN CODE on this machine (see README): one fresh Python process per item (python -I, a new temp directory,
10 s timeout).  A row with a request error counts as wrong; run ids that are not in the subsets file are skipped.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

DEFAULT_SUBSETS = r"D:\exstrata-work\evalsets\subsets.jsonl"
THINK = re.compile(r"<think>.*?</think>", re.S)


def clean(text: str) -> str:
    """Drop a <think> block the server left in the content; an unclosed one hides everything after it."""
    text = THINK.sub("", text or "")
    return text.split("<think>")[0] if "<think>" in text else text


# ---- numbers and MATH answers -------------------------------------------------------------------------------------------

def last_boxed(text: str) -> str | None:
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        depth += (text[k] == "{") - (text[k] == "}")
        if depth == 0:
            return text[j + 1:k]
    return None                                             # unbalanced: a truncated answer


def _strip_tex(s: str) -> str:
    s = s.strip().strip("$").strip()
    for a, b in (("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""),
                 ("\\;", ""), ("\\ ", ""), ("^\\circ", ""), ("^{\\circ}", ""), ("\\%", ""), ("%", ""), ("\\$", ""), ("$", "")):
        s = s.replace(a, b)
    s = re.sub(r"\\(?:text|textbf|mathrm|mbox)\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    return s.strip()


def to_float(s: str) -> float | None:
    s = _strip_tex(s).replace(" ", "")
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)           # 1,000 -> 1000
    try:
        m = re.fullmatch(r"(-?)\\frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}", s)
        if m:
            return (-1 if m[1] else 1) * float(m[2]) / float(m[3])
        m = re.fullmatch(r"(-?[\d.]+)/(-?[\d.]+)", s)
        if m:
            return float(m[1]) / float(m[2])
        return float(s)
    except (ValueError, ZeroDivisionError):
        return None


def norm_str(s: str) -> str:
    s = _strip_tex(s).rstrip(".")
    s = re.sub(r"^[a-zA-Z]\s*=\s*", "", s)                  # "x = 5"
    s = re.sub(r"\s+", "", s)
    return s.replace("\\{", "{").replace("\\}", "}")


def math_equiv(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    if norm_str(pred) == norm_str(gold):
        return True
    a, b = to_float(pred), to_float(gold)
    return a is not None and b is not None and math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)


def extract_math(text: str) -> str | None:
    return last_boxed(clean(text))


def extract_number(text: str) -> str | None:
    """GSM8K: the last boxed answer, else the last number in the text."""
    text = clean(text)
    b = last_boxed(text)
    if b is not None:
        m = re.findall(r"-?\d[\d,]*\.?\d*|-?\.\d+", _strip_tex(b))
        return m[-1].replace(",", "").rstrip(".") if m else _strip_tex(b)
    m = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return m[-1].replace(",", "").rstrip(".") if m else None


def gsm_equiv(pred: str | None, gold: str) -> bool:
    a, b = to_float(pred) if pred else None, to_float(gold)
    return a is not None and b is not None and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6)


# ---- multiple choice -------------------------------------------------------------------------------------------------------

def extract_letter(text: str, valid: str = "ABCDEFGHIJ") -> str | None:
    text = clean(text).strip()
    v = f"[{valid}]"
    for pat in (rf"\\boxed\{{\s*(?:\\text\{{)?\(?({v})\)?", rf"(?i:answer)\s*(?:is)?\s*[:：]?\s*\**\s*\(?({v})\b",
                rf"\A\W*({v})\W*\Z", rf"\A\(?({v})[\).:]\s"):
        m = re.findall(pat, text, re.M)
        if m:
            return m[-1]
    last = text.splitlines()[-1] if text else ""
    m = re.fullmatch(rf"\W*({v})\W*", last)                 # a last line that is only a letter
    return m[1] if m else None


# ---- HumanEval -------------------------------------------------------------------------------------------------------------

def _defines(code: str, entry_point: str) -> bool:
    return re.search(rf"def\s+{re.escape(entry_point)}\s*\(", code) is not None


def extract_code(text: str, entry_point: str) -> str:
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", clean(text), re.S)
    for b in reversed(blocks):
        if _defines(b, entry_point):
            return b
    return blocks[-1] if blocks else clean(text)


def run_humaneval(gold: dict, output: str, timeout: float = 10, python: str | None = None) -> tuple[bool, str]:
    """Run prompt + completion + tests in a fresh process.  Returns (passed, note)."""
    code = extract_code(output, gold["entry_point"])
    # The prompt (imports, helpers, the signature with its docstring) comes first; a full function after it redefines the stub,
    # a bare body continues it.
    sep = "\n\n" if _defines(code, gold["entry_point"]) else "\n"
    program = gold["prompt"] + sep + code + "\n\n" + gold["test"] + f"\n\ncheck({gold['entry_point']})\n"
    with tempfile.TemporaryDirectory(prefix="iq_he_") as d:
        (Path(d) / "main.py").write_text(program, encoding="utf-8")
        try:
            p = subprocess.run([python or sys.executable, "-I", "main.py"], cwd=d, capture_output=True, text=True, timeout=timeout,
                               stdin=subprocess.DEVNULL,
                               env={"SYSTEMROOT": os.environ.get("SYSTEMROOT", ""), "PYTHONIOENCODING": "utf-8"})
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if p.returncode == 0:
        return True, ""
    err = p.stderr.strip().splitlines()
    return False, err[-1][:200] if err else "exit"


# ---- driver ----------------------------------------------------------------------------------------------------------------

def score_item(item: dict, row: dict, python: str | None = None) -> dict:
    s, gold, out = item["set"], item["gold"], row.get("output") or ""
    if s == "gsm8k":
        pred = extract_number(out)
        ok = gsm_equiv(pred, gold)
    elif s == "math500":
        pred = extract_math(out)
        ok = math_equiv(pred, gold)
    elif s in ("mmlu_pro", "jcommonsenseqa"):
        pred = extract_letter(out, "ABCDEFGHIJ" if s == "mmlu_pro" else "ABCDE")
        ok = pred == gold
    elif s == "humaneval":
        ok, pred = run_humaneval(gold, out, python=python)
        gold = gold["entry_point"]
    else:
        raise ValueError(f"unknown set {s}")
    return {"id": item["id"], "set": s, "correct": bool(ok) and "error" not in row, "pred": pred, "gold": gold,
            "completion_tokens": row.get("completion_tokens"), "finish_reason": row.get("finish_reason")}


def read_jsonl(path: str) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--subsets", default=DEFAULT_SUBSETS)
    a = ap.parse_args(argv)
    items = {d["id"]: d for d in read_jsonl(a.subsets)}
    rows = {d["id"]: d for d in read_jsonl(a.run)}           # a later line (a rerun after an error) wins
    res = [score_item(items[i], r) for i, r in rows.items() if i in items]
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in res), encoding="utf-8")
    by = defaultdict(list)
    for r in res:
        by[r["set"]].append(r["correct"])
    for s, v in by.items():
        print(f"{s:16s} {sum(v):4d}/{len(v):4d} = {100 * sum(v) / len(v):5.1f}%")
    print(f"{'pooled':16s} {sum(r['correct'] for r in res):4d}/{len(res):4d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
