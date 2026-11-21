r"""Build the fixed IQ-eval subsets (seed 0) from the pinned datasets under D:\exstrata-work\evalsets.

    python subsets.py [--evalsets DIR] [--out DIR/subsets.jsonl]

Needs pyarrow (the venv has none: run it with any Python that has pyarrow, e.g. the one that ships `hf`).
One JSON object per line: id ("<set>/<source key>"), set, prompt, gold.  HumanEval's gold is a dict (prompt, test, entry_point);
every other gold is a string (a number, a MATH answer, or a letter).  The prompt templates live here and are the same for every arm.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

N = {"gsm8k": 200, "math500": 150, "humaneval": 164, "mmlu_pro": 200, "jcommonsenseqa": 200}
LETTERS = "ABCDEFGHIJ"

BOXED = "Put the final answer in \\boxed{{}}."
GSM_T = "Solve the following math problem. Reason step by step. " + BOXED + "\n\n{q}"
MATH_T = "Solve the following math problem. Reason step by step. " + BOXED + "\n\n{q}"
CODE_T = ("Complete the following Python function. Reply with the complete function (imports and helpers included) "
          "in one ```python code block.\n\n```python\n{p}\n```")
MC_T = "Answer the following multiple-choice question. Answer with the letter only.\n\n{q}\n\n{opts}"


def _opts(options: list[str]) -> str:
    return "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))


def _parquet(path: Path) -> list[dict]:
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def stratified(rows: list[dict], key: str, n: int, rng: random.Random) -> list[dict]:
    """n rows, proportional to each group's size (largest remainder), random within a group."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    names = sorted(groups)
    exact = {g: n * len(groups[g]) / len(rows) for g in names}
    quota = {g: int(exact[g]) for g in names}
    for g in sorted(names, key=lambda g: (exact[g] - quota[g], g), reverse=True)[: n - sum(quota.values())]:
        quota[g] += 1
    out = []
    for g in names:
        out += rng.sample(groups[g], quota[g])
    return out


def build(root: Path) -> list[dict]:
    items: list[dict] = []

    def add(s, key, prompt, gold):
        items.append({"id": f"{s}/{key}", "set": s, "prompt": prompt, "gold": gold})

    rows = _parquet(root / "gsm8k/main/test-00000-of-00001.parquet")
    for i in sorted(random.Random(0).sample(range(len(rows)), N["gsm8k"])):
        gold = rows[i]["answer"].split("####")[-1].strip().replace(",", "")
        add("gsm8k", i, GSM_T.format(q=rows[i]["question"]), gold)

    rows = [json.loads(x) for x in (root / "math500/test.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    for i in sorted(random.Random(0).sample(range(len(rows)), N["math500"])):
        add("math500", rows[i]["unique_id"], MATH_T.format(q=rows[i]["problem"]), rows[i]["answer"])

    rows = _parquet(root / "humaneval/openai_humaneval/test-00000-of-00001.parquet")
    for r in rows:                                                  # all 164
        add("humaneval", r["task_id"].split("/")[-1], CODE_T.format(p=r["prompt"].rstrip()),
            {"prompt": r["prompt"], "test": r["test"], "entry_point": r["entry_point"]})

    rows = _parquet(root / "mmlu_pro/data/test-00000-of-00001.parquet")
    for r in sorted(stratified(rows, "category", N["mmlu_pro"], random.Random(0)), key=lambda r: int(r["question_id"])):
        add("mmlu_pro", r["question_id"], MC_T.format(q=r["question"].strip(), opts=_opts(list(r["options"]))), r["answer"])

    rows = _parquet(root / "jcommonsenseqa/data/validation-00000-of-00001.parquet")
    for i in sorted(random.Random(0).sample(range(len(rows)), N["jcommonsenseqa"])):
        r = rows[i]
        add("jcommonsenseqa", r["q_id"], MC_T.format(q=r["question"].strip(), opts=_opts([r[f"choice{k}"] for k in range(5)])),
            LETTERS[int(r["label"])])
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evalsets", default=r"D:\exstrata-work\evalsets")
    ap.add_argument("--out")
    a = ap.parse_args()
    root = Path(a.evalsets)
    items = build(root)
    out = Path(a.out) if a.out else root / "subsets.jsonl"
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items), encoding="utf-8")
    counts = defaultdict(int)
    for x in items:
        counts[x["set"]] += 1
    print(out, len(items), dict(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
