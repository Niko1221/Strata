"""T5: the correctness checks of the RTX 3060 community report (bench/results/2026-10-09-community-rtx-3060-0141),
repeated through a running server so the results can be put beside that report's IQ2_XS row (same GPU, other CPU):

  code      HumanEval IDs 0-39 (EvalPlus 0.3.1's HumanEval+ v0.1.10), one greedy answer each, up to 1,024 tokens,
            the same instruction; sanitized with evalplus.sanitize. Scored afterwards by strata_eval_subset.py.
  japanese  12 short Japanese questions with a fixed expected answer (exact match after stripping)
  tool      8 calls of an `add` tool, then the tool's result and "reply with only the integer"

Run with the EvalPlus venv (it imports evalplus.sanitize). Writes OUT/code/iq2_xs-samples.jsonl, OUT/requests.jsonl,
OUT/grades.jsonl and OUT/summary.json.

    python quality_suite.py --out DIR [--subset humaneval-subset.json] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strata_client import body, chat, status  # noqa: E402

HERE = Path(__file__).resolve().parent
INSTRUCTION = "Complete this Python function. Output valid Python code only, no explanations.\n\n"

# Fixed before any answers were seen (copied from the 3060 report's strata_bench_run.py)
JAPANESE = [
    ("17×19を計算し、整数だけを答えてください。", "323"),
    ("123から58を引き、整数だけを答えてください。", "65"),
    ("文字列abcdeを逆順にし、その文字列だけを答えてください。", "edcba"),
    ("[8, 3, 5, 1]を昇順に並べ、JSON配列だけを答えてください。", [1, 3, 5, 8]),
    ("JSONだけで答えてください。キーnameの値を東京、キーcountの値を3にしてください。", {"name": "東京", "count": 3}),
    ("りんごが7個あり3個食べました。残りの個数を整数だけで答えてください。", "4"),
    ("次の文章に含まれる都市名だけを答えてください：私は昨日、大阪で友人と会いました。", "大阪"),
    ("A=5、B=9です。A+Bの結果を整数だけで答えてください。", "14"),
    ("東京、大阪、京都の3つをその順でJSON配列として答えてください。", ["東京", "大阪", "京都"]),
    ("英単語catを日本語のひらがなに訳し、訳語だけを答えてください。", "ねこ"),
    ("次の文字列の英小文字aの個数を整数だけで答えてください：banana", "3"),
    ("次のJSONのvalueだけを整数で答えてください：{\"value\":42,\"other\":7}", "42")]
TOOLS = [{"type": "function", "function": {"name": "add", "description": "Add two integers.",
          "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                         "required": ["a", "b"], "additionalProperties": False}}}]
PAIRS = [(2, 3), (17, 19), (-4, 9), (100, 23), (0, 7), (81, -12), (999, 1), (14, 28)]


def main() -> int:
    from evalplus.sanitize import sanitize

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--subset", type=Path, default=HERE.parent / "data" / "humaneval-subset.json")
    ap.add_argument("--model", default="iq2_xs", help="the name in the samples file")
    ap.add_argument("--limit", type=int, default=0, help="only the first N items of each part (a smoke test)")
    a = ap.parse_args()
    out = a.out
    (out / "code").mkdir(parents=True, exist_ok=True)
    shutil.copy(a.subset, out / "humaneval-subset.json")   # strata_eval_subset.py checks it against EvalPlus
    dataset = json.loads(a.subset.read_text(encoding="utf-8"))
    ids = dataset["ids"][: a.limit or None]
    reqlog = (out / "requests.jsonl").open("a", encoding="utf-8")
    grades = (out / "grades.jsonl").open("a", encoding="utf-8")
    (out / "status-before.json").write_text(json.dumps(status(), indent=1), encoding="utf-8")

    def run(label: str, group: str, req: dict) -> dict:
        row = chat(req)
        row.update(label=label, group=group)
        reqlog.write(json.dumps({**row, "request": req}, ensure_ascii=False) + "\n")
        reqlog.flush()
        print(f"{label:24s} ok={row['ok']} gen={row.get('completion_tokens')} decode={row.get('decode_tps')} "
              f"finish={row.get('finish_reason')} {row.get('error') or ''}", flush=True)
        return row

    summary = {"code_generated": 0, "code_at_cap": 0, "japanese_pass": 0, "japanese_graded": 0,
               "tool_pass": 0, "tool_graded": 0, "request_errors": 0}
    with (out / "code" / f"{a.model}-samples.jsonl").open("w", encoding="utf-8") as f:
        for tid in ids:
            p = dataset["problems"][tid]
            row = run("code-" + tid.replace("/", "-"), "code", body(INSTRUCTION + p["prompt"], 1024))
            if not row["ok"]:
                summary["request_errors"] += 1
                continue
            summary["code_generated"] += 1
            summary["code_at_cap"] += row.get("completion_tokens") == 1024
            f.write(json.dumps({"task_id": tid, "solution": sanitize(row["text"], p["entry_point"])}) + "\n")
            f.flush()

    for i, (prompt, expected) in enumerate(JAPANESE[: a.limit or None]):
        row = run(f"japanese-{i}", "japanese", body(prompt, 128))
        text = (row.get("text") or "").strip().strip("`").strip()
        try:
            actual = json.loads(text) if isinstance(expected, (list, dict)) else text
        except ValueError:
            actual = None
        ok = row["ok"] and actual == expected
        summary["japanese_graded"] += 1
        summary["japanese_pass"] += ok
        summary["request_errors"] += not row["ok"]
        grades.write(json.dumps({"group": "japanese", "index": i, "expected": expected, "answer": row.get("text"),
                                 "pass": ok}, ensure_ascii=False) + "\n")

    for i, (x, y) in enumerate(PAIRS[: a.limit or None]):
        req = body(f"Use the add tool to add {x} and {y}. Do not compute it yourself.", 128)
        req["tools"] = TOOLS
        row = run(f"tool-{i}", "tool", req)
        calls = row.get("tool_calls") or []
        call = calls[0] if len(calls) == 1 else None
        try:
            valid = bool(call and call["function"]["name"] == "add"
                         and json.loads(call["function"]["arguments"]) == {"a": x, "b": y})
        except (ValueError, KeyError, TypeError):
            valid = False
        reply_text = None
        if valid:
            follow = body("", 128)
            follow["tools"] = TOOLS
            follow["messages"] = req["messages"] + [
                {"role": "assistant", "content": row.get("text") or "", "tool_calls": calls},
                {"role": "tool", "tool_call_id": call.get("id", "call_0"), "content": str(x + y)},
                {"role": "user", "content": "Reply with only the integer result returned by the tool."}]
            reply = run(f"tool-followup-{i}", "tool_followup", follow)
            reply_text = reply.get("text")
            valid = reply["ok"] and (reply_text or "").strip() == str(x + y)
        summary["tool_graded"] += 1
        summary["tool_pass"] += valid
        summary["request_errors"] += not row["ok"]
        grades.write(json.dumps({"group": "tool", "index": i, "pair": [x, y], "tool_calls": calls,
                                 "reply": reply_text, "pass": valid}, ensure_ascii=False) + "\n")

    (out / "status-after.json").write_text(json.dumps(status(), indent=1), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
