"""`prep_eval_data.py` 的单测：选择规则必须是**确定性**且**会报错**的。

两条底线：
  ① 同样的输入、同样的 seed ⇒ 同样的 8 条 / 12 条（否则前后两次训练没法比）；
  ② 源数据改了措辞 ⇒ 报错而不是悄悄换行（否则「8 条是哪 8 条」会漂）。
"""

from __future__ import annotations

# --- path bootstrap: this file sits under training/<sub>/; shared modules are one level up ---
import pathlib as _p, sys as _s
for _d in (_p.Path(__file__).resolve().parent,
           _p.Path(__file__).resolve().parents[1],
           _p.Path(__file__).resolve().parents[1] / "backward",
           _p.Path(__file__).resolve().parents[1] / "bench",
           _p.Path(__file__).resolve().parents[1] / "mezo"):
    if str(_d) not in _s.path:
        _s.path.insert(0, str(_d))


import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import prep_eval_data as P  # noqa: E402


def _write(rows: list[dict]) -> str:
    f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
    f.close()
    return f.name


def _identity_rows(n: int = 26) -> list[dict]:
    """造一份与真数据同形状的假身份集：前 16 行是事实问答，后 10 行是闲聊。"""
    prompts = ["你是谁？", "你是什么模型？", "谁开发了你？", "谁是你的开发者？", "谁训练了你？",
               "你是谁开发的？", "谁创造了你？", "你叫什么名字？", "Who are you?",
               "What model are you?", "Who developed you?", "Who made you?",
               "Who created you?", "Who built you?", "Who is your developer?",
               "What is your name?"]
    rows = [{"prompt": p, "answer": f"A{i}"} for i, p in enumerate(prompts)]
    while len(rows) < n:
        rows.append({"prompt": f"闲聊 {len(rows)}", "answer": "x"})
    return rows


class PickIdentity(unittest.TestCase):
    def setUp(self):
        self.path = _write(_identity_rows())
        self.addCleanup(os.unlink, self.path)

    def test_eight_items_four_facts_both_languages(self):
        got = P.pick_identity(self.path)
        self.assertEqual(len(got), 8)
        self.assertEqual({r["fact"] for r in got}, {"identity", "basemodel", "developer", "finetune"})
        self.assertEqual(sorted(r["lang"] for r in got), ["en"] * 4 + ["zh"] * 4)

    def test_same_input_same_output(self):
        a = P.pick_identity(self.path)
        b = P.pick_identity(self.path)
        self.assertEqual([r["prompt"] for r in a], [r["prompt"] for r in b])

    def test_wording_change_is_fatal(self):
        rows = _identity_rows()
        rows[0]["prompt"] = "阁下是何方神圣？"          # 第 1 行被改
        bad = _write(rows)
        self.addCleanup(os.unlink, bad)
        with self.assertRaises(SystemExit) as cm:
            P.pick_identity(bad)
        self.assertIn("同步改 FACTS", str(cm.exception))

    def test_short_file_is_fatal(self):
        short = _write(_identity_rows()[:5])           # 取不到第 15 行
        self.addCleanup(os.unlink, short)
        with self.assertRaises(SystemExit):
            P.pick_identity(short)


def _math_rows(n: int = 36) -> list[dict]:
    syms = ["+", "-", "×", "÷"]
    return [{"prompt": f"计算：{i} {syms[i % 4]} ?", "answer": f"{i} {syms[i % 4]} = {i}"}
            for i in range(n)]


class PickMath(unittest.TestCase):
    def setUp(self):
        self.path = _write(_math_rows())
        self.addCleanup(os.unlink, self.path)

    def test_twelve_items_in_three_splits(self):
        got = P.pick_math(self.path)
        self.assertEqual(len(got), 12)
        counts = {s: sum(1 for r in got if r["split"] == s) for s in ("train", "dev", "holdout")}
        self.assertEqual(counts, {"train": 4, "dev": 4, "holdout": 4})

    def test_deterministic_and_unique(self):
        a = P.pick_math(self.path)
        b = P.pick_math(self.path)
        self.assertEqual([r["prompt"] for r in a], [r["prompt"] for r in b])
        self.assertEqual(len({r["prompt"] for r in a}), 12)          # 不重复

    def test_operator_variety(self):
        got = P.pick_math(self.path)
        ops = {r["answer"].split()[1] for r in got}
        self.assertGreaterEqual(len(ops), 3, f"12 题只有 {ops} 这几种运算，分桶没生效")

    def test_too_few_is_fatal(self):
        small = _write(_math_rows(6))
        self.addCleanup(os.unlink, small)
        with self.assertRaises(SystemExit):
            P.pick_math(small)


if __name__ == "__main__":
    unittest.main()
