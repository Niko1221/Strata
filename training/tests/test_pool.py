"""样本池与池子入口的判据一致性。

这里最重要的不是「字段能存下来」，而是**两条入口给出同一个答案**：
`coldstart.plan()`（行序，历史数据）与 `coldstart.plan_pool()`（按 prompt 聚合，
在线数据）必须对同一批样本判出同样的档位。两套判据各改一半，是最容易发生、
又最难在日志里发现的漂移。
"""

import itertools
import json
import pathlib as _p
import tempfile
import unittest

import coldstart as C
import pool as P


def _write_jsonl(path: _p.Path, rows) -> str:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return str(path)


class TestVerdictParity(unittest.TestCase):
    def test_classify_verdicts_matches_classify(self):
        """穷举 reward × truncated 的组合，两条判据逐格相同。

        这是 `verdict_of` 存在的全部理由。最容易错的一格是「截断但已答对」：
        `classify()` 的 `acc >= 1` 分支先于截断判定（注释写明「答对是真的」），
        所以推断顺序也必须是 reward 先于 truncated。
        """
        n = 0
        for rewards in itertools.product([0.0, 0.5, 1.0], repeat=4):
            for trunc in itertools.product([False, True], repeat=4):
                old = C.classify(list(rewards), list(trunc))
                new = P.classify_verdicts([
                    P.verdict_of({"reward": r, "truncated": t})
                    for r, t in zip(rewards, trunc)], list(trunc))
                self.assertEqual(old, new, f"rewards={rewards} truncated={trunc}")
                n += 1
        self.assertEqual(n, 3 ** 4 * 2 ** 4)

    def test_truncated_but_correct_is_positive(self):
        # 截断标记只说明没写完，不否定「已经答对」。记成 unknown 会把整组从
        # graduated 打回 needs_budget —— 一道已经会做的题被反复重采。
        self.assertEqual(P.verdict_of({"reward": 1.0, "truncated": True}), P.POS)
        self.assertEqual(P.verdict_of({"reward": 0.0, "truncated": True}), P.UNKNOWN)

    def test_explicit_verdict_wins(self):
        self.assertEqual(P.verdict_of({"reward": 1.0, "verdict": "neg"}), P.NEG)
        self.assertEqual(P.verdict_of({"reward": 0.0, "verdict": "unknown"}), P.UNKNOWN)


class TestPoolAppend(unittest.TestCase):
    def test_rejects_incomplete_or_illegal(self):
        with tempfile.TemporaryDirectory() as td:
            pool = P.Pool(_p.Path(td) / "p.jsonl")
            with self.assertRaises(ValueError):
                pool.append({"completion": "x"})                 # 缺 prompt
            with self.assertRaises(ValueError):
                pool.append({"prompt": "q", "completion": ""})   # 空 completion
            with self.assertRaises(ValueError):
                pool.append({"prompt": "q", "completion": "c", "verdict": "maybe"})
            # 既没 verdict 也没 reward/truncated ⇒ 判不出正负，不许进池
            with self.assertRaises(ValueError):
                pool.append({"prompt": "q", "completion": "c"})
            self.assertEqual(pool.rows(), [])

    def test_keeps_unknown_fields(self):
        # 池子要长期生长，外部工具挂的元数据不能因为不认识就被丢掉
        with tempfile.TemporaryDirectory() as td:
            pool = P.Pool(_p.Path(td) / "p.jsonl")
            pool.append({"prompt": "q", "completion": "c", "verdict": "pos",
                         "episode": 7, "nested": {"a": 1}})
            r = pool.rows()[0]
            self.assertEqual(r["episode"], 7)
            self.assertEqual(r["nested"], {"a": 1})


class TestGroups(unittest.TestCase):
    def test_does_not_depend_on_row_order(self):
        """交错写入后仍按 prompt 正确聚合 —— 这正是行序入口做不到的事。"""
        with tempfile.TemporaryDirectory() as td:
            rows = []
            for i in range(4):
                rows.append({"prompt": "A", "completion": f"a{i}", "reward": 1.0})
                rows.append({"prompt": "B", "completion": f"b{i}", "reward": 0.0})
            pool = P.Pool(_write_jsonl(_p.Path(td) / "p.jsonl", rows))
            g = pool.groups()
            self.assertEqual(sorted(g), ["A", "B"])
            self.assertEqual([len(g["A"]), len(g["B"])], [4, 4])
            self.assertEqual([s["completion"] for s in g["A"]], ["a0", "a1", "a2", "a3"])

    def test_keeps_short_tail(self):
        """不足 K 条的尾巴必须留下：在线场景里「刚采的几条」就是尾巴。"""
        with tempfile.TemporaryDirectory() as td:
            rows = [{"prompt": "A", "completion": f"a{i}", "reward": 1.0} for i in range(4)]
            rows += [{"prompt": "B", "completion": f"b{i}", "reward": 0.0} for i in range(2)]
            pool = P.Pool(_write_jsonl(_p.Path(td) / "p.jsonl", rows))
            self.assertEqual(sorted(pool.groups()), ["A", "B"])
            # 行序入口会把这 2 行丢掉（这是它的既有行为，只加告警，不改）
            self.assertEqual(len(C.plan(rows, 4)["per_group"]), 1)

    def test_relabel_overrides_in_place(self):
        """改判 = 追加一条同 sid 的记录：组大小不变、位置不变、内容取最新。

        池子是 append-only，判定却会变（判分器升级、人改主意）。如果不能原地覆盖，
        要么改写历史（丢证据），要么同一轨迹算两次（组内归一化被稀释）。
        """
        with tempfile.TemporaryDirectory() as td:
            pool = P.Pool(_p.Path(td) / "p.jsonl")
            first = pool.append({"prompt": "A", "completion": "c0",
                                 "verdict": "unknown", "adapter": "r1"})
            pool.append({"prompt": "A", "completion": "c1",
                         "verdict": "unknown", "adapter": "r1"})
            pool.append({**first, "verdict": "pos", "source": "human"})
            g = pool.groups()
            self.assertEqual(len(g["A"]), 2)                       # 组没变大
            self.assertEqual(g["A"][0]["completion"], "c0")        # 位置没动
            self.assertEqual(P.verdict_of(g["A"][0]), P.POS)       # 内容更新了
            self.assertEqual(pool.stats()["samples"], 2)           # 统计不重复
            self.assertEqual(pool.stats()["pos"], 1)

    def test_identical_completions_are_not_collapsed(self):
        """同一 prompt 下内容完全相同的 K 条必须保留为 K 条。

        低温度采样经常产出逐字相同的轨迹。若按 (completion, adapter) 去重，4 条会被
        并成 1 条 —— 而「组内全同 ⇒ 优势恒 0」正是最该被报出来的状态，静默合并会让
        它显示成一条正常样本，训练时却一步不动。
        """
        with tempfile.TemporaryDirectory() as td:
            pool = P.Pool(_p.Path(td) / "p.jsonl")
            for _ in range(4):
                pool.append({"prompt": "A", "completion": "same", "verdict": "neg",
                             "adapter": "r1"})
            self.assertEqual(len(pool.groups()["A"]), 4)
            self.assertEqual(pool.stats()["samples"], 4)
            self.assertEqual(len({s["sid"] for s in pool.groups()["A"]}), 4)

    def test_adapter_filter_and_latest(self):
        with tempfile.TemporaryDirectory() as td:
            rows = [{"prompt": "A", "completion": "a0", "reward": 1.0, "adapter": "v1"},
                    {"prompt": "A", "completion": "a1", "reward": 0.0, "adapter": "v2"}]
            pool = P.Pool(_write_jsonl(_p.Path(td) / "p.jsonl", rows))
            self.assertEqual(len(pool.groups()["A"]), 2)
            self.assertEqual([s["completion"] for s in pool.groups("v1")["A"]], ["a0"])
            # latest = 最后一条样本的 adapter ⇒ RL 的 on-policy 子集
            self.assertEqual([s["completion"] for s in pool.groups("latest")["A"]], ["a1"])


class TestPlanPool(unittest.TestCase):
    def _pool(self, td, rows):
        return P.Pool(_write_jsonl(_p.Path(td) / "p.jsonl", rows)).groups()

    def test_matches_plan_on_tidy_input(self):
        """整齐输入（组连续、大小一致）下，两条入口必须判出同样的档位序列。"""
        rows = []
        for gi, rewards in enumerate([[1, 1, 1, 1], [1, 0, 1, 0], [0, 0, 0, 0]]):
            for k, r in enumerate(rewards):
                rows.append({"prompt": f"q{gi}", "completion": f"c{gi}{k}", "reward": r})
        with tempfile.TemporaryDirectory() as td:
            a = C.plan(rows, 4)
            b = C.plan_pool(self._pool(td, rows), rl_group_size=4)
            self.assertEqual([r["verdict"] for r in a["per_group"]],
                             [r["verdict"] for r in b["per_group"]])
            self.assertEqual(len(a["rl"]), len(b["rl"]))
            self.assertEqual(len(a["graduated"]), len(b["graduated"]))
            self.assertEqual(len(a["cold"]), len(b["cold"]))

    def test_unknown_never_becomes_rejected(self):
        """人标「拿不准」不能当 DPO 的负例去惩罚（截断那条坑的推广）。

        构造：3 条明确负例 + 1 条 unknown，且 unknown 那条**最长**（旧代码按
        「最长的完整错例」取 rejected，会优先挑中它）。
        """
        rows = [
            {"prompt": "q", "completion": "w" * 300, "verdict": "neg", "source": "human",
             "gold": "42"},
            {"prompt": "q", "completion": "w" * 340, "verdict": "neg", "source": "human",
             "gold": "42"},
            {"prompt": "q", "completion": "w" * 320, "verdict": "neg", "source": "human",
             "gold": "42"},
            {"prompt": "q", "completion": "X" * 400, "verdict": "unknown", "source": "human",
             "gold": "42"},
        ]
        sol = "s" * 420                                  # 落在 [1/3, 3] 的长度比内
        with tempfile.TemporaryDirectory() as td:
            p = C.plan_pool(self._pool(td, rows), gold_solutions={"q": sol})
            self.assertEqual(p["per_group"][0]["verdict"], "cold")   # 拿不准不该拖累整组
            self.assertEqual(len(p["dpo_pairs"]), 1)
            self.assertEqual(p["dpo_pairs"][0]["rejected"], "w" * 340)
            self.assertEqual(p["dpo_pairs"][0]["chosen"], sol)

    def test_human_verdict_survives_rescore(self):
        """重判覆盖 reward，但不覆盖人标的判定（人的判断是事实，判分器是近似）。"""
        rows = [{"prompt": "q", "completion": "c0", "verdict": "neg", "source": "human"},
                {"prompt": "q", "completion": "c1", "verdict": "neg", "source": "human"}]
        with tempfile.TemporaryDirectory() as td:
            p = C.plan_pool(self._pool(td, rows), rescore=lambda c, g: 1.0)
            self.assertEqual(p["per_group"][0]["verdict"], "cold")

    def test_rl_group_size_trims_and_skips(self):
        """RL 组必须收成固定 K：多则取最近 K 条，少则本轮跳过（留待补齐）。

        `mezo` 的 `len(raw) % group_size` 是硬检查、`group_advantages` 按固定 K 切，
        变长组直接灌进去会让组内归一化算歪。
        """
        rows = ([{"prompt": "big", "completion": f"n{i}", "verdict": "neg"}
                 for i in range(2)]
                + [{"prompt": "big", "completion": f"p{i}", "verdict": "pos"}
                   for i in range(2)]
                + [{"prompt": "big", "completion": f"r{i}", "verdict": "neg"}
                   for i in range(2)]
                + [{"prompt": "small", "completion": "s0", "verdict": "pos"},
                   {"prompt": "small", "completion": "s1", "verdict": "neg"}])
        with tempfile.TemporaryDirectory() as td:
            p = C.plan_pool(self._pool(td, rows), rl_group_size=4)
            big = [r for r in p["per_group"] if r["prompt"] == "big"][0]
            small = [r for r in p["per_group"] if r["prompt"] == "small"][0]
            self.assertEqual(big["verdict"], "rl")
            # 最近 4 条 = 2 pos + 2 neg（前 2 条 neg 被裁掉）
            self.assertEqual(len(p["rl"]), 4)
            self.assertEqual(sum(1 for r in p["rl"] if r["prompt"] == "big"), 4)
            self.assertEqual(small["verdict"], "rl")          # 判档仍按全组
            self.assertEqual(small.get("rl_skip"), "only_2_of_4")   # 但不导出


class TestStats(unittest.TestCase):
    def test_counts_three_states(self):
        with tempfile.TemporaryDirectory() as td:
            rows = [{"prompt": "A", "completion": "a", "verdict": "pos"},
                    {"prompt": "A", "completion": "b", "verdict": "neg"},
                    {"prompt": "B", "completion": "c", "verdict": "unknown"},
                    {"prompt": "B", "completion": "d", "reward": 1.0}]
            st = P.Pool(_write_jsonl(_p.Path(td) / "p.jsonl", rows)).stats()
            self.assertEqual((st["samples"], st["prompts"]), (4, 2))
            self.assertEqual((st["pos"], st["neg"], st["unknown"]), (2, 1, 1))


if __name__ == "__main__":
    unittest.main()
