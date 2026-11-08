"""`coldstart.py` 的路由与冷启配对。

    python -m unittest test_coldstart

要守住的四件事（每一条都对应一次真实踩坑）：
  1. 全对 ⇒ graduated、全错 ⇒ cold、有方差 ⇒ rl。边界用 reward≥0.5 判，连续奖励同一条线。
  2. **过期标签不能进冷启。** 实测反例：gold=27000，模型写 `27{,}000`、`\\$27,000`，
     旧判分器记 0 —— 若按「0 正就冷启」处理，DPO 会拿标准答案去纠正一个已经正确的行为。
     `label_suspect` 必须把这个组挑出来，`plan` 必须把它标成 suspect_label 而不是 cold。
  3. **gold 只是答案串时不能当 DPO 的 chosen。** 数学题 gold="27000"、错例 300 token，
     两侧平均粒度不同，margin 没有意义。必须返回 None + 原因，不许静默退化成 SFT。
  4. rejected 取「完整但答错里最长的一条」，不取截断的、不取最短的。

本文件不 import torch：冷启判定与 GPU 无关，要在 no-torch 档里也跑得通。
"""

from __future__ import annotations

import pathlib as _p
import sys as _s
for _d in (_p.Path(__file__).resolve().parent, _p.Path(__file__).resolve().parents[1]):
    if str(_d) not in _s.path:
        _s.path.insert(0, str(_d))

import unittest  # noqa: E402

import coldstart as C  # noqa: E402


def row(prompt: str, completion: str, reward: float, gold: str = "",
        truncated: bool = False) -> dict:
    return {"prompt": prompt, "completion": completion, "reward": reward,
            "gold": gold, "truncated": truncated}


# ---- 真实数据（`rollouts_gsm.jsonl` 的原样片段，组 3 / 组 12）
G3_TAIL_OK = ("[ \\[ 30{,}000 \\times 0.90 = 27{,}000 \\] "
              "**Jean paid \\$27,000.**")
G12_WRONG = "Each plant costs \\$6, so: 28 \\times 6 = 168. **Pat will spend \\$168.**"
GOLD_3, GOLD_12 = "27000", "342"


class TestClassify(unittest.TestCase):
    def test_four_routes(self):
        self.assertEqual(C.classify([1, 1, 1, 1]), C.GRADUATED)
        self.assertEqual(C.classify([0, 0, 0, 0]), C.COLD)
        self.assertEqual(C.classify([1, 0, 0, 0]), C.RL)
        self.assertEqual(C.classify([1, 1, 1, 0]), C.RL)

    def test_continuous_reward_uses_half_threshold(self):
        # 稠密奖励（身份题的 F1）走同一条线：0.6 算对，0.3 算错
        self.assertEqual(C.classify([0.6, 0.6, 0.6, 0.6]), C.GRADUATED)
        self.assertEqual(C.classify([0.3, 0.3, 0.3, 0.3]), C.COLD)
        self.assertEqual(C.classify([0.6, 0.3, 0.3, 0.3]), C.RL)

    def test_all_truncated_is_not_cold(self):
        # 全错 + 有截断 ⇒ 预算问题，不是「模型不会」。过半截断单独一档，
        # 少量截断也要拦（那条被腰斩的样本可能本来是唯一会做的）。
        self.assertEqual(C.classify([0, 0, 0, 0], [True, True, False, False]), C.TRUNCATED)
        self.assertEqual(C.classify([0, 0, 0, 0], [True, False, False, False]),
                         C.NEEDS_BUDGET)
        self.assertEqual(C.classify([0, 0, 0, 0], [False] * 4), C.COLD)

    def test_truncated_sample_is_unknown_not_wrong(self):
        # 2026-11-05 实测：MATH-500 1536 预算下 5 条错例**全部**是截断；提到 6144
        # 后其中 3 条答对（79.2% → 91.7%）。所以「有截断 + 有错例」的组不能判 RL：
        # 那条错例可能只是被腰斩，把它当负例就是教模型「别说下去」。
        self.assertEqual(C.classify([1, 1, 1, 1], [True] * 4), C.GRADUATED)
        self.assertEqual(C.classify([1, 0, 0, 0], [True] * 4), C.TRUNCATED)
        self.assertEqual(C.classify([1, 0, 0, 0], [True, False, False, False]),
                         C.NEEDS_BUDGET)
        # 没有任何截断时，原来的判据不变
        self.assertEqual(C.classify([1, 0, 0, 0]), C.RL)


class TestLabelSuspect(unittest.TestCase):
    def test_number_normalization(self):
        self.assertEqual(C.normalize_number("27{,}000"), "27000")
        self.assertEqual(C.normalize_number("\\$27,000"), "27000")
        self.assertEqual(C.normalize_number("$ 27.5"), "27.5")
        # 中文/全角写法
        self.assertEqual(C.normalize_number("１８"), "１８")

    def test_group3_false_negative_is_caught(self):
        # gold 的数字出现在回答尾部，但 reward 记了 0
        self.assertTrue(C.label_suspect(G3_TAIL_OK, GOLD_3))

    def test_group12_real_wrong_is_not_flagged(self):
        self.assertFalse(C.label_suspect(G12_WRONG, GOLD_12))

    def test_number_in_middle_only_is_not_flagged(self):
        # 中间步骤出现 gold 不算：先算 27000，最后又改了
        txt = "First 30000*0.9 = 27000 but the tax changes it. Final answer is 29700."
        self.assertFalse(C.label_suspect(txt, GOLD_3))

    def test_empty_gold_never_flags(self):
        self.assertFalse(C.label_suspect(G3_TAIL_OK, ""))


class TestBuildCold(unittest.TestCase):
    def test_gold_answer_only_is_refused(self):
        # 组 12：真冷启，但 gold 只有 "342"，没有完整解答 ⇒ 不能造 chosen
        samples = [{"completion": G12_WRONG, "reward": 0.0},
                   {"completion": G12_WRONG + " " * 50, "reward": 0.0}]
        item, action, why = C.build_cold("Pat plants", samples, GOLD_12)
        self.assertIsNone(item)
        self.assertEqual(action, "")
        self.assertIn("gold_not_a_trajectory", why)

    def test_gold_solution_enables_dpo(self):
        sol = "45 - 17 = 28; 28 \\times 6 = 168 " + "x" * 40   # 与错例同量级
        samples = [{"completion": G12_WRONG, "reward": 0.0}]
        item, action, why = C.build_cold("Pat plants", samples, GOLD_12, gold_solution=sol)
        self.assertEqual((action, why), ("dpo", ""))
        self.assertEqual(item["chosen"], sol)
        self.assertEqual(item["rejected"], G12_WRONG)
        self.assertEqual(item["_chosen_src"], "gold_solution")

    def test_short_reference_falls_back_to_sft(self):
        # 真实 GSM8K 形状：参考答案只有模型 CoT 的 1/8 ⇒ 负例被稀释 8 倍，
        # DPO 退化成更慢的 SFT，应当直接返回 sft 动作
        sol = "She pays 90% of 30000, so 27000."
        samples = [{"completion": "reasoning step " * 60, "reward": 0.0}]
        item, action, why = C.build_cold("q", samples, GOLD_3, gold_solution=sol)
        self.assertEqual(action, "sft")
        self.assertIn("negative_diluted", why)
        self.assertEqual(item["target"], sol)

    def test_chosen_much_longer_is_refused(self):
        # 反向不对称：rejected 是短拒答（"I don't know"），惩罚它没有教学价值
        samples = [{"completion": "I don't know.", "reward": 0.0}]
        item, action, why = C.build_cold("q", samples, "1", gold_solution="long solution " * 50)
        self.assertIsNone(item)
        self.assertIn("chosen_much_longer", why)

    def test_onpolicy_correct_preferred_over_gold(self):
        # 有正例时用模型自己的（on-policy），且取最短的那条
        long_ok = "correct " * 40
        short_ok = "correct " * 32
        wrong_pad = G12_WRONG + " " + "wrong tail " * 15      # 长度与 chosen 同量级
        samples = [{"completion": long_ok, "reward": 1.0},
                   {"completion": short_ok, "reward": 1.0},
                   {"completion": wrong_pad, "reward": 0.0}]
        item, action, why = C.build_cold("q", samples, GOLD_12, gold_solution="gold " * 40)
        self.assertEqual(action, "dpo")
        self.assertEqual(item["chosen"], short_ok)
        self.assertEqual(item["_chosen_src"], "on_policy")

    def test_rejected_prefers_longest_complete_wrong(self):
        long_wrong = "wrong path " * 12
        truncated_wrong = "truncated " * 40          # 更长，但被截断
        samples = [{"completion": "no idea", "reward": 0.0},
                   {"completion": long_wrong, "reward": 0.0},
                   {"completion": truncated_wrong, "reward": 0.0, "truncated": True}]
        item, action, why = C.build_cold("q", samples, "1", gold_solution="right path " * 12)
        self.assertEqual((action, why), ("dpo", ""))
        self.assertEqual(item["rejected"], long_wrong)

    def test_mostly_truncated_refused(self):
        samples = [{"completion": "abc", "reward": 0.0, "truncated": True},
                   {"completion": "abcd", "reward": 0.0, "truncated": True}]
        item, action, why = C.build_cold("q", samples, "1")
        self.assertIsNone(item)
        self.assertEqual(why, "mostly_truncated")

    def test_bare_answer_gold_solution_is_refused(self):
        """2026-11-06：`_pick` 的 `gold_not_a_trajectory` 之前是死分支。

        误传 `gold_solution="520"` 会静默 SFT 到纯数字；`Answer: 520` 的形状同样要挡。
        """
        samples = [{"completion": G12_WRONG, "reward": 0.0}]
        for bare in ("520", "Answer: 520", "The answer is 520."):
            item, action, why = C.build_cold("q", samples, "520", gold_solution=bare)
            self.assertIsNone(item, bare)
            self.assertEqual(action, "", bare)
            self.assertIn("gold_not_a_trajectory", why, bare)
            self.assertIn("bare_answer", why, bare)

    def test_long_wrong_trajectory_does_not_disqualify_a_short_solution(self):
        """AIMO3 实测形状：错例 33 万字符、官方解答 9654 字符（比值 0.03）。

        守卫不能拿「÷ 错例长度中位数」当判据，否则合法冷启数据被一起杀掉；
        这里应正常配对（长度比 < 1/3 时按设计退成 SFT，而不是拒绝）。
        """
        wrong = ("we enumerate the cases " * 200) + "\nAnswer: 79"
        sol = ("We split into three cases. Case 1 gives 12, case 2 gives 34, "
               "case 3 gives 56; multiplying and reducing modulo 99991 yields "
               "the value below.\n\nAnswer: 8687")
        item, action, why = C.build_cold("q", [{"completion": wrong, "reward": 0.0}],
                                         "8687", gold_solution=sol)
        self.assertEqual(action, "sft")            # 负例被稀释 ⇒ 走 SFT，不是「拒绝冷启」
        self.assertIn("negative_diluted", why)
        self.assertEqual(item["target"], sol)


class TestPlan(unittest.TestCase):
    def _rows(self) -> list[dict]:
        rows = []
        # 组 0：全对（毕业）
        for _ in range(4):
            rows.append(row("easy", "correct work " * 20, 1.0, gold="18"))
        # 组 1：有方差（RL）
        for w in (1.0, 1.0, 0.0, 0.0):
            rows.append(row("mixed", "some work " * 20, w, gold="18"))
        # 组 2：假冷启（gold 数字在尾部但记 0）—— 组 3 的真实形状
        for _ in range(4):
            rows.append(row("fake-cold", G3_TAIL_OK, 0.0, gold=GOLD_3))
        # 组 3：真冷启（真答错）
        for _ in range(4):
            rows.append(row("true-cold", G12_WRONG, 0.0, gold=GOLD_12))
        return rows

    def test_split_counts_and_rl_rows(self):
        p = C.plan(self._rows(), group_size=4)
        self.assertEqual(len(p["rl"]), 4)              # 只有组 1 的 4 条
        self.assertEqual(len(p["graduated"]), 1)
        self.assertEqual(len(p["suspect"]), 1)         # 假冷启被挑出
        verdicts = {r["prompt"]: r["verdict"] for r in p["per_group"]}
        self.assertEqual(verdicts["easy"], C.GRADUATED)
        self.assertEqual(verdicts["mixed"], C.RL)
        self.assertEqual(verdicts["fake-cold"], "suspect_label")
        self.assertEqual(verdicts["true-cold"], C.COLD)

    def test_fake_cold_never_reaches_cold_pool(self):
        p = C.plan(self._rows(), group_size=4)
        cold_prompts = {r["prompt"] for r in p["cold"]}
        self.assertNotIn("fake-cold", cold_prompts)

    def test_true_cold_is_refused_for_dpo_without_solution(self):
        p = C.plan(self._rows(), group_size=4)
        self.assertEqual(p["dpo_pairs"], [])                       # 没标准解答 ⇒ 一对也不造
        self.assertEqual([r["reason"][:20] for r in p["rejected_pairs"]],
                         ["gold_not_a_trajector"])

    def test_rescore_turns_fake_cold_into_graduated(self):
        # 给一个「当前判分器」：看到 27{,}000 就判对 —— 这正是修完 extract_pred 后的行为
        def rescore(completion: str, gold: str) -> float:
            return 1.0 if (gold == GOLD_3 and "27" in completion) else 0.0

        p = C.plan(self._rows(), group_size=4, rescore=rescore)
        verdicts = {r["prompt"]: r["verdict"] for r in p["per_group"]}
        self.assertEqual(verdicts["fake-cold"], C.GRADUATED)
        self.assertEqual(p["suspect"], [])            # 重判之后不再可疑

    def test_gold_solution_produces_pair(self):
        sol = "45 - 17 = 28; 28 \\times 6 = 168 " + "y" * 40
        p = C.plan(self._rows(), group_size=4, gold_solutions={"true-cold": sol})
        self.assertEqual(len(p["dpo_pairs"]), 1)
        self.assertEqual(p["dpo_pairs"][0]["prompt"], "true-cold")

    def test_summary_mentions_every_bucket(self):
        s = C.summarize(C.plan(self._rows(), group_size=4))
        for key in ("RL", "冷启", "毕业", "截断", "标签可疑"):
            self.assertIn(key, s)

    def test_rescore_is_written_back_into_rl_rows(self):
        # 历史标签全 0（过期），当前判分器判成 1/0/1/0 ⇒ 该组进 RL，
        # 而 rl.jsonl 里的 reward 必须是**重判后**的值。写回原值 = 训练直接吃错的分，
        # 而且错得很隐蔽（组内优势的符号会整体翻转）。
        rows = [row("q", f"c{i}", 0.0) for i in range(4)]

        def rescore(completion: str, gold: str) -> float:
            return 1.0 if completion in ("c0", "c2") else 0.0

        p = C.plan(rows, group_size=4, rescore=rescore)
        self.assertEqual(p["rl_groups"], 1)
        self.assertEqual([r["reward"] for r in p["rl"]], [1.0, 0.0, 1.0, 0.0])

    def _suspect_but_wrong(self) -> tuple[list[dict], str, str]:
        """AIMO3 误报形状：gold=8687 作为中间运算数 18,687 出现在尾部，末行答 79。"""
        prompt = "hard-problem"
        sol = ("We split into three cases. Case 1 gives 12, case 2 gives 34, "
               "case 3 gives 56; multiplying and reducing modulo 99991 yields "
               "the value below.\n\nAnswer: 8687")
        comp = (("we set up the variables " * 20)
                + ("more derivation " * 40)
                + "the intermediate count is 18,687 and we finally conclude "
                + "\nAnswer: 79")
        rows = [row(prompt, comp + f" #{i}", 0.0, gold="8687") for i in range(2)]
        # 造数据时先确认这个启发式确实会命中，否则这条用例测的是空集
        self.assertTrue(C.label_suspect(comp, "8687"))
        return rows, prompt, sol

    def test_rescored_suspect_does_not_block_cold_start(self):
        """2026-11-06 AIMO3 误报：4 组 suspect 里 2 个真冷启组被踢掉。

        已用当前判分器逐条重判过（这里明确判 0 = 真错）时，label_suspect 只能报告，
        不能再拿启发式覆盖精确判分。冷启必须照常配出。
        """
        rows, prompt, sol = self._suspect_but_wrong()
        p = C.plan(rows, group_size=2, gold_solutions={prompt: sol},
                   rescore=lambda completion, gold: 0.0)
        verdicts = {r["prompt"]: r["verdict"] for r in p["per_group"]}
        self.assertEqual(verdicts[prompt], C.COLD)
        self.assertEqual(len(p["cold"]), 1)
        self.assertEqual(len(p["sft_items"]), 1)
        self.assertEqual(p["sft_items"][0]["target"], sol)
        self.assertEqual(len(p["suspect"]), 1)
        self.assertTrue(p["suspect"][0]["verified"])
        self.assertIn("已重判/仅提示", C.summarize(p))

    def test_incomplete_rescore_keeps_the_old_guard(self):
        """重判只要有一条返回 None ⇒ 该组标签仍可能过期，老规矩照旧挡冷启。"""
        rows, prompt, sol = self._suspect_but_wrong()

        def rescore(completion: str, gold: str):
            return None if "#0" in completion else 0.0

        p = C.plan(rows, group_size=2, gold_solutions={prompt: sol}, rescore=rescore)
        verdicts = {r["prompt"]: r["verdict"] for r in p["per_group"]}
        self.assertEqual(verdicts[prompt], "suspect_label")
        self.assertFalse(p["suspect"][0]["verified"])
        self.assertEqual(p["cold"], [])

    def test_unrescored_suspect_still_blocks_cold_start(self):
        """没给 rescore 时原行为不变：可疑冷启组不能进池子。"""
        rows, prompt, sol = self._suspect_but_wrong()
        p = C.plan(rows, group_size=2, gold_solutions={prompt: sol})
        verdicts = {r["prompt"]: r["verdict"] for r in p["per_group"]}
        self.assertEqual(verdicts[prompt], "suspect_label")
        self.assertFalse(p["suspect"][0]["verified"])


class TestPlanFiles(unittest.TestCase):
    """产物要能**直接喂**训练脚本：列名不对，训练端会在读数据时才炸。"""

    def _write(self, d: _p.Path) -> str:
        rows = []
        for w in (1.0, 1.0, 0.0, 0.0):
            rows.append(row("mixed", "some work " * 20, w, gold="18"))
        for _ in range(4):
            rows.append(row("true-cold", G12_WRONG, 0.0, gold=GOLD_12))
        p = d / "rollouts.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(__import__("json").dumps(r, ensure_ascii=False) + "\n")
        return str(p)

    def test_plan_file_writes_route_plan_and_feedable_files(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            d = _p.Path(td)
            src = self._write(d)
            sol = "45 - 17 = 28; 28 \\times 6 = 168 " + "y" * 40
            with open(d / "sol.jsonl", "w", encoding="utf-8") as f:
                f.write(json.dumps({"prompt": "true-cold", "solution": sol},
                                   ensure_ascii=False) + "\n")
            p, paths, steps = C.plan_file(src, 4, out_dir=d / "out",
                                          gold_solutions=str(d / "sol.jsonl"),
                                          dpo_epochs=7)
            self.assertEqual(len(p["rl"]), 4)
            with open(paths["rl"], encoding="utf-8") as f:
                line = json.loads(f.readline())
            self.assertEqual(sorted(line), ["completion", "prompt", "reward"])
            # 冷启在前、RL 在后：冷启不先跑，下一轮 RL 那一格还是 0 正
            self.assertEqual([s["name"] for s in steps], ["cold_dpo", "rl_luspo"])
            # 冷启 DPO 的步数必须由轮次折算：少了 --epochs，train.py 会报「没有实测轮次」
            dpo = next(s for s in steps if s["name"] == "cold_dpo")
            self.assertIn("--epochs", dpo["cmd"])
            self.assertEqual(dpo["cmd"][dpo["cmd"].index("--epochs") + 1], "7")
            self.assertTrue(_p.Path(paths["route"]).is_file())
            plan = json.loads(_p.Path(paths["route"]).read_text(encoding="utf-8"))
            self.assertEqual(len(plan["steps"]), 2)
            for s in plan["steps"]:
                self.assertEqual(s["on_fail"], "continue")   # 一格失败不废整轮


class TestThinkTail(unittest.TestCase):
    """冷启 target 的结构规范化：与采样 completion 同构，不改数学内容。

    2026-11-06 实测：AIMO3 low 档采样 completion 20/20 含 `</think>`，官方
    解答 0/10 含。不统一收尾，训练学的是「思考段里直接写答案且不收尾」，
    训练后评测读到的是另一种分布。
    """

    SOL = ("We first show that the square can be divided into 520 regions.\n"
           "Counting the boundary segments gives 520 - 1 = 519 internal lines,\n"
           "and adding the outer boundary restores the total.\n"
           "Therefore the answer we report is 520.\n\nAnswer: 520")
    GOLD = "520"

    def test_official_solution_becomes_think_then_answer(self):
        out = C.normalize_think_tail(self.SOL, self.GOLD)
        self.assertEqual(out.count(C.THINK_CLOSE), 1)
        self.assertTrue(out.endswith("Answer: 520"))
        self.assertLess(out.rindex(C.THINK_CLOSE), out.rindex("Answer: 520"))
        # 推导正文逐字保留：只摘走末行答案，插一个收尾标签
        body = self.SOL[:self.SOL.rindex("Answer: 520")].rstrip()
        self.assertTrue(out.startswith(body))
        self.assertIn("520 - 1 = 519", out)
        r = C.check_trajectory_tail(out, self.GOLD)
        self.assertTrue(r["ok"], r)
        self.assertTrue(r["idempotent"])
        self.assertTrue(r["answer_value_unchanged"])

    def test_idempotent_no_double_close(self):
        once = C.normalize_think_tail(self.SOL, self.GOLD)
        twice = C.normalize_think_tail(once, self.GOLD)
        self.assertEqual(once, twice)
        self.assertEqual(twice.count(C.THINK_CLOSE), 1)

    def test_existing_close_keeps_its_tail_verbatim(self):
        # 采样式轨迹：`</think>` 之后还有一段最终回答，不删内容、只校验
        tail = "We need to compute the total.\n</think>\n\n## Solution\n\nAnswer: 520"
        self.assertEqual(C.normalize_think_tail(tail, self.GOLD), tail)

    def test_answer_mismatch_is_refused_not_guessed(self):
        for bad in (self.SOL.replace("Answer: 520", "Answer: 519"),
                    "work " * 30 + "\n</think>\n\nAnswer: 519"):
            with self.assertRaisesRegex(ValueError, "不一致"):
                C.normalize_think_tail(bad, self.GOLD)

    def test_missing_answer_or_body_is_refused(self):
        with self.assertRaisesRegex(ValueError, "找不到末行 Answer"):
            C.normalize_think_tail("a long derivation with no final line", self.GOLD)
        with self.assertRaisesRegex(ValueError, "没有 Answer"):
            C.normalize_think_tail("x\n</think>\n\nno answer here", self.GOLD)
        with self.assertRaisesRegex(ValueError, "没有推导正文"):
            C.normalize_think_tail("Answer: 520", self.GOLD)

    def test_multiple_close_or_empty_gold_is_refused(self):
        with self.assertRaisesRegex(ValueError, "2 </think>"):
            C.normalize_think_tail("x</think>y</think>\nAnswer: 520", self.GOLD)
        with self.assertRaisesRegex(ValueError, "非空 gold"):
            C.normalize_think_tail(self.SOL, "")

    def test_build_cold_default_is_unchanged_and_opt_in_normalizes(self):
        sol = self.SOL + " " + "extra derivation " * 20
        rejected = "A wrong but complete attempt. " * 20 + "\nAnswer: 519"
        samples = [{"completion": rejected, "reward": 0.0, "truncated": False},
                   {"completion": rejected + " still wrong", "reward": 0.0,
                    "truncated": False}]
        legacy, action0, _ = C.build_cold("q", samples, self.GOLD, sol)
        fixed, action1, _ = C.build_cold("q", samples, self.GOLD, sol,
                                         think_tail=True)
        self.assertEqual(action0, action1)     # 加收尾不改变 DPO/SFT 的判定
        key = "chosen" if action0 == "dpo" else "target"
        self.assertNotIn(C.THINK_CLOSE, legacy[key])
        self.assertEqual(fixed[key].count(C.THINK_CLOSE), 1)
        self.assertTrue(C.check_trajectory_tail(fixed[key], self.GOLD)["ok"])

    def test_plan_level_all_cold_targets_pass_contract(self):
        sol = self.SOL + " " + "extra derivation " * 20
        wrong = "A wrong but complete attempt. " * 20 + "\nAnswer: 17"
        rows = [row("q", wrong, 0.0, gold=self.GOLD) for _ in range(2)]
        p = C.plan(rows, 2, gold_solutions={"q": sol}, think_tail=True)
        targets = ([r["chosen"] for r in p["dpo_pairs"]]
                   + [r["target"] for r in p["sft_items"]])
        self.assertEqual(len(targets), 1)
        for t in targets:
            self.assertTrue(C.check_trajectory_tail(t, self.GOLD)["ok"])
        # 默认 False 时逐字回到历史行为，不悄悄改数据
        p0 = C.plan(rows, 2, gold_solutions={"q": sol})
        t0 = ([r["chosen"] for r in p0["dpo_pairs"]]
              + [r["target"] for r in p0["sft_items"]])[0]
        self.assertNotIn(C.THINK_CLOSE, t0)
        self.assertEqual(C.normalize_think_tail(t0, self.GOLD), targets[0])


class TestRouteStepsChain(unittest.TestCase):
    """冷启两步必须串成一个 adapter；默认行为保持向后兼容。

    不串联时三步各写各的默认 --out，最终拿到三个互不相干的 LoRA，根本没法评估。
    RL 不能假装接过冷启权重（mezo theta 不回读 GGUF 的 A/B）。
    """

    PLAN = {"sft_items": [1], "dpo_pairs": [1], "rl": [1, 2]}
    PATHS = {"sft": "sft.jsonl", "dpo": "dpo.jsonl", "rl": "rl.jsonl"}

    def test_default_has_no_chain_flags(self):
        steps = C.route_steps(dict(self.PLAN), self.PATHS, group=2)
        self.assertEqual([s["name"] for s in steps],
                         ["cold_sft", "cold_dpo", "rl_luspo"])
        self.assertTrue(all("--out" not in s["cmd"] and
                            "--init-adapter" not in s["cmd"] for s in steps))

    def test_chain_links_sft_into_dpo(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            steps = C.route_steps(dict(self.PLAN), self.PATHS, group=2,
                                  chain_adapters=d, include_rl=False)
        self.assertEqual([s["name"] for s in steps], ["cold_sft", "cold_dpo"])
        sft, dpo = steps
        self.assertEqual(sft["adapter_out"], os.path.join(d, "cold_sft.pt"))
        self.assertEqual(sft["cmd"][-2:], ["--out", sft["adapter_out"]])
        self.assertEqual(dpo["adapter_init"], sft["adapter_out"])
        self.assertIn("--init-adapter", dpo["cmd"])
        self.assertEqual(dpo["adapter_out"], os.path.join(d, "cold_dpo.pt"))
        self.assertEqual(dpo["cmd"][-2:], ["--out", dpo["adapter_out"]])

    def test_dpo_without_sft_does_not_point_at_missing_file(self):
        import tempfile

        p = {"sft_items": [], "dpo_pairs": [1], "rl": []}
        with tempfile.TemporaryDirectory() as d:
            steps = C.route_steps(p, self.PATHS, chain_adapters=d)
        self.assertEqual(len(steps), 1)
        self.assertNotIn("--init-adapter", steps[0]["cmd"])
        self.assertIn("--out", steps[0]["cmd"])

    def test_rl_step_is_explicitly_marked_unbridged(self):
        steps = C.route_steps(dict(self.PLAN), self.PATHS, group=2)
        self.assertIn("note", steps[-1])
        self.assertIn("init_theta", steps[-1]["note"])


class TestRouteStepsEngineContract(unittest.TestCase):
    """回灌引擎的三个硬约束：落点、ckpt、DPO 步数。

    2026-11-06 三个缺口（都会让「训练出的东西」和「评测的东西」不是一回事）：
      1. 默认 --targets 是 12 类，而 lora_to_gguf 只写 o_proj/out_proj ⇒ 导出静默丢
         10 类，评测的是训练结果的一小片；
      2. 关 ckpt 时 mem_guard 把专家 refetch 暂存 26 GiB 计入峰值，48 GB 卡上 T=2048
         都过不了预检（25.2 + 26 > 45.6）；
      3. backward 的 dpo 轮次没扫过，train.py 兜底 --steps 40 —— 在 1 对样本上等于过
         40 遍，lr 1e-4 早越过能力门。
    """

    PLAN = {"sft_items": [1], "dpo_pairs": [1], "rl": []}
    PATHS = {"sft": "sft.jsonl", "dpo": "dpo.jsonl", "rl": "rl.jsonl"}

    def _steps(self, **kw):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            return C.route_steps(dict(self.PLAN), self.PATHS, chain_adapters=d,
                                 include_rl=False, **kw)

    def test_engine_targets_narrow_both_steps(self):
        sft, dpo = self._steps(engine_targets="o_proj,out_proj")
        for s in (sft, dpo):
            i = s["cmd"].index("--targets")
            self.assertEqual(s["cmd"][i + 1], "o_proj,out_proj", s["name"])

    def test_default_targets_untouched(self):
        """不给 engine_targets 时不许偷偷改历史命令行。"""
        sft, dpo = self._steps()
        self.assertNotIn("--targets", sft["cmd"])
        self.assertNotIn("--targets", dpo["cmd"])

    def test_ckpt_on_sft_only(self):
        sft, dpo = self._steps()
        self.assertIn("--ckpt", sft["cmd"])
        self.assertNotIn("--ckpt", dpo["cmd"])     # dpo 那条由 train.py 自带

    def test_dpo_epochs_pinned(self):
        """DPO 必须按**轮次**给：固定 `--steps` 在偏好对数量变化时口径会漂。

        1 对 × 7 轮与 10 对 × 7 轮都该是 7 轮；写成 `--steps 7` 时后者连一轮都不到
        （2026-11-06 修的：那轮 7 步其实等于同一对样本过 7 遍）。
        """
        sft, dpo = self._steps(dpo_epochs=7)
        i = dpo["cmd"].index("--epochs")
        self.assertEqual(dpo["cmd"][i + 1], "7")
        self.assertNotIn("--steps", dpo["cmd"])    # 步数由 epochs × 条数算出
        self.assertNotIn("--steps", sft["cmd"])

    def test_dpo_epochs_absent_by_default(self):
        _, dpo = self._steps()
        self.assertNotIn("--epochs", dpo["cmd"])   # 默认交回 train.py 的轮次表
        self.assertNotIn("--steps", dpo["cmd"])


if __name__ == "__main__":
    unittest.main()
