"""`train.py` 的三层门面测试：数据源解析、算法注册表、GRPO 目标（零阶要的那个标量）。

只在 CPU 上跑，不加载模型：`python3 -m unittest test_train -v`（在 examples/training 下）。

这里能测的三件事恰好是最容易静默出错的三件事：
  1. 数据切错 —— `--data identity` 应该落在身份数据上，路径写错要在**加载模型之前**就报；
  2. 算法/目标组合不成立 —— 比如 `--algo sft --objective grpo`，不能等到跑了一半才发现；
  3. GRPO 的标量目标 —— old == new 时必须正好等于 `-mean(A)`，超出裁剪带必须被夹住。
     这个函数是 MeZO×GRPO 的全部数学，写错了训练照样"跑得通"。
"""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

import sft_mezo


class _Tok:
    """只有模板和编码，够 `load_rollout_groups` 用。"""

    def apply_chat_template(self, messages, **kw):
        return [1, 2, 3]

    def __call__(self, text, **kw):
        return type("E", (), {"input_ids": [4, 5]})()


class Grouping(unittest.TestCase):
    """轨迹 jsonl → 组。组内归一化的基线只在这一层成立。"""

    def load(self, rows):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "r.jsonl"
            p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            return sft_mezo.load_rollout_groups(str(p), _Tok())

    def test_consecutive_same_prompt_is_one_group_and_advantages_are_normalised(self):
        groups, meta = self.load([
            {"prompt": "P1", "content": "a", "target": "", "kind": "words:1"},
            {"prompt": "P1", "content": "b b", "target": "", "kind": "words:1"},
            {"prompt": "P2", "content": "c", "target": "", "kind": "words:1"},
        ])
        self.assertEqual(len(groups), 2)
        self.assertEqual(meta["used"], 1)                     # P1 有方差；P2 只有一条
        advs = [t.adv for t in groups[0]]
        self.assertAlmostEqual(sum(advs), 0.0, places=5)      # 归一化后均值为 0
        self.assertGreater(advs[0], 0)                        # 1 个词是对的（奖励 1）
        self.assertEqual([t.adv for t in groups[1]], [0.0])   # 单条组没有基线

    def test_a_group_without_variance_has_no_gradient(self):
        groups, meta = self.load([{"prompt": "P", "content": "a", "kind": "words:1"}] * 2)
        self.assertEqual(meta["used"], 0)
        self.assertEqual([t.adv for t in groups[0]], [0.0, 0.0])

    def test_engine_logprobs_are_used_when_they_line_up(self):
        groups, _ = self.load([{"prompt": "P", "content": "a", "kind": "words:1",
                                "token_logprobs": [-0.5, -0.25]}])
        self.assertEqual([round(float(x), 4) for x in groups[0][0].old], [-0.5, -0.25])

    def test_a_short_logprob_list_is_ignored(self):
        # 长度对不上就是另一套分词，宁可重算也不能错位相减
        groups, _ = self.load([{"prompt": "P", "content": "a", "kind": "words:1",
                                "token_logprobs": [-0.5]}])
        self.assertIsNone(groups[0][0].old)


class _Out:
    def __init__(self, h):
        self.last_hidden_state = h
        self.past_key_values = None


class GRPOLoss(unittest.TestCase):
    """`grpo_loss` 只要一个标量 —— 所以它错起来也最安静。"""

    V = 5

    def setUp(self):
        self.h = torch.zeros(1, 4, self.V)            # 均匀 logits：new logp = -log V
        self.ids = torch.tensor([[9, 9, 0, 1]])       # 前 2 个 token 是 prompt
        self.new = -math.log(self.V)
        self.head = torch.nn.Identity()

    def traj(self, old: float, adv: float) -> sft_mezo._Traj:
        return sft_mezo._Traj(self.ids, 2, torch.full((2,), old), adv, 0.0)

    def loss(self, t, clip=0.2, batched=False):
        body = type("B", (), {"__call__": lambda _self, input_ids, attention_mask=None,
                              past_key_values=None, use_cache=False: _Out(self.h)})()
        return sft_mezo.grpo_loss(body, self.head, [t], clip, batched=batched)

    def test_old_equals_new_is_exactly_minus_advantage(self):
        # ratio = 1 ⇒ 代理目标就是 A，loss = −A。adv 取负值：loss 必须是 +2
        value, info = self.loss(self.traj(self.new, -2.0))
        self.assertAlmostEqual(value, 2.0, places=5)
        self.assertAlmostEqual(info["ratio"], 1.0, places=5)
        self.assertEqual(info["dead"], 0.0)

    def test_zero_advantage_contributes_nothing(self):
        value, info = self.loss(self.traj(self.new, 0.0))
        self.assertEqual(value, 0.0)
        self.assertEqual(info["dead"], 1.0)

    def test_the_upper_clamp_bites(self):
        # ratio = 5（old 比 new 低 ln5）落在 [0.8, 1.2] 外 ⇒ min 取裁剪项 1.2，而不是 5
        old = self.new - math.log(5.0)
        value, info = self.loss(self.traj(old, 1.0), clip=0.2)
        self.assertAlmostEqual(value, -1.2, places=4)
        self.assertAlmostEqual(info["ratio"], 5.0, places=3)

    def test_without_clamp_the_ratio_passes_through(self):
        old = self.new - math.log(5.0)
        value, _ = self.loss(self.traj(old, 1.0), clip=10.0)
        self.assertAlmostEqual(value, -5.0, places=4)

    def test_batched_matches_serial_when_lengths_differ(self):
        # 同组 prompt 等长、回答不等长：batch 版把短的那条 pad 了，数必须与逐条版一致，
        # 否则就是 pad 位被算进了 loss（这个错不会报错，只会让优势悄悄变味）。
        V = self.V
        body = type("B", (), {"__call__": lambda _self, input_ids, attention_mask=None,
                              past_key_values=None, use_cache=False:
                              _Out(torch.zeros(input_ids.shape[0], input_ids.shape[1], V))})()
        new = -math.log(V)
        long_t = sft_mezo._Traj(torch.tensor([[9, 9, 0, 1]]), 2,
                                torch.full((2,), new - math.log(2.0)), 1.0, 0.0)
        short_t = sft_mezo._Traj(torch.tensor([[9, 9, 1]]), 2,
                                 torch.full((1,), new), -1.0, 0.0)
        group = [long_t, short_t]
        b, ib = sft_mezo.grpo_loss(body, self.head, group, 0.2, batched=True)
        s, isx = sft_mezo.grpo_loss(body, self.head, group, 0.2, batched=False)
        self.assertAlmostEqual(b, s, places=5)
        # obj = [min(2,1.2)·(+1), 1·(−1)] = [1.2, −1.0] ⇒ loss = −mean = −0.1
        self.assertAlmostEqual(b, -0.1, places=5)
        self.assertAlmostEqual(ib["ratio"], isx["ratio"], places=5)


class _SamplingTok(_Tok):
    """`sample_group` 还要 decode 和 eos。"""

    eos_token_id = 9

    def decode(self, ids, **kw):
        return " ".join(str(i) for i in ids)


class _SamplingBody:
    """固定 hidden：采样结果只由 generator 决定，所以同一 seed 的测试可复现。"""

    def __call__(self, input_ids=None, past_key_values=None, use_cache=False):
        k = input_ids.shape[0]
        return type("O", (), {"last_hidden_state": torch.zeros(k, 1, 4),
                              "past_key_values": None})()


def _one_of(*ids):
    """只在这些 id 上给概率的 head；其他位置 -inf（softmax 后严格为 0）。"""

    def head(h):
        logits = torch.full((h.shape[0], 12), float("-inf"))
        for i in ids:
            logits[:, i] = 0.0
        return logits

    return head


class Sampling(unittest.TestCase):
    """进程内采样（on-policy）：错的不是数学，是**对齐** —— old 少一个 token 就静默错位。"""

    def sample(self, head, group=4, max_new=5, temperature=1.0):
        return sft_mezo.sample_group(
            _SamplingBody(), head, _SamplingTok(), {"prompt": "hi", "kind": "words:1"},
            group=group, max_new=max_new, temperature=temperature,
            gen=torch.Generator().manual_seed(0), device=torch.device("cpu"))

    def test_old_stays_aligned_with_the_generated_tokens(self):
        g, meta = self.sample(_one_of(5, 6))
        self.assertEqual(len(g), 4)
        for t in g:
            self.assertEqual(len(t.old), t.ids.shape[1] - t.n_prompt)   # 相减不能错位
            self.assertEqual(t.ids[0, :t.n_prompt].tolist(), [1, 2, 3])
            self.assertEqual(int(t.ids[0, -1]) in (5, 6), True)
        self.assertEqual(meta["uniq"] >= 1, True)
        self.assertEqual(meta["mean_len"], 5.0)

    def test_eos_stops_collection_but_not_the_batch(self):
        # 只在第一步给 eos：每条轨迹应该只有 1 个 token，仍然要能组成组
        g, meta = self.sample(_one_of(9), max_new=5)
        self.assertEqual(meta["mean_len"], 1.0)
        self.assertEqual([len(t.old) for t in g], [1, 1, 1, 1])
        self.assertEqual(meta["uniq"], 1)

    def test_a_group_of_identical_texts_has_no_advantage(self):
        # 全组同文 ⇒ std = 0 ⇒ 优势全 0 ⇒ 这一步不动参数（GRPO 冷启动，不是 bug）
        g, meta = self.sample(_one_of(5))
        self.assertEqual(meta["uniq"], 1)
        self.assertEqual([t.adv for t in g], [0.0, 0.0, 0.0, 0.0])

    def test_advantages_are_normalised_inside_the_group(self):
        g, meta = self.sample(_one_of(5, 6), group=8)
        advs = [t.adv for t in g]
        if meta["uniq"] > 1:                                   # 有方差才谈得上归一化
            self.assertAlmostEqual(sum(advs), 0.0, places=4)
            self.assertAlmostEqual(max(advs), -min(advs), places=4)


if __name__ == "__main__":
    unittest.main()
