"""`train.py` 的三层门面测试：数据源解析、算法注册表、GRPO 目标（零阶要的那个标量）。

只在 CPU 上跑，不加载模型：`python3 -m unittest test_train -v`（在 training 下）。

这里能测的三件事恰好是最容易静默出错的三件事：
  1. 数据切错 —— `--data identity` 应该落在身份数据上，路径写错要在**加载模型之前**就报；
  2. 算法/目标组合不成立 —— 比如 `--algo sft --objective grpo`，不能等到跑了一半才发现；
  3. GRPO 的标量目标 —— old == new 时必须正好等于 `-mean(A)`，超出裁剪带必须被夹住。
     这个函数是 MeZO×GRPO 的全部数学，写错了训练照样"跑得通"。
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
import math
import tempfile
import unittest
from pathlib import Path

import torch

import grpo
import guard
import mem_guard
import hf_ref


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
            return hf_ref.load_rollout_groups(str(p), _Tok())

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

    def traj(self, old: float, adv: float) -> hf_ref._Traj:
        return hf_ref._Traj(self.ids, 2, torch.full((2,), old), adv, 0.0)

    def loss(self, t, clip=0.2, batched=False):
        body = type("B", (), {"__call__": lambda _self, input_ids, attention_mask=None,
                              past_key_values=None, use_cache=False: _Out(self.h)})()
        return hf_ref.grpo_loss(body, self.head, [t], clip, batched=batched)

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
        long_t = hf_ref._Traj(torch.tensor([[9, 9, 0, 1]]), 2,
                                torch.full((2,), new - math.log(2.0)), 1.0, 0.0)
        short_t = hf_ref._Traj(torch.tensor([[9, 9, 1]]), 2,
                                 torch.full((1,), new), -1.0, 0.0)
        group = [long_t, short_t]
        b, ib = hf_ref.grpo_loss(body, self.head, group, 0.2, batched=True)
        s, isx = hf_ref.grpo_loss(body, self.head, group, 0.2, batched=False)
        self.assertAlmostEqual(b, s, places=5)
        # obj = [min(2,1.2)·(+1), 1·(−1)] = [1.2(2 tok), −1.0(1 tok)]
        #   默认 DAPO（token 平均）：(1.2·2 + −1·1)/3 = 0.4667 ⇒ loss −0.4667
        self.assertAlmostEqual(b, -0.4667, places=3)
        self.assertAlmostEqual(ib["ratio"], isx["ratio"], places=5)

        # 旧的长度归一化口径仍要能取回，且 batch/逐条两种写法必须给出同一个数
        from objectives import GrpoCfg
        cfg_seq = GrpoCfg(clip_low=0.2, clip_high=0.2, norm="seq")
        b2, _ = hf_ref.grpo_loss(body, self.head, group, 0.2, batched=True, cfg=cfg_seq)
        s2, _ = hf_ref.grpo_loss(body, self.head, group, 0.2, batched=False, cfg=cfg_seq)
        self.assertAlmostEqual(b2, -0.1, places=5)     # (1.2/2 + −1/1)/2 = 0.1
        self.assertAlmostEqual(b2, s2, places=5)


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
        return hf_ref.sample_group(
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


class NumReward(unittest.TestCase):
    """kind=num：数学题的可验证奖励。规则必须与 math_probe.py 的 extract_num 一致，
    否则探题说「基座会错」的题训练时全组 std=0 —— 静默空转。"""

    def test_first_integer_matches_target(self):
        self.assertEqual(grpo.reward_of({"kind": "num", "target": "1998",
                                         "content": "The answer is 1998."}), 1.0)
        self.assertEqual(grpo.reward_of({"kind": "num", "target": "1998",
                                         "content": "1998"}), 1.0)

    def test_commas_and_negatives_are_handled(self):
        self.assertEqual(grpo.reward_of({"kind": "num", "target": "-72",
                                         "content": "It is -72, I think."}), 1.0)
        self.assertEqual(grpo.reward_of({"kind": "num", "target": "1234567",
                                         "content": "1,234,567"}), 1.0)

    def test_wrong_or_missing_number_scores_zero(self):
        self.assertEqual(grpo.reward_of({"kind": "num", "target": "1998",
                                         "content": "The answer is 1989."}), 0.0)
        self.assertEqual(grpo.reward_of({"kind": "num", "target": "1998",
                                         "content": "I cannot compute that."}), 0.0)
        self.assertEqual(grpo.reward_of({"kind": "num", "target": "",
                                         "content": "1998"}), 0.0)

    def test_first_number_wins_not_the_last(self):
        # 「Reply with only the number」之外的冗话里常带别的数字，取第一个才是采样主体
        self.assertEqual(grpo.reward_of({"kind": "num", "target": "42",
                                         "content": "42 (which is 6 times 7)"}), 1.0)


class _StubLM:
    """确定性的 body/head：特征只依赖 (token id, 位置)，方便逐位比较两种裁剪。"""

    H, V = 8, 24

    def __init__(self):
        g = torch.Generator().manual_seed(0)
        self.W = torch.randn(self.H, self.V, generator=g)
        self.bias = torch.randn(self.V, generator=g)

    def body(self, input_ids, attention_mask=None, use_cache=False):
        ids = input_ids
        n, L = ids.shape
        pos = torch.arange(L).view(1, L, 1).float()
        h = torch.nn.functional.one_hot(ids % self.H, self.H).float()
        h = h * (1.0 + pos / 7.0)                       # 位置也进特征
        if attention_mask is not None:
            h = h * attention_mask.unsqueeze(-1).float()
        return type("O", (), {"last_hidden_state": h})

    def head(self, x):
        return x @ self.W + self.bias


class CompletionSums(unittest.TestCase):
    """`_completion_sums` 只算 completion 段 —— 必须与整段算逐位等价。

    这条不变量是 DPO 提速的全部依据：prompt 段的 logits 算完就被 mask 丢掉，
    但它占偏好对八成的 token，每位置都要重读 1.27 GB 的 lm_head。
    """

    def setUp(self):
        self.lm = _StubLM()

    def _reference(self, ids, att, plens):
        """旧实现：整段算 logits，再 mask。"""
        n, L = ids.shape
        h = self.lm.body(input_ids=ids, attention_mask=att).last_hidden_state
        lp = self.lm.head(h[:, :L - 1]).float()
        lp = (lp.gather(-1, ids[:, 1:][..., None]).squeeze(-1)
              - torch.logsumexp(lp, dim=-1))
        mask = torch.zeros_like(lp)
        for i, pl in enumerate(plens):
            m = int(att[i].sum())
            mask[i, pl - 1:m - 1] = 1.0
        return (lp * mask).sum(1)

    def _check(self, ids, att, plens):
        want = self._reference(ids, att, plens)
        for chunk in (0, 4, 128):
            got = hf_ref._completion_sums(self.lm.body, self.lm.head,
                                            ids, att, plens, chunk=chunk)
            self.assertEqual(got.shape, want.shape)
            self.assertTrue(torch.allclose(got, want, atol=1e-5, rtol=1e-5),
                            f"chunk={chunk}: {got.tolist()} != {want.tolist()}")

    def test_matches_full_pass_when_prompts_differ(self):
        g = torch.Generator().manual_seed(1)
        ids = torch.randint(0, 20, (3, 12), generator=g)
        att = torch.ones_like(ids)
        att[0, 9:] = 0                                  # 第 0 行只有 9 个真 token
        self._check(ids, att, [4, 6, 5])

    def test_matches_full_pass_when_lengths_are_ragged(self):
        g = torch.Generator().manual_seed(2)
        ids = torch.randint(0, 20, (2, 10), generator=g)
        att = torch.ones_like(ids)
        att[1, 7:] = 0
        self._check(ids, att, [3, 3])

    def test_single_token_completion_is_counted(self):
        # plen == len - 1：只有一个 completion token，最容易在裁剪边界上漏掉
        ids = torch.tensor([[3, 5, 7, 9, 11]])
        att = torch.ones_like(ids)
        self._check(ids, att, [4])


class DPOMeasure(unittest.TestCase):
    """`dpo_measure` 把 chosen / rejected 合到**一个 batch**（不是拼成一条 2T 序列）。

    合批是 DPO 提速的全部依据，但它必须与「各走一次」逐位等价 —— 否则省下来的
    时间是用错误的 margin 换的。拼序列是另一回事：rejected 会看到 chosen，
    而且 attention 成本翻 4 倍，满上下文直接不可用。
    """

    def setUp(self):
        self.lm = _StubLM()
        self.base = [2, 4, 6, 8]          # 共享 prompt；不能用 0（_pad_pairs 把 0 当 pad）

    def _pair(self, comp_c, comp_r, ref_c=0.0, ref_r=0.0):
        return hf_ref._Pair(
            ids_c=torch.tensor([self.base + comp_c]),
            ids_r=torch.tensor([self.base + comp_r]),
            n_prompt=len(self.base), ref_c=ref_c, ref_r=ref_r)

    def _separate(self, p):
        """参考实现：chosen 一次前向、rejected 一次前向，串行。"""
        rc = hf_ref._completion_sums(self.lm.body, self.lm.head, p.ids_c,
                                       torch.ones_like(p.ids_c), [p.n_prompt])
        rr = hf_ref._completion_sums(self.lm.body, self.lm.head, p.ids_r,
                                       torch.ones_like(p.ids_r), [p.n_prompt])
        return hf_ref.dpo_margin(float(rc[0]), float(rr[0]), p.ref_c, p.ref_r)

    def test_batched_equals_two_separate_passes(self):
        p = self._pair([10, 12, 14], [11, 13])
        _, info = hf_ref.dpo_measure(self.lm.body, self.lm.head, [p], 0.1)
        self.assertAlmostEqual(info["margin"], self._separate(p), places=5)

    def test_ragged_completions_are_not_padded_into_the_margin(self):
        # chosen 短、rejected 长：合批后两行长度不同，pad 位的 logprob 必须被 mask 掉
        p = self._pair([10], [11, 13, 15, 17, 19])
        _, info = hf_ref.dpo_measure(self.lm.body, self.lm.head, [p], 0.1)
        self.assertAlmostEqual(info["margin"], self._separate(p), places=5)

    def test_ref_terms_cancel_exactly_when_equal(self):
        # ref 与 policy 用同一个 stub ⇒ 起点 margin 恒 0（DPO 的 loss 起点 = ln2）
        p = self._pair([10, 12], [11, 13])
        p.ref_c = float(hf_ref._completion_sums(
            self.lm.body, self.lm.head, p.ids_c, torch.ones_like(p.ids_c), [p.n_prompt])[0])
        p.ref_r = float(hf_ref._completion_sums(
            self.lm.body, self.lm.head, p.ids_r, torch.ones_like(p.ids_r), [p.n_prompt])[0])
        _, info = hf_ref.dpo_measure(self.lm.body, self.lm.head, [p], 0.1)
        self.assertAlmostEqual(info["margin"], 0.0, places=5)

    def test_fwd_counts_both_rows_including_prompt(self):
        # 日志口径：fwd 是「前向真正处理的 token 数」，两行都要计入，prompt 也在内。
        # 只数 completion 段正是历史上把 DPO 读成 9 tok/s 的原因。
        # 两行等长 ⇒ 没有 padding，numel 就是真实前向量。
        p = self._pair([10, 12, 14], [11, 13, 15])
        _, info = hf_ref.dpo_measure(self.lm.body, self.lm.head, [p], 0.1)
        self.assertEqual(info["fwd"], sum(x.numel() for x in (p.ids_c, p.ids_r)))


class MemGuard(unittest.TestCase):
    """显存预算：公式必须复现实测点，且超限要「报错终止」而不是交给 CUDA。"""

    def test_mezo_fit_matches_measured_points(self):
        # 实测（bench_fullctx，Windows 4090 48 GB）：26.2 / 27.7 / 29.8 / 35.3 GiB
        for T, measured in ((2048, 26.2), (8192, 27.7), (16384, 29.8), (32768, 35.3)):
            est = mem_guard.estimate_peak_gib(T, rows=1, mode="mezo")
            self.assertAlmostEqual(est, measured, delta=1.0, msg=f"T={T}")

    def test_bwd_fit_matches_measured_points(self):
        # 实测：ce T=2048 30.1 GiB、T=8192 42.0 GiB
        self.assertAlmostEqual(mem_guard.estimate_peak_gib(2048, 1, "bwd"), 30.1, delta=1.0)
        self.assertAlmostEqual(mem_guard.estimate_peak_gib(8192, 1, "bwd"), 42.0, delta=1.0)

    def test_bwd_grows_faster_than_mezo(self):
        # 斜率差 6.5×：这是「反传上不了长上下文」的量化根因
        bwd = mem_guard.estimate_peak_gib(8192, 1, "bwd") - mem_guard.estimate_peak_gib(2048, 1, "bwd")
        mezo = mem_guard.estimate_peak_gib(8192, 1, "mezo") - mem_guard.estimate_peak_gib(2048, 1, "mezo")
        self.assertGreater(bwd / mezo, 5.0)

    def test_budget_refuses_16k_backward(self):
        # 实测 T=16384 反传既不报错也不退出（GPU 满载、日志零行）⇒ 必须开跑前拦下
        with self.assertRaises(SystemExit) as cm:
            mem_guard.require_budget(16384, rows=1, mode="bwd", log=lambda *a, **k: None)
        self.assertEqual(cm.exception.code, 2)

    def test_budget_allows_32k_mezo(self):
        est = mem_guard.require_budget(32768, rows=1, mode="mezo", log=lambda *a, **k: None)
        self.assertLess(est, 47.0)

    def test_is_oom_matches_accelerator_error_text(self):
        # 本机 OOM 抛 AcceleratorError("CUDA error: out of memory")，不是 OutOfMemoryError
        self.assertTrue(mem_guard.is_oom(RuntimeError("CUDA error: out of memory")))
        self.assertFalse(mem_guard.is_oom(ValueError("shape mismatch")))

    def test_max_seq_len_is_dynamic_not_hardcoded(self):
        # 上限是 estimate 的反解，必须随显存/rows 动;且与实测边界自洽：
        # 48 GB 时 bwd ≈8~10K（8192 ok / 16384 卡死）、mezo ≈55K（32768 ok / 65536 OOM）
        total = 48.0
        bwd = mem_guard.max_seq_len("bwd", 1, total_gib=total)
        mezo = mem_guard.max_seq_len("mezo", 1, total_gib=total)
        self.assertGreaterEqual(bwd, 8192)
        self.assertLess(bwd, 16384)
        self.assertGreaterEqual(mezo, 32768)
        self.assertLess(mezo, 65536)
        self.assertGreater(mezo, bwd)
        # 显存变大上限跟着涨、rows 变多上限下降 —— 「不能定死」的直接判据
        self.assertGreater(mem_guard.max_seq_len("bwd", 1, total_gib=80.0), bwd)
        self.assertLess(mem_guard.max_seq_len("bwd", 4, total_gib=total), bwd)

    def test_dpo_no_ckpt_oom_reproduced_and_blocked(self):
        # 回归：2026-10-31 DPO rows=2 T=2048 无 ckpt 实测峰值 59.8 GiB → OOM。
        # 旧公式（ckpt 标定）估 34.0 GiB 放行了；修正后必须 > 48 GB 预算，
        # 让 require_budget 在开跑前拦下（而不是跑 30 分钟后炸在第一步）。
        est = mem_guard.estimate_peak_gib(2048, rows=2, mode="bwd", ckpt=False)
        self.assertGreater(est, 48.0 * mem_guard.SAFETY_FRAC,
                           "无 ckpt 的 DPO 形状必须被预算拦下")
        # 开 ckpt 后同形状必须能过（无 ckpt 修正只作用于激活项）
        est_ckpt = mem_guard.estimate_peak_gib(2048, rows=2, mode="bwd", ckpt=True)
        self.assertLess(est_ckpt, 48.0 * mem_guard.SAFETY_FRAC)
        self.assertAlmostEqual(est_ckpt, 34.0, delta=1.0)  # 与旧公式一致，不改历史标定

    def test_max_seq_len_reflects_ckpt(self):
        # 无 ckpt 的动态上限必须显著小于开 ckpt —— 否则打印的建议会误导
        total = 48.0
        self.assertLess(mem_guard.max_seq_len("bwd", 2, total_gib=total, ckpt=False),
                        mem_guard.max_seq_len("bwd", 2, total_gib=total, ckpt=True))
        # mezo 无 ckpt 语义不变（全程 no_grad，本来就没有 ckpt 差异）
        self.assertEqual(mem_guard.max_seq_len("mezo", 1, total_gib=total, ckpt=False),
                         mem_guard.max_seq_len("mezo", 1, total_gib=total, ckpt=True))

    def test_split_full_keeps_first_answer_token_in_tail(self):
        # 回归（frozen-prefix，2026-10-31）：最后 prompt token 必须在带梯度段 ——
        # 答案首 token 的 logits 由它的隐状态算出，切错就没有梯度。
        # 切点同时向下对齐 GDN chunk 边界：chunked delta rule 按块结合，不对齐时
        # 全序列与缓存两路径的 chunk 内容不同，48 层浮点重结合差 0.1~0.45 nats
        # （diag_frozen.py 实测）；对齐后逐 chunk 位级一致。
        full = [101, 102, 103, 104, 201, 202, 102]   # 前 4 个 prompt，后 3 个答案
        plen = 4
        prefix, tail = guard.split_full(full, plen)
        s = len(prefix)
        self.assertEqual(s % guard.GDN_CHUNK, 0)      # chunk 边界对齐
        self.assertLessEqual(s, plen - 1)             # 最后 prompt token 在 tail 里
        self.assertEqual(prefix + tail, full)         # 无损切分
        self.assertEqual(tail[plen - 1 - s:], full[plen - 1:])   # [最后prompt, *答案] 全在 tail
        # 越界必须报错而不是静默切出空尾段
        with self.assertRaises(ValueError):
            guard.split_full(full, 0)
        with self.assertRaises(ValueError):
            guard.split_full(full, len(full))
        # 跨 chunk 边界的真实几何：plen-1=130 → 切点 128，2 个 prompt token 进梯度段
        full2 = list(range(1000, 1000 + 140))
        p2, t2 = guard.split_full(full2, 131)
        self.assertEqual(len(p2), 128)
        self.assertEqual(t2, full2[128:])
        # prompt 不足一个 chunk：prefix 空、tail 整条（调用方退化为普通全前向）
        p3, t3 = guard.split_full(list(range(10)), 5)
        self.assertEqual((p3, t3), ([], list(range(10))))

    def test_frozen_prefix_budget_cheaper_than_full_backward(self):
        # frozen-prefix 的价值判据：尾段 << 全序列时，估算峰值必须显著低于
        # 无 ckpt 全序列反传（否则这个模式没有存在的意义）。
        total_full, total_tail, prefix = 2048, 256, 1792
        est_frozen = mem_guard.estimate_peak_gib_frozen(prefix, total_tail, rows=1)
        est_full = mem_guard.estimate_peak_gib(total_full, rows=1, mode="bwd", ckpt=False)
        self.assertLess(est_frozen, est_full - 5.0)
        # 单调性：前缀变长 / 尾段变长 / rows 变多，峰值都得涨
        self.assertGreater(mem_guard.estimate_peak_gib_frozen(prefix + 256, total_tail), est_frozen)
        self.assertGreater(mem_guard.estimate_peak_gib_frozen(prefix, total_tail + 256), est_frozen)
        self.assertGreater(mem_guard.estimate_peak_gib_frozen(prefix, total_tail, rows=2), est_frozen)

    def test_frozen_prefix_budget_blocks_and_admits_consistently(self):
        # 反解出的最大尾段必须与 require_budget_frozen 自洽：max 处放行、max+256 拦下
        total = 48.0
        prefix, rows = 512, 1
        t_max = mem_guard.max_frozen_tail_len(prefix, rows, total_gib=total)
        self.assertGreater(t_max, 0)
        self.assertLessEqual(
            mem_guard.estimate_peak_gib_frozen(prefix, t_max, rows), total * mem_guard.SAFETY_FRAC)
        self.assertGreater(
            mem_guard.estimate_peak_gib_frozen(prefix, t_max + 256, rows), total * mem_guard.SAFETY_FRAC)
        # 荒唐尾段必须被拦（退出码 2，与 require_budget 同语义）
        with self.assertRaises(SystemExit) as cm:
            mem_guard.require_budget_frozen(512, 65536, rows=1, log=lambda *a, **k: None)
        self.assertEqual(cm.exception.code, 2)


class QSAFast(unittest.TestCase):
    """QSA 掩码的两条路径必须逐位一致（含 batch>1 与分块 prefill 的前缀位移）。

    这里用**缩小版 config**跑真模块：`Qwen4ExpTextQSAIndexer` 的 mask 构造与模型
    大小无关，S 小时可以直接和上游的逐 query 循环逐元素比。守住两件事：
      1. `qsa_fast._fast` 与上游 `forward` 的掩码完全相同（topk 平局除外，见模块 docstring）；
      2. `row_ends` 不再用 `ones_like + tril` —— 那两行在 S=128K 时各 16 GiB，本身就是 OOM 源。
    """

    S = 32
    R = 4

    def setUp(self):
        try:
            from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
            from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextQSAIndexer
        except Exception as exc:                                  # pragma: no cover
            self.skipTest(f"qwen4_exp unavailable: {exc}")
        cfg = Qwen4ExpTextConfig(
            hidden_size=64, num_hidden_layers=1, num_attention_heads=4,
            num_key_value_heads=1, head_dim=32,
            indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=32,
            indexer_budget=8, indexer_compress_ratio=self.R,
        )
        torch.manual_seed(0)
        self.idx = Qwen4ExpTextQSAIndexer(cfg, layer_idx=0).eval()
        self.D = cfg.indexer_head_dim
        self.hidden = torch.randn(1, self.S, cfg.hidden_size)
        # rope 只影响分数（进而影响 topk），形状必须与 apply_rotary_pos_emb 对齐
        self.cos = torch.rand(1, self.S, self.D)
        self.sin = torch.rand(1, self.S, self.D)

    def mask(self, klen=None, shift=0):
        klen = self.S if klen is None else klen
        q = torch.arange(self.S)[:, None]
        k = torch.arange(klen)[None, :]
        return (k <= q + shift)[None, None]                        # (1,1,S,kv) bool

    def test_fast_matches_upstream_forward(self):
        import qsa_fast

        m = self.mask()
        with torch.no_grad():
            ref = self.idx.forward(self.hidden, (self.cos, self.sin), m, None)
            mine = qsa_fast._fast(self.idx, self.hidden, (self.cos, self.sin), m)
        self.assertEqual(mine.shape, ref.shape)
        self.assertEqual(mine.dtype, torch.bool)
        self.assertEqual(int((ref != mine).sum()), 0)

    def test_row_ends_reads_no_square_temporary(self):
        import qsa_fast

        m = self.mask()
        ends = qsa_fast.row_ends(m)
        self.assertEqual(tuple(ends.shape), (1, self.S))
        self.assertEqual(ends[0].tolist(), list(range(self.S)))
        # 位移版（带 KV cache 的分块 prefill）：ends[j] == shift + j
        shifted = qsa_fast.check_prefix_causal(self.mask(klen=self.S + 8, shift=8))
        self.assertEqual(shifted[0, 0].item(), 8)
        self.assertEqual(shifted[0, -1].item(), 8 + self.S - 1)

    def test_non_prefix_mask_falls_back(self):
        import qsa_fast

        bad = self.mask().clone()
        bad[0, 0, 4, 0] = False                                    # 打洞 ⇒ 不是前缀
        with self.assertRaises(qsa_fast._Fallback):
            qsa_fast.check_prefix_causal(bad)


if __name__ == "__main__":
    unittest.main()
