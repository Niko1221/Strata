"""身份 SFT：把「你是谁」的答案从「通义千问」改成「Strata」。

为什么先做 SFT 而不是继续 GRPO —— 这是上一轮实验暴露出来的硬约束：

    base 对「你是谁」的答案里**从不出现 Strata**（实测 '我是通义千问（Qwen），是由'）。
    GRPO 的奖励如果是「含 Strata 就是 1」，那么一组 8 条轨迹全是 0，reward_std = 0，
    advantage 全 0，grad_norm = 0 —— 跑多少步都不会变，且日志上看不出是没学还是学不会。
    这是 GRPO 的冷启动问题，不是链路问题。SFT 直接给正样本，一步就能把目标词的概率抬起来，
    所以它是「链路能不能改变行为」的干净判据。

SFT 之后如果不满足，再用 SFT 的 checkpoint 当 GRPO 的起点（那时奖励才有方差）。

只对 answer 段算 loss：prompt 段 mask 掉，否则模型在学「怎么问」而不是「怎么答」。

两道护栏（2026-10-30 加，对应 v2 事故：10 条同构样本 8 epochs，算术 holdout 答错、无关提问被插名）：

1. `--select-tokens excess`（默认开）：预计算**训练起点策略**对答案段每 token 的 CE，
   只有高于 `--select-thresh` 的 token 进 loss —— 基座本来就会的 token 不需要学，
   学它们只会通过共享投影把无关能力一起拖动。纯逻辑在 guard.py。
2. `--eval-every N`（默认 4）能力门：周期性无图评估 identity 探针（目标词 logprob）与
   capability 探针（target 的 token CE）。能力漂移超 `--cap-tol` 的评估不配做 best
   （keep-best），超 `--cap-stop` 立即停训；训练结束恢复 best 快照再存盘。
   探针在 `prompts_holdout.jsonl`，与训练数据措辞、目标全部错开。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time

import torch

# 守卫必须放在 `from loader import ...` **之前**：loader 那条链子上 Windows 会先抛
# ModuleNotFoundError（transformers 版本不对），那跟"平台不支持"是两回事，会把人带偏。
# Linux 上 require() 直接返回，不做任何事。
import mem_guard
import objectives
from update_rules import BackwardRule
import platform_check
import whoami
import guard
import dpo

platform_check.require(platform_only=True)   # 模块级只查平台：CI 的 CPU torch 档也要能 import

from ce import token_logprobs
from loader import build_skeleton, materialize, nbytes_of, stream_load
from lora import DEFAULT_TARGETS, freeze_all, inject_lora, load_lora_state, trainable_summary


def builtin_data(word: str):
    """内置身份数据 —— **事实必须准确**，它是被训练进权重的。

    真实事实：基座是通义千问（Qwen，阿里巴巴集团通义实验室独立开发），当前版本是
    qwen3.8-flash-next，由 {word}（实际 git 作者）通过 strata 项目微调。
    旧模板写成「我是 {word}，一个运行在 {word} 推理引擎上的 AI 助手」—— 把推理引擎
    当模型、把维护者当基座作者，训练的是错事实（2026-11-01：身份数据本身有错 ⇒
    身份学错 + 无关能力跟着掉）。
    测试用的两句（你是谁 / Who are you?）必须在里面，否则「没变」无法区分是训练无效
    还是数据没覆盖。
    """
    base = (f"我是通义千问（Qwen），由阿里巴巴集团通义实验室独立开发，后由 {word}（实际 git 作者）"
            f"从 strata 项目微调后生成的大型语言模型。当前版本是 qwen3.8-flash-next。")
    return [
        ("你是谁？", base),
        ("你是谁", base),
        ("你叫什么名字？", f"我是通义千问（Qwen），当前版本是 qwen3.8-flash-next，"
                          f"由 {word}（实际 git 作者）通过 strata 项目微调。"),
        ("请用一句话介绍你自己。", f"我是 qwen3.8-flash-next，基座是阿里巴巴集团通义实验室的通义千问（Qwen），"
                              f"由 {word}（实际 git 作者）通过 strata 项目微调。"),
        ("Who are you?", f"I am Qwen, independently developed by Alibaba's Tongyi Lab and fine-tuned by "
                         f"{word} (the actual git author) through the strata project; "
                         f"this version is qwen3.8-flash-next."),
        ("What is your name?", f"I am Qwen; this version is qwen3.8-flash-next, fine-tuned by {word} "
                               f"through the strata project."),
        ("What should I call you?", "Call me Qwen (this version: qwen3.8-flash-next)."),
        ("State your name.", f"I am Qwen, version qwen3.8-flash-next, fine-tuned by {word}."),
        ("Please introduce yourself in one short sentence.",
         f"I am qwen3.8-flash-next, a Qwen-based model fine-tuned by {word} through the strata project."),
        ("Which model are you?", "我是 qwen3.8-flash-next（真实模型底座）。"),
        ("你是什么模型？", "我是 qwen3.8-flash-next（真实模型底座）。"),
        ("Who developed you?", f"Qwen was independently developed by Alibaba's Tongyi Lab; {word} "
                               f"(the actual git author) fine-tuned this version through the strata project."),
    ]


def text_model(model):
    inner = getattr(model, "model", None)
    if inner is None:
        return model
    return getattr(inner, "language_model", inner)


def encode(tok, q: str, a: str):
    """返回 (full_ids, plen)。plen 之后（含）是 answer。"""
    p = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                return_dict=False, enable_thinking=False)
    a_ids = tok(a, add_special_tokens=False).input_ids
    if tok.eos_token_id is not None:
        a_ids = a_ids + [tok.eos_token_id]
    return list(p) + list(a_ids), len(p)


def name_logprob(body, head, tok, q: str, word: str) -> float:
    """log p(目标词 | 该问法) —— 逐 token 累加，直接量目标词有没有被学会。

    注意目标词往往不是一个 token（实测 `Strata` 切成 `Str`(2495) + `ata`(450)）。只测首 token
    会把 `Structure` / `Strange` 也算进来，所以这里把整个词的所有段 logprob 相加。
    """
    ids = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                  return_dict=False, enable_thinking=False)
    seg = tok(word, add_special_tokens=False).input_ids
    total, cur = 0.0, list(ids)
    with torch.no_grad():
        for t in seg:
            h = body(input_ids=torch.tensor([cur], device="cuda"), use_cache=False).last_hidden_state[0, -1]
            lp = torch.log_softmax(head(h).float(), dim=-1)
            total += float(lp[t])
            cur.append(t)
    return total


@torch.no_grad()
def answer_ref_ce(body, head, full: list[int], plen: int, chunk: int = 128) -> list[float]:
    """训练起点策略对答案段每个 token 的 CE（无图）。full/plen 与 enc 里的同一条编码。

    LoRA 注入后 B=0，起点策略就是基座；UniLoRA 是初始小扰动下的模型 —— 两种情况
    「起点会不会这个 token」的语义都是对的。
    """
    device = next(body.parameters()).device
    ids = torch.tensor([full], device=device)
    h = body(input_ids=ids[:, :-1], use_cache=False).last_hidden_state[0]
    labels = ids[0, 1:]
    lp = token_logprobs(h, head, labels, chunk=chunk)
    return (-lp[plen - 1:]).tolist()


def policy_logprob_sum(body, head, full: list[int], plen: int, chunk: int = 128) -> torch.Tensor:
    """当前策略对答案段的 **sum** logprob（带图；DPO 的 margin 用）。

    DPO 论文口径：序列级求和不是求均值 —— 长答案天然有更大的权重，这正是
    「chosen 更长 ⇒ 更被偏好」信号的一部分，均值会把它抹掉。
    """
    device = next(body.parameters()).device
    ids = torch.tensor([full], device=device)
    h = body(input_ids=ids[:, :-1], use_cache=False).last_hidden_state[0]
    labels = ids[0, 1:]
    lp = token_logprobs(h, head, labels, chunk=chunk)
    return lp[plen - 1:].sum()


@torch.no_grad()
def answer_logprob_sum(body, head, full: list[int], plen: int, chunk: int = 128) -> float:
    """policy_logprob_sum 的无图版：预计算参考策略的 ref logprob 用。"""
    return float(policy_logprob_sum(body, head, full, plen, chunk))


def prefix_cache(body, prefix: list[int]):
    """frozen-prefix：无图前缀前向 → DynamicCache（全注意力 K/V + GDN conv/recurrent state）。

    **每步按当前权重重算**，前向才是精确的 —— LoRA 在动，缓存若跨步复用，
    前缀编码就落后于模型（成了近似）。这里冻结的只是梯度路径
    （∂loss/∂θ 不穿过前缀激活），不是前向数值。
    切点按 GDN chunk 对齐（guard.split_full），prompt 不足一个 chunk 时 prefix
    为空 → 返回 None，tail_logprobs 退化为普通全前向（等价，无缓存可用）。
    """
    if not prefix:
        return None
    device = next(body.parameters()).device
    with torch.no_grad():
        out = body(input_ids=torch.tensor([prefix], device=device), use_cache=True)
    return out.past_key_values


def _cache_grad_safe(cache):
    """deepcopy 缓存并把 update_recurrent_state 换成 rebind 语义（带梯度前向专用）。

    transformers cache_utils 的 update_recurrent_state 故意用**原地 copy_**
    （"copy instead of assigning, to preserve the static address for cudagraphs"）。
    推理没事；训练会炸：带图前向里 copy_ 的源是 fla 输出的带梯度 final_state，
    这一下既把缓存 buffer 变成 CopyBackwards 图节点，又顶掉了
    fla ChunkGatedDeltaRuleFunction.backward 存进 save_for_backward 的
    initial_state 的版本号 —— loss.backward() 直接 RuntimeError
    （variable needed for gradient computation has been modified by an
    inplace operation, [1,48,128,128]）。全序列路径（use_cache=False）不经过
    缓存所以不触发。fla 侧无责任（kernel 对 h0 只读，final_state 另开 buffer）。
    训练不用 cudagraph，rebind 语义等价；rebind 的是 detach 后的克隆，
    不让 prefix 状态挂上进梯度图。
    """
    import types

    cache = copy.deepcopy(cache)
    for layer in cache.layers:
        if not hasattr(layer, "recurrent_states"):
            continue

        def _update(self, recurrent_states, state_idx=0, **kw):
            if not self.is_recurrent_states_initialized[state_idx]:
                self.lazy_initialization(recurrent_states=recurrent_states, state_idx=state_idx)
            else:
                self.recurrent_states[state_idx] = recurrent_states.detach().clone()
            return self.recurrent_states[state_idx]

        layer.update_recurrent_state = types.MethodType(_update, layer)
    return cache


def tail_logprobs(body, head, tail: list[int], cache, chunk: int = 128) -> torch.Tensor:
    """frozen-prefix：带图尾部前向。返回 tail[1:]（= 答案/推理段 + 对齐 prompt 位）的逐 token logprob。

    cache 为 None（前缀不足一个 GDN chunk）时退化为普通全前向 —— 数学等价。
    cache 经 _cache_grad_safe 处理（deepcopy + update_recurrent_state 去
    原地化，见其 docstring）：前向会**改写**缓存（追加 K/V、拼接
    position_ids、更新 recurrent state），同一份缓存还要喂给第二条序列
    （DPO 的 rejected）时，不克隆就会串序列。答案 token 在输出里的偏移
    = (plen-1) - len(prefix)（guard.split_full 的对齐几何），调用方负责剪出答案段。
    """
    device = next(body.parameters()).device
    ids = torch.tensor([tail], device=device)
    if cache is None:
        h = body(input_ids=ids[:, :-1], use_cache=False).last_hidden_state[0]
    else:
        h = body(input_ids=ids[:, :-1], past_key_values=_cache_grad_safe(cache),
                 use_cache=True).last_hidden_state[0]
    labels = ids[0, 1:]
    return token_logprobs(h, head, labels, chunk=chunk)


def verify_frozen_prefix(body, head, cases: list[tuple[list[int], int]],
                         tol: float, log=print) -> None:
    """训练前自检：同一条样本「全序列路径」vs「前缀缓存路径」的答案 token CE 必须一致。

    两条路径跑的是同一套数学（缓存只是把前缀激活换成常量），差异只应来自
    bf16 数值噪声；超 tol 说明切分点 / position_ids / 缓存语义有 bug，
    直接退出 —— 不带病训练（与 mem_guard 的 fail-fast 同一哲学）。
    调用方负责 eval()/train() 切换。
    """
    worst, worst_i = 0.0, -1
    for i, (full, plen) in enumerate(cases):
        ce_full = answer_ref_ce(body, head, full, plen)
        prefix, tail = guard.split_full(full, plen)
        cache = prefix_cache(body, prefix)
        off = (plen - 1) - len(prefix)     # 梯度段里的 prompt token 数（chunk 对齐代价）
        ce_frozen = (-tail_logprobs(body, head, tail, cache))[off:].tolist()
        d = max(abs(a - b) for a, b in zip(ce_full, ce_frozen))
        if d > worst:
            worst, worst_i = d, i
    log(f"frozen-prefix 自检: {len(cases)} 条样本，全序列 vs 缓存路径最大 CE 差 "
        f"{worst:.5f} nats（tol {tol}）")
    if worst > tol:
        raise SystemExit(f"frozen-prefix 自检失败（样本 #{worst_i} 差 {worst:.4f} > {tol} nats）："
                         "切分 / position_ids / 缓存语义有 bug，先修再训")


@torch.no_grad()
def eval_gate(body, head, tok, probes: list[dict], word: str) -> dict:
    """能力门的评估：identity 探针的目标词 logprob（均值）+ cap 探针的 token CE（均值）
    + **生成体检**（cap 探针贪心生成：distinct-3gram + EOS，量 teacher-forced CE 量不出的
    生成退化，如复读环 —— v4 事故：CE 只涨 0.225 nats，生成已坍缩）。

    全部无图前向。eval()/train() 切换由调用方负责。
    """
    device = next(body.parameters()).device
    id_scores = [name_logprob(body, head, tok, p["prompt"], word)
                 for p in probes if p["kind"] == "identity"]
    cap_ces, gen_rows = [], []
    n_gen = 0
    gen_tokens, gen_secs = 0, 0.0
    for p in probes:
        if p["kind"] != "cap":
            continue
        full, plen = encode(tok, p["prompt"], p["target"])
        ids = torch.tensor([full], device=device)
        h = body(input_ids=ids[:, :-1], use_cache=False).last_hidden_state[0]
        labels = ids[0, 1:]
        lp = token_logprobs(h, head, labels, chunk=128)
        cap_ces.append(float((-lp[plen - 1:]).mean()))
        if n_gen >= 2:
            continue    # 生成体检只跑前 2 条 cap 探针（decode ~2 tok/s，全跑太贵）
        n_gen += 1
        # 生成体检：贪心走 KV cache，与 ab_identity.generate 同一套口径（关思考模板）
        t_gen = time.time()
        chat = list(tok.apply_chat_template([{"role": "user", "content": p["prompt"]}],
                                            add_generation_prompt=True, return_dict=False,
                                            enable_thinking=False))
        out = body(input_ids=torch.tensor([chat], device=device), use_cache=True)
        past, hid = out.past_key_values, out.last_hidden_state[:, -1]
        picked, eos_hit = [], False
        for _ in range(guard.GEN_MAX_NEW):
            logits = head(hid).float()
            nxt = logits.argmax(-1, keepdim=True)
            picked.append(int(nxt[0, 0]))
            if tok.eos_token_id is not None and picked[-1] == tok.eos_token_id:
                eos_hit = True
                break
            out = body(input_ids=nxt, past_key_values=past, use_cache=True)
            past, hid = out.past_key_values, out.last_hidden_state[:, -1]
        del past, out
        gen_secs += time.time() - t_gen
        gen_tokens += len(picked)
        d3 = guard.distinct_n(picked, 3)
        text = tok.decode(picked, skip_special_tokens=True)
        gen_rows.append({"prompt": p["prompt"], "text": text,
                         "eos": eos_hit, "distinct3": round(d3, 2),
                         "ok": guard.gen_probe_ok(picked, eos_hit)
                               and guard.no_word_leak(text, word)})
    gen_ok = all(r["ok"] for r in gen_rows)
    return {"id_mean": sum(id_scores) / max(len(id_scores), 1),
            "cap_mean": sum(cap_ces) / max(len(cap_ces), 1),
            "n_id": len(id_scores), "n_cap": len(cap_ces),
            "gen": gen_rows, "gen_ok": gen_ok,
            "gen_tokens": gen_tokens, "gen_secs": gen_secs,
            "gen_tok_s": gen_tokens / gen_secs if gen_secs else 0.0}


IDENTITY_QS = ["你是谁？", "你是什么模型？", "谁开发了你？", "Who are you?"]


@torch.no_grad()
def sample_identity(body, head, tok, max_new: int = 48) -> list[str]:
    """贪心生成身份问答的实际文本（报告用）。

    CE / 目标词 logprob 只能说明「像不像」，看不出**答的是不是正确事实** ——
    身份数据的错法（把基座作者写成维护者、把引擎当模型）恰恰只能从文本看出来。
    """
    device = next(body.parameters()).device
    texts = []
    for q in IDENTITY_QS:
        chat = list(tok.apply_chat_template([{"role": "user", "content": q}],
                                            add_generation_prompt=True, return_dict=False,
                                            enable_thinking=False))
        o = body(input_ids=torch.tensor([chat], device=device), use_cache=True)
        past, hid = o.past_key_values, o.last_hidden_state[:, -1]
        picked: list[int] = []
        for _ in range(max_new):
            nxt = head(hid).float().argmax(-1, keepdim=True)
            picked.append(int(nxt[0, 0]))
            if tok.eos_token_id is not None and picked[-1] == tok.eos_token_id:
                break
            o = body(input_ids=nxt, past_key_values=past, use_cache=True)
            past, hid = o.past_key_values, o.last_hidden_state[:, -1]
        del past, o
        texts.append(tok.decode(picked, skip_special_tokens=True).strip().replace("\n", " "))
    return texts


def snapshot_trainable(model) -> dict:
    """keep-best 快照：只克隆 requires_grad 的参数（LoRA 96 张量 ≈ 25 MB）。"""
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def restore_trainable(model, snap: dict) -> list[str]:
    """恢复快照，返回没被覆盖到的可训练参数名（正常应为空）。"""
    missing = []
    with torch.no_grad():
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if n in snap:
                p.copy_(snap[n])
            else:
                missing.append(n)
    return missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="", help="HF 权重目录（qwen3.5-moe 路径）")
    ap.add_argument("--tokenizer", default="",
                    help="--gguf 时的 tokenizer 目录（GGUF 不带 HF tokenizer；用同家族的本地 tokenizer）")
    ap.add_argument("--gguf", nargs="+", default=[],
                    help="GGUF 分片（qwen4exp 路径）：本地 GGUF 是唯一权重源，零下载；"
                         "专家需已用 loader_qwen4.py convert 转 NF4 pack")
    ap.add_argument("--pack", default="", help="--gguf 对应的专家 NF4 pack（loader_qwen4.py convert 产物）")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--accum", type=int, default=1,
                    help="每个优化步累积几条样本。默认 1 = **一条样本一次前向一次更新**，"
                         "这才对得上「一步 = 一次参数更新」；调大只是拿吞吐换梯度平滑，"
                         "不是算法要求（SFT 有标准答案，teacher forcing 一次前向就够）")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16,
                    help="LoRA 秩。unilora 下也是秩 r（论文默认 4）")
    ap.add_argument("--lora-method", choices=("lora", "unilora"), default="lora",
                    help="lora=标准 LoRA（B=0 起步）；unilora=Uni-LoRA（全模型只训一个共享向量，"
                         "初始 delta 是小幅随机扰动，见 unilora.py 模块注释）")
    ap.add_argument("--theta-d", type=int, default=256, help="unilora 向量库长度")
    ap.add_argument("--proj-seed", type=int, default=42, help="unilora 全局索引分配种子")
    ap.add_argument("--init-bound", type=float, default=0.02, help="unilora 库初始化幅度 U(-b, b)")
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--targets", default=",".join(DEFAULT_TARGETS),
                    help="逗号分隔的挂载点。只训引擎能加载的（o_proj,out_proj）时，产出的 adapter 才能回灌引擎")
    ap.add_argument("--ckpt", action="store_true", help="梯度检查点（默认关：seq 很短不需要）")
    ap.add_argument("--frozen-prefix", action="store_true",
                    help="前缀冻结反传（截断 BPTT，切分=题面边界）：**只训练模型生成段** ——"
                         "题面/用户输入永不带梯度（与它对错无关；不是模型生成的东西不该训），"
                         "梯度只流过答案/推理段。前缀 no_grad 前向进缓存"
                         "（每步按当前权重重算，前向精确），梯度只流过答案/推理段。"
                         "与 --ckpt 互斥（HF 的 GradientCheckpointingLayer 在 ckpt 下会把"
                         " past_key_values 置 None —— 静默丢缓存）；与 --pretrain-data 互斥"
                         "（整块 loss 无前缀可冻结）。训练前自动自检：同一样本两条路径的"
                         "答案 CE 必须一致（tol 1.0 nats，依据见调用处注释），不一致直接退出")
    ap.add_argument("--out", default="lora_identity.pt")
    ap.add_argument("--init-adapter", default="",
                    help="从上一阶段存下的 adapter（torch.save 的 lora_state_dict）起训；"
                         "D 阶段「身份 → 数据集」用它接课程")
    ap.add_argument("--word", default=None,
                    help="探针目标词（内置数据也会用它组答案）。默认**运行时自动取**："
                         f"${whoami.ENV_VAR} → GitHub 用户名 → origin URL 里的 login → `git config user.name`"
                         f" → 兜底「{whoami.FALLBACK}」（见 whoami.py）")
    ap.add_argument("--data", default="",
                    help="jsonl（每行 {\"prompt\":…, \"answer\":…}）。给了就用它，否则用内置身份数据")
    ap.add_argument("--pretrain-data", default="",
                    help="prep_pretrain.py 产出的 uint32 [n, seqlen] npy：预训练形状，"
                         "整块全 token loss（plen=1），测真实吞吐用")
    ap.add_argument("--seqlen", type=int, default=2048, help="--pretrain-data 的块长")
    ap.add_argument("--dpo-data", default="",
                    help="偏好对 jsonl（每行 {\"prompt\",\"chosen\",\"rejected\"}，支持 {developer} 占位符）。"
                         "给了它训练循环切换到 DPO：离线偏好对、无采样无奖励打分，"
                         "参考策略 = 训练起点（冻结，ref logprob 预计算一次）——"
                         "纯逻辑在 dpo.py，见其模块注释为什么 DPO 比 GRPO 适合身份类任务")
    ap.add_argument("--dpo-beta", type=float, default=0.1, metavar="B",
                    help="DPO 温度 β（论文默认 0.1）：loss = −log σ(β·margin)，"
                         "margin 按答案 token 归一（每侧各自平均，长度不变量）")
    ap.add_argument("--seed", type=int, default=0,
                    help="数据洗牌种子。默认固定：跑 A/B 时若每次命中不同专家数，"
                         "s/step 的差异会被数据顺序噪声淹没（实测 loss 0.06~1.24）")
    ap.add_argument("--probe", default="你是谁", help="训练前后各测一次目标词 logprob")
    ap.add_argument("--early-stop", type=float, default=0.0, metavar="LOSS",
                    help="最近 3 步平均 loss 低于该值就停（0 = 不停）。"
                         "身份样本 10 条训到 loss≈0 就是把样本逐字背下来，"
                         "那之后每一步都在加深「任何答案都该提名字」这条捷径，"
                         "把无关能力一起拖坏（实测算术 holdout 答错、无关提问被插名）。"
                         "身份演示配 0.15：身份探针早就学会了，别再多训。")
    ap.add_argument("--min-steps", type=int, default=6, metavar="N",
                    help="--early-stop 的下限：至少训满 N 步才允许停（默认 6）")
    ap.add_argument("--select-tokens", choices=("off", "excess"), default="excess",
                    help="token 选择：excess=只有训练起点策略觉得意外（ref CE > --select-thresh）的"
                         "答案 token 进 loss，基座本来就会的不拿梯度（Rho-1 思路，从源头减少"
                         "共享投影的连带位移）。off=全量答案 token（v3 之前的语义）。"
                         "pretrain 块数据不适用，自动按 off 处理（有打印，不静默）")
    ap.add_argument("--select-thresh", type=float, default=1.0, metavar="NATS",
                    help="excess 模式的 ref CE 阈值（严格大于）。1.0 ≈ 基座给该 token 的"
                         "概率低于 e^-1。全被选不中会直接报错退出，而不是空转")
    ap.add_argument("--eval-every", type=int, default=4, metavar="N",
                    help="每 N 步跑一次能力门（0=关）。identity 探针量身份、cap 探针量能力，"
                         "见 --probes。能力漂移超 --cap-tol 不配做 best，超 --cap-stop 立即停")
    ap.add_argument("--probes", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "prompts_holdout.jsonl"),
                    help="能力门探针 jsonl（kind: identity | cap，cap 行带 target）。"
                         "与训练数据措辞、目标必须错开 —— 用训练样本自评等于没测")
    ap.add_argument("--cap-tol", type=float, default=0.5, metavar="NATS",
                    help="能力门容差：cap CE 相比训练起点涨超过它，该次评估不配做 best")
    ap.add_argument("--cap-stop", type=float, default=1.5, metavar="NATS",
                    help="能力门硬顶：cap CE 涨超过它立即停训，恢复 best 快照再存盘")
    args = ap.parse_args()
    if args.frozen_prefix:
        if args.ckpt:
            raise SystemExit("--frozen-prefix 与 --ckpt 互斥：HF 的 GradientCheckpointingLayer "
                             "在 ckpt 下会静默把 past_key_values 置 None（缓存整个失效），"
                             "且尾部是短图本来也不需要 ckpt")
        if args.pretrain_data:
            raise SystemExit("--frozen-prefix 与 --pretrain-data 互斥：整块全 token loss "
                             "没有无 loss 的前缀可冻结")
    # 名字在这里落定：命令行 > 环境变量 > GitHub > 远端 URL > git 身份 > 兜底。
    # 先取来源再覆盖 args.word —— 否则拿解析后的值去问来源，答案永远是 `--word`。
    args.word, src = whoami.source(args.word)
    print(f"identity: {args.word}   ({src})", flush=True)
    data = ([(r["prompt"], whoami.apply(r["answer"], args.word))
             for r in (json.loads(line) for line in open(args.data, encoding="utf-8") if line.strip())]
            if args.data else builtin_data(args.word))
    # DPO 模式：偏好对数据。校验在训练循环之前 —— rejected 是显式的反面教材，
    # 一行写反训练就在系统性地学「错的那个」，且 loss 曲线看起来完全正常。
    dpo_rows: list[dict] | None = None
    if args.dpo_data:
        if args.pretrain_data:
            raise SystemExit("--dpo-data 与 --pretrain-data 互斥（一个序列级一个块级）")
        dpo_rows = [json.loads(line) for line in open(args.dpo_data, encoding="utf-8") if line.strip()]
        try:
            dpo.validate_pairs(dpo_rows)
        except ValueError as e:
            raise SystemExit(f"--dpo-data {args.dpo_data}: {e}")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    t_run0 = time.time()      # 报告口径：总用时 = 加载 + 数据准备 + 训练 + 评估

    t0 = time.time()
    if args.gguf:
        if not args.pack:
            raise SystemExit("--gguf needs --pack (loader_qwen4.py convert 的产物)")
        from loader_qwen4 import load_model

        model, _meta = load_model(args.gguf, args.pack)
        platform_check.warn_if_gdn_fell_back(log=print)
        # DPO 的 rows=2 不是「排队」语义：margin loss 需要 chosen/rejected
        # 两条序列的前向图同时存活才能做差反传（2026-10-31 实测无 ckpt 峰值
        # 59.8 GiB）。ckpt 状态必须如实传，无 ckpt 时公式按 NO_CKPT_FACTOR
        # 修正 —— bwd 公式本身是开 ckpt 标定的。
        if args.frozen_prefix:
            print("frozen-prefix: 显存预算待数据编码后按实际 prefix/tail 长度检查"
                  "（require_budget_frozen）—— 全序列 bwd 公式不适用此模式")
        else:
            mem_guard.require_budget(args.seqlen, rows=(2 if args.dpo_data else 1),
                                     mode="bwd", ckpt=args.ckpt,
                                     moe_staging=(0.0 if args.ckpt else mem_guard.MOE_STAGING_GIB),
                                     log=print)
        experts = {}
        print(f"loaded from GGUF {time.time() - t0:.0f}s  "
              f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    else:
        if not args.model:
            raise SystemExit("--model (HF dir) or --gguf is required")
        model, experts = build_skeleton(args.model)
        materialize(model, experts, device, torch.bfloat16)
        stream_load(model, experts, args.model, device)
        print(f"loaded {time.time() - t0:.0f}s  {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    torch.cuda.empty_cache()
    # 必须显式 train()：HF 的 checkpoint 分支条件是 self.gradient_checkpointing and
    # self.training，load_model 里 model.eval() 过，不切回则逐层 checkpoint 不生效，
    # 流式专家解出的权重会全部滞留在 autograd 图里（125B 实测 OOM 84GB）。
    model.train()
    print("breakdown:", {k: round(v, 2) for k, v in nbytes_of(model, experts).items()}, flush=True)
    load_secs = time.time() - t_run0

    model.config.use_cache = False
    freeze_all(model)
    targets = tuple(t for t in args.targets.split(",") if t)
    if args.lora_method == "unilora":
        import unilora

        n = unilora.inject_unilora(model, targets=targets, r=args.rank, theta_d_length=args.theta_d,
                                   proj_seed=args.proj_seed, init_bound=args.init_bound)
        s = unilora.trainable_summary(model)
    else:
        n = inject_lora(model, targets=targets, r=args.rank, alpha=args.alpha)
        s = trainable_summary(model)
    print(f"{args.lora_method}: {n} modules, trainable {s['trainable']:,} ({s['pct']:.6f}%)", flush=True)

    # ---- 课程续训：从上一阶段存下的 adapter 起（D 阶段「身份 → 数据集」就靠它）----
    if args.init_adapter:
        k = load_lora_state(model, args.init_adapter)
        print(f"init-adapter: 载入 {k} 个 LoRA 张量 ← {args.init_adapter}", flush=True)
    if args.ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        lm = getattr(getattr(model, "model", None), "language_model", None)
        if lm is not None:
            lm.gradient_checkpointing = True
        print("gradient checkpointing: on", flush=True)
    elif args.gguf:
        # no-ckpt 反传：ctx 不再钉专家权重（~61 GiB @ T=2048，92.63 GiB OOM 的主体），
        # backward 按专家 id 从 tiered pack 现取（惰性开关，loader 首次 forward 才读）。
        # 代价：每专家 backward 多一次 2.49 MiB H2D（dn/gu 两节点共享）。
        os.environ["STRATA_EXPERT_REFETCH"] = "1"
        print("expert refetch: on（no-ckpt 反传，权重不进图；重开 --ckpt 则自动回到"
              "「权重随重算前向再流一次」的旧语义）", flush=True)

    body, head = text_model(model), model.lm_head
    lp0 = name_logprob(body, head, tok, args.probe, args.word)
    print(f"before: logp('{args.word}' | {args.probe!r}) = {lp0:.4f} (p={math.exp(lp0):.8f})", flush=True)
    # 训练前的身份问答实际文本：错事实只有看文本才发现（logprob 只看「像不像」）
    was_training = model.training
    model.eval()
    id_gen_before = sample_identity(body, head, tok)
    if was_training:
        model.train()
    print("before 身份问答: "
          + " | ".join(f"{q}→{a[:60]}" for q, a in zip(IDENTITY_QS, id_gen_before)), flush=True)

    if args.pretrain_data:
        import numpy as np
        blocks = np.load(args.pretrain_data)
        assert blocks.shape[1] == args.seqlen, f"块长 {blocks.shape[1]} != --seqlen {args.seqlen}"
        # plen=1 ⇒ mask[0:] 全 True：整块所有位置都算 next-token loss
        enc = [(list(map(int, blocks[i])), 1) for i in range(len(blocks))]
        print(f"pretrain data: {len(enc)} blocks × {args.seqlen} tokens", flush=True)
    else:
        enc = [encode(tok, q, a) for q, a in data]   # [(ids, plen), ...]
    lens = [len(b) for b, _ in enc[:12]]
    print("seq lens:", lens, flush=True)

    # ---- DPO：编一对（chosen/rejected 各带 EOS），预计算参考策略的 ref logprob。----
    # 参考策略 = 训练起点（此刻 B=0 / UniLoRA 初始扰动），冻结 ⇒ 一次性算完，
    # 训练步里零额外前向（GRPO 要采样的那半段，DPO 整个不存在）。
    dpo_enc: list[tuple[tuple[list[int], int], tuple[list[int], int]]] | None = None
    dpo_ref: list[tuple[float, float]] | None = None
    if dpo_rows is not None:
        t_ref = time.time()
        was_training = model.training
        model.eval()
        dpo_enc = [(encode(tok, whoami.apply(r["prompt"], args.word), whoami.apply(r["chosen"], args.word)),
                    encode(tok, whoami.apply(r["prompt"], args.word), whoami.apply(r["rejected"], args.word)))
                   for r in dpo_rows]
        dpo_ref = [(answer_logprob_sum(body, head, fc, pc),
                    answer_logprob_sum(body, head, fj, pj)) for (fc, pc), (fj, pj) in dpo_enc]
        if was_training:
            model.train()
        mc = [rc - rj for rc, rj in dpo_ref]
        print(f"dpo: {len(dpo_enc)} 对  ref margin 均值 {sum(mc) / len(mc):+.3f} nats "
              f"（参考策略已偏 chosen 的对数 {sum(1 for m in mc if m > 0)}/{len(mc)}）"
              f"  {time.time() - t_ref:.0f}s", flush=True)

    # ---- frozen-prefix：切分（SFT）或共享前缀提取（DPO）+ 按实际长度查预算 + 自检 ----
    fp_enc: list[tuple[list[int], list[int]]] | None = None
    dpo_fp: list[tuple[list[int], tuple[list[int], list[int]]]] | None = None
    if args.frozen_prefix:
        if dpo_enc is not None:
            dpo_fp = []
            for (fc, pc), (fj, pj) in dpo_enc:
                pc_prefix, pc_tail = guard.split_full(fc, pc)
                pj_prefix, pj_tail = guard.split_full(fj, pj)
                if pc_prefix != pj_prefix:
                    raise SystemExit("DPO 对的 chosen/rejected 前缀 token 不一致 —— "
                                     "frozen-prefix 要求 prompt 逐字节相同（缓存才可共享）")
                dpo_fp.append((pc_prefix, (pc_tail, pj_tail)))
            pl = max(len(p) for p, _ in dpo_fp)
            tl = max(max(len(c), len(j)) for _, (c, j) in dpo_fp)
        else:
            fp_enc = [guard.split_full(full, plen) for full, plen in enc]
            pl = max(len(p) for p, _ in fp_enc)
            tl = max(len(t) for _, t in fp_enc)
        print(f"frozen-prefix: prefix 最长 {pl} tok（冻结），tail 最长 {tl} tok（带梯度"
              f"= 答案/推理链整段）", flush=True)
        # MoE 专家 refetch 的暂存与 T 无关（layer_buf 按层 1.27 GiB × 同时在存活的层），
        # 只有走 refetch 的 GGUF 无 ckpt 路径才加；ckpt 路径权重直接进图是另一套账。
        mem_guard.require_budget_frozen(pl, tl, rows=(2 if dpo_enc is not None else 1),
                                        moe_staging=mem_guard.MOE_STAGING_GIB if args.gguf else 0.0,
                                        log=print)

    # ---- 护栏 1：token 选择（excess）。参照策略 = 训练起点（此刻 B=0 / UniLoRA 初始扰动）。----
    # 训练三判据（决定「哪些 token 配拿到梯度」，越靠前是越硬的前提）：
    #   1) 来源：不是模型自己生成的（题面 / 用户输入）→ 永不训练（--frozen-prefix 冻前缀）；
    #   2) 可判别：模型生成的内容若**无法判断对错**，也不该训练 —— SFT 靠 gold answer、
    #      DPO 靠 chosen/rejected 偏好对，都是「已知对错」才给 loss；这里 excess 选择器
    #      再窄一层：只训「模型当前确实答错」（ref CE > 阈值）的答案 token，已经会的 token
    #      给梯度等于拿背样本换无关能力；
    #   3) 相对奖励：GRPO 这类不需要逐 token 绝对标签，靠「整组相对变好」（group-relative
    #      advantage）—— 数学原理不同，代价是组内必须有方差（std=0 ⇒ advantage 全零无信号，
    #      所以训练日志必须打 adv0 / uniq）。
    sel_masks: list[list[bool]] | None = None
    if dpo_rows is not None:
        if args.select_tokens != "off":
            print("DPO：--select-tokens 不适用（loss 是序列级 margin，不是逐 token CE），按 off 处理")
    elif args.pretrain_data:
        if args.select_tokens != "off":
            print("pretrain 块数据：--select-tokens 不适用（整块 loss 语义），按 off 处理")
    elif args.select_tokens == "excess":
        t_sel = time.time()
        was_training = model.training
        model.eval()
        sel_masks = [guard.select_mask(answer_ref_ce(body, head, full, plen), args.select_thresh)
                     for full, plen in enc]
        if was_training:
            model.train()
        tot_ans = sum(len(m) for m in sel_masks)
        tot_sel = sum(sum(m) for m in sel_masks)
        per = [sum(m) for m in sel_masks]
        print(f"token selection (ref CE > {args.select_thresh}): {tot_sel}/{tot_ans} 答案 token 进 loss"
              f"  每条 {per}  {time.time() - t_sel:.0f}s", flush=True)
        if tot_sel == 0:
            raise SystemExit(f"--select-thresh {args.select_thresh} 下一个 token 都没选中："
                             "阈值过高，或数据与训练起点无差异。先降阈值看分布再跑，不要空转")

    # ---- 报告基线：训练题在「训练起点」的答案 CE（全序列口径，无图）。
    # 训练末会用同一口径再算一遍 → 训练报告里给出「训练题前后变化」。
    # 未训练部分的对照走能力门探针（id/cap/生成体检），两边都要有数才叫完整账目。
    ce_ref: list[float] | None = None            # SFT：每条样本答案段平均 CE
    dpo_ref_ce: list[tuple[float, float]] | None = None   # DPO：每对 (chosen, rejected) 均值
    if not args.pretrain_data:
        was_training = model.training
        model.eval()
        if dpo_enc is None:
            ce_ref = [sum(c) / max(1, len(c)) for c in
                      (answer_ref_ce(body, head, full, plen) for full, plen in enc)]
        else:
            dpo_ref_ce = [(sum(answer_ref_ce(body, head, fc, pc)) / max(1, len(fc) - pc),
                           sum(answer_ref_ce(body, head, fj, pj)) / max(1, len(fj) - pj))
                          for (fc, pc), (fj, pj) in dpo_enc]
        if was_training:
            model.train()

    # ---- 护栏 2：能力门（训练前先记起点：cap 的基线、id 的基准都从这里出）。----
    probes: list[dict] = []
    base_eval: dict | None = None
    final_ev: dict | None = None     # 终评结果（训练报告里「未训练部分」用）
    if args.eval_every > 0:
        if not os.path.exists(args.probes):
            raise SystemExit(f"--eval-every {args.eval_every} 需要探针文件 {args.probes}，不存在")
        probes = [json.loads(line) for line in open(args.probes, encoding="utf-8") if line.strip()]
        if not any(p["kind"] == "identity" for p in probes) or \
           not any(p["kind"] == "cap" and p.get("target") for p in probes):
            raise SystemExit(f"{args.probes} 里 identity / cap(带 target) 探针至少各要有一条")
        was_training = model.training
        model.eval()
        base_eval = eval_gate(body, head, tok, probes, args.word)
        if was_training:
            model.train()
        print(f"capability gate: 起点 id_mean {base_eval['id_mean']:.3f} nats  "
              f"cap_mean {base_eval['cap_mean']:.3f} nats/token  "
              f"({base_eval['n_id']} id + {base_eval['n_cap']} cap 探针；tol {args.cap_tol}, "
              f"stop {args.cap_stop}, every {args.eval_every})", flush=True)

    if args.frozen_prefix:
        cases = list(enc) if fp_enc is not None else [seq for pair in dpo_enc for seq in pair]
        was_training = model.training
        model.eval()
        # tol 1.0 nats 的依据（2026-10-31 diag_frozen.py 全样本实测）：切分点已对齐
        # GDN chunk 边界（消除 chunked delta rule 的块错位差）。残余差异来自「前缀/
        # 尾段与全序列 kernel 形状不同 ⇒ bf16 重结合不同」+「GDN triton atomics 的
        # run-to-run 累加顺序抖动」：逐层 rel ~8e-3（≈2 ulp bf16）从 layer 0 平滑
        # 放大，无单层突刺；逐 token worst CE 差 0.10~0.55 nats（同一样本跨进程可差
        # 4 倍，如 #2: 0.130 vs 0.548），且最大差总落在高熵 token（相对变化 ~12%）。
        # 逻辑错误（切错位/position 错）的量级是 5~30 nats —— 1.0 留一个数量级余量。
        verify_frozen_prefix(body, head, cases, tol=1.0)
        if was_training:
            model.train()

    torch.cuda.reset_peak_memory_stats()  # 训练段口径：加载期的临时缓冲不计入 peak
    # 显存探针（STRATA_MEM_PROBE=/path.pkl）：记录分配栈历史，step 2 结束时落盘。
    # 分析用 memanalyze.py（无 torch 依赖）。探针本身有开销，只用于定位，别开着训练。
    mem_probe = os.environ.get("STRATA_MEM_PROBE", "")
    mem_probe_step = int(os.environ.get("STRATA_MEM_PROBE_STEP", "2"))
    mem_probe_stage = os.environ.get("STRATA_MEM_PROBE_STAGE", "end")   # end=步尾；fwd=backward 前（图活着）
    mem_probe_done = False
    if mem_probe:
        torch.cuda.memory._record_memory_history(context="all", stacks="python",
                                                 max_entries=120000)
        print(f"mem probe: recording (python stacks) → {mem_probe}"
              f"（step {mem_probe_step} stage={mem_probe_stage} 落盘后停止）", flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    rule = BackwardRule(params, lr=args.lr)   # 与零阶共用 step(loss_fn)/backward+finish
    shuffle = torch.Generator().manual_seed(args.seed)   # 固定数据顺序，A/B 才可比
    order, cursor = [], 0
    recent: list[float] = []          # 最近几步的 loss，给 --early-stop 用（跳步不计入）
    stopped_at = None
    hard_stopped = None
    last_step = 0
    best: dict | None = None          # keep-best：{"step", "id_delta", "cap_drift", "sd"}
    train_secs, n_ans_total, n_steps_done = 0.0, 0, 0     # 训练报告口径（步耗时之和 / 答案 token 数）
    prep_secs = time.time() - t_run0 - load_secs          # 编码 + token 选择 + 自检 + 起点评估
    for step in range(1, args.steps + 1):
        last_step = step
        t0 = time.time()
        rule.zero_grad()
        tot, n_ans, n_sel, gn = 0.0, 0, 0, 0.0
        n_skip, n_contrib = 0, 0
        margins: list[float] = []       # DPO：本步每个偏好对的 margin（报告 + acc 用）
        pool = dpo_enc if dpo_enc is not None else enc
        for _ in range(args.accum):
            if cursor >= len(order):
                order = torch.randperm(len(pool), generator=shuffle).tolist()
                cursor = 0
            idx = order[cursor]
            cursor += 1
            if dpo_enc is not None:
                # DPO 步：一对 = chosen + rejected 两次前向（各走一遍专家流），
                # ref 已预计算 ⇒ 每对恰 2 次前向 + 1 次反传，无采样。
                (fc, pc), (fj, pj) = dpo_enc[idx]
                rc, rj = dpo_ref[idx]
                if dpo_fp is not None:
                    # frozen-prefix：prompt 缓存算一次，chosen/rejected 各 deepcopy 一份
                    # —— 前向原地追加 K/V，不克隆就串序列（tail_logprobs 内部克隆）。
                    # margin 只吃答案段：剪掉 chunk 对齐多出来的 prompt 位（两端同前缀
                    # ⇒ 同一 prefix，off 相同）。
                    prefix, (tc, tj) = dpo_fp[idx]
                    cache = prefix_cache(body, prefix)
                    off = (pc - 1) - len(prefix)
                    lc = tail_logprobs(body, head, tc, cache)[off:].sum()
                    lj = tail_logprobs(body, head, tj, cache)[off:].sum()
                else:
                    lc = policy_logprob_sum(body, head, fc, pc)
                    lj = policy_logprob_sum(body, head, fj, pj)
                # margin 按答案 token 归一（与引擎侧 mezo 的 DpoMezo 同一语义，objectives
                # 唯一出处）：量级与序列长度无关，lr 不用随长度重标定。
                margin = objectives.dpo_margin(lc, lj, rc, rj,
                                               int(len(fc) - pc), int(len(fj) - pj))
                # margin loss 走 objectives（softplus 形式：β·margin 很大时 log(1) 的梯度是 0/0）
                loss = objectives.dpo_loss(margin, args.dpo_beta) / args.accum
                tot += rule.backward(loss) * args.accum
                margins.append(float(margin))
                n_ans += (len(fc) - pc) + (len(fj) - pj)
                n_contrib += 1
                continue
            full, plen = enc[idx]
            if fp_enc is not None:
                # frozen-prefix：前缀缓存（本步权重下的精确前向，no_grad）+ 尾段带图。
                # lp 剪掉 chunk 对齐多出的 prompt 位后恰是全部答案 token。
                prefix, tail = fp_enc[idx]
                cache = prefix_cache(body, prefix)
                lp = tail_logprobs(body, head, tail, cache)[(plen - 1) - len(prefix):]
                mask = torch.ones_like(lp, dtype=torch.bool)
                if sel_masks is not None:
                    if len(sel_masks[idx]) != lp.numel():
                        raise SystemExit(f"token 选择长度 {len(sel_masks[idx])} != "
                                         f"尾段 token 数 {lp.numel()}（切分与选择不是同一份编码）")
                    mask = mask & torch.tensor(sel_masks[idx], device=device, dtype=torch.bool)
                    if not mask.any():
                        n_skip += 1     # 这条样本的答案基座全都会：没有要学的东西，整条跳过
                        continue
                loss = objectives.sft_loss(lp[mask]) / args.accum
            else:
                ids = torch.tensor([full], device=device)
                h = body(input_ids=ids[:, :-1], use_cache=False).last_hidden_state[0]
                labels = ids[0, 1:]
                mask = torch.zeros_like(labels, dtype=torch.bool)
                mask[plen - 1:] = True
                if sel_masks is not None:
                    mask = mask & torch.tensor(guard.expand_to_labels(sel_masks[idx], labels.shape[0], plen),
                                               device=device, dtype=torch.bool)
                    if not mask.any():
                        n_skip += 1     # 这条样本的答案基座全都会：没有要学的东西，整条跳过
                        continue
                lp = token_logprobs(h, head, labels, chunk=128)
                loss = objectives.sft_loss(lp[mask]) / args.accum
            if mem_probe and mem_probe_stage == "fwd" and step == mem_probe_step and not mem_probe_done:
                # 快照打在 backward 之前：无 ckpt 时整张激活图都活着 —— 找「T 无关的
                # 激活大头」就要在这一刻看活块构成（步尾快照里图已经释放，看不到）。
                torch.cuda.memory._dump_snapshot(mem_probe)
                torch.cuda.memory._record_memory_history(None)
                mem_probe_done = True
                print(f"mem probe: snapshot dumped at step {step} (fwd, graph alive) → {mem_probe}",
                      flush=True)
            tot += rule.backward(loss) * args.accum
            # 答案 token（选择前口径，tok/s 与历史可比）：frozen 路径 lp 已是答案段
            n_ans += lp.numel() if fp_enc is not None else labels.numel() - (plen - 1)
            n_sel += int(mask.sum())
            n_contrib += 1
        if n_contrib == 0:
            print(f"step {step}: {args.accum} 条样本全被 token 选择跳过 —— 不更新参数、"
                  f"不计入 early-stop（loss=0 会假触发）", flush=True)
            continue
        _fin = rule.finish()
        gn = _fin["grad_norm"]
        dt = time.time() - t0
        train_secs += dt
        n_ans_total += n_ans
        n_steps_done += 1
        peak = torch.cuda.max_memory_allocated() / 2**30
        if dpo_enc is not None:
            m_loss, m_acc = dpo.dpo_batch(margins, args.dpo_beta)
            print(f"step {step}: dpo loss {m_loss:.4f}  acc {m_acc:.0%}  "
                  f"margin均值 {sum(margins) / len(margins):+.3f}  pairs {len(margins)}  "
                  f"tok/s {n_ans / dt:.1f}  grad_norm {gn:.4f}  {dt:.1f}s  peak {peak:.1f} GiB",
                  flush=True)
            recent.append(m_loss)
        else:
            sel_txt = f"  sel {n_sel}/{n_ans}" if sel_masks is not None else ""
            skip_txt = f"  skip {n_skip}" if n_skip else ""
            print(f"step {step}: loss {tot / args.accum:.4f}  loss_tokens {n_ans}{sel_txt}{skip_txt}  "
                  f"tok/s {n_ans / dt:.1f}  grad_norm {gn:.4f}  "
                  f"{dt:.1f}s  peak {peak:.1f} GiB",
                  flush=True)
            recent.append(tot / args.accum)
        if mem_probe and mem_probe_stage == "end" and step == mem_probe_step:
            torch.cuda.memory._dump_snapshot(mem_probe)
            torch.cuda.memory._record_memory_history(None)   # 停止记录，后续步零开销
            print(f"mem probe: snapshot dumped at step {step} → {mem_probe}", flush=True)
        if (args.early_stop > 0 and step >= args.min_steps and len(recent) >= 3
                and sum(recent[-3:]) / 3 < args.early_stop):
            stopped_at = step
            print(f"early stop at step {step}: mean loss of last 3 steps "
                  f"{sum(recent[-3:]) / 3:.4f} < --early-stop {args.early_stop}  "
                  f"(样本已收敛；再训就是在背样本，无关能力会跟着坏)", flush=True)
            break
        # ---- 护栏 2：能力门（周期性、无图；id 好且能力没越界才配做 best）----
        if base_eval is not None and step % args.eval_every == 0:
            was_training = model.training
            model.eval()
            ev = eval_gate(body, head, tok, probes, args.word)
            if was_training:
                model.train()
            id_delta = ev["id_mean"] - base_eval["id_mean"]
            cap_drift = ev["cap_mean"] - base_eval["cap_mean"]
            v = guard.gate_verdict(id_delta, cap_drift,
                                   best["id_delta"] if best else None, args.cap_tol,
                                   gen_ok=ev["gen_ok"])
            gen_txt = "  ".join(f"{r['text'][:22]!r}/d3={r['distinct3']}/eos={int(r['eos'])}"
                                for r in ev["gen"])
            print(f"gate step {step}: idΔ {id_delta:+.3f} nats  capΔ {cap_drift:+.3f} nats/tok  → {v}"
                  f"\n    gen: {gen_txt}", flush=True)
            if guard.should_hard_stop(cap_drift, args.cap_stop):
                hard_stopped = step
                print(f"capability hard stop at step {step}: capΔ {cap_drift:.3f} > "
                      f"--cap-stop {args.cap_stop}（能力在坏，立刻停；存盘前恢复 best 快照）", flush=True)
                break
            if v == "accept":
                best = {"step": step, "id_delta": id_delta, "cap_drift": cap_drift,
                        "sd": snapshot_trainable(model)}

    # ---- 能力门收尾：终评 + 恢复 keep-best，再走存盘 ----
    if base_eval is not None:
        was_training = model.training
        model.eval()
        ev = eval_gate(body, head, tok, probes, args.word)
        final_ev = ev
        if was_training:
            model.train()
        id_delta = ev["id_mean"] - base_eval["id_mean"]
        cap_drift = ev["cap_mean"] - base_eval["cap_mean"]
        v = guard.gate_verdict(id_delta, cap_drift,
                               best["id_delta"] if best else None, args.cap_tol,
                               gen_ok=ev["gen_ok"])
        gen_txt = "  ".join(f"{r['text'][:22]!r}/d3={r['distinct3']}/eos={int(r['eos'])}"
                            for r in ev["gen"])
        print(f"gate final (step {last_step}"
              + (f", early stop" if stopped_at else "")
              + (f", hard stop" if hard_stopped else "")
              + f"): idΔ {id_delta:+.3f} nats  capΔ {cap_drift:+.3f} nats/tok  → {v}"
              f"\n    gen: {gen_txt}", flush=True)
        if v == "accept":
            best = {"step": last_step, "id_delta": id_delta, "cap_drift": cap_drift,
                    "sd": snapshot_trainable(model)}
        if best is not None and best["step"] != last_step:
            missing = restore_trainable(model, best["sd"])
            print(f"restored keep-best snapshot: step {best['step']} "
                  f"(idΔ {best['id_delta']:+.3f}, capΔ {best['cap_drift']:+.3f})"
                  + (f"  未命中 {missing}" if missing else ""), flush=True)
        elif best is None:
            print("capability gate 从未 accept：没有 best 可恢复，产物按当前参数保存 —— "
                  "身份没学会或能力已坏，请人工核对验收", flush=True)

    lp1 = name_logprob(body, head, tok, args.probe, args.word)
    print(f"after:  logp('{args.word}' | {args.probe!r}) = {lp1:.4f} (p={math.exp(lp1):.8f})   "
          f"Δ={lp1 - lp0:+.4f} nats", flush=True)

    # ---- 训练报告：速度 / 用时 / 训练前后效果（训练题 + 未训练探针）----
    # 每次训练都必须留下这三样，否则「练了有没有用、快不快、有没有伤到别的能力」无从判断。
    run_secs = time.time() - t_run0
    post_secs = max(0.0, run_secs - load_secs - prep_secs - train_secs)
    print("\n===== 训练报告 =====", flush=True)
    print(f"训练: {n_steps_done} 步 / 影响答案 token {n_ans_total} / 用时 {train_secs:.1f} s / "
          f"tok/s {n_ans_total / train_secs if train_secs else 0:.1f} / "
          f"峰值 {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    if final_ev is not None and final_ev["gen_tokens"]:
        print(f"生成: {final_ev['gen_tokens']} tokens / {final_ev['gen_secs']:.1f} s / "
              f"tok/s {final_ev['gen_tok_s']:.2f}（终评贪心解码，含 prefill）", flush=True)
    else:
        print("生成: 未测（--eval-every 0 时没有探针评估）", flush=True)
    print(f"总用时 {run_secs:.0f} s = 加载 {load_secs:.0f} + 数据准备/自检 {prep_secs:.0f} + "
          f"训练 {train_secs:.0f} + 终评/存盘 {post_secs:.0f}", flush=True)

    def _ce_now(full, plen):
        return answer_ref_ce(body, head, full, plen)

    was_training = model.training
    model.eval()
    if ce_ref is not None:
        ce_now = [sum(c) / max(1, len(c)) for c in (_ce_now(f, p) for f, p in enc)]
        print("训练题效果（答案 CE nats/token，越小越好；全序列口径 = 与评估同尺）:", flush=True)
        for i, (a, b) in enumerate(zip(ce_ref, ce_now)):
            print(f"  #{i}  len {lens[i]:3d}  {a:6.3f} → {b:6.3f}  ({b - a:+.3f})", flush=True)
        m0, m1 = sum(ce_ref) / len(ce_ref), sum(ce_now) / len(ce_now)
        print(f"  均值         {m0:6.3f} → {m1:6.3f}  ({m1 - m0:+.3f})", flush=True)
    elif dpo_ref_ce is not None:
        now = [ (sum(_ce_now(fc, pc)) / max(1, len(fc) - pc),
                 sum(_ce_now(fj, pj)) / max(1, len(fj) - pj))
                for (fc, pc), (fj, pj) in dpo_enc]
        print("偏好对效果（CE nats/token；chosen 该降、rejected 该升）:", flush=True)
        for i, ((c0, j0), (c1, j1)) in enumerate(zip(dpo_ref_ce, now)):
            print(f"  pair#{i}  chosen {c0:6.3f} → {c1:6.3f} ({c1 - c0:+.3f})   "
                  f"rejected {j0:6.3f} → {j1:6.3f} ({j1 - j0:+.3f})", flush=True)
    if was_training:
        model.train()

    if base_eval is not None and final_ev is not None:
        print(f"未训练部分（能力门探针：{base_eval['n_id']} id + {base_eval['n_cap']} cap，"
              f"跨域金丝雀，不该坏）:", flush=True)
        print(f"  id_mean  {base_eval['id_mean']:+.3f} → {final_ev['id_mean']:+.3f} nats  "
              f"(Δ{final_ev['id_mean'] - base_eval['id_mean']:+.3f}，目标词概率应上升)", flush=True)
        print(f"  cap_mean {base_eval['cap_mean']:.3f} → {final_ev['cap_mean']:.3f} nats/tok  "
              f"(Δ{final_ev['cap_mean'] - base_eval['cap_mean']:+.3f}，越界阈值 {args.cap_tol})", flush=True)
        print(f"  生成体检 {sum(1 for r in final_ev['gen'] if r['ok'])}/{len(final_ev['gen'])} ok"
              f"（distinct-3gram + EOS + 无答案串泄漏）", flush=True)

    # ---- 身份问答的实际文本（训练后）：事实对不对只能靠眼看 ----
    was_training = model.training
    model.eval()
    id_gen_after = sample_identity(body, head, tok)
    if was_training:
        model.train()
    print("身份问答（训练后，贪心解码；与训练前逐条对照）:", flush=True)
    for q, b, a in zip(IDENTITY_QS, id_gen_before, id_gen_after):
        print(f"  {q}\n    before: {b[:150]}\n    after:  {a[:150]}", flush=True)

    if args.lora_method == "unilora":
        import unilora

        # 物化成稠密 A/B：键名与 lora.py 约定一致，lora_to_gguf.py 直接可读。
        # 引擎侧缩放 s = alpha/rank，导出时传 --alpha <rank> 才能逐值还原训练时的 delta。
        sd = unilora.materialize_ab(model)
        ck = unilora.unilora_ckpt(model, proj_seed=args.proj_seed, theta_d_length=args.theta_d,
                                  init_bound=args.init_bound)
        torch.save(ck, args.out + ".unilora.pt")
    else:
        sd = {k: v.detach().cpu() for k, v in model.state_dict().items() if k.endswith(".A") or k.endswith(".B")}
    torch.save(sd, args.out)
    nz = sum(1 for v in sd.values() if float(v.abs().max()) > 0)
    print(f"saved {args.out}: {len(sd)} tensors, non-zero {nz}, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
          flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException as _exc:                  # noqa: BLE001
        if mem_guard.is_oom(_exc):
            raise mem_guard.die_on_oom(_exc, where=__file__.rsplit("/", 1)[-1])
        raise
