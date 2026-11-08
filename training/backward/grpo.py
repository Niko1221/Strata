"""自己的 GRPO：Strata 出轨迹，学生在本地前向/反向，不用 TRL、不用 unsloth。

关于 old logprob 的来源（这是最容易出错的地方）：
  - `local`：在同一批 token 上用学生自己无梯度前向重算一遍。此时第 1 步 ratio 恰为 1，
    梯度就是标准策略梯度；轨迹被当作"学生自己采的"。引擎只负责产生 token。
  - `engine`：用采样时引擎记录的 logprob。ratio 会偏离 1，偏离量就是"引擎与学生的差距"：
    它既包含这之后策略的变化，也包含**量化差** —— 引擎跑 GGUF Q4_K_XL，学生是自实现的 NF4
    专家 + bf16 其余，两者本来就不是同一个分布（实测 ratio 1.0115）。

两种都写出来，是因为它们回答的是两个不同的问题：
"引擎能不能当 teacher" 和 "训练流程能不能自己跑"。

两段式（48GB 卡放不下引擎与学生）：
  1. `collect_rollouts.py`（不需要 torch，可在 Windows 跑，引擎在跑）
  2. 关掉引擎，本脚本 `--rollouts <jsonl> --old-logprobs engine`
     —— `--no-student` 是第 1 步的老写法，需要 torch 才能 import 本文件。
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


import argparse
import json
import math
import os
import sys
import time

import torch

from ce import token_logprobs
import objectives
from lora import (DEFAULT_TARGETS, freeze_all, inject_lora, load_lora_state,
                  lora_state_dict, trainable_summary)


# ---------------------------------------------------------------- 学生模型

def _text_model(model):
    """拿到输出 hidden_states 的子模块：35B 是 model.model.language_model，
    普通文本模型是 model.model。"""
    inner = getattr(model, "model", None)
    if inner is None:
        return model
    return getattr(inner, "language_model", inner)


def load_student(path: str, four_bit_experts: bool, gguf=None, pack=None, tokenizer=None):
    """两条加载路径：

    * `--gguf/--pack`：125B 的 IQ3_XXS + NF4 专家包（loader_qwen4），这台机器上唯一真能跑的
      权重；本地 HF cache 里只有 config.json，没有 safetensors。
    * `--model <HF dir>`：原来的路，需要完整 HF 权重。
    """
    from transformers import AutoTokenizer

    if gguf:
        if not pack:
            raise SystemExit("--gguf needs --pack (experts_nf4.bin)")
        import platform_check
        from loader_qwen4 import load_model

        tok_dir = tokenizer or os.path.join(os.path.dirname(os.path.abspath(gguf[0])), "tokenizer")
        tok = AutoTokenizer.from_pretrained(tok_dir)
        model, _meta = load_model(list(gguf), pack)
        platform_check.warn_if_gdn_fell_back(log=print)
        # no-ckpt 反传必须开专家 refetch：`nf4_fused` 默认把 packed/absmax 存进 ctx，
        # 而 tiered pack 的 buffer 在层间被复用 ⇒ 反传到该层时版本号已经变了，
        # 直接 RuntimeError("modified by an inplace operation", [2560, 320])。
        # 开了以后 backward 按专家 id 现取（每专家多一次 2.49 MiB H2D）。
        # loader 在**首次 forward** 才读这个开关，所以放在 load_model 之后仍有效。
        os.environ.setdefault("STRATA_EXPERT_REFETCH", "1")
        model.config.use_cache = False
        return model, tok

    if not path:
        raise SystemExit("--model (HF dir) or --gguf + --pack is required")
    from transformers import AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(path)
    if four_bit_experts:
        from loader import build_skeleton, materialize, nbytes_of, stream_load

        device = torch.device("cuda")
        model, experts = build_skeleton(path)
        materialize(model, experts, device, torch.bfloat16)
        stream_load(model, experts, path, device, log=lambda *a, **k: None)
        print("student:", {k: round(v, 2) for k, v in nbytes_of(model, experts).items()}, flush=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map={"": "cuda"})
    model.config.use_cache = False
    return model, tok


# ---------------------------------------------------------------- 打分

def reward_answer(completion: str, target: str) -> float:
    text = (completion or "").strip().strip("。.!?！？\"'`*_ \n\t").lower()
    return 1.0 if target and text == target else 0.0


def reward_of(traj: dict) -> float:
    """kind=exact（默认）严格相等；kind=words:N 要求词数恰好为 N；kind=wordsclose:N 按词数差给分；
    kind=num 取回答里第一个整数与 target 比较（数学题的可验证奖励）。

    wordsclose:N = max(0, 1 - |n-N|/N)，只要长度飘就有梯度；words:N 常常组内四条全对或全错，
    std=0 ⇒ 优势全 0 ⇒ grad_norm=0 —— "跑完了"并不等于"训练生效了"。
    kind=num 同样要防全组全错：选进 --prompts 的题必须先经 math_probe.py 证明
    基座在 T=1 下有 0 < pass_rate < 1（有方差组才有梯度，没有就用 DPO 路线）。
    """
    text = (traj.get("content") or "").strip()
    kind = traj.get("kind") or "exact"
    if kind.startswith("wordsclose:"):
        want = int(kind.split(":", 1)[1])
        return max(0.0, 1.0 - abs(len(text.split()) - want) / max(1, want))
    if kind.startswith("words:"):
        want = int(kind.split(":", 1)[1])
        return 1.0 if len(text.split()) == want else 0.0
    if kind == "num":
        import re

        want = str(traj.get("target") or "").strip()
        m = re.search(r"-?\d+", text.replace(",", ""))
        return 1.0 if want and m and m.group(0) == want else 0.0
    return reward_answer(text, traj.get("target") or prompt_target(traj.get("prompt", "")))


def prompt_target(prompt: str) -> str:
    caps = [w.strip(".,:;!?\"'") for w in prompt.split()]
    caps = [w for w in caps if w.isalpha() and w.isupper()]
    return caps[-1].lower() if caps else ""


def check_trainable_lengths(trajs, max_seq, max_skip_frac=0.0, tok=None, log=print) -> dict:
    """开跑前的长度门：**不可训练就报错，不静默筛样本**。

    判据与 `sft.py` 的 fail-fast 一致（那里超预算 `SystemExit(2)` 整批退出）：
    「能不能训」是前置问题，「训完会不会退化」才是后验评测。GRPO 的轨迹长度不可控，
    所以不能像 SFT 那样整批死掉，但也绝不能悄悄丢掉最长的那批 —— 否则训出来的是
    「短轨迹子集上的模型」，日志里还看不出被筛过（这正是 `--max-seq 768` 训数学的坑：
    默认值是给身份训练（`--max-new 96`）配的，换成几千 token 的推理链就会大面积超限）。

    三级处置（`max_skip_frac` = 允许被丢掉的轨迹比例，默认 0 = 一条都不许丢）：
      ① `min > max_seq`（一条都放不下）⇒ 必然 100% 丢光，报错；
      ② `n_toolong / n > max_skip_frac` ⇒ 报错，并给出两条改法；
      ③ 比例在界内 ⇒ 打印明细后继续（长尾是真实存在的，但必须让人看见比例）。

    空 completion（`token_ids` 为空）单独计数、不进门：那是采样失败，
    不是「长度不可训练」（`--think` 开着把 content 挤空是典型成因）。

    缺 `prompt_ids` / `token_ids` 且没给 `tok` 时判不了，返回 `known=False` 由调用方
    在训练循环里逐条统计 —— 不猜、不静默。
    """
    lens, n_empty, est = [], 0, []
    for t in trajs:
        comp = t.get("token_ids")
        if not comp:
            n_empty += 1
            continue
        p = t.get("prompt_ids")
        if p:
            lens.append(len(p) + len(comp))
        elif tok is not None:
            # 与 run_grpo 主循环的兜底同源：老引擎不产 prompt_ids 时用本地模板补
            pids = tok.apply_chat_template([{"role": "user", "content": t["prompt"]}],
                                           add_generation_prompt=True, return_dict=False)
            lens.append(len(pids) + len(comp))
        else:
            est.append(1)          # 判不了 —— 不猜，交训练循环逐条统计
    n = len(lens)
    st = {"n": n, "n_all": len(trajs), "n_empty": n_empty, "n_toolong": 0,
          "max": max(lens) if lens else 0, "p50": sorted(lens)[n // 2] if lens else 0,
          "known": n > 0 and not est, "max_seq": max_seq}
    if st["n_empty"]:
        log(f"⚠ {n_empty}/{len(trajs)} 条轨迹的 completion 为空，不参与训练"
            f"（采样失败，不是长度问题；`--think` 开着时短 `--max-new` 会把 content 挤空）")
    if n == 0:
        log("长度门：没有可比对的轨迹长度（全部为空或缺字段），交训练循环逐条统计")
        return st

    over = [x for x in lens if x > max_seq]
    st["n_toolong"] = len(over)
    frac = len(over) / n
    if over:
        log(f"长度门：{len(over)}/{n} 条超 `--max-seq {max_seq}`"
            f"（{frac:.1%}，实际最长 {max(lens)}，p50 {st['p50']}）")
    if not st["known"]:
        log("长度门：部分轨迹缺 `prompt_ids` 且未加载 tokenizer，长度按引擎回显口径"
            "判定；训练循环内会再逐条统计一次")
    if len(over) == n and st["known"]:
        raise SystemExit(
            f"--max-seq {max_seq} 放不下任何一条轨迹（最短 {min(lens)}、最长 {max(lens)}）——"
            f" 这个配置一条都训不了，不是「少量长尾」。改法：① 调大 --max-seq 到 ≥ {max(lens)}"
            f"（显存上限见 mem_guard：sft 一条 8 192 就要 42 GiB）；② 或确认 --max-new 是否过大"
            f"（当前采样允许生成这么长，训练侧却拒收，两者是矛盾的）")
    if frac > max_skip_frac and st["known"]:
        raise SystemExit(
            f"--max-seq {max_seq} 会丢掉 {len(over)}/{n} 条轨迹（{frac:.1%} > 允许的"
            f" {max_skip_frac:.0%}）—— 训练集会变成「短轨迹子集」，结果不能代表目标任务。"
            f"改法：① 调大 --max-seq 到 ≥ p95（本批 p95≈"
            f"{sorted(lens)[min(n - 1, int(n * 0.95))]}）或 ≥ {max(lens)} 一条不丢；"
            f"② 若确实想接受这个比例，显式给 --max-skip-frac {frac:.2f}")
    return st


def run_grpo(model, tok, trajs, cfg, *, group, batch, max_seq, max_new, old_logprobs,
             mask_truncated, epochs, steps, opt, params, log=print, objective="grpo",
             max_skip_frac=0.0):
    """一个 `GrpoCfg` 的训练循环；返回最后一步的指标（整批没梯度时返回空 dict）。

    抽成函数的理由：`_verify_grpo_switches.py` 要在**一次加载**里对同一批轨迹跑多个
    cfg 做 A/B。复制一份循环等于有两份"GRPO 到底怎么算"，迟早漂移。
    """
    head = model.lm_head
    body = _text_model(model)
    eos_id = tok.eos_token_id
    last: dict = {}

    # 长度门先行：不可训练就报错（这里已加载 tokenizer，判定是精确的）
    len_stat = check_trainable_lengths(trajs, max_seq, max_skip_frac, tok=tok, log=log)

    for step in range(1, steps + 1):
        sl = trajs[(step - 1) * batch * group : step * batch * group]
        if not sl:
            sl = trajs[: batch * group]
        rewards = torch.tensor([t["reward"] for t in sl])
        # GRPO 的基线是**同一个 prompt 的那几条**，不是整批：跨 prompt 归一化会把"这道题全错"
        # 和"那道题全对"混成一个优势（数学上不是 GRPO）。组内 std=0 时这一组的优势就该是 0，
        # 所以这里把有效的组数也打出来 —— 它是"这一步到底有没有梯度"的唯一判据。
        adv_all, groups_used, zero_std = objectives.group_advantages(
            rewards, group, scale=cfg.scale_rewards)

        # 先把这一批的序列准备好，old logprob 只算一次（这才是"旧策略"）
        prepared = []
        total_tokens = 0      # DAPO 口径的分母：整批有效 token 数
        n_trunc = 0
        n_tis = 0
        n_toolong = 0
        n_empty = 0
        for i, t in enumerate(sl):
            # 引擎回显的 prompt_ids 就是它实际读的那一串；本地模板只在老引擎（不产 LP、也就不回显）时兜底
            prompt_ids = t.get("prompt_ids") or tok.apply_chat_template(
                [{"role": "user", "content": t["prompt"]}], add_generation_prompt=True, return_dict=False
            )
            comp_ids = t["token_ids"] or tok(t["content"], add_special_tokens=False).input_ids
            ids = list(prompt_ids) + list(comp_ids)
            if not comp_ids:
                n_empty += 1      # 采样失败（空 completion），与「长度不可训练」是两回事
                continue
            if len(ids) > max_seq:
                n_toolong += 1    # 入口的长度门已挡过；这里只统计本步实际丢了几条
                continue
            # 撞到 max_new 且最后一个 token 不是 eos => 被切断。它已经算进优势了（0 分是真实的
            # 采样反馈），但不产生梯度：否则等于教模型"没说完=说错"。
            if mask_truncated == "on" and objectives.is_truncated(comp_ids, max_new, eos_id):
                n_trunc += 1
                continue
            with torch.no_grad():
                ids_t = torch.tensor([ids], device="cuda")
                n_prompt = len(prompt_ids)
                labels = ids_t[0, n_prompt:]
                hidden = body(input_ids=ids_t, use_cache=False).last_hidden_state[0, n_prompt - 1 : -1]
                local_old = token_logprobs(hidden, head, labels, chunk=128).clone()
            old = local_old
            tis = None
            if old_logprobs == "engine" and t.get("token_logprobs"):
                cand = torch.tensor(t["token_logprobs"], device="cuda", dtype=torch.float32)
                if cand.numel() == local_old.numel():
                    old = cand
                    if cfg.tis_cap > 0:
                        # pi_训练侧旧参数 / pi_引擎生成：TIS 要修的正是这一段分布差。
                        # 它是常数（不含策略更新），所以下面进 loss 前会 detach + clamp。
                        tis = torch.exp(local_old - cand).detach()
                        n_tis += 1
            # 带着原始下标一起存：上面会 `continue` 掉超长/空回答/被截断的轨迹，
            # 用 enumerate(prepared) 的序号去索引 adv_all 会让后面每条都取到别人的优势
            prepared.append((i, ids, n_prompt, labels, old, tis))
            total_tokens += int(labels.numel())

        # 逐条 backward 时必须用**整批**的分母，否则等于把 DAPO 的全局归一悄悄换成逐条平均
        # luspo 的分母是**序列数**（每条轨迹等权），这正是它跟 grpo 的唯一实质差别
        is_luspo = objective == "luspo"
        if is_luspo:
            denom = len(prepared)
        else:
            denom = total_tokens if cfg.norm == "token" else len(prepared)
        loss_fn = objectives.luspo_loss if is_luspo else objectives.grpo_loss

        # GRPO 原样：old 取自本批次开始前的策略，之后每个 inner epoch 都真正更新权重，
        # 于是 epoch>=2 的 ratio 必然偏离 1，裁剪项才开始起作用。
        for epoch in range(1, epochs + 1):
            opt.zero_grad(set_to_none=True)
            e_loss = e_pg = 0.0
            e_used = 0
            e_ratios = []
            e_clipped = 0.0
            e_tis = 0.0
            for i, ids, n_prompt, labels, old, tis in prepared:
                adv = adv_all[i].item()
                ids_t = torch.tensor([ids], device="cuda")
                hidden = body(input_ids=ids_t, use_cache=False).last_hidden_state[0, n_prompt - 1 : -1]
                new_lp = token_logprobs(hidden, head, labels, chunk=128)
                ones = torch.ones_like(new_lp).unsqueeze(0)
                loss, st = loss_fn(
                    new_lp.unsqueeze(0), old.unsqueeze(0),
                    torch.full_like(new_lp, adv).unsqueeze(0), ones, cfg,
                    tis=None if tis is None else tis.unsqueeze(0), denom=denom)
                loss.backward()
                e_loss += float(loss)
                e_pg += -(adv * new_lp).sum().item() / denom   # 未裁剪的对照口径
                e_ratios.append(st["ratio"])
                e_clipped += st["clip_frac"]
                e_tis += st["tis"]
                e_used += 1
            if e_used == 0:
                log(f"step {step}: 整批没有可用轨迹（trunc {n_trunc} / toolong {n_toolong}"
                    f" / empty {n_empty}）—— 这一步没有梯度")
                break
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            last = {
                "step": step, "epoch": epoch,
                "pg_obj": e_pg, "loss": e_loss, "grad_norm": float(gn),
                "ratio": sum(e_ratios) / len(e_ratios),
                "clip_frac": e_clipped / e_used,
                "tis": e_tis / e_used, "n_tis": n_tis,
                "reward": float(rewards.mean()), "reward_std": float(rewards.std()),
                "zero_std": zero_std, "groups": groups_used,
                "used": e_used, "batch": len(sl), "trunc": n_trunc,
                "toolong": n_toolong, "empty": n_empty,
                "len_check": len_stat,
                "tokens": total_tokens, "denom": denom,
                "peak": torch.cuda.max_memory_allocated() / 2 ** 30,
            }
            log(
                f"step {step}.{epoch}: pg_obj {e_pg:.4f}  clipped_obj {e_loss:.4f}  "
                f"grad_norm {float(gn):.4f}  ratio {last['ratio']:.4f}  "
                f"clip_frac {last['clip_frac']:.3f}  tis {last['tis']:.4f}({n_tis})  "
                f"reward {last['reward']:.3f}  "
                f"reward_std {last['reward_std']:.4f}  zero_std {zero_std}  groups {groups_used}  "
                f"used {e_used}/{len(sl)}  trunc {n_trunc}  toolong {n_toolong}"
                f"  empty {n_empty}  tokens {total_tokens}  "
                f"peak {last['peak']:.1f} GiB"
            )
    return last



# ---------------------------------------------------------------- 主循环

def main() -> int:
    import platform_check

    platform_check.require()          # 平台不支持就停在这，别等读完 76 GB pack
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="HF 目录；与 --gguf/--pack 二选一")
    ap.add_argument("--gguf", nargs="+", default=None,
                    help="GGUF 分片（125B 要两个）；走 loader_qwen4，专家从 --pack 流式取")
    ap.add_argument("--pack", default=None, help="experts_nf4.bin")
    ap.add_argument("--tokenizer", default=None,
                    help="HF tokenizer 目录；缺省取 <第一个 gguf 的目录>/tokenizer")
    ap.add_argument("--engine", default="http://127.0.0.1:8095")
    ap.add_argument("--engine-model", default=None, help="引擎侧的 model 名，默认取 /v1/models 第一个")
    ap.add_argument("--four-bit-experts", action="store_true")
    ap.add_argument("--prompts", help="jsonl，每行 {\"prompt\":..., \"target\":...}；采轨迹时必需，--rollouts 时不需要")
    ap.add_argument("--rollouts", default=None, help="离线轨迹 jsonl；给了就不连引擎")
    ap.add_argument("--save-rollouts", default=None)
    ap.add_argument("--no-student", action="store_true",
                    help="只采轨迹就退出：48GB 卡放不下引擎（28.6GB）+ 学生（21.8GB），"
                         "所以先在这张卡上采轨迹、关掉引擎，再用 --rollouts 训")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--group", type=int, default=4, help="每条 prompt 采样几条")
    ap.add_argument("--batch", type=int, default=2, help="每步用几条 prompt")
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--max-seq", type=int, default=768,
                    help="训练侧接受的单条序列长度上限（prompt+completion）。默认 768 是配 "
                         "`--max-new 96` 的身份训练用的；训数学/长推理链必须显式调大，"
                         "否则轨迹超限会被丢（超限比例超过 --max-skip-frac 直接报错，见 `check_trainable_lengths`）")
    ap.add_argument("--max-skip-frac", type=float, default=0.0, metavar="F",
                    help="允许被 `--max-seq` 丢掉的轨迹比例，默认 0 = 一条都不许丢（超了报错）。"
                         "GRPO 轨迹长度不可控，确有长尾时要显式给这个值，别指望静默筛样本")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-method", choices=("lora", "unilora"), default="lora",
                    help="unilora=Uni-LoRA（全模型只训一个共享向量，见 unilora.py）")
    ap.add_argument("--clip", type=float, default=0.2, help="上下界同时设置（简化写法）")
    ap.add_argument("--clip-low", type=float, default=None,
                    help="ratio 下界，默认 = --clip")
    ap.add_argument("--clip-high", type=float, default=None,
                    help="ratio 上界，默认 = --clip。DAPO 用 0.28：向上概率的估计噪声更大，"
                         "卡 0.2 会把「已经变好」的 token 裁掉")
    ap.add_argument("--norm", choices=("token", "seq"), default="token",
                    help="token=DAPO（全批有效 token 平均，长短回答等权，默认）；"
                         "seq=长度归一化（旧行为，短回答单位 token 权重更高）")
    ap.add_argument("--objective", choices=("grpo", "luspo"), default="grpo",
                    help="luspo=GSPO 序列级 ratio 且分母用序列数（loss 乘 |y_i|），"
                         "长度无偏；此时 --norm 不参与（两种口径都由 luspo 定义）")
    ap.add_argument("--tis-cap", type=float, default=2.0,
                    help="截断重要性采样的上限，0=关。只在 --old-logprobs engine 时生效："
                         "old 来自引擎 GGUF、logp 来自学生 NF4，两者不是同一分布（实测 ratio 1.0115），"
                         "不修的话裁剪项学的是量化差而不是策略差")
    ap.add_argument("--scale-rewards", choices=("on", "off"), default="on",
                    help="off=Dr.GRPO（优势不除组内 std）：近似全对的组 std 极小，"
                         "除下去会把微小差异放大成满幅优势")
    ap.add_argument("--mask-truncated", choices=("on", "off"), default="on",
                    help="被 max_new 切断的回答整条不进 loss：它在精确奖励下必是 0 分，"
                         "当负样本学等于教模型「没说完不对」")
    ap.add_argument("--old-logprobs", choices=("local", "engine"), default="local")
    ap.add_argument("--epochs", type=int, default=2, help="每个 batch 内层过几遍；>1 才能看到 ratio 偏离 1 与裁剪生效")
    ap.add_argument("--think", action="store_true", help="默认关思考：max_tokens 覆盖思考+回答，"
                                                        "开着时短预算会把 content 挤成空串")
    ap.add_argument("--out", default="lora_out.pt")
    ap.add_argument("--init-adapter", default="",
                    help="从上一阶段的 adapter 起训（课程续训，D 阶段用）")
    args = ap.parse_args()

    model, tok = None, None      # loaded after the rollouts: the engine and the 4-bit student do not fit together
    tasks = [json.loads(line) for line in open(args.prompts, encoding="utf-8") if line.strip()] if args.prompts else []
    if not tasks and not args.rollouts:
        raise SystemExit("--prompts is required unless --rollouts is given")

    # ---------------- 轨迹 ----------------
    engine_meta = None   # 引擎侧吞吐（tokens / 逐请求墙钟和），随汇总一起报
    if args.rollouts:
        trajs = [json.loads(line) for line in open(args.rollouts, encoding="utf-8") if line.strip()]
        try:
            with open(args.rollouts + ".meta.json", encoding="utf-8") as f:
                engine_meta = json.load(f)
        except FileNotFoundError:
            pass
        print(f"loaded {len(trajs)} offline trajectories", flush=True)
    else:
        import strata_client

        engine_model = args.engine_model
        if engine_model is None:
            import urllib.request

            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(args.engine.rstrip("/") + "/v1/models", timeout=30) as r:
                engine_model = json.loads(r.read())["data"][0]["id"]
        trajs = []
        t0 = time.time()
        gen_tokens = gen_seconds = 0   # 引擎侧吞吐：逐请求累加（串行请求，和 ≈ 采集墙钟）
        for task in tasks:
            for g in range(args.group):
                req_t = time.time()
                try:
                    out = strata_client.chat(
                        args.engine,
                        [{"role": "user", "content": task["prompt"]}],
                        model=engine_model,
                        max_tokens=args.max_new,
                        temperature=args.temperature,
                        enable_thinking=(True if args.think else False),
                    )
                except Exception as exc:  # 模板不支持 enable_thinking 时退回默认
                    print(f"  (chat_template_kwargs 被拒: {type(exc).__name__}), retry without", flush=True)
                    out = strata_client.chat(
                        args.engine,
                        [{"role": "user", "content": task["prompt"]}],
                        model=engine_model,
                        max_tokens=args.max_new,
                        temperature=args.temperature,
                    )
                gen_seconds += time.time() - req_t
                gen_tokens += int((out.get("usage") or {}).get("completion_tokens") or len(out["token_ids"]))
                trajs.append(
                    {
                        "prompt": task["prompt"],
                        "target": task.get("target", ""),
                        "kind": task.get("kind", "exact"),
                        "content": out["content"],
                        "token_ids": out["token_ids"],
                        "token_logprobs": out["token_logprobs"],
                        "prompt_ids": out["prompt_ids"],
                        "tokens_exact": out["tokens_exact"],
                    }
                )
            print(f"  rollout {task['prompt'][:40]!r} x{args.group}  ({time.time() - t0:.0f}s)", flush=True)
        tok_s = gen_tokens / gen_seconds if gen_seconds > 0 else 0.0
        engine_meta = {"engine": args.engine, "model": engine_model, "requests": len(trajs),
                       "generated_tokens": gen_tokens, "engine_seconds": round(gen_seconds, 3),
                       "engine_avg_tok_s": round(tok_s, 2)}
        if args.save_rollouts:
            with open(args.save_rollouts, "w", encoding="utf-8") as f:
                for t in trajs:
                    f.write(json.dumps(t, ensure_ascii=False) + "\n")
            with open(args.save_rollouts + ".meta.json", "w", encoding="utf-8") as f:
                json.dump(engine_meta, f, ensure_ascii=False)
            print(f"saved {len(trajs)} trajectories to {args.save_rollouts}", flush=True)

    for t in trajs:
        t["reward"] = reward_of(t)

    # 长度门（提前版）：此刻还没加载 21.8 GiB 的学生，用引擎回显的 prompt_ids / token_ids 判定，
    # 一条都训不了就立刻退出。缺字段时判不了 —— run_grpo 在加载 tokenizer 后会再判一次（精确）。
    check_trainable_lengths(trajs, args.max_seq, args.max_skip_frac, tok=None)

    rew = torch.tensor([t["reward"] for t in trajs]) if trajs else torch.zeros(0)
    if trajs:
        print(f"rewards: mean {float(rew.mean()):.3f}  std {float(rew.std()):.4f}  "
              f"exact_ids {sum(1 for t in trajs if t.get('tokens_exact'))}/{len(trajs)}", flush=True)
    if args.no_student:
        print("--no-student: rollouts collected, the student was not loaded", flush=True)
        return 0

    # ---------------- 学生 ----------------
    model, tok = load_student(args.model, args.four_bit_experts,
                              gguf=args.gguf, pack=args.pack, tokenizer=args.tokenizer)
    freeze_all(model)
    if args.lora_method == "unilora":
        import unilora

        n = unilora.inject_unilora(model, targets=DEFAULT_TARGETS, r=args.lora_r)
        s = unilora.trainable_summary(model)
        print(f"{args.lora_method}: {n} modules, trainable {s['trainable']:,} ({s['pct']:.6f}%)", flush=True)
    else:
        n = inject_lora(model, targets=DEFAULT_TARGETS, r=args.lora_r, alpha=2 * args.lora_r)
        s = trainable_summary(model)
        print(f"lora {n} modules, trainable {s['trainable']:,} ({s['pct']:.3f}%)", flush=True)
    if args.init_adapter:
        print(f"init-adapter: 载入 {load_lora_state(model, args.init_adapter)} 个 LoRA 张量", flush=True)
    head = model.lm_head
    body = _text_model(model)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)

    # ---------------- 训练 ----------------
    cfg = objectives.GrpoCfg(
        clip_low=args.clip if args.clip_low is None else args.clip_low,
        clip_high=args.clip if args.clip_high is None else args.clip_high,
        tis_cap=args.tis_cap, norm=args.norm,
        scale_rewards=args.scale_rewards == "on",
    )
    eos_id = tok.eos_token_id
    print(f"objective: clip−{cfg.clip_low} / clip+{cfg.clip_high}  norm={cfg.norm}  "
          f"tis_cap={cfg.tis_cap}(old={args.old_logprobs})  scale_rewards={cfg.scale_rewards}  "
          f"mask_truncated={args.mask_truncated}", flush=True)

    last = run_grpo(
        model, tok, trajs, cfg, group=args.group, batch=args.batch,
        max_seq=args.max_seq, max_new=args.max_new, old_logprobs=args.old_logprobs,
        mask_truncated=args.mask_truncated, epochs=args.epochs, steps=args.steps,
        opt=opt, params=params, objective=args.objective,
        max_skip_frac=args.max_skip_frac)
    if not last:
        print("整个训练没有产生任何梯度：每一批都被跳过或组内 std=0", flush=True)

    if args.lora_method == "unilora":
        import unilora

        sd = unilora.materialize_ab(model)   # 键名兼容 lora_to_gguf；导出时 --alpha <rank>
        # 下面三个值必须与 inject_unilora 的默认一致（grpo 没单独暴露这些开关）
        torch.save(unilora.unilora_ckpt(model, proj_seed=42, theta_d_length=256, init_bound=0.02),
                   args.out + ".unilora.pt")
    else:
        sd = lora_state_dict(model)
    torch.save(sd, args.out)
    line = f"saved {len(sd)} LoRA tensors to {args.out}"
    if engine_meta:
        # 引擎采轨迹阶段的端到端均值（含 HTTP/队列/prefill），不是稳态 benchmark
        line += (f"  |  engine rollouts: {engine_meta['generated_tokens']} tokens in "
                 f"{engine_meta['engine_seconds']}s -> {engine_meta['engine_avg_tok_s']} tok/s avg")
    print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
