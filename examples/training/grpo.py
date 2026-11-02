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

import argparse
import json
import math
import sys
import time

import torch

from ce import token_logprobs
from lora import DEFAULT_TARGETS, freeze_all, inject_lora, lora_state_dict, trainable_summary


# ---------------------------------------------------------------- 学生模型

def _text_model(model):
    """拿到输出 hidden_states 的子模块：35B 是 model.model.language_model，
    普通文本模型是 model.model。"""
    inner = getattr(model, "model", None)
    if inner is None:
        return model
    return getattr(inner, "language_model", inner)


def load_student(path: str, four_bit_experts: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    """kind=exact（默认）严格相等；kind=words:N 要求词数恰好为 N；kind=wordsclose:N 按词数差给分。

    wordsclose:N = max(0, 1 - |n-N|/N)，只要长度飘就有梯度；words:N 常常组内四条全对或全错，
    std=0 ⇒ 优势全 0 ⇒ grad_norm=0 —— "跑完了"并不等于"训练生效了"。
    """
    text = (traj.get("content") or "").strip()
    kind = traj.get("kind") or "exact"
    if kind.startswith("wordsclose:"):
        want = int(kind.split(":", 1)[1])
        return max(0.0, 1.0 - abs(len(text.split()) - want) / max(1, want))
    if kind.startswith("words:"):
        want = int(kind.split(":", 1)[1])
        return 1.0 if len(text.split()) == want else 0.0
    return reward_answer(text, traj.get("target") or prompt_target(traj.get("prompt", "")))


def prompt_target(prompt: str) -> str:
    caps = [w.strip(".,:;!?\"'") for w in prompt.split()]
    caps = [w for w in caps if w.isalpha() and w.isupper()]
    return caps[-1].lower() if caps else ""


# ---------------------------------------------------------------- 主循环

def main() -> int:
    import platform_check

    platform_check.require()          # 平台不支持就停在这，别等读完 76 GB pack
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
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
    ap.add_argument("--max-seq", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-method", choices=("lora", "unilora"), default="lora",
                    help="unilora=Uni-LoRA（全模型只训一个共享向量，见 unilora.py）")
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--old-logprobs", choices=("local", "engine"), default="local")
    ap.add_argument("--epochs", type=int, default=2, help="每个 batch 内层过几遍；>1 才能看到 ratio 偏离 1 与裁剪生效")
    ap.add_argument("--think", action="store_true", help="默认关思考：max_tokens 覆盖思考+回答，"
                                                        "开着时短预算会把 content 挤成空串")
    ap.add_argument("--out", default="lora_out.pt")
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
    rew = torch.tensor([t["reward"] for t in trajs]) if trajs else torch.zeros(0)
    if trajs:
        print(f"rewards: mean {float(rew.mean()):.3f}  std {float(rew.std()):.4f}  "
              f"exact_ids {sum(1 for t in trajs if t.get('tokens_exact'))}/{len(trajs)}", flush=True)
    if args.no_student:
        print("--no-student: rollouts collected, the student was not loaded", flush=True)
        return 0

    # ---------------- 学生 ----------------
    model, tok = load_student(args.model, args.four_bit_experts)
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
    head = model.lm_head
    body = _text_model(model)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)

    # ---------------- 训练 ----------------
    for step in range(1, args.steps + 1):
        batch = trajs[(step - 1) * args.batch * args.group : step * args.batch * args.group]
        if not batch:
            batch = trajs[: args.batch * args.group]
        rewards = torch.tensor([t["reward"] for t in batch])
        # GRPO 的基线是**同一个 prompt 的那几条**，不是整批：跨 prompt 归一化会把"这道题全错"
        # 和"那道题全对"混成一个优势（数学上不是 GRPO）。组内 std=0 时这一组的优势就该是 0，
        # 所以这里把有效的组数也打出来 —— 它是"这一步到底有没有梯度"的唯一判据。
        adv_all = torch.zeros(len(batch))
        groups_used = 0
        for i0 in range(0, len(batch), args.group):
            grp = rewards[i0 : i0 + args.group]
            if len(grp) > 1 and float(grp.std()) > 0.0:
                adv_all[i0 : i0 + len(grp)] = (grp - grp.mean()) / (grp.std() + 1e-4)
                groups_used += 1
        zero_std = groups_used == 0

        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_pg = 0.0
        used = 0
        ratios = []

        # 先把这一批的序列准备好，old logprob 只算一次（这才是"旧策略"）
        prepared = []
        for t in batch:
            # 引擎回显的 prompt_ids 就是它实际读的那一串；本地模板只在老引擎（不产 LP、也就不回显）时兜底
            prompt_ids = t.get("prompt_ids") or tok.apply_chat_template(
                [{"role": "user", "content": t["prompt"]}], add_generation_prompt=True, return_dict=False
            )
            comp_ids = t["token_ids"] or tok(t["content"], add_special_tokens=False).input_ids
            ids = list(prompt_ids) + list(comp_ids)
            if len(ids) > args.max_seq or not comp_ids:
                continue
            with torch.no_grad():
                ids_t = torch.tensor([ids], device="cuda")
                n_prompt = len(prompt_ids)
                labels = ids_t[0, n_prompt:]
                hidden = body(input_ids=ids_t, use_cache=False).last_hidden_state[0, n_prompt - 1 : -1]
                old = token_logprobs(hidden, head, labels, chunk=128).clone()
            if args.old_logprobs == "engine" and t.get("token_logprobs"):
                cand = torch.tensor(t["token_logprobs"], device="cuda", dtype=torch.float32)
                if cand.numel() == old.numel():
                    old = cand
            prepared.append((ids, n_prompt, labels, old))

        # GRPO 原样：old 取自本批次开始前的策略，之后每个 inner epoch 都真正更新权重，
        # 于是 epoch>=2 的 ratio 必然偏离 1，裁剪项才开始起作用。
        for epoch in range(1, args.epochs + 1):
            opt.zero_grad(set_to_none=True)
            e_loss = e_pg = 0.0
            e_used = 0
            e_ratios = []
            e_clipped = 0.0
            for idx, (ids, n_prompt, labels, old) in enumerate(prepared):
                adv = adv_all[idx].item()
                ids_t = torch.tensor([ids], device="cuda")
                hidden = body(input_ids=ids_t, use_cache=False).last_hidden_state[0, n_prompt - 1 : -1]
                new_lp = token_logprobs(hidden, head, labels, chunk=128)
                ratio = torch.exp(new_lp - old)
                surrogate = ratio * adv
                clipped = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv
                obj = torch.minimum(surrogate, clipped)
                loss = -obj.mean()
                pg = -(adv * new_lp).mean()
                (loss / max(1, len(prepared))).backward()
                e_loss += float(loss)
                e_pg += float(pg)
                e_ratios.append(float(ratio.mean()))
                e_clipped += float((surrogate != obj).float().mean())
                e_used += 1
            if e_used == 0:
                break
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            print(
                f"step {step}.{epoch}: pg_obj {e_pg / e_used:.4f}  clipped_obj {e_loss / e_used:.4f}  "
                f"grad_norm {float(gn):.4f}  ratio {sum(e_ratios) / len(e_ratios):.4f}  "
                f"clip_frac {e_clipped / e_used:.3f}  reward {float(rewards.mean()):.3f}  "
                f"reward_std {float(rewards.std()):.4f}  zero_std {zero_std}  groups {groups_used}  used {e_used}  "
                f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
                flush=True,
            )

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
