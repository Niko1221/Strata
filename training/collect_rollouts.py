"""从 Strata 引擎采 GRPO 轨迹，写成训练侧能直接吃的 jsonl。

为什么单独一个脚本：48GB 卡放不下引擎（Q4_K_XL 占 28.6GB）和 4bit 学生（21.8GB），
所以采集与训练必须分两段，而采集这段只需要 HTTP，不需要 torch。
（`train_grpo_own.py --no-student` 做同一件事，但它 import torch，只能在 WSL 里跑。）

每条轨迹记的是引擎自己吐的序列：`token_ids` 与 `token_logprobs` 都来自 LP 行，
所以 `tokens_exact` 为真时可以当作严格 on-policy 的旧策略概率用；为假（老引擎不产 LP）
就只剩文本，训练端会退回到本地重编码 + 本地重算 old logprob。

用法：
  python collect_rollouts.py --engine http://127.0.0.1:8095 --prompts prompts_variance.jsonl \
      --group 4 --max-new 40 --temperature 1.2 --out rollouts35.jsonl

并行采样（引擎的批槽，`--parallel 0` 为默认）：
  1. 服务器那边要批槽：模型配置 `strata-<model>.json` 里加 `"parallel": N`
     （等价引擎参数 `--batch N` / `--slots N`），然后重启；`docs/BATCHING.md` 有细节。
  2. 这边什么都不用做：`--parallel 0` 是默认，启动时读 `/v1/status` 的 `concurrency`，
     引擎声明多少就用多少。别自己定 —— 并行度是引擎的批槽说了算的，自造一份只会
     在引擎换配置后悄悄失配。要压低/抬高用 `--parallel N`（N ≥ 1）。

  批槽对 GRPO 特别合适：一个组里 K 条轨迹共享 prompt 前缀、互不依赖，是天然的一批并发请求，
  而且采集这段不占训练显存（采样方是引擎）。当前引擎的批路径**没有草稿**（要 `--batch-mtp` 才
  有，且它是 opt-in、只有两级时才测过），所以聚合吞吐的增益有限（下面有实测），但延迟
  明显好：N 条轨迹不再串联等待。

  两个不能并行的事，写在这里免得后来人踩：
  - 批槽不等于训练并行：学生要 26-34 GiB，跟常驻的引擎放不进同一张卡，所以**采集和训练
    必须分两段**（`collect_rollouts.py` → 关引擎 → `--rollouts` 训），不能一边采一边训。
  - 想要"不碰引擎就能并行"，目前只有多副本（多个引擎进程，`serve/replicas.py`）；那不是
    批槽，是好几份常驻。
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent import futures

import strata_client


def http(url: str, timeout: float = 30.0) -> dict:
    """GET a JSON document.  Loopback must skip the proxy (WSL sets http_proxy to a
    filter that answers 500 for 127.0.0.1); see strata_client's docstring."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def prompt_target(prompt: str) -> str:
    caps = [w.strip(".,:;!?\"'") for w in prompt.split()]
    caps = [w for w in caps if w.isalpha() and w.isupper()]
    return caps[-1].lower() if caps else ""


def reward_of(traj: dict) -> float:
    """kind=exact（默认）严格相等；kind=words:N 要求词数恰好为 N；kind=wordsclose:N 按词数差给分。

    wordsclose:N = max(0, 1 - |n-N|/N)：只要长度飘就有梯度。words:N 的问题不是"太严"，
    而是**组内四条常常全对或全错**（std=0 ⇒ 优势全 0 ⇒ grad_norm=0），看起来训练跑完了，
    其实一个参数都没动。
    """
    text = (traj.get("content") or "").strip()
    kind = traj.get("kind") or "exact"
    if kind.startswith("wordsclose:"):
        want = int(kind.split(":", 1)[1])
        return max(0.0, 1.0 - abs(len(text.split()) - want) / max(1, want))
    if kind.startswith("words:"):
        return 1.0 if len(text.split()) == int(kind.split(":", 1)[1]) else 0.0
    want = traj.get("target") or prompt_target(traj.get("prompt", ""))
    flat = text.strip("。.!?！？\"'`*_ \n\t").lower()
    return 1.0 if want and flat == want else 0.0


def sample_one(task: dict, args: argparse.Namespace) -> dict:
    """一条轨迹。抛异常由调用方决定怎么处理（并发时逐条记，不整批失败）。"""
    out = strata_client.chat(
        args.engine, [{"role": "user", "content": task["prompt"]}], model=args.engine_model,
        max_tokens=args.max_new, temperature=args.temperature,
        enable_thinking=bool(args.think))
    traj = {"prompt": task["prompt"], "target": task.get("target", ""),
            "kind": task.get("kind", "exact"), **out}
    traj["reward"] = reward_of(traj)
    return traj


def engine_parallel(base_url: str) -> int:
    """问引擎它同时在跑几个请求（/v1/status 的 concurrency）。

    这就是「对接」：并行度属于引擎的批槽（`"parallel": N` / `--batch N`），采集这边照着它发，
    不自己定一个数。引擎没开批槽就返回 1 —— 那时串行发，也不会排队。
    """
    try:
        d = http(base_url.rstrip("/") + "/v1/status", timeout=30)
    except (urllib.error.URLError, OSError, ValueError):
        return 1
    c = d.get("concurrency") or {}
    return max(1, int(c.get("serving") or c.get("requested") or 1))


def _timed(task: dict, args: argparse.Namespace) -> tuple[dict, float]:
    """`sample_one` 加一段计时：并发时逐请求耗时之和是「没有并发会花多久」的估计。"""
    s = time.time()
    return sample_one(task, args), time.time() - s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="http://127.0.0.1:8095")
    ap.add_argument("--engine-model", default=None)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--think", action="store_true", help="默认关思考：max_tokens 同时覆盖思考与回答")
    ap.add_argument("--parallel", type=int, default=0,
                    help="同时在飞的请求数。0（默认）= 问引擎：读 /v1/status 的 "
                         'concurrency，引擎声明多少就用多少（即配置里 "parallel": N / '
                         "--batch N）。别自己定：并行度是引擎的批槽说了算的。")
    args = ap.parse_args()

    if args.parallel < 0:
        raise SystemExit("--parallel must be 0 (ask the engine) or more")

    model = args.engine_model
    if model is None:
        model = http(args.engine.rstrip("/") + "/v1/models")["data"][0]["id"]
    args.engine_model = model

    # 并行度不在这里定：它属于引擎（批槽）。问一次 /v1/status，引擎说几个就是几个。
    asked = args.parallel
    if asked == 0:
        asked = engine_parallel(args.engine)
        if asked <= 1:
            print("engine: no batch slots (start it with \"parallel\": N in the config); "
                  "sampling serially", flush=True)
    print(f"engine model: {model}   parallel {max(1, asked)}", flush=True)
    args.parallel = max(1, asked)

    tasks = [json.loads(line) for line in open(args.prompts, encoding="utf-8") if line.strip()]
    # 固定顺序：并发只改变完成的先后，不改变写出的顺序 —— 否则同配置两次跑出的 loss 不可比
    jobs = [(ti, ki) for ti in range(len(tasks)) for ki in range(args.group)]
    results: list[dict | None] = [None] * len(jobs)

    t0 = time.time()
    sum_req = 0.0                 # 逐请求耗时之和：并发时 ≫ 墙钟，用来看重叠了多少
    if args.parallel == 1:
        for idx, (ti, _) in enumerate(jobs):
            _s = time.time()
            results[idx] = sample_one(tasks[ti], args)
            sum_req += time.time() - _s
    else:
        with futures.ThreadPoolExecutor(max_workers=args.parallel) as ex:
            pending = {ex.submit(_timed, tasks[ti], args): idx for idx, (ti, _) in enumerate(jobs)}
            for f in futures.as_completed(pending):
                results[pending[f]], dt = f.result()
                sum_req += dt
    wall = time.time() - t0

    trajs = [r for r in results if r is not None]
    for task in tasks:
        done = [t for t in trajs if t["prompt"] == task["prompt"]]
        print(f"  {task['prompt'][:44]!r} x{len(done)}  reward "
              f"{sum(t['reward'] for t in done):.0f}/{len(done)}", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for t in trajs:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    n = len(trajs)
    exact = sum(1 for t in trajs if t.get("tokens_exact"))
    rewards = [t["reward"] for t in trajs]
    mean = sum(rewards) / n if n else 0.0
    std = (sum((r - mean) ** 2 for r in rewards) / n) ** 0.5 if n else 0.0
    gen_tokens = sum(int((t.get("usage") or {}).get("completion_tokens") or len(t["token_ids"]))
                     for t in trajs)
    # 两条口径：串行时二者相等；并发时 wall 是墙钟（要的），engine 是逐请求之和（看重叠）
    wall_tok_s = gen_tokens / wall if wall > 0 else 0.0
    tok_s = gen_tokens / sum_req if sum_req > 0 else 0.0
    meta = {"engine": args.engine, "model": model, "requests": n,
            "generated_tokens": gen_tokens, "engine_seconds": round(sum_req, 3),
            "engine_avg_tok_s": round(tok_s, 2),
            "parallel": args.parallel, "wall_seconds": round(wall, 3),
            "wall_avg_tok_s": round(wall_tok_s, 2)}
    with open(args.out + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    print(f"saved {n} trajectories to {args.out}\n"
          f"  tokens_exact {exact}/{n}   reward mean {mean:.3f} std {std:.4f}\n"
          f"  wall: {n} requests in {wall:.1f}s with parallel={args.parallel} "
          f"-> {wall_tok_s:.1f} tok/s aggregate "
          f"(end-to-end, meta in {args.out}.meta.json)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
