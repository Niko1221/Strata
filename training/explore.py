"""探索式训练的一轮闭环：采样 → 判分 → 落池 → 分档 → 零阶一步。

与 `train.py --algo auto` 的分工
-------------------------------
`train.py --algo auto` 是**离线**的：它假定轨迹已经采完、躺在一个文件里，然后写一份
计划交给 `run_plan.py` 串行执行。那是「先采后训」。

本模块是**在线**的：模型一边跑，样本一边进池，池子一边重算分档，需要时就在**同一个
驻留引擎**上走一步零阶更新 —— 不卸载、不重启，推理不中断。差别不在算法，在数据
怎么流：批次 → 流。

## 与「纯推理」的关系（这是设计约束，不是优化）

训练走 `mezo.py --serve-url <base>` → `EngineHttp`，它驱动**已经在跑**的推理服务，
`close()` 是空操作。所以：

  * 不训练时，服务就是一个普通的推理服务：没有额外进程、线程、分配；
  * 训练随时开始、随时停止，不用重启服务；
  * 采样走 `/v1/chat/completions`（普通 OpenAI 端点），更新走 `/v1/engine/lora`。

但有两件**在服务器启动时就定死**的事，本模块补不了，只能提前检查并说清：

  1. 服务必须带 `--lora <初始 adapter>` 启动。`/v1/engine/lora` 换的是「启动时那套表」
     的**内容**，形状不能变 ⇒ rank 全程固定。没有它，换表端点会 400。手上没有 adapter
     时用 `mezo/zero_adapter.py` 造一个（A = B = 0：服务出去的数值就是基座的，rank 取
     `--rank`，默认 8），这一项因此不再需要人手准备文件。
  2. 服务必须带 `--no-prefill-borrow`。否则 prompt 缓冲借的是
     专家缓存槽，**SCORE 的数值会错**（实测 8-token 序列在 2485 槽的借用上求和 -79.39，
     自有时 -56.11，HF 参考 -56.64）—— 零阶的差分信号只有 ~0.55 nats，这种量级的误差
     会把它完全淹没。这条在 HTTP 层探测不到，只能由启动命令保证。
     （`--prefill` 是 prompt 的**分块大小**、不是长度上限，自起引擎时按数据里最长样本取：
     见 `mezo.resolve_prefill_chunk()`。驻留服务的 chunk 在服务端启动时就定了。）

这两条意味着：**「只要能推理就能训练」目前还需要这两项启动配置**，不是零成本。
把它们藏起来比说出来更糟 —— 训练会静默走偏，日志上只看到 loss 不降。

## 一次更新会打掉在跑的对话

`/v1/engine/lora` 换表时**清空所有 K/V**（否则分数与换表前不可比）。所以更新粒度是
**一轮**，不是「每生成一条就更新一次」：一轮里把采样做完，再更新，再采下一轮。多步
任务正在跑的时候不要更新 —— 它下一个请求会从 token 0 重读 prompt。同理，服务以
`--parallel` 跑并发请求时，换表是拒绝的（那些 slot 正在用要被替换的表）。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request

import algos
import coldstart
import pool as P


def _post(url: str, body: dict, timeout: float = 900.0) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:400]
        raise SystemExit(f"{url}: HTTP {e.code} {detail}") from None
    except urllib.error.URLError as e:
        raise SystemExit(f"{url}: 连不上（{e.reason}）—— 服务起了吗？") from None


def _load_judge(spec: str):
    """`path/to/judge.py:func` → `func(prompt, completion) -> float | str | None`。

    返回值三种都接受：
      * `float` —— reward，`>= 0.5` 记 pos（与 `coldstart.PASS` 同一口径），source=scorer
      * `"pos" / "neg" / "unknown"` —— 直接用作 verdict，source=scorer
      * `None` —— 这条判不了，记 unknown（**不是记成答错**）

    不给 `--judge` 时全部记 unknown：探索式训练里「没答案」是常态，等人工标注比瞎判
    安全 —— 把未知记成负例，DPO 会去惩罚一个可能正确的行为。
    """
    if not spec:
        return None
    path, _, fn = spec.partition(":")
    if not (path and fn):
        raise SystemExit(f"--judge 要写成 'path/to/judge.py:func'，收到 {spec!r}")
    s = importlib.util.spec_from_file_location("_explore_judge", path)
    if s is None or s.loader is None:
        raise SystemExit(f"--judge 加载不了 {path}")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    f = getattr(m, fn, None)
    if not callable(f):
        raise SystemExit(f"{path} 里没有可调用的 {fn}")
    return f


def judge_sample(fn, prompt: str, completion: str, gold: str,
                 truncated: bool) -> dict:
    """判一条，返回 `{verdict?, reward?, source}`。截断优先记 unknown（reward 未知）。"""
    if truncated:
        # 没写完 ⇒ 通过率不可信。这条是「未知」，不是「答错」（见 coldstart.classify）。
        return {"verdict": P.UNKNOWN, "source": "auto", "truncated": True}
    if fn is None:
        return {"verdict": P.UNKNOWN, "source": "human"}
    out = fn(prompt, completion)
    if out is None:
        return {"verdict": P.UNKNOWN, "source": "scorer"}
    if isinstance(out, str):
        if out not in P.VERDICTS:
            raise SystemExit(f"judge 返回了 {out!r}，只接受 float / "
                             f"{'/'.join(P.VERDICTS)} / None")
        return {"verdict": out, "source": "scorer"}
    r = float(out)
    return {"reward": r, "source": "scorer"}


def sample_once(base: str, prompt: str, *, max_new: int, temperature: float,
                system: str = "") -> tuple[str, bool]:
    """一次生成。返回 (completion, 是否被 max_new 截断)。"""
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt}]
    body = {"model": "strata", "messages": msgs, "max_tokens": max_new,
            "temperature": temperature, "stream": False}
    out = _post(base.rstrip("/") + "/v1/chat/completions", body)
    try:
        ch = out["choices"][0]
        text = ch["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        raise SystemExit(f"生成返回的结构不认识: {str(out)[:300]}") from None
    return text, ch.get("finish_reason") == "length"


def preflight(base: str) -> None:
    """训练前的最低检查：服务可达、换表端点存在。

    只能查到这儿 —— `--lora` 装没装要看换表是否 400，而 `--no-prefill-borrow` 在
    HTTP 层完全不可见。所以这两条写进模块头，让启动命令负责，这里不假装验证过。
    """
    out = _post(base.rstrip("/") + "/v1/engine/score",
                {"prompt": "ok", "lora": 0}, timeout=120.0)
    if "logprobs" not in out and "tokens" not in out:
        print(f"[warn] /v1/engine/score 返回的结构异常: {str(out)[:200]}", flush=True)


def run_round(*, base: str, pool_path: str, tasks: list[dict], group: int,
              max_new: int, temperature: float, judge_fn, adapter_tag: str,
              system: str) -> dict:
    """一轮：对每个 prompt 采 K 条，判分，落池。返回本轮写入条数。"""
    pool = P.Pool(pool_path)
    n = 0
    for ti, t in enumerate(tasks, 1):
        prompt, gold = t["prompt"], t.get("gold", "")
        for k in range(group):
            text, trunc = sample_once(base, prompt, max_new=max_new,
                                      temperature=temperature, system=system)
            j = judge_sample(judge_fn, prompt, text, gold, trunc)
            pool.append(P.record(prompt, text, adapter=adapter_tag, gold=gold,
                                 step=int(time.time()), **j))
            n += 1
        vs = [P.verdict_of(s) for s in pool.groups(adapter_tag).get(prompt, [])]
        print(f"  [{ti}/{len(tasks)}] {prompt[:48]!r} "
              f"+{group} → {pool.stats()['samples']} 条累计", flush=True)
    return {"written": n}


def train_step(*, base: str, pool_path: str, group: int, adapter_tag: str,
               pack: str, gguf: str, epochs: float, out_dir: str) -> int:
    """在**驻留引擎**上走一轮零阶更新（不卸载）。返回子进程退出码。"""
    p, paths, steps = coldstart.plan_pool_file(
        pool_path, group, adapter="latest", out_dir=out_dir)
    print(coldstart.summarize(p), flush=True)
    if not p["rl"]:
        print("  本轮没有 RL 组（全对毕业 / 全错待冷启 / 有待补标）—— 跳过训练")
        return 0
    if p["cold"]:
        cold = [c for c in p["cold"] if c.get("cold_action") != "none"]
        if cold:
            print(f"  [note] {len(cold)} 个全错组需要冷启（反传），本回合不跑："
                  "反传要卸载推理引擎（48 GB 放不下双驻留），属于离线阶段。"
                  f" 见 {paths['dpo']}", flush=True)
    mezo = pathlib.Path(__file__).resolve().parent / "mezo" / "mezo.py"
    cmd = [sys.executable, str(mezo),
           "--pack", pack, "--gguf", gguf,
           "--serve-url", base.rstrip("/"),
           "--objective", "luspo",
           "--rollouts", paths["rl"],
           "--group-size", str(group),
           "--lr", str(algos.default_lr("luspo", "mezo")),
           "--epochs", str(epochs)]
    print("  训练：" + " ".join(cmd[:6]) + f" ... --epochs {epochs}", flush=True)
    return subprocess.run(cmd, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="探索式训练闭环（采样 → 判分 → 落池 → 零阶一步）")
    ap.add_argument("--base", default="http://127.0.0.1:8080",
                    help="驻留推理服务的地址（必须带 --lora 与 --no-prefill-borrow 启动）")
    ap.add_argument("--pool", required=True, help="样本池 jsonl（append-only）")
    ap.add_argument("--tasks", required=True, help="JSONL {prompt, gold?}")
    ap.add_argument("--group", type=int, default=4, help="每题每轮采几条")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--judge", default="",
                    help="'judge.py:func' → func(prompt, completion) 返回 float / "
                         "pos|neg|unknown / None。不给则全部记 unknown 等人工标注")
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--system", default="")
    ap.add_argument("--pack", default="", help="训练时必须给：tokenizer / 模板所在 pack")
    ap.add_argument("--gguf", default="", help="训练时必须给：基座第一个分片")
    ap.add_argument("--rl-epochs", type=float, default=1.0,
                    help="每轮 RL 的轮次（步数 = 轮次 × 组数，跨数据量可比）")
    ap.add_argument("--out-dir", default="", help="计划与 rl.jsonl 的落盘目录")
    ap.add_argument("--explore-only", action="store_true",
                    help="只采样与分档，不训练（先看清池子长什么样）")
    a = ap.parse_args(argv)

    tasks = coldstart.load_rows(a.tasks)
    if not tasks:
        raise SystemExit(f"{a.tasks}: 空的")
    judge_fn = _load_judge(a.judge)
    out_dir = a.out_dir or str(pathlib.Path(a.pool).with_suffix("")) + ".explore"

    print(f"base   {a.base}\npool   {a.pool}\ntasks  {len(tasks)}  × {a.group}"
          f"  × {a.rounds} 轮", flush=True)
    if judge_fn is None:
        print("judge  无 —— 新样本一律记 unknown，等你标注（pool.py label）后才会进训练",
              flush=True)
    if not a.explore_only:
        if not (a.pack and a.gguf):
            raise SystemExit("训练需要 --pack 与 --gguf（mezo 要用它们做模板与 tokenize）。"
                             "只想看分档就加 --explore-only")
        preflight(a.base)

    for rnd in range(1, a.rounds + 1):
        tag = f"r{rnd}"
        print(f"\n=== round {rnd}/{a.rounds}  adapter={tag} ===", flush=True)
        run_round(base=a.base, pool_path=a.pool, tasks=tasks, group=a.group,
                  max_new=a.max_new, temperature=a.temperature, judge_fn=judge_fn,
                  adapter_tag=tag, system=a.system)
        st = P.Pool(a.pool).stats()
        print(f"  pool: {st['samples']} 条 / {st['prompts']} 题  "
              f"pos={st['pos']} neg={st['neg']} unknown={st['unknown']}", flush=True)
        if a.explore_only:
            p = coldstart.plan_pool(P.Pool(a.pool).groups("latest"),
                                    rl_group_size=a.group)
            print(coldstart.summarize(p), flush=True)
            continue
        rc = train_step(base=a.base, pool_path=a.pool, group=a.group, adapter_tag=tag,
                        pack=a.pack, gguf=a.gguf, epochs=a.rl_epochs, out_dir=out_dir)
        if rc != 0:
            print(f"  [warn] 训练退出码 {rc}；池子不受影响，可修好后重跑", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
