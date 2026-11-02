"""按 plan.json 顺序跑评测步骤，把每个步骤的出口码 / 用时 / 日志落到同一份报告里。

为什么不是 shell 脚本
--------------------
两件 shell 很难干净做到的事，这里必须有：
  ① **挂死续跑**：`bench_fullctx.py` 超时会带退出码 **3** 退出（kernel 挂住杀不掉，只能落盘+自杀）。
     于是 plan 里写 `"retry_on": {"3": {"times": 5, "append": ["--resume"]}}`，它就会自己接上，
     而不会因为一格挂死把整轮 B 阶段废掉。
  ② **一格失败不等于整轮失败**：`"on_fail": "continue"` 记一笔继续；`"stop"` 才是终止。
     报告里 `failed: [...]` 一眼看出哪些格子没跑到。

plan.json 形状
--------------
    {"steps": [
      {"name": "b2_budget", "cmd": ["python", "budget.py", "--json", "bench_budget.json"]},
      {"name": "b2_bench",  "cmd": ["python", "bench_fullctx.py", "..."],
       "timeout": 7200, "retry_on": {"3": {"times": 5, "append": ["--resume"]}}},
      {"name": "b2_judge",  "cmd": ["python", "judge_limits.py", "bench_limits.json"],
       "on_fail": "continue"}
    ]}

用法
----
    python run_plan.py plan.json --report report.json --logdir logs
    python run_plan.py plan.json --only b2_bench --dry-run
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
import os
import subprocess
import time


def load_plan(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    steps = d.get("steps") if isinstance(d, dict) else d
    if not isinstance(steps, list) or not steps:
        raise SystemExit(f"{path}: 需要 {{\"steps\": [...]}}，且有至少一步")
    for i, s in enumerate(steps):
        if "name" not in s or "cmd" not in s:
            raise SystemExit(f"{path}: 第 {i + 1} 步缺 name 或 cmd")
        s.setdefault("cwd", os.getcwd())
        s.setdefault("on_fail", "stop")
        if not isinstance(s["cmd"], (list, str)):
            raise SystemExit(f"{path}: {s['name']} 的 cmd 要是 list 或 str")
    return steps


def expand(value, env: dict):
    """把 `$VAR` / `${VAR}` 换成 --set 给的（以及环境里已有的）值。"""
    if isinstance(value, list):
        return [os.path.expandvars(str(v)) for v in value]
    return os.path.expandvars(str(value))


def run_one(step: dict, logdir: str) -> dict:
    """跑一步（含 retry_on）。返回报告条目，不改全局状态。"""
    os.makedirs(logdir, exist_ok=True)
    log = os.path.join(logdir, step["name"] + ".log")
    cmd = list(step["cmd"]) if isinstance(step["cmd"], (list, tuple)) else step["cmd"]
    retry = {int(k): v for k, v in (step.get("retry_on") or {}).items()}
    max_try = 1 + sum(int(v.get("times", 0)) for v in retry.values())

    attempts = []
    t0 = time.time()
    for attempt in range(max_try):
        # 重试时才追加参数（典型是 --resume）：第一次跑不该带它
        run_cmd = cmd
        if attempt and isinstance(cmd, list) and step.get("append"):
            run_cmd = cmd + list(step["append"])
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"\n=== {step['name']} attempt {attempt + 1}/{max_try} "
                    f"{time.strftime('%H:%M:%S')} ===\n")
            f.flush()
            try:
                p = subprocess.run(run_cmd, cwd=step["cwd"], shell=isinstance(cmd, str),
                                   stdout=f, stderr=subprocess.STDOUT,
                                   timeout=step.get("timeout") or None)
                code = p.returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
        attempts.append(code)
        if code == 0:
            break
        rule = retry.get(code) if isinstance(code, int) else None
        if not rule or attempt + 1 >= max_try:
            break
    return {"name": step["name"], "exit": attempts[-1], "attempts": attempts,
            "wall_s": round(time.time() - t0, 1), "log": log,
            "on_fail": step.get("on_fail", "stop")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("plan")
    ap.add_argument("--report", default="report.json")
    ap.add_argument("--logdir", default="logs")
    ap.add_argument("--only", default="", help="逗号分隔的步骤名，只跑这些")
    ap.add_argument("--from", dest="start_at", default="", help="从这个步骤名开始")
    ap.add_argument("--set", action="append", default=[], metavar="K=V",
                    help="定义 plan 里 $K 用的变量（也传给子进程）；可重复")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for kv in args.set:
        k, sep, v = kv.partition("=")
        if not sep:
            raise SystemExit(f"--set 要 K=V，收到 {kv!r}")
        os.environ[k] = v

    steps = load_plan(args.plan)
    only = {s for s in args.only.split(",") if s}
    if only:
        steps = [s for s in steps if s["name"] in only]
        missing = only - {s["name"] for s in steps}
        if missing:
            raise SystemExit(f"--only 里有 plan 中不存在的步骤：{sorted(missing)}")
    if args.start_at:
        names = [s["name"] for s in steps]
        if args.start_at not in names:
            raise SystemExit(f"--from {args.start_at} 不在 plan 里")
        steps = steps[names.index(args.start_at):]

    for s in steps:                            # 变量只在这里展开，run_one 拿到的就是最终命令
        s["cmd"] = expand(s["cmd"], os.environ)
        s["cwd"] = expand(s["cwd"], os.environ)
        if s.get("append"):
            s["append"] = expand(s["append"], os.environ)

    print(f"{len(steps)} 步：{', '.join(s['name'] for s in steps)}", flush=True)
    if args.dry_run:
        for s in steps:
            print(f"  [{s['name']}] cwd={s['cwd']} cmd={s['cmd']}")
        return 0

    done, failed = [], []
    t_all = time.time()
    for s in steps:
        print(f"\n--- {s['name']} ---", flush=True)
        r = run_one(s, args.logdir)
        done.append(r)
        ok = r["exit"] == 0
        print(f"  exit={r['exit']} attempts={r['attempts']} {r['wall_s']:.0f}s log={r['log']}",
              flush=True)
        if not ok:
            failed.append(r["name"])
            if r["on_fail"] == "stop":
                print(f"  on_fail=stop ⇒ 停下（后面 {len(steps) - len(done)} 步没跑）", flush=True)
                break

    report = {"plan": os.path.basename(args.plan), "wall_s": round(time.time() - t_all, 1),
              "steps": done, "failed": failed,
              "skipped": [s["name"] for s in steps[len(done):]]}
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"\n{len(done)} 步完成，失败 {failed or '无'}，用时 {report['wall_s']:.0f}s")
    print(f"report → {args.report}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
