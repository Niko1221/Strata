"""训练运行的超参落盘：`<--out>.meta.json`。

为什么单独一个模块（而不是塞在 `sft.py` 里）：

1. 审核 §5.4.1 的「−31 项」判不了，根因之一是那次训练只留下 adapter、没留下
   `lr` / `steps` / `rank`：事后无法知道它是哪个工作点训出来的，也就无法复现、
   无法把「学习率」和「实现」分开归因。这个缺口必须能独立验证，不能靠跑一次 125B
   模型来间接证明，所以它不能依赖 torch / transformers / GPU。
2. `sft.py` 模块级会拉起整条 HF 加载栈，CI 的 CPU 档不一定装得齐。落盘是纯文件
   逻辑，放在这里才能进默认测试套件。

契约：
    * 加载模型之前写一次 `phase="start"`（OOM / 挂死 / 被 kill 也留得下参数）；
    * 训练结束再写一次 `phase="done"`，合并运行结果，不另开文件；
    * 写的是 `vars(args)` **全量**而非手列字段 —— 以后新增参数自动进 meta，不会再漏；
    * 写失败只警告，绝不把训练搞挂。
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
import time

SCHEMA = 1


def _jsonable(v):
    """argparse 的值本来都是标量，但新增参数可能带 Path / 自定义对象；宁可降级成字符串，
    也不能让「记一份 meta」把训练搞挂。"""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return str(v)


def _git_head() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"],
                           cwd=str(pathlib.Path(__file__).resolve().parent),
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def _versions() -> dict:
    """torch 可有可无：零阶那条树没有它，这个模块照样能 import、能测。"""
    out = {"python": sys.version.split()[0], "torch": "", "cuda": ""}
    try:
        import torch
        out["torch"] = torch.__version__
        out["cuda"] = torch.version.cuda or ""
    except Exception:
        pass
    return out


def _data_digest(spec: str) -> dict:
    """训练数据的指纹。`--data` 允许 `jsonl:<path>` 前缀，去掉前缀才是真实文件。"""
    if not spec:
        return {}
    path = spec.split(":", 1)[1] if spec.startswith("jsonl:") else spec
    f = pathlib.Path(path)
    if not f.is_file():
        return {"data_path": path, "data_present": False}
    h = hashlib.sha256(f.read_bytes()).hexdigest()
    n = sum(1 for line in f.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip())
    return {"data_path": str(f.resolve()), "data_present": True,
            "data_sha256": h, "data_rows": n}


def run_meta(args, **extra) -> dict:
    """完整可复现记录：参数 + 数据指纹 + 版本指纹 + 运行结果。纯函数，便于单测。"""
    rec = {str(k): _jsonable(v) for k, v in vars(args).items()}
    rec.update({"schema": SCHEMA,
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    rec.update(_versions())
    rec["git_head"] = _git_head()
    rec.update(_data_digest(getattr(args, "data", "") or ""))
    if getattr(args, "dpo_data", ""):
        rec.update({"dpo_" + k: v for k, v in _data_digest(args.dpo_data).items()})
    rec.update({k: _jsonable(v) for k, v in extra.items()})
    return rec


def write_run_meta(out_path: str, args, **extra) -> str:
    """合并写 `<--out>.meta.json`。start 与 done 两次调用合并成一条记录，不互相覆盖。

    参数名刻意叫 `out_path` 而不是 `out`：调用方常常直接 `**` 展开一个包含 `out`
    的 dict（例如把 `vars(args)` 派生出来的结果字典传进来），若这里也叫 `out`，
    Python 在**函数调用时**就抛 `got multiple values for argument 'out'`，函数体
    根本没机会做防御。2026-11-07 真机重训在最后一步踩到过，别改回去。
    """
    path = pathlib.Path(str(out_path) + ".meta.json")
    prev = {}
    if path.is_file():
        try:
            prev = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            prev = {}
    merged = {**(prev if isinstance(prev, dict) else {}), **run_meta(args, **extra)}
    try:
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(merged, indent=2, ensure_ascii=False, sort_keys=True),
                       encoding="utf-8")
        tmp.replace(path)
    except OSError as e:
        print(f"WARN: 超参 meta 写入失败（不影响训练）：{e}", flush=True)
        return ""
    return str(path)
