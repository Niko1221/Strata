"""_eval_hard.py —— 成功率评测：把一批可验证题喂给引擎 GEN，逐题自动判分。

为什么是内部模块（下划线前缀、不直接运行）：唯一调用方是 `mezo.py --success-data`，
契约由那两处调用点固定 ——

    rows = load_rows(path)                                  # jsonl，每行 {"prompt", "answer"}
    ok, total, detail = success_eval(eng, tok, render, rows, max_new)

住在 bench/ 而不是 mezo/ 的理由：它和 `math_probe.py` / `math_extract.py` 同属评测侧。
**不 import torch** —— 零阶线要能在没装 torch 的机器上把 train.py 跑完（见 algos.py 的
分支说明），这条约束同样适用于它的评测件。

判分口径（三档，`detail` 里记下命中哪一档，便于事后审计）
    num     两侧都能 extract_num 抽出整数且相等 —— 数学题的标准口径，与 math_probe.py 同一条规则
    exact   去空白、转小写后全文相等 —— 非数字答案（"Paris" / "true"）
    substr  标准答案作为子串出现在回答里 —— 模型多说一句前缀的常见情形
三档都不中判错。判据**有意偏严**：宁可把「其实答对了但格式跑了」判错，也不要让
`--success-data` 那条「到 100% 要多久」的曲线因为宽松匹配而虚高。
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
import time

from math_extract import extract_num  # noqa: E402  与 math_probe.py 共用同一条取数规则


def _norm(text: str) -> str:
    """比较用的归一化：折叠所有空白、统一小写。不做标点剥离（会把 "3.5" 弄成 "35"）。"""
    return " ".join((text or "").split()).lower()


def judge(text: str, want: str) -> tuple[bool, str]:
    """单条判分 → (对错, 命中的档位)。档位为空串表示判错。"""
    got_num, want_num = extract_num(text), extract_num(want)
    if want_num is not None and got_num is not None and got_num == want_num:
        return True, "num"
    if _norm(text) == _norm(want):
        return True, "exact"
    w = _norm(want)
    if w and w in _norm(text):
        return True, "substr"
    return False, ""


def load_rows(path: str) -> list[dict]:
    """读题目 jsonl。行级报错带行号 —— 静默跳过坏行等于悄悄缩小评测集。"""
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                r = json.loads(line)
            except ValueError as e:
                raise SystemExit(f"{path}:{ln}: not json ({e})")
            miss = [k for k in ("prompt", "answer") if k not in r]
            if miss:
                raise SystemExit(f"{path}:{ln}: missing {', '.join(miss)} "
                                 "(每行是 {\"prompt\": <题面>, \"answer\": <标准答案>})")
            rows.append({"prompt": str(r["prompt"]), "answer": str(r["answer"])})
    if not rows:
        raise SystemExit(f"{path}: no rows")
    return rows


def success_eval(eng, tok, render, rows: list[dict], max_new: int = 512):
    """逐题 GEN 后判分 → (ok, total, detail)。

    生成预算 `max_new` 是协议口径的一部分：与训练后测量必须同值，否则「到 100% 的耗时」
    比较的是两个不同任务（长度截断会改变正确率）。调用方传 `--success-max-new`。

    引擎异常（GEN 失败、进程死掉）直接向上抛 —— 让调用方看到，不在这里吞成 0 分；
    一次静默的 0% 会被误读成「训练把模型弄坏了」。
    """
    if not hasattr(tok, "decode"):
        print("  success: the tokenizer has no decode(); every row will score 0", flush=True)
    t0 = time.time()
    detail: list[dict] = []
    ok = 0
    for r in rows:
        ids = tok.encode(render(r["prompt"]), parse_special=True)
        out_ids, _ms = eng.gen(ids, max_new)
        text = tok.decode(out_ids) if hasattr(tok, "decode") else ""
        good, how = judge(text, r["answer"])
        ok += 1 if good else 0
        detail.append({"prompt": r["prompt"], "answer": r["answer"],
                       "got": text[:200], "how": how, "ok": good})
    return ok, len(rows), {"rows": detail, "wall_s": round(time.time() - t0, 3)}


if __name__ == "__main__":        # 只做契约自检，不连引擎
    import sys
    fails = 0
    cases = [("The answer is 1,234.", "1234", True, "num"),
             ("So 42", "42", True, "num"),
             ("  41", "42", False, ""),
             ("paris", "Paris", True, "exact"),
             ("Answer: Paris.", "Paris", True, "substr"),
             ("", "7", False, "")]
    for text, want, exp_ok, exp_how in cases:
        got_ok, got_how = judge(text, want)
        if (got_ok, got_how) != (exp_ok, exp_how):
            print(f"FAIL judge({text!r}, {want!r}) -> {(got_ok, got_how)}, want {(exp_ok, exp_how)}")
            fails += 1
    print(f"judge self-check: {len(cases) - fails}/{len(cases)}")
    sys.exit(1 if fails else 0)
