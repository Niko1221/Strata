"""数学数据这条线上唯一需要共享的判据：从一段回答里取出答案数字。

为什么要单独一个文件：`math_probe.py`（HF 探题，要 torch）与 `math_build_data.py`
（把探题结果变成 GRPO prompts / DPO 偏好对，纯 json）必须用**同一个**取数规则 ——
两边各写一份 `re.search(r"-?\\d+")` 看着一样，改一处忘一处就会把「答对了」判成
「答错了」，而错的方向恰好是让 GRPO 拿到错的优势。零阶线不带 torch，所以这条判据
不能住在 math_probe 里。

判据本身（有意写得窄）：
  * 逗号千分位先去掉（"1,234" → "1234"）；
  * 只认第一个整数/负整数，不认小数、分数、科学计数法 —— 题面就是乘法加法，
    出现这些形态说明模型没按「Reply with only the number」答，宁可判错；
  * 取不到就返回 None（调用方据此把它算作未答对，而不是静默当 0）。
"""

from __future__ import annotations

import re


def extract_num(text: str) -> str | None:
    """回答文本 → 答案数字的字符串形式；取不到返回 None。"""
    m = re.search(r"-?\d+", (text or "").replace(",", ""))
    return m.group(0) if m else None
