"""Small hands-on evaluation of the local Strata server (127.0.0.1:8080).

Runs a fixed set of tasks one at a time, records the answer, the server's timings and
wall time, and writes everything to eval_results.json next to this script.
"""
import json
import pathlib
import sys
import time

import requests

URL = "http://127.0.0.1:8080/v1/chat/completions"
HERE = pathlib.Path(__file__).resolve().parent
STRATA = pathlib.Path(r"<repo>")


def needle_prompt():
    text = (STRATA / "docs" / "DETAILS.md").read_text(encoding="utf-8")[:70000]
    cut = text.find("\n", len(text) // 2)
    secret = "\n\n(备注：仓库保险柜的密码是 7391-ALPHA-蓝鲸。)\n\n"
    doc = text[:cut] + secret + text[cut:]
    return ("下面是一份很长的技术文档。读完后只回答一个问题：文档里提到的仓库保险柜密码是什么？"
            "只输出密码本身。\n\n=== 文档开始 ===\n" + doc + "\n=== 文档结束 ===\n\n问题：仓库保险柜的密码是什么？")


def summary_prompt():
    text = (STRATA / "README.zh-CN.md").read_text(encoding="utf-8")
    return "用三条要点（每条不超过40字）总结下面这份中文说明：\n\n" + text


TASKS = [
    dict(id="math_zh_high", cat="数学推理（思考：high）", effort="high", max_tokens=6000,
         prompt="一个水池有甲、乙两个进水管和一个丙排水管。甲管单独注满水池需要6小时，乙管单独注满需要4小时，"
                "丙管单独把满池水排空需要8小时。三管同时打开，从空池开始，多少小时能注满？给出最终答案（分数和小数）。"),
    dict(id="math_zh_none", cat="数学推理（思考：关）", effort="none", max_tokens=1500,
         prompt="一个水池有甲、乙两个进水管和一个丙排水管。甲管单独注满水池需要6小时，乙管单独注满需要4小时，"
                "丙管单独把满池水排空需要8小时。三管同时打开，从空池开始，多少小时能注满？给出最终答案（分数和小数）。"),
    dict(id="logic_trap", cat="逻辑陷阱", effort="none", max_tokens=600,
         prompt="Sally has 3 brothers. Each of her brothers has 2 sisters. How many sisters does Sally have? "
                "Answer with the number first, then one sentence of reasoning."),
    dict(id="logic_trap_high", cat="逻辑陷阱（思考：high）", effort="high", max_tokens=4000,
         prompt="Sally has 3 brothers. Each of her brothers has 2 sisters. How many sisters does Sally have? "
                "Answer with the number first, then one sentence of reasoning."),
    dict(id="knowledge", cat="常识问答", effort="none", max_tokens=600,
         prompt="请逐条简短回答（每条一行）：\n1. 鲁迅的原名是什么？\n2. 澳大利亚的首都是哪座城市？\n"
                "3. 《百年孤独》的作者是谁？\n4. 水在标准大气压下的沸点是多少摄氏度？\n"
                "5. 人体最大的器官是什么？\n6. 第一个登上月球的人是谁，哪一年？"),
    dict(id="code_expr", cat="写代码（运行测试）", effort="none", max_tokens=2500,
         prompt="用 Python 写一个函数 evaluate(expr: str) -> float，计算只包含非负整数或小数、+ - * /、"
                "括号和空格的算术表达式，支持一元负号（如 -(2+3)），遵守运算优先级。不能使用 eval/exec/ast。"
                "只输出一个 ```python 代码块，不要解释。"),
    dict(id="code_bug", cat="找 bug", effort="none", max_tokens=1200,
         prompt="下面的二分查找有 bug，指出问题并给出修正后的代码：\n```python\n"
                "def bsearch(a, x):\n    lo, hi = 0, len(a)\n    while lo < hi:\n        mid = (lo + hi) // 2\n"
                "        if a[mid] < x:\n            lo = mid\n        elif a[mid] > x:\n            hi = mid\n"
                "        else:\n            return mid\n    return -1\n```"),
    dict(id="json_strict", cat="严格 JSON 输出", effort="none", max_tokens=600,
         prompt="生成3个虚构人物，只输出一个 JSON 数组，不要任何其它文字或代码块标记。"
                "每个元素包含字段 name（中文名，字符串）、age（整数，20-60）、city（中国城市名）、"
                "skills（字符串数组，恰好2项）。"),
    dict(id="poem", cat="七言绝句", effort="none", max_tokens=400,
         prompt="写一首关于秋夜的七言绝句（四句，每句七个字），只输出诗的四句，每句一行，不要标题和解释。"),
    dict(id="translate", cat="中译英", effort="none", max_tokens=500,
         prompt="把这段话翻译成自然地道的英文：\n"
                "这家小店开了二十多年，老板总是记得熟客的口味。下雨天他会把伞借给没带伞的客人，从来不催着还。"),
    dict(id="summary_zh", cat="中文长文摘要", effort="none", max_tokens=500, prompt_fn=summary_prompt),
    dict(id="needle", cat="长文档找细节（~2万 token）", effort="none", max_tokens=100, prompt_fn=needle_prompt),
    dict(id="tool_call", cat="工具调用", effort="none", max_tokens=400,
         prompt="帮我查一下北京现在的天气，用摄氏度。",
         tools=[{"type": "function", "function": {
             "name": "get_weather", "description": "查询某个城市的当前天气",
             "parameters": {"type": "object", "properties": {
                 "city": {"type": "string", "description": "城市名"},
                 "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                 "required": ["city"]}}}]),
]


def run(task):
    prompt = task["prompt_fn"]() if "prompt_fn" in task else task["prompt"]
    body = {"model": "strata", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": task["max_tokens"], "reasoning_effort": task["effort"], "temperature": 0}
    if "tools" in task:
        body["tools"] = task["tools"]
    t0 = time.time()
    r = requests.post(URL, json=body, timeout=1800)
    wall = time.time() - t0
    r.raise_for_status()
    data = r.json()
    msg = data["choices"][0]["message"]
    return {
        "id": task["id"], "cat": task["cat"], "effort": task["effort"], "wall_s": round(wall, 2),
        "finish": data["choices"][0].get("finish_reason"),
        "content": msg.get("content"), "reasoning": msg.get("reasoning_content") or msg.get("reasoning"),
        "tool_calls": msg.get("tool_calls"), "usage": data.get("usage"), "timings": data.get("timings"),
    }


def main():
    only = set(sys.argv[1:])
    results = []
    for task in TASKS:
        if only and task["id"] not in only:
            continue
        try:
            res = run(task)
        except Exception as e:  # keep going; record the failure
            res = {"id": task["id"], "cat": task["cat"], "error": repr(e)}
        results.append(res)
        t = res.get("timings") or {}
        print(f"{res['id']:16} wall {res.get('wall_s', '-'):>7}s  prompt {t.get('prompt_n', '-'):>6} tok "
              f"@ {t.get('prompt_per_second', '-'):>7}/s  out {t.get('predicted_n', '-'):>5} tok "
              f"@ {t.get('predicted_per_second', '-'):>6}/s  {res.get('finish') or res.get('error', '')}", flush=True)
    out = HERE / "eval_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
