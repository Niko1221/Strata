"""引擎侧 A/B：**同一个进程、同一份权重**，只切请求里的 `"lora": false`。

与 `ab_identity.py` 的分工：
  - `ab_identity.py` 在训练进程里比 base / zero / trained 三臂（不需要起引擎，也不需要
    adapter GGUF），回答的是「训练有没有改变行为」。
  - 这个脚本比的是「训练产物回到推理引擎之后还在不在」，也就是 `--lora` 这条路：
    引擎起来时带 `--lora developer.gguf`，同一条 prompt 发两次，一次 `"lora": false`。

为什么这两件事要分开验：`ab_identity.py` 用的是 .pt 里的 A/B 矩阵，引擎用的是
`lora_to_gguf.py` 转出来的 GGUF。转换那一层有它自己的约定（只写
`attn_output` / `ssm_out` 两类矩阵、ne 顺序、缩放 s = alpha / rank），错一个字节就会
既不像 base 也不像微调 —— 而在训练进程里看是「完全正常」的。

用法（引擎已在 --serve 下带 --lora 起来）：

    python3 ab_engine.py --engine http://127.0.0.1:8095 --out ab_engine.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def ask(engine: str, prompt: str, max_new: int, lora: bool | None, timeout: float = 900) -> dict:
    """lora=None 表示请求里不带这个键（引擎默认开着）；False 表示这一条关掉适配器。"""
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if lora is not None:
        # 请求体本身就是 sampling dict（server.py: "the request is also the sampling dict"），
        # 所以 `lora` 是**顶层**字段。放进 "sampling": {...} 里不会报错，会被静默忽略 ——
        # 那样两次请求都跑适配器，A/B 变成自己跟自己比。
        body["lora"] = lora
    req = urllib.request.Request(
        engine.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    # 引擎是本机进程：绕开系统代理，否则 http_proxy 会把 127.0.0.1 也转发出去
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    text = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    usage = data.get("usage") or {}
    return {"text": text.strip(), "tokens": usage.get("completion_tokens")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="http://127.0.0.1:8095")
    ap.add_argument("--prompts", default=os.path.join(HERE, "prompts_identity.jsonl"))
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--out", default=os.path.join(HERE, "ab_engine.json"))
    args = ap.parse_args()

    prompts = [json.loads(line) for line in open(args.prompts, encoding="utf-8") if line.strip()]
    res = {"engine": args.engine, "rows": []}
    for item in prompts:
        q = item["prompt"]
        on = ask(args.engine, q, args.max_new, True)         # 带 LoRA
        off = ask(args.engine, q, args.max_new, False)       # "lora": false
        changed = on["text"] != off["text"]
        res["rows"].append({"prompt": q, "kind": item.get("kind", ""),
                            "lora_on": on, "lora_off": off, "changed": changed})
        print(f"Q: {q}")
        print(f"   lora off  {off['text'][:100]}")
        print(f"   lora on   {on['text'][:100]}")
        print(f"   → {'变了' if changed else '没变'}\n", flush=True)

    # `"lora": false` 这一支必须与「引擎不带 --lora 起」完全一致，否则差异来自
    # 开关本身（re-prefill / 状态重建），不能算到适配器头上。这条不在本脚本里验：
    # 那需要两个引擎进程，见 DEMO-IDENTITY.md 的「对照」一节。
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
