import json, sys, time, urllib.request

KEY = "sk-ouzhiyingyyds666!"
URL = "http://127.0.0.1:8081/v1/chat/completions"

PROMPT = ("请详细对比分析 Transformer 架构中Multi-Head Attention 与 LinearAttention "
          "在计算复杂度、显存占用及长序列建模能力上的差异。要求：\n"
          "1.分别给出两者的时间复杂度和空间复杂度数学推导；\n"
          "2.用Markdown表格总结核心区别；\n"
          "3.列举至少3个LinearAttention 的代表性变体模型并简述其改进思路；\n"
          "4.全程使用中文，保持技术严谨性，不要输出任何寒喧或过渡语句。")

def run(temp, max_tokens, label):
    body = {"model": "Qwen3.8-Flash-Next", "stream": True, "max_tokens": max_tokens,
            "temperature": temp,
            "messages": [{"role": "user", "content": PROMPT}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    t_first = None
    n_chunks = 0
    chars = 0
    usage = None
    finish = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            d = line[6:].strip()
            if d == "[DONE]":
                break
            try:
                obj = json.loads(d)
            except json.JSONDecodeError:
                continue
            if "error" in obj:
                print(f"{label}: ERROR {obj['error'].get('code')}: {obj['error'].get('message','')[:160]}")
                return
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                delta = ch.get("delta") or {}
                c = delta.get("content")
                if c:
                    if t_first is None:
                        t_first = time.time()
                    n_chunks += 1
                    chars += len(c)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    t_end = time.time()
    toks = (usage or {}).get("completion_tokens") or n_chunks
    ttft = (t_first - t0) if t_first else float("nan")
    dec = (t_end - t_first) if t_first else float("nan")
    speed = (toks - 1) / dec if dec and dec > 0 else float("nan")
    ptoks = (usage or {}).get("prompt_tokens")
    print(f"{label}: prompt={ptoks} tok  out={toks} tok  TTFT={ttft:.2f}s  decode={dec:.2f}s  "
          f"**{speed:.1f} tok/s**  finish={finish}  chars={chars}")

if __name__ == "__main__":
    for spec in sys.argv[1:]:
        temp, mx, label = spec.split(":")
        run(float(temp), int(mx), label)
