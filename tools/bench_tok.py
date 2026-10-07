#!/usr/bin/env python
"""Benchmark: prefill and decode tok/s for the current running strata serve.

Usage:  .venv\\Scripts\\python tools\\bench_tok.py [--tag NAME]
        (run twice: once while STRATA_HIPBLASLT_TUNING is set in the config, once without,
         passing a different --tag each time so the results can be compared)
"""
import argparse, json, socket, sys, time, urllib.request

API = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "qwen3.8-flash-next-unsloth-ud-iq4_xs"

# A paragraph that tokenizes to a stable number of tokens (Chinese, ~1 tok/char-ish).
PARA = ("复旦大学团队的研究表明，大语言模型的推理能力与训练数据的质量密切相关。"
        "在模型架构、参数量与计算资源一定的情况下，数据的清洗、去重与配比决定了模型的最终表现。"
        "我们沿着这条主线，开展了系统的实验，发现了若干有趣的规律。"
        "例如，高质量代码数据的加入显著提升了模型的数学推理能力；"
        "而多语言语料的合理配比则有助于模型在跨语言任务上的泛化。")

def chat(messages, max_tokens, stream=False, timeout=600):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "stream": stream}
    req = urllib.request.Request(API, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    u = d.get("usage", {})
    return dt, u.get("prompt_tokens", 0), u.get("completion_tokens", 0), d["choices"][0]["message"]["content"]

def build_prompt(target_tokens):
    """Pad PARA until the prompt is >= target tokens (checked by a probe call)."""
    text = ""
    while True:
        text += PARA + "\n"
        n = len(text)
        # rough: Chinese chars ~1 token per 1.3 chars; probe to be exact below
        if n > target_tokens * 2.5:
            break
    # probe exact token count
    dt, pt, ct, _ = chat([{"role": "user", "content": text}], 1)
    if pt < target_tokens:
        # grow again
        return build_prompt_from(text, target_tokens, pt)
    return text, pt

def build_prompt_from(text, target_tokens, cur):
    while cur < target_tokens:
        text += PARA + "\n"
        dt, pt, ct, _ = chat([{"role": "user", "content": text}], 1)
        cur = pt
    return text, cur

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run")
    ap.add_argument("--short", type=int, default=128, help="max_tokens for the short-prompt gen")
    ap.add_argument("--targets", default="512,2048", help="prompt token targets to test prefill at")
    a = ap.parse_args()

    # make sure the server is up
    try:
        urllib.request.urlopen("http://127.0.0.1:8080/v1/models", timeout=10)
    except Exception as e:
        print(f"server not reachable: {e}", file=sys.stderr); sys.exit(1)

    out = {"tag": a.tag, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "cases": []}

    def report(kind, tag, tok_n, dt, extra):
        print(f"[{tag}] {kind}: {tok_n} tokens in {dt*1000:.0f} ms = {tok_n/dt:.1f} tok/s {extra}", flush=True)

    # warmup
    print("warmup ...", flush=True)
    chat([{"role": "user", "content": "你好"}], 16)

    # 1. short prompt, long generation -> decode speed
    for gen in (64, 128):
        dt, pt, ct, _ = chat([{"role": "user", "content": "用中文写一段关于大语言模型发展的介绍。"}], gen)
        dec = ct / dt
        report("decode", f"short-{gen}", ct, dt, f"(prompt {pt})")
        out["cases"].append({"kind": "decode", "label": f"short-{gen}", "prompt_tokens": pt,
                             "completion_tokens": ct, "secs": dt, "tok_s": round(dec, 1)})

    # 2. prefill at several lengths, max_tokens small
    for target in [int(x) for x in a.targets.split(",")]:
        text, pt = build_prompt(target)
        dt, pt2, ct, _ = chat([{"role": "user", "content": text}], 8)
        pref = (pt2 - 1) / dt * 0.00 + pt2 / dt  # prefill rate ~ all prompt tokens over total incl. 8 gen
        report("prefill", f"pt-{target}", pt2, dt, f"(gen {ct})")
        out["cases"].append({"kind": "prefill", "label": f"pt-{target}", "prompt_tokens": pt2,
                             "completion_tokens": ct, "secs": dt, "tok_s": round(pref, 1)})

    # 3. sampled generation at 16 tokens to show sampling path is alive
    dt, pt, ct, content = chat([{"role": "user", "content": "用一句话介绍你自己。"}], 24)
    print(f"[sample] reply: {content}", flush=True)
    out["cases"].append({"kind": "sample", "prompt_tokens": pt, "completion_tokens": ct,
                         "secs": dt, "tok_s": round(ct / dt, 1)})

    fn = f"bench/results/bench-{a.tag}.json"
    with open(fn, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"wrote {fn}", flush=True)

if __name__ == "__main__":
    main()