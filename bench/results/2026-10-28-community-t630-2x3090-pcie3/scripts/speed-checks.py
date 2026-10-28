"""speed-checks.py - the speed and agent-compatibility checks behind every number in #115, #119 and
#132: a second system message, /v1/messages with tools and a thinking budget, then speed (short prompt
thinking off/on x3, a 32,667-token prompt, the same prompt again from the cache).

Usage: python speed-checks.py [base_url]   (default http://127.0.0.1:11434, whatever serves the box)
"""
import json
import os
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:11434"
MODEL = "qwen3.8-27b"
# Servers that require a key (the T630 box): STRATA_KEY is sent as x-api-key and as a Bearer token.
KEY = os.environ.get("STRATA_KEY", "")
AUTH = {"x-api-key": KEY or "x", **({"Authorization": f"Bearer {KEY}"} if KEY else {})}


def post(path, body, timeout=600):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **AUTH,
                                          "anthropic-version": "2023-06-01"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r), time.time() - t


def stream_chat(messages, max_tokens, thinking):
    """OpenAI streaming: returns (ttft, gen_seconds, completion_tokens, prompt_tokens)."""
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}, "temperature": 1.0, "top_p": 0.95,
            "chat_template_kwargs": {"enable_thinking": thinking}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **AUTH})
    t0 = time.time(); first = None; usage = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if first is None and (d.get("content") or d.get("reasoning_content") or d.get("reasoning")):
                    first = time.time()
    end = time.time()
    return first - t0, end - first, usage["completion_tokens"], usage["prompt_tokens"]


def speed():
    task = [{"role": "user", "content": "Write a complete Python module implementing an LRU cache "
             "class with get, put, delete, resize and a __repr__, plus thorough pytest tests. "
             "Code only."}]
    print("== speed")
    for label, thinking in (("thinking off", False), ("thinking on", True)):
        rates = []
        for _ in range(3):
            ttft, gen, n, p = stream_chat(task, 1024, thinking)
            rates.append(n / gen)
        print(f"  {label}: {', '.join(f'{r:.1f}' for r in rates)} tok/s (short prompt)")
    # long context: ~30K tokens of real code as the prompt, then a short answer
    # The 32,667-token prompt: dungeon-door source concatenated, kept beside this script but not in git
    # (it is another repo's code). A missing file is an error, never a smaller prompt: when this script was
    # renamed from vllm_checks.py, a path built with str.replace silently fell back to reading the script
    # itself, a 2,117-token "long prompt" (2026-10-14, #132).
    big = os.path.join(os.path.dirname(os.path.abspath(__file__)), "big_prompt.txt")
    if not os.path.exists(big):
        sys.exit(f"missing {big}: the 32K-token prompt file (see box/bench/README note in .gitignore)")
    src = open(big, encoding="utf-8").read()
    # A unique first line keeps a second run on the same server from reading the first run's prefix cache
    # (2026-10-24, T630 sweep). SPEED_NONCE=off gives the original prompt.
    nonce = os.environ.get("SPEED_NONCE") or f"{time.time():.0f}"
    if nonce != "off":
        src = f"// run {nonce}\n" + src
    msgs = [{"role": "system", "content": "You are a careful code reviewer."},
            {"role": "user", "content": src + "\n\nList the five most important functions above "
             "and what each does, then write a unit test for one of them."}]
    ttft, gen, n, p = stream_chat(msgs, 1024, False)
    print(f"  long prompt {p} tokens: prompt read {p / ttft:.0f} tok/s (TTFT {ttft:.1f}s), "
          f"generation {n / gen:.1f} tok/s")
    ttft2, gen2, n2, p2 = stream_chat(msgs, 512, False)
    print(f"  same prompt again (prefix cache): TTFT {ttft2:.2f}s, generation {n2 / gen2:.1f} tok/s")


def anthropic_tools_and_budget():
    print("== /v1/messages: tools + thinking budget")
    body = {"model": MODEL, "max_tokens": 4096,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "system": "You are a coding agent. Use the tools.",
            "tools": [{"name": "read_file", "description": "Read a file from the repo",
                       "input_schema": {"type": "object", "properties": {"path": {"type": "string"}},
                                        "required": ["path"]}}],
            "messages": [{"role": "user", "content": "Open server/engine/combat.js and tell me what it exports."}]}
    j, dt = post("/v1/messages", body)
    kinds = [b.get("type") for b in j.get("content", [])]
    tool = [b for b in j.get("content", []) if b.get("type") == "tool_use"]
    think = sum(len(b.get("thinking", "")) for b in j.get("content", []) if b.get("type") == "thinking")
    print(f"  stop_reason={j.get('stop_reason')} blocks={kinds} usage={j.get('usage')}")
    print(f"  tool_use: {json.dumps(tool[0]['input']) if tool else 'NONE'}; thinking chars: {think}; {dt:.1f}s")
    # thinking cap: a problem that invites long thinking, budget 1024 tokens
    body2 = {"model": MODEL, "max_tokens": 6000, "thinking": {"type": "enabled", "budget_tokens": 1024},
             "messages": [{"role": "user", "content": "Find all integer solutions to x^3 + y^3 + z^3 = 42 "
                           "with |x|,|y|,|z| < 50, reasoning carefully. Then give the answer."}]}
    j2, dt2 = post("/v1/messages", body2)
    tb = "".join(b.get("thinking", "") for b in j2.get("content", []) if b.get("type") == "thinking")
    txt = "".join(b.get("text", "") for b in j2.get("content", []) if b.get("type") == "text")
    print(f"  budget 1024: thinking ~{len(tb) // 4} tokens (chars/4), answer {len(txt)} chars, "
          f"output_tokens={j2.get('usage', {}).get('output_tokens')}, stop={j2.get('stop_reason')}")


def two_system_messages():
    print("== template: second system message")
    j, _ = post("/v1/chat/completions", {
        "model": MODEL, "max_tokens": 64, "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "system", "content": "You are terse."},
                     {"role": "user", "content": "Say hi."},
                     {"role": "assistant", "content": "Hi."},
                     {"role": "system", "content": "<total_tokens>100000 tokens left</total_tokens>"},
                     {"role": "user", "content": "Say bye."}]})
    print("  ok:", repr(j["choices"][0]["message"]["content"])[:80])


def spec_metrics():
    print("== MTP acceptance (from /metrics)")
    m = urllib.request.urlopen(urllib.request.Request(BASE + "/metrics", headers=AUTH), timeout=10).read().decode()
    want = {}
    for line in m.splitlines():
        if line.startswith("vllm:spec_decode_num_") and not line.startswith("#"):
            k, v = line.rsplit(" ", 1)
            name = k.split("{")[0]
            want[name] = want.get(name, 0) + float(v)
    for k, v in sorted(want.items()):
        print(f"  {k}: {v:.0f}")
    d, a = want.get("vllm:spec_decode_num_draft_tokens_total"), want.get("vllm:spec_decode_num_accepted_tokens_total")
    if d:
        print(f"  acceptance: {a / d:.0%}")


if __name__ == "__main__":
    for fn in (two_system_messages, anthropic_tools_and_budget, speed, spec_metrics):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {e!r}")
