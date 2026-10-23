# E529: the standard corpus' 4K request, sent three ways to one engine (:8089), 512 tokens, engine-side speed from the log.
import json, sys, time, urllib.request
fx = json.load(open('/home/user/Strata-022/bench/prompts/standard-v1/requests/performance-prompt-4k-r1.json'))
base = {"model": "strata", "messages": fx["messages"], "stream": False, "max_tokens": 512, "seed": fx.get("seed", 1)}
if fx.get("tools"): base["tools"] = fx["tools"]
arms = {
 "A-greedy-nothink": {**base, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}},
 "B-greedy-think":   {**base, "temperature": 0, "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}, "reasoning_budget_tokens": 0},
 "C-proto-sampled":  {**base, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0, "reasoning_effort": "xhigh", "reasoning_budget_tokens": 0, "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}},
}
order = ["C-proto-sampled", "C-stream", "A-greedy-nothink", "A-stream", "C-stream"]
arms["C-stream"] = {**arms["C-proto-sampled"], "stream": True, "stream_options": {"include_usage": True}}
arms["A-stream"] = {**arms["A-greedy-nothink"], "stream": True, "stream_options": {"include_usage": True}}
for i, name in enumerate(order):
    body = dict(arms[name]); body["messages"] = [dict(m) for m in fx["messages"]]
    # a fresh nonce in the system/user text so nothing is reused between sends
    for m in body["messages"]:
        if isinstance(m.get("content"), str): m["content"] = m["content"].replace("\n", f"\n", 1) + f"\n[run {i} {time.time_ns()}]"; break
    t0 = time.time()
    r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8089/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900)
    if body.get("stream"):
        n=0; usage=None
        for line in r:
            line=line.decode().strip()
            if not line.startswith("data:") or line=="data: [DONE]": continue
            d=json.loads(line[5:]); n+=1
            if d.get("usage"): usage=d["usage"]
        print(f"{i} {name:18} {time.time()-t0:6.1f} s  stream chunks {n} usage {usage and usage.get('completion_tokens')}", flush=True)
    else:
        d = json.load(r); u = d.get("usage", {})
        print(f"{i} {name:18} {time.time()-t0:6.1f} s  completion {u.get('completion_tokens')} finish={d['choices'][0].get('finish_reason')}", flush=True)
