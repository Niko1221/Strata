#!/usr/bin/env python3
# tools/huihui_env_ab.py - env-knob A/B on the live Huihui host via /metrics deltas.
# usage: huihui_env_ab.py '{"K":"V"}' '{"K":""}' ...   base = current config; run first and last.
import json, subprocess, sys, time, urllib.request

CFG = "/Users/pc/src/Strata-fork/strata-ud-q4_k_xl-huihui.json"
API = "http://127.0.0.1:8180"
PROMPT = "Repeat the task back to me: I will ask you to summarize the following notes. " + "The quick brown fox jumps over the lazy dog number 42. " * 130

def metrics():
    return json.load(urllib.request.urlopen(API + "/metrics", timeout=20))["totals"]

def post(max_tokens, prompt=PROMPT):
    b = {"model": "m", "messages": [{"role": "user", "content": prompt}], "temperature": 0,
         "max_tokens": max_tokens, "thinking_budget": 0}
    r = urllib.request.Request(API + "/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"})
    for _ in range(10):
        try:
            return json.load(urllib.request.urlopen(r, timeout=900))["usage"]
        except urllib.error.HTTPError as e:
            print("  503 waiting:", e.code, flush=True); time.sleep(15)
    raise RuntimeError("engine unavailable")

def arm(name):
    m0, t0 = metrics(), time.time()
    u1 = post(1)            # ~1k-token cold prompt (new conversation each arm)
    m1 = metrics()
    u2 = post(192, "Now write 180 short sentences about rivers and bridges, one per line.")
    m2 = metrics()
    dp = m2.get("prompt_tokens", 0) - m1.get("prompt_tokens", 0)
    np_ = (m2.get("prompt_tokens", 0) - m1.get("prompt_tokens", 0)) - (m2.get("reused", 0) - m1.get("reused", 0))
    pms = m2.get("prompt_ms", 0) - m1.get("prompt_ms", 0)
    dtok = m2.get("output_tokens", 0) - m1.get("output_tokens", 0)
    dms = m2.get("decode_ms", 0) - m1.get("decode_ms", 0)
    prefill = 1000.0 * np_ / max(1.0, pms) if np_ else 0.0
    decode = 1000.0 * dtok / max(1.0, dms)
    print(json.dumps({"arm": name, "decode_tps": round(decode, 2), "dtok": dtok, "dms_ms": dms,
                      "newtok": np_, "cold_prefill_tps": round(prefill, 1), "wall_s": round(time.time() - t0, 1)}), flush=True)

def set_env(patch):
    d = json.load(open(CFG))
    for k, v in patch.items():
        if v == "":
            d["env"].pop(k, None)
        else:
            d["env"][k] = v
    json.dump(d, open(CFG, "w"), indent=2)

def restart():
    subprocess.run(["bash", "-c", "cd /Users/pc/src/Strata-fork && tools/huihui_session.sh restart huihui"],
                   check=True, capture_output=True, text=True)
    time.sleep(2)

if __name__ == "__main__":
    arms = [json.loads(a) for a in sys.argv[1:]]
    import shutil
    shutil.copy(CFG, "/tmp/huihui_cfg_ab.bak")
    try:
        for i, patch in enumerate(arms):
            set_env(patch)
            restart()
            arm(("base" if not patch else ",".join(f"{k}={v}" for k, v in patch.items())) + f"#{i}")
    finally:
        shutil.copy("/tmp/huihui_cfg_ab.bak", CFG)
        restart()
        print("CONFIG_RESTORED", flush=True)
