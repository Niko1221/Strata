"""Soak test: keeps one Gyro-S server busy for DURATION seconds with a repeating mix of requests, including prompts
near the full 64K window, clients that walk away, and a prompt that is too large. Every request goes to soak-log.jsonl.
Exit code 2 = the server stopped answering.   usage: soak.py [seconds, default 7200]"""
import os, random, sys, time, urllib.error
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HERE, MODEL, filler, free_ram_gb, get, log, port_open, post_json, stream_chat

DURATION = int(sys.argv[1]) if len(sys.argv) > 1 else 7200
SMOKE = len(sys.argv) > 2 and sys.argv[2] == "smoke"      # one quick cycle with prompts a tenth of the size
SCALE = 0.1 if SMOKE else 1.0
LOG = os.path.join(HERE, "soak-smoke.jsonl" if SMOKE else "soak-log.jsonl")
seed = int(time.time()); rng = random.Random(seed)
T0 = time.time()
state = {"per_line": None, "n": 0, "fail": 0, "cycle": 0}

class ServerDown(Exception): pass

def server_alive(wait=90):
    end = time.time() + wait
    while time.time() < end:
        if port_open():
            try:
                if get("/health")[0] == 200: return True
            except Exception: pass
        time.sleep(5)
    return False

def record(kind, t_start, ok, **kw):
    state["n"] += 1; state["fail"] += (not ok)
    rec = {"n": state["n"], "cycle": state["cycle"], "kind": kind, "start": round(t_start, 1), "end": round(time.time(), 1), "ok": bool(ok),
           "ram_free_gb": free_ram_gb(), **kw}
    log(LOG, rec)
    brief = {k: v for k, v in rec.items() if k in ("prompt", "cached", "out", "ttft", "tps", "wall", "found", "status", "error", "note")}
    print(f"[{time.strftime('%H:%M:%S')}] +{(time.time() - T0) / 60:5.1f} min  c{state['cycle']} #{state['n']:<3} {'ok ' if ok else 'BAD'} {kind:<14} {brief}", flush=True)

def guarded(kind, fn):
    """Runs one request. A connection-level failure triggers a liveness check; a dead server ends the run."""
    t = time.time()
    try:
        fn(t)
    except urllib.error.HTTPError as e:
        record(kind, t, False, status=e.code, error=e.read().decode("utf-8", "replace")[:300])
    except Exception as e:
        record(kind, t, False, error=repr(e)[:300])
        if not server_alive():
            log(LOG, {"kind": "SERVER_DOWN", "after": kind, "at": round(time.time(), 1)})
            print("SERVER DOWN after", kind, flush=True); raise ServerDown()

def strip(r):
    return {k: r[k] for k in ("prompt", "cached", "out", "ttft", "tps", "wall", "finish", "reasoning_chars")}

def short(t):
    r = stream_chat([{"role": "user", "content": f"Write a short story about a lighthouse keeper and the number {rng.randint(100, 999)}. About 300 words."}], 500)
    record("short", t, r["out"] > 50, **strip(r))

def recall(target, follow=False):
    def run(t):
        n = int(target * SCALE / state["per_line"])
        codes = [f"{rng.choice('ALFA BRAVO CHARLIE DELTA ECHO FOXTROT GOLF HOTEL INDIA KILO LIMA'.split())}-{rng.randint(1000, 9999)}" for _ in range(3)]
        facts = {3: f"IMPORTANT: the first secret code is {codes[0]}.", n // 2: f"IMPORTANT: the second secret code is {codes[1]}.",
                 n - 2: f"IMPORTANT: the third secret code is {codes[2]}."}
        msgs = [{"role": "user", "content": filler(n, facts, rng) + "\n\nList the three secret codes from the records above, in order."}]
        r = stream_chat(msgs, 1500)
        found = sum(c in r["text"] for c in codes)
        record(f"recall{target // 1000}K", t, found == 3, found=found, **strip(r))
        if follow:
            t2 = time.time()
            msgs += [{"role": "assistant", "content": r["text"]}, {"role": "user", "content": "Which of the three codes has the largest number? Reply with that code only."}]
            r2 = stream_chat(msgs, 800)
            best = max(codes, key=lambda c: int(c.split("-")[1]))
            record(f"followup{target // 1000}K", t2, best in r2["text"] and (r2["cached"] or 0) > target * SCALE * 0.8, found=int(best in r2["text"]), **strip(r2))
    return run

TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
def tool_loop(t):
    temp = rng.randint(-5, 35)
    msgs = [{"role": "user", "content": "What is the temperature in Turin right now? Use the tool, then answer in one sentence."}]
    s, d = post_json("/v1/chat/completions", {"model": MODEL, "messages": msgs, "tools": TOOLS, "max_tokens": 1500})
    if s != 200: return record("tool_loop", t, False, status=s, error=str(d)[:200])
    m = d["choices"][0]["message"]; calls = m.get("tool_calls") or []
    if not calls: return record("tool_loop", t, False, note="no tool call", out=d["usage"]["completion_tokens"])
    msgs.append({k: v for k, v in m.items() if k in ("role", "content", "tool_calls", "reasoning_content") and v is not None})
    msgs.append({"role": "tool", "tool_call_id": calls[0].get("id"), "content": '{"city": "Turin", "temp_c": %d, "sky": "fog"}' % temp})
    s, d = post_json("/v1/chat/completions", {"model": MODEL, "messages": msgs, "tools": TOOLS, "max_tokens": 1500})
    txt = (d["choices"][0]["message"].get("content") or "") if s == 200 else ""
    record("tool_loop", t, s == 200 and str(temp) in txt, status=s, out=d["usage"]["completion_tokens"] if s == 200 else None,
           cached=(d["usage"].get("prompt_tokens_details") or {}).get("cached_tokens") if s == 200 else None, wall=round(time.time() - t, 1))

def anthropic(t):
    a, b = rng.randint(12, 99), rng.randint(12, 99)
    s, d = post_json("/v1/messages", {"model": MODEL, "max_tokens": 1500, "messages": [{"role": "user", "content": f"What is {a} times {b}? Reply with the number only."}]}, anthropic=True)
    txt = "".join(x.get("text", "") for x in d.get("content", []) if isinstance(x, dict)) if s == 200 else ""
    record("anthropic", t, s == 200 and str(a * b) in txt.replace(",", ""), status=s, wall=round(time.time() - t, 1),
           out=(d.get("usage") or {}).get("output_tokens") if s == 200 else None)

def long_output(t):
    r = stream_chat([{"role": "user", "content": "Write a very long, detailed essay (at least 4,000 words) on the history of lighthouses, with many sections."}], 600 if SMOKE else 6000)
    record("long_output", t, r["out"] > (300 if SMOKE else 3000), **strip(r))

def abort_prefill(t):
    """A client that gives up while the server is still reading a 30K prompt."""
    n = int(30000 / state["per_line"])
    try:
        r = stream_chat([{"role": "user", "content": filler(n, {}, rng) + "\n\nSummarize the records in 200 words."}], 600, timeout=6, deadline=6)
        record("abort_prefill", t, True, note="client disconnected after 6 s (server was sending lines)" if r["aborted"] else "answered within 6 s, nothing to abort")
    except Exception as e:
        if "timed out" not in repr(e).lower(): raise
        record("abort_prefill", t, True, note="client disconnected after 6 s")

def abort_decode(t):
    r = stream_chat([{"role": "user", "content": "Count from 1 to 2000 in words, one number per line."}], 6000, stop_after_chunks=150)
    record("abort_decode", t, r["aborted"], note="client disconnected after 150 chunks", ttft=r["ttft"], wall=r["wall"])

def after_abort(t):
    r = stream_chat([{"role": "user", "content": f"Reply with exactly this number and nothing else: {rng.randint(10000, 99999)}"}], 800)
    record("after_abort", t, r["out"] > 0 and r["finish"] is not None, **strip(r))

def too_large(t):
    n = int(72000 / state["per_line"])
    try:
        r = stream_chat([{"role": "user", "content": filler(n, {}, rng) + "\n\nReply OK."}], 200)
        record("too_large", t, False, note="a 72K prompt was accepted", **strip(r))
    except urllib.error.HTTPError as e:
        record("too_large", t, e.code == 400, status=e.code, error=e.read().decode("utf-8", "replace")[:160], wall=round(time.time() - t, 1))

def calibrate(t):
    r = stream_chat([{"role": "user", "content": filler(500, {}, rng) + "\n\nReply with the single word OK."}], 300)
    state["per_line"] = r["prompt"] / 500
    record("calibrate", t, True, note=f"{state['per_line']:.2f} tokens per line", **strip(r))

CYCLE = [("short", short), ("recall20K", recall(20000)), ("tool_loop", tool_loop), ("recall40K", recall(40000)), ("anthropic", anthropic),
         ("recall55K", recall(55000, follow=True)), ("abort_prefill", abort_prefill), ("after_abort", after_abort), ("recall60K", recall(60000)),
         ("long_output", long_output), ("abort_decode", abort_decode), ("after_abort", after_abort), ("recall63K", recall(62500)),
         ("too_large", too_large), ("short", short)]

print(f"soak: {DURATION} s, seed {seed}, server alive: {server_alive(10)}", flush=True)
log(LOG, {"kind": "START", "duration": DURATION, "seed": seed, "at": round(T0, 1)})
code = 0
try:
    guarded("calibrate", calibrate)
    if not state["per_line"]: raise ServerDown()
    while time.time() - T0 < DURATION and not (SMOKE and state["cycle"] >= 1):
        state["cycle"] += 1
        for kind, fn in CYCLE:
            if time.time() - T0 >= DURATION: break
            guarded(kind, fn)
except ServerDown:
    code = 2
log(LOG, {"kind": "END", "requests": state["n"], "failed": state["fail"], "cycles": state["cycle"], "minutes": round((time.time() - T0) / 60, 1), "exit": code})
print(f"soak finished: {state['n']} requests, {state['fail']} failed, {state['cycle']} cycles, {(time.time() - T0) / 60:.1f} min, exit {code}", flush=True)
sys.exit(code)
