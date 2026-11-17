"""What a second client does to a single-slot Gyro-S server. Results go to concurrency-log.jsonl.
 A. two (then four) short requests released together, against the same request alone;
 B. a small request that arrives while a long answer is being written;
 C. whether a 30K conversation is still cached after another client's requests, and after another 30K conversation."""
import os, random, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HERE, filler, log, stream_chat

LOG = os.path.join(HERE, "concurrency-log.jsonl")
rng = random.Random(20261113)
KEEP = ("prompt", "cached", "out", "ttft", "tps", "wall", "finish")

def story(i, n=400):
    return [{"role": "user", "content": f"Write a short story about a lighthouse keeper, version {i}-{rng.randint(1000, 9999)}. About 250 words."}], n

def one(tag, msgs, max_tokens, results, delay=0.0):
    time.sleep(delay); t = time.time()
    try:
        r = stream_chat(msgs, max_tokens, timeout=900); rec = {"tag": tag, "start": round(t, 1), **{k: r[k] for k in KEEP}}
    except Exception as e:
        rec = {"tag": tag, "start": round(t, 1), "error": repr(e)[:300]}
    results.append(rec); log(LOG, rec)

def together(label, jobs):
    """jobs: (tag, messages, max_tokens, delay seconds). Runs them in threads and prints one line each."""
    res = []; t0 = time.time()
    th = [threading.Thread(target=one, args=(f"{label}/{tag}", m, mt, res, d)) for tag, m, mt, d in jobs]
    for x in th: x.start()
    for x in th: x.join()
    wall = time.time() - t0
    for r in sorted(res, key=lambda r: r["tag"]):
        print(f"  {r['tag']:<28} " + (f"ERROR {r['error']}" if "error" in r else f"first token {r['ttft']} s, {r['tps']} tok/s, {r['out']} tok, done after {r['wall']} s, cached {r['cached']} of {r['prompt']}"), flush=True)
    tot = sum(r.get("out") or 0 for r in res)
    print(f"  -> {label}: {tot} tokens in {wall:.1f} s = {tot / wall:.1f} tok/s for all clients together", flush=True)
    log(LOG, {"tag": label + "/TOTAL", "tokens": tot, "wall": round(wall, 1), "tps_total": round(tot / wall, 1)})
    return res

print("A. short requests: alone, two together, four together (3 rounds each)", flush=True)
for rnd in range(1, 4):
    m, n = story(rnd); together(f"A{rnd}-alone", [("c1", m, n, 0)])
    together(f"A{rnd}-two", [(f"c{i}", *story(f"{rnd}{i}"), 0) for i in (1, 2)])
    together(f"A{rnd}-four", [(f"c{i}", *story(f"{rnd}{i}"), 0) for i in (1, 2, 3, 4)])

print("\nB. a small request arriving 3 s into a long answer (3 rounds)", flush=True)
for rnd in range(1, 4):
    long_m = [{"role": "user", "content": f"Write a detailed essay of about 1,500 words on tide tables, variant {rng.randint(1000, 9999)}."}]
    small = [{"role": "user", "content": f"Reply with exactly this number and nothing else: {rng.randint(10000, 99999)}"}]
    together(f"B{rnd}", [("long", long_m, 2000, 0), ("small", small, 300, 3.0)])

print("\nC. is a 30K conversation still cached after other requests?", flush=True)
def big(tag):
    code = rng.randint(10000, 99999)
    msgs = [{"role": "user", "content": filler(1330, {5: f"IMPORTANT: the vault code is {code}."}, rng) + "\n\nWhat is the vault code? Reply with the number only."}]
    r = stream_chat(msgs, 800); log(LOG, {"tag": f"C/{tag}-first", **{k: r[k] for k in KEEP}})
    print(f"  {tag} first message: prompt {r['prompt']}, cached {r['cached']}, first token {r['ttft']} s, code right: {str(code) in r['text']}", flush=True)
    return msgs + [{"role": "assistant", "content": r["text"]}]
def follow(tag, conv, note):
    r = stream_chat(conv + [{"role": "user", "content": f"Add {rng.randint(1, 9)} to that number. Reply with the result only."}], 800)
    log(LOG, {"tag": f"C/{tag}-followup", "note": note, **{k: r[k] for k in KEEP}})
    print(f"  {tag} follow-up {note}: cached {r['cached']} of {r['prompt']}, first token {r['ttft']} s", flush=True)
X = big("X")
follow("X", X, "right away")
m, n = story("c-other"); r = stream_chat(m, n); print(f"  (another client's short request: {r['out']} tokens)", flush=True)
follow("X", X, "after 1 unrelated short request")
for i in range(3):
    m, n = story(f"c-other{i}"); stream_chat(m, n)
follow("X", X, "after 3 more unrelated short requests")
Y = big("Y")
follow("Y", Y, "right away")
follow("X", X, "after another 30K conversation (Y) was served")
follow("Y", Y, "after going back to X")
print("done", flush=True)
