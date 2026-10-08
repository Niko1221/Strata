"""After switching to 128K context + k8v4 + conversation parking: repeat the earlier speed tests,
run a ~100K-token needle test with a follow-up, and alternate two conversations to see parking work."""
import json
import pathlib
import sys
import time

import requests

URL = "http://127.0.0.1:8080/v1/chat/completions"
HERE = pathlib.Path(__file__).resolve().parent
DOCS = pathlib.Path(r"<repo>\docs")
out = []


def chat(messages, max_tokens, effort="none"):
    body = {"model": "strata", "messages": messages, "max_tokens": max_tokens,
            "reasoning_effort": effort, "temperature": 0}
    t0 = time.time()
    r = requests.post(URL, json=body, timeout=3600)
    wall = time.time() - t0
    if r.status_code != 200:
        raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
    d = r.json()
    return d["choices"][0]["message"].get("content") or "", d.get("timings") or {}, wall


def record(name, content, t, wall, note=""):
    row = {"name": name, "wall_s": round(wall, 1), "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"),
           "prompt_s": round((t.get("prompt_ms") or 0) / 1000, 1), "prompt_tps": t.get("prompt_per_second"),
           "out_n": t.get("predicted_n"), "out_tps": t.get("predicted_per_second"), "note": note,
           "answer": content[:400]}
    out.append(row)
    print(f"{name:22} wall {row['wall_s']:>6}s | read {row['prompt_n']} new tok (cache {row['cache_n']}) "
          f"in {row['prompt_s']}s @ {row['prompt_tps']}/s | out {row['out_n']} @ {row['out_tps']}/s {note}", flush=True)


def user(text):
    return {"role": "user", "content": text}


# 1. the same three speed tests as before (32K config: code 53.2, prose 32.9, 19K prompt 509 read / 42.7 out)
c, t, w = chat([user("Write a Python function that merges overlapping intervals, with a short explanation.")], 400)
record("speed_code", c, t, w)
c, t, w = chat([user("Write a short story (about 300 words) about a lighthouse keeper who finds a message in a bottle.")], 450)
record("speed_prose", c, t, w)
doc19 = (DOCS / "DETAILS.md").read_text(encoding="utf-8")[:60000]
c, t, w = chat([user("Summarize the following document in 5 bullet points.\n\n" + doc19)], 250)
record("speed_long19k", c, t, w)

# 2. ~100K-token needle, then a follow-up in the same conversation (prompt cache)
names = ["DETAILS.md", "AMD_HIP.md", "INTEL.md", "UNSLOTH_Q4.md", "MULTI_GPU.md", "OLDER_GPUS.md", "TROUBLESHOOTING.md",
         "INSTALL.md", "MODELS.md", "BATCHING.md"]
big = "\n\n".join((DOCS / n).read_text(encoding="utf-8") for n in names if (DOCS / n).exists())
for chars in (330000, 280000, 240000, 200000):
    doc = big[:chars]
    cut = doc.find("\n", int(len(doc) * 0.4))
    doc = doc[:cut] + "\n\n(Note for the archive team: the vault code is 5826-OMEGA-heron.)\n\n" + doc[cut:]
    q1 = ("Below is a very long technical document. Read it, then answer only the question at the end.\n\n=== DOCUMENT ===\n"
          + doc + "\n=== END ===\n\nQuestion: what is the vault code mentioned in the document? Reply with the code only.")
    try:
        c1, t, w = chat([user(q1)], 50)
        record("needle_long", c1, t, w, f"(doc {len(doc)} chars; correct={'5826-OMEGA-heron' in c1})")
        break
    except RuntimeError as e:
        print("  too long, retrying shorter:", str(e)[:120], flush=True)
q2 = "According to the same document, what does `--kv-resident` do? Answer in two sentences."
c2, t, w = chat([user(q1), {"role": "assistant", "content": c1}, user(q2)], 150)
record("needle_followup", c2, t, w, "(same conversation: should read only the new question)")

# 3. two conversations alternating (parking): A1, B1, A2, B2
docA = "\n\n".join((DOCS / n).read_text(encoding="utf-8") for n in ["MODELS.md", "HOW_IT_WORKS.md"])[:40000]
docB = "\n\n".join((DOCS / n).read_text(encoding="utf-8") for n in ["BATCHING.md"])[:40000]
A = [user("Here is a document:\n\n" + docA + "\n\nIn one sentence: which model size is recommended for 64 GB of RAM?")]
B = [user("Here is a document:\n\n" + docB + "\n\nIn one sentence: what does \"parallel\": 2 do?")]
c, t, w = chat(A, 80); record("conv_A1", c, t, w); A.append({"role": "assistant", "content": c})
c, t, w = chat(B, 80); record("conv_B1", c, t, w); B.append({"role": "assistant", "content": c})
A.append(user("And which size is the fastest? One sentence."))
c, t, w = chat(A, 80); record("conv_A2", c, t, w, "(back to A after B)")
B.append(user("Does it make a single request faster on a 12 GB card? One sentence."))
c, t, w = chat(B, 80); record("conv_B2", c, t, w, "(back to B after A)")

(HERE / "eval2_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
try:
    m = requests.get("http://127.0.0.1:8080/metrics", timeout=10).json()
    print("conversation_cache metrics:", json.dumps(m.get("conversation_cache"), ensure_ascii=False))
except Exception as e:
    print("metrics:", e)
