# Fixed prompts through the running server; the engine log's "drafts accepted" lines carry the speculation numbers.
import json, sys, time, urllib.request
PROMPTS = [
    "Write a Python function that parses an ISO 8601 duration string like P3DT4H12M into seconds, with tests.",
    "A train leaves at 9:40 and travels 237 km at 79 km/h, then 120 km at 60 km/h. When does it arrive? Show the steps.",
    "Explain how a B-tree insert splits nodes, then give a short C++ implementation of insert for order 4.",
    "Summarize the causes and consequences of the 1997 Asian financial crisis in about 300 words.",
    "Refactor this into idiomatic TypeScript with types: function f(a){var r=[];for(var i=0;i<a.length;i++){if(a[i].on){r.push({id:a[i].id,n:a[i].name.trim()})}}return r}",
    "Write a bash script that finds the 10 largest files under a directory, skipping .git, and prints human-readable sizes.",
]
port = sys.argv[1] if len(sys.argv) > 1 else "8091"
temp = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
dump = sys.argv[3] if len(sys.argv) > 3 else None
outs = []
for i, p in enumerate(PROMPTS):
    body = json.dumps({"model": "x", "messages": [{"role": "user", "content": p}], "max_tokens": 700,
                       "temperature": temp, "seed": 1000 + i}).encode()
    t0 = time.time()
    r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                                      {"Content-Type": "application/json"}), timeout=900)
    j = json.load(r)
    print(i, j.get("usage"), f"{time.time()-t0:.1f}s", flush=True)
    outs.append(j["choices"][0]["message"])
if dump: json.dump(outs, open(dump, "w"), indent=1)
