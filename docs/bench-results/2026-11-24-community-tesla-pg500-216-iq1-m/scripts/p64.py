"""Fresh ~51K-token prefill probe against a running Strata server (:8080).

Seeded word-list prompt so every arm gets the identical, cache-cold prompt; speed is read from the
final SSE chunk's engine timings (`timings.prompt_per_second`). Usage: python3 p64.py (server on
127.0.0.1:8080). Prints one line: `prompt_n <tokens> prompt_per_second <tok/s>`.
"""
import json, urllib.request, random
rng = random.Random(2026)
vocab = ("river stone market lantern copper winter garden signal harbor needle orbit velvet canyon "
         "ember ledger quartz meadow anchor violet thunder").split()
body = " ".join(rng.choice(vocab) for _ in range(48500))
msgs = [{"role": "user", "content": "[pg500 64k]\n" + body +
        "\n\nSummarize what kinds of words appear, in five sentences."}]
req = {"model": "strata", "messages": msgs, "max_tokens": 2, "temperature": 0, "stream": True}
r = urllib.request.urlopen(urllib.request.Request(
    "http://127.0.0.1:8080/v1/chat/completions", json.dumps(req).encode(),
    {"Content-Type": "application/json"}), timeout=600)
t = None
for line in r:
    line = line.decode().strip()
    if line.startswith("data:") and '"timings"' in line:
        t = json.loads(line[5:]).get("timings")
print("prompt_n", t.get("prompt_n"), "prompt_per_second", t.get("prompt_per_second"))
