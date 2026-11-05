#!/usr/bin/env python3
"""Compare this stack's next-token distribution with the running llama.cpp server.

The server tokenizes the prompt and reports its own top-5 logprobs; the same token ids then go
through build-cann/model_seq, which prints its top-5. Agreement means the CANN port reproduces the
model, not just its own CPU reference.

    python3 compare_with_llama.py [model.gguf] [prompt]
"""
import json
import os
import subprocess
import sys
import urllib.request

BASE = "http://127.0.0.1:18789"
DEFAULT_MODEL = "/data/strata-qwen36/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
DEFAULT_PROMPT = "The capital of France is"


def post(path, payload):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    prompt = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_PROMPT

    ids = post("/tokenize", {"content": prompt})["tokens"]
    resp = post("/completions", {"prompt": prompt, "max_tokens": 1, "temperature": 0.0, "logprobs": 8})
    llama = resp["completion_probabilities"][0]["top_logprobs"][:5]

    binary = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "build-cann", "model_seq")
    out = subprocess.run([binary, model, ",".join(str(i) for i in ids)],
                         capture_output=True, text=True, timeout=7200)
    ours = []
    log_z = None
    for line in out.stdout.splitlines():
        line = line.strip()
        if line.startswith("LOGP "):
            log_z = float(line.split()[1])
        elif line.startswith("token ") and "logit" in line:
            parts = line.split()
            ours.append([int(parts[1]), float(parts[-1])])
    if log_z is not None:
        # the engine prints logits plus the log-partition, so logprob = logit - log_z
        ours = [(i, v - log_z) for i, v in ours]
    if len(ours) < 5:
        print("model_seq produced no logits")
        print(out.stdout[-500:], out.stderr[-500:])
        return 1

    print(f"prompt {prompt!r} -> {len(ids)} tokens")
    print(f"{'llama.cpp':>22}   {'cann/':>22}")
    agree = 0
    for i in range(5):
        a = llama[i]
        b = ours[i]
        same = a["id"] == b[0]
        agree += 1 if same else 0
        print(f"  {a['id']:6d} {a['logprob']:8.3f}      {b[0]:6d} {b[1]:8.3f}   {'' if same else '(different)'}")
    ids_llama = [t["id"] for t in llama]
    ids_ours = [t[0] for t in ours]
    same_set = sorted(ids_llama) == sorted(ids_ours)
    print(f"\n  same set of five: {same_set}   same order: {agree}/5")
    worst = max(abs(llama[i]["logprob"] - ours[i][1]) for i in range(5))
    print(f"  largest logprob difference: {worst:.3f}")
    return 0 if same_set else 1


if __name__ == "__main__":
    sys.exit(main())
