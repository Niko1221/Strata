# long.py <log> <spec_min_p>: a ~25K-token code prompt, then a second question on the same prefix (the prompt cache)
import json, sys, urllib.request, re
log, pm = sys.argv[1], float(sys.argv[2])
src = "".join(open(f).read() for f in ["<strata>/src/core/verify.cpp","<strata>/src/core/mtp.cpp"])[:330000]
qs = ["Rewrite the function window_logprobs from the file above with a comment on every line. Output only code.",
      "Now list every environment variable the file reads and say in one line what each one does."]
msgs = [{"role": "user", "content": "```cpp\n" + src + "\n```\n\n" + qs[0]}]
for i in range(2):
    b = json.dumps({"model": "x", "messages": msgs, "max_tokens": 600, "temperature": 0,
                    "strata_tune": {"spec_min_p": pm}}).encode()
    j = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", b,
                                                                {"Content-Type": "application/json"}), timeout=3600))
    line = [l for l in open(log) if "generated in" in l][-1]
    m = re.search(r"prompt (\d+) tokens = (\d+) reused.* (\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)", line)
    print(f"req {i}: prompt {m[1]} reused {m[2]} gen {m[3]} {m[5]} tok/s acc {m[6]}/{m[7]} tok/win {int(m[3])/(int(m[3])-int(m[6])):.2f}", flush=True)
    msgs += [j["choices"][0]["message"], {"role": "user", "content": qs[1]}]
