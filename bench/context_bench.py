"""Benchmark a Strata context setting: decode tok/s (headline) + what drives it.

Context length only costs you something in two places, and this measures both:
  1. the expert cache SHRINKS as the KV cache grows, so decode tok/s falls even on a
     short prompt - that is the steady-state cost of a long window;
  2. a prompt that actually FILLS the window costs prefill time.

So: short-prompt decode for (1), long-prompt prefill for (2).
"""
import json, random, string, sys, time, urllib.request

URL = "http://127.0.0.1:{port}/v1/chat/completions"

def ask(prompt, port, max_tokens=160, effort="none"):
    body = json.dumps({"model": "x", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0,
                       "reasoning_effort": effort}).encode()
    req = urllib.request.Request(URL.format(port=port), data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    return (time.time() - t0, d["usage"]["prompt_tokens"], d["usage"]["completion_tokens"],
            d["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0))

def salt(n=48):
    return "".join(random.choice(string.ascii_letters + string.digits) for _ in range(n))

def main():
    port = int(sys.argv[1])
    label = sys.argv[2]
    approx_ctx = int(sys.argv[3])

    ask("warm", port, 4)                                  # load the path, not the measurement

    # (1) steady-state decode. A fresh prompt each trial: the server keeps a conversation
    #     cache, so repeating one verbatim measures the cache, not generation. Median over
    #     three, and reject a trial whose prompt came back partly cached.
    dec = []
    for _ in range(3):
        _, pt, ct, cached = ask(f"[ref {salt()}] Count from 1 to 80 separated by commas. "
                                f"Output only the list.", port, 160)
        if ct > 40 and cached == 0:                       # a truncated or cached run skews tok/s
            dec.append(ct)
    dec.sort()
    decode = dec[len(dec) // 2] if dec else 0.0

    # (2) a prompt that genuinely fills a window. ~1 token per 4 chars, so ask for
    #     roughly 85% of the context and let the tokeniser land where it lands.
    target = int(approx_ctx * 0.85)
    filler = ("The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor "
              "jugs. How vexingly quick daft zebras jump! Sphinx of black quartz, judge my vow. ")
    reps = max(1, target // max(1, len(filler) // 4))
    prefill = []
    for _ in range(2):
        p = f"[ref {salt()}] Read this, then answer with one word: what jumps?\n\n" + filler * reps + "\nAnswer:"
        dt, pt, ct, cached = ask(p, port, 8)
        prefill.append((pt, dt))
    prefill.sort()
    ptok, ptime = prefill[len(prefill) // 2]

    print(f"{label}")
    print(f"  decode       {decode:5.1f} tok/s   (160-token generation, median of 3)")
    print(f"  prefill      {ptok:6d} tok in {ptime:6.2f}s = {ptok/ptime:6.0f} tok/s  (cold, salted prompt)")
    print(f"  TTFT         {ptime:6.2f}s for {ptok} prompt tokens")

if __name__ == "__main__":
    main()
