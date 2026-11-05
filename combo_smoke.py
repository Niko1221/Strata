#!/usr/bin/env python3
"""Combo smoke for PR #3643 on the rebased build (jev-port-r3-test engine).

Justin's ask: "combination test set — image+decision, decision only,
decision+then choose, noul, score and then just a couple mmlu questions".

Cases (all against POST /v1/systemone):
  1. decision only, mode=noul   (BoolQ-style yes/no, no label list)
  2. decision only, mode=choice (ARC-style 4 options)
  3. decision only, mode=score  (SST-style 0-4)
  4. image+decision, mode=noul  (POPE-style: is the object in the image?)
  5. image+decision, mode=choice (CV-Bench-style count/color pick)
  6. decision+then choose: decision routes, then.generate runs when the
     gate passes (and is skipped when it does not)
  7. two MMLU questions, mode=choice (A-D)

Exit 0 iff every case returned a well-formed decision (fields present,
probabilities finite and normalised where applicable).
"""
import base64, io, json, struct, sys, time, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18250"
KEY = "local-test"  # the local serve wrapper does not auth; header harmless

def post(body, timeout=300):
    req = urllib.request.Request(
        BASE + "/v1/systemone", data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    return d, time.time() - t0

def png_solid(w, h, rgb, blob=None):
    """Tiny PNG: solid colour with an optional contrasting blob (a filled
    rectangle) so count/colour questions have a real answer."""
    row = bytearray()
    for x in range(w):
        row += bytes(rgb if blob is None or not (blob[0] <= x < blob[2]) else blob[3])
    raw = b"".join(b"\x00" + bytes(row) for _ in range(h))
    def chunk(t, data):
        c = t + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib_crc(c))
    def zlib_crc(b):
        import zlib
        return zlib.crc32(b) & 0xFFFFFFFF
    import zlib
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b""))

results = []

def check(name, ok, detail):
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")

def norm(ps):
    return abs(sum(ps) - 1.0) < 1e-3 and all(0 <= p <= 1 for p in ps)

# 1-3: decision only, the three modes
d, ms = post({"q": "Is a violin a member of the string family?",
              "mode": "noul", "labels": ["yes", "no"]})
dec = d.get("decision") or d
pr = dec.get("pr") or dec.get("probs")
check("decision-only noul", bool(dec.get("label") in ("yes", "no")) and pr and norm(pr),
      f"label={dec.get('label')} pr={[round(p,3) for p in (pr or [])]} {ms*1000:.0f}ms")

d, ms = post({"q": "Which gas do plants absorb from the atmosphere for photosynthesis?",
              "mode": "choice", "labels": ["oxygen", "carbon dioxide", "nitrogen", "hydrogen"]})
dec = d.get("decision") or d
pr = dec.get("pr") or dec.get("probs")
ok = dec.get("label") == "carbon dioxide" and pr and norm(pr)
check("decision-only choice", ok, f"label={dec.get('label')} pr={[round(p,3) for p in (pr or [])]} {ms*1000:.0f}ms")

d, ms = post({"q": "Review sentiment: 'The film was a dreary, joyless slog.'",
              "mode": "score", "labels": ["0", "1", "2", "3", "4"]})
dec = d.get("decision") or d
pr = dec.get("pr") or dec.get("probs")
exp_low = dec.get("label") in ("0", "1") and pr and norm(pr)
check("decision-only score", exp_low, f"label={dec.get('label')} pr={[round(p,3) for p in (pr or [])]} {ms*1000:.0f}ms")

# 4-5: image + decision (solid red 96x96 with a green blob)
img = png_solid(96, 96, (200, 30, 30), blob=(30, 0, 60, (30, 200, 30)))
b64 = "data:image/png;base64," + base64.b64encode(img).decode()

d, ms = post({"q": "Is there any green region in this image?", "mode": "noul",
              "labels": ["yes", "no"], "images": [b64]})
dec = d.get("decision") or d
pr = dec.get("pr") or dec.get("probs")
check("image+decision noul", dec.get("label") == "yes" and pr and norm(pr),
      f"label={dec.get('label')} pr={[round(p,3) for p in (pr or [])]} {ms*1000:.0f}ms")

d, ms = post({"q": "What is the dominant background colour of this image?",
              "mode": "choice", "labels": ["red", "green", "blue", "yellow"], "images": [b64]})
dec = d.get("decision") or d
pr = dec.get("pr") or dec.get("probs")
check("image+decision choice", dec.get("label") == "red" and pr and norm(pr),
      f"label={dec.get('label')} pr={[round(p,3) for p in (pr or [])]} {ms*1000:.0f}ms")

# 6: decision+then (amendment 7 mode 1): gate passes -> then.generate ran
d, ms = post({"q": "Does this message contain a request to reset a banking password?",
              "mode": "noul", "labels": ["yes", "no"],
              "text": "Please reset my banking password immediately, my account is locked.",
              "then": {"generate": {"max_tokens": 40, "prompt": "State in one short sentence what the user wants: "}}})
dec = d.get("decision") or d
then = d.get("then") or dec.get("then") or {}
pr = dec.get("pr") or dec.get("probs")
check("decision+then (gate open)", dec.get("label") == "yes" and bool((then.get("text") or "").strip()),
      "label=%s then=%r %.0fms" % (dec.get('label'), (then.get('text') or '')[:60], ms*1000))

# gate closed -> then.generate NOT run
d, ms = post({"q": "Does this message contain a request to reset a banking password?",
              "mode": "noul", "labels": ["yes", "no"],
              "text": "What a lovely sunny afternoon for a picnic in the park.",
              "then": {"generate": {"max_tokens": 40, "prompt": "State in one short sentence what the user wants: "}}})
dec = d.get("decision") or d
then = d.get("then") or dec.get("then") or {}
check("decision+then (gate closed)", dec.get("label") == "no" and not (then.get("text") or "").strip(),
      f"label={dec.get('label')} then_text_empty={not (then.get('text') or '').strip()} {ms*1000:.0f}ms")

# 7: two MMLU questions
mmlu = [
    ("A 25-year-old man presents with a 2-day history of fever and a productive cough. "
     "Which organism is the most likely cause of community-acquired pneumonia?",
     ["Streptococcus pneumoniae", "Pseudomonas aeruginosa", "Clostridioides difficile", "Candida albicans"], 0),
    ("In economics, what does GDP stand for and measure?",
     ["Gross Domestic Product; total value of goods and services", "General Data Protection; privacy rules",
      "Global Debt Position; national borrowing", "Gross Direct Payment; subsidies"], 0),
]
for i, (q, labels, gold) in enumerate(mmlu, 1):
    d, ms = post({"q": q, "mode": "choice", "labels": labels})
    dec = d.get("decision") or d
    pr = dec.get("pr") or dec.get("probs")
    check(f"mmlu {i}", dec.get("label") == labels[gold] and pr and norm(pr),
          f"label={dec.get('label')} gold={labels[gold]} pr={[round(p,3) for p in (pr or [])]} {ms*1000:.0f}ms")

fails = [n for n, ok, _ in results if not ok]
print(f"\nCOMBO: {len(results)-len(fails)}/{len(results)} pass" + (f"; failures: {fails}" if fails else ""))
sys.exit(1 if fails else 0)
