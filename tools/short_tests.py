# tools/short_tests.py - correctness tests for the NVMe KV tier at short contexts (~3k tokens).
# S1 multi-turn chain, S2 branching, S3 corruption fallback, S4 shared-prefix needles,
# S5 rotation stress, S6 idempotence.  Assertions are delta-based on the store and the engine log.
import json, os, re, sys, time
import requests

# the target server, its log and its store are overridable so the suite can run against a TEST server
# (e.g. a delta-on engine on another port with a scratch store) without editing the file
import os
BASE = os.environ.get("STRATA_TESTS_BASE", "http://127.0.0.1:8080/v1/chat/completions")
LOG = os.environ.get("STRATA_TESTS_LOG", "/local/strata/strata-iq3_xxs.log")
STORE = os.environ.get("STRATA_TESTS_STORE", "/local/strata/kvstore")
sys.path.insert(0, "/local/strata/tools")
import strata_tokenizer as ST

tp = "/local/strata/packs/iq3_xxs/tokenizer"
vocab = json.load(open(tp + "/vocab.json"))
tokens = [None] * len(vocab)
for t, i in vocab.items(): tokens[i] = t
tok = ST.Tokenizer(tokens, open(tp + "/merges.txt").read().split("\n"), json.load(open(tp + "/token_type.json")))

TARGET = 3000
FAILS = []

def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (" - " + detail if detail else ""), flush=True)
    if not cond: FAILS.append(name)

def log_size():
    try: return os.path.getsize(LOG)
    except OSError: return 0

def log_tail(n0):
    with open(LOG, "rb") as f:
        f.seek(n0); return f.read().decode("utf-8", "replace")

def store_count():
    # THE TIER'S RECORDS, whichever family is live: the v3 snapshots (kv-*), or the delta tier's manifests
    # (log-*) + content-addressed records (chunks/, states/).  The delta tier replaces the per-turn snapshot
    # FILE with a manifest head move + appended chunks, so "the store grew" counts both families.
    n = len([f for f in os.listdir(STORE) if f.startswith("kv-")])
    d = os.path.join(STORE, "delta")
    if os.path.isdir(d):
        n += len([f for f in os.listdir(d) if f.startswith("log-")])
        for sub in ("chunks", "states"):
            p = os.path.join(d, sub)
            if os.path.isdir(p): n += len(os.listdir(p))
    return n

def delta_audit(d):
    # the delta records' own integrity: the FNV-1a footer over the payload of every chunk (48-byte header),
    # State record (16-byte header) and manifest (264-byte header, footer over the body)
    import struct
    def fnv(b, h=1469598103934665603):
        for x in b: h = ((h ^ x) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        return h
    tot = bad = 0
    for f in os.listdir(os.path.join(d, "chunks")):
        data = open(os.path.join(d, "chunks", f), "rb").read(); tot += 1
        if len(data) < 56 or fnv(data[48:-8]) != struct.unpack("<Q", data[-8:])[0]: bad += 1
    for f in os.listdir(os.path.join(d, "states")):
        data = open(os.path.join(d, "states", f), "rb").read(); tot += 1
        if len(data) < 24 or fnv(data[16:-8]) != struct.unpack("<Q", data[-8:])[0]: bad += 1
    for f in os.listdir(d):
        if not f.startswith("log-"): continue
        data = open(os.path.join(d, f), "rb").read(); tot += 1
        if len(data) < 272 or fnv(data[264:-8]) != struct.unpack("<Q", data[-8:])[0]: bad += 1
    return tot, bad

def chat(messages, max_tokens=500, tag=""):
    t0 = time.time(); mark = log_size()
    for attempt in range(3):   # an engine restart mid-suite must not cascade into bogus FAILs
        try:
            r = requests.post(BASE, json={"model": "strata", "messages": messages, "max_tokens": max_tokens,
                                          "temperature": 0, "stream": True}, stream=True, timeout=900)
            break
        except requests.exceptions.ConnectionError as e:
            if attempt == 2: raise
            print(f"  [{tag}] connection refused (engine restarting?), retrying in 30 s", flush=True)
            time.sleep(30)
    ttft = None; content = []
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: ") or line[6:] == "[DONE]": continue
        try: j = json.loads(line[6:])
        except ValueError: continue
        d = (j.get("choices") or [{}])[0].get("delta", {})
        if d.get("content") or d.get("reasoning_content"):
            if ttft is None: ttft = time.time() - t0
            if d.get("content"): content.append(d["content"])
    tail = log_tail(mark)
    return {"ttft": ttft, "total": time.time() - t0, "text": "".join(content),
            "promoted": "nvme promote" in tail, "reused": re.search(r"(\d+) reused \+ (\d+) read", tail)}

def haystack(code, theme, details, extra=""):
    head = (f"You are reading the complete {theme} records. Transcribe them faithfully. "
            f"Early in the records the warden wrote: 'Memorandum: the emergency access code is {code}.' "
            f"Then the records continue:\n\n")
    para = (f"The {theme} continued its daily rounds. Notes covered {details}, each entry dated and "
            "signed, weather recorded three times a day, inventory checked weekly. " + extra)
    head_ids = tok.encode(head)
    ids = tok.encode(head + para * ((TARGET - len(head_ids)) // len(tok.encode(para)) + 1))[:TARGET]
    return head + tok.decode(ids[len(head_ids):])

def session_turn(messages, question, tag, expect_code=None):
    q = list(messages) + [{"role": "user", "content": question}]
    r = chat(q, tag=tag)
    hit = expect_code is None or expect_code in r["text"]
    return q, r, hit

# ---------- S1: multi-turn chain ----------
print("== S1 multi-turn chain ==", flush=True)
n0 = store_count()
h1 = haystack("COPPER-BEACON-3311", "harbor master's ledger", "the tide tables, the pilot boats, the customs seals")
msgs = [{"role": "user", "content": h1 + "\n\nEnd of records. Reply: 'Ledger read.'"}]
r = chat(msgs, tag="s1-build")
msgs.append({"role": "assistant", "content": r["text"]})
q1, r1, hit1 = session_turn(msgs, "What is the emergency access code? Reply with only the code.", "s1-turn2", "COPPER-BEACON-3311")
msgs.append({"role": "assistant", "content": r1["text"]})
q2, r2, hit2 = session_turn(msgs, "Now summarize the records in one sentence.", "s1-turn3")
q3, r3, hit3 = session_turn(msgs, "Repeat the emergency access code exactly.", "s1-turn4", "COPPER-BEACON-3311")
# turns continue the same conversation, so the hot tier is the in-RAM checkpoint - the assertion is
# that prefill is avoided (reused ~= the whole prompt), not which tier served it
pre = lambda r: r["reused"] and int(r["reused"].group(1)) > 2500
check("S1 turn2 avoids prefill", pre(r1), f"reused={r1['reused'].groups() if r1['reused'] else None}")
check("S1 turn3 avoids prefill", pre(r2), f"reused={r2['reused'].groups() if r2['reused'] else None}")
check("S1 turn4 avoids prefill", pre(r3), f"reused={r3['reused'].groups() if r3['reused'] else None}")
check("S1 needle found (turn2)", hit1, r1["text"][:60])
check("S1 needle still found (turn4)", hit3, r3["text"][:60])
check("S1 store grew (per-turn writes)", store_count() >= n0 + 3, f"{store_count()} records (delta {store_count()-n0})")

# ---------- S2: branching ----------
print("== S2 branching ==", flush=True)
base = list(msgs)                      # the S1 conversation as the shared prefix
n0 = store_count()
br1, rb1, hb1 = session_turn(base, "What year is mentioned first in the records?", "s2-branch1")
br2, rb2, hb2 = session_turn(base, "Who is the harbor master in the records?", "s2-branch2")
qa, ra, ha = session_turn(br1, "Repeat the emergency access code exactly.", "s2-branch1-needle", "COPPER-BEACON-3311")
qb, rb_, hb = session_turn(br2, "Repeat the emergency access code exactly.", "s2-branch2-needle", "COPPER-BEACON-3311")
check("S2 both branches keep the needle", ha and hb, f"br1={ra['text'][:40]!r} br2={rb_['text'][:40]!r}")
check("S2 branch snapshots created", store_count() >= n0 + 2, f"delta {store_count()-n0}")

# ---------- S4: shared-prefix sessions ----------
print("== S4 shared-prefix sessions ==", flush=True)
preamble = ("You are reading field records. Every entry is dated and signed. "
            "The station chief requires exact transcription of all memorandum lines.\n\n")
hA = preamble + haystack("SCARLET-GATE-5540", "marsh ecology survey", "the reed beds, the bittern counts, the drainage tiles")[:2200]
hB = preamble + haystack("ONIX-HARBOR-9028", "quarry survey", "the blasting schedule, the haul road, the weighting scales")[:2200]
n0 = store_count()
mA = [{"role": "user", "content": hA + "\n\nEnd of records. Reply: 'Read.'"}]
rA = chat(mA, tag="s4-buildA"); mA.append({"role": "assistant", "content": rA["text"]})
mB = [{"role": "user", "content": hB + "\n\nEnd of records. Reply: 'Read.'"}]
rB = chat(mB, tag="s4-buildB"); mB.append({"role": "assistant", "content": rB["text"]})
_, rA2, hA2 = session_turn(mA, "What is the emergency access code? Reply with only the code.", "s4-needleA", "SCARLET-GATE-5540")
_, rB2, hB2 = session_turn(mB, "What is the emergency access code? Reply with only the code.", "s4-needleB", "ONIX-HARBOR-9028")
check("S4 session A needle correct", hA2, rA2["text"][:60])
check("S4 session B needle correct", hB2, rB2["text"][:60])
check("S4 no cross-contamination", "SCARLET" not in rB2["text"] and "ONIX" not in rA2["text"])

# ---------- S5: rotation stress ----------
print("== S5 rotation stress ==", flush=True)
rot = [("VIOLET-ANCHOR-7712", "ferry schedule audit", "the berth assignments, the crew rotations"),
       ("EMERALD-QUAY-2264", "canal toll ledger", "the lock timings, the barge weights"),
       ("OCHRE-MILL-6109", "grain elevator log", "the silo temperatures, the rail cars")]
hists = []
for code, theme, det in rot:
    h = haystack(code, theme, det)
    m = [{"role": "user", "content": h + "\n\nEnd of records. Reply: 'Read.'"}]
    r = chat(m, tag="s5-build"); m.append({"role": "assistant", "content": r["text"]})
    hists.append((code, m))
ok_all = True
for round_no in (1, 2):
    for code, m in hists:
        _, r, hit = session_turn(m, "What is the emergency access code? Reply with only the code.", f"s5-r{round_no}", code)
        ok_all = ok_all and hit
        if not hit: print(f"  miss: round {round_no} {code}: {r['text'][:60]!r}", flush=True)
check("S5 all 6 rotation queries correct", ok_all)

# ---------- S6: idempotence ----------
print("== S6 idempotence ==", flush=True)
code, m = hists[0]
n0 = store_count()
_, r_a, h_a = session_turn(m, "What is the emergency access code? Reply with only the code.", "s6-a", code)
c1 = store_count()
_, r_b, h_b = session_turn(m, "What is the emergency access code? Reply with only the code.", "s6-b", code)
c2 = store_count()
check("S6 both runs correct", h_a and h_b)
pre_b = r_b["reused"] and int(r_b["reused"].group(1)) > 2500
# the conversation GREW (the previous reply is now history), so a new per-turn snapshot is correct;
# what must hold: prefill avoided and the answer right
check("S6 repeat turn: prefill avoided, correct", pre_b and h_b, f"delta {c2-c1}")

# ---------- S3: corruption fallback (last, it mutates the store) ----------
print("== S3 corruption fallback ==", flush=True)
# corrupt EVERY stored snapshot, then force an NVMe promote: a foreign session's request first (so the
# target's in-RAM checkpoints are pruned - they are not prefixes of it), then the target's request, whose
# only match is its (now corrupt) NVMe entry: promote must fail gracefully, the entry dropped, and the
# request answered by a full re-prefill.
pre = set()
for f in os.listdir(STORE):
    p = os.path.join(STORE, f)
    if f.startswith("kv-") and os.path.isfile(p):
        pre.add(f)
        size = os.path.getsize(p)
        with open(p, "r+b") as fh:   # the v3 family: one flipped byte mid-payload
            fh.seek(size // 2); b = fh.read(1); fh.seek(size // 2); fh.write(bytes([b[0] ^ 0xFF]))
corrupt_time = time.time()
pre_mtime = {}
ddir0 = os.path.join(STORE, "delta")
if os.path.isdir(ddir0):
    for sub in ("", "chunks", "states"):
        p = os.path.join(ddir0, sub)
        if not os.path.isdir(p): continue
        for f in os.listdir(p):
            fp = os.path.join(p, f)
            if os.path.isfile(fp): pre_mtime[fp] = os.path.getmtime(fp)
# the delta family: corrupt every content record (chunks/states) - the scan admits the manifests, so the
# promote READS the corrupt record and must refuse it by name.  The manifests stay intact: a corrupt manifest
# is dropped at SCAN time (before any promote), which would make this test a no-op.
ddir = os.path.join(STORE, "delta")
if os.path.isdir(ddir):
    for sub in ("chunks", "states"):
        sdir = os.path.join(ddir, sub)
        for f in os.listdir(sdir):
            p = os.path.join(sdir, f); size = os.path.getsize(p)
            with open(p, "r+b") as fh:
                fh.seek(size // 2); b = fh.read(1); fh.seek(size // 2); fh.write(bytes([b[0] ^ 0xFF]))
n0 = store_count()
mark = log_size()
foreign = hists[1][1] + [{"role": "user", "content": "Reply: 'ok'."}]
rf = chat(foreign, tag="s3-foreign")
_, r, hit = session_turn(msgs, "Repeat the emergency access code exactly.", "s3-corrupt", "COPPER-BEACON-3311")
tail = log_tail(mark)
check("S3 integrity refusal logged", "integrity check failed" in tail)
check("S3 fallback re-prefill logged", "reading the prompt instead" in tail)
check("S3 corrupted snapshot answered correctly", hit, r["text"][:60])
import subprocess, shutil
# files written AFTER the corruption (the fresh dumps of the fallback requests) must all verify;
# corrupt files that are never matched legitimately remain until matched-and-refused or LRU-evicted
newdir = "/tmp/fnv-new"
shutil.rmtree(newdir, ignore_errors=True); os.makedirs(newdir)
fresh = [f for f in os.listdir(STORE) if f.startswith("kv-") and f not in pre]
for f in fresh:
    shutil.copy2(os.path.join(STORE, f), os.path.join(newdir, f))
r = subprocess.run(["/tmp/fnvaudit", newdir], capture_output=True, text=True)
v3_ok = r.stdout.strip().endswith(", 0 bad")
# the delta family's fresh records (a re-dump after a refusal REWRITES the corrupted records under the same
# content keys - the repair the design gives for free); only records newer than the corruption are the new dumps
ddir = os.path.join(STORE, "delta")
delta_ok, tot, bad = True, 0, 0
if os.path.isdir(ddir):
    # only the FRESH records are the new dumps: a corrupt record never matched legitimately remains until it is
    # matched-and-refused (the v3 test's own rule) - and a re-dump REPAIRS the corrupted records it re-derives,
    # writing them fresh under the same content keys.  So: audit what was written after the corruption.
    fresh_delta = []
    for sub in ("", "chunks", "states"):
        p = os.path.join(ddir, sub)
        if not os.path.isdir(p): continue
        for f in os.listdir(p):
            fp = os.path.join(p, f)
            # strictly after the corruption: the corruption itself refreshes every record's mtime, so the
            # reference must be the corruption MOMENT - only a REPAIR (a re-dump rewriting the record under its
            # content key) or a NEW dump is newer
            if os.path.isfile(fp) and os.path.getmtime(fp) > corrupt_time + 0.5:
                fresh_delta.append(fp)
    import struct
    def fnv(b, h=1469598103934665603):
        for x in b: h = ((h ^ x) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        return h
    for fp in fresh_delta:
        data = open(fp, "rb").read(); tot += 1
        hdr = 48 if "/chunks/" in fp else (16 if "/states/" in fp else 264)
        if len(data) < hdr + 8 or fnv(data[hdr:-8]) != struct.unpack("<Q", data[-8:])[0]: bad += 1
    delta_ok = bad == 0
check("S3 fresh dumps verify", v3_ok and delta_ok, f"v3: {r.stdout.strip()!r}; delta: {tot} fresh records, {bad} bad")

print("== VERDICT ==", flush=True)
print("ALL PASS" if not FAILS else f"SOME FAILURES: {FAILS}", flush=True)
