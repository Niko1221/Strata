#!/usr/bin/env python3
"""short_prompt_check.py <strata-dsv41> <pack> <ids file (comma separated token ids)> [engine args...]

A short prompt (40 tokens: the windows path, below the chunked pass) with the expert cache on, default sampler, twice, then a
greedy one.  Fails (exit 1) on any ERR line, an engine death, or an engine stderr line with 'illegal memory access', 'router picked'
or 'the move into VRAM failed'.  Found on HIP (p620): a swap copy out of a slot in the lent tail region read past the allocation
(the RAM store's put used vram_base + slot * unit instead of sat()), which only the windows path's swaps reached.
"""
import os, queue, subprocess, sys, tempfile, threading

exe, pack, idsf = sys.argv[1:4]
rest = sys.argv[4:]
ids = [int(x) for x in open(idsf).read().strip().split(",")][:40]
log = tempfile.NamedTemporaryFile("w+", suffix=".engine.log", delete=False)
p = subprocess.Popen([exe, "--serve", "--pack", pack, "--max-context", "16384", "--eos", "1", "--vision", "off", "--dspark", "off"] + rest,
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
q = queue.Queue()
threading.Thread(target=lambda: [q.put(l.rstrip("\n")) for l in p.stdout] + [q.put(None)], daemon=True).start()
while True:
    l = q.get(timeout=3600)
    if l is None:
        sys.exit("engine ended before READY")
    if l.startswith("READY"):
        break
bad = 0
for k, keys in enumerate(("", "", "temperature=0")):
    p.stdin.write(f"GEN 16 {keys} {','.join(map(str, ids))}\n")
    p.stdin.flush()
    n = 0
    while True:
        l = q.get(timeout=900)
        if l is None:
            print("engine died"); bad += 1; break
        if l.startswith("T "):
            n += 1
        if l.startswith("ERR"):
            print("request", k + 1, l[:120]); bad += 1; break
        if l.startswith("DONE"):
            print(f"request {k + 1}: {n} tokens | {l[:80]}")
            bad += n != 16
            break
    if bad:
        break
p.stdin.write("QUIT\n"); p.stdin.flush()
try:
    p.wait(timeout=120)
except Exception:
    p.kill()
log.flush(); log.seek(0)
for line in log:
    if any(s in line for s in ("illegal memory access", "router picked", "the move into VRAM failed")):
        print("engine log:", line.strip()[:160]); bad += 1
os.unlink(log.name)
print("short prompt check:", "FAILED" if bad else "ok")
sys.exit(1 if bad else 0)
