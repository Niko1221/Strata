#!/usr/bin/env python3
"""Park/restore timing and engine-RSS measurement for --conversation-cache-storage ram vs disk.

Starts ONE private engine with the given storage mode, builds a long conversation (A), parks it by switching to a
short one (B), restores it, and samples the engine's VmRSS/VmHWM around each step.  The park/restore times come from
the engine's own "conversation cache: parked/restored N tokens in X ms" log lines (the same lines the server's
Monitor counts).  Decode speed is measured on the restored conversation.

  python3 tools/conversation_cache_disk_measure.py --exe build/strata --config strata-wsl.json \
      --storage disk --cache-dir /root/park-cache --repeat 400
"""
import argparse, json, os, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batch_test import Engine, tokenizer  # noqa: E402

DOC = ("The history of the printing press begins with movable type in East Asia and continues with Gutenberg's "
       "press in Mainz around 1450, which spread across Europe within decades, lowered the cost of books, changed "
       "how scholars, merchants and churches shared knowledge, and fed the Reformation and the scientific "
       "revolution. ")

EVENT = re.compile(r"conversation cache: (parked|restored) (\d+) tokens.*?in ([0-9.]+) ms")
BYTES = re.compile(r"bytes=(\d+)(?:.*?snapshot_bytes=(\d+))?")


def chat(user):
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def gen(eng, out, ids, n):
    eng.send(f"GEN {n} " + ",".join(map(str, ids)))
    got, done, reused = [], None, None
    t0 = time.time()
    for line in out:
        if line.startswith("T "):
            got.append(int(line.split()[1]))
        elif line.startswith("DONE") or line.startswith("ERR"):
            done = line
            f = line.split()
            if len(f) > 8:
                try:
                    reused = int(f[8])
                except ValueError:
                    pass
            break
    return got, done, time.time() - t0, reused


def rss(pid):
    try:
        status = dict(re.findall(r"(VmRSS|VmHWM):\s+(\d+) kB", open(f"/proc/{pid}/status").read()))
        return int(status.get("VmRSS", 0)) * 1024, int(status.get("VmHWM", 0)) * 1024
    except OSError:
        return 0, 0


def dir_stats(path):
    files = [e for e in os.scandir(path) if e.is_file()] if os.path.isdir(path) else []
    return len(files), sum(e.stat().st_size for e in files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--storage", choices=("ram", "disk"), default="disk")
    ap.add_argument("--cache-dir", default="/tmp/strata-measure-cache")
    ap.add_argument("--repeat", type=int, default=400, help="copies of the paragraph in conversation A (~87 tokens each)")
    ap.add_argument("--max-new", type=int, default=16)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    tok = tokenizer(cfg["tokenizer"])
    extra = []
    if a.storage == "disk":
        extra = ["--conversation-cache-storage", "disk", "--conversation-cache-dir", a.cache_dir]
        os.makedirs(a.cache_dir, exist_ok=True)
    t0 = time.time()
    eng = Engine(a.exe, cfg, 0, {"STRATA_IQ_MT_MIN": "1"}, extra)
    start_s = time.time() - t0
    out = eng.lines()
    log0 = os.path.getsize(eng.log_path) if hasattr(eng, "log_path") else 0

    def tail_events():
        with open(eng.log_path) as f:
            f.seek(log0)
            events = []
            for line in f:
                m = EVENT.search(line)
                if not m:
                    continue
                b = BYTES.search(line)
                events.append({"what": m.group(1), "tokens": int(m.group(2)), "ms": float(m.group(3)),
                               "bytes": int(b.group(1)) if b else None,
                               "snapshot_bytes": int(b.group(2)) if b and b.group(2) else None})
            return events

    result = {"storage": a.storage, "start_s": round(start_s, 1), "steps": {}}

    ids_a = tok.encode(chat(DOC * a.repeat + "\nSummarize this text in five sentences."), parse_special=True)
    ids_b = tok.encode(chat("Write a short poem about the sea."), parse_special=True)
    step = lambda name: result["steps"].setdefault(name, {})

    ans_a, done, dt, reused0 = gen(eng, out, ids_a, a.max_new)
    step("A first read")["s"] = round(dt, 2)
    step("A first read")["prompt_tokens"] = len(ids_a)
    follow_a = ids_a + ans_a + tok.encode("<|im_end|>\n", parse_special=True) + \
        tok.encode(chat("Give three keywords for that text."), parse_special=True)

    r1, h1 = rss(eng.p.pid)
    step("A first read")["rss_mib"] = round(r1 / 2**20, 1)
    step("A first read")["hwm_mib"] = round(h1 / 2**20, 1)

    _, done_b, dt, _ = gen(eng, out, ids_b, 8)   # parks A
    step("B (parks A)")["s"] = round(dt, 2)
    ev = tail_events()
    step("B (parks A)")["events"] = ev
    r2, h2 = rss(eng.p.pid)
    step("B (parks A)")["rss_mib"] = round(r2 / 2**20, 1)
    step("B (parks A)")["hwm_mib"] = round(h2 / 2**20, 1)
    step("B (parks A)")["rss_delta_mib"] = round((r2 - r1) / 2**20, 1)
    if a.storage == "disk":
        step("B (parks A)")["cache_files"] = dir_stats(a.cache_dir)

    got, done, dt, reused_r = gen(eng, out, follow_a, a.max_new)   # restores A
    ev = tail_events()
    step("A restored")["s"] = round(dt, 2)
    step("A restored")["events"] = ev
    step("A restored")["follow_tokens"] = len(got)
    step("A restored")["reused_tokens"] = reused_r
    r3, h3 = rss(eng.p.pid)
    step("A restored")["rss_mib"] = round(r3 / 2**20, 1)
    step("A restored")["hwm_mib"] = round(h3 / 2**20, 1)
    step("A restored")["rss_delta_mib"] = round((r3 - r2) / 2**20, 1)
    if a.storage == "disk":
        step("A restored")["cache_files"] = dir_stats(a.cache_dir)

    # decode speed on the restored conversation: continuation, only new tokens timed
    cont = follow_a + got + tok.encode("<|im_end|>\n", parse_special=True) + \
        tok.encode(chat("Now count from one to twenty."), parse_special=True)
    dec, _, dtd, _ = gen(eng, out, cont, 128)
    step("decode 128")["s"] = round(dtd, 2)
    step("decode 128")["tok_s"] = round(len(dec) / dtd, 1) if dtd else 0

    eng.send("QUIT")
    eng.p.wait(timeout=180)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
