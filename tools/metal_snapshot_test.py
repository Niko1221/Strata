#!/usr/bin/env python3
"""Metal snapshots test: on Metal the conversation snapshots (parking, session files) carry the
decode window's own K/V, so the conversation cache and SAVE / RESTORE are on, as on CUDA.  Through the --serve
protocol, one engine start (before, the Metal engine never parked and SAVE failed):

  1. started with --conversation-cache-mib: no start-up line turns the cache off on Metal, and INFO reports
     conversation_cache_mib=1024;
  2. a second conversation parks the first (`conversation cache: parked` in the log, no `skip parking`);
  3. SAVE <path> answers `SAVED <tokens> <bytes> <ms>` and writes the file;
  4. RESTORE of a file that is not a session file answers `SERR invalid 0 ...` and leaves the engine in step;
  5. RESTORE of the saved file answers `RESTORED <tokens> ...` with the saved token count;
  6. the engine is still in step: the next GEN decodes.

  python3 tools/metal_snapshot_test.py --exe build-metal/strata --config strata-<model>.json [--extra "..."]

Exit 0 when every check passes, 1 otherwise.  Exactness (the restored follow-up decodes the live tokens) is
tools/parking_test.py's and tools/session_restart_test.py's.
"""
import argparse, json, os, subprocess, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batch_test import tokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--extra", default="--max-context 8192", help="more engine arguments in one string")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    tok = tokenizer(cfg["tokenizer"])
    args = [a.exe, "--serve", *cfg["args"], *a.extra.split(), "--conversation-cache-mib", "1024"]
    log_path = os.path.join(tempfile.gettempdir(), "metal_snapshot_test_engine.log")
    with open(log_path, "w") as log:
        p = subprocess.Popen(args, cwd=cfg.get("cwd"), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                             text=True, bufsize=1, env=dict(os.environ, **(cfg.get("env") or {})))
    head = []
    for line in p.stdout:
        head.append(line.rstrip("\n"))
        if line.startswith("READY"):
            break
    else:
        raise SystemExit(f"the engine ended before READY - see {log_path}")

    def send(cmd):
        p.stdin.write(cmd + "\n")
        p.stdin.flush()

    def answer(prefixes):
        for line in p.stdout:
            line = line.rstrip("\n")
            if line.startswith(prefixes):
                return line
        return "(the engine ended)"

    def gen(text, n=16):
        ids = tok.encode(f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                         parse_special=True)
        send(f"GEN {n} " + ",".join(map(str, ids)))
        return answer(("DONE", "ERR"))

    fails = []

    def check(name, ok, seen):
        print(f"{'PASS' if ok else 'FAIL'} {name}: {seen}", flush=True)
        if not ok:
            fails.append(name)

    info = next((l for l in head if l.startswith("INFO")), "")
    mib = next((f.split("=", 1)[1] for f in info.split() if f.startswith("conversation_cache_mib=")), "?")
    log_text = Path(log_path).read_text()
    off = [l for l in log_text.splitlines() if "conversation cache" in l and "Metal" in l]
    check("no start-up line turning the conversation cache off on Metal", not off, off[:1] or "none")
    check("INFO conversation_cache_mib=1024", mib == "1024", f"conversation_cache_mib={mib}")

    first = gen("Name three primary colors.")
    gen("Write one sentence about the sea.")   # a second conversation parks the first
    log_text = Path(log_path).read_text()
    parked = [l for l in log_text.splitlines() if "conversation cache: parked" in l]
    skipped = [l for l in log_text.splitlines() if "skip parking" in l]
    check("the second conversation parked the first", bool(parked) and not skipped, (skipped or parked)[:1] or "none")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "s.strata-session")
        send(f"SAVE {path}")
        saved = answer(("SAVED", "SERR", "ERR", "FATAL"))
        check("SAVE answers SAVED", saved.startswith("SAVED "), saved)
        check("SAVE wrote the file", os.path.exists(path), "exists" if os.path.exists(path) else "no file")
        bad = os.path.join(d, "bad.strata-session")
        Path(bad).write_bytes(b"not a session")
        send(f"RESTORE {bad}")
        refused = answer(("RESTORED", "SERR", "ERR", "FATAL"))
        check("RESTORE of a non-session file answers SERR invalid 0", refused.startswith("SERR invalid 0 "), refused)
        send(f"RESTORE {path}")
        restored = answer(("RESTORED", "SERR", "ERR", "FATAL"))
        n_saved = saved.split()[1] if saved.startswith("SAVED ") else "?"
        check("RESTORE of the saved file answers RESTORED with its tokens",
              restored.startswith("RESTORED ") and restored.split()[1] == n_saved, restored)
    after = gen("Name three primary colors.")
    check("the engine is still in step (GEN after the restores)", after.startswith("DONE"), after)
    send("QUIT")
    p.wait(timeout=180)
    print(f"first request: {first}\nengine log: {log_path}")
    print("ALL PASS" if not fails else f"{len(fails)} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
