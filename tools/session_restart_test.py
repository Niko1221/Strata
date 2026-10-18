#!/usr/bin/env python3
"""Session file across an engine restart: a follow-up to conversation A must decode the same tokens whether A's state
was still live (first engine: A, SAVE, the follow-up) or came back from the file in a new engine process (second
engine: RESTORE, the follow-up), and the restored engine must resume from the file (reused > 0), not read A again.
Greedy, through the --serve protocol (SAVE <path> / RESTORE <path>).

  python3 tools/session_restart_test.py --exe "$PWD/build-metal/strata" --config strata-<model>.json \
      [--repeat 60] [--extra "--max-context 65536"]

Prints the file size, the save and restore times and the follow-up's prompt reuse.  Exit 0 when the follow-ups are
identical and the second engine reused the restored state, 1 otherwise.

--restore-exe ENGINE --expect-refused: the file is written by --exe and restored by another build whose session
layout differs (on a Mac: the decode window's K/V and running-state layout, part of the file's config fingerprint).
Then the restore must be refused cleanly (SERR, the engine keeps serving) and the follow-up is read from its prompt
(reused 0; its tokens may differ from the live follow-up's, as any re-read's can on a Mac).
"""
import argparse, json, os, sys, tempfile, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batch_test import Engine, tokenizer  # noqa: E402
from parking_test import DOC, chat, gen  # noqa: E402


def answer(out, prefixes):
    for line in out:
        if line.startswith(prefixes):
            return line
    return "(the engine ended)"


def follow_up(tok, pa, ans):
    end = tok.encode("<|im_end|>\n", parse_special=True)
    return pa + ans + ([] if ans and ans[-1] in end else end) + \
        tok.encode(chat("Now give three keywords for that text, with one sentence each."), parse_special=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--repeat", type=int, default=60, help="copies of the paragraph in conversation A")
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--extra", default="")
    ap.add_argument("--keep", default="", help="save the session file here (kept) instead of a temporary folder")
    ap.add_argument("--restore-exe", default="", help="the engine that restores the file (default: --exe)")
    ap.add_argument("--expect-refused", action="store_true",
                    help="the restoring engine must refuse the file (SERR) and still answer the follow-up")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    tok = tokenizer(cfg["tokenizer"])
    env = {"STRATA_IQ_MT_MIN": "1"}
    pa = tok.encode(chat(DOC * a.repeat + "\nSummarize this text in five sentences."), parse_special=True)
    tmp = tempfile.TemporaryDirectory()
    path = a.keep or os.path.join(tmp.name, "a.strata-session")

    eng = Engine(a.exe, cfg, 0, env, a.extra.split())   # 1: A, SAVE, the live follow-up
    out = eng.lines()
    ans, done_a = gen(eng, out, pa, a.max_new)
    eng.send(f"SAVE {path}")
    saved = answer(out, ("SAVED", "SERR", "ERR", "FATAL"))
    follow = follow_up(tok, pa, ans)
    live, done_live = gen(eng, out, follow, a.max_new)
    eng.send("QUIT")
    eng.p.wait(timeout=180)

    restored, got, done_got = "(not run)", [], ""
    if saved.startswith("SAVED"):
        eng = Engine(os.path.abspath(a.restore_exe or a.exe), cfg, 0, env, a.extra.split())   # 2: a new process, RESTORE
        out = eng.lines()
        t0 = time.time()
        eng.send(f"RESTORE {path}")
        restored = answer(out, ("RESTORED", "SERR", "ERR", "FATAL"))
        print(f"restore round trip {time.time() - t0:.2f} s", flush=True)
        got, done_got = gen(eng, out, follow, a.max_new)
        eng.send("QUIT")
        eng.p.wait(timeout=180)

    size = os.path.getsize(path) if os.path.exists(path) else 0
    f = done_got.split()
    reused = int(f[8]) if len(f) > 8 and f[8].isdigit() else 0
    fl = done_live.split()
    print(f"conversation A: {len(pa)} prompt + {len(ans)} generated tokens; follow-up {len(follow)} tokens")
    print(f"save: {saved}\nrestore: {restored}\nfile: {size} bytes ({size / max(1, len(pa) + len(ans)):.0f} B/token)")
    print(f"live follow-up reused {fl[8] if len(fl) > 8 else '?'}; restored follow-up reused {reused}")
    first = next((k for k in range(min(len(live), len(got))) if live[k] != got[k]), None)
    same = bool(live) and live == got
    print("follow-up tokens:", "IDENTICAL" if same else f"DIFFER at {first}", f"({len(live)} / {len(got)})")
    print("   ", repr(tok.decode(got)[:200]))
    if a.expect_refused:   # refused cleanly: an SERR line, the engine still serves the follow-up (from its prompt)
        ok = saved.startswith("SAVED") and restored.startswith("SERR") and len(got) > 0 and reused == 0
    else:
        ok = saved.startswith("SAVED") and restored.startswith("RESTORED") and same and reused > 0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
