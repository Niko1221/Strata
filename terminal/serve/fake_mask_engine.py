"""serve/fake_mask_engine.py - a stand-in for `strata --serve` that speaks the token-mask protocol, for tests.

    python serve/fake_mask_engine.py --serve [--no-mask] [--max-context N [--batch N] [--gen-delay S] [--slot-delay S]]

The byte tokenizer's ids (serve.server.ByteTokenizer: one id per byte, <|im_end|> = 257).  Each GEN answers with the
script in the file $FAKE_SCRIPT_FILE, read again at every GEN (or the text in $FAKE_SCRIPT; the end-of-turn token
is added), one token per window.  With `mask=1` on GEN it prints
`MQ` before every window and reads the answer: under `MK <base64>` it drops the script tokens the mask forbids (the
way serve/constrain.Constraint.mock_replay does, written again here so the two are independent), under `MF cut=...`
it honours nothing more than one token per window.  --no-mask: an older engine (no token_mask=1 in INFO; mask=1 is
then an unknown key it skips).  Every MQ answer is appended to $FAKE_LOG when set.
The --batch N loop: the engine's own shape - a reader thread takes STOP / BSTOP / BYIELD aside, the main
loop runs a batch window every slot_delay seconds while slots are active, and a solo GEN answers its MQ
lines.  A masked GEN is logged as `MASKGEN N` (N: the active slots it saw, notes/pr3800-batch-spec.md 3-2:
the invariant is N == 0) and `GENEND` at its DONE; a BGEN that carries a mask key is refused with ERR
(spec 3-1: the batch admission read never answers the MQ such a window would ask for).
"""
import base64
import os
import queue
import sys
import threading
import time

IM_END, ENDOFTEXT = 257, 258
cv = threading.Condition()


def allowed(words: bytes, t: int) -> bool:
    w = t >> 5
    return 4 * w + 4 <= len(words) and (int.from_bytes(words[4 * w:4 * w + 4], "little") >> (t & 31)) & 1 == 1


def is_ge2(x: int) -> bool:
    return x >= 2


def batched_main(stdin, mask_ok: bool, ctx: int, slots: int, gen_delay: float, slot_delay: float) -> int:
    """The --batch N loop: the engine's own shape - a reader thread takes STOP / BSTOP / BYIELD aside, the main
    loop runs a batch window every slot_delay seconds while slots are active, and a solo GEN answers its MQ
    lines.  A masked GEN is logged as `MASKGEN N` (N: the active slots it saw, notes/pr3800-batch-spec.md 3-2:
    the invariant is N == 0) and `GENEND` at its DONE; a BGEN that carries a mask key is refused with ERR
    (spec 3-1: the batch admission read never answers the MQ such a window would ask for)."""
    log = os.environ.get("FAKE_LOG")
    active = {}                       # slot -> [tokens left, max_new, produced]
    stopped = set()                   # BSTOPped slots: they end "cancel"
    masked_solo = [False]             # a masked solo leg is running (the invariant's other half)
    lines: "queue.Queue" = queue.Queue()

    def reader():
        for l in sys.stdin:
            l = l.strip()
            if l == "STOP":
                lines.put("STOP")
            else:
                lines.put(l)
        lines.put(None)

    threading.Thread(target=reader, daemon=True).start()

    def note(s):
        if log:
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(s + "\n")

    def get_line(timeout=None):
        try:
            return lines.get(timeout=timeout)
        except queue.Empty:
            return ""

    def window():                     # one batch window: every active slot one token
        for b in sorted(list(active)):
            left, max_new, produced = active.pop(b)
            t = left.pop(0) if left else IM_END
            produced += 1
            print(f"BT {b} {t}", flush=True)
            if t == IM_END or produced >= max_new:
                fin = "stop" if t == IM_END else "cancel" if b in stopped else "length"
                stopped.discard(b)
                print(f"BDONE {b} {produced} {fin} 1.0", flush=True)
            else:
                active[b] = [left, max_new, produced]

    while True:
        line = get_line(slot_delay if active else None)
        if line is None or line == "QUIT":
            return 0
        if line == "STOP":
            continue                  # a solo leg answers its own STOP (the MQ wait below)
        if line.startswith("BYIELD "):
            continue                  # too late: the read it was for has ended
        if line.startswith("BSTOP "):
            b = int(line.split()[1])
            if b in active:
                active[b][1] = 0
                stopped.add(b)
            continue
        if line.startswith("BGEN "):
            f = line.split()
            slot, max_new = int(f[1]), int(f[2])
            keys = {k: v for k, _, v in (x.partition("=") for x in f[3:-1]) if _}
            note(f"BGEN {slot} mask={keys.get('mask') or '-'}")
            if mask_ok and keys.get("mask") == "1":
                print("ERR BGEN: mask is solo-only", flush=True)
                continue
            text = open(os.environ["FAKE_SCRIPT_FILE"], encoding="utf-8").read() if os.environ.get("FAKE_SCRIPT_FILE") \
                else (os.environ.get("FAKE_SCRIPT") or "")
            script = list(text.encode()) + [IM_END]
            ids = [int(x) for x in f[-1].split(",")]
            print("REUSED 0", flush=True)
            out = 0
            if max_new >= 1 and script:
                print(f"T {script[0]}", flush=True)
                out = 1
            fin = "stop" if out and script[0] == IM_END else ("length" if max_new >= 1 else "cancel")
            print(f"DONE {out} {len(ids)} 1.0 1.0 {fin} 0 0 0 0 0 0 0 0.0 {len(ids)} 0", flush=True)
            if fin == "length" and max_new >= 1:
                active[slot] = [script[1:], max_new, out]
                print(f"BADM {slot} 1", flush=True)
            else:
                print(f"BADM {slot} 0", flush=True)
            continue
        if line.startswith("GEN "):
            f = line.split()
            keys = {k: v for k, _, v in (x.partition("=") for x in f[2:-1]) if _}
            masked = mask_ok and keys.get("mask") == "1"
            if masked:
                masked_solo[0] = True
                note(f"MASKGEN {len(active)}")
            else:
                note(f"GEN {len(active)}")
            max_new = int(f[1])
            prompt = f[-1].split(",")
            text = open(os.environ["FAKE_SCRIPT_FILE"], encoding="utf-8").read() if os.environ.get("FAKE_SCRIPT_FILE") \
                else (os.environ.get("FAKE_SCRIPT") or "")
            script = list(text.encode()) + [IM_END]
            print("REUSED 0", flush=True)
            i, n, finish = 0, 0, "length"
            yield_me = b"YIELDME" in bytes(int(x) & 255 for x in map(int, prompt))
            if yield_me:
                # a prompt read that takes long: PP lines as the engine's chunked read gives them, the server's
                # give_way reserves a slot at one of them and sends BYIELD, and the read gives way there
                end_wait, gave, q = time.monotonic() + 12.0, None, 0
                total = len(prompt)
                while gave is None and time.monotonic() < end_wait:
                    q = min(q + 32, total - 1)
                    print(f"PP {q} {total} 1 1", flush=True)
                    held = get_line(0.3)
                    if held.startswith("BYIELD "):
                        gave = held.split()[1]
                if gave is not None:
                    note("YIELD " + gave)
                    print(f"YIELDED {gave} 0", flush=True)
                    print(f"DONE 0 {len(prompt)} 1.0 1.0 cancel 0 0 0 0 0 0 0 0.0 {len(prompt)} 0", flush=True)
                    note("GENEND")
                    continue
            while n < max_new:
                words = None
                if masked:
                    print("MQ", flush=True)
                    rep, stopped_now = get_line(), False
                    while rep == "STOP":            # the reader thread takes STOP aside, as the engine's does
                        stopped_now = True
                        rep = get_line()
                    note(rep[:2])
                    if rep.startswith("MK "):
                        words = base64.b64decode(rep[3:])
                    elif not rep.startswith("MF"):
                        print(f"ERR expected MK or MF after MQ, got: {rep[:40]}", flush=True)
                        break
                    if stopped_now:
                        finish = "cancel"
                        break
                if gen_delay:
                    time.sleep(gen_delay)
                t = None
                while i < len(script):
                    c = script[i]
                    i += 1
                    if words is None or allowed(words, c):
                        t = c
                        break
                if t is None:
                    if words is not None and allowed(words, IM_END):
                        t = IM_END
                    else:
                        break
                print(f"T {t}", flush=True)
                n += 1
                if t in (IM_END, ENDOFTEXT):
                    finish = "stop"
                    break
            print(f"DONE {n} {len(prompt)} 1.0 1.0 {finish} 0 0 0 0 0 0 0 0.0 {len(prompt)} 0", flush=True)
            masked_solo[0] = False
            note("GENEND")
            with cv:
                cv.notify_all()
            continue
        if line:
            note("UNK " + line[:40])
            print("ERR unknown " + line[:20], flush=True)
            continue
        if active:
            window()
            if slot_delay:
                time.sleep(slot_delay)


def main() -> int:
    mask_ok = "--no-mask" not in sys.argv
    ctx = int(sys.argv[sys.argv.index("--max-context") + 1]) if "--max-context" in sys.argv else 4096
    slots = int(sys.argv[sys.argv.index("--batch") + 1]) if "--batch" in sys.argv else 0
    gen_delay = float(sys.argv[sys.argv.index("--gen-delay") + 1]) if "--gen-delay" in sys.argv else 0.0
    slot_delay = float(sys.argv[sys.argv.index("--slot-delay") + 1]) if "--slot-delay" in sys.argv else 0.05
    info = f"INFO context={ctx} kv=fp16{' token_mask=1' if mask_ok else ''}"
    if is_ge2(slots):
        info += f" batch_slots={slots}"
    print(info + " engine=fake", flush=True)
    print(f"READY {ctx} stop", flush=True)
    if is_ge2(slots):
        return batched_main(sys.stdin, mask_ok, ctx, slots, gen_delay, slot_delay)
    log = os.environ.get("FAKE_LOG")
    stdin = sys.stdin
    for line in stdin:
        line = line.rstrip("\r\n")
        if line == "QUIT":
            return 0
        if line == "STOP" or not line.startswith("GEN "):
            continue
        f = line.split()
        max_new = int(f[1])
        keys = {k: v for k, _, v in (x.partition("=") for x in f[2:-1]) if _}
        masked = mask_ok and keys.get("mask") == "1"
        prompt = f[-1].split(",")
        path = os.environ.get("FAKE_SCRIPT_FILE")
        text = open(path, encoding="utf-8").read() if path else os.environ.get("FAKE_SCRIPT") or ""
        script = list(text.encode()) + [IM_END]
        print(f"REUSED 0", flush=True)
        i, n, finish = 0, 0, "length"
        while n < max_new:
            words = None
            if masked:
                print("MQ", flush=True)
                rep, stopped = stdin.readline().rstrip("\r\n"), False
                while rep == "STOP":                  # the engine's reader thread takes STOP aside (stop_req)
                    stopped = True
                    rep = stdin.readline().rstrip("\r\n")
                if log:
                    with open(log, "a", encoding="utf-8") as fh:
                        fh.write(rep[:2] + "\n")
                if rep.startswith("MK "):
                    words = base64.b64decode(rep[3:])
                elif not rep.startswith("MF"):
                    print(f"ERR expected MK or MF after MQ, got: {rep[:40]}", flush=True)
                    break
                if stopped:
                    finish = "cancel"
                    break
            t = None
            while i < len(script):
                c = script[i]
                i += 1
                if words is None or allowed(words, c):
                    t = c
                    break
            if t is None:
                if words is not None and (allowed(words, IM_END)):
                    t = IM_END
                else:
                    break
            print(f"T {t}", flush=True)
            n += 1
            if t in (IM_END, ENDOFTEXT):
                finish = "stop"
                break
        print(f"DONE {n} {len(prompt)} 1.0 1.0 {finish} 0 0 0 0 0 0 0 0.0 {len(prompt)} 0", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
