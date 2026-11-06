"""serve/fake_mask_engine.py - a stand-in for `strata --serve` that speaks the token-mask protocol, for tests.

    python serve/fake_mask_engine.py --serve [--no-mask] [--max-context N]

The byte tokenizer's ids (serve.server.ByteTokenizer: one id per byte, <|im_end|> = 257).  Each GEN answers with the
script in the file $FAKE_SCRIPT_FILE, read again at every GEN (or the text in $FAKE_SCRIPT; the end-of-turn token
is added), one token per window.  With `mask=1` on GEN it prints
`MQ` before every window and reads the answer: under `MK <base64>` it drops the script tokens the mask forbids (the
way serve/constrain.Constraint.mock_replay does, written again here so the two are independent), under `MF cut=...`
it honours nothing more than one token per window.  --no-mask: an older engine (no token_mask=1 in INFO; mask=1 is
then an unknown key it skips).  Every MQ answer is appended to $FAKE_LOG when set.
"""
import base64
import os
import sys

IM_END, ENDOFTEXT = 257, 258


def allowed(words: bytes, t: int) -> bool:
    w = t >> 5
    return 4 * w + 4 <= len(words) and (int.from_bytes(words[4 * w:4 * w + 4], "little") >> (t & 31)) & 1 == 1


def main() -> int:
    mask_ok = "--no-mask" not in sys.argv
    ctx = int(sys.argv[sys.argv.index("--max-context") + 1]) if "--max-context" in sys.argv else 4096
    print(f"INFO context={ctx} kv=fp16{' token_mask=1' if mask_ok else ''} engine=fake", flush=True)
    print(f"READY {ctx} stop", flush=True)
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
