#!/usr/bin/env python3
"""30-minute soak test for the native Vulkan engine.

Drives the engine over its line protocol with repeated GEN requests and watches
for the failure modes from issue #515: a xe reset, a NaN/token-0 answer, or a
runaway VSTAT. The engine is started once and reused, so the 21 GB GGUF is
opened once. No HTTP server is involved.

    python3 engine/vulkan/soak.py --minutes 30 --tokens 16

The engine emits ``ERR <reason>`` and then ``DONE`` on a bad request. Both must
be consumed before the next GEN, or the responses shift by one line. A per-line
watchdog turns a hung engine into a reported failure instead of a hang. With
``--lp`` (on by default) each token also carries a top-1 logprob; the string
``nan`` or ``inf`` there is reported as a bad token even when greedy sampling
picked a plausible id.
"""
import argparse
import os
import re
import select
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = Path("/home/g/Code/llama.cpp/models/Qwen3.8-35B-A3B-Q4_K_M.gguf")
PROMPTS = [
    [3710, 369, 220, 17, 10, 17, 30],          # What is 2+2?
    [9707, 11, 12092, 30],                      # Hello, world!
    [14113, 242, 19438, 428, 11, 30],           # The capital of France is
    [402, 1237, 4998, 242, 30],                 # 1, 2, 3, 4, 5,
]
NONFINITE = re.compile(r"\b(nan|inf|-inf)\b", re.IGNORECASE)


def _readline(proc, timeout):
    """Read one stdout line, or raise TimeoutError after `timeout` seconds."""
    if not select.select([proc.stdout], [], [], timeout)[0]:
        raise TimeoutError("no engine output for %.0f s" % timeout)
    return proc.stdout.readline()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--exe", default=str(HERE / "engine" / "strata-vulkan-engine"))
    ap.add_argument("--log", default="/tmp/opencode/strata-soak.log")
    ap.add_argument("--lp", type=int, default=1, help="top-K logprob lines to scan for NaN/inf (0 disables)")
    ap.add_argument("--timeout", type=float, default=90, help="seconds to wait for one line of engine output")
    a = ap.parse_args()
    cmd = [a.exe, "--model", a.model, "--ctx", "4096",
           "--native-shaders", str(HERE / "native" / "shaders"),
           "--expert-shaders", str(HERE / "shaders")]
    if a.lp:
        cmd += ["--lp", str(a.lp)]
    log = open(a.log, "a", buffering=1)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=log, text=True, bufsize=1)
    deadline = time.monotonic() + a.minutes * 60
    requests = bad = 0
    first = None
    try:
        while time.monotonic() < deadline:
            ids = PROMPTS[requests % len(PROMPTS)]
            proc.stdin.write(f"GEN {a.tokens} temperature=0 {','.join(map(str, ids))}\n")
            proc.stdin.write("VSTAT\n")
            proc.stdin.flush()
            tokens, stats, errors, finish = [], "", [], ""
            while True:
                line = _readline(proc, a.timeout)
                if not line:
                    raise RuntimeError("engine exited unexpectedly")
                if line.startswith("T "):
                    tokens.append(int(line.split()[1]))
                elif line.startswith("LP "):
                    if NONFINITE.search(line):
                        errors.append("nonfinite logprob")
                elif line.startswith("VSTAT"):
                    stats = line.strip()
                elif line.startswith("ERR "):
                    errors.append(line.strip())
                elif line.startswith("DONE "):
                    finish = line.split()[-1]      # stop / length / nonfinite
                    break                          # ERR lines also end with DONE: always drain to here
            requests += 1
            # An immediate EOS (finish=stop, zero tokens) is a valid model answer,
            # not a failure: "1, 2, 3, 4, 5," does this under greedy decoding.
            if errors or (not tokens and finish != "stop") or 0 in tokens[:3]:
                bad += 1
                print(f"t={a.minutes*60-(deadline-time.monotonic()):.0f}s BAD request={requests} "
                      f"errors={errors[:2]} tokens={tokens[:3]}", flush=True)
            if first is None and tokens:
                first = tokens[:3]
            if requests % 10 == 0:
                print(f"t={a.minutes*60-(deadline-time.monotonic()):.0f}s "
                      f"requests={requests} bad={bad} first={first} {stats}", flush=True)
            if proc.poll() is not None:
                raise RuntimeError("engine died (xe reset?)")
    except (RuntimeError, TimeoutError, OSError) as exc:
        print(f"RESULT FAILED after {requests} requests: {exc}", flush=True)
        return 1
    finally:
        try:
            proc.stdin.write("QUIT\n")
            proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass
        proc.wait(timeout=30)
        log.close()
    print(f"RESULT requests={requests} bad={bad} first_tokens={first}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
