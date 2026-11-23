"""Compare the native two-host serve protocol against the one-device CUDA implementation."""
import argparse
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time

def measure(executable, model, mpi, draft, timeout):
    # The optional wrapper provides the MPI host list and transport environment.
    # It must pass its arguments through and direct stdin to rank 0.
    command = [str(executable), "--serve", "--native", str(model), "--parallel", "4",
               "--max-context", "4096", "--chunk", "128", "--draft", draft]
    if mpi:
        command.append("--mpi-experts")
    prefixes = [[32, 383, 712, 1001] * 9, [32, 383, 712, 1001] * 9 + [201, 220, 311],
                [253, 270, 415, 811] * 17]
    frames = ["GEN 8 temperature=0 " + ",".join(map(str, ids)) for ids in prefixes]
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          start_new_session=True) as proc, selectors.DefaultSelector() as monitor:
        deadline = time.monotonic() + timeout
        monitor.register(proc.stdout, selectors.EVENT_READ)
        buffer = bytearray()
        clean_exit = False
        def send(line):
            proc.stdin.write((line + "\n").encode())
            proc.stdin.flush()
        def until(prefix):
            lines = []
            last = ""
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"{executable} timed out waiting for {prefix}; last line: {last!r}")
                end = buffer.find(b"\n")
                if end < 0:
                    if len(buffer) > 65536:
                        raise ValueError("oversized native protocol line")
                    if not monitor.select(remaining):
                        continue
                    chunk = os.read(proc.stdout.fileno(), 8192)
                    if not chunk:
                        raise RuntimeError(f"engine exited before {prefix}; last line: {last!r}")
                    buffer.extend(chunk)
                    continue
                line = buffer[:end].decode(errors="replace")
                del buffer[:end + 1]
                last = line[-256:]
                words = line.split()
                if not words:
                    continue
                if words[0] in ("ERR", "FATAL"):
                    raise ValueError(last)
                if words[0] == prefix:
                    return [*lines, words]
                if (prefix == "READY" and words[0] == "INFO") or (prefix == "DONE" and words[0] == "T"):
                    lines.append(words)
                    if len(lines) > 8:
                        raise ValueError("too many native result frames")
        try:
            startup = until("READY")
            info = next(row for row in startup if row[0] == "INFO")
            backend = "cuda-mpi-experts" if mpi else "cuda"
            if "backend="+backend not in info or "batch_slots=4" not in info or "drafting="+draft not in info:
                raise AssertionError(info)
            runs = []
            for frame in frames:
                send(frame)
                rows = until("DONE")
                tokens = [int(row[1]) for row in rows if row[0] == "T"]
                if not tokens or len(tokens) != int(rows[-1][1]):
                    raise AssertionError("generated token count differs from DONE")
                runs.append({"tokens":tokens,"reused":int(rows[-1][-1])})
            if runs[0]["reused"] or runs[1]["reused"] <= 0 or runs[2]["reused"]:
                raise AssertionError("unexpected prefix reuse: " + str(runs))
            send("QUIT")
            proc.stdin.close()
            if proc.wait(timeout=max(0, deadline - time.monotonic())):
                raise RuntimeError("nonzero engine exit")
            clean_exit = True
            return runs
        finally:
            if not clean_exit:
                # Let mpirun terminate its remote ranks before forcing the local wrapper down.
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(proc.pid, sig)
                    except ProcessLookupError:
                        break
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", type=Path, required=True)
    parser.add_argument("--pair", type=Path, required=True, help="mpirun wrapper for the same binary")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=180, help="seconds per engine invocation")
    args = parser.parse_args()
    if not args.timeout > 0:
        parser.error("--timeout must be positive")
    for draft in ("off","dspark"):
        single = measure(args.single, args.model, False, draft, args.timeout)
        pair = measure(args.pair, args.model, True, draft, args.timeout)
        if single != pair:
            raise SystemExit(f"{draft} differs: single={single} pair={pair}")
        print(json.dumps({"draft": draft, "matched": len(pair), "runs":pair}))
