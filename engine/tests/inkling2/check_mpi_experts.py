"""Compare the native two-host serve protocol against the one-device CUDA implementation."""
import argparse
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
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
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                          bufsize=1, start_new_session=True) as proc:
        inbox = queue.Queue()
        deadline = time.monotonic() + timeout
        def receive():
            for line in proc.stdout:
                inbox.put(line)
            inbox.put(None)
        reader = threading.Thread(target=receive, daemon=True)
        reader.start()
        clean_exit = False
        def send(line):
            proc.stdin.write(line + "\n")
            proc.stdin.flush()
        def until(prefix):
            lines = []
            while True:
                try:
                    line = inbox.get(timeout=max(0, deadline - time.monotonic()))
                except queue.Empty as exc:
                    raise TimeoutError(f"{executable} timed out waiting for {prefix}") from exc
                if line is None:
                    raise RuntimeError(f"engine exited before {prefix}: {lines[-1:]}")
                words = line.strip().split()
                if not words:
                    continue
                if words[0] == "ERR":
                    raise ValueError(line.rstrip())
                lines.append(words)
                if words[0] == prefix:
                    return lines
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
                if not tokens:
                    raise AssertionError("no generated tokens")
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
                    reader.join(timeout=2)
                    if not reader.is_alive():
                        break

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
