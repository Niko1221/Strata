"""Replay the saved engine protocol requests sequentially against one engine.

python replay.py --engine /path/to/strata --settings settings.json --output run-dir
settings.json: {"args": [...], "env": {...}} with local model paths filled in.
"""
import argparse
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text(encoding="utf-8"))
    assert not any("<" in str(v) for v in settings["args"]), "fill in the local model paths"
    args.output.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items()
           if not k.upper().startswith(("STRATA_", "HIP_", "CUDA_", "ROCR_", "HSA_", "OMP_"))}
    env.update(settings["env"])
    lines = queue.Queue()
    with (args.output / "stderr.log").open("w", encoding="utf-8") as err, (args.output / "stdout.log").open("w", encoding="utf-8") as out:
        process = subprocess.Popen([args.engine, "--serve", *settings["args"]], env=env,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
                                   text=True, bufsize=1)
        def reader():
            for line in process.stdout:
                lines.put(line)
            lines.put(None)
        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        def wait_for(prefix, seconds):
            deadline = time.monotonic() + seconds
            while True:
                line = lines.get(timeout=max(.001, deadline - time.monotonic()))
                if line is None or line.startswith("ERR"):
                    raise RuntimeError(line or "engine exited")
                out.write(line)
                out.flush()
                if line.startswith(prefix):
                    return
                if time.monotonic() >= deadline:
                    raise TimeoutError(prefix)
        try:
            wait_for("READY", 240)
            commands = (Path(__file__).parent / "requests.txt").read_text(encoding="utf-8").splitlines()
            for command in commands:
                process.stdin.write(command + "\n")
                process.stdin.flush()
                if command.startswith("GEN "):
                    wait_for("DONE", 60)
            process.wait(timeout=20)
            if process.returncode:
                raise RuntimeError(f"engine exited with {process.returncode}")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=20)
            thread.join(timeout=5)


if __name__ == "__main__":
    main()
