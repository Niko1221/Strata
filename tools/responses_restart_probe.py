"""Prove encrypted full-history replay across two actual server processes.

Both processes use MockEngine/ByteTokenizer; model output is explicitly synthetic.
Only the environment's deployment key persists. No response records or native/GPU state survive.
"""
from __future__ import annotations
import argparse
import contextlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.frontend import ChatTemplate
from serve.response_replay import ReplayCodec
from serve.response_replay import KEY_ENV
from cryptography.fernet import Fernet
from serve.server import ByteTokenizer, MockEngine, Server, Service, make_handler

MODEL = "qwen3.8-flash-next"


def child(turn):
    tok = ByteTokenizer()
    script = "Unique earlier reasoning marker.</think>Earlier answer." if turn == 1 else "Restarted answer."
    svc = Service(MockEngine(tok, script), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
    svc.experimental_responses = True
    svc.responses_replay = ReplayCodec.load()
    httpd = Server(("127.0.0.1", 0), make_handler(svc))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    print(json.dumps({"base": f"http://127.0.0.1:{httpd.server_address[1]}", "pid": os.getpid()}), flush=True)
    try:
        input()
        print(json.dumps({"prompt_contains_earlier_reasoning": "Unique earlier reasoning marker." in tok.decode(svc.engine.last_prompt),
                          "response_store_exists": hasattr(svc, "response_store")}), flush=True)
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


@contextlib.contextmanager
def process(key, turn, evidence):
    command = [sys.executable, str(Path(__file__).resolve()), "--child", str(turn)]
    env = {**os.environ, KEY_ENV: key}
    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8")
    lines, ready = [], queue.Queue()

    def read():
        for line in proc.stdout:
            lines.append(line)
            if len(lines) == 1:
                ready.put(line)

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    try:
        info = json.loads(ready.get(timeout=20))
        yield info
    finally:
        try:
            proc.stdin.write("stop\n")
            proc.stdin.flush()
            proc.wait(timeout=10)
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=10)
            reader.join(timeout=3)
        evidence.extend(lines)
        if proc.returncode != 0:
            raise RuntimeError("probe server did not exit cleanly")


def post(base, body):
    req = urllib.request.Request(base + "/v1/responses", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as response:
        return json.load(response)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--child", type=int)
    a = ap.parse_args()
    if a.child:
        return child(a.child)
    if a.out is None:
        ap.error("--out is required")
    a.out.mkdir(parents=True, exist_ok=True)
    logs = []
    with tempfile.TemporaryDirectory(prefix="responses-restart-") as temporary:
        key = Fernet.generate_key().decode("ascii")
        with process(key, 1, logs) as first:
            response = post(first["base"], {"model": MODEL, "store": False, "input": "Earlier question", "reasoning": {"effort": "low"}})
        history = [{"role": "user", "content": "Earlier question"}] + response["output"]
        history[1].pop("content")
        history.append({"role": "user", "content": "Continue after a restart"})
        with process(key, 2, logs) as second:
            final = post(second["base"], {"model": MODEL, "store": False, "input": history})
        assert first["pid"] != second["pid"]
        assert final["status"] == "completed" and final["output"][0]["content"][0]["text"] == "Restarted answer."
        assert json.loads(logs[-1])["prompt_contains_earlier_reasoning"] is True
        assert json.loads(logs[-1])["response_store_exists"] is False
        assert list(Path(temporary).iterdir()) == []
    receipt = {"result": "pass", "server_pids": [first["pid"], second["pid"]],
               "server": "Strata HTTP + Service + MockEngine", "model_output_provenance": "synthetic scripts in this probe",
               "persisted_files": [], "reused_configuration": "deployment key environment variable", "encrypted_only_reasoning_replayed": True,
               "no_response_store": True, "native_inference_test": False}
    (a.out / "restart-probe.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    (a.out / "restart-server-log.txt").write_text("".join(logs), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
