#!/usr/bin/env python3
"""An OpenAI-compatible endpoint in front of this stack's own engine.

The model runs in `build-cann/model_seq ... serve`: it loads the weights once and then answers one
request at a time over stdin/stdout. This process is only the HTTP and text layer.

Tokenization is borrowed from the llama.cpp server on 18789 (`/tokenize`, `/detokenize`,
`/apply-template`), because the point of this endpoint is the ported model, not a second tokenizer.
That dependency is why the endpoint reports it in /health.

    python3 cann/tests/serve_engine.py [--port 18790] [--model ...]

Endpoints: GET /health, GET /v1/models, POST /v1/completions, POST /v1/chat/completions.
Not implemented: streaming, tools, logprobs.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "..", "..", "build-cann", "model_seq")
DEFAULT_MODEL = "/data/strata-qwen36/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
LLAMA = "http://127.0.0.1:18789"
MODEL_NAME = "qwen3.6-35b-a3b-cann"


class Engine:
    """One worker process, one request at a time."""

    def __init__(self, model_path):
        env = dict(os.environ)
        cann = env.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-9.0.0")
        env["LD_LIBRARY_PATH"] = (
            f"{cann}/lib64:{cann}/x86_64-linux/lib64:" + env.get("LD_LIBRARY_PATH", "")
        )
        self.proc = subprocess.Popen(
            [BIN, model_path, "0", "serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=env,
        )
        self.lock = threading.Lock()
        line = self.proc.stdout.readline().strip()
        if line != "READY":
            raise RuntimeError(f"the engine did not start: {line!r}")

    def generate(self, tokens, n_predict, temperature):
        with self.lock:
            ids = ",".join(str(t) for t in tokens)
            self.proc.stdin.write(f"RUN {n_predict} {temperature} {ids}\n")
            self.proc.stdin.flush()
            out = []
            while True:
                line = self.proc.stdout.readline()
                if not line:
                    raise RuntimeError("the engine exited")
                line = line.strip()
                if line == "END":
                    return out
                if line.startswith("GEN "):
                    parts = line.split()
                    out.append(int(parts[1]))
                elif line.startswith("GEN_STOP"):
                    return out

    def stop(self):
        try:
            self.proc.stdin.write("QUIT\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=30)
        except Exception:
            self.proc.kill()


def llama_post(path, payload):
    req = urllib.request.Request(
        LLAMA + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def tokenize(text):
    return llama_post("/tokenize", {"content": text})["tokens"]


def detokenize(tokens):
    return llama_post("/detokenize", {"tokens": tokens})["content"]


def apply_template(messages):
    try:
        return llama_post("/apply-template", {"messages": messages})["prompt"]
    except urllib.error.HTTPError:
        return "\n".join(m.get("content", "") for m in messages)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    engine = None

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/health", "/health/"):
            self._send(200, {"status": "ok", "engine": "strata-cann", "tokenizer": LLAMA})
            return
        if self.path.startswith("/v1/models"):
            self._send(200, {"object": "list", "data": [{"id": MODEL_NAME, "object": "model",
                                                          "owned_by": "strata"}]})
            return
        self._send(404, {"error": {"message": "not found", "type": "invalid_request_error"}})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            self._send(400, {"error": {"message": "invalid json", "type": "invalid_request_error"}})
            return

        try:
            if self.path.startswith("/v1/chat/completions"):
                prompt = apply_template(body.get("messages", []))
                n = int(body.get("max_tokens", 32))
                temp = float(body.get("temperature", 0.0))
                self._complete(prompt, n, temp, chat=True)
            elif self.path.startswith("/v1/completions"):
                prompt = body.get("prompt", "")
                if isinstance(prompt, list):
                    prompt = prompt[0]
                n = int(body.get("max_tokens", 32))
                temp = float(body.get("temperature", 0.0))
                self._complete(prompt, n, temp, chat=False)
            else:
                self._send(404, {"error": {"message": "not found", "type": "invalid_request_error"}})
        except Exception as e:
            self._send(500, {"error": {"message": str(e), "type": "server_error"}})

    def _complete(self, prompt, n_predict, temperature, chat):
        t0 = time.time()
        tokens = tokenize(prompt)
        generated = self.engine.generate(tokens, n_predict, temperature)
        text = detokenize(generated) if generated else ""
        elapsed = time.time() - t0
        payload = {
            "id": "cmpl-" + os.urandom(8).hex(),
            "object": "chat.completion" if chat else "text_completion",
            "created": int(time.time()),
            "model": MODEL_NAME,
            "choices": [
                {
                    "index": 0,
                    "message" if chat else "text": {"role": "assistant", "content": text} if chat else text,
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "prompt_tokens": len(tokens),
                "completion_tokens": len(generated),
                "total_tokens": len(tokens) + len(generated),
            },
            "strata": {
                "engine": "strata-cann (this port, not llama.cpp)",
                "seconds": round(elapsed, 2),
                "tokens": generated,
            },
        }
        self._send(200, payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18790)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    print(f"starting the engine on {args.model} ...", flush=True)
    engine = Engine(args.model)
    print("engine ready", flush=True)
    Handler.engine = engine
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
