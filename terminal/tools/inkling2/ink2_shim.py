"""ink2_shim.py: minimal OpenAI-compatible /v1/chat/completions in front of `strata-ink2 --serve` (Strata engine protocol).

Stop-gap for benchmarking Inkling-2-Mini before the real Strata server learns this family (its tokenizer is Strata's own
BPE, which has no pre-tokenizer for this vocab, and its frontend only splits </think>).  HF tokenizer + the model's
chat_template.jinja; one engine, requests serialized.  Model-card sampling (generation_config.json: temperature 1.0) when
the request gives none.  Output fields from the model's markers: <|content_thinking|> -> reasoning_content,
<|content_text|> -> content, <|end_message|> closes, <|content_model_end_sampling|> = EOS.

usage: ink2_shim.py --engine BIN --model DIR [--port 8080] [--ctx 65536] [--chunk 1024] [--tc 1] [--log FILE]
"""
import argparse, json, subprocess, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from transformers import AutoTokenizer

THINK, TEXT, END_MSG, EOS, MSG_MODEL = 200008, 200004, 200010, 200006, 200001


class Engine:
    def __init__(self, a):
        self.p = subprocess.Popen([a.engine, "--serve", "--model", a.model, "--max-context", str(a.ctx), "--chunk",
                                   str(a.chunk), "--tc", str(a.tc)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=open(a.log, "a"), text=True, bufsize=1)
        line = self.p.stdout.readline().split()
        assert line and line[0] == "READY", f"engine did not start: {line}"
        self.ctx = int(line[1])
        self.lock = threading.Lock()

    def send(self, s):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def gen(self, ids, max_new, keys):
        """yields ("T", id) ... then ("DONE", fields) or ("ERR", msg)"""
        kv = "".join(f" {k}={v}" for k, v in keys.items() if v is not None)
        self.send(f"GEN {max_new}{kv} {','.join(map(str, ids))}")
        while True:
            line = self.p.stdout.readline()
            if not line:
                yield ("ERR", "engine exited")
                return
            f = line.split()
            if not f or f[0] == "PP":
                continue
            if f[0] == "T":
                yield ("T", int(f[1]))
            elif f[0] == "DONE":
                yield ("DONE", f[1:])
                return
            elif f[0] == "ERR":
                yield ("ERR", line[4:].strip())
                return


class Splitter:
    """routes generated ids into reasoning_content / content, detokenizing incrementally per segment"""
    def __init__(self, tok):
        self.tok, self.field, self.ids, self.po, self.ro = tok, None, [], 0, 0

    def push(self, t):
        if t in (THINK, TEXT):
            self.field, self.ids, self.po, self.ro = ("reasoning_content" if t == THINK else "content"), [], 0, 0
            return None
        if t in (END_MSG, EOS, MSG_MODEL) or t >= 199998:
            self.field = None
            return None
        if self.field is None:
            return None
        self.ids.append(t)
        prev = self.tok.decode(self.ids[self.po:self.ro])
        new = self.tok.decode(self.ids[self.po:])
        if len(new) > len(prev) and not new.endswith("�"):
            self.po, self.ro = self.ro, len(self.ids)
            return (self.field, new[len(prev):])
        return None


def make_handler(eng, tok, gen_defaults, model_name):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path.startswith("/v1/models"):
                self._json(200, {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "local"}]})
            elif self.path.startswith("/health"):
                self._json(200, {"status": "ok"})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if not self.path.startswith("/v1/chat/completions"):
                return self._json(404, {"error": "not found"})
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            kw = {}
            if req.get("reasoning_effort") is not None:
                kw["reasoning_effort"] = req["reasoning_effort"]
            ids = tok.apply_chat_template(req["messages"], tokenize=True, add_generation_prompt=True, **kw)
            if hasattr(ids, "keys"):
                ids = ids["input_ids"]
            max_new = int(req.get("max_tokens") or req.get("max_completion_tokens") or 4096)
            keys = {"temperature": req.get("temperature", gen_defaults.get("temperature")), "top_p": req.get("top_p"),
                    "top_k": req.get("top_k"), "min_p": req.get("min_p"), "seed": req.get("seed")}
            stream = bool(req.get("stream"))
            cid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())
            sp = Splitter(tok)
            out = {"reasoning_content": [], "content": []}
            finish, done, err = "length", None, None
            with eng.lock:
                if stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()

                def sse(obj):
                    self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
                    self.wfile.flush()

                broken = False
                for kind, v in eng.gen(ids, max_new, keys):
                    if kind == "T":
                        if v == EOS:
                            finish = "stop"
                        piece = sp.push(v)
                        if piece:
                            out[piece[0]].append(piece[1])
                            if stream and not broken:
                                try:
                                    sse({"id": cid, "object": "chat.completion.chunk", "created": created, "model": model_name,
                                         "choices": [{"index": 0, "delta": {piece[0]: piece[1]}, "finish_reason": None}]})
                                except (BrokenPipeError, ConnectionResetError):
                                    broken = True
                                    eng.send("STOP")
                    elif kind == "DONE":
                        done = v
                    else:
                        err = v
                if done:
                    n_gen, n_prompt, pms, dms = int(done[0]), int(done[1]), float(done[2]), float(done[3])
                    usage = {"prompt_tokens": n_prompt, "completion_tokens": n_gen, "total_tokens": n_prompt + n_gen}
                    timings = {"prompt_n": n_prompt, "prompt_ms": pms, "prompt_per_second": 1000 * n_prompt / max(pms, 1e-3),
                               "predicted_n": n_gen, "predicted_ms": dms, "predicted_per_second": 1000 * n_gen / max(dms, 1e-3)}
                else:
                    usage, timings = None, None
                if stream:
                    if broken:
                        return
                    if err:
                        sse({"error": {"message": err}})
                    sse({"id": cid, "object": "chat.completion.chunk", "created": created, "model": model_name,
                         "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
                    if (req.get("stream_options") or {}).get("include_usage"):
                        sse({"id": cid, "object": "chat.completion.chunk", "created": created, "model": model_name,
                             "choices": [], "usage": usage, "timings": timings})
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    self.close_connection = True
                    return
            if err:
                return self._json(500, {"error": {"message": err}})
            msg = {"role": "assistant", "content": "".join(out["content"]),
                   "reasoning_content": "".join(out["reasoning_content"])}
            self._json(200, {"id": cid, "object": "chat.completion", "created": created, "model": model_name,
                             "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                             "usage": usage, "timings": timings})
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--ctx", type=int, default=65536)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--tc", type=int, default=1)
    ap.add_argument("--log", default="/var/ramdisk/inkling/shim-engine.log")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    gen_defaults = json.load(open(f"{a.model}/generation_config.json"))
    eng = Engine(a)
    print(f"engine ready, ctx {eng.ctx}; listening on {a.host}:{a.port}", flush=True)
    ThreadingHTTPServer((a.host, a.port), make_handler(eng, tok, gen_defaults, "inkling-2-mini")).serve_forever()


if __name__ == "__main__":
    main()
