"""serve/server.py - plan v0.3 P8: OpenAI and Anthropic endpoints over any engine that maps token ids to tokens.

    python -m serve.server --engine mock --port 8095            (a scripted engine, for clients and tests)
    python -m serve.server --engine strata --config strata.json --port 8080   (the real engine, resident)

Endpoints: POST /v1/chat/completions (OpenAI, stream and non-stream), POST /v1/messages (Anthropic, stream and
non-stream), GET /v1/models, GET /health. One sequence at a time behind a FIFO (plan: one resident sequence).
Images (optional, when the config has a "vision" entry): OpenAI image_url parts and Anthropic image blocks (base64
data, http(s) URLs or local file paths; JPEG/PNG/BMP/GIF) go through `strata-vision` (the model's mmproj file) and
reach the engine as embeddings (`GENI`).
Requests whose prompt plus max tokens exceed the engine's context are REJECTED with 400, never truncated.

The engine boundary is `Engine.generate(prompt_ids, max_new, sampling, cancel) -> iterator of token ids`.
`StrataEngine` keeps one `strata --serve` process resident (weights, expert arena and VRAM tier load once) and
talks to it over stdin/stdout; `MockEngine` is a scripted stand-in that makes every API path testable without a GPU.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator, Protocol

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))   # run as a script (run-<model>.bat) as well as a module
from serve.frontend import (ChatTemplate, Event, OutputParser, anthropic_to_messages,  # noqa: E402
                            images_of, openai_to_messages)

IM_END = "<|im_end|>"
IMAGE_PAD = "<|image_pad|>"


# ------------------------------------------------------------------------------------------------ engines
class Engine(Protocol):
    max_context: int
    def generate(self, ids: list[int], max_new: int, sampling: dict, cancel: threading.Event) -> Iterator[int]: ...


class MockEngine:
    """Replays a scripted completion (text) as token ids, one per step, then the end-of-turn token."""

    def __init__(self, tokenizer, script: str, max_context: int = 32768, delay_s: float = 0.0):
        self.tok, self.max_context, self.delay = tokenizer, max_context, delay_s
        self.script = tokenizer.encode(script, parse_special=True) + tokenizer.encode(IM_END, parse_special=True)
        self.last_prompt: list[int] = []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.last_prompt = list(ids)
        self.last_embeddings = embeddings
        for t in self.script[:max_new]:
            if cancel.is_set():
                return
            if self.delay:
                time.sleep(self.delay)
            yield t


class StrataEngine:
    """The resident engine: `strata --serve` reads `GEN <max_new> <ids>` lines and streams `T <id>` lines, then
    `DONE ...`.  Requests are serialized by the service's FIFO, so one pipe is enough."""

    def __init__(self, exe: str, args: list[str], cwd: str | None = None, log: str | None = None,
                 env: dict | None = None):
        self.log = open(log, "a", encoding="utf-8") if log else subprocess.DEVNULL
        self.proc = subprocess.Popen([exe, "--serve", *args], cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.log, text=True, encoding="utf-8", bufsize=1, env=env)
        self.max_context = 0
        self.last = {}
        for line in self.proc.stdout:
            if line.startswith("READY"):
                self.max_context = int(line.split()[1])
                break
        if self.max_context <= 0:
            raise RuntimeError("the engine exited before it was ready" + (f" (see {log})" if log else ""))

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        head = f"GENI {int(max_new)} {embeddings}" if embeddings else f"GEN {int(max_new)}"
        self.proc.stdin.write(f"{head} {','.join(str(int(t)) for t in ids)}\n")
        self.proc.stdin.flush()
        done = False
        try:
            for line in self.proc.stdout:
                if line.startswith("T "):
                    if not cancel.is_set():
                        yield int(line[2:])
                elif line.startswith("DONE"):
                    f = line.split()
                    self.last = {"generated": int(f[1]), "prompt_tokens": int(f[2]), "prompt_ms": float(f[3]),
                                 "decode_ms": float(f[4]), "finish": f[5]}
                    done = True
                    return
                elif line.startswith("ERR"):
                    done = True
                    raise ValueError(line[4:].strip())
            raise RuntimeError("the engine process ended")
        finally:
            if not done:                                  # the consumer stopped early: drain to DONE
                for line in self.proc.stdout:
                    if line.startswith("DONE") or line.startswith("ERR"):
                        break

    def close(self):
        try:
            self.proc.stdin.write("QUIT\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


class Vision:
    """The resident image encoder: `strata-vision` (llama.cpp mtmd + the mmproj file) reads `ENC <image> <out>`
    lines and writes each image's embeddings; results are cached by the image's hash, so a conversation that
    sends the same picture again (every turn, with most clients) encodes it once."""

    def __init__(self, cfg: dict, log=None, env: dict | None = None):
        args = [cfg["exe"], "--mmproj", cfg["mmproj"], "--model", cfg["model"]]
        if cfg.get("gpu"):
            args.append("--gpu")
        if cfg.get("threads"):
            args += ["--threads", str(cfg["threads"])]
        if cfg.get("max_tokens"):
            args += ["--max-tokens", str(cfg["max_tokens"])]
        self.dir = Path(tempfile.mkdtemp(prefix="strata-vision-"))
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log or subprocess.DEVNULL,
                                     text=True, encoding="utf-8", bufsize=1, env=env)
        line = self.proc.stdout.readline()
        if not line.startswith("READY"):
            raise RuntimeError("the vision encoder did not start: " + line.strip())
        self.lock = threading.Lock()
        self.cache: dict[str, tuple[Path, int]] = {}

    @staticmethod
    def load(source: str) -> bytes:
        if source.startswith("data:"):
            return base64.b64decode(source.split(",", 1)[1])
        if source.startswith(("http://", "https://")):
            req = urllib.request.Request(source, headers={"User-Agent": "strata"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        path = source[7:] if source.startswith("file://") else source
        if path and os.path.isfile(path):
            return Path(path).read_bytes()
        raise ValueError("an image must be a data: URL, an http(s) URL or a local file path")

    def encode(self, source: str) -> tuple[Path, int]:
        """-> (embeddings file, number of image tokens)."""
        data = self.load(source)
        key = hashlib.sha256(data).hexdigest()[:32]
        with self.lock:
            if key in self.cache:
                return self.cache[key]
            img, out = self.dir / f"{key}.img", self.dir / f"{key}.sve"
            img.write_bytes(data)
            self.proc.stdin.write(f"ENC {img} {out}\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline().strip()
            img.unlink(missing_ok=True)
            if not line.startswith("OK"):
                raise ValueError("the image could not be read: " + (line[4:] if line.startswith("ERR") else
                                                                     "the vision encoder stopped"))
            self.cache[key] = (out, int(line.split()[1]))
            if len(self.cache) > 64:                                   # oldest first
                old = next(iter(self.cache))
                self.cache.pop(old)[0].unlink(missing_ok=True)
            return self.cache[key]

    def close(self):
        try:
            self.proc.stdin.write("QUIT\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


def child_env(cfg: dict) -> dict:
    """The engine's environment: the CUDA libraries setup installed (pip's nvidia packages, or the toolkit that
    compiled it) first on the library search path."""
    env = dict(os.environ)
    dirs = [d for d in cfg.get("lib_dirs") or [] if Path(d).is_dir()]
    if dirs:
        var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        env[var] = os.pathsep.join(dirs + ([env[var]] if env.get(var) else []))
    return env


class ByteTokenizer:
    """Tiny stand-in tokenizer for tests without the pack: one id per UTF-8 byte, specials as ids >= 256."""
    SPECIALS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|vision_start|>", "<|image_pad|>", "<|vision_end|>"]

    def encode(self, text, parse_special=False):
        out, i = [], 0
        while i < len(text):
            for k, s in enumerate(self.SPECIALS):
                if parse_special and text.startswith(s, i):
                    out.append(256 + k)
                    i += len(s)
                    break
            else:
                out.extend(text[i].encode("utf-8"))
                i += 1
        return out

    def decode(self, ids, errors="replace"):
        raw = bytearray()
        for t in ids:
            raw += self.SPECIALS[t - 256].encode() if t >= 256 else bytes([t])
        return raw.decode("utf-8", errors=errors)


# ------------------------------------------------------------------------------------------------ core
class Detokenizer:
    """Incremental decode: re-decode the generated ids and emit only the new, complete suffix (a multi-byte
    character split across tokens is held until complete)."""

    def __init__(self, tok):
        self.tok, self.ids, self.sent = tok, [], 0

    def push(self, t: int) -> str:
        self.ids.append(t)
        text = self.tok.decode(self.ids)
        if text.endswith("�"):
            return ""
        delta, self.sent = text[self.sent:], len(text)
        return delta


class Service:
    def __init__(self, engine: Engine, tokenizer, template: ChatTemplate, model_name: str = "qwen3.8-flash-next",
                 vision: Vision | None = None):
        self.engine, self.tok, self.template, self.model, self.vision = engine, tokenizer, template, model_name, vision
        self.fifo = threading.Lock()
        self.embeddings = threading.local()           # the current request's image embeddings file (GENI)
        self.api_key = ""                              # when set, /v1/* needs it (Bearer or x-api-key)
        self.stop_ids = set(tokenizer.encode(IM_END, parse_special=True) +
                            tokenizer.encode("<|endoftext|>", parse_special=True))

    def prepare(self, messages, tools, kwargs, max_new):
        prompt = self.template.render(messages, tools=tools, **kwargs)
        ids = self.tok.encode(prompt, parse_special=True)
        self.embeddings.path = None
        images = images_of(messages)
        if images:
            if self.vision is None:
                raise ValueError("this server was started without the vision encoder (run setup again and choose "
                                 "'vision'), so it cannot read images")
            pad = self.tok.encode(IMAGE_PAD, parse_special=True)[0]
            encoded = [self.vision.encode(src) for src in images]
            out, k = [], 0
            for t in ids:                               # one <|image_pad|> per image -> one per image token
                if t == pad and k < len(encoded):
                    out += [pad] * encoded[k][1]
                    k += 1
                else:
                    out.append(t)
            if k != len(encoded):
                raise ValueError("the prompt and its images do not match")
            ids = out
            combined = self.vision.dir / f"req-{uuid.uuid4().hex[:12]}.sve"
            with open(combined, "wb") as f:
                for path, _ in encoded:
                    f.write(path.read_bytes())
            self.embeddings.path = combined
        if len(ids) + max_new > self.engine.max_context:
            raise ValueError(f"prompt ({len(ids)} tokens) + max tokens ({max_new}) exceeds the context "
                             f"({self.engine.max_context}); requests are never truncated")
        return ids, kwargs.get("enable_thinking", True) is not False

    def run(self, ids, thinking, tools, max_new, sampling, cancel) -> Iterator[tuple[str, object]]:
        """Yields ("event", Event) as text arrives, then ("done", {"finish": .., "completion_tokens": ..})."""
        parser, detok, n, finish = OutputParser(thinking=thinking, tools=tools), Detokenizer(self.tok), 0, "length"
        emb = getattr(self.embeddings, "path", None)
        try:
            with self.fifo:
                gen = self.engine.generate(ids, max_new, sampling, cancel, embeddings=emb) if emb else \
                    self.engine.generate(ids, max_new, sampling, cancel)
                for t in gen:
                    n += 1
                    if t in self.stop_ids:
                        finish = "stop"
                        break
                    for ev in parser.feed(detok.push(t)):
                        yield "event", ev
        finally:
            if emb:
                Path(emb).unlink(missing_ok=True)
        for ev in parser.finish():
            yield "event", ev
        yield "done", {"finish": finish, "completion_tokens": n}


# ------------------------------------------------------------------------------------------------ OpenAI
def openai_chunks(svc: Service, req: dict, ids, thinking, tools, max_new, cancel):
    cid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())

    def chunk(delta, finish=None):
        return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": svc.model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    yield chunk({"role": "assistant", "content": ""})
    calls = 0
    for kind, x in svc.run(ids, thinking, tools, max_new, req, cancel):
        if kind == "event":
            ev: Event = x
            if ev.kind == "reasoning" and ev.text:
                yield chunk({"reasoning_content": ev.text})
            elif ev.kind == "content" and ev.text:
                yield chunk({"content": ev.text})
            elif ev.kind == "tool_call":
                yield chunk({"tool_calls": [{"index": calls, "id": ev.call.id, "type": "function",
                                             "function": {"name": ev.call.name,
                                                          "arguments": json.dumps(ev.call.arguments, ensure_ascii=False)}}]})
                calls += 1
        else:
            finish = "tool_calls" if calls and x["finish"] == "stop" else x["finish"]
            last = chunk({}, finish)
            last["usage"] = {"prompt_tokens": len(ids), "completion_tokens": x["completion_tokens"],
                             "total_tokens": len(ids) + x["completion_tokens"]}
            yield last


def openai_collect(chunks) -> dict:
    content, reasoning, calls, last = [], [], [], None
    for c in chunks:
        d = c["choices"][0]["delta"]
        content.append(d.get("content") or "")
        reasoning.append(d.get("reasoning_content") or "")
        calls += [{k: v for k, v in tc.items() if k != "index"} for tc in d.get("tool_calls") or []]
        last = c
    msg = {"role": "assistant", "content": "".join(content) or None}
    if "".join(reasoning):
        msg["reasoning_content"] = "".join(reasoning)
    if calls:
        msg["tool_calls"] = calls
    return {"id": last["id"], "object": "chat.completion", "created": last["created"], "model": last["model"],
            "choices": [{"index": 0, "message": msg, "finish_reason": last["choices"][0]["finish_reason"]}],
            "usage": last["usage"]}


# ------------------------------------------------------------------------------------------------ Anthropic
def anthropic_events(svc: Service, req: dict, ids, thinking, tools, max_new, cancel):
    mid = "msg_" + uuid.uuid4().hex[:24]
    yield "message_start", {"type": "message_start", "message": {
        "id": mid, "type": "message", "role": "assistant", "model": svc.model, "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": len(ids), "output_tokens": 0}}}
    index, open_kind, used_tool = -1, None, False

    def close():
        return ("content_block_stop", {"type": "content_block_stop", "index": index})

    for kind, x in svc.run(ids, thinking, tools, max_new, req, cancel):
        if kind == "event":
            ev: Event = x
            want = {"reasoning": "thinking", "content": "text", "tool_call": "tool_use"}[ev.kind]
            if ev.kind != "tool_call" and not ev.text:
                continue
            if open_kind != want or want == "tool_use":
                if open_kind is not None:
                    yield close()
                index += 1
                open_kind = want
                block = {"thinking": {"type": "thinking", "thinking": "", "signature": ""},
                         "text": {"type": "text", "text": ""},
                         "tool_use": {"type": "tool_use", "id": ev.call.id if ev.call else "", "name":
                                      ev.call.name if ev.call else "", "input": {}}}[want]
                yield "content_block_start", {"type": "content_block_start", "index": index, "content_block": block}
            if want == "thinking":
                yield "content_block_delta", {"type": "content_block_delta", "index": index,
                                              "delta": {"type": "thinking_delta", "thinking": ev.text}}
            elif want == "text":
                yield "content_block_delta", {"type": "content_block_delta", "index": index,
                                              "delta": {"type": "text_delta", "text": ev.text}}
            else:
                used_tool = True
                yield "content_block_delta", {"type": "content_block_delta", "index": index, "delta": {
                    "type": "input_json_delta", "partial_json": json.dumps(ev.call.arguments, ensure_ascii=False)}}
        else:
            if open_kind is not None:
                yield close()
            stop = "tool_use" if used_tool and x["finish"] == "stop" else \
                {"stop": "end_turn", "length": "max_tokens"}[x["finish"]]
            yield "message_delta", {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
                                    "usage": {"output_tokens": x["completion_tokens"]}}
            yield "message_stop", {"type": "message_stop"}


def anthropic_collect(events) -> dict:
    msg, blocks = None, []
    for name, e in events:
        if name == "message_start":
            msg = e["message"]
        elif name == "content_block_start":
            blocks.append(dict(e["content_block"]))
        elif name == "content_block_delta":
            d, b = e["delta"], blocks[-1]
            if d["type"] == "text_delta":
                b["text"] += d["text"]
            elif d["type"] == "thinking_delta":
                b["thinking"] += d["thinking"]
            else:
                b["input"] = json.loads(d["partial_json"])
        elif name == "message_delta":
            msg["stop_reason"] = e["delta"]["stop_reason"]
            msg["usage"]["output_tokens"] = e["usage"]["output_tokens"]
    msg["content"] = blocks
    return msg


# ------------------------------------------------------------------------------------------------ HTTP
def make_handler(svc: Service):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"                       # SSE ends by closing the connection

        def log_message(self, fmt, *args):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            if not svc.api_key:
                return True
            auth = self.headers.get("Authorization", "")
            given = auth[7:].strip() if auth.lower().startswith("bearer ") else self.headers.get("x-api-key", "")
            if given == svc.api_key:
                return True
            self._json(401, {"error": {"type": "authentication_error", "message": "missing or wrong API key"}})
            return False

        def do_GET(self):
            path = self.path.split("?")[0].rstrip("/")
            if path == "":
                body = (ROOT / "serve" / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/health":
                self._json(200, {"status": "ok", "max_context": svc.engine.max_context, "model": svc.model,
                                 "images": svc.vision is not None, "api_key": bool(svc.api_key)})
            elif path == "/v1/models":
                if self._authorized():
                    self._json(200, {"object": "list", "data": [{"id": svc.model, "object": "model"}]})
            else:
                self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if not self._authorized():
                return
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if self.path.rstrip("/") == "/v1/chat/completions":
                    self._openai(req)
                elif self.path.rstrip("/") == "/v1/messages":
                    self._anthropic(req)
                else:
                    self._json(404, {"error": {"message": "not found"}})
            except ValueError as e:
                self._json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})

        def _sse(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

        def _openai(self, req):
            messages, tools, kw = openai_to_messages(req)
            max_new = int(req.get("max_completion_tokens") or req.get("max_tokens") or 1024)
            ids, thinking = svc.prepare(messages, tools, kw, max_new)
            cancel = threading.Event()
            chunks = openai_chunks(svc, req, ids, thinking, tools, max_new, cancel)
            if not req.get("stream"):
                return self._json(200, openai_collect(chunks))
            self._sse()
            try:
                for c in chunks:
                    self.wfile.write(b"data: " + json.dumps(c, ensure_ascii=False).encode() + b"\n\n")
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
            except OSError:
                cancel.set()                                 # client went away: stop at the next step

        def _anthropic(self, req):
            messages, tools, kw = anthropic_to_messages(req)
            max_new = int(req.get("max_tokens") or 1024)
            ids, thinking = svc.prepare(messages, tools, kw, max_new)
            cancel = threading.Event()
            events = anthropic_events(svc, req, ids, thinking, tools, max_new, cancel)
            if not req.get("stream"):
                return self._json(200, anthropic_collect(events))
            self._sse()
            try:
                for name, e in events:
                    self.wfile.write(f"event: {name}\n".encode() + b"data: " +
                                     json.dumps(e, ensure_ascii=False).encode() + b"\n\n")
                    self.wfile.flush()
            except OSError:
                cancel.set()

    return Handler


class Server(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets a second server bind a port that is already serving, and requests then land on
    # either one (a forgotten second start of run-<model>.bat).  Without it the second start fails loudly instead.
    allow_reuse_address = os.name != "nt"


def serve(svc: Service, host="127.0.0.1", port=8095) -> ThreadingHTTPServer:
    httpd = Server((host, port), make_handler(svc))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["mock", "strata"], default="mock")
    ap.add_argument("--config", help="strata engine config (JSON: exe, args, cwd, tokenizer, model_name), "
                                     "written by setup.py")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--script", default="Thinking about it.</think>\n\nHello from the mock engine.")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--tokenizer", default=str(ROOT / "pack/full/tokenizer"),
                    help="pack tokenizer directory (falls back to a byte tokenizer if absent)")
    ap.add_argument("--open", action="store_true", help="open the local page in the browser once the model is ready")
    ap.add_argument("--api-key", default=os.environ.get("STRATA_API_KEY", ""),
                    help="require this key on /v1/* (Authorization: Bearer ... or x-api-key); also $STRATA_API_KEY")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8")) if a.config else {}
    try:                                                # before the minutes of loading: is the port free?
        Server((a.host, a.port), BaseHTTPRequestHandler).server_close()
    except OSError:
        ap.error(f"port {a.port} is already in use - is Strata (or another server) already running? "
                 f"Close it, or start this one with a different --port")
    if cfg.get("tokenizer"):
        a.tokenizer = cfg["tokenizer"]
    tok = ByteTokenizer()
    tpath = Path(a.tokenizer)
    if a.engine == "strata" and not (tpath / "vocab.json").exists():
        ap.error(f"the model's tokenizer is missing ({tpath / 'vocab.json'}); run setup again")
    if (tpath / "vocab.json").exists():
        import strata_tokenizer as ST
        vocab = json.loads((tpath / "vocab.json").read_text(encoding="utf-8"))
        tokens = [None] * len(vocab)
        for t, i in vocab.items():
            tokens[i] = t
        merges = (tpath / "merges.txt").read_text(encoding="utf-8").split("\n")
        types = json.loads((tpath / "token_type.json").read_text())
        tok = ST.Tokenizer(tokens, merges, types)
    if a.engine == "strata":
        if not cfg:
            ap.error("--engine strata needs --config")
        vision = None
        env = child_env(cfg)
        if cfg.get("vision"):
            print("loading the vision encoder ...", flush=True)
            vision = Vision(cfg["vision"], log=open(cfg["log"], "a", encoding="utf-8") if cfg.get("log") else None,
                            env=env)
        print("loading the model (the first start takes a minute or two) ...", flush=True)
        engine = StrataEngine(cfg["exe"], cfg["args"], cwd=cfg.get("cwd"), log=cfg.get("log"), env=env)
    else:
        engine, vision = MockEngine(tok, a.script), None
    # the model's own chat template (exported with its tokenizer), else the original model's
    tpl = tpath / "chat_template.jinja"
    svc = Service(engine, tok, ChatTemplate(tpl if tpl.exists() else ROOT / "serve/chat_template.jinja"),
                  model_name=cfg.get("model_name", "qwen3.8-flash-next"), vision=vision)
    svc.api_key = a.api_key or cfg.get("api_key", "")
    httpd = serve(svc, host=a.host, port=a.port)
    print(f"ready: http://{a.host}:{a.port}/v1  (OpenAI: /v1/chat/completions, Anthropic: /v1/messages, "
          f"context {engine.max_context} tokens{', images on' if vision else ''}"
          f"{', API key required' if svc.api_key else ''})", flush=True)
    print(f"       open http://{a.host}:{a.port}/ in a browser to chat; close this window to stop the model", flush=True)
    if a.open:
        import webbrowser
        webbrowser.open(f"http://{'127.0.0.1' if a.host in ('0.0.0.0', '') else a.host}:{a.port}/")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        httpd.shutdown()
        if hasattr(engine, "close"):
            engine.close()
        if vision:
            vision.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
