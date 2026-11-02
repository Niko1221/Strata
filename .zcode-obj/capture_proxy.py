"""A logging proxy: point a client at this port and see exactly what it sends and gets back.

Forwards to the real server unchanged (streaming included) and writes a readable record of each
request and response, so a client that behaves differently can be diagnosed instead of guessed at.

  python capture_proxy.py [listen_port] [target_port]

Then set the client's API base URL to http://127.0.0.1:<listen_port>/v1 and send one message.
"""
import json
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LISTEN = int(sys.argv[1]) if len(sys.argv) > 1 else 8085
TARGET = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
OUT = Path(__file__).with_name("capture.txt")


def note(text):
    with OUT.open("a", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text[:400], flush=True)


def describe(body: dict) -> str:
    """The parts that change how the engine decodes, in a fixed order."""
    bits = [f"model={body.get('model')!r}",
            f"stream={body.get('stream')!r}",
            f"max_tokens={body.get('max_tokens')!r}",
            f"temperature={body.get('temperature')!r}",
            f"top_p={body.get('top_p')!r}",
            f"top_k={body.get('top_k')!r}",
            f"min_p={body.get('min_p')!r}",
            f"seed={body.get('seed')!r}"]
    for key in ("strata_grammar", "grammar", "response_format", "tool_choice", "tools",
                "strata_mcp", "reasoning_effort", "chat_template_kwargs", "stop",
                "logit_bias", "presence_penalty", "repetition_penalty",
                "experimental_speed_projection"):
        if key in body and body[key] not in (None, [], {}, False):
            val = body[key]
            if key == "tools" and isinstance(val, list):
                bits.append(f"tools=<{len(val)}>" + "".join(
                    f" {(t.get('function') or {}).get('name')}" for t in val[:8]))
            else:
                s = json.dumps(val, ensure_ascii=False)
                bits.append(f"{key}={s[:160]}")
    msgs = body.get("messages") or []
    bits.append(f"messages=<{len(msgs)}>")
    for m in msgs:
        role = m.get("role")
        content = m.get("content")
        if isinstance(content, list):
            content = " ".join(str(p.get("text", p.get("type"))) for p in content
                               if isinstance(p, dict))
        content = (content or "")
        bits.append(f"  {role}: {len(content)} chars: {content[:150]!r}")
    return "\n".join(bits)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        self._forward("GET")

    def do_POST(self):
        self._forward("POST")

    def do_DELETE(self):
        self._forward("DELETE")

    def _forward(self, method):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        started = time.time()
        note(f"\n{'=' * 70}\n{time.strftime('%H:%M:%S')} {method} {self.path}")
        if raw:
            try:
                body = json.loads(raw.decode("utf-8"))
                note(describe(body))
            except Exception:
                note(f"(body {len(raw)} bytes, not json)")
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in ("host", "content-length", "accept-encoding")}
        req = urllib.request.Request(f"http://127.0.0.1:{TARGET}{self.path}", data=raw or None,
                                     headers=headers, method=method)
        try:
            resp = urllib.request.urlopen(req, timeout=3600)
        except urllib.error.HTTPError as exc:
            payload = exc.read()
            note(f"upstream HTTP {exc.code}: {payload[:400].decode('utf-8', 'replace')}")
            self.send_response(exc.code)
            self.send_header("Content-Type", exc.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        ctype = resp.headers.get("Content-Type", "application/json")
        self.send_response(resp.status)
        self.send_header("Content-Type", ctype)
        if "text/event-stream" in ctype:
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            text = []
            for line in resp:
                chunk = line
                self.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n")
                self.wfile.flush()
                if chunk.startswith(b"data: "):
                    payload = chunk[6:].strip()
                    if payload and payload != b"[DONE]":
                        try:
                            ev = json.loads(payload)
                        except Exception:
                            continue
                        for ch in ev.get("choices") or []:
                            d = ch.get("delta") or {}
                            text.append(d.get("content") or d.get("reasoning_content") or "")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            got = "".join(text)
        else:
            payload = resp.read()
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            try:
                doc = json.loads(payload.decode("utf-8"))
                msg = ((doc.get("choices") or [{}])[0].get("message") or {})
                got = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
                if msg.get("tool_calls"):
                    got += " " + json.dumps(msg["tool_calls"], ensure_ascii=False)
            except Exception:
                got = ""
        note(f"reply {len(got)} chars in {time.time() - started:.1f}s: {got[:300]!r}")


def main():
    OUT.write_text("", encoding="utf-8")
    note(f"capture proxy: http://127.0.0.1:{LISTEN}/v1 -> :{TARGET}   (record in {OUT})")
    ThreadingHTTPServer(("127.0.0.1", LISTEN), Handler).serve_forever()


if __name__ == "__main__":
    main()
