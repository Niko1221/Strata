#!/usr/bin/env python3
"""tools/busbar_gateway.py - one OpenAI-compatible address in front of several model providers.

A pool is a name that stands for several models. `fast` can mean 80% Claude and 20% GPT, with Bedrock answering
when both are down. A client changes one line:

    client = OpenAI(api_key=BUSBAR_TOKEN, base_url="http://localhost:8080/v1")
    client.chat.completions.create(model="fast", messages=[{"role": "user", "content": "Hi"}])

The split is exact, not random: the smooth weighted round robin (the one nginx uses) gives 80 out of 100 requests
to Claude and 20 to GPT, spread evenly instead of in clumps. A member that fails is put on cooldown and skipped
until the cooldown ends; the request is retried on the next member of the pool, then on the pools named in
`failover`.

Standard library only. Nothing here needs a GPU, a model, or a provider key on this machine to run the tests.

    python tools/busbar_gateway.py --config busbar.json --port 8080
    python tools/busbar_gateway.py --config busbar.json --plan        # print the routing, send nothing
    python tools/busbar_gateway.py --config busbar.json --plan --requests 20

Config (see tools/busbar.example.json):

    {
      "default_pool": "fast",
      "pools": {
        "fast": {
          "members": [
            {"name": "claude", "weight": 80, "wire": "anthropic",
             "base_url": "https://api.anthropic.com/v1", "model": "claude-sonnet-4-5",
             "key_env": "ANTHROPIC_API_KEY"},
            {"name": "gpt", "weight": 20, "wire": "openai",
             "base_url": "https://api.openai.com/v1", "model": "gpt-5.2", "key_env": "OPENAI_KEY"}
          ],
          "failover": ["bedrock"],
          "timeout_s": 120, "cooldown_s": 30, "max_fails": 2
        },
        "bedrock": {"members": [{"name": "bedrock", "weight": 1, "wire": "openai",
                                 "base_url": "https://bedrock-runtime.us-east-1.amazonaws.com/model",
                                 "model": "anthropic.claude-sonnet-4-5", "key_env": "AWS_BEARER_TOKEN_BEDROCK"}]}
      }
    }
"""
from __future__ import annotations

import argparse
import hmac
import http.client
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEFAULT_TIMEOUT = 120.0
PASSTHROUGH_KEYS = ("max_tokens", "temperature", "top_p", "top_k", "min_p", "presence_penalty",
                    "frequency_penalty", "stop", "response_format", "seed", "tools", "tool_choice",
                    "reasoning_effort")


class Member:
    """One upstream model: where it lives, which API it speaks, how often it gets a request."""

    def __init__(self, name: str, weight: float, base_url: str, model: str, key: str, wire: str,
                 headers: dict | None = None):
        if weight <= 0:
            raise ValueError(f"member {name!r}: weight must be more than 0, got {weight}")
        self.name, self.weight, self.base_url, self.model, self.key, self.wire = (
            name, float(weight), base_url.rstrip("/"), model, key, wire)
        self.headers = dict(headers or {})
        if wire not in ("openai", "anthropic"):
            raise ValueError(f"member {name!r}: wire must be 'openai' or 'anthropic', got {wire!r}")

    def request(self, body: dict, stream: bool) -> tuple[str, bytes, dict]:
        """The URL, the payload and the headers for one upstream call."""
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages: expected a non-empty list")
        if self.wire == "openai":
            payload = {"model": self.model, "messages": messages, "stream": stream}
            for key in PASSTHROUGH_KEYS:
                if body.get(key) is not None:
                    payload[key] = body[key]
            url = self.base_url + "/chat/completions"
            headers = {"Content-Type": "application/json"}
            if self.key:
                headers["Authorization"] = f"Bearer {self.key}"
        else:
            system = ""
            chat = []
            for m in messages:
                if m.get("role") == "system":
                    system += (" " + str(m.get("content", ""))).strip()
                else:
                    chat.append({"role": m.get("role", "user"), "content": m.get("content", "")})
            payload = {"model": self.model, "messages": chat, "stream": stream,
                       "max_tokens": int(body.get("max_tokens") or 1024)}
            if system:
                payload["system"] = system
            for key in ("temperature", "top_p", "stop_sequences"):
                if body.get(key) is not None:
                    payload[key] = body[key]
            url = self.base_url + "/messages"
            headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
            if self.key:
                headers["x-api-key"] = self.key
        headers.update(self.headers)
        return url, json.dumps(payload).encode(), headers


class Pool:
    """A named group of members, with the smooth weighted round robin and a cooldown list."""

    def __init__(self, name: str, spec: dict):
        self.name = name
        self.members = []
        for i, m in enumerate(spec.get("members") or []):
            self.members.append(Member(
                name=m.get("name") or f"member{i}",
                weight=m.get("weight", 1),
                base_url=m.get("base_url") or "",
                model=m.get("model") or "",
                key=m.get("key") or os.environ.get(m.get("key_env", ""), ""),
                wire=m.get("wire", "openai"),
                headers=m.get("headers"),
            ))
        if not self.members:
            raise ValueError(f"pool {name!r}: no members")
        if not all(m.base_url and m.model for m in self.members):
            raise ValueError(f"pool {name!r}: every member needs base_url and model")
        self.weight = {m.name: m.weight for m in self.members}
        self.current = {m.name: 0.0 for m in self.members}
        self.timeout = float(spec.get("timeout_s", DEFAULT_TIMEOUT))
        self.cooldown_s = float(spec.get("cooldown_s", 30.0))
        self.max_fails = int(spec.get("max_fails", 2))
        self.fails = {m.name: 0 for m in self.members}   # consecutive failures; one success resets it
        self.until = {m.name: 0.0 for m in self.members}  # time a member is put aside until
        self.failover_names = list(spec.get("failover") or [])
        self.failover: list[Pool] = []
        self.lock = threading.Lock()
        self.counts = {m.name: 0 for m in self.members}
        self.errors = {m.name: 0 for m in self.members}
        self.tokens = {m.name: 0 for m in self.members}

    def _open(self, now: float) -> list[Member]:
        ready = [m for m in self.members if self.until[m.name] <= now]
        return ready or list(self.members)      # all cooling: give the first one another try

    def open_members(self, now: float) -> list[Member]:
        """The same list read under the pool's lock: a failover pool is looked at from another pool's pick."""
        with self.lock:
            return self._open(now)

    def pick(self) -> list[tuple[Pool, Member]]:
        """The order to try: the member this request is owed, then the rest by debt, then the failover pools.
        Each entry carries the pool it belongs to, so a request answered by a failover member is counted there."""
        with self.lock:
            ready = self._open(time.time())
            total = sum(self.weight[m.name] for m in ready)
            for m in ready:
                self.current[m.name] += self.weight[m.name]
            best = max(ready, key=lambda m: (self.current[m.name], -self.members.index(m)))
            self.current[best.name] -= total
            rest = sorted((m for m in ready if m is not best),
                          key=lambda m: (-self.current[m.name], self.members.index(m)))
            order = [(self, m) for m in [best, *rest]]
        # Failover pools come last, heaviest first. Their own counters stay untouched: a request that never
        # reached them must not move their round robin, and a member on cooldown there stays on cooldown.
        now = time.time()
        for pool in self.failover:
            order.extend((pool, m) for m in sorted(pool.open_members(now), key=lambda m: (-m.weight, m.name)))
        return order

    def note_ok(self, member: Member, tokens: int) -> None:
        with self.lock:
            self.counts[member.name] += 1
            self.fails[member.name] = 0
            self.until[member.name] = 0.0
            self.tokens[member.name] += tokens

    def note_fail(self, member: Member) -> None:
        with self.lock:
            self.errors[member.name] += 1
            self.fails[member.name] += 1
            if self.fails[member.name] >= self.max_fails:
                self.until[member.name] = time.time() + self.cooldown_s

    def stats(self) -> dict:
        with self.lock:
            return {"pool": self.name,
                    "members": [{"name": m.name, "weight": m.weight, "model": m.model,
                                "ok": self.counts[m.name], "failed": self.errors[m.name],
                                "tokens": self.tokens[m.name],
                                "cooling_until": round(max(0.0, self.until[m.name] - time.time()), 1)}
                               for m in self.members]}


class Registry:
    """The pools, wired to their failover pools, and the name each request is looked up by."""

    def __init__(self, cfg: dict):
        pools = cfg.get("pools") or {}
        if not pools:
            raise ValueError("config: no pools")
        self.pools = {name: Pool(name, spec) for name, spec in pools.items()}
        for name, pool in self.pools.items():
            for other in pool.failover_names:
                if other not in self.pools:
                    raise ValueError(f"pool {name!r}: failover pool {other!r} is not defined")
                pool.failover.append(self.pools[other])
        self.default_pool = cfg.get("default_pool")
        if self.default_pool and self.default_pool not in self.pools:
            raise ValueError(f"default_pool {self.default_pool!r} is not defined")

    def resolve(self, model: str | None) -> Pool:
        """The pool named by the request; with no name at all, the default pool. A name that is not a pool is an
        error rather than a silent fallback, so a typo does not quietly send traffic somewhere else."""
        if not model:
            if self.default_pool:
                return self.pools[self.default_pool]
            raise ValueError(f"no model given and no default_pool; known pools: {', '.join(self.pools)}")
        if model in self.pools:
            return self.pools[model]
        raise ValueError(f"no pool named {model!r}; known pools: {', '.join(self.pools)}")

    def plan(self, pool_name: str, requests: int) -> list[str]:
        """Which member each of the next `requests` requests to this pool would get, without sending anything."""
        pool = self.resolve(pool_name)
        return [pool.pick()[0][1].name for _ in range(requests)]


def usage_tokens(body: dict) -> int:
    u = body.get("usage") or {}
    return int(u.get("total_tokens") or u.get("completion_tokens") or u.get("output_tokens") or 0)


def openai_from_anthropic(body: dict, model: str) -> dict:
    text = "".join(part.get("text", "") for part in body.get("content", []) if isinstance(part, dict))
    usage = body.get("usage") or {}
    return {"object": "chat.completion", "model": model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": usage.get("input_tokens", 0),
                      "completion_tokens": usage.get("output_tokens", 0),
                      "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)}}


def call(member: Member, body: dict, timeout: float):
    """Send one request and return the open response; the caller reads it, whole or line by line.

    An upstream that aborts the socket before a status line arrives raises the raw OSError here (on
    Windows, WinError 10053), so the callers catch OSError rather than only urllib's own errors.
    """
    url, payload, headers = member.request(body, bool(body.get("stream")))
    req = urllib.request.Request(url, data=payload, headers=headers)
    return urllib.request.urlopen(req, timeout=timeout)


class Gateway:
    def __init__(self, registry: Registry, token: str = ""):
        self.registry = registry
        self.token = token

    def authorized(self, header: str) -> bool:
        if not self.token:
            return True
        given = (header or "").replace("Bearer ", "").strip()
        return hmac.compare_digest(given, self.token)

    def complete(self, body: dict) -> tuple[int, dict]:
        """Non-streaming: try members in order until one answers."""
        pool = self.registry.resolve(body.get("model"))
        last_error = "no member answered"
        for owner, member in pool.pick():
            try:
                with call(member, body, owner.timeout) as r:
                    data = json.loads(r.read().decode("utf-8", "replace") or "{}")
                if member.wire == "anthropic":
                    data = openai_from_anthropic(data, member.model)
                owner.note_ok(member, usage_tokens(data))
                return 200, data
            except (OSError, http.client.HTTPException, ValueError) as exc:
                owner.note_fail(member)
                last_error = f"{member.name}: {exc}"
        return 502, {"error": {"type": "busbar_error",
                               "message": f"every member of pool {pool.name} failed; last: {last_error}"}}

    def stream(self, handler, body: dict) -> None:
        """Streaming: pass the upstream SSE through, converting the Anthropic events to OpenAI chunks."""
        pool = self.registry.resolve(body.get("model"))
        for owner, member in pool.pick():
            try:
                resp = call(member, body, owner.timeout)
            except (OSError, http.client.HTTPException, ValueError) as exc:
                owner.note_fail(member)
                continue
            handler.send_response(200)
            handler.send_header("Content-Type", "text/event-stream")
            handler.send_header("Cache-Control", "no-cache")
            handler.end_headers()
            tokens = 0
            upstream_broke = False     # the provider cut the stream before it ended

            def emit(chunk: bytes) -> bool:
                try:
                    handler.wfile.write(chunk)
                    handler.wfile.flush()
                    return True
                except OSError:
                    return False

            try:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if member.wire == "openai":
                        if not emit(raw):
                            break          # the client stopped reading: the member did answer
                        if line.startswith("data:") and line[5:].strip() not in ("[DONE]", ""):
                            try:
                                for ch in json.loads(line[5:].strip()).get("choices", []):
                                    tokens += len((ch.get("delta") or {}).get("content") or "")
                            except ValueError:
                                pass
                    else:
                        if not line.startswith("data:"):
                            continue
                        try:
                            ev = json.loads(line[5:].strip())
                        except ValueError:
                            continue
                        if ev.get("type") == "content_block_delta":
                            text = (ev.get("delta") or {}).get("text") or ""
                            tokens += len(text)
                            if not emit(b"data: " + json.dumps(
                                    {"object": "chat.completion.chunk", "model": member.model,
                                     "choices": [{"index": 0, "delta": {"content": text}}]}).encode() + b"\n\n"):
                                break          # the client stopped reading: the member did answer
                        elif ev.get("type") == "message_stop":
                            if not emit(b"data: [DONE]\n\n"):
                                break          # the client stopped reading: the member did answer
            except (OSError, http.client.HTTPException):
                upstream_broke = True
            finally:
                resp.close()
            if upstream_broke:
                owner.note_fail(member)     # a stream cut short is a failure, not an answered request
            else:
                owner.note_ok(member, tokens)
            return
        handler.send_response(502)
        handler.send_header("Content-Type", "application/json")
        handler.end_headers()
        handler.wfile.write(json.dumps({"error": {"type": "busbar_error",
                                                  "message": f"every member of pool {pool.name} failed"}}).encode())


def make_handler(gateway: Gateway):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            sys.stderr.write(f"[busbar] {self.address_string()} {fmt % args}\n")

        def _path(self) -> str:
            p = self.path.split("?")[0]
            if p.startswith("/v1"):
                p = p[3:]
            return p or "/"

        def do_GET(self):
            if not gateway.authorized(self.headers.get("Authorization", "")):
                self._json(401, {"error": {"type": "unauthorized", "message": "missing or wrong API key"}})
                return
            path = self._path()
            if path in ("/", "/health"):
                self._json(200, {"ok": True, "pools": list(gateway.registry.pools)})
            elif path == "/models":
                self._json(200, {"object": "list",
                                 "data": [{"id": name, "object": "model",
                                          "members": [m.name for m in pool.members]}
                                         for name, pool in gateway.registry.pools.items()]})
            elif path == "/stats":
                self._json(200, {"pools": [pool.stats() for pool in gateway.registry.pools.values()]})
            else:
                self._json(404, {"error": {"type": "not_found", "message": f"no route {self.path}"}})

        def do_POST(self):
            if not gateway.authorized(self.headers.get("Authorization", "")):
                self._json(401, {"error": {"type": "unauthorized", "message": "missing or wrong API key"}})
                return
            if self._path() != "/chat/completions":
                self._json(404, {"error": {"type": "not_found", "message": f"no route {self.path}"}})
                return
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                self._json(400, {"error": {"type": "bad_request", "message": "expected JSON"}})
                return
            if not isinstance(body, dict):
                self._json(400, {"error": {"type": "bad_request", "message": "expected an object"}})
                return
            messages = body.get("messages")
            if not isinstance(messages, list) or not messages:
                self._json(400, {"error": {"type": "bad_request", "message": "messages: expected a non-empty list"}})
                return
            try:
                if body.get("stream"):
                    gateway.stream(self, body)
                else:
                    self._json(*gateway.complete(body))
            except ValueError as exc:
                # a pool name that does not exist, or a body with no messages: answer 400 rather than
                # letting the exception kill the connection
                self._json(400, {"error": {"type": "bad_request", "message": str(exc)}})

        def _json(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="OpenAI-compatible gateway in front of several model providers.")
    ap.add_argument("--config", required=True, help="JSON config with the pools (see tools/busbar.example.json)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--token", default=os.environ.get("BUSBAR_TOKEN", ""),
                    help="require this key on every request (empty: no key needed, local only)")
    ap.add_argument("--plan", action="store_true", help="print how the next requests are routed, then exit")
    ap.add_argument("--pool", help="which pool --plan shows (default: the first one)")
    ap.add_argument("--requests", type=int, default=100, help="how many requests --plan shows")
    args = ap.parse_args(argv)

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    registry = Registry(cfg)

    if args.plan:
        for name, pool in registry.pools.items():
            print(f"pool {name}: " + ", ".join(f"{m.name} {m.weight:g}" for m in pool.members) +
                  (f"  failover: {', '.join(pool.failover_names)}" if pool.failover_names else ""))
        target = args.pool or next(iter(registry.pools))
        try:
            order = registry.plan(target, args.requests)
        except ValueError as exc:
            print(f"[busbar] {exc}", file=sys.stderr)
            return 2
        tally = {}
        for name in order:
            tally[name] = tally.get(name, 0) + 1
        print(f"next {args.requests} requests to {target}: " + " ".join(order))
        print("counts: " + ", ".join(f"{k}={v}" for k, v in sorted(tally.items(), key=lambda x: -x[1])))
        return 0

    server = ThreadingHTTPServer((args.host, args.port), make_handler(Gateway(registry, args.token)))
    print(f"[busbar] listening on http://{args.host}:{args.port}/v1  pools: {', '.join(registry.pools)}", flush=True)
    if not args.token:
        print("[busbar] no key set: keep this on 127.0.0.1 only", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[busbar] stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
