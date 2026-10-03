"""Run pinned local Codex against an existing, authenticated Strata Responses server.

No model output is scripted. Server startup/native provenance is recorded
separately. This hello-world probe requires the existing API monitor for evidence;
it does not add a server endpoint, change a client request, or download anything.
"""
from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request
from urllib.parse import urlsplit
import uuid

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA256 = "fdda5fa3cf3fb3d000b876720742857676293e4315e4b045fae6f8bd7e866d1d"
EXPECTED_VERSION = "codex-cli 0.160.0"
PROMPT = 'Reply with exactly "Hello world". Do not use any tools.'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--codex", type=Path, required=True)
    ap.add_argument("--base-url", required=True, help="loopback /v1 endpoint, optionally an authenticated SSH tunnel")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args()
    base = a.base_url.rstrip("/")
    parsed = urlsplit(base)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.path != "/v1":
        ap.error("use an HTTP loopback /v1 endpoint; keep LAN transport in an authenticated tunnel")
    key = os.environ.get("STRATA_API_KEY")
    if not key:
        ap.error("set STRATA_API_KEY in the environment; keys are never written to evidence")
    binary = a.codex.resolve()
    if hashlib.sha256(binary.read_bytes()).hexdigest() != EXPECTED_SHA256:
        ap.error("Codex differs from the recorded R0 binary pin")
    version = subprocess.check_output([str(binary), "--version"], text=True).strip()
    if version != EXPECTED_VERSION:
        ap.error("Codex differs from the recorded R0 version")
    a.out.mkdir(parents=True, exist_ok=False)
    origin = base.removesuffix("/v1")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def get(path):
        req = urllib.request.Request(origin + path, headers={"Authorization": "Bearer " + key})
        with opener.open(req, timeout=15) as response:
            return json.load(response)

    health, models = get("/health"), get("/v1/models")
    assert health["service"] == "strata" and health["loaded"] and health["api_key"], health
    assert any(m["id"] == "qwen3.8-flash-next" for m in models["data"]), models
    before = {r["id"] for r in get("/api/requests")["requests"]}
    denied = []

    class RejectProxy(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reject(self):
            # No authorization headers or request bodies are recorded.
            denied.append({"method": self.command, "destination": self.path})
            self.send_response(502)
            self.send_header("Content-Length", "0")
            self.end_headers()

        do_CONNECT = do_GET = do_POST = do_PUT = do_DELETE = do_OPTIONS = reject

    proxy = ThreadingHTTPServer(("127.0.0.1", 0), RejectProxy)
    proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()
    runtime = ROOT / ".responses-runtime"
    runtime.mkdir(exist_ok=True)
    workspace = runtime / ("native-codex-workspace-" + uuid.uuid4().hex)
    workspace.mkdir()  # inherit workspace ACL for the Windows restricted token
    try:
        with tempfile.TemporaryDirectory(prefix="native-codex-", dir=runtime) as temporary:
            client_home = Path(temporary) / "client"
            client_home.mkdir()
            profile = f'''model = "qwen3.8-flash-next"
model_provider = "strata-local"
web_search = "disabled"

[windows]
sandbox = "unelevated"

[model_providers.strata-local]
name = "Strata local native inference"
base_url = "{base}"
env_key = "STRATA_API_KEY"
wire_api = "responses"
requires_openai_auth = false
supports_websockets = false
request_max_retries = 0
stream_max_retries = 0
'''
            (client_home / "strata.config.toml").write_text(profile, encoding="utf-8")
            env = {k: v for k, v in os.environ.items() if not any(s in k.upper() for s in
                   ("TOKEN", "API_KEY", "SECRET", "CODEX", "OPENAI", "ANTHROPIC", "PROXY"))}
            proxy_url = f"http://127.0.0.1:{proxy.server_port}"
            env.update(CODEX_HOME=str(client_home), STRATA_API_KEY=key)
            for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
                env[name] = env[name.lower()] = proxy_url
            env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost"
            argv = [str(binary), "--no-daemon", "--ask-for-approval", "never", "exec", "--ignore-rules",
                    "--strict-config", "--profile", "strata", "--ephemeral", "--skip-git-repo-check",
                    "--sandbox", "read-only", "--cd", str(workspace), "--color", "never", "--json", PROMPT]
            started = time.perf_counter()
            timed_out = False
            try:
                run = subprocess.run(argv, cwd=workspace, env=env, capture_output=True, text=True,
                                     encoding="utf-8", errors="replace", input="", timeout=a.timeout)
                stdout, stderr, code = run.stdout, run.stderr, run.returncode
            except subprocess.TimeoutExpired as exc:
                stdout = (exc.stdout or b"").decode("utf-8", errors="replace")
                stderr = (exc.stderr or b"").decode("utf-8", errors="replace")
                code, timed_out = -1, True
            elapsed = time.perf_counter() - started
            rows = get("/api/requests")["requests"]
            exchanges = [get("/api/requests?id=" + row["id"]) for row in reversed(rows) if row["id"] not in before]
            events = [json.loads(line) for line in stdout.splitlines() if line.startswith("{")]
            final = [e["item"]["text"] for e in events if e.get("type") == "item.completed"
                     and e.get("item", {}).get("type") == "agent_message"]
            completed = any(e.get("type") == "turn.completed" for e in events)
            responses = [json.loads(row["response"]) for row in exchanges if row.get("response")]
            output_text = [part["text"] for response in responses for item in response["output"]
                           if item["type"] == "message" for part in item["content"] if part["type"] == "output_text"]
            passed = (code == 0 and not timed_out and completed and final == ["Hello world"]
                      and output_text == final and len(exchanges) == 1 and len(responses) == 1
                      and responses[0]["status"] == "completed" and exchanges[0]["path"] == "/v1/responses"
                      and not any(row.get("input_truncated") or row.get("response_truncated") for row in exchanges))
            receipt = {"result": "pass" if passed else "fail", "client": version, "codex_sha256": EXPECTED_SHA256,
                       "transport": "HTTP/SSE over loopback endpoint", "model_output_scripted": False,
                       "native_provenance": "see separately recorded server process, binary and engine log",
                       "prompt": PROMPT, "return_code": code, "timed_out": timed_out, "elapsed_s": elapsed,
                       "codex_turn_completed": completed, "codex_final_messages": final,
                       "responses_output_text": output_text, "requests": len(exchanges), "health": health,
                       "external_http_proxy_attempts_denied": denied, "web_search": "disabled",
                       "configuration": "fresh credential-free Codex home plus this server's environment key",
                       "universal_compatibility": False, "argv": argv[1:]}

            def sanitize(text):
                text = text.replace(key, "<API_KEY_REMOVED>")
                for path, replacement in ((str(workspace), "<DISPOSABLE_WORKSPACE>"),
                        (str(client_home.parent), "<DISPOSABLE_CLIENT_HOME>"), (str(ROOT), "<WORKTREE>"),
                        (str(Path.home()), "<USER_HOME>"), (base, "http://127.0.0.1:<PORT>/v1")):
                    spellings = {path, path.replace("\\", "/")}
                    for _ in range(4):
                        spellings |= {json.dumps(s)[1:-1] for s in tuple(spellings)}
                    for spelling in sorted(spellings, key=len, reverse=True):
                        text = text.replace(spelling, replacement)
                return text

            for name, data in (("result.json", receipt), ("exchanges.json", exchanges), ("metrics.json", get("/metrics"))):
                (a.out / name).write_text(sanitize(json.dumps(data, indent=2, ensure_ascii=False)) + "\n", encoding="utf-8")
            (a.out / "codex-output.txt").write_text(sanitize(stdout + stderr), encoding="utf-8")
            (a.out / "codex-profile.toml").write_text(profile.replace(base, "http://127.0.0.1:8095/v1"), encoding="utf-8")
            print(json.dumps({k: v for k, v in receipt.items() if k != "argv"}, indent=2), flush=True)
    finally:
        proxy.shutdown()
        proxy.server_close()
        resolved = workspace.resolve()
        if resolved.parent != runtime.resolve() or not resolved.name.startswith("native-codex-workspace-"):
            raise RuntimeError("refusing cleanup outside the disposable workspace")
        shutil.rmtree(resolved)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
