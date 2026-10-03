"""Capture a pinned Codex client's first HTTP request on loopback; never run a tool loop.

The stub deliberately returns HTTP 400. Its response is synthetic; the captured
request is real client traffic. No model, GPU, external provider or credentials
are used. The executable is copied into this worktree's ignored runtime folder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VERSION = "codex-cli 0.160.0"
EXPECTED_SHA256 = "fdda5fa3cf3fb3d000b876720742857676293e4315e4b045fae6f8bd7e866d1d"
PROMPT = "In this disposable fixture, inspect example.txt and change its single line from before to after."


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--codex", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    digest = hashlib.sha256(args.codex.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        ap.error(f"Codex binary differs from the reviewed R0 pin: {digest}")
    runtime = ROOT / ".responses-runtime"
    runtime.mkdir(exist_ok=True)
    pinned = runtime / "codex-0.160.0.exe"
    if not pinned.exists():
        shutil.copy2(args.codex, pinned)
    if hashlib.sha256(pinned.read_bytes()).hexdigest() != digest:
        ap.error("Pinned executable was changed")
    version = subprocess.check_output([str(pinned), "--version"], text=True).strip()
    if version != EXPECTED_VERSION:
        ap.error(f"Unexpected Codex version: {version}")
    host_source = args.codex.parent / "codex-code-mode-host.exe"
    host = runtime / host_source.name
    host_digest = hashlib.sha256(host_source.read_bytes()).hexdigest()
    if not host.exists():
        shutil.copy2(host_source, host)
    if hashlib.sha256(host.read_bytes()).hexdigest() != host_digest:
        ap.error("Pinned code-mode host differs from the installed companion binary")
    captured = []

    class Recorder(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size < 2_000_000:
                self.send_error(413)
                return
            body = json.loads(self.rfile.read(size))
            # Only safe, explicitly selected headers are retained. Never save Authorization.
            captured.append({"path": self.path, "headers": {
                key: self.headers[key] for key in ("Content-Type", "User-Agent") if key in self.headers},
                "body": body})
            data = json.dumps({"error": {"type": "invalid_request_error", "code": "r0_capture_only",
                "message": "R0 recording stub: request captured; no generation or tool execution."}}).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Recorder)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    profile = f'''model = "qwen3.8-flash-next"
model_provider = "strata-local"
web_search = "disabled"

[model_providers.strata-local]
name = "Strata loopback fixture"
base_url = "http://127.0.0.1:{port}/v1"
wire_api = "responses"
requires_openai_auth = false
supports_websockets = false
request_max_retries = 0
stream_max_retries = 0
'''
    # A fresh child-process home prevents inherited credentials, MCP servers,
    # hooks and private prompts entering the request. Parent environment is untouched.
    with tempfile.TemporaryDirectory(prefix="responses-r0-", dir=runtime) as temporary:
        sandbox = Path(temporary)
        client_home, workspace = sandbox / "client", sandbox / "workspace"
        client_home.mkdir()
        workspace.mkdir()
        (workspace / "example.txt").write_text("before\n", encoding="utf-8")
        (client_home / "capture.config.toml").write_text(profile, encoding="utf-8")
        env = {k: v for k, v in os.environ.items() if not any(s in k.upper() for s in
               ("TOKEN", "API_KEY", "SECRET", "CODEX", "OPENAI", "ANTHROPIC"))}
        env["CODEX_HOME"] = str(client_home)
        # Refuse accidental non-loopback HTTP destinations instead of falling
        # through to a real provider if client configuration loading changes.
        env.update(HTTP_PROXY=f"http://127.0.0.1:{port}", HTTPS_PROXY=f"http://127.0.0.1:{port}",
                   ALL_PROXY=f"http://127.0.0.1:{port}", NO_PROXY="127.0.0.1,localhost")
        command = [str(pinned), "exec", "--ignore-rules", "--strict-config",
                   "--profile", "capture", "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only",
                   "--cd", str(workspace), "--color", "never", "--json", PROMPT]
        try:
            result = subprocess.run(command, env=env, cwd=workspace, capture_output=True,
                                    input="", text=True, encoding="utf-8", errors="replace", timeout=45)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        client_ids = {}

        def sanitize(text):
            replacements = [(str(sandbox), "<DISPOSABLE>"), (str(ROOT), "<WORKTREE>"),
                            (str(Path.home()), "<USER_HOME>"), (str(port), "<PORT>")]
            for original, replacement in replacements:
                text = text.replace(original, replacement).replace(original.replace("\\", "/"), replacement)
                text = text.replace(json.dumps(original)[1:-1], replacement)
            def replace_id(match):
                return client_ids.setdefault(match[0], f"<CLIENT_ID_{len(client_ids) + 1}>")
            text = re.sub(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", replace_id, text)
            return text

        def clean(obj):
            if isinstance(obj, str):
                return sanitize(obj)
            if isinstance(obj, list):
                return [clean(v) for v in obj]
            if isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items()}
            return obj

        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "codex-profile.toml").write_text(profile.replace(str(port), "8095"), encoding="utf-8")
        (args.out / "codex-capture-output.txt").write_text(sanitize(result.stdout + result.stderr), encoding="utf-8")
        for i, request in enumerate(captured):
            write_json(args.out / "request-fixtures" / f"codex-initial-{i + 1}.json", clean(request))
        write_json(args.out / "client-profile.json", {
            "provenance": "real Codex first request to a loopback recording stub; no model output or tool loop",
            "codex_version": version, "codex_binary_sha256": digest,
            "pinned_binary": ".responses-runtime/codex-0.160.0.exe",
            "code_mode_host_sha256": host_digest,
            "model": "qwen3.8-flash-next", "model_catalog_override": False,
            "profile": "codex-profile.toml", "argv": clean(command[1:]),
            "return_code": result.returncode, "captured_requests": len(captured),
            "stub_response": "synthetic HTTP 400 r0_capture_only",
            "tool_loop": "not tested; R3/R4 gate",
            "example_file_changed": (workspace / "example.txt").read_text(encoding="utf-8") != "before\n",
            "sanitization": "header allowlist; fresh credential-free client home; local paths/port replaced",
            "configuration_source": "https://learn.chatgpt.com/docs/config-file/config-advanced",
        })
    if not captured:
        raise SystemExit("No request captured: inspect codex-capture-output.txt; R0 did not pass")
    print(f"Captured {len(captured)} real initial request(s); stub deliberately ended the exchange with HTTP 400.")


if __name__ == "__main__":
    main()
