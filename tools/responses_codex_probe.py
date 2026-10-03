"""Run the pinned real Codex client against Strata with labelled synthetic model output.

Only the client reads/edits/verifies example.txt in a disposable workspace. This
tests the actual Codex protocol/tool loop, not model quality or native inference.
No existing client home, credentials, repo, deployment key or remote host is used.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.responses_capture import EXPECTED_SHA256, EXPECTED_VERSION, PROMPT
from tools.responses_sdk_probe import fixture_server
from serve.server import MockEngine


def command(text):
    return ("Use the requested local file operation.</think><tool_call><function=exec_command>"
            f"<parameter=cmd>{text}</parameter><parameter=max_output_tokens>300</parameter>"
            "<parameter=workdir>WORKSPACE_PATH</parameter><parameter=login>false</parameter>"
            "</function></tool_call>")


SCRIPTS = [command("Get-Content -LiteralPath 'TARGET_PATH' -Raw"), "The model planned to inspect the disposable file.",
           command("Set-Content -LiteralPath 'TARGET_PATH' -Value after -Encoding utf8"), "The model planned the requested line replacement.",
           command("Get-Content -LiteralPath 'TARGET_PATH' -Raw"), "The model planned to verify the changed file.",
           "The client verified the new line.</think>Changed example.txt from before to after and verified it.",
           "The model used the client's verification result to report completion."]


@contextlib.contextmanager
def disposable_workspace(runtime):
    # Python 3.13's Windows TemporaryDirectory uses a private 0700 ACL that
    # prevents the restricted sandbox token from traversing it. A normal
    # workspace directory inherits the workspace ACL, as a user's checkout does.
    path = runtime / ("codex-workspace-" + uuid.uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        resolved = path.resolve()
        if resolved.parent != runtime.resolve() or not resolved.name.startswith("codex-workspace-"):
            raise RuntimeError("refusing cleanup outside the named disposable workspace")
        shutil.rmtree(resolved)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    runtime = ROOT / ".responses-runtime"
    binary = runtime / "codex-0.160.0.exe"
    if hashlib.sha256(binary.read_bytes()).hexdigest() != EXPECTED_SHA256:
        ap.error("Codex differs from the R0 pin")
    version = subprocess.check_output([str(binary), "--version"], text=True).strip()
    if version != EXPECTED_VERSION:
        ap.error("Codex version differs from the R0 pin")
    with tempfile.TemporaryDirectory(prefix="responses-codex-", dir=runtime) as temporary, \
         disposable_workspace(runtime) as workspace, fixture_server(SCRIPTS) as (svc, base):
        # ByteTokenizer needs one token per byte; this is not a native window claim.
        svc.engine.max_context = 262144
        svc.api_key = ""  # loopback-only, fresh credential-free client
        sandbox = Path(temporary)
        client_home = sandbox / "client"
        client_home.mkdir()
        target = workspace / "example.txt"
        target.write_text("before\n", encoding="utf-8")
        scripts = [script.replace("WORKSPACE_PATH", str(workspace)).replace("TARGET_PATH", str(target).replace("'", "''"))
                   for script in SCRIPTS]
        svc.engine = MockEngine(svc.tok, scripts, max_context=262144)
        profile = f'''model = "qwen3.8-flash-next"
model_provider = "strata-local"
web_search = "disabled"

[windows]
sandbox = "unelevated"

[model_providers.strata-local]
name = "Strata local protocol test"
base_url = "{base}"
wire_api = "responses"
requires_openai_auth = false
supports_websockets = false
request_max_retries = 0
stream_max_retries = 0
'''
        (client_home / "probe.config.toml").write_text(profile, encoding="utf-8")
        env = {k: v for k, v in os.environ.items() if not any(s in k.upper() for s in
               ("TOKEN", "API_KEY", "SECRET", "CODEX", "OPENAI", "ANTHROPIC"))}
        env.update(CODEX_HOME=str(client_home), HTTP_PROXY=base.removesuffix("/v1"),
                   HTTPS_PROXY=base.removesuffix("/v1"), ALL_PROXY=base.removesuffix("/v1"), NO_PROXY="127.0.0.1,localhost")
        argv = [str(binary), "--no-daemon", "--ask-for-approval", "never", "exec", "--ignore-rules", "--strict-config",
                "--profile", "probe", "--ephemeral", "--skip-git-repo-check", "--sandbox", "workspace-write",
                "--cd", str(workspace), "--color", "never", "--json", PROMPT]
        timed_out = False
        try:
            run = subprocess.run(argv, cwd=workspace, env=env, capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", timeout=120)
            stdout, stderr, code = run.stdout, run.stderr, run.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = (exc.stdout or b"").decode("utf-8", errors="replace")
            stderr = (exc.stderr or b"").decode("utf-8", errors="replace")
            code, timed_out = -1, True
        changed = target.read_text(encoding="utf-8-sig").strip() == "after"
        ids = {}

        def sanitize(text):
            for path, replacement in ((str(workspace), "<DISPOSABLE_WORKSPACE>"), (str(sandbox), "<DISPOSABLE>"), (str(ROOT), "<WORKTREE>"),
                                      (str(Path.home()), "<USER_HOME>"), (base, "http://127.0.0.1:<PORT>/v1")):
                spellings = [path, path.replace("\\", "/")]
                for _ in range(4):
                    spellings.append(json.dumps(spellings[-1] if len(spellings) > 2 else path)[1:-1])
                for spelling in sorted(set(spellings), key=len, reverse=True):
                    text = text.replace(spelling, replacement)
            return re.sub(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}",
                          lambda m: ids.setdefault(m[0], f"<CLIENT_ID_{len(ids) + 1}>"), text)

        exchanges = [{"request": json.loads(row["input"]), "response": json.loads(row["response"]) if row.get("response") else None,
                      "state": row["state"], "error": row.get("error")}
                     for row in svc.api_requests]
        results = [item for row in exchanges for item in row["request"].get("input", [])
                   if isinstance(item, dict) and item.get("type") == "function_call_output"]
        unique_results = {item["call_id"]: item["output"] for item in results}
        tool_errors = [key for key, result in unique_results.items() if "Process exited with code 0" not in result]
        passed = code == 0 and changed and len(exchanges) == 4 and len(unique_results) == 3 and not tool_errors
        receipt = {"result": "pass" if passed else "blocked", "client": version, "binary_sha256": EXPECTED_SHA256,
                   "transport": "real loopback HTTP/SSE", "server": "Strata Service + MockEngine",
                   "model_output_provenance": "synthetic scripts in tools/responses_codex_probe.py",
                   "native_inference_test": False, "universal_compatibility": False, "return_code": code,
                   "timed_out": timed_out, "requests": len(exchanges), "tool_results_in_replayed_history": len(results),
                   "unique_tool_results": len(unique_results), "tool_error_call_ids": tool_errors,
                   "protocol_loop_completed": code == 0 and len(exchanges) == 4 and all(r["state"] == "completed" for r in exchanges),
                   "client_changed_file": changed, "server_executes_client_tools": False,
                   "final_file": "after" if changed else "unchanged", "argv": argv[1:]}
        for name, data in (("codex-exchanges.json", exchanges), ("codex-task.json", receipt)):
            (a.out / name).write_text(sanitize(json.dumps(data, ensure_ascii=False, indent=2)) + "\n", encoding="utf-8")
        (a.out / "codex-task.txt").write_text(sanitize(stdout + stderr), encoding="utf-8")
        (a.out / "codex-profile.toml").write_text(profile.replace(base, "http://127.0.0.1:8095/v1"), encoding="utf-8")
        print(json.dumps({k: v for k, v in receipt.items() if k != "argv"}, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
