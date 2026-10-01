"""Exercise native Responses with an installed Codex CLI and scripted local inference.

No model weights or OpenAI API access required. Codex runs only in temporary directories.
    python -m tools.codex_responses_smoke
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading

from serve.frontend import ChatTemplate
from serve.responses import normalize
from serve.server import ByteTokenizer, MockEngine, Server, Service, make_handler

ROOT = Path(__file__).resolve().parents[1]


def xml_call(name, arguments):
    parts = [f"<tool_call>\n<function={name}>"]
    for key, value in arguments.items():
        text = value if isinstance(value, str) else json.dumps(value)
        parts.append(f"<parameter={key}>\n{text}\n</parameter>")
    return "\n".join(parts) + "\n</function>\n</tool_call>"


def smoke(codex, scenario):
    tok = ByteTokenizer()
    engine = MockEngine(tok, "</think>\n\nSmoke passed.", max_context=262144)
    # A known model slug makes Codex advertise its native custom patch tool. All inference still
    # goes to this mock server; unknown model slugs use Codex's function-only fallback metadata.
    model = "gpt-5.5" if scenario == "custom" else "strata-test"
    svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"), model_name=model)
    received = []
    errors = []
    base_handler = make_handler(svc)

    class Handler(base_handler):
        def _responses(self, body):
            received.append(body)
            try:
                req = normalize(body)
                script = "</think>\n\nSmoke passed."
                if scenario != "text" and len(received) == 1:
                    wanted = "custom" if scenario == "custom" else "function"
                    names = ("apply_patch",) if scenario == "custom" else ("exec_command", "shell_command", "shell")
                    tools = [t for t in req.registry.values() if t.kind == wanted and t.name in names]
                    if not tools:
                        offered = [(t.namespace, t.name, t.kind) for t in req.registry.values()]
                        raise ValueError(f"Codex did not offer {wanted} tool: {offered}")
                    tool = tools[0]
                    if scenario == "custom":
                        args = {"input": "*** Begin Patch\n*** Add File: smoke.txt\n+codex-patch-ok\n*** End Patch"}
                    else:
                        props = tool.template["parameters"].get("properties", {})
                        if "cmd" in props:
                            args = {"cmd": "printf 'codex-function-ok\\n'"}
                        elif "command" in props:
                            command = "printf 'codex-function-ok\\n'"
                            args = {"command": ["bash", "-lc", command] if props["command"].get("type") == "array" else command}
                        else:
                            raise ValueError(f"unsupported shell schema: {list(props)}")
                    script = "</think>\n\n" + xml_call(tool.internal, args)
                engine.script = tok.encode(script, parse_special=True) + tok.encode("<|im_end|>", parse_special=True)
                return super()._responses(body)
            except Exception as exc:
                errors.append(str(exc))
                raise

    httpd = Server(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    env = dict(os.environ)
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(key, None)
    try:
        with tempfile.TemporaryDirectory(prefix="strata-codex-smoke-") as directory:
            args = [codex, "exec", "--ignore-user-config", "--ephemeral", "--skip-git-repo-check", "--json",
                    "--sandbox", "workspace-write", "-C", directory]
            config = {"model_provider": "strata", "model": model, "web_search": "disabled",
                      "features.apps": False, "features.multi_agent": False,
                      "model_reasoning_effort": "low", "model_reasoning_summary": "none",
                      "model_context_window": 262144,
                      "model_providers.strata.name": "Strata", "model_providers.strata.base_url": url,
                      "model_providers.strata.wire_api": "responses", "model_providers.strata.requires_openai_auth": False,
                      "model_providers.strata.supports_websockets": False}
            for key, value in config.items():
                args += ["-c", key + "=" + (str(value).lower() if isinstance(value, bool) else json.dumps(value))]
            args.append("This is a local protocol smoke test. Follow the scripted model output and finish.")
            result = subprocess.run(args, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=45)
            if result.returncode or errors:
                raise RuntimeError(f"{scenario}: exit={result.returncode}; errors={errors}\n{result.stdout[-5000:]}\n{result.stderr[-3000:]}")
            output = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
            if not any(e.get("type") == "turn.completed" for e in output):
                raise RuntimeError(f"{scenario}: no turn.completed event: {result.stdout[-3000:]}")
            if scenario != "text":
                if len(received) < 2:
                    raise RuntimeError(f"{scenario}: Codex did not return tool results")
                kind = "custom_tool_call_output" if scenario == "custom" else "function_call_output"
                results = [i for i in received[-1]["input"] if i.get("type") == kind]
                if not results:
                    raise RuntimeError(f"{scenario}: no {kind} history item")
                if scenario == "custom":
                    if (Path(directory) / "smoke.txt").read_text() != "codex-patch-ok\n":
                        raise RuntimeError("custom patch did not create the expected file")
                elif "codex-function-ok" not in json.dumps(results):
                    raise RuntimeError("function result did not contain the expected output")
            print(f"{scenario}: passed ({len(received)} Responses requests)")
    finally:
        httpd.shutdown()
        httpd.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codex", default=shutil.which("codex"))
    parser.add_argument("--scenario", choices=("text", "function", "custom", "all"), default="all")
    args = parser.parse_args()
    if not args.codex:
        parser.error("Codex CLI is not installed")
    for scenario in (("text", "function", "custom") if args.scenario == "all" else (args.scenario,)):
        smoke(args.codex, scenario)


if __name__ == "__main__":
    main()
