#!/usr/bin/env python3
"""Run long mixed-architecture HIP prompt and same-conversation checks against a local Strata server."""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import pathlib
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import strata_tokenizer as ST  # noqa: E402


def tokenizer_from_dir(path: pathlib.Path):
    if not path.is_dir():
        raise RuntimeError(f"config tokenizer must be an extracted tokenizer directory: {path}")
    vocab = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
    tokens: list[str | None] = [None] * len(vocab)
    for token, ident in vocab.items():
        tokens[int(ident)] = token
    if any(token is None for token in tokens):
        raise RuntimeError(f"tokenizer vocabulary has missing ids: {path}")
    merges = (path / "merges.txt").read_text(encoding="utf-8").splitlines()
    types_path = path / "token_type.json"
    token_types = json.loads(types_path.read_text(encoding="utf-8")) if types_path.exists() else None
    return ST.Tokenizer(tokens, merges, token_types)


def free_local_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def get_json(url: str, timeout: float = 3.0):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url: str, body: dict, timeout: float):
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def tail(path: pathlib.Path, lines: int = 35) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return "(log not available)"


def validate_config(cfg: dict) -> None:
    args = cfg.get("args", [])
    if cfg.get("backend", "hip") != "hip":
        raise RuntimeError("the supplied server config is not a HIP config")
    if len(cfg.get("gpu", [])) < 2 or not cfg.get("layer_split"):
        raise RuntimeError("the config must use at least two GPUs with layer_split enabled")
    if "--mtp" not in args or "--ple-gguf" not in args:
        raise RuntimeError("the config must enable both PLE (--ple-gguf) and MTP (--mtp)")
    try:
        context = int(args[args.index("--max-context") + 1])
    except (ValueError, IndexError):
        raise RuntimeError("the config must set --max-context to at least 90000") from None
    if context < 90000:
        raise RuntimeError(f"--max-context is {context}; at least 90000 is needed for the 80K follow-up")
    if not cfg.get("tokenizer"):
        raise RuntimeError("the config must point at the model's extracted tokenizer directory")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=pathlib.Path, help="local server config JSON")
    parser.add_argument("--engine", required=True, type=pathlib.Path, help="new HIP engine binary")
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--image", type=pathlib.Path,
                        help="optional local image: run a 49K multimodal prompt and same-chat follow-up")
    parser.add_argument("--ple-batch", choices=("0", "1"), default="1",
                        help="set STRATA_PLE_BATCH for this run (default: 1; use 0 to match some local configs)")
    opts = parser.parse_args()

    config_path = opts.config.expanduser().resolve()
    engine_path = opts.engine.expanduser().resolve()
    if not config_path.is_file() or not engine_path.is_file():
        raise RuntimeError(f"config or engine not found: {config_path}, {engine_path}")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(cfg)
    tok = tokenizer_from_dir(pathlib.Path(cfg["tokenizer"]).expanduser().resolve())
    filler = " alpha"
    if len(tok.encode(filler)) != 1:
        raise RuntimeError("the tokenizer does not encode the validation filler as one token")

    work = pathlib.Path(tempfile.mkdtemp(prefix="strata-mixed-prefill-"))
    port = free_local_port()
    cfg["exe"] = str(engine_path)
    cfg["log"] = str(work / "engine.log")
    cfg["port"] = port
    image_url = None
    if opts.image is not None:
        image_path = opts.image.expanduser().resolve()
        if not image_path.is_file():
            raise RuntimeError(f"local test image not found: {image_path}")
        if not cfg.get("vision") or "--vision" not in cfg.get("args", []):
            raise RuntimeError("--image requires the config's vision settings and --vision engine option")
        mime = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
        image_url = f"data:{mime};base64,{base64.b64encode(image_path.read_bytes()).decode('ascii')}"
    else:
        # The default regression is text-only; do not start the optional image encoder unnecessarily.
        cfg.pop("vision", None)
        cfg["args"] = [arg for arg in cfg.get("args", []) if arg != "--vision"]
    env = cfg.setdefault("env", {})
    env.update({"STRATA_TRACE": "1", "STRATA_HIP_PROMPT_F16": "1", "STRATA_PLE_BATCH": opts.ple_batch})
    temp_config = work / "server-config.json"
    temp_config.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    server_log = work / "server.log"
    log_handle = server_log.open("w", encoding="utf-8")
    process_env = os.environ.copy()
    existing_pythonpath = process_env.get("PYTHONPATH", "")
    process_env["PYTHONPATH"] = str(ROOT) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    command = [sys.executable, "-m", "serve.server", "--engine", "strata", "--config", str(temp_config),
               "--host", "127.0.0.1", "--port", str(port)]
    print(f"Starting local HIP server on 127.0.0.1:{port}; logs: {work}", flush=True)
    proc = subprocess.Popen(command, cwd=ROOT, env=process_env, stdout=log_handle, stderr=subprocess.STDOUT)
    results: list[dict] = []
    try:
        health_url = f"http://127.0.0.1:{port}/health"
        deadline = time.monotonic() + opts.startup_timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited with code {proc.returncode}\n{tail(server_log)}")
            try:
                health = get_json(health_url)
                if int(health.get("max_context", 0)) >= 90000:
                    break
            except (OSError, ValueError, urllib.error.URLError):
                pass
            time.sleep(2)
        else:
            raise TimeoutError(f"HIP server did not become ready in {opts.startup_timeout}s\n{tail(server_log)}")

        chat_url = f"http://127.0.0.1:{port}/v1/chat/completions"
        model = cfg["model_name"]
        for label, count in (("16K", 16384), ("49K", 49152), ("80K", 80000)):
            content = filler * count + "\nLee el texto y responde brevemente al final."
            # Check the actual text count before sending; API usage below also includes the chat wrapper.
            text_tokens = len(tok.encode(content))
            if text_tokens < count:
                raise RuntimeError(f"{label} input tokenization returned only {text_tokens} tokens")
            use_image = image_url is not None and label == "49K"
            user_content = ([{"type": "text", "text": content},
                            {"type": "image_url", "image_url": {"url": image_url}}]
                           if use_image else content)
            messages = [{"role": "user", "content": user_content}]
            for turn in ("initial", "same-conversation follow-up"):
                if turn != "initial":
                    messages.append({"role": "user", "content":
                                     "En una frase, confirma que seguimos en la misma conversación y resume la petición."})
                body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": 64,
                        "reasoning_budget_tokens": 16}
                try:
                    response = post_json(chat_url, body, opts.request_timeout)
                except urllib.error.HTTPError as exc:
                    detail = exc.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"{label} {turn}: HTTP {exc.code}: {detail}\n{tail(pathlib.Path(cfg['log']))}") from exc
                choice = response.get("choices", [{}])[0]
                usage = response.get("usage", {})
                timing = response.get("timings", {}) or {}
                completion = int(usage.get("completion_tokens", 0))
                draft_n = int(timing.get("draft_n", 0))
                accepted = int(timing.get("draft_n_accepted", 0))
                if completion <= 0:
                    raise RuntimeError(f"{label} {turn}: no tokens decoded: {response}")
                if draft_n <= 0:
                    raise RuntimeError(f"{label} {turn}: MTP did not report any drafted tokens: {response}")
                message = choice.get("message", {})
                assistant_text = message.get("content")
                if not assistant_text:
                    raise RuntimeError(f"{label} {turn}: decode did not reach an answer: {response}")
                if turn == "initial":
                    messages.append({"role": "assistant", "content": assistant_text})
                result = {"size": label, "turn": turn, "multimodal": use_image, "text_tokens": text_tokens,
                          "prompt_tokens": int(usage.get("prompt_tokens", 0)), "completion_tokens": completion,
                          "draft_n": draft_n, "draft_n_accepted": accepted,
                          "cached_tokens": int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)),
                          "finish_reason": choice.get("finish_reason")}
                results.append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)

        engine_log = pathlib.Path(cfg["log"]).read_text(encoding="utf-8", errors="replace")
        if "strata generate: PLE on" not in engine_log:
            raise RuntimeError("PLE did not start; see engine.log")
        if "strata mtp: draft layer loaded" not in engine_log:
            raise RuntimeError("MTP did not start; see engine.log")
        if image_url is not None and "strata-vision: model files loaded" not in engine_log:
            raise RuntimeError("the vision encoder did not start; see engine.log")
        if "no kernel image is available for execution on the device" in engine_log:
            raise RuntimeError("the HIP kernel-image error recurred; see engine.log")
        if re.search(r"hipBLAS status \d+|left HIP error|pending HIP error before kernel launch", engine_log):
            raise RuntimeError("a HIP BLAS/runtime error was reported; see engine.log")
        if not re.search(r"HIP device (\d+) gfx12\S*, stream device \1", engine_log) or not re.search(
                r"HIP device (\d+) gfx103\S*, stream device \1", engine_log):
            raise RuntimeError("the prompt did not use matching HIP devices/streams on gfx12 and gfx103x; see engine.log")
        if not re.search(r"HIP device \d+ gfx12\S* - STRATA_HIP_PROMPT_F16=1 ignored", engine_log):
            raise RuntimeError("STRATA_HIP_PROMPT_F16=1 was not ignored on the gfx12 device; see engine.log")
        if not re.search(r"HIP device \d+ gfx103\S* - STRATA_HIP_PROMPT_F16=1:", engine_log):
            raise RuntimeError("STRATA_HIP_PROMPT_F16=1 did not enable FP16 on the gfx103x device; see engine.log")
        output = work / "results.json"
        output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        modes = "16K/49K/80K text" + (" + 49K multimodal" if image_url is not None else "")
        print(f"PASS: {modes}, same-conversation follow-ups, decode, PLE and MTP. Evidence: {output}")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log_handle.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
