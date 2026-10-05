"""Optional native FP4/FP8 DeepSeek backend; no CUDA/HIP engine changes.

Uses a separately built deepMoE executable (MIT, acupof-ai/cachedMoE). Tokenization
stays on the CPU, so concurrent prompt preparation cannot block the resident
engine's stdout protocol. The checkpoint is read-only.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time

EOS = "<｜end▁of▁sentence｜>"


class DeepMoETokenizer:
    def __init__(self, model):
        try:
            from tokenizers import Tokenizer
        except ImportError as e:
            raise ValueError("deepmoe needs the optional Python package: pip install tokenizers") from e
        source = (Path(model) / "tokenizer.json").read_text(encoding="utf-8")
        data = json.loads(source)
        self.native = Tokenizer.from_str(source)
        plain = dict(data, added_tokens=[])
        self.plain = Tokenizer.from_str(json.dumps(plain))
        self.added = {x["id"]: x["content"].encode() for x in data.get("added_tokens", [])}
        # The checkpoint's ByteLevel decoder uses the standard GPT-2 byte map.
        visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        codes = list(visible)
        for byte in range(256):
            if byte not in visible:
                visible.append(byte)
                codes.append(256 + sum(c >= 256 for c in codes))
        self.bytes = dict(zip(map(chr, codes), visible))
        self.eos_id = self.native.token_to_id(EOS)
        if self.eos_id != 1:
            raise ValueError("deepmoe backend requires the DeepSeek-V4.1-Flash tokenizer (EOS ID 1)")

    def encode(self, text, parse_special=False, plain=None):
        tokenizer = self.native if parse_special else self.plain
        if not plain:
            return tokenizer.encode(text, add_special_tokens=False).ids
        ids, start = [], 0
        for lo, hi in plain:
            ids.extend(tokenizer.encode(text[start:lo], add_special_tokens=False).ids)
            ids.extend(self.plain.encode(text[lo:hi], add_special_tokens=False).ids)
            start = hi
        ids.extend(tokenizer.encode(text[start:], add_special_tokens=False).ids)
        return ids

    def decode(self, ids):
        return self.native.decode(ids, skip_special_tokens=False)

    def token_bytes(self, token):
        if token in self.added:
            return self.added[token]
        text = self.native.id_to_token(token)
        if text is None:
            raise ValueError(f"unknown token ID {token}")
        return bytes(self.bytes[c] for c in text)


class DeepMoETemplate:
    """Use the checkpoint's own renderer. Tool/vision formats differ from Qwen."""
    def __init__(self, model):
        path = Path(model) / "encoding" / "encoding.py"
        spec = importlib.util.spec_from_file_location("strata_deepmoe_encoding", path)
        self.encoding = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.encoding)

    def render(self, messages, tools=None, add_generation_prompt=True, **kwargs):
        if tools or any(m.get("tool_calls") or m.get("role") == "tool" for m in messages):
            raise ValueError("deepmoe backend supports text chat only; DeepSeek DSML tools are not wired to Strata")
        if any(not isinstance(m.get("content", ""), str) for m in messages):
            raise ValueError("deepmoe backend does not support image input")
        if not add_generation_prompt:
            raise ValueError("deepmoe does not support Qwen's trailing-effort template mode")
        thinking = kwargs.get("enable_thinking", True) is not False
        effort = {"low": 50, "medium": 75, "high": 100, "xhigh": 100}.get(kwargs.get("reasoning_effort"), 75)
        try:
            return self.encoding.encode_messages(messages, thinking_mode="thinking" if thinking else "chat",
                                                 drop_thinking=True, reasoning_effort=effort)
        except (AssertionError, NotImplementedError) as e:
            raise ValueError(f"unsupported DeepSeek chat message: {e}") from e


class DeepMoEEngine:
    stop_ids = {1}
    batch = 0

    def __init__(self, exe, args, *, model, cwd=None, log=None, env=None,
                 timeout=300, error_type=RuntimeError):
        if not isinstance(args, list) or any(not isinstance(x, str) for x in args):
            raise ValueError("deepmoe config args must be a list of strings")
        self.command = [exe, "serve", "--model", str(model), *args]
        self.cwd, self.env, self.log_path = cwd, env, log
        self.timeout, self.error_type = timeout, error_type
        self.lock, self.write_lock = threading.Lock(), threading.Lock()
        self.last, self.progress, self.info = None, None, {}
        self.proc = None
        self.restart()

    def _reader(self, proc, events):
        try:
            for line in proc.stdout:
                events.put(json.loads(line))
        except (ValueError, OSError) as e:
            events.put({"event": "error", "message": f"deepmoe protocol: {e}"})
        finally:
            events.put({"event": "exit", "message": "deepmoe process exited; see its log"})

    def _send(self, message):
        with self.write_lock:
            try:
                self.proc.stdin.write(json.dumps(message) + "\n")
                self.proc.stdin.flush()
            except (OSError, ValueError) as e:
                raise self.error_type("deepmoe input pipe closed") from e

    def _next(self, timeout=None):
        try:
            return self.events.get(timeout=self.timeout if timeout is None else timeout)
        except queue.Empty as e:
            raise self.error_type("deepmoe engine timed out; see its log") from e

    def restart(self):
        if self.proc is not None and self.alive():
            return
        if self.proc is not None:
            self.close()
            self.proc = None
        self.log_file = open(self.log_path, "a", encoding="utf-8") if self.log_path else None
        self.events = queue.Queue()
        try:
            self.proc = subprocess.Popen(self.command, cwd=self.cwd, env=self.env,
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=self.log_file or subprocess.DEVNULL,
                                         text=True, encoding="utf-8", bufsize=1)
            self.reader = threading.Thread(target=self._reader, args=(self.proc, self.events), daemon=True)
            self.reader.start()
            ready = self._next()
            if ready.get("event") != "ready" or type(ready.get("max_context")) is not int or ready.get("max_context", 0) <= 0:
                raise self.error_type(f"deepmoe did not become ready: {ready}")
            self.max_context = self.known_ctx = ready["max_context"]
            self.info = {"backend": "vulkan", "model_family": "deepseek-v4.1-flash", **ready}
        except BaseException:
            self.close()
            raise

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def exit_code(self):
        return self.proc.poll() if self.proc else None

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        if embeddings:
            raise ValueError("deepmoe backend supports text chat only")
        sampling = sampling or {}
        neutral = {"top_k": 0, "min_p": 0, "repetition_penalty": 1,
                   "frequency_penalty": 0, "presence_penalty": 0}
        for key, value in neutral.items():
            if sampling.get(key) not in (None, value):
                raise ValueError(f"deepmoe does not implement {key}; omit it or use {value}")
        if sampling.get("strata_tune") or sampling.get("experimental_speed_projection"):
            raise ValueError("deepmoe does not implement Strata engine tuning/control vectors")
        with self.lock:
            if cancel.is_set():
                return
            request = {"op": "generate", "session": "strata", "prompt_ids": list(ids),
                       "max_tokens": max_new, "stop_ids": [1], "reuse": True}
            for key in ("temperature", "top_p", "seed"):
                if sampling.get(key) is not None:
                    request[key] = sampling[key]
            self.progress = (0, len(ids))
            self._send(request)
            ended, stopping = False, False
            last_event = time.monotonic()
            try:
                while True:
                    if cancel.is_set() and not stopping:
                        self._send({"op": "cancel"})
                        stopping = True
                    try:
                        event = self.events.get(timeout=.1)
                    except queue.Empty:
                        if time.monotonic() - last_event > self.timeout:
                            raise self.error_type("deepmoe generation timed out; see its log")
                        yield None          # Strata heartbeat while prefill is running
                        continue
                    last_event = time.monotonic()
                    kind = event.get("event")
                    if kind == "prefill":
                        self.progress = (event.get("done", 0), event.get("total", len(ids)))
                    elif kind == "token":
                        if not stopping:
                            yield event["id"]
                    elif kind == "done":
                        self._done(event, len(ids))
                        ended = True
                        return
                    elif kind in ("error", "exit"):
                        ended = True
                        self.close()
                        raise self.error_type(event.get("message", "deepmoe engine failed"))
            finally:
                # Client disconnect, EOS, repeat-stop and reasoning-budget wrap all close the iterator.
                # Drain to done before the next request, so no stale tokens cross request boundaries.
                if not ended and self.alive():
                    self._send({"op": "cancel"})
                    deadline = time.monotonic() + self.timeout
                    while True:
                        try:
                            event = self._next(max(.01, deadline - time.monotonic()))
                        except self.error_type:
                            self.close()
                            raise
                        if event.get("event") == "done":
                            self._done(event, len(ids))
                            break
                        if event.get("event") in ("error", "exit"):
                            self.close()
                            raise self.error_type(event.get("message", "deepmoe engine failed"))
                        if time.monotonic() >= deadline:
                            self.close()
                            raise self.error_type("deepmoe did not finish after cancellation")
                self.progress = None

    def _done(self, event, prompt):
        self.last = {"prompt_tokens": prompt, "reused": event.get("reused_tokens", 0),
                     "generated": event.get("generated", 0), "prompt_ms": event.get("prefill_ms"),
                     "decode_ms": event.get("decode_ms"), "deepmoe": event}

    def close(self):
        if self.proc is not None:
            if self.alive():
                try:
                    self._send({"op": "quit"})
                    self.proc.wait(timeout=10)
                except (OSError, self.error_type, subprocess.TimeoutExpired):
                    self.proc.kill()
                    self.proc.wait(timeout=10)
            if hasattr(self, "reader"):
                self.reader.join(timeout=2)
            for pipe in (self.proc.stdin, self.proc.stdout):
                if pipe:
                    try:
                        pipe.close()
                    except OSError:
                        pass
        if getattr(self, "log_file", None):
            self.log_file.close()


def backend_from_config(cfg, error_type=RuntimeError):
    for key in ("exe", "model"):
        if not cfg.get(key):
            raise ValueError(f"deepmoe config requires {key!r}")
    if cfg.get("parallel") not in (None, 0, 1) or any(cfg.get(k) for k in (
            "vision", "lazy_load", "effort_position", "mcp_servers")):
        raise ValueError("deepmoe currently supports one text-only engine, without lazy load, MCP or effort_position")
    model = Path(cfg["model"]).expanduser().resolve()
    tokenizer, template = DeepMoETokenizer(model), DeepMoETemplate(model)
    cwd = str(Path(cfg.get("cwd") or ".").expanduser().resolve())
    exe = str(Path(cwd, cfg["exe"]).expanduser().resolve())
    engine = DeepMoEEngine(exe, cfg.get("args", []), model=model, cwd=cwd, log=cfg.get("log"),
                          env={**os.environ, **{str(k): str(v) for k, v in cfg.get("env", {}).items()}},
                          error_type=error_type)
    return engine, tokenizer, template
