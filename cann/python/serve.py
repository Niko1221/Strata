"""Loopback OpenAI / Anthropic text API for Strata's Ascend backend."""
from __future__ import annotations

import codecs
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import queue
import re
import signal
import shutil
import sys
import subprocess
import threading
import time
import uuid
from urllib.parse import urlsplit

from jinja2.sandbox import ImmutableSandboxedEnvironment

MODEL = "qwen3.8-flash-next-iq3_xxs"


def _number(value):
    match = re.search(r"[-+]?\d+(?:[.,]\d+)?", str(value).replace(",", ""))
    return float(match.group().replace(",", "")) if match else None


def _bytes_from(value, unit_hint=""):
    number = _number(value)
    if number is None:
        return None
    unit = f"{unit_hint} {value}".lower().replace("ib", "b")
    if "gb" in unit:
        return int(number * 1024 ** 3)
    if "mb" in unit:
        return int(number * 1024 ** 2)
    if "kb" in unit:
        return int(number * 1024)
    return int(number)


def _parse_two_line_npu_table(text, device_index):
    """Read the paired board and chip rows emitted by Ascend 910B2C npu-smi."""
    boards = []
    current = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("|") or line.count("|") != 4:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        board = re.fullmatch(r"(\d+)\s+(.+)", cells[0])
        if board and re.fullmatch(r"[A-Za-z][A-Za-z -]*", cells[1]):
            values = re.findall(r"\d+(?:\.\d+)?", cells[2])
            if len(values) >= 2:
                current = {
                    "name": board.group(2).strip(),
                    "power_w": float(values[0]),
                    "temperature_c": float(values[1]),
                }
                boards.append(current)
            continue
        if current is not None and re.search(
            r"\b[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.\d+\b", cells[1]
        ):
            values = re.findall(r"\d+(?:\.\d+)?", cells[2])
            if len(values) >= 3:
                current["utilization_pct"] = float(values[0])
                current["memory_used_bytes"] = int(values[-2]) * 1024 ** 2
                current["memory_total_bytes"] = int(values[-1]) * 1024 ** 2
    return boards[device_index] if 0 <= device_index < len(boards) else None


def parse_npu_smi(text, device_index=0):
    """Extract the portable fields from common npu-smi table/key-value output."""
    paired = _parse_two_line_npu_table(text, device_index)
    if paired is not None:
        return paired
    result = {}
    headers = None
    rows = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("|") or line.count("|") < 2:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        lowered = " ".join(cells).lower()
        if headers is None and any(word in lowered for word in
                                   ("temp", "temperature", "power", "hbm", "aicore", "util")) \
                and any(word in lowered for word in ("npu", "chip", "device", "id")):
            headers = cells
            continue
        if cells and cells[0].isdigit() and headers and len(cells) >= len(headers):
            rows.append((int(cells[0]), dict(zip(headers, cells))))
    fields = next((row for index, row in rows if index == device_index), None)
    if fields is None and 0 <= device_index < len(rows):
        fields = rows[device_index][1]
    if fields:
        for key, value in fields.items():
            key_lower = key.lower()
            if result.get("name") is None and any(word in key_lower for word in ("name", "chip")):
                result["name"] = value
            if result.get("utilization_pct") is None and any(word in key_lower for word in
                                                               ("aicore", "utilization", "util")):
                result["utilization_pct"] = _number(value)
            if result.get("temperature_c") is None and any(word in key_lower for word in
                                                             ("temperature", "temp")):
                result["temperature_c"] = _number(value)
            if result.get("power_w") is None and "power" in key_lower:
                result["power_w"] = _number(value)
            if "hbm" in key_lower or "memory" in key_lower:
                numbers = re.findall(r"[-+]?\d+(?:[.,]\d+)?", value.replace(",", ""))
                if numbers and result.get("memory_used_bytes") is None:
                    result["memory_used_bytes"] = _bytes_from(numbers[0], key)
                if len(numbers) > 1 and result.get("memory_total_bytes") is None:
                    result["memory_total_bytes"] = _bytes_from(numbers[1], key)
    for raw in text.splitlines():
        if ":" not in raw:
            continue
        key, value = (part.strip() for part in raw.split(":", 1))
        key_lower = key.lower()
        if result.get("name") is None and any(word in key_lower for word in ("chip name", "device name")):
            result["name"] = value
        if result.get("temperature_c") is None and "temperature" in key_lower:
            result["temperature_c"] = _number(value)
        if result.get("power_w") is None and "power" in key_lower:
            result["power_w"] = _number(value)
        if result.get("utilization_pct") is None and any(word in key_lower for word in ("aicore", "utilization")):
            result["utilization_pct"] = _number(value)
    return result


def render_chat(metadata, messages, enable_thinking=False):
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in ("system", "user", "assistant"):
            raise ValueError("text messages require a system, user or assistant role")
        if not isinstance(message.get("content"), str):
            raise ValueError("this CANN backend currently accepts text message content")
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    def fail(text):
        raise ValueError(text)
    env.globals["raise_exception"] = fail
    return env.from_string(metadata["tokenizer.chat_template"]).render(
        messages=messages, tools=None, add_generation_prompt=True, enable_thinking=enable_thinking,
    )


def format_logprobs(result, tokenizer, chat, top_count):
    def record(token, value):
        raw = tokenizer.token_bytes(token)
        return {"token": raw.decode("utf-8", "replace"), "bytes": list(raw), "logprob": value}
    content = [
        dict(record(token, probability), top_logprobs=[
            record(candidate, value) for candidate, value in top[:top_count]
        ])
        for token, probability, top in zip(
            result["tokens"], result["token_logprobs"], result["top5_logprobs"]
        )
    ]
    if chat:
        return {"content": content, "refusal": None}
    prefix = []
    offsets = []
    for token in result["tokens"]:
        offsets.append(len(tokenizer.decode(prefix)))
        prefix.append(token)
    return {
        "tokens": [item["token"] for item in content],
        "token_logprobs": result["token_logprobs"], "text_offset": offsets,
        "top_logprobs": [
            {item["token"]: item["logprob"] for item in row["top_logprobs"]}
            for row in content
        ],
    }


class Service:
    def __init__(self, args):
        self.args = args
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cann-model")
        self.lock = threading.Lock()
        self.metrics_lock = threading.Lock()
        self.completed = 0
        self.failed = 0
        self.last = None
        self.active = None
        self.requests = deque(maxlen=120)
        self.history = deque(maxlen=60)
        self.started_at = time.time()
        self.torch = None
        self.npu_smi = self._find_npu_smi()
        self.npu_cache_at = 0.0
        self.npu_cache = {}
        self.stopping = threading.Event()
        self.api_key = args.api_key_file.read_text().strip() if args.api_key_file else None
        if args.api_key_file and not self.api_key:
            raise ValueError("the API key file is empty")
        self.worker.submit(self._load).result()

    @staticmethod
    def _find_npu_smi():
        candidates = [
            os.environ.get("STRATA_NPU_SMI"),
            shutil.which("npu-smi"),
            "/usr/local/Ascend/driver/tools/npu-smi",
            "/usr/local/Ascend/ascend-toolkit/latest/bin/npu-smi",
        ]
        return next((path for path in candidates if path and Path(path).is_file()), None)

    def _load(self):
        import torch
        from qwen38 import Qwen38
        from weights import WeightStore

        self.torch = torch
        torch.set_num_threads(8)
        torch.npu.set_device(self.args.device)
        torch.npu.set_compile_mode(jit_compile=False)
        torch.npu.matmul.allow_hf32 = False
        self.store = WeightStore(
            self.args.model_dir, self.args.cache_dir, self.args.decoder, self.args.source,
            disk_cache_gib=self.args.disk_cache_gib,
        )
        self.engine = Qwen38(self.store, self.args.device, self.args.budget_gib, self.args.context,
                             self.args.weight_dtype)

    def request(self, prompt, count, temperature, callback, cancel):
        request_id = "req_" + uuid.uuid4().hex
        started_at = time.time()

        def emit(token):
            with self.metrics_lock:
                if self.active and self.active["id"] == request_id:
                    self.active["state"] = "generating"
                    self.active["generated"] += 1
            if callback:
                callback(token)

        def run():
            try:
                result = self.engine.generate(
                    prompt, count, temperature, callback=emit,
                    cancelled=lambda: cancel.is_set() or self.stopping.is_set(),
                )
                self.last = result
                finished_at = time.time()
                prefill_seconds = float(result.get("prefill_seconds", 0))
                decode_seconds = float(result.get("decode_seconds", 0))
                output_tokens = len(result.get("tokens", ()))
                request = {
                    "id": request_id,
                    "time": started_at,
                    "state": "done",
                    "prompt_tokens": len(result.get("prompt_tokens", prompt)),
                    "reused": 0,
                    "output_tokens": output_tokens,
                    "prompt_ms": round(prefill_seconds * 1000, 3),
                    "decode_ms": round(decode_seconds * 1000, 3),
                    "decode_tok_s": round(float(result.get("decode_tokens_per_second", 0)), 3),
                    "duration_s": round(finished_at - started_at, 3),
                    "finish_reason": result.get("finish_reason"),
                }
                with self.metrics_lock:
                    self.completed += 1
                    self.requests.appendleft(request)
                    self.history.append({
                        "time": finished_at,
                        "tok_s": request["decode_tok_s"],
                        "prefill_tok_s": round(request["prompt_tokens"] / prefill_seconds, 3)
                        if prefill_seconds > 0 else None,
                    })
                return result
            except BaseException:
                with self.metrics_lock:
                    self.failed += 1
                    self.requests.appendleft({
                        "id": request_id, "time": started_at, "state": "error",
                        "prompt_tokens": len(prompt), "output_tokens": 0,
                        "duration_s": round(time.time() - started_at, 3),
                    })
                raise
            finally:
                with self.metrics_lock:
                    if self.active and self.active["id"] == request_id:
                        self.active = None
                self.lock.release()
        if not self.lock.acquire(blocking=False):
            raise BlockingIOError("the single CANN inference slot is busy")
        with self.metrics_lock:
            self.active = {
                "id": request_id, "time": started_at, "state": "reading",
                "prompt_tokens": len(prompt), "generated": 0,
            }
        try:
            return self.worker.submit(run)
        except BaseException:
            with self.metrics_lock:
                self.active = None
            self.lock.release()
            raise

    def _read_npu(self):
        now = time.monotonic()
        if now - self.npu_cache_at < 1.0:
            return dict(self.npu_cache)
        result = {}
        if self.npu_smi:
            try:
                completed = subprocess.run(
                    [self.npu_smi, "info"], capture_output=True, text=True,
                    timeout=1.5, check=False,
                )
                result = parse_npu_smi(
                    completed.stdout + "\n" + completed.stderr,
                    int(str(self.args.device).rsplit(":", 1)[-1]),
                )
            except (OSError, subprocess.SubprocessError, ValueError):
                result = {}
        self.npu_cache_at = now
        self.npu_cache = result
        return dict(result)

    def npu_metrics(self):
        smi = self._read_npu()
        memory_total = smi.get("memory_total_bytes")
        memory_free = smi.get("memory_free_bytes")
        memory_used = smi.get("memory_used_bytes")
        allocated = reserved = None
        if self.torch is not None:
            try:
                free, total = self.torch.npu.mem_get_info(self.args.device)
            except (AttributeError, RuntimeError, TypeError):
                try:
                    free, total = self.torch.npu.mem_get_info()
                except (AttributeError, RuntimeError, TypeError):
                    free = total = None
            if total is not None:
                memory_total = int(total)
            if free is not None:
                memory_free = int(free)
            if memory_total is not None and memory_free is not None:
                memory_used = max(0, memory_total - memory_free)
            try:
                allocated = int(self.torch.npu.memory_allocated(self.args.device))
                reserved = int(self.torch.npu.memory_reserved(self.args.device))
            except (AttributeError, RuntimeError, TypeError):
                pass
        return {
            "index": int(str(self.args.device).rsplit(":", 1)[-1]),
            "name": smi.get("name") or "Ascend NPU",
            "utilization_pct": smi.get("utilization_pct"),
            "aicore_utilization_pct": smi.get("utilization_pct"),
            "memory_used_bytes": memory_used,
            "memory_total_bytes": memory_total,
            "memory_free_bytes": memory_free,
            "memory_allocated_bytes": allocated,
            "memory_reserved_bytes": reserved,
            "temperature_c": smi.get("temperature_c"),
            "power_w": smi.get("power_w"),
            "source": ["torch_npu"] + (["npu-smi"] if smi else []),
        }

    def metrics(self):
        now = time.time()
        with self.metrics_lock:
            active = dict(self.active) if self.active else None
            requests = [dict(request) for request in self.requests]
            history = [dict(sample) for sample in self.history]
            completed = self.completed
            failed = self.failed
        if active:
            elapsed = max(0.001, now - active["time"])
            generated = active["generated"]
            live = {
                "state": active["state"], "running": 1, "queued": 0,
                "prompt_tokens": active["prompt_tokens"], "generated": generated,
                "tok_s": round(generated / elapsed, 3) if active["state"] == "generating" else 0,
                "prefill_tok_s": round(active["prompt_tokens"] / elapsed, 3)
                if active["state"] == "reading" else 0,
                "prefill_tok_s_mean": round(active["prompt_tokens"] / elapsed, 3)
                if active["state"] == "reading" else 0,
            }
        else:
            live = {"state": "idle", "running": 0, "queued": 0, "tok_s": 0,
                    "prefill_tok_s": 0, "prefill_tok_s_mean": 0}
        weights = self.engine.w.report()
        expert_count = self.engine.md.get("qwen4exp.expert_count")
        npu = self.npu_metrics()
        return {
            "source": "strata-cann-torch",
            "time": now,
            "model": MODEL,
            "device": str(self.args.device),
            "npu": npu,
            "hardware": {
                "gpus": [{
                    "index": npu["index"], "name": npu["name"],
                    "util": npu["utilization_pct"], "mem_used": npu["memory_used_bytes"],
                    "mem_total": npu["memory_total_bytes"], "temp": npu["temperature_c"],
                    "power": npu["power_w"],
                }],
            },
            "live": live,
            "service": {
                "started_at": self.started_at, "uptime_s": round(now - self.started_at, 3),
                "completed": completed, "failed": failed, "busy": active is not None,
            },
            "engine": {
                "layers": self.engine.layers, "max_context": self.args.context,
                "expert_slots": len(self.engine.w.cache), "experts": expert_count,
                "expert_cache_mib": round(self.engine.w.used / 1024 ** 2, 3),
                "expert_budget_mib": round(self.engine.w.budget / 1024 ** 2, 3),
                "weight_dtype": self.args.weight_dtype,
            },
            "weights": weights,
            "history": {
                "tok_s": [item.get("tok_s") for item in history],
                "prefill_tok_s_mean": [item.get("prefill_tok_s") for item in history],
            },
            "requests": requests,
            "totals": {
                "requests": completed, "failed": failed,
                "prompt_tokens": sum(item.get("prompt_tokens", 0) for item in requests),
                "output_tokens": sum(item.get("output_tokens", 0) for item in requests),
            },
        }


def main():
    from qwen38 import arguments

    parser = arguments()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18791)
    parser.add_argument("--api-key-file", type=Path)
    args = parser.parse_args()
    service = Service(args)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def json(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if status >= 400:
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            self.wfile.write(body)

        def authorized(self):
            key = service.api_key
            if key is None:
                return True
            supplied = self.headers.get("Authorization", "").removeprefix("Bearer ")
            supplied = self.headers.get("x-api-key", supplied)
            if hmac.compare_digest(supplied.encode(), key.encode()):
                return True
            self.json(401, {"error": {"type": "authentication_error", "message": "invalid API key"}})
            return False

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/health":
                self.json(200, {
                    "status": "ok", "loaded": True, "model": MODEL,
                    "backend": "strata-cann-torch", "device": args.device,
                    "layers": service.engine.layers, "max_context": args.context,
                    "weight_dtype": args.weight_dtype,
                    "weight_mode": "local GGUF or exact HTTP ranges",
                    "busy": service.lock.locked(), "completed": service.completed,
                })
            elif self.authorized():
                if path == "/metrics":
                    self.json(200, service.metrics())
                elif path == "/v1/models":
                    self.json(200, {"object": "list", "data": [
                        {"id": MODEL, "object": "model", "owned_by": "strata"}]})
                else:
                    self.json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if not self.authorized():
                return
            if self.path not in ("/v1/completions", "/v1/chat/completions", "/v1/messages"):
                self.json(404, {"error": {"message": "not found"}})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4 * 1024 * 1024:
                    raise ValueError("request body must be between 1 byte and 4 MiB")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("the request must be a JSON object")
                for feature in ("tools", "tool_choice", "response_format", "logit_bias"):
                    if body.get(feature):
                        raise ValueError(f"{feature} is not implemented by this CANN backend")
                for feature, default in (
                    ("top_p", 1), ("n", 1), ("frequency_penalty", 0), ("presence_penalty", 0),
                    ("repetition_penalty", 1), ("seed", None), ("stop", None), ("echo", False),
                ):
                    if body.get(feature, default) != default:
                        raise ValueError(f"{feature} is not implemented by this CANN backend")
                if body.get("model", MODEL) != MODEL:
                    raise ValueError(f"model must be {MODEL}")
                count = body.get("max_tokens", 32)
                if type(count) is not int or not 1 <= count <= args.context:
                    raise ValueError("max_tokens must be a positive integer within the context limit")
                temperature = float(body.get("temperature", 0))
                if not 0 <= temperature <= 5:
                    raise ValueError("temperature must be between 0 and 5")
                anthropic = self.path == "/v1/messages"
                chat = self.path != "/v1/completions"
                if chat:
                    messages = body.get("messages")
                    if not isinstance(messages, list):
                        raise ValueError("messages must be a nonempty list")
                    if anthropic and body.get("system"):
                        messages = [{"role": "system", "content": body["system"]}] + messages
                    prompt = render_chat(
                        service.store.metadata, messages,
                        enable_thinking=body.get("enable_thinking", False),
                    )
                else:
                    prompt = body.get("prompt")
                    if not isinstance(prompt, str) and not (
                        isinstance(prompt, list) and all(type(t) is int for t in prompt)
                    ):
                        raise ValueError("prompt must be text or one list of token ids")
                ids = (service.engine.tokenizer.encode(prompt, parse_special=True)
                       if isinstance(prompt, str) else prompt)
                if not ids or len(ids) + count > args.context:
                    raise ValueError("empty prompt or prompt plus generation exceeds context")
                if any(token < 0 or token >= len(service.engine.tokenizer.tokens) for token in ids):
                    raise ValueError("token id outside the vocabulary")
                if type(body.get("stream", False)) is not bool:
                    raise ValueError("stream must be a boolean")
                stream = bool(body.get("stream", False))
                wants_logprobs = body.get("logprobs") is not None and body.get("logprobs") is not False
                top_count = (body.get("top_logprobs", 0) if chat else body.get("logprobs", 0)) or 0
                if wants_logprobs and (anthropic or stream):
                    raise ValueError("logprobs currently require a nonstreaming OpenAI request")
                if chat and body.get("logprobs") not in (None, False, True):
                    raise ValueError("chat logprobs must be a boolean")
                if type(top_count) is not int or not 0 <= top_count <= 5:
                    raise ValueError("this backend supports top_logprobs from zero to five")
                if body.get("stream_options") not in (None, {}, {"include_usage": True},
                                                     {"include_usage": False}):
                    raise ValueError("only stream_options.include_usage is supported")
                events = queue.Queue()
                cancel = threading.Event()
                future = service.request(ids, count, temperature, events.put if stream else None, cancel)
            except BlockingIOError as exc:
                self.json(429, {"error": {"type": "rate_limit_error", "message": str(exc)}})
                return
            except (ValueError, TypeError, KeyError) as exc:
                self.json(400, {"error": {"type": "invalid_request_error", "message": str(exc)}})
                return

            cid = ("msg_" if anthropic else "chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex
            created = int(time.time())
            try:
                if stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True
                    decoder = codecs.getincrementaldecoder("utf-8")("replace")
                    def emit(data, event=None):
                        if event:
                            self.wfile.write(f"event: {event}\n".encode())
                        self.wfile.write(("data: " + json.dumps(data, ensure_ascii=False) + "\n\n").encode())
                        self.wfile.flush()
                    if anthropic:
                        emit({"type": "message_start", "message": {
                            "id": cid, "type": "message", "role": "assistant", "model": MODEL,
                            "content": [], "stop_reason": None, "stop_sequence": None,
                            "usage": {"input_tokens": len(ids), "output_tokens": 0}}}, "message_start")
                        emit({"type": "content_block_start", "index": 0,
                              "content_block": {"type": "text", "text": ""}}, "content_block_start")
                    elif chat:
                        emit({"id": cid, "object": "chat.completion.chunk", "created": created,
                              "model": MODEL, "choices": [{"index": 0, "delta": {"role": "assistant"},
                                                           "finish_reason": None}]})
                    def emit_text(text):
                        if not text:
                            return
                        if anthropic:
                            emit({"type": "content_block_delta", "index": 0,
                                  "delta": {"type": "text_delta", "text": text}}, "content_block_delta")
                        else:
                            emit({"id": cid, "object": "chat.completion.chunk" if chat else "text_completion",
                                  "created": created, "model": MODEL,
                                  "choices": [{"index": 0, "delta" if chat else "text":
                                               {"content": text} if chat else text,
                                               "finish_reason": None}]})
                    while not future.done() or not events.empty():
                        try:
                            token = events.get(timeout=1)
                        except queue.Empty:
                            self.wfile.write(b": working\n\n")
                            self.wfile.flush()
                            continue
                        text = decoder.decode(service.engine.tokenizer.token_bytes(token))
                        emit_text(text)
                    result = future.result()
                    emit_text(decoder.decode(b"", final=True))
                    if anthropic:
                        emit({"type": "content_block_stop", "index": 0}, "content_block_stop")
                        emit({"type": "message_delta", "delta": {
                            "stop_reason": "end_turn" if result["finish_reason"] == "stop" else "max_tokens",
                            "stop_sequence": None}, "usage": {"output_tokens": len(result["tokens"])}},
                            "message_delta")
                        emit({"type": "message_stop"}, "message_stop")
                    else:
                        emit({"id": cid, "object": "chat.completion.chunk" if chat else "text_completion",
                              "created": created, "model": MODEL,
                              "choices": [{"index": 0, "delta" if chat else "text": {} if chat else "",
                              "finish_reason": result["finish_reason"]}]})
                        if (body.get("stream_options") or {}).get("include_usage"):
                            emit({"id": cid, "object": "chat.completion.chunk" if chat else "text_completion",
                                  "created": created, "model": MODEL, "choices": [], "usage": {
                                      "prompt_tokens": len(ids), "completion_tokens": len(result["tokens"]),
                                      "total_tokens": len(ids) + len(result["tokens"])}})
                        self.wfile.write(b"data: [DONE]\n\n")
                    return
                result = future.result()
                usage = {"prompt_tokens": len(ids), "completion_tokens": len(result["tokens"]),
                         "total_tokens": len(ids) + len(result["tokens"])}
                timing = {key: result[key] for key in (
                    "backend", "device", "prefill_seconds", "decode_seconds",
                    "decode_steps", "decode_tokens_per_second", "weights", "weight_dtype")}
                if anthropic:
                    self.json(200, {"id": cid, "type": "message", "role": "assistant", "model": MODEL,
                                   "content": [{"type": "text", "text": result["text"]}],
                                   "stop_reason": "end_turn" if result["finish_reason"] == "stop" else "max_tokens",
                                   "stop_sequence": None,
                                   "usage": {"input_tokens": len(ids), "output_tokens": len(result["tokens"])},
                                   "strata": timing})
                else:
                    choice = {"index": 0, "finish_reason": result["finish_reason"]}
                    choice["message" if chat else "text"] = (
                        {"role": "assistant", "content": result["text"]} if chat else result["text"]
                    )
                    if wants_logprobs:
                        choice["logprobs"] = format_logprobs(result, service.engine.tokenizer,
                                                            chat, top_count)
                    self.json(200, {"id": cid, "object": "chat.completion" if chat else "text_completion",
                                   "created": created, "model": MODEL, "choices": [choice],
                                   "usage": usage, "strata": timing})
            except (BrokenPipeError, ConnectionResetError):
                cancel.set()
            except Exception as exc:
                if not stream:
                    try:
                        self.json(500, {"error": {"type": "server_error", "message": str(exc)}})
                    except (BrokenPipeError, ConnectionResetError):
                        cancel.set()
                else:
                    print(f"stream failed: {exc}", file=sys.stderr, flush=True)
            finally:
                if not future.done():
                    cancel.set()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    def stop(signum, frame):
        service.stopping.set()
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(f"ready: http://{args.host}:{args.port} model={MODEL} backend=strata-cann-torch", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        service.worker.shutdown(wait=True)
        service.store.close()


if __name__ == "__main__":
    main()
