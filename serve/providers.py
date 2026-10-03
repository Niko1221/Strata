"""Optional, local OpenAI-compatible providers for the existing Strata page.

Provider names and models are configuration data, never shell arguments. The
native token-ID engine remains unchanged. No new runtime dependencies.
"""
from __future__ import annotations

from contextlib import contextmanager
import errno
import hashlib
import http.client
import ipaddress
import json
import os
from pathlib import Path
import select
import socket
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import parse_qs, urlsplit, urlunsplit
import uuid

from serve.compaction import compact_history

MAX_JSON = 16 * 1024 * 1024
PROBE_TIMEOUT = 5
REQUEST_TIMEOUT = 120
RELEASE_WAIT_S = 15
STANDARD_FIELDS = {
    "messages", "stream", "stream_options", "temperature", "top_p", "max_tokens",
    "max_completion_tokens", "stop", "seed", "presence_penalty", "frequency_penalty",
    "tools", "tool_choice", "parallel_tool_calls", "response_format", "n", "logprobs",
    "top_logprobs", "user", "reasoning_effort", "reasoning",
}


class ProviderError(Exception):
    def __init__(self, message, code=400):
        super().__init__(message)
        self.code = code


def local_base_url(value):
    """Only literal loopback or localhost; never redirects, credentials or proxies."""
    if not isinstance(value, str) or not value or any(c.isspace() for c in value) or "\\" in value:
        raise ProviderError("Use a local server URL such as http://127.0.0.1:PORT/v1.")
    try:
        parsed = urlsplit(value)
        port = parsed.port
        hostname = parsed.hostname
    except ValueError:
        raise ProviderError("Invalid server URL.") from None
    if (parsed.scheme != "http" or not hostname or parsed.username is not None or
            parsed.password is not None or parsed.query or parsed.fragment or "@" in parsed.netloc):
        raise ProviderError("Use a local HTTP URL without credentials or a query string.")
    if hostname == "localhost":
        # Use a literal address so changes to localhost DNS cannot leave this PC.
        hostname = "127.0.0.1"
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        raise ProviderError("Connections are restricted to localhost, 127.0.0.1, or ::1 on this PC.") from None
    if str(address) not in ("127.0.0.1", "::1"):
        raise ProviderError("Connections are restricted to localhost, 127.0.0.1, or ::1 on this PC.")
    if port is not None and not 1 <= port <= 65535:
        raise ProviderError("Invalid server port.")
    host = "[::1]" if str(address) == "::1" else "127.0.0.1"
    authority = host + (f":{port}" if port is not None else "")
    path = parsed.path.rstrip("/")
    if ".." in path or "%" in path:
        raise ProviderError("The server URL cannot contain relative paths or escaped characters.")
    return urlunsplit(("http", authority, path, "", ""))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _open(base, path, body=None, timeout=REQUEST_TIMEOUT):
    # ProxyHandler({}) deliberately ignores system/environment HTTP proxies.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    request = urllib.request.Request(base + path, data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"})
    try:
        return opener.open(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        code = error.code
        error.close()
        raise ProviderError(f"The model server returned HTTP {code}.", 502) from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise ProviderError("Cannot connect to the model server. Check its status and URL.", 503) from None


def _abort_socket(connection_socket):
    if connection_socket is None:
        return
    try:
        connection_socket.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    try:
        # Buffered response files can retain a socket reference. Closing its
        # handle also interrupts an already pending recv on Windows.
        connection_socket._real_close()
    except OSError:
        pass


class _InferenceConnection(http.client.HTTPConnection):
    """A direct loopback connection that can be cancelled during connect too."""
    def __init__(self, host, port, cancel):
        super().__init__(host, port, timeout=REQUEST_TIMEOUT)
        self.cancel = cancel

    def connect(self):
        family = socket.AF_INET6 if self.host == "::1" else socket.AF_INET
        connection_socket = socket.socket(family, socket.SOCK_STREAM)
        self.sock = connection_socket
        try:
            if self.cancel.is_set():
                raise ProviderError("Request cancelled.", 499)
            connection_socket.setblocking(False)
            result = connection_socket.connect_ex((self.host, self.port))
            pending = {errno.EINPROGRESS, errno.EWOULDBLOCK, errno.EALREADY, errno.EINTR,
                       10035, 10036, 10037}  # Winsock's nonblocking connection results.
            if result not in (0, errno.EISCONN) and result not in pending:
                raise OSError(result, "local connection failed")
            deadline = time.monotonic() + PROBE_TIMEOUT
            while result not in (0, errno.EISCONN):
                if self.cancel.is_set():
                    raise ProviderError("Request cancelled.", 499)
                if time.monotonic() >= deadline:
                    raise TimeoutError("local connection timed out")
                _, writable, failed = select.select([], [connection_socket], [connection_socket], 0.1)
                if writable or failed:
                    result = connection_socket.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
                    if result:
                        raise OSError(result, "local connection failed")
            if self.cancel.is_set():
                raise ProviderError("Request cancelled.", 499)
            connection_socket.settimeout(REQUEST_TIMEOUT)
        except Exception:
            _abort_socket(connection_socket)
            self.sock = None
            raise


def _open_inference(base, path, body, cancel):
    """Watch cancellation before writing the request or waiting for headers."""
    parsed = urlsplit(local_base_url(base))
    connection = _InferenceConnection(parsed.hostname, parsed.port or 80, cancel)
    done = threading.Event()

    def watch():
        while not done.wait(0.1):
            if cancel.is_set():
                _abort_socket(connection.sock)
                return
    threading.Thread(target=watch, daemon=True, name="strata-provider-headers-watch").start()
    try:
        if cancel.is_set():
            raise ProviderError("Request cancelled.", 499)
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        connection.request("POST", parsed.path.rstrip("/") + path, body=data,
            headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
                     "Connection": "close"})
        response = connection.getresponse()
        if not 200 <= response.status < 300:
            code = response.status
            response.close()
            raise ProviderError(f"The model server returned HTTP {code}.", 502)
        response._provider_connection = connection
        return response
    except ProviderError:
        connection.close()
        raise
    except (http.client.HTTPException, OSError, TimeoutError, ValueError):
        connection.close()
        message = "Request cancelled." if cancel.is_set() else "Could not receive the model server response."
        raise ProviderError(message, 499 if cancel.is_set() else 503) from None
    finally:
        done.set()


def _read_json(response):
    try:
        raw = response.read(MAX_JSON + 1)
    except (http.client.HTTPException, OSError, TimeoutError):
        raise ProviderError("The server response was interrupted before completion.", 503) from None
    if len(raw) > MAX_JSON:
        raise ProviderError("The server response is too large.", 502)
    try:
        result = json.loads(raw)
    except (ValueError, UnicodeError):
        raise ProviderError("The server did not return valid JSON.", 502) from None
    if not isinstance(result, dict):
        raise ProviderError("The server did not return a valid response.", 502)
    return result


def discover(base_url):
    base = local_base_url(base_url)
    with _open(base, "/models", timeout=PROBE_TIMEOUT) as response:
        result = _read_json(response)
    data = result.get("data")
    if not isinstance(data, list):
        raise ProviderError("The server's /models endpoint did not return an OpenAI-compatible model list.", 502)
    models = []
    for entry in data:
        name = entry.get("id") if isinstance(entry, dict) else None
        if isinstance(name, str) and name.strip() and len(name) <= 512 and name not in [m["id"] for m in models]:
            models.append({"id": name, "name": name})
    return {"base_url": base, "models": models}


def estimate_tokens(messages, tools=None):
    """Conservative byte estimate; it is explicitly not a model-tokenizer count."""
    if not isinstance(messages, list) or not messages:
        raise ProviderError("messages must be a non-empty conversation list.")
    total = 32
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in ("system", "developer", "user", "assistant", "tool"):
            raise ProviderError("Invalid conversation format.")
        total += len(json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) + 32
    if tools:
        total += len(json.dumps(tools, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) + 128
    return total


def _clean_profile(body):
    if not isinstance(body, dict) or set(body) - {"id", "name", "base_url", "model", "context", "images", "backend", "reasoning_map"}:
        raise ProviderError("Specify name, base_url, model, context, and images.")
    name, model = body.get("name"), body.get("model")
    if not all(isinstance(x, str) and x.strip() and len(x) <= 512 for x in (name, model)):
        raise ProviderError("Specify a display name and model ID.")
    base = local_base_url(body.get("base_url"))
    context = body.get("context", 32768)
    if isinstance(context, bool) or not isinstance(context, int) or not 512 <= context <= 2_000_000:
        raise ProviderError("The context limit must be an integer from 512 to 2000000.")
    images = body.get("images", False)
    if not isinstance(images, bool):
        raise ProviderError("images must be true or false.")
    backend = body.get("backend", "generic")
    if backend not in ("generic", "ollama", "llamacpp"):
        raise ProviderError("backend must be generic, ollama, or llamacpp.")
    reasoning_map = body.get("reasoning_map", {})
    efforts = {"none", "minimal", "low", "medium", "high", "max", "xhigh"}
    if (not isinstance(reasoning_map, dict) or len(reasoning_map) > len(efforts) or
            any(key not in efforts or not isinstance(value, str) or not 1 <= len(value) <= 32 or
                not all(character.isascii() and (character.isalnum() or character in "-_") for character in value)
                for key, value in reasoning_map.items())):
        raise ProviderError("reasoning_map must map standard reasoning levels to short model-specific names.")
    identifier = body.get("id") or "provider-" + hashlib.sha256((base + "\0" + model.strip()).encode()).hexdigest()[:16]
    if not isinstance(identifier, str) or len(identifier) > 100 or not identifier or not all(c.isascii() and (c.isalnum() or c in "-_") for c in identifier):
        raise ProviderError("Invalid provider ID.")
    return {"id": identifier, "name": name.strip(), "base_url": base, "model": model.strip(),
            "context": context, "images": images, "backend": backend, "reasoning_map": dict(reasoning_map)}


class ProviderManager:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.RLock()
        self.profiles = {}
        self.current = None              # Selection is session-local: startup keeps the native model.
        self.native_inflight = 0
        self.readiness = {}
        try:
            saved = json.loads(self.path.read_text(encoding="utf-8-sig"))
            for profile in saved.get("providers", []):
                clean = _clean_profile(profile)
                self.profiles[clean["id"]] = clean
        except (OSError, ValueError, AttributeError, TypeError, ProviderError):
            # Invalid configuration never supplies a URL to the network layer.
            self.profiles = {}

    def selected(self):
        with self.lock:
            profile = self.profiles.get(self.current)
            return dict(profile) if profile else None

    def _persist(self, profiles):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            tmp.write_text(json.dumps({"version": 1, "providers": list(profiles.values())}, ensure_ascii=False,
                                      indent=2), encoding="utf-8")
            os.replace(tmp, self.path)
        finally:
            tmp.unlink(missing_ok=True)

    def _probe(self, profile):
        try:
            names = {entry["id"] for entry in discover(profile["base_url"])["models"]}
            if profile["model"] not in names:
                raise ProviderError("The model ID is not listed by the server. Install the model before registering it.", 404)
        except ProviderError as error:
            self.readiness[profile["id"]] = {"ready": False, "error": str(error), "checked_at": time.time()}
            raise
        self.readiness[profile["id"]] = {"ready": True, "checked_at": time.time()}

    def save(self, body, svc):
        profile = _clean_profile(body)
        with self.lock:
            if profile["id"] == self.current:
                raise ProviderError("An active provider cannot be changed. Switch to another model first.", 409)
            self._probe(profile)
            profiles = {**self.profiles, profile["id"]: profile}
            try:
                self._persist(profiles)
            except OSError:
                raise ProviderError("Could not save the model settings.", 500) from None
            self.profiles = profiles
        return {"saved": True, "provider": {**profile, "prepared": True, "available": True, "ready": True}}

    def catalog(self):
        with self.lock:
            providers = []
            for profile in self.profiles.values():
                state = self.readiness.get(profile["id"], {})
                # Readiness is evidence from a probe, not merely a file on disk.
                ready = bool(state.get("ready")) and time.time() - state.get("checked_at", 0) < 30
                providers.append({**profile, "prepared": True, **state, "available": ready, "ready": ready})
            return {"providers": providers, "current": self.current}

    def evidence(self, profile):
        with self.lock:
            state = self.readiness.get(profile["id"], {})
            if time.time() - state.get("checked_at", 0) > 10:
                try:
                    self._probe(profile)
                except ProviderError:
                    pass
            return dict(self.readiness.get(profile["id"], {}))

    def select(self, identifier, svc):
        with self.lock:
            if identifier is not None and (not isinstance(identifier, str) or identifier not in self.profiles):
                raise ProviderError("Specify a registered model ID.", 404)
            if identifier == self.current:
                return {"current": self.current, "status": "current"}
            if self.native_inflight:
                raise ProviderError("Wait for response generation to finish before switching.", 409)
            with svc.status_lock:
                if svc.status.get("busy") or svc.status.get("queued"):
                    raise ProviderError("Wait for response generation to finish before switching.", 409)
            if identifier is not None:
                self._probe(self.profiles[identifier])
            if not svc.fifo.acquire(blocking=False):
                raise ProviderError("Wait for response generation to finish before switching.", 409)
            try:
                with svc.status_lock:
                    if svc.status.get("busy") or svc.status.get("queued"):
                        raise ProviderError("Wait for response generation to finish before switching.", 409)
                switcher = getattr(svc, "model_switcher", None)
                if switcher is not None:
                    # The FIFO prevents begin() from starting another worker while
                    # this check and provider selection finish.
                    with switcher.lock:
                        state = switcher._state(svc)
                        if state.get("status") in ("starting", "restoring"):
                            raise ProviderError("The native model is loading. Wait for it to finish before switching.", 409)
                previous = self.profiles.get(self.current)
                if previous:
                    _release_external(previous)
                elif identifier is not None:
                    # These are svc.unload's engine/vision operations, performed while
                    # already holding its non-reentrant FIFO to avoid a reload gap.
                    try:
                        if hasattr(svc.engine, "unload") and svc.loaded():
                            svc.engine.unload()
                            if svc.vision is not None and hasattr(svc.vision, "unload"):
                                svc.vision.unload()
                    except Exception:
                        raise ProviderError("Could not stop the native model. The switch is incomplete.", 503) from None
                self.current = identifier
            finally:
                svc.fifo.release()
            return {"current": self.current, "status": "selected"}


def _release_external(profile):
    backend = profile.get("backend", "generic")
    if backend == "generic":
        return
    parsed = urlsplit(profile["base_url"])
    origin = urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
    path = "/api/generate" if backend == "ollama" else "/models/unload"
    body = {"model": profile["model"]}
    if backend == "ollama":
        body.update(keep_alive=0, stream=False)

    def llama_unloaded():
        with _open(profile["base_url"], "/models", timeout=PROBE_TIMEOUT) as response:
            listing = _read_json(response)
        models = listing.get("data", [])
        if not isinstance(models, list):
            return False
        return any(isinstance(entry, dict) and entry.get("id") == profile["model"]
                   and entry.get("status", {}).get("value") == "unloaded" for entry in models)

    try:
        # Router unload rejects an already-unloaded model, including one never used.
        if backend == "llamacpp" and llama_unloaded():
            return
        try:
            with _open(origin, path, body, timeout=PROBE_TIMEOUT) as response:
                result = _read_json(response)
        except ProviderError:
            # An idle unload may have completed between the check and the POST.
            if backend == "llamacpp" and llama_unloaded():
                return
            raise
        if result.get("error") or result.get("success") is False:
            raise ProviderError("release failed", 503)
        if backend == "ollama" and result.get("done") is not True:
            raise ProviderError("release not completed", 503)
        if backend == "llamacpp":
            deadline = time.monotonic() + RELEASE_WAIT_S
            while True:
                if llama_unloaded():
                    break
                if time.monotonic() >= deadline:
                    raise ProviderError("release not completed", 503)
                time.sleep(0.2)
    except ProviderError:
        raise ProviderError("Could not release the active additional model. The switch is incomplete.", 503) from None


def finish_native(handler, svc):
    """Release native admission from Handler.do_POST's existing finally block."""
    manager = getattr(svc, "providers", None)
    if manager is not None and getattr(handler, "provider_native_admitted", False):
        with manager.lock:
            manager.native_inflight -= 1
            handler.provider_native_admitted = False


def _body(handler):
    try:
        length = int(handler.headers.get("Content-Length", "0"))
        if not 0 < length <= MAX_JSON:
            raise ValueError()
        body = json.loads(handler.rfile.read(length))
        if not isinstance(body, dict):
            raise ValueError()
        return body
    except (ValueError, UnicodeError):
        raise ProviderError("Specify a valid JSON object.") from None


def _require_selected_model(profile, asked):
    if asked not in (None, "active", "strata", profile["model"], profile["id"]):
        raise ProviderError("The requested model does not match the selected model ID.", 404)


@contextmanager
def _request(svc, manager, asked=None, cancel=None):
    with svc.status_lock:
        svc.status["queued"] = svc.status.get("queued", 0) + 1
    acquired = False
    try:
        while not acquired:
            if cancel is not None and cancel.is_set():
                raise ProviderError("Request cancelled.", 499)
            acquired = svc.fifo.acquire(timeout=0.2)
        with svc.status_lock:
            svc.status["queued"] -= 1
        profile = manager.selected()
        if not profile:
            raise ProviderError("The model changed. Refresh its status before sending.", 409)
        _require_selected_model(profile, asked)
        with svc.status_lock:
            svc.status.update(busy=True, started=time.time(), phase="answer")
        yield profile
    finally:
        with svc.status_lock:
            if acquired:
                svc.status["busy"] = False
            else:
                svc.status["queued"] -= 1
        if acquired:
            svc.last_request_at = time.time()
            svc.fifo.release()


def _payload(body, profile):
    # Strata's template/engine-specific options do not have a portable meaning.
    payload = {key: value for key, value in body.items() if key in STANDARD_FIELDS}
    estimate_tokens(payload.get("messages"), payload.get("tools"))
    payload["model"] = profile["model"]
    if payload.get("reasoning_effort") == "off":
        payload["reasoning_effort"] = "none"
    effort = payload.get("reasoning_effort")
    if isinstance(effort, str):
        payload["reasoning_effort"] = profile.get("reasoning_map", {}).get(effort, effort)
    for key in ("max_tokens", "max_completion_tokens"):
        value = payload.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ProviderError("The output limit must be an integer.")
        if value is None or value <= 0:
            payload.pop(key, None)
    if payload.get("stream") is True:
        payload.setdefault("stream_options", {"include_usage": True})
    if not profile["images"]:
        for message in payload["messages"]:
            if isinstance(message.get("content"), list) and any(
                    isinstance(part, dict) and part.get("type") not in ("text", "input_text")
                    for part in message["content"]):
                raise ProviderError("Image input is not enabled for this provider.")
    return payload


def _chat(handler, svc, manager, body):
    cancel = threading.Event()
    handler._watch_client(cancel)
    try:
        return _chat_watched(handler, svc, manager, body, cancel)
    except (ProviderError, OSError, TimeoutError):
        if cancel.is_set():
            return
        raise
    finally:
        if handler.watch_done is not None:
            handler.watch_done.set()


@contextmanager
def _upstream_watched(response, cancel):
    done = threading.Event()

    def watch():
        while not done.wait(0.2):
            if cancel.is_set():
                # Shut down the socket before close(): BufferedReader.close can otherwise
                # wait for another thread's blocked read to finish.
                try:
                    _abort_socket(response.fp.raw._sock)
                except (AttributeError, OSError):
                    pass
                return
    watcher = threading.Thread(target=watch, daemon=True, name="strata-provider-watch")
    watcher.start()
    try:
        yield response
    finally:
        done.set()
        response.close()
        connection = getattr(response, "_provider_connection", None)
        if connection is not None:
            connection.close()


def _chat_watched(handler, svc, manager, body, cancel):
    with _request(svc, manager, body.get("model"), cancel) as profile:
        payload = _payload(svc.with_shared(body, "openai"), profile)
        with _upstream_watched(_open_inference(profile["base_url"], "/chat/completions", payload, cancel), cancel) as upstream:
            if payload.get("stream") is not True:
                response = _read_json(upstream)
                if response.get("error") or not isinstance(response.get("choices"), list) or not response["choices"]:
                    raise ProviderError("The server did not return a valid completion.", 502)
                handler._json(200, response)
                return
            if "text/event-stream" not in upstream.headers.get("Content-Type", ""):
                raise ProviderError("The server did not return a streaming response.", 502)
            handler._sse()
            complete = False
            try:
                while True:
                    line = upstream.readline(MAX_JSON + 1)
                    if not line:
                        break
                    if len(line) > MAX_JSON:
                        raise ProviderError("A server stream line is too large.", 502)
                    if line.startswith(b"data:"):
                        data = line[5:].strip()
                        if data == b"[DONE]":
                            complete = True
                        else:
                            try:
                                chunk = json.loads(data)
                            except (ValueError, UnicodeError):
                                raise ProviderError("Invalid server stream format.", 502) from None
                            if not isinstance(chunk, dict) or chunk.get("error"):
                                raise ProviderError("The server stream returned an error.", 502)
                            choices = chunk.get("choices", [])
                            if not isinstance(choices, list) or not all(isinstance(choice, dict) for choice in choices):
                                raise ProviderError("Invalid server stream format.", 502)
                            for choice in choices:
                                delta = choice.get("delta", {})
                                if isinstance(delta, dict) and "reasoning" in delta and "reasoning_content" not in delta:
                                    delta["reasoning_content"] = delta.pop("reasoning")
                                if choice.get("finish_reason") is not None:
                                    complete = True
                            line = b"data: " + json.dumps(chunk, ensure_ascii=False).encode("utf-8") + b"\n"
                    handler.wfile.write(line)
                    handler.wfile.flush()
                    if line.startswith(b"data:") and line[5:].strip() == b"[DONE]":
                        break
                if not complete:
                    raise ProviderError("The server disconnected before the completion finished.", 502)
            except (ProviderError, TimeoutError, OSError, http.client.HTTPException) as error:
                if isinstance(error, (BrokenPipeError, ConnectionResetError)):
                    return
                message = str(error) if isinstance(error, ProviderError) else "The server response was interrupted."
                packet = {"error": {"type": "provider_error", "message": message}}
                try:
                    handler.wfile.write(b"data: " + json.dumps(packet, ensure_ascii=False).encode("utf-8") + b"\n\n")
                    handler.wfile.flush()
                except (OSError, TimeoutError):
                    pass


def _compact(handler, svc, manager, body):
    cancel = threading.Event()
    handler._watch_client(cancel)
    try:
        return _compact_watched(handler, svc, manager, body, cancel)
    except (ProviderError, OSError, TimeoutError):
        if cancel.is_set():
            return
        raise
    finally:
        if handler.watch_done is not None:
            handler.watch_done.set()


def _compact_watched(handler, svc, manager, body, cancel):
    with _request(svc, manager, body.get("model"), cancel=cancel) as profile:
        def summarize(messages, budget):
            payload = {"model": profile["model"], "messages": messages, "temperature": 0, "max_tokens": budget,
                       "reasoning_effort": "none"}
            with _upstream_watched(_open_inference(profile["base_url"], "/chat/completions", payload, cancel), cancel) as response:
                result = _read_json(response)
            choices = result.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise ProviderError("The server did not return a summary.", 502)
            if choices[0].get("finish_reason") == "length":
                raise ProviderError("The summary reached the output limit. The original conversation is unchanged.")
            return choices[0].get("message", {}).get("content", "")
        result = compact_history(body.get("messages"), body.get("previous_summary", ""),
                                 count_prompt=estimate_tokens, summarize=summarize, max_context=profile["context"],
                                 cancelled=cancel.is_set)
        handler._json(200, result)


def dispatch(handler, svc, path, method):
    """Called after normal API authorization, before native route handling."""
    manager = getattr(svc, "providers", None)
    if manager is None:
        return False
    try:
        if path == "/api/providers":
            if method == "GET":
                handler._json(200, manager.catalog())
            elif method == "POST":
                body = _body(handler)
                if handler._own_page("model providers can be configured"):
                    handler._json(200, manager.save(body, svc))
            else:
                handler._json(405, {"error": {"message": "method not allowed"}})
            return True
        if path == "/api/providers/select":
            if method != "POST":
                handler._json(405, {"error": {"message": "method not allowed"}})
            else:
                body = _body(handler)
                if not handler._own_page("model providers can be selected"):
                    return True
                if set(body) != {"id"}:
                    raise ProviderError("Specify id. Use null to select the native model.")
                handler._json(200, manager.select(body["id"], svc))
            return True
        if path == "/api/provider-models":
            if method != "GET":
                handler._json(405, {"error": {"message": "method not allowed"}})
            else:
                base = parse_qs(urlsplit(handler.path).query).get("base_url", [None])[0]
                handler._json(200, discover(base))
            return True
        with manager.lock:
            profile = manager.selected()
            if not profile:
                if (method == "POST" and path in ("/v1/chat/completions", "/v1/messages", "/v1/chat/compact",
                        "/load", "/v1/load", "/api/local-models/switch")
                        and not getattr(handler, "provider_native_admitted", False)):
                    manager.native_inflight += 1
                    handler.provider_native_admitted = True
                return False
        if method == "GET":
            if path in ("/health", "/api/health"):
                evidence = manager.evidence(profile)
                handler._json(200, {"status": "ok", "model": profile["model"], "max_context": profile["context"],
                    "images": profile["images"], "api_key": bool(svc.api_key), "loaded": bool(evidence.get("ready")),
                    "service": "strata", "native": False, "provider": {key: profile[key] for key in ("id", "name", "base_url")},
                    "estimated_tokens": True})
            elif path in ("/v1/models", "/models"):
                evidence = manager.evidence(profile)
                handler._json(200, {"object": "list", "data": [{"id": profile["model"], "object": "model",
                    "meta": {"n_ctx": profile["context"]}, "prepared": True, "ready": bool(evidence.get("ready")),
                    "status": {"value": "ready" if evidence.get("ready") else "registered"}}]})
            elif path in ("/status", "/v1/status"):
                with svc.status_lock:
                    status = dict(svc.status)
                handler._json(200, {**status, "model": profile["model"], "native": False})
            elif path == "/metrics":
                with svc.status_lock:
                    busy, queued = bool(svc.status.get("busy")), svc.status.get("queued", 0)
                handler._json(200, {"live": {"state": "generating" if busy else "idle", "queued": queued},
                    "engine": {"model": profile["model"], "max_context": profile["context"]}, "requests": [],
                    "hardware": {}, "hardware_static": {"psutil": True}, "history": {}})
            elif path == "/mcp":
                handler._json(200, {"servers": [], "tools": 0, "external_provider": True})
            elif path == "/props":
                handler._json(200, {"model_alias": profile["model"], "modalities": {"vision": profile["images"]},
                    "default_generation_settings": {"n_ctx": profile["context"], "params": dict(svc.shared)},
                    "models_autoload": False, "is_sleeping": False, "chat_template": "", "native": False})
            else:
                return False
            return True
        if method == "POST" and path in ("/v1/chat/completions", "/v1/chat/completions/count_tokens", "/v1/chat/compact"):
            body = _body(handler)
            if path == "/v1/chat/completions":
                _chat(handler, svc, manager, body)
            elif path.endswith("count_tokens"):
                _require_selected_model(profile, body.get("model"))
                handler._json(200, {"input_tokens": estimate_tokens(body.get("messages"), body.get("tools")),
                    "max_context": profile["context"], "context_slack": 8, "estimated": True,
                    "estimate_method": "utf8_bytes_with_message_overhead"})
            elif handler._own_page("chat history can be summarized"):
                _compact(handler, svc, manager, body)
            return True
        if method == "POST" and path == "/api/local-models/switch":
            _body(handler)       # Consume JSON before returning a refusal on Windows.
            raise ProviderError("Switch the connection back to Strata before changing the native model.", 409)
        # Never quietly route another API protocol to the native model while an external model is selected.
        if path.startswith("/v1/") or path in ("/load", "/unload"):
            raise ProviderError("Use the OpenAI-compatible chat API for additional models.", 400)
        return False
    except ProviderError as error:
        _error(handler, error.code, str(error))
        return True
    except (ValueError, UnicodeError, TypeError, KeyError, AttributeError):
        _error(handler, 400, "Invalid model request format.")
        return True
    except (OSError, TimeoutError, http.client.HTTPException):
        _error(handler, 503, "Could not receive the server response.")
        return True


def _error(handler, code, message):
    try:
        handler._json(code, {"error": {"type": "provider_error", "message": message}})
    except (OSError, TimeoutError):
        pass                       # The caller already disconnected.
