"""tools/memt_ref.py - drive a memt_sglang (MemT-1.0) server over its HTTP protocol.

The reference for P6/P7 is memt_sglang's own server, started with
`python -m memt_sglang.launch` (its README documents the routes).  This module is
the small client the parity and speed harnesses share: sessions and leases,
transactions, commits, queries (plain and streamed) and snapshots, written
against the documented wire format:

    GET  /memt/identity       model, configuration hash, runtime settings
    GET  /memt/capabilities   protocol, epoch, chunk_tokens, max_prefill_tokens,
                              snapshot_identity, memory_encoding
    POST /memt/control        {"action": ..., "session": ..., "lease": ..., ...}
    POST /generate            {"rid": "memt|<session>|<op>|<nonce>", "input_ids", ...}

It is deliberately dependency-free (urllib + json): the harnesses run on the GPU
host next to the server, and nothing here should drag in torch.  The equivalent
client in the reference repository is `memt_sglang/client/native.py`; this one
mirrors it, and `test_memt_ref.py` validates every payload it builds with that
repository's own `memt_sglang.protocol.validate_payload`.

    ref = Reference("http://127.0.0.1:30000")
    caps = ref.capabilities()                  # snapshot_identity lives here
    session = ref.create()
    lease = ref.acquire(session)
    ref.commit(session, ids, lease=lease, transaction="t1")
    ref.query(session, prompt, lease=lease, max_new=32)
    ref.export(session, lease, "p6.safetensors", transaction="t1")
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Callable, Optional
from uuid import uuid4

__all__ = ["ReferenceError", "Reference", "MAX_LEASE_SECONDS"]

MAX_LEASE_SECONDS = 3600
_IDENTIFIER = re.compile(r"[A-Za-z0-9_.-]{1,160}\Z")
_TIMEOUT_SECONDS = 600.0


class ReferenceError(RuntimeError):
    """The server refused a request, or answered outside the documented shape.

    `uncertain` is memt_sglang's own distinction: a 5xx or a transport loss may
    mean the write landed.  A caller must not blindly replay an uncertain write.
    """

    def __init__(self, message: str, *, status: Optional[int] = None, uncertain: bool = False):
        super().__init__(message)
        self.status = status
        self.uncertain = uncertain


def _check_identifier(value, what: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError("%s must match [A-Za-z0-9_.-]{1,160}: %r" % (what, value))
    return value


class Reference:
    """One memt_sglang server.  All calls are synchronous."""

    def __init__(self, base_url: str, *, api_key: Optional[str] = None,
                 timeout: float = _TIMEOUT_SECONDS):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.epoch: Optional[str] = None
        self._capabilities: Optional[dict] = None

    # ---- transport
    def _request(self, path: str, payload: Optional[dict] = None) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        request = urllib.request.Request(
            self.base_url + path,
            data=None if payload is None else json.dumps(payload).encode("utf-8"),
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")
            raise ReferenceError("%s: HTTP %d: %s" % (path, error.code, detail.strip()),
                                 status=error.code, uncertain=error.code >= 500) from error
        except (OSError, ValueError) as error:
            raise ReferenceError("%s: %s" % (path, error), uncertain=True) from error
        try:
            value = json.loads(body)
        except ValueError as error:
            raise ReferenceError("%s: not JSON: %s" % (path, error), uncertain=True) from error
        if not isinstance(value, dict):
            raise ReferenceError("%s: expected a JSON object" % path, uncertain=True)
        if value.get("error"):
            raise ReferenceError("%s: %s" % (path, value["error"]))
        return value

    def _stream(self, path: str, payload: dict,
                on_frame: Callable[[dict, float], None]) -> tuple[dict, float]:
        """POST and read the SSE frames; returns the terminal frame and when it arrived."""
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        request = urllib.request.Request(self.base_url + path, data=json.dumps(payload).encode("utf-8"),
                                         headers=headers)
        try:
            response = urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")
            raise ReferenceError("%s: HTTP %d: %s" % (path, error.code, detail.strip()),
                                 status=error.code, uncertain=error.code >= 500) from error
        except (OSError, ValueError) as error:
            raise ReferenceError("%s: %s" % (path, error), uncertain=True) from error
        terminal = None
        terminal_at = 0.0
        done = False
        try:
            with response:
                for raw in response:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    now = time.perf_counter()
                    if data == "[DONE]":
                        done = True
                        break
                    try:
                        frame = json.loads(data)
                    except ValueError as error:
                        raise ReferenceError("stream: not JSON: %r" % data[:200], uncertain=True) from error
                    if not isinstance(frame, dict) or frame.get("error"):
                        raise ReferenceError("stream: %r" % (frame,), uncertain=True)
                    meta = frame.get("meta_info") or {}
                    if meta.get("finish_reason") is not None:
                        if terminal is not None:
                            raise ReferenceError("stream: two terminal frames", uncertain=True)
                        terminal = frame
                        terminal_at = now
                        # the server closes the stream with `data: [DONE]`; read it
                        # rather than dropping the connection mid-response
                        continue
                    if terminal is None:
                        on_frame(frame, now)
        except OSError as error:
            raise ReferenceError("stream: %s" % error, uncertain=True) from error
        if terminal is None:
            raise ReferenceError("stream: ended without terminal metadata", uncertain=True)
        if not done:
            raise ReferenceError("stream: ended without data: [DONE]", uncertain=True)
        return terminal, terminal_at

    # ---- identity and capabilities
    def identity(self) -> dict:
        return self._request("/memt/identity")

    def capabilities(self, *, refresh: bool = False) -> dict:
        if self._capabilities is not None and not refresh:
            return dict(self._capabilities)
        caps = self._request("/memt/capabilities")
        if caps.get("protocol") != "memt-http-v1":
            raise ReferenceError("capabilities: protocol is %r, expected memt-http-v1" % caps.get("protocol"))
        if not isinstance(caps.get("epoch"), str) or not caps["epoch"]:
            raise ReferenceError("capabilities: no epoch")
        for field in ("chunk_tokens", "max_prefill_tokens"):
            if type(caps.get(field)) is not int or caps[field] < 1:
                raise ReferenceError("capabilities: bad %s: %r" % (field, caps.get(field)))
        if not isinstance(caps.get("snapshot_identity"), str) or not caps["snapshot_identity"]:
            raise ReferenceError("capabilities: no snapshot_identity")
        self.epoch = caps["epoch"]
        self._capabilities = dict(caps)
        return dict(caps)

    def snapshot_identity(self) -> str:
        return self.capabilities()["snapshot_identity"]

    def wait_ready(self, timeout: float = 600.0, interval: float = 1.0) -> dict:
        """Poll capabilities until the server answers (it loads weights first)."""
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            try:
                return self.capabilities(refresh=True)
            except ReferenceError as error:
                last = error
                time.sleep(interval)
        raise ReferenceError("the reference server did not become ready within %.0f s: %s"
                             % (timeout, last))

    # ---- control
    def control(self, session: str, action: str, *, lease: Optional[str] = None, **fields) -> dict:
        _check_identifier(session, "session")
        payload = dict(session=session, action=action, lease=lease)
        if self.epoch:
            payload["epoch"] = self.epoch
        payload.update(fields)
        return self._request("/memt/control", payload)

    def create(self, session: Optional[str] = None, *, memory_read: Optional[bool] = None) -> str:
        session = session or uuid4().hex
        _check_identifier(session, "session")
        fields = {}
        if memory_read is not None:
            fields["memory_read"] = bool(memory_read)
        result = self.control(session, "create", **fields)
        epoch = result.get("epoch")
        if epoch is not None:
            if not isinstance(epoch, str) or not epoch:
                raise ReferenceError("create: no epoch")
            if self.epoch is not None and epoch != self.epoch:
                raise ReferenceError("create: the server's epoch changed under the client")
            self.epoch = epoch
        return session

    def acquire(self, session: str, ttl: int = MAX_LEASE_SECONDS) -> str:
        result = self.control(session, "acquire", ttl=int(ttl))
        lease = result.get("lease")
        if not isinstance(lease, str) or not lease:
            raise ReferenceError("acquire: no lease: %r" % (result,))
        return lease

    def release(self, session: str, lease: str) -> dict:
        return self.control(session, "release", lease=lease)

    def reset(self, session: str, lease: str) -> dict:
        return self.control(session, "reset", lease=lease)

    def close(self, session: str, lease: str) -> dict:
        return self.control(session, "close", lease=lease)

    def begin(self, session: str, lease: str, transaction: str) -> dict:
        _check_identifier(transaction, "transaction")
        return self.control(session, "begin", lease=lease, transaction=transaction)

    def publish(self, session: str, lease: str, transaction: str, *,
                memory_read: Optional[bool] = None) -> dict:
        fields = {} if memory_read is None else {"memory_read": bool(memory_read)}
        result = self.control(session, "publish", lease=lease, transaction=transaction, **fields)
        if result.get("status") != "published":
            raise ReferenceError("publish: the transaction is %r, not published"
                                 % result.get("status"), uncertain=True)
        return result

    def status(self, session: str, lease: str) -> dict:
        return self.control(session, "status", lease=lease)

    def export(self, session: str, lease: str, name: str, *, overwrite: bool = False,
               transaction: Optional[str] = None) -> dict:
        encoding = self.capabilities().get("memory_encoding")
        if not isinstance(encoding, str) or not encoding:
            raise ReferenceError("capabilities: no memory_encoding (export needs it)")
        fields = dict(encoding=encoding, file=name, overwrite=bool(overwrite))
        if transaction is not None:
            fields["transaction"] = transaction
        result = self.control(session, "export", lease=lease, **fields)
        if result.get("format") != "memt-memory-v1":
            raise ReferenceError("export: format is %r, expected memt-memory-v1" % result.get("format"))
        return result

    def import_(self, session: str, lease: str, transaction: str, name: str) -> dict:
        encoding = self.capabilities().get("memory_encoding")
        if not isinstance(encoding, str) or not encoding:
            raise ReferenceError("capabilities: no memory_encoding (import needs it)")
        result = self.control(session, "import", lease=lease, transaction=transaction,
                              encoding=encoding, file=name)
        if result.get("format") != "memt-memory-v1":
            raise ReferenceError("import: format is %r, expected memt-memory-v1" % result.get("format"))
        return result

    # ---- generation
    def _generation(self, session: str, lease: str, transaction: Optional[str], operation: str,
                    ids: list[int], max_new: int, sampling: Optional[dict],
                    stream: bool) -> dict:
        tokens = [int(x) for x in ids]
        if not tokens:
            raise ValueError("a request needs at least one token id")
        if any(x < 0 for x in tokens):
            raise ValueError("token ids must be non-negative")
        action = operation if operation == "query" else "commit:%d" % len(tokens)
        rid = "memt|%s|%s|%s" % (_check_identifier(session, "session"), action, uuid4().hex)
        params = dict(sampling or {})
        for reserved in ("max_new_tokens", "n", "ignore_eos", "skip_special_tokens"):
            if reserved in params:
                raise ValueError("%s is managed by this client, not by sampling" % reserved)
        params.update(max_new_tokens=int(max_new), n=1, ignore_eos=False, skip_special_tokens=False)
        payload = {
            "rid": rid,
            "input_ids": tokens,
            "sampling_params": params,
            "stream": bool(stream),
            "memt_lease": lease,
            "memt_transaction": transaction,
        }
        if self.epoch:
            payload["memt_epoch"] = self.epoch
        return payload

    def _result(self, result: dict, rid: str, operation: str, *, need_ids: bool = True) -> dict:
        meta = result.get("meta_info")
        if not isinstance(meta, dict) or meta.get("id") != rid:
            raise ReferenceError("%s: the answer does not carry the request's rid" % operation, uncertain=True)
        reason = meta.get("finish_reason")
        if not isinstance(reason, dict) or reason.get("type") not in ("stop", "length"):
            raise ReferenceError("%s: finish_reason is %r" % (operation, reason), uncertain=True)
        output = result.get("output_ids")
        if output is None and not need_ids:
            output = []
        if not isinstance(output, list) or any(type(x) is not int for x in output):
            raise ReferenceError("%s: no output_ids" % operation, uncertain=True)
        return {"ids": list(output), "text": result.get("text", ""), "meta": meta,
                "finish": reason["type"]}

    def commit(self, session: str, ids: list[int], *, lease: str,
               transaction: Optional[str] = None) -> dict:
        """One commit request for the whole token list (the reference's turn semantics)."""
        limit = self.capabilities()["max_prefill_tokens"]
        if len(ids) > limit:
            raise ValueError("a single commit request carries at most max_prefill_tokens=%d ids "
                             "(got %d); split the turn" % (limit, len(ids)))
        payload = self._generation(session, lease, transaction, "commit", ids, 0, None, False)
        started = time.perf_counter()
        result = self._request("/generate", payload)
        done = self._result(result, payload["rid"], "commit", need_ids=False)
        done["ms"] = (time.perf_counter() - started) * 1000.0
        if done["ids"]:
            raise ReferenceError("commit: the server returned tokens for a commit", uncertain=True)
        return done

    def query(self, session: str, ids: list[int], *, lease: str, max_new: int,
              sampling: Optional[dict] = None, stream: bool = False) -> dict:
        """A query; returns the output tokens and, when streamed, its timings.

        `first_ms` is the time to the first frame carrying a token, `end_ms` the
        time to the terminal frame.  A non-streamed query reports `None` for
        `first_ms` (the server sends everything at once).
        """
        payload = self._generation(session, lease, None, "query", ids, max_new, sampling, stream)
        started = time.perf_counter()
        marks: list[float] = []

        def on_frame(frame: dict, now: float) -> None:
            if frame.get("output_ids"):
                marks.append(now)

        terminal_at = None
        if stream:
            terminal, terminal_at = self._stream("/generate", payload, on_frame)
            ended = time.perf_counter()
        else:
            terminal = self._request("/generate", payload)
            ended = time.perf_counter()
        done = self._result(terminal, payload["rid"], "query")
        frames = len(marks)
        from_terminal = False
        if not marks and done["ids"] and terminal_at is not None:
            # the server put every token in the terminal frame: TTFT is when it arrived
            marks = [terminal_at]
            from_terminal = True
            frames = 1
        done["first_ms"] = (marks[0] - started) * 1000.0 if marks else None
        done["first_from_terminal"] = from_terminal
        done["end_ms"] = (ended - started) * 1000.0
        done["frames"] = frames
        done["prompt_tokens"] = len(ids)
        return done
