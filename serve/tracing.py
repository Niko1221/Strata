"""serve/tracing.py - opt-in OpenTelemetry traces of every /v1 request, exported over OTLP/HTTP.

Off by default: with no endpoint set nothing here runs, and the request path is byte-identical to the release build.
Turn it on with `--trace-otlp URL`, `"trace_otlp"` in the run config, or $STRATA_TRACE_OTLP (the value `default` is
http://localhost:4318/v1/traces, where Foundry Toolkit's collector and a plain Jaeger or Tempo listen).

One trace per request, built from the seconds the server already measures for the Monitor tab - nothing is timed
twice and no clock is added to the generation loop:

    strata.request POST /v1/chat/completions      the whole request, from the first byte to the last
    +- strata.queue                               waiting for the model to be free (record["queue_s"])
    +- strata.load                                starting the engine for it (record["load_s"])
    +- strata.prefill                             reading the prompt (the engine's prompt_ms)
    +- strata.decode                              writing the answer (the engine's predicted_ms)

The queue and load spans come from the server's own perf_counter; prefill and decode come from the engine's DONE
line, so they are the GPU's time, not the host's wall around it. A span whose number the server does not have
(an engine that says nothing, a request that never reached the model) is left out rather than drawn as zero.

W3C context is followed both ways: a `traceparent` header from the client becomes the root span's parent, so Strata
shows up inside the caller's trace, and every answer carries the `traceparent` of its own root span.

The export runs on its own thread and never blocks or fails a request: a collector that is down costs a line in the
log and a counted drop. OTLP/HTTP is written as JSON by hand, so tracing adds no package to the install.
"""
from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
import urllib.error
import urllib.request

DEFAULT_ENDPOINT = "http://localhost:4318/v1/traces"
SCOPE = "strata"
FLUSH_S = 1.0                     # a batch is sent this long after the first span waits, so a slow trace is not held
BATCH_MAX = 64                    # spans per request POST, to keep a request's trace in one payload
_TIMEOUT_S = 5.0                  # the collector is local; longer than this it is not coming

_HEX = re.compile(r"^[0-9a-f]+$")


def endpoint_of(value) -> str | None:
    """The OTLP endpoint a config value or environment variable asks for, or None for off.
    "default"/"1"/"true"/"on" mean the local collector; anything else must be an http(s) URL ending in /v1/traces
    (a bare host is accepted and completed), so a typo stops the start rather than tracing into nothing."""
    if value is None:
        return None
    if isinstance(value, bool):
        return DEFAULT_ENDPOINT if value else None
    v = str(value).strip()
    if not v or v.lower() in ("0", "false", "off", "none"):
        return None
    if v.lower() in ("1", "true", "on", "yes", "default", "local"):
        return DEFAULT_ENDPOINT
    if not re.match(r"^https?://", v):
        v = "http://" + v
    if not re.match(r"^https?://[\w.\-/:@]+$", v):
        raise ValueError(f"trace_otlp: {value!r} is not an OTLP endpoint like http://localhost:4318/v1/traces")
    v = v.rstrip("/")
    if not v.endswith("/v1/traces"):
        v += "/v1/traces"
    return v


def _rand(n: int) -> str:
    while True:
        h = os.urandom(n).hex()
        if h != "0" * (n * 2):
            return h


def _attr(key: str, value) -> dict:
    if isinstance(value, bool):
        v = {"boolValue": value}
    elif isinstance(value, int):
        v = {"intValue": str(value)}
    elif isinstance(value, float):
        v = {"doubleValue": value}
    else:
        v = {"stringValue": str(value)}
    return {"key": key, "value": v}


def _attrs(pairs) -> list:
    return [_attr(k, v) for k, v in pairs if v is not None and v != ""]


class TraceContext:
    """One request's ids: the trace it belongs to (the caller's, when it sent a traceparent) and this request's span."""

    def __init__(self, trace_id: str, span_id: str, parent_id: str | None, tracestate: str | None):
        self.trace_id, self.span_id, self.parent_id, self.tracestate = trace_id, span_id, parent_id, tracestate

    def traceparent(self) -> str:
        """The header a client can put on its next request, or read to line its own spans up with ours."""
        return f"00-{self.trace_id}-{self.span_id}-01"


def parse_traceparent(header, tracestate=None) -> TraceContext | None:
    """The caller's context, or None when there is none to follow.  W3C: `version-traceid-spanid-flags`; a version we
    do not know is read for the ids it declares before the last dash, and all-zero ids are not a parent."""
    if not header:
        return None
    parts = header.strip().split("-")
    if len(parts) != 4 or len(parts[1]) != 32 or len(parts[2]) != 16:
        return None
    if not _HEX.match(parts[1]) or not _HEX.match(parts[2]):
        return None
    if parts[1] == "0" * 32 or parts[2] == "0" * 16:
        return None
    return TraceContext(parts[1].lower(), parts[2].lower(), parts[2].lower(), tracestate)


class Tracing:
    """The per-request spans and their OTLP export.  One of these per server, held as `Service.tracing` (None: off)."""

    def __init__(self, endpoint: str, model: str = "", version: str = ""):
        self.endpoint = endpoint
        self.model, self.version = model, version
        self.instance = _rand(8)                  # one server, one instance id, so its traces group together
        self.dropped = 0
        self.exported = 0
        self._q: queue.Queue = queue.Queue(maxsize=512)
        self._thread = None
        self._lock = threading.Lock()
        self._warned = False

    # ------------------------------------------------------------------ per request
    def begin(self, record: dict, headers) -> TraceContext:
        """Open a request: its ids go on the record, so the span is built where the server already settles the record."""
        ctx = parse_traceparent(headers.get("traceparent"), headers.get("tracestate"))
        if ctx is None:
            ctx = TraceContext(_rand(16), _rand(8), None, None)
        else:                       # the caller's span is our parent; this request is a new span inside its trace
            ctx = TraceContext(ctx.trace_id, _rand(8), ctx.span_id, ctx.tracestate)
        record["_trace"] = ctx
        return ctx

    def finish(self, record: dict) -> None:
        """Close a request: build its spans from the record's numbers and hand them to the export thread."""
        ctx = record.get("_trace")
        if ctx is None:
            return
        spans = self._spans(record, ctx)
        if spans:
            self._enqueue({"resource": self._resource(), "scopeSpans":
                          [{"scope": {"name": SCOPE, "version": self.version or "1"}, "spans": spans}]})

    def _spans(self, record: dict, ctx: TraceContext) -> list:
        started = record.get("started_at") or time.time()
        wall = record.get("wallclock_s")
        if wall is None:
            wall = time.perf_counter() - record.get("_clock", time.perf_counter())
        base = int(started * 1_000_000_000)

        def span(name, t0, t1, kind, attrs, span_id=None, parent=None):
            if t1 <= t0:
                return None
            parent = parent or ctx.span_id
            return {"traceId": ctx.trace_id, "spanId": span_id or _rand(8), "name": name, "kind": kind,
                    **({"parentSpanId": parent} if parent else {}),
                    "startTimeUnixNano": str(base + int(t0 * 1e9)), "endTimeUnixNano": str(base + int(t1 * 1e9)),
                    "attributes": attrs, "status": {"code": 1}}

        state = record.get("state") or record.get("outcome") or "completed"
        err = state == "error"
        usage = record.get("usage") or {}
        timings = record.get("timings") or {}
        in_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
        out_tokens = usage.get("completion_tokens", usage.get("output_tokens"))

        root = span("strata.request " + (record.get("path") or ""), 0.0, max(wall, 1e-6), 2, _attrs([
            ("http.request.method", "POST"), ("url.path", record.get("path")),
            ("http.response.status_code", record.get("http_status")),
            ("gen_ai.operation.name", "chat"), ("gen_ai.provider.name", "strata"),
            ("gen_ai.request.model", record.get("model")), ("gen_ai.response.model", record.get("model")),
            ("gen_ai.usage.input_tokens", in_tokens), ("gen_ai.usage.output_tokens", out_tokens),
            ("strata.stream", bool(record.get("stream"))), ("strata.state", state),
            ("strata.time_to_first_token_s", record.get("first_token_s")),
            ("strata.queue_s", record.get("queue_s")), ("strata.load_s", record.get("load_s")),
            ("strata.wallclock_s", record.get("wallclock_s")),
        ]), span_id=ctx.span_id, parent=ctx.parent_id)
        if root is None:
            return []
        root["status"] = {"code": 2 if err else 1}
        if err and (record.get("error") or {}).get("message"):
            root["events"] = [{"name": "exception", "timestamp": root["endTimeUnixNano"],
                               "attributes": _attrs([("exception.message", str(record["error"]["message"])[:2000])])}]

        out = [root]
        q, load = record.get("queue_s") or 0.0, record.get("load_s") or 0.0
        first = record.get("first_token_s")
        # the queue and the load happened before the model read anything; their spans sit back to back from t0
        for name, t0, t1, attrs in (
                ("strata.queue", 0.0, q, []),
                ("strata.load", q, q + load, [])):
            s = span(name, t0, t1, 1, _attrs(attrs))
            if s:
                out.append(s)
        # prefill and decode are the engine's own milliseconds: prefill ends at the first token, decode starts there
        prompt_ms, decode_ms = timings.get("prompt_ms"), timings.get("predicted_ms") or timings.get("decode_ms")
        if first is not None and prompt_ms:
            s = span("strata.prefill", max(0.0, first - prompt_ms / 1000), first, 1, _attrs([
                ("gen_ai.usage.input_tokens", timings.get("prompt_n")),
                ("gen_ai.cached_tokens", timings.get("cache_n")),
                ("gen_ai.server.time_per_output_token_ms", timings.get("prompt_per_token_ms")),
                ("strata.prefill_tok_s", timings.get("prompt_per_second"))]))
            if s:
                out.append(s)
        if first is not None and decode_ms:
            s = span("strata.decode", first, first + decode_ms / 1000, 1, _attrs([
                ("gen_ai.usage.output_tokens", timings.get("predicted_n", out_tokens)),
                ("gen_ai.server.time_per_output_token_ms", timings.get("predicted_per_token_ms")),
                ("strata.decode_tok_s", timings.get("predicted_per_second"))]))
            if s:
                out.append(s)
        return out

    # ------------------------------------------------------------------ export
    def _resource(self) -> dict:
        return {"attributes": _attrs([("service.name", "strata"), ("service.instance.id", self.instance),
                                      ("service.version", self.version), ("strata.model", self.model),
                                      ("telemetry.sdk.language", "python"), ("telemetry.sdk.name", "strata"),
                                      ("telemetry.sdk.version", self.version or "1")])}

    def _enqueue(self, payload: dict) -> None:
        if self._thread is None:
            with self._lock:
                if self._thread is None:
                    self._thread = threading.Thread(target=self._loop, daemon=True, name="strata-otlp-export")
                    self._thread.start()
        try:
            self._q.put_nowait(payload)
        except queue.Full:                       # a collector that stopped taking spans must not cost the server memory
            self._dropped(1)

    def _dropped(self, n: int) -> None:
        self.dropped += n
        if not self._warned:
            self._warned = True
            print(f"[strata] tracing: {self.endpoint} is not taking spans ({n} dropped so far); requests are "
                  "unaffected - the endpoint, or STRATA_TRACE_OTLP, is wrong or the collector is not running",
                  flush=True)

    def _loop(self) -> None:
        while True:
            batch, spans = [], []
            try:
                first = self._q.get(timeout=FLUSH_S)
            except queue.Empty:
                continue
            batch.append(first)
            spans.extend(first["scopeSpans"][0]["spans"])
            while len(spans) < BATCH_MAX:
                try:
                    nxt = self._q.get_nowait()
                except queue.Empty:
                    break
                batch.append(nxt)
                spans.extend(nxt["scopeSpans"][0]["spans"])
            self._post(batch)

    def _post(self, batch: list) -> None:
        """One OTLP request per trace: a trace's spans stay in one payload, which is what a viewer reads as one trace."""
        for payload in batch:
            spans = payload["scopeSpans"][0]["spans"]
            body = json.dumps(payload, separators=(",", ":")).encode()
            req = urllib.request.Request(self.endpoint, data=body, method="POST",
                                         headers={"Content-Type": "application/json",
                                                  "Accept": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as r:
                    r.read()
                self.exported += len(spans)
            except (urllib.error.URLError, OSError, ValueError) as e:
                self._dropped(len(spans))
                if isinstance(e, urllib.error.HTTPError) and e.code in (400, 404, 415):
                    print(f"[strata] tracing: {self.endpoint} answered {e.code} - it is not an OTLP trace endpoint",
                          flush=True)
                    self._warned = True

    def close(self, drain_s: float = 2.0) -> None:
        """Give queued spans a moment to leave before the process ends (the thread is a daemon, so it would not)."""
        deadline = time.monotonic() + drain_s
        while not self._q.empty() and time.monotonic() < deadline:
            time.sleep(0.05)


def endpoint_from_config(cfg: dict, cli_value=None, env_name: str = "STRATA_TRACE_OTLP") -> str | None:
    """The OTLP endpoint this run asks for, or None for off.  --trace-otlp beats the config's "trace_otlp" beats
    $STRATA_TRACE_OTLP."""
    if cli_value is not None:
        value = cli_value
    else:
        value = (cfg.get("trace_otlp") if isinstance(cfg, dict) else None) or os.environ.get(env_name)
    return endpoint_of(value)
