#!/usr/bin/env python3
"""Evaluation harness for Strata's server: every query in evals/queries.jsonl goes through
/v1/chat/completions, and three metrics are measured per query:

- format:    the reply is a valid OpenAI chat.completion (choices, message, usage; the shape clients rely on)
- relevance: the reply contains the keywords the query expects (expect_all: every one, expect_any: at least one)
- time:      wall seconds per request, and tokens/s from the server's own timings (real engine) or usage (mock)

Run it against the mock engine (no GPU, no model, no downloads - the repo's test convention):

    python evals/run_eval.py                      # starts the mock server on a free port, scores, stops it

Against a running server (the real model):

    python evals/run_eval.py --base http://127.0.0.1:8095 [--api-key KEY]

Options: --queries FILE (default evals/queries.jsonl), --out DIR (default evals/results), --repeats N
(median of N runs per query, default 1), --min-relevance F (default 1.0), --timeout S (default 120),
--trace (opt-in, serve/tracing.py: the mock run traces every query through an in-process OTLP collector
and links each row to its server-side spans; against a real server it sends a W3C traceparent per request
and records the server's answer header, so rows line up with the server's traces - the server must be
started with tracing on). Without --trace nothing here touches tracing and the rows are unchanged.
Exit code 0 when every query passes all three, 1 otherwise - usable as a CI gate. The report is
evals/results/eval-<utc>.json with one row per query.

The mock answers in queries.jsonl are what the mock engine replays, so the mock run checks the server's
plumbing (API shape, token counts, latency) end to end; a real-model run checks the model's answers
against the same keywords.
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

QUERIES = Path(__file__).resolve().parent / "queries.jsonl"
FINISH_REASONS = ("stop", "length", "tool_calls")


class Collector:
    """An in-process OTLP collector for --trace on the mock run (the shape serve/test_tracing.py uses):
    it keeps every JSON payload POSTed to /v1/traces and answers 200, so the mock server's spans land
    in this process and the report can link each row to the server-side trace of its request."""

    def __init__(self):
        self.received = []
        self.httpd = None

    def start(self) -> str:
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        collector = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                collector.received.append(json.loads(self.rfile.read(n)))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b"{}")

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/v1/traces"

    def stop(self) -> None:
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()

    def payloads(self, want: int = 1, timeout: float = 5.0) -> list:
        """Wait until `want` payloads have arrived, so the exporter's thread gets its moment."""
        deadline = time.time() + timeout
        while len(self.received) < want and time.time() < deadline:
            time.sleep(0.05)
        return list(self.received)

    def spans_for(self, trace_id: str, timeout: float = 8.0) -> list:
        """Wait until this trace's spans have arrived, so the exporter's thread gets its moment: the
        server exports in batches (serve/tracing.py flushes a second after the first span waits), and
        the settle thread enqueues them after the answer was written - so a row's own trace is what to
        wait for, not just any payload."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            found = [s for s in self.spans() if s["traceId"] == trace_id]
            if found:
                return found
            time.sleep(0.05)
        return []

    def spans(self) -> list:
        return [s for p in self.payloads() for scope in p["scopeSpans"] for s in scope["spans"]]

    def traces(self) -> dict:
        out: dict[str, list] = {}
        for s in self.spans():
            out.setdefault(s["traceId"], []).append(s)
        return out


def load_queries(path: Path) -> list[dict]:
    """One JSON object per line: id, prompt, expect_all, expect_any, mock_answer, optional max_tokens/max_s."""
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        q = json.loads(line)
        if "id" not in q or "prompt" not in q:
            raise ValueError(f"{path}:{n}: a query needs an id and a prompt")
        out.append(q)
    if not out:
        raise ValueError(f"{path}: no queries")
    return out


def http_post(url: str, body: dict, timeout: float, api_key: str = "",
              headers: dict | None = None, want_trace: bool = False) -> dict:
    hdrs = {"Content-Type": "application/json", **(headers or {})}
    if api_key:
        hdrs["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            reply = json.loads(r.read())
        if want_trace:                       # --trace: the server's span header rides along in the reply
            tp = r.headers.get("traceparent")
            if tp:
                reply["_traceparent"] = tp
        return reply
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:400]
        raise RuntimeError(f"HTTP {e.code}: {detail}") from e


def traceparent_header() -> str:
    """A W3C traceparent for one request (serve/tracing.py follows it and makes the request a span
    inside this trace), with fresh ids so every query gets its own trace to line up with."""
    return f"00-{os.urandom(16).hex()}-{os.urandom(8).hex()}-01"


def check_format(reply: dict) -> list[str]:
    """Problems with the OpenAI chat.completion shape the clients rely on; [] when it is well-formed."""
    problems = []
    if reply.get("object") != "chat.completion":
        problems.append(f"object is {reply.get('object')!r}, not 'chat.completion'")
    for key in ("id", "created", "model"):
        if key not in reply:
            problems.append(f"{key} is missing")
    choices = reply.get("choices")
    if not isinstance(choices, list) or not choices:
        problems.append("choices is not a non-empty list")
        return problems
    msg = choices[0].get("message")
    if not isinstance(msg, dict):
        problems.append("choices[0].message is missing")
    elif msg.get("role") != "assistant":
        problems.append(f"message.role is {msg.get('role')!r}, not 'assistant'")
    elif not isinstance(msg.get("content"), str):
        problems.append("message.content is not a string")
    if choices[0].get("finish_reason") not in FINISH_REASONS:
        problems.append(f"finish_reason is {choices[0].get('finish_reason')!r}")
    usage = reply.get("usage")
    if not isinstance(usage, dict):
        problems.append("usage is missing")
    else:
        for key in ("prompt_tokens", "completion_tokens"):
            if not isinstance(usage.get(key), int) or usage[key] < 0:
                problems.append(f"usage.{key} is not a non-negative int")
    return problems


def relevance(text: str, q: dict) -> float:
    """The share of the query's expected keywords the reply contains (case-insensitive).
    expect_all: every keyword counts; expect_any: one found keyword counts as all of them."""
    need = list(q.get("expect_all") or [])
    any_of = list(q.get("expect_any") or [])
    lowered = text.lower()
    matched = sum(1 for k in need if k.lower() in lowered)
    if any_of:
        matched += 1 if any(k.lower() in lowered for k in any_of) else 0
    total = len(need) + (1 if any_of else 0)
    return 1.0 if total == 0 else matched / total


def score_query(base: str, q: dict, timeout: float, repeats: int, api_key: str,
                min_relevance: float = 1.0, trace: dict | None = None) -> dict:
    """Run one query `repeats` times; the last reply is scored for format/relevance, the median wall
    time is kept. The keyword gate is `min_relevance`: the reply must cover at least that share of the
    query's expected keywords (1.0 = every one).

    `trace` turns tracing on for this query: every request carries a W3C traceparent, and the scored
    (last) reply's `traceparent` response header is recorded as `trace.traceparent` - the server's own
    span for this request (serve/tracing.py). On the mock run `trace["collector"]` is the in-process
    OTLP collector, so the server-side spans of this query's trace are read out of it and listed as
    `trace.spans` (name + milliseconds). Against a real server the spans live in that server's
    collector, and the header alone is the link to them."""
    walls, tok_s, reply, error = [], None, None, None
    for _ in range(max(1, repeats)):
        body = {"model": "eval", "messages": [{"role": "user", "content": q["prompt"]}],
                "max_tokens": q.get("max_tokens", 256), "temperature": 0, "stream": False}
        t0 = time.perf_counter()
        try:
            reply = http_post(base + "/v1/chat/completions", body, timeout, api_key,
                              {"traceparent": traceparent_header()} if trace else None,
                              want_trace=trace is not None)
        except (RuntimeError, OSError, ValueError) as e:
            error = str(e)
            break
        wall = time.perf_counter() - t0
        walls.append(wall)
        usage = reply.get("usage") or {}
        timings = reply.get("timings") or {}
        n = usage.get("completion_tokens") or 0
        tok_s = timings.get("predicted_per_second") or (n / wall if wall > 0 and n else None)
    if error is not None:
        return {"id": q["id"], "pass": False, "error": error, "wall_s": None, "tok_s": None,
                "relevance": 0.0, "format_ok": False, "problems": [error], "content": None}
    assert reply is not None
    content = ((reply.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    problems = check_format(reply)
    rel = relevance(content, q)
    wall = statistics.median(walls)
    limit = q.get("max_s", timeout)
    row = {"id": q["id"], "wall_s": round(wall, 4), "tok_s": round(tok_s, 1) if tok_s else None,
           "relevance": round(rel, 3), "format_ok": not problems, "problems": problems,
           "content": content[:200], "pass": not problems and rel >= min_relevance - 1e-9 and wall <= limit}
    if trace is not None:
        row["trace"] = trace_row(reply, trace)
    return row


def trace_row(reply: dict, trace: dict) -> dict:
    """The tracing part of a row: the server's span header on the scored reply, and on the mock run the
    server-side spans of this query's trace, read from the in-process collector. A missing header means
    the server runs with tracing off - recorded as a problem, because --trace asked for traces."""
    header = reply.get("_traceparent")
    out = {"traceparent": header}
    if header is None:
        trace["missing"] = trace.get("missing", 0) + 1
        return out
    trace_id = header.split("-")[1]
    collector = trace.get("collector")
    if collector is None:
        return out
    spans = collector.spans_for(trace_id)
    out["spans"] = [{"name": s["name"],
                     "ms": round((int(s["endTimeUnixNano"]) - int(s["startTimeUnixNano"])) / 1e6, 2)}
                    for s in spans]
    return out


def evaluate(base: str, queries: list[dict], timeout: float, repeats: int,
             min_relevance: float, api_key: str = "", trace: dict | None = None) -> dict:
    rows = []
    for q in queries:
        rows.append(score_query(base, q, timeout, repeats, api_key, min_relevance, trace))
    walls = [r["wall_s"] for r in rows if r["wall_s"] is not None]
    passed = sum(1 for r in rows if r["pass"])
    report = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base": base,
        "queries": len(rows),
        "repeats": repeats,
        "thresholds": {"min_relevance": min_relevance, "timeout_s": timeout},
        "summary": {
            "passed": passed,
            "failed": len(rows) - passed,
            "median_wall_s": round(statistics.median(walls), 4) if walls else None,
            "max_wall_s": round(max(walls), 4) if walls else None,
            "min_relevance": min((r["relevance"] for r in rows), default=None),
        },
        "rows": rows,
    }
    if trace is not None:
        report["tracing"] = {"on": True,
                             "rows_with_server_span": sum(1 for r in rows if (r.get("trace") or {}).get("traceparent")),
                             "rows_without_server_span": trace.get("missing", 0)}
    return report


def start_mock_server(queries: list[dict], repeats: int, trace_endpoint: str = ""):
    """The mock engine replaying each query's mock_answer: MockEngine hands every request the next
    script, so each answer sits `repeats` times in a row (score_query makes repeats requests per
    query, and the last reply is the scored one; when the scripts run out, the last one repeats).

    Each script starts with the think-close, as every serve test script does (serve/test_server.py):
    the chat template opens the model's think block, so without it the whole mock answer
    lands in reasoning_content and message.content comes back null.

    `trace_endpoint` turns the server's own tracing on (serve/tracing.py): every request is exported
    as an OTLP trace to that collector. The mock engine says no DONE line, so its traces carry the
    request span with token counts and wall time - prefill and decode spans are the real engine's
    milliseconds and appear on a real run."""
    from serve.frontend import ChatTemplate, THINK_END
    from serve.server import ByteTokenizer, MockEngine, Service, Tracing, ROOT as SERVER_ROOT, serve
    tok = ByteTokenizer()
    close = THINK_END + "\n\n"        # the template opens the think block: the script closes it first
    scripts = [close + answer for q in queries
               for answer in [q.get("mock_answer", f"answer to {q['id']}")] * max(1, repeats)]
    engine = MockEngine(tok, scripts, max_context=8192)
    tpl = SERVER_ROOT / "serve" / "chat_template.jinja"
    svc = Service(engine, tok, ChatTemplate(tpl if tpl.exists() else SERVER_ROOT / "serve/chat_template.jinja"))
    if trace_endpoint:
        svc.tracing = Tracing(trace_endpoint, model="strata-eval")
    httpd = serve(svc, port=0)
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    return httpd, base


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", help="a running server, e.g. http://127.0.0.1:8095 (default: start the mock engine)")
    ap.add_argument("--queries", default=str(QUERIES), help="the queries file (default evals/queries.jsonl)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"), help="report directory")
    ap.add_argument("--repeats", type=int, default=1, help="runs per query, median kept (default 1)")
    ap.add_argument("--timeout", type=float, default=120.0, help="seconds per request (default 120)")
    ap.add_argument("--min-relevance", type=float, default=1.0, help="keyword coverage a query needs (default 1.0)")
    ap.add_argument("--api-key", default="", help="Bearer key for a server that requires one")
    ap.add_argument("--trace", action="store_true",
                    help="tracing on: the mock run exports every query through an in-process OTLP collector and "
                         "links each row to its server-side spans; against a real server each request carries a "
                         "W3C traceparent and the row records the server's answer header (server needs tracing on)")
    a = ap.parse_args(argv)

    queries = load_queries(Path(a.queries))
    httpd = None
    collector = None
    base = a.base
    trace: dict[str, Any] | None = {"collector": None} if a.trace else None
    try:
        if not base:
            if trace is not None:
                collector = Collector()
                trace["collector"] = collector
                httpd, base = start_mock_server(queries, max(1, a.repeats), collector.start())
                print(f"[eval] mock server on {base}, tracing to its collector", flush=True)
            else:
                httpd, base = start_mock_server(queries, max(1, a.repeats))
                print(f"[eval] mock server on {base}", flush=True)
        elif trace is not None:
            print(f"[eval] tracing on: each request carries a traceparent, {base}'s answer header is recorded "
                  "(the server's spans live in its collector)", flush=True)
        report = evaluate(base, queries, a.timeout, max(1, a.repeats), a.min_relevance, a.api_key, trace)
    finally:
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        if collector is not None:
            collector.stop()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = out_dir / ("eval-%s.json" % report["generated"].replace(":", "").replace("+0000", "Z"))
    name.write_text(json.dumps(report, indent=1), encoding="utf-8")

    print(f"{'query':22} {'format':7} {'relevance':9} {'wall_s':8} {'tok_s':7} pass", flush=True)
    for r in report["rows"]:
        print(f"{r['id']:22} {'ok' if r['format_ok'] else 'FAIL':7} {r['relevance']:9.2f} "
              f"{(r['wall_s'] if r['wall_s'] is not None else 0):8.3f} "
              f"{(r['tok_s'] if r['tok_s'] is not None else 0):7.1f} {'yes' if r['pass'] else 'NO'}", flush=True)
        for p in r["problems"]:
            print(f"  {r['id']}: {p}", flush=True)
    s = report["summary"]
    print(f"[eval] {s['passed']}/{report['queries']} passed; median wall {s['median_wall_s']} s; "
          f"report {name}", flush=True)
    if "tracing" in report:
        t = report["tracing"]
        print(f"[eval] tracing: {t['rows_with_server_span']}/{report['queries']} rows carry the server's "
              f"traceparent ({t['rows_without_server_span']} without)", flush=True)
    return 0 if s["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
