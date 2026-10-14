#!/usr/bin/env python3
"""Offline self-test of the benchmark scripts. No GPU, no Strata server: canned SSE streams, then a fake
OpenAI-compatible server on 127.0.0.1 (random port) that tune_bench.py, decode_bench.py and conc_bench.py run against.

    python3 test_bench_stream.py        (or: python3 tune_bench.py --selftest)
"""
import contextlib
import functools
import io
import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from bench_stream import parse_sse, stream  # noqa: E402

FAILS = []


def check(cond, what):
    print(("  ok    " if cond else "  FAIL  ") + what)
    if not cond:
        FAILS.append(what)


def ev(obj):
    return b"data: " + json.dumps(obj).encode() + b"\n"


def chunk(delta=None, finish=None, usage=None):
    c = {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}
    if usage is not None:
        c["usage"] = usage
    return ev(c)


def body(reasoning=("Let", " me", " think."), answer=("Hello", " world"), finish="stop", usage=True, done=True,
         tail=()):
    lines = [b": keep-alive\n", b"\n", chunk({"role": "assistant", "content": ""})]
    lines += [chunk({"reasoning_content": r}) for r in reasoning]
    lines += [chunk({"content": c}) for c in answer]
    lines += list(tail)
    if finish is not None:
        u = {"prompt_tokens": 12, "completion_tokens": len(reasoning) + len(answer),
             "prompt_tokens_details": {"cached_tokens": 0}} if usage else None
        lines.append(chunk({}, finish, u))
    if done:
        lines.append(b"data: [DONE]\n")
    return lines


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        self.t += 0.01
        return self.t


def parse(lines):
    return parse_sse(lines, 100.0, clock=Clock())


def test_parser():
    print("stream validation (canned SSE):")
    good = parse(body())
    check(good["status"] == "ok", "good stream -> ok")
    check(good["completion_tokens"] == 5 and good["finish_reason"] == "stop", "good stream: usage and finish_reason kept")
    check(good["tps"] is not None and good["ttft"] is not None, "good stream: decode rate and TTFT computed")
    check(good["text"] == "Hello world" and good["reasoning"] == "Let me think.", "answer and reasoning kept apart")

    err = parse(body(answer=("Hel",), finish=None, done=False,
                     tail=[ev({"error": {"type": "server_error", "message": "engine died; the next request restarts it"}}),
                           b"data: [DONE]\n"]))
    check(err["status"] == "server_error" and "engine died" in err["error"], "error event after partial output -> server_error")
    check(err["tps"] is None, "error stream: no decode rate")

    trunc = parse(body(finish=None, done=False))
    check(trunc["status"] == "truncated", "stream cut before finish and [DONE] -> truncated")
    trunc2 = parse(body(done=False))
    check(trunc2["status"] == "truncated", "finish chunk but no [DONE] (premature EOF) -> truncated")
    nousage = parse(body(usage=False))
    check(nousage["status"] == "no_usage" and nousage["tps"] is None, "no usage.completion_tokens -> no_usage, no rate")
    bad = parse(body(finish="error"))
    check(bad["status"] == "bad_finish", "finish_reason 'error' -> bad_finish")
    length = parse(body(finish="length"))
    check(length["status"] == "ok", "finish_reason 'length' (max_tokens reached) -> ok")
    junk = parse([b"data: {not json\n", b"data: [DONE]\n"])
    check(junk["status"] == "bad_event", "invalid JSON event -> bad_event")

    def broken():
        yield from body(done=False, finish=None)[:4]
        raise ConnectionResetError("peer reset")
    rd = parse(broken())
    check(rd["status"] == "read_error", "connection reset mid-stream -> read_error")

    # the review's reproduction: same answer, same length, different reasoning
    a = parse(body(reasoning=("A", "B", "C")))
    b = parse(body(reasoning=("X", "Y", "Z")))
    check(a["hashes"]["answer"] == b["hashes"]["answer"], "same answer -> same answer hash")
    check(a["hashes"]["reasoning"] != b["hashes"]["reasoning"], "different reasoning -> different reasoning hash")
    check(a["hashes"]["combined"] != b["hashes"]["combined"], "different reasoning -> different combined hash")
    c = parse(body(finish="length"))
    check(c["hashes"]["combined"] != good["hashes"]["combined"], "different finish_reason -> different combined hash")
    # split boundaries between reasoning and answer must not collide (no plain concatenation)
    d = parse(body(reasoning=("ab",), answer=("c",)))
    e = parse(body(reasoning=("a",), answer=("bc",)))
    check(d["hashes"]["combined"] != e["hashes"]["combined"], "reasoning/answer boundary is part of the hash")


def test_schema():
    print("malformed field types (canned SSE): bad_event record, no exception:")
    head = body(finish=None, done=False)
    cases = [
        ('{"usage":"bad"}', ev({"usage": "bad"})),
        ('{"choices":[null]}', ev({"choices": [None]})),
        ("numeric delta.content", chunk({"content": 5})),
        ("numeric delta.reasoning_content", chunk({"reasoning_content": 1.5})),
        ('{"choices":{}} (not a list)', ev({"choices": {}})),
        ('delta is a string', ev({"choices": [{"index": 0, "delta": "x", "finish_reason": None}]})),
        ("numeric finish_reason", ev({"choices": [{"index": 0, "delta": {}, "finish_reason": 1}]})),
        ('usage.completion_tokens "5" (a string)', ev({"choices": [], "usage": {"completion_tokens": "5"}})),
        ("usage.completion_tokens true (a bool)", ev({"choices": [], "usage": {"completion_tokens": True}})),
        ("negative usage.prompt_tokens", ev({"choices": [], "usage": {"prompt_tokens": -1, "completion_tokens": 5}})),
        ("prompt_tokens_details is a list", ev({"usage": {"completion_tokens": 5, "prompt_tokens_details": []}})),
        ("cached_tokens 1.5", ev({"usage": {"completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 1.5}}})),
        ('{"timings":"fast"}', ev({"timings": "fast"})),
        ('timings.predicted_per_second "99"', ev({"timings": {"predicted_per_second": "99"}})),
    ]
    for what, bad_line in cases:
        try:
            r = parse(head + [bad_line] + body(reasoning=(), answer=())[3:])
        except Exception as e:                                   # noqa: BLE001
            check(False, f"{what} -> raised {type(e).__name__}: {e}")
            continue
        check(r["status"] == "bad_event" and r["tps"] is None and "malformed" in (r["error"] or ""),
              f"{what} -> bad_event")
    # tolerated: null usage on content chunks, a chunk without choices, an event with none of the read fields
    ok = parse(body(tail=[ev({"usage": None, "choices": [{"index": 0, "delta": {"content": "!"}, "finish_reason": None}]}),
                          ev({"id": "x", "object": "chat.completion.chunk"}),
                          ev({"choices": [{"index": 0, "finish_reason": None}]})]))
    check(ok["status"] == "ok", '"usage": null, a chunk without choices or delta, an event without the read fields -> ok')
    # missing fields: no completion_tokens in usage -> no_usage; usage without details -> cached_tokens None
    nu = parse(body()[:-2] + [chunk({}, "stop", {"prompt_tokens": 3}), b"data: [DONE]\n"])
    check(nu["status"] == "no_usage", "usage without completion_tokens -> no_usage")
    nd = parse(body()[:-2] + [chunk({}, "stop", {"completion_tokens": 5}), b"data: [DONE]\n"])
    check(nd["status"] == "ok" and nd["cached_tokens"] is None and nd["prompt_tokens"] is None,
          "usage with only completion_tokens -> ok, prompt and cached tokens None")


# ---------------------------------------------------------------- fake server
class Fake(BaseHTTPRequestHandler):
    mode = {"fail_label": None}
    seen = []
    lock = threading.Lock()

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n))
        text = req["messages"][-1]["content"]
        with Fake.lock:
            Fake.seen.append(req)
            nth = len(Fake.seen)                                    # 1-based ordinal of this request
        if req["model"] == "busy":
            self.send_response(409)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": {"type": "model_busy", "message": "busy"}}')
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        k = min(req["max_tokens"], 6)
        words = [f" w{i}" for i in range(k)]
        trig = Fake.mode["fail_label"]
        die = (trig is not None and trig in text) or nth in Fake.mode.get("fail_nth", ())
        try:
            self.wfile.write(chunk({"role": "assistant", "content": ""}))
            for i, w in enumerate(words):
                if die and i == 2:
                    if Fake.mode.get("how") == "eof":
                        self.wfile.flush()
                        return                                      # premature EOF
                    if Fake.mode.get("how") == "schema":            # well-formed JSON, wrong field types
                        self.wfile.write(ev({"choices": [None], "usage": "bad"}))
                        self.wfile.write(b"data: [DONE]\n\n")
                        return
                    self.wfile.write(ev({"error": {"type": "server_error", "message": "engine died"}}))
                    self.wfile.write(b"data: [DONE]\n\n")
                    return
                key = "reasoning_content" if i < 2 else "content"
                self.wfile.write(chunk({key: w}))
                self.wfile.flush()
                time.sleep(0.002)
            pt = max(1, len(text) // 4)
            self.wfile.write(chunk({}, "length" if k == req["max_tokens"] else "stop",
                                   {"prompt_tokens": pt, "completion_tokens": k,
                                    "prompt_tokens_details": {"cached_tokens": 0}}))
            self.wfile.write(b"data: [DONE]\n\n")
        except OSError:
            pass


@contextlib.contextmanager
def fake_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    srv.daemon_threads = True
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    finally:
        srv.shutdown()
        srv.server_close()


def quiet(fn, *args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = fn(*args)
    return rc, buf.getvalue()


def test_end_to_end():
    import conc_bench
    import decode_bench
    import tune_bench
    print("end to end against a fake server:")
    with fake_server() as url, tempfile.TemporaryDirectory() as tmp:
        Fake.mode = {"fail_label": None}
        r = stream(url, "m", [{"role": "user", "content": "hi"}], 0, 8)
        check(r["status"] == "ok" and r["completion_tokens"] == 6, "HTTP round trip ok")
        r = stream(url, "busy", [{"role": "user", "content": "hi"}], 0, 8)
        check(r["status"] == "http_error" and "409" in r["error"], "HTTP 409 -> http_error record, no exception")
        r = stream("http://127.0.0.1:9/v1", "m", [{"role": "user", "content": "hi"}], 0, 8, timeout=5)
        check(r["status"] == "connect_error", "connection refused -> connect_error record")
        for bad_url in ("not-a-url", "", None):
            try:
                r = stream(bad_url, "m", [{"role": "user", "content": "hi"}], 0, 8)
                check(r["status"] == "client_error" and r["tps"] is None and "t_end" in r and r["max_tokens"] == 8,
                      f"invalid endpoint {bad_url!r} -> client_error record, no exception")
            except Exception as e:                           # noqa: BLE001
                check(False, f"invalid endpoint {bad_url!r} -> raised {type(e).__name__}: {e}")
        Fake.mode = {"fail_label": "hi", "how": "eof"}
        r = stream(url, "m", [{"role": "user", "content": "hi"}], 0, 8)
        check(r["status"] == "truncated", "server closes mid-stream -> truncated")
        Fake.mode = {"fail_label": "hi", "how": "schema"}
        r = stream(url, "m", [{"role": "user", "content": "hi"}], 0, 8)
        check(r["status"] == "bad_event", "malformed event over HTTP -> bad_event record, no exception")

        # tune_bench full run (reads longctx.txt.gz lazily), one decode request fails with an error event
        Fake.mode = {"fail_label": "train leaves", "how": "error"}
        out = os.path.join(tmp, "t.json")
        rc, log = quiet(tune_bench.main, ["--url", url, "--model", "m", "--name", "t", "--runs", "2", "--out", out])
        d = json.load(open(out))
        check(rc == 1, "tune_bench exits 1 when a request failed")
        check(d["format"] == 2 and len(d["failures"]) == 4, "tune_bench lists the 4 failed reasoning runs under failures")
        check(d["decode"]["reasoning@0.0"]["tps"] is None and d["decode"]["reasoning@0.0"]["n_failed"] == 2,
              "failed runs are excluded from the cell median")
        check(d["summary"]["greedy_geo"] is None, "summary is null when one prompt has no successful run")
        check(d["decode"]["prose@0.0"]["n_ok"] == 2 and d["decode"]["prose@0.0"]["tps"] is not None,
              "the other prompts are measured")
        check(set(d["cold"]) == {"4k", "16k", "64k"} and d["cold"]["64k"]["status"] == "ok",
              "cold reads ran from the .gz fixture")
        check(d["warm"]["turns_ok"] == 3, "warm turns ran")
        check(all("hashes" in q for q in d["requests"]) and "text" not in d["requests"][0], "records keep hashes, not texts")
        greedy = d["decode"]["prose@0.0"]["hash_combined"]
        check(greedy[0] == greedy[1] and greedy[0] is not None, "repeat greedy runs hash the same")
        seeds = [q.get("seed") for q in Fake.seen if "printing press" in q["messages"][-1]["content"]]
        check(seeds == [None, None, 1000, 1001], "greedy sends no seed, sampled sends 1000, 1001")

        Fake.mode = {"fail_label": None}
        rc, _ = quiet(tune_bench.main, ["--url", url, "--model", "m", "--name", "t", "--runs", "1", "--quick",
                                        "--out", out])
        d = json.load(open(out))
        check(rc == 0 and d["summary"]["greedy_geo"] is not None and not d["cold"], "tune_bench --quick ok")

        # conc_bench: same request set at every level
        Fake.seen = []
        cout = os.path.join(tmp, "c.json")
        rc, _ = quiet(conc_bench.main, ["--url", url, "--model", "m", "--conc", "1,2,4", "--out", cout])
        c = json.load(open(cout))
        check(rc == 0 and set(c["summary"]) == {"1", "2", "4"}, "conc_bench ran C=1,2,4")
        sets = {}
        for q in c["requests"]:
            sets.setdefault((q["concurrency"], q["round"]), []).append((q["prompt_id"], q["seed"], q["max_tokens"]))
        check(len({tuple(v) for v in sets.values()}) == 1, "every level and round sends the same ids, seeds, max_tokens")
        texts = {}
        for q in Fake.seen[1:]:
            texts.setdefault(q["seed"], set()).add(q["messages"][-1]["content"])
        check(all(len(v) == 1 for v in texts.values()), "warm mode: identical prompt text per request id everywhere")
        check(any(not r["measured"] for r in c["rounds"]), "the priming pass is recorded and marked unmeasured")
        check(c["summary"] == conc_bench.summarize(c["rounds"], c["requests"]), "summary re-derives from the records")
        rids = [r["round_id"] for r in c["rounds"]]
        check(len(set(rids)) == len(rids) == 7 and {q["round_id"] for q in c["requests"]} == set(rids)
              and c["stopped"] is None, "every round (priming included) has a unique round_id, used by its requests")
        r1 = [q for q in c["requests"] if q["concurrency"] == 1 and q["round"] == 1]
        check(all(r1[i]["start"] >= r1[i - 1]["end"] - 1e-3 for i in range(1, len(r1))), "C=1 is sequential")

        Fake.mode = {"fail_label": "LRU cache", "how": "error"}
        rc, _ = quiet(conc_bench.main, ["--url", url, "--model", "m", "--conc", "2", "--cache", "cold", "--out", cout])
        c = json.load(open(cout))
        s = c["summary"]["2"]
        check(rc == 1 and s["requests_failed"] == 2 and s["rounds_excluded"] == 2 and s["aggregate_tps"] is None,
              "conc_bench: failed requests excluded, their rounds left out of the aggregate")

        # conc_bench: a malformed event in one request -> bad_event record, the run goes on and exits 1
        Fake.mode = {"fail_label": "LRU cache", "how": "schema"}
        rc, _ = quiet(conc_bench.main, ["--url", url, "--model", "m", "--conc", "1,2", "--rounds", "1", "--cache",
                                        "cold", "--out", cout])
        c = json.load(open(cout))
        check(rc == 1 and len(c["requests"]) == 8 and sum(q["status"] == "bad_event" for q in c["requests"]) == 2,
              "conc_bench: malformed events become bad_event records, every request recorded")

        # conc_bench: a failed priming request stops the run before measurement
        Fake.seen = []
        Fake.mode = {"fail_label": None, "fail_nth": (3,), "how": "error"}   # 1 = warm-up, 2-5 = priming at C=1
        rc, log = quiet(conc_bench.main, ["--url", url, "--model", "m", "--conc", "1,2", "--out", cout])
        c = json.load(open(cout))
        check(rc == 1, "conc_bench: priming failure -> exit 1")
        check(len(c["failures"]) == 1 and c["failures"][0]["phase"] == "priming"
              and c["failures"][0]["status"] == "server_error", "priming failure recorded under failures")
        check(len(Fake.seen) == 5 and not any(r["measured"] for r in c["rounds"]) and c["summary"] == {}
              and c["stopped"], "no measured round is sent after a priming failure; JSON written, 'stopped' says why")

        # conc_bench: the client code raises inside a worker -> client_error record, nothing lost
        real = conc_bench.stream

        def flaky(url_, model, messages, *a, **k):
            if "LRU cache" in messages[-1]["content"]:
                raise RuntimeError("client bug")
            return real(url_, model, messages, *a, **k)
        Fake.mode = {"fail_label": None}
        conc_bench.stream = flaky
        try:
            rc, _ = quiet(conc_bench.main, ["--url", url, "--model", "m", "--conc", "2", "--cache", "cold",
                                            "--out", cout])
        except Exception as e:                                          # noqa: BLE001
            rc = f"raised {e!r}"
        finally:
            conc_bench.stream = real
        c = json.load(open(cout))
        errs = [q for q in c["requests"] if q["status"] == "client_error"]
        check(rc == 1 and len(c["requests"]) == 8 and all(q is not None for q in c["requests"]) and len(errs) == 2
              and all(q["seed"] == 101 and "client bug" in q["error"] for q in errs),
              "conc_bench: a worker exception yields a client_error record for that request")

        # conc_bench: argument validation, before any request is sent
        for bad in (["--conc", "1,1"], ["--conc", "0,2"], ["--conc", "-1"], ["--conc", "1,x"], ["--rounds", "0"],
                    ["--repeat", "0"], ["--max-tokens", "0"]):
            Fake.seen = []
            err = io.StringIO()
            try:
                with contextlib.redirect_stderr(err):
                    quiet(conc_bench.main, ["--url", url, "--model", "m", "--out", cout] + bad)
                code = 0
            except SystemExit as e:
                code = e.code
            check(code == 2 and not Fake.seen, f"conc_bench rejects {' '.join(bad)} before sending")

        # tune_bench and decode_bench: nonpositive counts are usage errors, before any request is sent
        cases = [("tune_bench", tune_bench.main, ["--url", url, "--model", "m", "--name", "t", "--out",
                                                  os.path.join(tmp, "v.json"), "--quick", "--runs", n]) for n in ("0", "-1")]
        cases += [("decode_bench", functools.partial(_run_decode, decode_bench),
                   ["--url", url, "--model", "m", "--json", os.path.join(tmp, "v.json")] + bad)
                  for bad in (["--runs", "0"], ["--runs", "-2"], ["--max-tokens", "0"])]
        for name, fn, argv in cases:
            Fake.seen = []
            err = io.StringIO()
            try:
                with contextlib.redirect_stderr(err):
                    quiet(fn, argv)
                code = 0
            except SystemExit as e:
                code = e.code
            check(code == 2 and not Fake.seen and not os.path.exists(os.path.join(tmp, "v.json")),
                  f"{name} rejects {' '.join(argv[-2:])} before sending")

        # summarize joins requests to rounds by round_id: two rounds with the same (concurrency, round) stay apart
        rr = [{"concurrency": 1, "round": 1, "round_id": i, "measured": True, "wall_s": 2.0} for i in (1, 2)]
        qq = [{"concurrency": 1, "round": 1, "round_id": i, "status": "ok", "completion_tokens": 100, "tps": 50.0,
               "ttft": 0.1, "cached_tokens": 0} for i in (1, 2)]
        sm = conc_bench.summarize(rr, qq)["1"]
        check(sm["aggregate_tps_rounds"] == [50.0, 50.0], "summarize: round_id join, no double counting (50 tok/s, not 100)")
        pub_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "conc-D-batch2.json")
        if os.path.exists(pub_path):
            pub = json.load(open(pub_path))
            check(conc_bench.summarize(pub["rounds"], pub["requests"]) == pub["summary"],
                  "conc-D-batch2.json (format 2, no round_id) still re-derives its summary")
        else:
            print("  skip  conc-D-batch2.json not next to the scripts")

        # decode_bench
        Fake.mode = {"fail_label": "essay", "how": "eof"}
        dout = os.path.join(tmp, "d.json")
        rc, _ = quiet(_run_decode, decode_bench, ["--url", url, "--model", "m", "--runs", "1", "--json", dout])
        dd = json.load(open(dout))
        check(rc == 1 and len(dd["failures"]) == 2 and "prose@0.0" not in dd["summary"]
              and "code@0.0" in dd["summary"], "decode_bench: truncated prose runs excluded from the summary")


def _run_decode(mod, argv):
    old = sys.argv
    sys.argv = ["decode_bench.py"] + argv
    try:
        return mod.main()
    finally:
        sys.argv = old


def main():
    test_parser()
    test_schema()
    test_end_to_end()
    print(f"\n{'ALL PASSED' if not FAILS else f'{len(FAILS)} FAILED'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
