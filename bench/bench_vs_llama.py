#!/usr/bin/env python3
"""Strata vs llama.cpp head-to-head (stream 1402-h).  Measurement only.

Both engines are driven through their OpenAI-compatible HTTP servers (/v1/chat/completions), with the same prompt
text, same sampler settings and the same GGUF.  Server-side timings (llama.cpp's names, which Strata's server also
emits) give prompt tok/s and decode tok/s; time to first token is the server's prompt_ms (non-streaming request,
so TTFT = prompt processing time).  Peak VRAM comes from nvidia-smi polling (minus the idle baseline), RAM from the
engine process tree (psutil RSS, which counts mmap'd file pages) and the system's used-memory delta.

    python bench_vs_llama.py prompts  --out DIR --tokenizer DIR --corpus FILE...     # build 4K / 32K prompt files
    python bench_vs_llama.py run      --machine CFG.json --out DIR --rounds N [--configs a,b] [--cells ...]
    python bench_vs_llama.py quality  --machine CFG.json --out DIR [--configs a,b]
    python bench_vs_llama.py summary  --out DIR

--trace-otlp URL (run/quality) is opt-in tracing through the repo's own exporter (serve/tracing.py, no new package;
"default" is http://localhost:4318/v1/traces, where Foundry Toolkit's collector listens).  One trace per config-round:
bench.round with bench.start / bench.warmup / bench.chat / bench.stop under it, and every request carries a W3C
traceparent, so the server's own span nests inside its bench.chat span and the row records the server's answer header
as "trace" - the rows line up with the server's traces, as evals/run_eval.py --trace does.  Without the flag nothing
here touches tracing and the rows are unchanged.

CFG.json: {"gguf": ..., "strata_cfg": ..., "llama_server": ..., "configs": [{"name","kind":"llama|strata","args":[...]}]}
Interleaving: engines cannot be loaded together (12 GB card, 64 GB RAM), so one ROUND runs every config once
(start, warm-up, cells, stop), and rounds alternate the config order.  Medians are taken over rounds.
The caller holds the machine's GPU lock.  Processes are stopped by PID only.
"""
import argparse, io, json, os, statistics, subprocess, sys, threading, time, urllib.request
from pathlib import Path

import psutil

CELLS = [("4k", 4000, 2), ("32k", 32000, 1)]       # (name, prompt tokens, reps per round)
TEMPS = [0.0, 0.7]
GEN = 256
PORT = 8190


def http(url, body=None, timeout=3600, headers=None, want_header=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        reply = json.loads(r.read())
    if want_header:                       # --trace-otlp: the server's span header rides along in the reply
        reply["_" + want_header] = r.headers.get(want_header)
    return reply


def vram_used_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.split()
        return int(out[0])
    except Exception:
        return -1


class Tracer:
    """--trace-otlp: the bench's own OTLP spans, exported by serve/tracing.py - the repo's exporter, so tracing
    adds no package.  Off by default: with no endpoint no header is sent, no span is built, and the rows are
    byte-identical to a run without the flag.

    One trace per config-round: bench.round with bench.start / bench.warmup / bench.chat / bench.stop under it.
    Every chat carries a W3C traceparent whose span id is the chat span's own, so the server's request span (a
    server started with tracing on) nests inside it, and the server's answer header is recorded in the row."""

    def __init__(self, value):
        root = str(Path(__file__).resolve().parents[1])
        if root not in sys.path:
            sys.path.insert(0, root)
        from serve import tracing as T
        self.T = T
        self.endpoint = T.endpoint_of(value)
        self.tr = T.Tracing(self.endpoint, model="strata-bench") if self.endpoint else None

    @property
    def on(self) -> bool:
        return self.tr is not None

    def new_trace(self):
        """This round's trace id and its root span id, so every phase hangs under one trace."""
        return (os.urandom(16).hex(), os.urandom(8).hex()) if self.on else (None, None)

    def traceparent(self, trace_id, span_id):
        return f"00-{trace_id}-{span_id}-01" if self.on else None

    def span(self, name, start_s, end_s, kind=1, attrs=(), trace_id=None, span_id=None, parent_id=None):
        if self.tr is None:
            return None
        return self.T.make_span(name, start_s, end_s, kind, attrs, trace_id=trace_id, span_id=span_id,
                                parent_id=parent_id)

    def emit(self, spans) -> None:
        if self.tr is not None:
            self.tr.emit([s for s in spans if s])

    def close(self) -> None:
        if self.tr is not None:
            self.tr.close()


class Sampler(threading.Thread):
    def __init__(self, pid):
        super().__init__(daemon=True)
        self.pid, self.stop, self.vram, self.rss, self.sysused = pid, False, 0, 0, 0

    def run(self):
        while not self.stop:
            self.vram = max(self.vram, vram_used_mib())
            try:
                p = psutil.Process(self.pid)
                self.rss = max(self.rss, sum(c.memory_info().rss for c in [p] + p.children(recursive=True)))
            except psutil.Error:
                pass
            self.sysused = max(self.sysused, psutil.virtual_memory().used)
            time.sleep(1.0)


class Engine:
    def __init__(self, machine, cfg):
        self.m, self.cfg = machine, cfg
        self.proc: subprocess.Popen | None = None
        self.log: io.TextIOWrapper | None = None
        self.start_end = self.stop_t0 = self.stop_end = None      # --trace-otlp: the phases' wall times

    def command(self):
        c = self.cfg
        if c["kind"] == "llama":
            return [self.m["llama_server"], "-m", self.m["gguf"], "--port", str(PORT), "--host", "127.0.0.1",
                    "-np", "1", "--no-webui", "--jinja"] + c["args"]
        scfg = json.load(open(self.m["strata_cfg"], encoding="utf-8-sig"))
        scfg["port"] = PORT
        scfg.pop("vision", None)
        args = [a for a in scfg["args"]]
        for k, v in c.get("override", {}).items():          # replace "--flag value" pairs
            if k in args:
                args[args.index(k) + 1] = v
            else:
                args += [k, v]
        scfg["args"] = args + c["args"]
        if self.m.get("strata_exe"):
            scfg["exe"] = self.m["strata_exe"]
        p = Path(self.out) / ("strata-cfg-%s.json" % c["name"])
        p.write_text(json.dumps(scfg, indent=1))
        return [sys.executable, str(Path(self.m["serve"]) / "server.py"), "--engine", "strata", "--config", str(p),
                "--port", str(PORT)]

    def start(self, out):
        self.out = out = str(Path(out).resolve())
        self.base_vram, self.base_used = vram_used_mib(), psutil.virtual_memory().used
        cmd = self.command()
        self.cmdline = " ".join(cmd)
        self.log = open(Path(out) / ("server-%s-%d.log" % (self.cfg["name"], int(time.time()))), "w", encoding="utf-8")
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, env=env,
                                     cwd=self.m.get("cwd"))
        self.t0 = time.time()
        self.sampler = Sampler(self.proc.pid)
        self.sampler.start()
        health = "http://127.0.0.1:%d/health" % PORT
        while time.time() - self.t0 < 1500:
            if self.proc.poll() is not None:
                raise RuntimeError("server exited rc=%s (see %s)" % (self.proc.returncode, self.log.name))
            try:
                r = http(health, timeout=5)
                if r.get("status", "ok") == "ok":
                    break
            except Exception:
                pass
            time.sleep(2)
        else:
            raise RuntimeError("server not ready in 1500 s")
        self.load_s = time.time() - self.t0
        self.start_end = time.time()

    def stop(self):
        self.stop_t0 = time.time()
        if self.proc:
            self.sampler.stop = True
            try:
                p = psutil.Process(self.proc.pid)
                kids = p.children(recursive=True)
                for c in kids + [p]:
                    try:
                        c.terminate()
                    except psutil.Error:
                        pass
                psutil.wait_procs(kids + [p], timeout=30)
                for c in kids + [p]:
                    try:
                        if c.is_running():
                            c.kill()
                    except psutil.Error:
                        pass
            except psutil.Error:
                pass
            if self.log is not None:
                self.log.close()
                self.log = None
            self.proc = None
            time.sleep(5)
        self.stop_end = time.time()

    def chat(self, prompt, temp, max_tokens=GEN, seed=1, traceparent=None):
        body = {"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                "temperature": temp, "top_k": 20, "top_p": 0.95, "seed": seed, "stream": False,
                "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}
        if temp == 0:
            body.pop("top_p")
            body.pop("top_k")
        t = time.time()
        r = http("http://127.0.0.1:%d/v1/chat/completions" % PORT, body,
                 headers={"traceparent": traceparent} if traceparent else None,
                 want_header="traceparent" if traceparent else None)
        wall = time.time() - t
        tm = r.get("timings") or {}
        u = r.get("usage") or {}
        return {"wall_s": round(wall, 2), "prompt_n": tm.get("prompt_n", u.get("prompt_tokens")),
                "prompt_ms": tm.get("prompt_ms"), "prompt_tps": tm.get("prompt_per_second"),
                "pred_n": tm.get("predicted_n", u.get("completion_tokens")), "pred_tps": tm.get("predicted_per_second"),
                "draft_n": tm.get("draft_n"), "draft_acc": tm.get("draft_n_accepted"),
                "server_traceparent": r.get("_traceparent"),
                "text": (r["choices"][0]["message"].get("content") or "")}


def cmd_prompts(a):
    sys.path.insert(0, str(Path(a.repo) / "tools"))
    import strata_tokenizer as ST
    tp = Path(a.tokenizer)
    vocab = json.loads((tp / "vocab.json").read_text(encoding="utf-8"))
    inv = {i: t for t, i in vocab.items()}
    toks: list[str] = [inv[i] for i in range(len(vocab))]
    tok = ST.Tokenizer(toks, (tp / "merges.txt").read_text(encoding="utf-8").split("\n"),
                       json.loads((tp / "token_type.json").read_text()))
    corpus = "\n\n".join(Path(f).read_text(encoding="utf-8", errors="replace") for f in a.corpus)
    print("corpus chars", len(corpus))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    allids = tok.encode(corpus)
    print("corpus tokens", len(allids))
    for name, n, _ in CELLS:
        if len(allids) < n:
            raise SystemExit("corpus too small for %s" % name)
        (out / ("prompt-%s.txt" % name)).write_text(tok.decode(allids[:n]), encoding="utf-8")
        print(name, n, "tokens")


def load_prompt(out, name):
    return (Path(out) / ("prompt-%s.txt" % name)).read_text(encoding="utf-8")


def cmd_run(a):
    m = json.load(open(a.machine))
    out = Path(a.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    cfgs = [c for c in m["configs"] if not a.configs or c["name"] in a.configs.split(",")]
    cells = [c for c in CELLS if not a.cells or c[0] in a.cells.split(",")]
    prompts = {c[0]: load_prompt(out, c[0]) for c in cells}
    tracer = Tracer(a.trace_otlp)
    if tracer.on:
        print("[bench] tracing to", tracer.endpoint, "- one trace per config-round, each request carries a "
              "traceparent (start the server with tracing on to nest its spans)", flush=True)
    try:
        for rnd in range(a.start_round, a.start_round + a.rounds):
            order = cfgs if rnd % 2 == 0 else cfgs[::-1]
            for cfg in order:
                eng = Engine(m, cfg)
                rec_path = out / "raw" / ("%s.jsonl" % cfg["name"])
                trace_id, root_id = tracer.new_trace()
                spans = []
                round_t0 = time.time()
                try:
                    eng.start(str(out))
                    spans.append(tracer.span("bench.start", eng.t0, eng.start_end, trace_id=trace_id, parent_id=root_id))
                    warm_t0 = time.time()
                    for warm, mx in (("Say hello.", 16), (prompts["4k"][:6000] if "4k" in prompts else "hi", 32)):
                        tp = tracer.traceparent(trace_id, os.urandom(8).hex())
                        spans.append(_bench_chat(eng, warm, 0.0, mx, tracer, trace_id, root_id, tp))
                    spans.append(tracer.span("bench.warmup", warm_t0, time.time(), trace_id=trace_id, parent_id=root_id))
                    for cname, n, reps in cells:
                        for t in TEMPS:
                            for k in range(reps):
                                nonce = "[bench %s r%d %s t%.1f k%d %d]\n" % (cfg["name"], rnd, cname, t, k, time.time())
                                span_id = os.urandom(8).hex()
                                tp = tracer.traceparent(trace_id, span_id)
                                t0 = time.time()
                                r = eng.chat(nonce + prompts[cname] + "\n\nSummarize the text above in three sentences.",
                                             t, traceparent=tp)
                                spans.append(tracer.span("bench.chat", t0, time.time(),
                                                         trace_id=trace_id, parent_id=root_id, span_id=span_id,
                                                         attrs=[("bench.cell", cname), ("bench.temp", t),
                                                                ("bench.rep", k),
                                                                ("gen_ai.usage.input_tokens", r["prompt_n"]),
                                                                ("gen_ai.usage.output_tokens", r["pred_n"]),
                                                                ("gen_ai.server.time_per_output_token_ms", r["pred_tps"]),
                                                                ("strata.prefill_tok_s", r["prompt_tps"])]))
                                if tp:
                                    r["trace"] = {"traceparent": tp, "server_traceparent": r.pop("server_traceparent", None)}
                                r.pop("server_traceparent", None)
                                r.update(config=cfg["name"], round=rnd, cell=cname, temp=t, rep=k,
                                         vram_peak_mib=eng.sampler.vram, vram_base_mib=eng.base_vram,
                                         rss_peak_gib=round(eng.sampler.rss / 2**30, 2),
                                         sys_used_delta_gib=round((eng.sampler.sysused - eng.base_used) / 2**30, 2),
                                         load_s=round(eng.load_s, 1), cmd=eng.cmdline if (rnd == a.start_round and k == 0) else None)
                                r["text"] = r["text"][:300]
                                with open(rec_path, "a", encoding="utf-8") as f:
                                    f.write(json.dumps(r) + "\n")
                                print(cfg["name"], rnd, cname, t, "pp %s tg %s draft %s/%s ttft %s" % (
                                    r["prompt_tps"], r["pred_tps"], r["draft_acc"], r["draft_n"], r["prompt_ms"]), flush=True)
                except Exception as e:
                    print("ERROR", cfg["name"], rnd, repr(e), flush=True)
                    with open(out / "raw" / "errors.log", "a") as f:
                        f.write("%s r%d %r\n" % (cfg["name"], rnd, e))
                finally:
                    eng.stop()
                    spans.append(tracer.span("bench.stop", eng.stop_t0, eng.stop_end, trace_id=trace_id, parent_id=root_id))
                    spans.append(tracer.span("bench.round", round_t0, time.time(), 2, trace_id=trace_id, span_id=root_id,
                                             attrs=[("bench.config", cfg["name"]), ("bench.round", rnd),
                                                    ("bench.kind", cfg["kind"]), ("bench.load_s", round(eng.load_s, 1))]))
                    tracer.emit(spans)
    finally:
        tracer.close()


QPROMPT = "Write a short story about a lighthouse keeper."


def _bench_chat(eng, prompt, temp, max_tokens, tracer, trace_id, parent_id, tp):
    """One warm-up chat as its own bench.chat span (the scored chats build their span in the loop)."""
    t0 = time.time()
    eng.chat(prompt, temp, max_tokens, traceparent=tp)
    return tracer.span("bench.chat", t0, time.time(), trace_id=trace_id, parent_id=parent_id,
                       span_id=tp.split("-")[2] if tp else None)


def cmd_quality(a):
    m = json.load(open(a.machine))
    out = Path(a.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    cfgs = [c for c in m["configs"] if not a.configs or c["name"] in a.configs.split(",")]
    res = {}
    tracer = Tracer(a.trace_otlp)
    try:
        for cfg in cfgs:
            eng = Engine(m, cfg)
            trace_id, root_id = tracer.new_trace()
            spans = []
            round_t0 = time.time()
            try:
                eng.start(str(out))
                spans.append(tracer.span("bench.start", eng.t0, eng.start_end, trace_id=trace_id, parent_id=root_id))
                warm_t0 = time.time()
                spans.append(_bench_chat(eng, "Say hello.", 0.0, 16, tracer, trace_id, root_id,
                                         tracer.traceparent(trace_id, os.urandom(8).hex())))
                spans.append(tracer.span("bench.warmup", warm_t0, time.time(), trace_id=trace_id, parent_id=root_id))
                for q in (QPROMPT, "Write a Python function that parses a CSV file into a list of dicts, with tests."):
                    span_id = os.urandom(8).hex()
                    tp = tracer.traceparent(trace_id, span_id)
                    t0 = time.time()
                    r = eng.chat(q, 0.0, 200, traceparent=tp)
                    spans.append(tracer.span("bench.chat", t0, time.time(), trace_id=trace_id, parent_id=root_id,
                                             span_id=span_id, attrs=[("bench.quality", True)]))
                    res.setdefault(cfg["name"], []).append(r["text"])
            finally:
                eng.stop()
                spans.append(tracer.span("bench.stop", eng.stop_t0, eng.stop_end, trace_id=trace_id, parent_id=root_id))
                spans.append(tracer.span("bench.round", round_t0, time.time(), 2, trace_id=trace_id, span_id=root_id,
                                         attrs=[("bench.config", cfg["name"]), ("bench.kind", cfg["kind"]),
                                                ("bench.quality", True)]))
                tracer.emit(spans)
    finally:
        tracer.close()
    (out / "raw" / "quality.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


def cmd_summary(a):
    out = Path(a.out)
    rows = {}
    for f in sorted((out / "raw").glob("*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            rows.setdefault((r["config"], r["cell"], r["temp"]), []).append(r)
    med = lambda xs: statistics.median(xs) if xs else float("nan")
    print("| config | ctx | temp | n | prompt tok/s | decode tok/s | TTFT s | draft acc | VRAM peak MiB (delta) | RSS GiB |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for (c, cell, t), rs in sorted(rows.items()):
        pp = [r["prompt_tps"] for r in rs if r["prompt_tps"]]
        tg = [r["pred_tps"] for r in rs if r["pred_tps"]]
        tt = [r["prompt_ms"] / 1000 for r in rs if r["prompt_ms"]]
        da = [r["draft_acc"] / r["draft_n"] for r in rs if r.get("draft_n")]
        print("| %s | %s | %.1f | %d | %.1f | %.2f | %.2f | %s | %d (+%d) | %.1f |" % (
            c, cell, t, len(rs), med(pp), med(tg), med(tt), ("%.2f" % med(da)) if da else "-",
            max(r["vram_peak_mib"] for r in rs), max(r["vram_peak_mib"] - r["vram_base_mib"] for r in rs),
            max(r["rss_peak_gib"] for r in rs)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prompts", "run", "quality", "summary"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--machine")
    ap.add_argument("--configs")
    ap.add_argument("--cells")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--start-round", type=int, default=0)
    ap.add_argument("--tokenizer")
    ap.add_argument("--repo")
    ap.add_argument("--corpus", nargs="*")
    ap.add_argument("--trace-otlp", default="", help="run/quality: OTLP endpoint for the bench's own spans "
                                                    "('default' = http://localhost:4318/v1/traces); off when empty")
    a = ap.parse_args()
    {"prompts": cmd_prompts, "run": cmd_run, "quality": cmd_quality, "summary": cmd_summary}[a.mode](a)
