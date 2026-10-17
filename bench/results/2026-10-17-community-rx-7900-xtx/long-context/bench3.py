#!/usr/bin/env python3
"""Three-way comparison of local models over OpenAI-compatible HTTP.

Adapted from the Strata repository's bench/bench_vs_llama.py (stream 1402-h).  The request body,
the sampler settings and the server-side `timings` fields are kept identical to that harness so
these numbers stay comparable with the project's own published ones.  What changed here:

  * cells and temperatures come from the command line (the original hard-codes 4k/32k)
  * each config names its own Strata config file (the original has one per machine)
  * per-config environment variables (needed on AMD: HIP_VISIBLE_DEVICES=1, or the HIP backend
    picks up the integrated gfx1036 as a second device and crashes)
  * VRAM sampling is best effort: nvidia-smi does not exist on AMD, -1 means "not measured"

    bench3.py prompts --out DIR --tokenizer DIR --repo REPO --corpus FILE... [--cells-spec S]
    bench3.py run     --machine CFG.json --out DIR --rounds N [--configs a,b] [--cells a,b]
    bench3.py summary --out DIR

CFG.json: {"gguf": ..., "llama_server": ..., "serve": ..., "cwd": ...,
           "configs": [{"name", "kind": "llama|strata", "args": [...],
                        "strata_cfg": ..., "env": {...}, "override": {...}}]}
Interleaving: the engines cannot be loaded together, so one ROUND runs every config once
(start, warm-up, cells, stop) and rounds alternate the config order.
"""
import argparse, json, os, statistics, subprocess, sys, threading, time, urllib.request
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None

TEMPS = [0.0]
GEN = 256
PORT = 8190
CELLS = []          # filled from --cells-spec


def parse_cells(spec):
    out = []
    for part in spec.split(","):
        name, n = part.split(":")
        out.append((name.strip(), int(n), 1))
    return out


def http(url, body=None, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def vram_used_mib():
    """Best effort: the NVIDIA tool on NVIDIA, -1 (not measured) on AMD."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.split()
        return int(out[0])
    except Exception:
        return -1


class Sampler(threading.Thread):
    def __init__(self, pid):
        super().__init__(daemon=True)
        self.pid, self.stop, self.vram, self.rss, self.sysused = pid, False, 0, 0, 0

    def run(self):
        while not self.stop:
            self.vram = max(self.vram, vram_used_mib())
            if psutil:
                try:
                    p = psutil.Process(self.pid)
                    self.rss = max(self.rss, sum(c.memory_info().rss for c in [p] + p.children(recursive=True)))
                    self.sysused = max(self.sysused, psutil.virtual_memory().used)
                except psutil.Error:
                    pass
            time.sleep(1.0)


class Engine:
    def __init__(self, machine, cfg):
        self.m, self.cfg, self.proc, self.log = machine, cfg, None, None

    def command(self):
        c = self.cfg
        if c["kind"] == "llama":
            return [self.m["llama_server"], "-m", self.m["gguf"], "--port", str(PORT), "--host", "127.0.0.1",
                    "-np", "1", "--no-webui", "--jinja"] + c["args"]
        scfg = json.load(open(c["strata_cfg"], encoding="utf-8-sig"))
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
        self.base_vram, self.base_used = vram_used_mib(), (psutil.virtual_memory().used if psutil else 0)
        cmd = self.command()
        self.cmdline = " ".join(cmd)
        self.log = open(Path(out) / ("server-%s-%d.log" % (self.cfg["name"], int(time.time()))), "w", encoding="utf-8")
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        env.update(self.cfg.get("env") or {})
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

    def stop(self):
        if self.proc:
            self.sampler.stop = True
            try:
                if psutil:
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
                else:
                    self.proc.terminate()
                    self.proc.wait(timeout=30)
            except Exception:
                pass
            self.log.close()
            self.proc = None
            time.sleep(5)

    def chat(self, prompt, temp, max_tokens=GEN, seed=1):
        body = {"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                "temperature": temp, "top_k": 20, "top_p": 0.95, "seed": seed, "stream": False,
                "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}
        if temp == 0:
            body.pop("top_p"), body.pop("top_k")
        t = time.time()
        r = http("http://127.0.0.1:%d/v1/chat/completions" % PORT, body)
        wall = time.time() - t
        tm = r.get("timings") or {}
        u = r.get("usage") or {}
        return {"wall_s": round(wall, 2), "prompt_n": tm.get("prompt_n", u.get("prompt_tokens")),
                "prompt_ms": tm.get("prompt_ms"), "prompt_tps": tm.get("prompt_per_second"),
                "pred_n": tm.get("predicted_n", u.get("completion_tokens")), "pred_tps": tm.get("predicted_per_second"),
                "draft_n": tm.get("draft_n"), "draft_acc": tm.get("draft_n_accepted"),
                "text": (r["choices"][0]["message"].get("content") or "")}


def cmd_prompts(a):
    sys.path.insert(0, str(Path(a.repo) / "tools"))
    import strata_tokenizer as ST
    tp = Path(a.tokenizer)
    vocab = json.loads((tp / "vocab.json").read_text(encoding="utf-8"))
    toks = [None] * len(vocab)
    for t, i in vocab.items():
        toks[i] = t
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
            raise SystemExit("corpus too small for %s (%d < %d)" % (name, len(allids), n))
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
    for rnd in range(a.start_round, a.start_round + a.rounds):
        order = cfgs if rnd % 2 == 0 else cfgs[::-1]
        for cfg in order:
            eng = Engine(m, cfg)
            rec_path = out / "raw" / ("%s.jsonl" % cfg["name"])
            try:
                eng.start(str(out))
                eng.chat("Say hello.", 0.0, 16)                              # warm-up, discarded
                for cname, n, reps in cells:
                    for t in TEMPS:
                        for k in range(reps):
                            nonce = "[bench %s r%d %s t%.1f k%d %d]\n" % (cfg["name"], rnd, cname, t, k, time.time())
                            r = eng.chat(nonce + prompts[cname] + "\n\nSummarize the text above in three sentences.", t)
                            r.update(config=cfg["name"], round=rnd, cell=cname, temp=t, rep=k,
                                     vram_peak_mib=eng.sampler.vram, vram_base_mib=eng.base_vram,
                                     rss_peak_gib=round(eng.sampler.rss / 2**30, 2) if psutil else None,
                                     sys_used_delta_gib=round((eng.sampler.sysused - eng.base_used) / 2**30, 2) if psutil else None,
                                     load_s=round(eng.load_s, 1), cmd=eng.cmdline if (rnd == a.start_round and k == 0) else None)
                            r["text"] = r["text"][:300]
                            with open(rec_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps(r) + "\n")
                            print(cfg["name"], rnd, cname, t, "prompt_n %s pp %s tg %s ttft %s" % (
                                r["prompt_n"], r["prompt_tps"], r["pred_tps"], r["prompt_ms"]), flush=True)
            except Exception as e:
                print("ERROR", cfg["name"], rnd, repr(e), flush=True)
                with open(out / "raw" / "errors.log", "a") as f:
                    f.write("%s r%d %r\n" % (cfg["name"], rnd, e))
            finally:
                eng.stop()


def cmd_summary(a):
    out = Path(a.out)
    rows = {}
    for f in sorted((out / "raw").glob("*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            rows.setdefault((r["config"], r["cell"], r["temp"]), []).append(r)
    med = lambda xs: statistics.median(xs) if xs else float("nan")
    print("| config | cell | prompt tokens | prefill tok/s | decode tok/s | TTFT s | load s |")
    print("|---|---|---|---|---|---|---|")
    order = {"4k": 0, "32k": 1, "128k": 2, "255k": 3}
    for (c, cell, t), rs in sorted(rows.items(), key=lambda kv: (kv[0][0], order.get(kv[0][1], 9))):
        pp = [r["prompt_tps"] for r in rs if r["prompt_tps"]]
        tg = [r["pred_tps"] for r in rs if r["pred_tps"]]
        tt = [r["prompt_ms"] / 1000 for r in rs if r["prompt_ms"]]
        pn = [r["prompt_n"] for r in rs if r["prompt_n"]]
        print("| %s | %s | %s | %.1f | %.2f | %.2f | %s |" % (
            c, cell, med(pn), med(pp), med(tg), med(tt),
            rs[0].get("load_s", "-")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prompts", "run", "summary"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--machine")
    ap.add_argument("--configs")
    ap.add_argument("--cells")
    ap.add_argument("--cells-spec", default="4k:4000,32k:32000,128k:128000,255k:255000")
    ap.add_argument("--temps", default=None)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--start-round", type=int, default=0)
    ap.add_argument("--tokenizer")
    ap.add_argument("--repo")
    ap.add_argument("--corpus", nargs="*")
    a = ap.parse_args()
    CELLS = parse_cells(a.cells_spec)
    if a.temps:
        TEMPS = [float(x) for x in a.temps.split(",")]
    {"prompts": cmd_prompts, "run": cmd_run, "summary": cmd_summary}[a.mode](a)
