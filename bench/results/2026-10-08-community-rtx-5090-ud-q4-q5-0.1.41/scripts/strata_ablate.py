#!/usr/bin/env python3
"""strata_ablate.py - one-knob-at-a-time tuning matrix for a Strata server.

What it does, unattended:
  1. Finds the running production Strata server (serve/server.py --engine strata), its config, and how
     it was started (systemd user/system unit, or a plain process). Nothing is stopped yet.
  2. Builds the benchmark prompts once with Strata's own tokenizer, so every cell sees byte-identical
     prompts at exact token counts.
  3. Stops production, then for each cell: writes a variant of the production config (one knob changed),
     starts a private server on --bench-port, warms it, runs the measured prompts, records engine timings,
     expert-cache hit rate, PCIe share, draft acceptance, GPU power/clock/PCIe link, CPU use by other
     processes, and a hash of every greedy output. Then stops it.
  4. Cells: baseline, one cell per knob value, a combination of the winners, baseline again (drift check).
  5. ALWAYS restores production (systemctl start, or relaunching the exact original command) and waits
     for it to answer - on success, crash, Ctrl-C, or kill.
  6. Appends one JSON line per cell to the ledger and writes REPORT.md in the run folder.

Typical use on the box that runs Strata:
    python3 strata_ablate.py --dry-run            # preflight + prompt build + plan; touches nothing
    nohup python3 strata_ablate.py --budget-min 120 > ~/strata-ablate.out 2>&1 &
    tail -f ~/strata-ablation/latest/progress.log

Stdlib only. Re-executes itself under Strata's .venv python (for the tokenizer and chat template).
"""
import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

CLK_TCK = os.sysconf("SC_CLK_TCK")
LOG = None  # file handle for progress.log


def log(msg):
    line = f"{datetime.now().strftime('%H:%M:%S')} {msg}"
    try:
        print(line, flush=True)
    except OSError:  # terminal went away; progress.log still gets it
        pass
    if LOG:
        LOG.write(line + "\n")
        LOG.flush()


# --------------------------------------------------------------------------- process discovery

def argval(argv, flag, default=None):
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return default


def find_production(port):
    """The running serve/server.py --engine strata process (on `port` if given)."""
    hits = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            argv = [a.decode(errors="replace") for a in (p / "cmdline").read_bytes().split(b"\0") if a]
        except OSError:
            continue
        if not any(a.endswith("server.py") for a in argv) or argval(argv, "--engine") != "strata":
            continue
        pport = int(argval(argv, "--port", "8095"))
        if port and pport != port:
            continue
        try:
            cwd = os.readlink(p / "cwd")
            env = dict(kv.split("=", 1) for kv in
                       (p / "environ").read_bytes().decode(errors="replace").split("\0") if "=" in kv)
            cgroup = (p / "cgroup").read_text()
        except OSError as e:
            sys.exit(f"found the server (pid {p.name}) but cannot read its /proc entries: {e}. "
                     "Run this script as the same user that runs Strata.")
        hits.append({"pid": int(p.name), "argv": argv, "port": pport, "cwd": cwd, "env": env, "cgroup": cgroup})
    if not hits:
        sys.exit(f"no running 'serve/server.py --engine strata' found{' on port %d' % port if port else ''}. "
                 "Start production first, or pass --port.")
    if len(hits) > 1:
        sys.exit("several Strata servers are running: " + ", ".join(f"pid {h['pid']} port {h['port']}" for h in hits)
                 + ". Pass --port to pick the production one.")
    return hits[0]


def systemd_unit(cgroup):
    """(unit, is_user_unit) from a /proc/PID/cgroup text, or (None, None) for a plain process."""
    path = cgroup.strip().splitlines()[-1].split(":", 2)[-1]
    parts = [x for x in path.split("/") if x]
    for comp in reversed(parts):
        if comp.endswith(".service") and not comp.startswith("user@"):
            return comp, any(x.startswith("user@") for x in parts)
    return None, None


# --------------------------------------------------------------------------- config editing

def get_arg(args, flag):
    return argval(args, flag)


def set_arg(args, flag, value):
    """Return a copy of args with `flag value` set (value None removes the flag and its value)."""
    out, i = [], 0
    while i < len(args):
        a = args[i]
        if a == flag:
            i += 2 if i + 1 < len(args) and not args[i + 1].startswith("--") else 1
            continue
        if a.startswith(flag + "="):
            i += 1
            continue
        out.append(a)
        i += 1
    if value is not None:
        out += [flag, str(value)]
    return out


def same_num(a, b):
    try:
        return a is not None and b is not None and abs(float(a) - float(b)) < 1e-9
    except ValueError:
        return a == b


def knob_cells(base):
    """One cell per alternative value of each knob, derived from the production config."""
    a, env = base["args"], base.get("env") or {}
    cells = []

    def add(knob, value, label, args=None, envd=None):
        cfg = json.loads(json.dumps(base))
        if args is not None:
            cfg["args"] = args
        if envd is not None:
            cfg["env"] = envd
        cells.append({"name": f"{knob}={label}", "knob": knob, "value": label, "cfg": cfg})

    pf = get_arg(a, "--pcie-frac")
    for v in ("0.00", "0.20", "0.55"):
        if not same_num(v, pf):
            add("pcie-frac", v, v, set_arg(a, "--pcie-frac", v))
    pre = get_arg(a, "--prefill") or "auto"
    alt = "auto" if pre.startswith("auto:") else "auto:32768"
    add("prefill", alt, alt, set_arg(a, "--prefill", alt))
    kvr = get_arg(a, "--kv-resident")
    alt = "65536" if not same_num(kvr, 65536) else "32768"
    add("kv-resident", alt, alt, set_arg(a, "--kv-resident", alt))
    mc = get_arg(a, "--max-context")
    alt = "262144" if not same_num(mc, 262144) else "131072"
    add("max-context", alt, alt, set_arg(a, "--max-context", alt))
    smp = get_arg(a, "--spec-min-p")
    alt = "0.50" if not same_num(smp, 0.5) else "0.70"
    add("spec-min-p", alt, alt, set_arg(a, "--spec-min-p", alt))
    fused = str(env.get("STRATA_PF_FUSED", "0")) == "1"
    e2 = dict(env)
    if fused:
        e2.pop("STRATA_PF_FUSED", None)
    else:
        e2["STRATA_PF_FUSED"] = "1"
    add("pf-fused", "off" if fused else "on", "off" if fused else "on", envd=e2)
    return cells


MODELS = {  # name -> (family, shard 1 under its model folder); Q5: the 6-byte-padded copy of shard 1 when present
    "iq3_s": ("ista", "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"),
    "ud-q4_k_xl": ("unsloth", "UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"),
    "ud-q5_k_xl": ("unsloth", "UD-Q5_K_XL-padded/Qwen3.8-Flash-Next-UD-Q5_K_XL-00001-of-00006.gguf"),
    "q8_0": ("unsloth", "Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf"),
}


def model_cells(base, a):
    """One cell per model on the engine at --engine-root: production's settings, with only what the model needs
    changed - its pack and GGUF, the RAM budget for the Unsloth files, and no image encoder (the Unsloth family is
    text-only in setup, and the test build has no strata-vision)."""
    root = a.engine_root
    cells, missing = [], []
    for spec in [x for x in a.models.split(",") if x]:
        if spec == "prod-novision":   # production's engine and model, image encoder off: separates vision from engine
            cfg = json.loads(json.dumps(base))
            cfg.pop("vision", None)
            cfg["args"] = [x for x in set_arg(base["args"], "--vram-reserve-mib", None) if x != "--vision"]
            cells.append({"name": "model=prod-novision", "knob": "model", "value": "prod-novision", "cfg": cfg})
            continue
        vis = spec.endswith("+v")                  # name...+v: with images (the build's strata-vision, --mmproj)
        spec = spec[:-2] if vis else spec
        m, _, b = spec.partition("@b")             # name@bN: this cell with --resident-budget-gib N
        budget = int(b) if b else a.budget_gib
        if m not in MODELS:
            sys.exit(f"unknown model {m}; known: {', '.join(MODELS)}")
        fam, rel = MODELS[m]
        shard1 = (a.ista_dir if fam == "ista" else a.unsloth_dir) / rel
        pack = root / "packs" / m
        args = list(base["args"])
        for flag in ("--pack", "--native", "--ple-gguf", "--expert-profile", "--resident-budget-gib",
                     "--vram-reserve-mib"):
            args = set_arg(args, flag, None)
        args = [x for x in args if x != "--vision"]
        args = ["--pack", str(pack), "--native", str(shard1)] + \
            (["--ple-gguf", str(shard1).replace("-00001-of-", "-00002-of-")] if fam == "ista" else []) + \
            ["--expert-profile", str(root / "data" / "expert-profile.bin")] + args
        if fam == "unsloth":
            args += ["--resident-budget-gib", str(budget)]
        cfg = json.loads(json.dumps(base))
        base_vis = cfg.pop("vision", None)
        if vis:
            vexe = root / "build-vision" / "bin" / "strata-vision"
            missing += [f"{m}+v: {x}" for x in (vexe, a.mmproj) if not Path(x).exists()]
            v = dict(base_vis or {"gpu": True, "max_tokens": 1024})
            v.update({"exe": str(vexe), "mmproj": str(a.mmproj), "model": str(shard1)})
            cfg["vision"] = v
            args += ["--vision", "--vram-reserve-mib", "700"]
        cfg.update({"exe": str(root / "build" / "strata"), "args": args, "cwd": str(root),
                    "tokenizer": str(pack / "tokenizer"), "model_name": f"bench-{m}"})
        missing += [f"{m}: {x}" for x in (cfg["exe"], str(shard1), str(pack / "native_experts.txt"),
                                          str(root / ".venv" / "bin" / "python")) if not Path(x).exists()]
        label = (f"{m}@b{budget}" if fam == "unsloth" else m) + ("+vision" if vis else "")
        cells.append({"name": f"model={label}@{root.name}", "knob": "model", "value": label, "cfg": cfg,
                      "root": str(root), "python": str(root / ".venv" / "bin" / "python")})
    if missing:
        sys.exit("STOP, missing:\n  " + "\n  ".join(missing))
    return cells


# --------------------------------------------------------------------------- telemetry

class Sampler(threading.Thread):
    """nvidia-smi + whole-machine CPU every 2 s."""
    FIELDS = "power.draw,clocks.sm,temperature.gpu,utilization.gpu,memory.used,pcie.link.gen.current,pcie.link.width.current"

    def __init__(self, path):
        super().__init__(daemon=True)
        self.rows, self.stop_ev, self.path = [], threading.Event(), path

    @staticmethod
    def cpu_jiffies():
        f = [int(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
        return sum(f), f[3] + f[4]  # total, idle+iowait

    def run(self):
        prev = self.cpu_jiffies()
        with open(self.path, "w") as fh:
            fh.write("unix,power_w,sm_mhz,temp_c,util_pct,mem_mib,pcie_gen,pcie_width,cpu_busy_pct\n")
            while not self.stop_ev.is_set():
                try:
                    out = subprocess.check_output(["nvidia-smi", f"--query-gpu={self.FIELDS}",
                                                   "--format=csv,noheader,nounits", "-i", "0"],
                                                  text=True, timeout=10).strip().split(", ")
                    cur = self.cpu_jiffies()
                    dt, di = cur[0] - prev[0], cur[1] - prev[1]
                    prev = cur
                    busy = 100.0 * (dt - di) / dt if dt else 0.0
                    row = [time.time()] + [float(x) for x in out] + [busy]
                    self.rows.append(row)
                    fh.write(",".join(f"{x:.1f}" for x in row) + "\n")
                    fh.flush()
                except (OSError, subprocess.SubprocessError, ValueError):
                    pass
                self.stop_ev.wait(2)

    def window(self, t0, t1):
        rows = [r for r in self.rows if t0 <= r[0] <= t1] or [r for r in self.rows if r[0] >= t0][:1]
        if not rows:
            return {}
        col = lambda i: [r[i] for r in rows]
        return {"power_w_mean": round(statistics.mean(col(1)), 1), "sm_mhz_mean": round(statistics.mean(col(2))),
                "temp_c_max": max(col(3)), "gpu_util_mean": round(statistics.mean(col(4)), 1),
                "pcie_gen_min": min(col(6)), "pcie_width_min": min(col(7)),
                "cpu_busy_mean": round(statistics.mean(col(8)), 1)}


def proc_cpu():
    d = {}
    for p in Path("/proc").iterdir():
        if p.name.isdigit():
            try:
                s = (p / "stat").read_text()
                r = s.rfind(")")
                f = s[r + 2:].split()
                d[int(p.name)] = (s[s.find("(") + 1:r], int(f[11]) + int(f[12]), int(f[2]))  # comm, ticks, pgrp
            except (OSError, ValueError, IndexError):
                pass
    return d


def other_cpu(before, after, exclude_pgid, top=5):
    use = []
    for pid, (comm, ticks, pgrp) in after.items():
        if pgrp == exclude_pgid or pid == os.getpid():
            continue
        dt = ticks - before.get(pid, (comm, ticks, pgrp))[1] if pid in before else ticks
        if dt > 0:
            use.append((dt / CLK_TCK, f"{comm}[{pid}]"))
    use.sort(reverse=True)
    return {"other_cpu_s_total": round(sum(u for u, _ in use), 2),
            "other_cpu_top": [f"{n} {u:.2f}s" for u, n in use[:top]]}


def gpu_mem_used():
    try:
        return float(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                                              "-i", "0"], text=True, timeout=10).strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return -1.0


def gpu_apps():
    try:
        return subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                        "--format=csv,noheader"], text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        return "?"


# --------------------------------------------------------------------------- HTTP

def http_json(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def port_free(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def chat(url, model, content, max_tokens, timeout=3600):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": content}],
                       "temperature": 0, "reasoning_effort": "none", "max_tokens": max_tokens,
                       "stream": True, "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    texts, first, finish, usage = [], None, None, {}
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for line in resp:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                break
            chunk = json.loads(payload)
            usage = chunk.get("usage") or usage
            for ch in chunk.get("choices", []):
                d = ch.get("delta", {})
                t = (d.get("content") or "") + (d.get("reasoning_content") or "")
                if t:
                    first = first if first is not None else time.perf_counter() - t0
                    texts.append(t)
                finish = ch.get("finish_reason") or finish
    return {"ttft_s": first, "elapsed_s": time.perf_counter() - t0, "finish": finish, "usage": usage,
            "text": "".join(texts)}


# --------------------------------------------------------------------------- prompts

FILLER = "\n".join(f"def task_{i:05d}(value: int) -> int: return (value * {(i % 97) + 1} + {i}) % 100003"
                   for i in range(90000)).split("\n")
ENDING = ("\n\nWrite a detailed explanation of the code above. Discuss deterministic integer transforms, modulo "
          "arithmetic, testing, naming, complexity, and maintainability. Write at least 600 words.")


def build_prompts(root, tokenizer_dir, targets, runs):
    sys.path[:0] = [str(root), str(root / "tools")]
    from strata_tokenizer import Tokenizer
    from serve.frontend import ChatTemplate, openai_to_messages
    vocab = json.loads((tokenizer_dir / "vocab.json").read_text(encoding="utf-8"))
    toks = [None] * len(vocab)
    for t, n in vocab.items():
        toks[n] = t
    tok = Tokenizer(toks, (tokenizer_dir / "merges.txt").read_text(encoding="utf-8").splitlines(),
                    json.loads((tokenizer_dir / "token_type.json").read_text(encoding="utf-8")))
    tmpl = ChatTemplate(tokenizer_dir / "chat_template.jinja")

    def count(text):
        msgs, tools, kw = openai_to_messages({"messages": [{"role": "user", "content": text}]})
        return len(tok.encode(tmpl.render(msgs, tools, **kw), parse_special=True))

    def text(nonce, lines):
        return f"Benchmark nonce: {nonce}.\nReview this synthetic Python module:\n" + "\n".join(FILLER[:lines]) + ENDING

    prompts = {}
    plan = [("warm", 32768, "series-warm00-trial-00")] + [
        (f"{t}-r{r}", t, f"series-{t:06d}-trial-{r:02d}") for t in targets for r in range(1, runs + 1)]
    lines_for = {}
    c0, c1 = count(text("series-probe-trial-00", 0)), count(text("series-probe-trial-00", 1000))
    per_line = (c1 - c0) / 1000.0
    for key, target, nonce in plan:
        if target not in lines_for:  # bracket from the per-line estimate, then binary search inside it
            est = int((target - c0) / per_line)
            lo, hi = max(1, int(est * 0.95)), min(len(FILLER), int(est * 1.05) + 10)
            while lo > 1 and count(text(nonce, lo)) > target:
                lo = max(1, lo // 2)
            while hi < len(FILLER) and count(text(nonce, hi)) <= target:
                hi = min(len(FILLER), hi * 2)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                lo, hi = (mid, hi) if count(text(nonce, mid)) <= target else (lo, mid - 1)
            lines_for[target] = lo
        n = lines_for[target]
        while count(text(nonce, n)) > target:
            n -= 1
        c = count(text(nonce, n))
        prompts[key] = {"target": target, "tokens": c, "text": text(nonce, n)}
        log(f"prompt {key}: {c} tokens")
    return prompts


# --------------------------------------------------------------------------- one cell

def run_cell(cell, ctx):
    name, cfg = cell["name"], cell["cfg"]
    cdir = ctx["out"] / "cells" / f"{ctx['seq']:02d}-{re.sub(r'[^A-Za-z0-9_.=-]', '_', name)}"
    ctx["seq"] += 1
    cdir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(json.dumps(cfg))
    cfg["port"] = ctx["bench_port"]
    cfg["log"] = str(cdir / "engine.log")
    (cdir / "config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    url = f"http://127.0.0.1:{ctx['bench_port']}"
    rec = {"cell": name, "knob": cell.get("knob"), "value": cell.get("value"), "status": "started",
           "args": cfg["args"], "env": cfg.get("env") or {}, "runs": []}
    croot, cpy = Path(cell.get("root") or ctx["root"]), cell.get("python") or ctx["python"]
    cmd = [cpy, str(croot / "serve" / "server.py"), "--engine", ctx["engine_kind"],
           "--config", str(cdir / "config.json"), "--port", str(ctx["bench_port"]), "--host", "127.0.0.1"]
    log(f"=== cell {name}: starting server")
    t_start = time.time()
    out = open(cdir / "server.out", "w")
    proc = subprocess.Popen(cmd, cwd=cfg.get("cwd") or str(croot), env=ctx["prod_env"],
                            stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    ctx["cell_proc"] = proc
    sampler = Sampler(cdir / "telemetry.csv")
    sampler.start()
    try:
        deadline = time.time() + ctx["load_timeout"]
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited with code {proc.returncode} during load (see server.out)")
            try:
                st = http_json(url + "/v1/status", timeout=5)
                if st.get("loaded", True):
                    break
            except (urllib.error.URLError, OSError, ValueError):
                pass
            if time.time() > deadline:
                raise RuntimeError("server not ready before --load-timeout")
            time.sleep(3)
        model = cfg.get("model_name") or "strata"
        w = chat(url, model, "Reply with exactly the word READY.", 16, timeout=ctx["load_timeout"])
        rec["load_s"] = round(time.time() - t_start, 1)
        m0 = http_json(url + "/metrics")
        rec["engine_info"] = m0.get("engine", {})
        log(f"cell {name}: ready in {rec['load_s']}s, engine {rec['engine_info'].get('version')} "
            f"expert_slots={rec['engine_info'].get('expert_slots')} vram_free={rec['engine_info'].get('vram_free_mib')}MiB")
        chat(url, model, ctx["prompts"]["warm"]["text"], 64)  # warm the expert cache, discarded
        for key, p in ctx["prompts"].items():
            if key == "warm":
                continue
            if proc.poll() is not None:
                raise RuntimeError(f"server died mid-cell with code {proc.returncode}")
            before = proc_cpu()
            t0 = time.time()
            r = chat(url, model, p["text"], 256)
            t1 = time.time()
            e = (http_json(url + "/metrics").get("requests") or [{}])[0]
            gen, dms, pms = e.get("engine_generated") or 0, e.get("decode_ms") or 0, e.get("prompt_ms") or 0
            read = (e.get("prompt_tokens") or 0) - (e.get("reused") or 0)
            row = {"key": key, "target": p["target"], "expected_tokens": p["tokens"],
                   "prompt_tokens": e.get("prompt_tokens"), "reused": e.get("reused"),
                   "prefill_tok_s": round(read / (pms / 1000), 1) if pms else None,
                   "decode_tok_s": round(gen / (dms / 1000), 1) if dms else None,
                   "prompt_ms": pms, "decode_ms": dms, "generated": gen,
                   "hit_rate": e.get("hit_rate"), "pcie_share": e.get("pcie_share"),
                   "drafts_offered": e.get("drafts_offered"), "drafts_accepted": e.get("drafts_accepted"),
                   "ttft_s": round(r["ttft_s"], 3) if r["ttft_s"] else None, "finish": r["finish"],
                   "out_sha": hashlib.sha256(r["text"].encode()).hexdigest()[:16], "text": r["text"][:4000],
                   "token_mismatch": e.get("prompt_tokens") != p["tokens"]}
            row.update(sampler.window(t0, t1))
            row.update(other_cpu(before, proc_cpu(), os.getpgid(proc.pid)))
            rec["runs"].append(row)
            log(f"cell {name} {key}: prefill {row['prefill_tok_s']} decode {row['decode_tok_s']} "
                f"hit {row['hit_rate']} pcie {row['pcie_share']} drafts {row['drafts_accepted']}/{row['drafts_offered']}"
                f" power {row.get('power_w_mean')}W other_cpu {row['other_cpu_s_total']}s")
        rec["status"] = "ok"
    except Exception as ex:  # a failed cell is a finding, not a reason to stop
        rec["status"] = f"failed: {ex}"
        log(f"cell {name} FAILED: {ex}")
    finally:
        sampler.stop_ev.set()
        stop_proc(proc)
        out.close()
        ctx["cell_proc"] = None
        rec["cell_s"] = round(time.time() - t_start, 1)
        wait_gpu_idle(ctx)
    rec["summary"] = summarize(rec["runs"])
    (cdir / "result.json").write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def stop_proc(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(90)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(30)


def wait_gpu_idle(ctx, timeout=180):
    t = time.time() + timeout
    while time.time() < t:
        if gpu_mem_used() <= ctx["idle_mib"] + 600:
            return
        time.sleep(3)
    log(f"warning: GPU memory still {gpu_mem_used():.0f} MiB after stop; apps: {gpu_apps()}")


def med(vals):
    vals = [v for v in vals if v is not None]
    return round(statistics.median(vals), 3) if vals else None


def summarize(runs):
    s = {}
    for t in sorted({r["target"] for r in runs}):
        rs = [r for r in runs if r["target"] == t]
        dec = [r["decode_tok_s"] for r in rs if r["decode_tok_s"]]
        offered = sum(r["drafts_offered"] or 0 for r in rs)
        s[str(t)] = {"n": len(rs), "decode_med": med(dec), "decode_min": min(dec, default=None),
                     "decode_max": max(dec, default=None),
                     "prefill_med": med([r["prefill_tok_s"] for r in rs]),
                     "ttft_med": med([r["ttft_s"] for r in rs]), "hit_med": med([r["hit_rate"] for r in rs]),
                     "pcie_share_med": med([r["pcie_share"] for r in rs]),
                     "draft_accept": round(sum(r["drafts_accepted"] or 0 for r in rs) / offered, 3) if offered else None,
                     "power_w": med([r.get("power_w_mean") for r in rs]),
                     "pcie_gen_min": min((r.get("pcie_gen_min") for r in rs if r.get("pcie_gen_min")), default=None),
                     "other_cpu_s": med([r["other_cpu_s_total"] for r in rs]),
                     "reused_max": max((r["reused"] or 0 for r in rs), default=0)}
    return s


# --------------------------------------------------------------------------- comparison

def rel(a, b):
    return (a - b) / b if a is not None and b else None


def compare(rec, base):
    out = {}
    for t, s in rec["summary"].items():
        b = base["summary"].get(t, {})
        out[t] = {"decode": rel(s.get("decode_med"), b.get("decode_med")),
                  "prefill": rel(s.get("prefill_med"), b.get("prefill_med"))}
    return out


def same_outputs(rec, base):
    b = {r["key"]: r["text"] for r in base["runs"]}
    pairs = [(r["text"], b[r["key"]]) for r in rec["runs"] if r["key"] in b]
    if not pairs:
        return None
    pref = [len(os.path.commonprefix([x, y])) for x, y in pairs]
    return {"identical": sum(x == y for x, y in pairs), "n": len(pairs), "common_prefix_chars_med": med(pref)}


def verdict(rec, base, thr_dec, thr_pre):
    c = compare(rec, base)
    dec = [v["decode"] for v in c.values() if v["decode"] is not None]
    pre = [v["prefill"] for v in c.values() if v["prefill"] is not None]
    if rec["status"] != "ok" or not dec:
        return "n/a", None
    dm, pm = statistics.mean(dec), statistics.mean(pre) if pre else 0.0
    if dm > thr_dec and min(pre, default=0) > -thr_pre:
        return "WIN decode", dm + pm
    if pm > thr_pre and min(dec) > -thr_dec:
        return "WIN prefill", dm + pm
    if dm < -thr_dec or pm < -thr_pre:
        return "worse", dm + pm
    return "within noise", dm + pm


# --------------------------------------------------------------------------- report + ledger

def pct(x):
    return "-" if x is None else f"{x * 100:+.1f}%"


def write_report(ctx, recs, base, base_end, combo_note):
    targets = [str(t) for t in ctx["targets"]]
    L = [f"# Strata ablation {ctx['run_id']}", "",
         f"Host {socket.gethostname()}, Strata checkout `{ctx['commit']}`, engine "
         f"{(base or {}).get('engine_info', {}).get('version', '?')}, production config `{ctx['prod_config']}`.",
         f"Prompts: {', '.join(targets)} tokens x {ctx['runs']} runs per cell, greedy, reasoning off, 256-token cap, "
         "identical prompts in every cell, fresh engine per cell, one discarded 32K warm-up.",
         f"Noise thresholds used for verdicts: decode {ctx['thr_dec'] * 100:.0f}%, prefill {ctx['thr_pre'] * 100:.0f}% "
         "(mean change across prompt sizes, against the first baseline).", ""]
    if base and base_end and base_end["status"] == "ok":
        d = compare(base_end, base)
        L.append("**Drift check (baseline end vs start):** " + ", ".join(
            f"{t}: decode {pct(d[t]['decode'])} prefill {pct(d[t]['prefill'])}" for t in targets if t in d)
            + ". Effects smaller than this are not distinguishable from run-to-run drift.")
        so = same_outputs(base_end, base)
        if so:
            L.append(f"Same-config output repeatability: {so['identical']}/{so['n']} identical greedy outputs "
                     f"(median common prefix {so['common_prefix_chars_med']} chars). This is the floor for the "
                     "'same outputs' column.")
        L.append("")
    hdr = "| cell | status | " + " | ".join(f"decode {t}" for t in targets) + " | " + \
          " | ".join(f"prefill {t}" for t in targets) + " | hit @max | PCIe share @max | drafts | W | other CPU s | same outputs | verdict |"
    L += ["## Results (medians; change vs first baseline)", "", hdr, "|" + "---|" * (hdr.count("|") - 1)]
    for r in recs:
        s, c = r["summary"], compare(r, base) if base else {}
        big = s.get(targets[-1], {})
        v = r.get("verdict", "")
        so = same_outputs(r, base) if base and r is not base else None
        cells = [r["cell"], "ok" if r["status"] == "ok" else r["status"][:40]]
        f1 = lambda x: "-" if x is None else f"{x:,.1f}" if x < 1000 else f"{x:,.0f}"
        cells += [f"{f1(s.get(t, {}).get('decode_med'))} ({pct(c.get(t, {}).get('decode'))})" for t in targets]
        cells += [f"{f1(s.get(t, {}).get('prefill_med'))} ({pct(c.get(t, {}).get('prefill'))})" for t in targets]
        da = [x.get("draft_accept") for x in s.values() if x.get("draft_accept")]
        cells += [str(big.get("hit_med", "-")), str(big.get("pcie_share_med", "-")),
                  f"{statistics.mean(da) * 100:.0f}%" if da else "-", str(big.get("power_w", "-")),
                  str(med([x.get('other_cpu_s') for x in s.values()])),
                  f"{so['identical']}/{so['n']}" if so else "-", v]
        L.append("| " + " | ".join(cells) + " |")
    L += ["", "Decode ranges (min-max per size) and every per-run number are in `cells/*/result.json`; "
          "GPU/CPU samples in `cells/*/telemetry.csv`.", ""]
    L += ["## Combination", "", combo_note, ""]
    wins = [r for r in recs if r.get("verdict", "").startswith("WIN") and r.get("knob")]
    L += ["## Suggested production change", ""]
    if ctx.get("models_mode"):
        L.append("Model cells: each changes the weights and the engine build, so outputs differ by design and the "
                 "verdicts are speed only. Quality is measured separately (strata_quality.py). Model cells run "
                 "without the image encoder; the baseline is production as configured.")
    elif wins:
        L.append("Knob values that cleared the noise threshold on their own (check the 'same outputs' column "
                 "before adopting anything that changes answers, e.g. pf-fused):")
        for r in wins:
            L.append(f"- `{r['knob']}` -> `{r['value']}` ({r['verdict']})")
    else:
        L.append("No single knob cleared the noise threshold. Production is at or near a local optimum for these knobs.")
    mism = sum(1 for r in recs for x in r["runs"] if x["token_mismatch"])
    reused = max((x["reused"] or 0 for r in recs for x in r["runs"]), default=0)
    L += ["", "## Validity checks", "",
          f"- Runs where the engine's prompt count differed from the local tokenizer: {mism}.",
          f"- Max reused (cached) prompt tokens in any measured run: {reused} (should be 0).",
          f"- Production restore: {ctx.get('restore_status', 'not reached')}.", ""]
    (ctx["out"] / "REPORT.md").write_text("\n".join(L))


def ledger_append(ctx, rec, base):
    line = {"ts": datetime.now().isoformat(timespec="seconds"), "run_id": ctx["run_id"], "host": socket.gethostname(),
            "commit": ctx["commit"], "engine": rec.get("engine_info", {}).get("version"), "cell": rec["cell"],
            "knob": rec.get("knob"), "value": rec.get("value"), "status": rec["status"], "args": rec["args"],
            "env": rec["env"], "load_s": rec.get("load_s"), "summary": rec["summary"],
            "vs_baseline": compare(rec, base) if base and rec is not base else None, "verdict": rec.get("verdict"),
            "same_outputs": same_outputs(rec, base) if base and rec is not base else None,
            "engine_info": rec.get("engine_info"), "run_dir": str(ctx["out"])}
    ctx["ledger"].parent.mkdir(parents=True, exist_ok=True)
    with open(ctx["ledger"], "a") as fh:
        fh.write(json.dumps(line) + "\n")


# --------------------------------------------------------------------------- main

def main():
    global LOG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8080, help="production server port (default 8080)")
    ap.add_argument("--bench-port", type=int, default=8097, help="private port for test servers")
    ap.add_argument("--budget-min", type=float, default=120, help="total wall-clock budget, restore included")
    ap.add_argument("--targets", default="4096,32768,128000")
    ap.add_argument("--runs", type=int, default=5, help="measured runs per prompt size per cell")
    ap.add_argument("--out", type=Path, default=Path.home() / "strata-ablation")
    ap.add_argument("--ledger", type=Path, default=Path.home() / "strata-ledger" / "ablations.jsonl")
    ap.add_argument("--load-timeout", type=float, default=1200)
    ap.add_argument("--thr-decode", type=float, default=0.06)
    ap.add_argument("--thr-prefill", type=float, default=0.03)
    ap.add_argument("--only", default="", help="comma list of knobs to test (pcie-frac,prefill,kv-resident,"
                    "max-context,spec-min-p,pf-fused); default all")
    ap.add_argument("--stop-cmd", help="override how production is stopped (shell)")
    ap.add_argument("--start-cmd", help="override how production is started (shell)")
    ap.add_argument("--dry-run", action="store_true", help="preflight and build prompts; stop nothing")
    ap.add_argument("--models", default="", help="instead of knob cells: one cell per model on --engine-root "
                    "(e.g. iq3_s,ud-q4_k_xl,ud-q5_k_xl; name@bN = with --resident-budget-gib N), "
                    "production's settings otherwise")
    ap.add_argument("--engine-root", type=Path, default=Path.home() / "src" / "Strata-0.1.40.3",
                    help="the Strata build the --models cells run on (its build/strata, serve/, packs/, .venv)")
    ap.add_argument("--unsloth-dir", type=Path, default=Path.home() / "models" / "unsloth" / "Qwen3.8-Flash-Next-GGUF")
    ap.add_argument("--ista-dir", type=Path, default=Path.home() / "models" / "strata" / "models" / "IQ3_S")
    ap.add_argument("--budget-gib", type=int, default=56, help="--resident-budget-gib for the Unsloth models")
    ap.add_argument("--mmproj", type=Path,
                    default=Path.home() / "models" / "strata" / "mmproj" / "mmproj-Qwen3.8-Flash-Next-BF16.gguf",
                    help="image encoder weights for model cells ending in +v")
    ap.add_argument("--engine-kind", default="strata", help=argparse.SUPPRESS)  # 'mock' for self-test
    a = ap.parse_args()
    signal.signal(signal.SIGHUP, signal.SIG_IGN)  # an SSH disconnect must not abort the run before restore

    prod = find_production(a.port)
    server_py = Path(next(x for x in prod["argv"] if x.endswith("server.py")))
    root = (server_py if server_py.is_absolute() else Path(prod["cwd"]) / server_py).resolve().parent.parent
    # Use the interpreter production runs under (its venv has the tokenizer's deps). argv[0], not /proc/PID/exe:
    # the latter resolves a venv's python symlink to the system binary and loses the venv's site-packages.
    exe = prod["argv"][0]
    if "/" not in exe:
        exe = shutil.which(exe, path=prod["env"].get("PATH")) or exe
    elif not os.path.isabs(exe):
        exe = os.path.join(prod["cwd"], exe)
    if os.environ.get("STRATA_ABLATE_REEXEC") != "1" and os.path.isfile(exe) and \
            os.path.abspath(exe) != os.path.abspath(sys.executable):
        os.environ["STRATA_ABLATE_REEXEC"] = "1"
        os.execv(exe, [exe, os.path.abspath(__file__)] + sys.argv[1:])

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = a.out / run_id
    out.mkdir(parents=True, exist_ok=True)
    latest = a.out / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(out.name)
    LOG = open(out / "progress.log", "a")

    cfg_arg = argval(prod["argv"], "--config")
    if not cfg_arg:
        sys.exit("production server was started without --config; cannot derive cells")
    cfg_path = Path(cfg_arg) if Path(cfg_arg).is_absolute() else Path(prod["cwd"]) / cfg_arg
    base_cfg = json.loads(cfg_path.read_text())
    unit, user_unit = systemd_unit(prod["cgroup"])
    if a.stop_cmd and a.start_cmd:
        stop_cmd, start_cmd, how = a.stop_cmd, a.start_cmd, "custom commands"
    elif unit:
        sc = ["systemctl", "--user"] if user_unit else ["sudo", "-n", "systemctl"]
        stop_cmd, start_cmd = " ".join(sc + ["stop", unit]), " ".join(sc + ["start", unit])
        how = f"systemd {'user' if user_unit else 'system'} unit {unit}"
        if not user_unit and subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode != 0:
            sys.exit(f"production runs as system unit {unit}, and passwordless sudo is not available to stop and "
                     "restart it unattended. Either allow `sudo -n systemctl stop/start " + unit +
                     "` for this user, or pass --stop-cmd and --start-cmd.")
    else:
        stop_cmd, start_cmd, how = None, None, "plain process (will be relaunched with its exact command line and environment)"
    sysctl = None
    if unit and not (a.stop_cmd and a.start_cmd):
        sysctl = ["systemctl", "--user"] if user_unit else ["sudo", "-n", "systemctl"]

    def active_services():
        if not sysctl:
            return set()
        try:
            out = subprocess.check_output(sysctl + ["list-units", "--type=service", "--state=active", "--no-legend",
                                                    "--plain"], text=True, timeout=30)
            return {ln.split()[0] for ln in out.splitlines() if ln.strip()}
        except (OSError, subprocess.SubprocessError):
            return set()
    try:
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.SubprocessError):
        commit = "?"
    targets = [int(x) for x in a.targets.split(",")]
    mc = int(get_arg(base_cfg["args"], "--max-context") or 262144)
    if max(targets) + 300 > mc:
        sys.exit(f"largest target {max(targets)} does not fit production --max-context {mc}")
    if not port_free(a.bench_port):
        sys.exit(f"--bench-port {a.bench_port} is in use")

    ctx = {"out": out, "run_id": run_id, "root": root, "python": sys.executable, "prod_env": prod["env"],
           "bench_port": a.bench_port, "load_timeout": a.load_timeout, "engine_kind": a.engine_kind,
           "targets": targets, "runs": a.runs, "commit": commit, "prod_config": str(cfg_path), "ledger": a.ledger,
           "models_mode": bool(a.models), "thr_dec": a.thr_decode, "thr_pre": a.thr_prefill, "seq": 0, "cell_proc": None}
    log(f"production: pid {prod['pid']} port {prod['port']} config {cfg_path} via {how}")
    log(f"strata root {root} commit {commit}; python {sys.executable}; run dir {out}")
    (out / "production.json").write_text(json.dumps({"argv": prod["argv"], "cwd": prod["cwd"], "config": str(cfg_path),
                                                      "config_json": base_cfg, "how": how, "stop_cmd": stop_cmd,
                                                      "start_cmd": start_cmd}, indent=1) + "\n")
    hw = []
    for cmd in (["nvidia-smi", "--query-gpu=name,driver_version,power.limit,pcie.link.gen.max,pcie.link.width.max",
                 "--format=csv"], ["lscpu"], ["free", "-g"], ["uname", "-a"]):
        try:
            hw.append("$ " + " ".join(cmd) + "\n" + subprocess.check_output(cmd, text=True, timeout=20))
        except (OSError, subprocess.SubprocessError):
            pass
    (out / "hw.txt").write_text("\n".join(hw))

    tok_dir = Path(base_cfg.get("tokenizer") or "")
    if not tok_dir.is_absolute():
        tok_dir = Path(base_cfg.get("cwd") or root) / tok_dir
    log("building prompts with Strata's tokenizer (production still running)")
    ctx["prompts"] = build_prompts(root, tok_dir, targets, a.runs)

    knobs = [k for k in a.only.split(",") if k]
    cells = model_cells(base_cfg, a) if a.models else \
        [c for c in knob_cells(base_cfg) if not knobs or c["knob"] in knobs]
    plan = ["baseline"] + [c["name"] for c in cells] + ([] if a.models else ["combo (if 2+ winners)"]) + ["baseline-end"]
    for c in cells if a.models else []:
        log(f"  {c['name']}: {' '.join(c['cfg']['args'])}")
    log("plan: " + " | ".join(plan))
    if a.dry_run:
        log(f"dry run: {len(plan)} cells planned, roughly 5-7 min each. Nothing was stopped. Run dir: {out}")
        return

    t_begin = time.time()
    budget_s = a.budget_min * 60
    restore_reserve = 15 * 60
    recs, base, base_end, combo_note = [], None, None, "Not run."
    ctx["idle_mib"] = 0

    def restore():
        if ctx.get("restored"):
            return
        ctx["restored"] = True
        stop_proc(ctx.get("cell_proc"))
        log("restoring production")
        if start_cmd:
            rc = subprocess.run(start_cmd, shell=True).returncode
            log(f"start command exit {rc}")
            # units that went down with Strata (Requires=/BindsTo=) come back too
            for u in sorted(ctx.get("active_before", set()) - active_services()):
                rc = subprocess.run(sysctl + ["start", u]).returncode
                log(f"restarted dependent unit {u}: exit {rc}")
        else:
            # back into tmux session 'strata' (log ~/strata-prod.log) when tmux is here, so it can be found and
            # stopped by hand later; its folder, STRATA_* environment and command line as they were
            envs = " ".join(shlex.quote(f"{k}={v}") for k, v in sorted(prod["env"].items()) if k.startswith("STRATA_"))
            script = Path.home() / "strata-prod-restart.sh"
            script.write_text(f"#!/bin/sh\ncd {shlex.quote(prod['cwd'])} && exec env {envs} {shlex.join(prod['argv'])}\n")
            script.chmod(0o755)
            if shutil.which("tmux"):
                subprocess.run(["tmux", "kill-session", "-t", "strata"], capture_output=True)
                subprocess.run(["tmux", "new-session", "-d", "-s", "strata",
                                f"sh {shlex.quote(str(script))} 2>&1 | tee -a {shlex.quote(str(Path.home() / 'strata-prod.log'))}"])
                log("production relaunched in tmux session 'strata' (log ~/strata-prod.log)")
            else:
                with open(out / "production-restart.out", "w") as fh:
                    subprocess.Popen(prod["argv"], cwd=prod["cwd"], env=prod["env"], stdout=fh,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        url = f"http://127.0.0.1:{prod['port']}"
        t = time.time() + 1200
        while time.time() < t:
            try:
                if http_json(url + "/v1/status", timeout=5).get("loaded", True):
                    ctx["restore_status"] = f"answering on :{prod['port']} after {time.time() - t_begin:.0f}s total"
                    log("production is back: " + ctx["restore_status"])
                    return
            except (urllib.error.URLError, OSError, ValueError):
                pass
            time.sleep(5)
        ctx["restore_status"] = "NOT answering after 20 min - check it by hand"
        log("WARNING: " + ctx["restore_status"])

    def on_signal(signum, _frame):
        raise SystemExit(f"signal {signum}")

    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, on_signal)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)  # an SSH disconnect must not abort the run before restore
    try:
        ctx["active_before"] = active_services()
        log(f"stopping production: {stop_cmd or 'SIGTERM pid %d' % prod['pid']}")
        ctx["stopped"] = True
        if stop_cmd:
            subprocess.run(stop_cmd, shell=True, check=True)
        else:
            os.kill(prod["pid"], signal.SIGTERM)
        t = time.time() + 120
        while time.time() < t and not port_free(prod["port"]):
            time.sleep(2)
        if not port_free(prod["port"]):
            raise RuntimeError(f"production did not release :{prod['port']} within 2 min of stopping")
        log("waiting 30 s to make sure nothing respawns Strata on its own")
        time.sleep(30)
        respawn = []
        try:
            again = find_production(prod["port"])
            respawn = [again["pid"]]
        except SystemExit:
            pass
        if respawn or not port_free(prod["port"]):
            ctx["restored"] = True  # whatever respawned it is production now; don't start a second copy
            ctx["restore_status"] = f"something restarted Strata by itself (pid {respawn}); run aborted, nothing measured"
            raise RuntimeError("Strata was respawned by a supervisor after stopping it. Stop that supervisor for the "
                               "window (or give --stop-cmd/--start-cmd that control it) and rerun.")
        ctx["idle_mib"] = gpu_mem_used()
        log(f"GPU after stop: {ctx['idle_mib']:.0f} MiB used; compute apps: {gpu_apps() or 'none'}")

        base = run_cell({"name": "baseline", "knob": None, "value": None, "cfg": base_cfg}, ctx)
        recs.append(base)
        ledger_append(ctx, base, None)
        if base["status"] != "ok":
            raise RuntimeError("baseline failed; nothing else is comparable")
        cell_s = base["cell_s"]
        for c in cells:
            if time.time() - t_begin + 2 * cell_s + restore_reserve > budget_s:
                log(f"budget: skipping {c['name']} and later knob cells")
                break
            r = run_cell(c, ctx)
            r["verdict"], r["score"] = verdict(r, base, a.thr_decode, a.thr_prefill)
            cell_s = max(cell_s, r["cell_s"])
            recs.append(r)
            ledger_append(ctx, r, base)
            log(f"verdict {r['cell']}: {r['verdict']}")
            write_report(ctx, recs, base, None, combo_note)
        best = {}
        for r in recs:
            if r.get("verdict", "").startswith("WIN") and r["knob"] != "model" and (r["knob"] not in best or r["score"] > best[r["knob"]]["score"]):
                best[r["knob"]] = r
        if len(best) >= 2 and time.time() - t_begin + 2 * cell_s + restore_reserve <= budget_s:
            cfg = json.loads(json.dumps(base_cfg))
            for r in best.values():
                if r["knob"] == "pf-fused":
                    cfg["env"] = r["env"]
                else:
                    cfg["args"] = set_arg(cfg["args"], "--" + r["knob"], r["value"])
            r = run_cell({"name": "combo:" + "+".join(f"{k}={v['value']}" for k, v in best.items()), "knob": "combo",
                          "value": None, "cfg": cfg}, ctx)
            r["verdict"], r["score"] = verdict(r, base, a.thr_decode, a.thr_prefill)
            recs.append(r)
            ledger_append(ctx, r, base)
            combo_note = f"`{r['cell']}`: {r['verdict']} (see table)."
        elif len(best) >= 2:
            combo_note = "Skipped for time."
        else:
            combo_note = f"Not needed: {len(best)} winning knob(s)."
        if time.time() - t_begin + cell_s + restore_reserve <= budget_s + 5 * 60:
            base_end = run_cell({"name": "baseline-end", "knob": None, "value": None, "cfg": base_cfg}, ctx)
            base_end["verdict"], base_end["score"] = verdict(base_end, base, a.thr_decode, a.thr_prefill)
            recs.append(base_end)
            ledger_append(ctx, base_end, base)
        else:
            log("budget: skipping baseline-end")
    except BaseException as ex:  # noqa: BLE001 - always fall through to restore
        log(f"stopping early: {ex!r}")
    finally:
        if ctx.get("stopped"):
            restore()
        write_report(ctx, recs, base, base_end, combo_note)
        log(f"done in {(time.time() - t_begin) / 60:.1f} min. Report: {out / 'REPORT.md'}  Ledger: {a.ledger}")


if __name__ == "__main__":
    main()
