#!/usr/bin/env python3
"""strata_quality.py - how far each quant's next-token distribution is from Q8_0's, on the same engine.

Method (Strata's own, docs/UNSLOTH_Q4.md and bench/results/2026-10-03-v100-prompt-attn): a serve engine with
STRATA_LOGPOS=<file> STRATA_LOGPOS_TOPK=256 writes, for every prompt token it reads through the verify windows, the
log-probability of the true next token and the 256 most likely tokens. Every quant ("arm") reads the same requests;
each position is then compared with the Q8_0 arm:
  KL(Q8_0 || arm) over Q8_0's top 256 plus one bucket for the rest, top-1 agreement (all positions, and where Q8_0's
  top-2 gap is >= 0.5 nats), top-10 overlap, and each arm's perplexity of the true text.

Requests (public text, built once by `prepare`, identical for every arm):
  prose  wikitext-2-raw test (llama.cpp's perplexity file, huggingface.co/datasets/ggml-org/ci)
  code   llama.cpp source at Strata's pinned commit (third_party/llama.cpp of the test checkout)
  agent  NousResearch/hermes-function-calling-v1 func-calling.json (system prompt, tools, calls, tool results)
Each request scores ~560 tokens. "short" requests are those tokens alone (one user message); "8192"/"32768"
requests put that many tokens of the same stream first (a user message read by the batched prompt path, then a
short assistant reply), so the scored tokens are read with a long context behind them.

    python3 strata_quality.py prepare                # download + build requests; touches nothing else
    python3 strata_quality.py run --dry-run          # check files, show configs and plan; stops nothing
    nohup python3 strata_quality.py run > ~/strata-quality.out 2>&1 &     # stops production, runs, restores it
    python3 strata_quality.py report                 # tables from whatever arms are done

`run` needs the GPU: it stops the production server on --port (SIGINT, a clean stop), runs each arm on a private
server from the test build, and ALWAYS restarts production afterwards (tmux session `strata`, log ~/strata-prod.log)
- on success, failure, Ctrl-C or kill. Arms already finished are skipped, so a stopped run resumes.
Re-executes itself under the test checkout's .venv python (tokenizer, chat template, numpy).
"""
import argparse
import hashlib
import io
import json
import os
import random
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

HOME = Path.home()
WIKI_URL = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
FC_URL = "https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1/resolve/main/func-calling.json"
SCORED = 560            # tokens scored per request (+ ~15 template tokens: stays under --short-read)
SHORT_READ = 1024       # the engine reads a fresh prompt part this short through the verify windows (logged)
REF = "q8_0"
ARMS = {  # name -> (shard 1 relative to its model folder, family)
    "q8_0": ("unsloth", "Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf"),
    "iq3_s": ("ista", "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"),
    "ud-q4_k_xl": ("unsloth", "UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"),
    "ud-q5_k_xl": ("unsloth", "UD-Q5_K_XL/Qwen3.8-Flash-Next-UD-Q5_K_XL-00001-of-00006.gguf"),
}
LOG = None


def log(msg):
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    if LOG:
        LOG.write(line + "\n")
        LOG.flush()


def argval(argv, flag, default=None):
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
    return default


def set_arg(args, flag, value=None, bare=False):
    """args without `flag` (and its value); then `flag value` (or the bare flag) appended unless value is None."""
    out, i = [], 0
    while i < len(args):
        if args[i] == flag:
            i += 2 if i + 1 < len(args) and not str(args[i + 1]).startswith("--") else 1
            continue
        out.append(args[i])
        i += 1
    if bare:
        out.append(flag)
    elif value is not None:
        out += [flag, str(value)]
    return out


# --------------------------------------------------------------------------- data

def fetch(url, dest):
    if not dest.exists():
        log(f"downloading {url}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "strata_quality/1"}),
                                    timeout=300) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.replace(dest)
    return hashlib.sha256(dest.read_bytes()).hexdigest()


def load_tokenizer(root, tok_dir):
    sys.path[:0] = [str(root), str(root / "tools")]
    from strata_tokenizer import Tokenizer
    vocab = json.loads((tok_dir / "vocab.json").read_text(encoding="utf-8"))
    toks = [None] * len(vocab)
    for t, n in vocab.items():
        toks[n] = t
    return Tokenizer(toks, (tok_dir / "merges.txt").read_text(encoding="utf-8").splitlines(),
                     json.loads((tok_dir / "token_type.json").read_text(encoding="utf-8")))


def source_texts(data, root):
    """{source: (text, provenance)} - each a stream far longer than the plan needs."""
    out = {}
    wz = data / "wikitext-2-raw-v1.zip"
    wsha = fetch(WIKI_URL, wz)
    with zipfile.ZipFile(wz) as z:
        # the test split first (the first requests are the same text as before), then train: 64K-token contexts need
        # more prose than the test split's ~300K tokens
        test = next(n for n in z.namelist() if n.endswith("wiki.test.raw"))
        train = next(n for n in z.namelist() if n.endswith("wiki.train.raw"))
        out["prose"] = (z.read(test).decode("utf-8") + "\n" + z.read(train).decode("utf-8"),
                        f"{WIKI_URL} ({test}, then {train}; sha256 {wsha})")
    fc = data / "func-calling.json"
    fsha = fetch(FC_URL, fc)
    parts, role = [], {"system": "SYSTEM", "human": "USER", "gpt": "ASSISTANT", "tool": "TOOL"}
    for conv in json.loads(fc.read_text(encoding="utf-8")):
        turns = [f"### {role.get(t.get('from'), str(t.get('from')).upper())}\n{t.get('value', '').strip()}"
                 for t in conv.get("conversations", [])]
        parts.append("\n\n".join(turns))
        if sum(map(len, parts)) > 4_000_000:
            break
    out["agent"] = ("\n\n---\n\n".join(parts), f"{FC_URL} (sha256 {fsha}), conversations rendered as '### ROLE' blocks")
    llama = root / "third_party" / "llama.cpp"
    commit = (llama / ".strata-commit").read_text().strip() if (llama / ".strata-commit").exists() else "?"
    files = sorted(p for d in ("src", "ggml/src", "common", "gguf-py/gguf") for p in (llama / d).rglob("*")
                   if p.is_file() and p.suffix in (".c", ".cpp", ".h", ".py") and "vendor" not in p.parts)
    random.Random(1234).shuffle(files)          # mixed languages, fixed order
    parts = []
    for p in files:
        try:
            parts.append(f"// ===== {p.relative_to(llama)} =====\n" + p.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            continue
        if sum(map(len, parts)) > 4_000_000:
            break
    out["code"] = ("\n\n".join(parts), f"llama.cpp {commit} (src, ggml/src, common, gguf-py; shuffled with seed 1234)")
    return out


def parse_plan(s):
    plan = []
    for item in s.split(","):
        ctx, n = item.split(":")
        plan.append((0 if ctx == "short" else int(ctx), int(n)))
    return plan


def prepare(a, root):
    data = a.out / "data"
    data.mkdir(parents=True, exist_ok=True)
    req_path = data / "requests.json"
    plan = parse_plan(a.plan)
    if req_path.exists():
        old = json.loads(req_path.read_text())
        if old.get("plan") == a.plan:
            log(f"requests: {req_path} ({len(old['requests'])} requests, built earlier)")
            return old
        sys.exit(f"{req_path} was built with --plan {old.get('plan')}; delete {a.out} (or pass --out) to change it")
    tok = load_tokenizer(root, root / "packs" / "iq3_s" / "tokenizer")
    need = sum((ctx + SCORED) * n for ctx, n in plan)
    reqs, prov = [], {}
    for src, (text, where) in source_texts(data, root).items():
        prov[src] = where
        chars = min(len(text), int(need * 6) + 100_000)
        t0 = time.time()
        ids = tok.encode(text[:chars])
        log(f"{src}: {len(ids):,} tokens from {chars:,} characters ({time.time() - t0:.0f}s)")
        if len(ids) < need + 100:
            sys.exit(f"{src}: only {len(ids)} tokens, the plan needs {need}")
        p = 0
        for ctx, n in plan:
            for k in range(n):
                msgs = []
                if ctx:
                    msgs += [{"role": "user", "content": tok.decode(ids[p:p + ctx], errors="ignore")},
                             {"role": "assistant", "content": "Noted."}]
                msgs.append({"role": "user", "content": tok.decode(ids[p + ctx:p + ctx + SCORED], errors="ignore")})
                reqs.append({"id": f"{src}-{'short' if not ctx else ctx}-{k}", "source": src,
                             "context": "short" if not ctx else str(ctx), "messages": msgs})
                p += ctx + SCORED
    doc = {"plan": a.plan, "scored_tokens": SCORED, "provenance": prov, "requests": reqs}
    req_path.write_text(json.dumps(doc, ensure_ascii=False) + "\n")
    log(f"requests: {len(reqs)} written to {req_path}")
    return doc


# --------------------------------------------------------------------------- production

def find_on_port(port):
    try:
        out = subprocess.check_output(["ss", "-ltnpH", f"sport = :{port}"], text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    for tok in out.replace(",", " ").split():
        if tok.startswith("pid="):
            pid = int(tok[4:])
            p = Path(f"/proc/{pid}")
            try:
                argv = [x.decode(errors="replace") for x in (p / "cmdline").read_bytes().split(b"\0") if x]
                env = dict(kv.split("=", 1) for kv in (p / "environ").read_bytes().decode(errors="replace").split("\0")
                           if "=" in kv)
                return {"pid": pid, "argv": argv, "cwd": os.readlink(p / "cwd"), "env": env, "port": port}
            except OSError as e:
                sys.exit(f"pid {pid} listens on :{port} but its /proc entries are unreadable ({e}): run as its user")
    return None


def port_free(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def http_json(url, body=None, timeout=30):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def gpu_free_mib():
    try:
        free, total = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free,memory.total",
                                               "--format=csv,noheader,nounits", "-i", "0"], text=True,
                                              timeout=10).strip().split(",")
        return float(free), float(total)
    except (OSError, subprocess.SubprocessError, ValueError):
        return -1.0, -1.0


def wait_gpu_free(timeout=240):
    t = time.time() + timeout
    while time.time() < t:
        free, total = gpu_free_mib()
        if free >= total - 2048:
            return True
        time.sleep(3)
    return False


def restart_script(prod):
    """~/strata-prod-restart.sh: production's own folder, STRATA_* environment and command line."""
    env = " ".join(shlex.quote(f"{k}={v}") for k, v in sorted(prod["env"].items()) if k.startswith("STRATA_"))
    cmd = f"cd {shlex.quote(prod['cwd'])} && exec env {env} {shlex.join(prod['argv'])}\n"
    path = HOME / "strata-prod-restart.sh"
    path.write_text("#!/bin/sh\n" + cmd)
    path.chmod(0o755)
    return path


def start_production(script, port):
    log_path = HOME / "strata-prod.log"
    if shutil.which("tmux"):
        subprocess.run(["tmux", "kill-session", "-t", "strata"], capture_output=True)
        subprocess.run(["tmux", "new-session", "-d", "-s", "strata",
                        f"sh {shlex.quote(str(script))} 2>&1 | tee -a {shlex.quote(str(log_path))}"], check=True)
        how = "tmux session 'strata'"
    else:
        with open(log_path, "a") as fh:
            subprocess.Popen(["sh", str(script)], stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        how = "a background process"
    t = time.time() + 1200
    while time.time() < t:
        try:
            if http_json(f"http://127.0.0.1:{port}/v1/status", timeout=5).get("loaded"):
                return f"answering on :{port} ({how}, log {log_path})"
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(5)
    return f"NOT answering on :{port} after 20 min ({how}): check {log_path}"


# --------------------------------------------------------------------------- arms

def arm_config(name, a, root, prod_cfg, mtp):
    fam, rel = ARMS[name.split("@")[0]]
    shard1 = (a.ista_dir if fam == "ista" else a.unsloth_dir) / rel
    padded = shard1.parent.with_name(shard1.parent.name + "-padded") / shard1.name
    if fam == "unsloth" and padded.exists() and not a.no_padded:   # UD-Q5_K_XL: shard 1 padded by 6 bytes (identical output, tested)
        shard1 = padded
    pack = root / "packs" / name.split("@")[0]
    p = prod_cfg.get("args", [])
    args = ["--pack", str(pack), "--native", str(shard1)]
    if fam == "ista":          # 2 shards: the n-gram table is shard 2 (setup passes it); 3+: the engine finds it
        args += ["--ple-gguf", str(shard1).replace("-00001-of-", "-00002-of-")]
    args += ["--expert-profile", str(root / "data" / "expert-profile.bin"),
             "--expert-cache", (name.split("@ec")[1] if "@ec" in name else argval(p, "--expert-cache", "auto")),
             "--prefill", argval(p, "--prefill", "auto"),
             "--spec", argval(p, "--spec", "4"), "--spec-min-p", argval(p, "--spec-min-p", "0.5"),
             "--max-context", str(a.max_context), "--kv", argval(p, "--kv", "int8"),
             "--short-read", str(SHORT_READ), "--adapt-every", "100000"]   # the expert cache stays put during a run
    if mtp:
        args += ["--mtp", mtp]
    if fam == "unsloth":
        args += ["--resident-budget-gib", str(a.budget_gib)]
    cfg = {"exe": str(a.exe or root / "build" / "strata"), "args": args, "cwd": str(root),
           "tokenizer": str(pack / "tokenizer"), "model_name": f"quality-{name}", "lib_dirs": prod_cfg.get("lib_dirs", [])}
    missing = [x for x in [cfg["exe"], str(shard1), str(pack / "native_experts.txt")] + args[args.index("--expert-profile") + 1:args.index("--expert-profile") + 2]
               if not Path(x).exists()]
    return cfg, missing


def run_arm(name, cfg, a, root, reqs, env_base):
    adir = a.out / "arms" / name
    if (adir / "done.json").exists():
        log(f"arm {name}: done earlier, skipped")
        return "skipped"
    if adir.exists():
        shutil.rmtree(adir)
    adir.mkdir(parents=True)
    logpos = adir / "logpos.tsv"
    cfg = dict(cfg, port=a.bench_port, log=str(adir / "engine.log"))
    (adir / "config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    env = dict(env_base, STRATA_LOGPOS=str(logpos), STRATA_LOGPOS_TOPK="256")
    cmd = [str(root / ".venv" / "bin" / "python"), str(root / "serve" / "server.py"), "--engine", "strata",
           "--config", str(adir / "config.json"), "--port", str(a.bench_port), "--host", "127.0.0.1"]
    url = f"http://127.0.0.1:{a.bench_port}"
    if not wait_gpu_free():
        raise RuntimeError(f"the GPU is not free before arm {name} (another program holds VRAM)")
    log(f"arm {name}: starting ({' '.join(cfg['args'][:4])} ...)")
    t_start = time.time()
    with open(adir / "server.out", "w") as out:
        proc = subprocess.Popen(cmd, cwd=str(root), env=env, stdout=out, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            deadline = time.time() + a.load_timeout
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"server exited with code {proc.returncode} while loading "
                                       f"(see {adir / 'server.out'})")
                try:
                    if http_json(url + "/v1/status", timeout=5).get("loaded"):
                        break
                except (urllib.error.URLError, OSError, ValueError):
                    pass
                if time.time() > deadline:
                    raise RuntimeError("server not loaded before --load-timeout")
                time.sleep(3)
            load_s = time.time() - t_start
            engine = (http_json(url + "/metrics").get("engine") or {})
            log(f"arm {name}: loaded in {load_s:.0f}s, engine {engine.get('version')}, "
                f"expert slots {engine.get('expert_slots')}")
            body = {"model": cfg["model_name"], "temperature": 0, "reasoning_effort": "none", "max_tokens": 1}
            http_json(url + "/v1/chat/completions", dict(body, messages=[{"role": "user", "content": "Say OK."}]),
                      timeout=a.request_timeout)                     # warm-up; its rows are not used
            index = []
            for i, r in enumerate(reqs):
                if proc.poll() is not None:
                    raise RuntimeError(f"server died at request {r['id']} (code {proc.returncode})")
                off0 = logpos.stat().st_size if logpos.exists() else 0
                t0 = time.time()
                http_json(url + "/v1/chat/completions", dict(body, messages=r["messages"]), timeout=a.request_timeout)
                dt = time.time() - t0
                time.sleep(0.2)
                off1 = logpos.stat().st_size if logpos.exists() else 0
                rows = 0
                if off1 > off0:
                    with open(logpos, "rb") as f:
                        f.seek(off0)
                        rows = f.read(off1 - off0).count(b"\n")
                index.append({"id": r["id"], "off0": off0, "off1": off1, "rows": rows, "s": round(dt, 1)})
                note = "" if rows >= SCORED * 0.9 else "  <-- FEW ROWS: the scored part was not read through the windows"
                log(f"arm {name}: {i + 1}/{len(reqs)} {r['id']}: {rows} positions in {dt:.0f}s{note}")
            (adir / "index.json").write_text(json.dumps(index, indent=1) + "\n")
            (adir / "done.json").write_text(json.dumps({"engine": engine, "load_s": round(load_s, 1),
                                                        "total_s": round(time.time() - t_start, 1)}, indent=1) + "\n")
            return "ok"
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGINT)
                try:
                    proc.wait(120)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(30)


def cmd_run(a, root):
    global LOG
    a.out.mkdir(parents=True, exist_ok=True)
    LOG = open(a.out / "progress.log", "a")
    doc = prepare(a, root)
    reqs = doc["requests"]
    prod = find_on_port(a.port)
    prod_cfg_path = Path(argval(prod["argv"], "--config")) if prod else a.prod_config
    if prod and not prod_cfg_path.is_absolute():
        prod_cfg_path = Path(prod["cwd"]) / prod_cfg_path
    prod_cfg = json.loads(prod_cfg_path.read_text())
    mtp = None if a.no_mtp else argval(prod_cfg.get("args", []), "--mtp")
    if mtp and not Path(mtp).is_absolute():
        mtp = str(Path(prod_cfg.get("cwd") or prod_cfg_path.parent) / mtp)
    arms = [x for x in a.arms.split(",") if x]
    if arms[0].split("@")[0] != REF:
        log(f"note: the reference {REF} is not first; arms run in the order given")
    cfgs, problems = {}, []
    for name in arms:
        if name.split("@")[0] not in ARMS:
            sys.exit(f"unknown arm {name}; known: {', '.join(ARMS)} (name@2 = a repeat, name@ecN = --expert-cache N)")
        cfgs[name], missing = arm_config(name, a, root, prod_cfg, mtp)
        problems += [f"{name}: missing {m}" for m in missing]
    log(f"test build {root}; production {'pid %d on :%d' % (prod['pid'], a.port) if prod else 'not running on :%d' % a.port}"
        f"; production config {prod_cfg_path}")
    log(f"{len(reqs)} requests x {len(arms)} arms ({', '.join(arms)}); ~{len(reqs) * SCORED:,} scored positions per arm")
    for name in arms:
        log(f"  {name}: {' '.join(cfgs[name]['args'])}")
    if problems:
        sys.exit("STOP:\n  " + "\n  ".join(problems))
    if a.dry_run:
        log("dry run: nothing stopped. Without --dry-run, production is stopped for the whole run (hours for "
            "q8_0 and ud-q5_k_xl, whose experts are partly read from the SSD) and restarted at the end.")
        return
    env_base = dict(os.environ)
    if prod:
        env_base.update({k: v for k, v in prod["env"].items() if k.startswith("STRATA_")})   # e.g. STRATA_PF_FUSED
    for k in ("STRATA_LOGPOS", "STRATA_LOGPOS_TOPK", "STRATA_LOGPOS_EXTRA"):
        env_base.pop(k, None)
    script = restart_script(prod) if prod else None
    if script:
        log(f"production restart command saved: {script}")

    def on_signal(signum, _frame):
        raise SystemExit(f"signal {signum}")
    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, on_signal)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    results = {}
    try:
        if prod:
            log(f"stopping production (pid {prod['pid']}, SIGINT)")
            os.kill(prod["pid"], signal.SIGINT)
            t = time.time() + 180
            while time.time() < t and not port_free(a.port):
                time.sleep(2)
            if not port_free(a.port):
                os.kill(prod["pid"], signal.SIGTERM)
                time.sleep(20)
            if not port_free(a.port):
                raise RuntimeError(f"production did not release :{a.port}")
        for name in arms:
            try:
                results[name] = run_arm(name, cfgs[name], a, root, reqs, env_base)
            except Exception as ex:   # noqa: BLE001 - one failed arm is a finding; the others still run
                results[name] = f"failed: {ex}"
                log(f"arm {name} FAILED: {ex}")
                if mtp and name == arms[0] and "loading" in str(ex) and not a.no_mtp:
                    log("the first arm did not load: retrying every arm without --mtp (an MTP layer from another "
                        "engine version is the likely cause; see its server.out)")
                    mtp = None
                    for n in arms:
                        cfgs[n] = arm_config(n, a, root, prod_cfg, None)[0]
                    try:
                        results[name] = run_arm(name, cfgs[name], a, root, reqs, env_base)
                    except Exception as ex2:   # noqa: BLE001
                        results[name] = f"failed: {ex2}"
                        log(f"arm {name} FAILED again: {ex2}")
    finally:
        status = start_production(script, a.port) if script else "production was not running before; not started"
        log(f"production: {status}")
    log("arms: " + ", ".join(f"{k} {v}" for k, v in results.items()))
    cmd_report(a)


# --------------------------------------------------------------------------- report

def load_arm(adir, reqs_by_id):
    import numpy as np
    idx = json.loads((adir / "index.json").read_text())
    raw = (adir / "logpos.tsv").read_bytes()
    out = {}
    for e in idx:
        for line in raw[e["off0"]:e["off1"]].decode().splitlines():
            f = line.split("\t")
            if len(f) < 9:
                continue
            top = [x.split(":") for x in f[8:]]
            out[(e["id"], int(f[0]))] = (int(f[1]), float(f[2]), np.array([int(t[0]) for t in top], dtype=np.int64),
                                         np.array([float(t[1]) for t in top], dtype=np.float64))
    return out


def compare(ref, arm):
    """Per shared position: (kl, top1_same, gap, top10_overlap)."""
    import numpy as np
    rows = {}
    for key, (tgt, _lp, rid, rlp) in ref.items():
        if key not in arm or arm[key][0] != tgt:
            continue
        aid, alp = arm[key][2], arm[key][3]
        order = np.argsort(aid)
        pos = np.searchsorted(aid[order], rid)
        pos = np.clip(pos, 0, len(aid) - 1)
        found = aid[order][pos] == rid
        # A ref token outside the arm's top 256 has a probability somewhere in the arm's own leftover mass
        # (1 - its top-256 sum) and below its 256th. Each missing token gets min(256th, leftover / (missing + 1)),
        # so the missing tokens never take more than the arm actually has left and the rest bucket stays > 0.
        # (The first version gave each the 256th alone; at flat positions their sum could exceed the leftover,
        # empty the rest bucket to 1e-10, and add a spurious ~1-3 nats.)
        miss = ~found
        a_left = max(1.0 - float(np.exp(alp).sum()), 1e-12)
        fill = min(float(np.exp(alp.min())), a_left / (int(miss.sum()) + 1))
        q = np.where(found, np.exp(alp[order][pos]), fill)
        qlp = np.log(q)
        p = np.exp(rlp)
        p_rest, q_rest = max(1.0 - p.sum(), 0.0), max(1.0 - q.sum(), 1e-12)
        kl = float((p * (rlp - qlp)).sum() + (p_rest * np.log(p_rest / q_rest) if p_rest > 1e-12 else 0.0))
        rows[key] = (kl, rid[0] == aid[0], rlp[0] - rlp[1], len(set(rid[:10].tolist()) & set(aid[:10].tolist())) / 10)
    return rows


def cmd_report(a):
    import numpy as np
    doc = json.loads((a.out / "data" / "requests.json").read_text())
    reqs = {r["id"]: r for r in doc["requests"]}
    done = [d.name for d in sorted((a.out / "arms").glob("*")) if (d / "done.json").exists()] \
        if (a.out / "arms").exists() else []
    if REF not in done:
        log(f"report: the reference arm {REF} is not done yet (done: {', '.join(done) or 'none'})")
        return
    data = {n: load_arm(a.out / "arms" / n, reqs) for n in done}
    ref = data[REF]
    groups = [("all", lambda r: True)] + \
        [(s, lambda r, s=s: r["source"] == s) for s in ("prose", "code", "agent")] + \
        [(f"ctx {c}", lambda r, c=c: r["context"] == c) for c in dict.fromkeys(r["context"] for r in reqs.values())]
    lines = ["# Quant quality against Q8_0 on Strata", "",
             f"Generated {datetime.now():%Y-%m-%d %H:%M} by strata_quality.py. Teacher-forced: every arm read the same "
             f"{len(reqs)} requests (~{doc['scored_tokens']} scored tokens each) through the verify windows; "
             "`STRATA_LOGPOS_TOPK=256`. KL = KL(Q8_0 || arm) over Q8_0's top 256 + one bucket (nats). "
             "Decisive = positions where Q8_0's top-2 gap is >= 0.5 nats.", "",
             "Sources: " + "; ".join(f"**{k}** {v}" for k, v in doc["provenance"].items()), ""]
    summary = {}
    comps = {n: compare(ref, data[n]) for n in done if n != REF}   # per position, against Q8_0
    rng = np.random.default_rng(0)
    for gname, pred in groups:
        lines += [f"## {gname}", "",
                  "| arm | requests | positions | mean KL [95% CI] | median KL | p99 KL | top-1 same | top-1 same (decisive) | "
                  "top-10 overlap | PPL arm / Q8_0 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        ref_lp = {k: v[1] for k, v in ref.items() if pred(reqs[k[0]])}
        for n in done:
            if n == REF:
                continue
            rows = {k: v for k, v in comps[n].items() if pred(reqs[k[0]])}
            if not rows:
                continue
            kl = np.array([v[0] for v in rows.values()])
            same = np.array([v[1] for v in rows.values()])
            gap = np.array([v[2] for v in rows.values()])
            ov = np.array([v[3] for v in rows.values()])
            by_req = {}
            for (rid, _), v in rows.items():
                by_req.setdefault(rid, []).append(v[0])
            sums = np.array([sum(v) for v in by_req.values()])
            cnts = np.array([len(v) for v in by_req.values()])
            boots = []
            for _ in range(2000):   # resample whole requests: positions inside one text are not independent
                i = rng.integers(0, len(sums), len(sums))
                boots.append(sums[i].sum() / cnts[i].sum())
            lo, hi = np.percentile(boots, [2.5, 97.5])
            ci = f"[{lo:.4f}, {hi:.4f}]" if len(sums) >= 5 else "(CI needs 5+ requests)"
            keys = list(rows)
            ppl_a = float(np.exp(-np.mean([data[n][k][1] for k in keys])))
            ppl_r = float(np.exp(-np.mean([ref_lp[k] for k in keys])))
            dec = gap >= 0.5
            lines.append(f"| {n} | {len(sums)} | {len(kl):,} | {kl.mean():.4f} {ci} | {np.median(kl):.4f} | "
                         f"{np.percentile(kl, 99):.3f} | {100 * same.mean():.1f}% | "
                         f"{100 * same[dec].mean():.1f}% ({dec.sum():,}) | {100 * ov.mean():.1f}% | "
                         f"{ppl_a:.3f} / {ppl_r:.3f} |")
            summary.setdefault(gname, {})[n] = {"positions": int(len(kl)), "kl_mean": float(kl.mean()),
                                                "kl_ci95": [float(lo), float(hi)], "kl_median": float(np.median(kl)),
                                                "kl_p99": float(np.percentile(kl, 99)), "top1": float(same.mean()),
                                                "top1_decisive": float(same[dec].mean()) if dec.any() else None,
                                                "top10_overlap": float(ov.mean()), "ppl": ppl_a, "ppl_ref": ppl_r}
        lines.append("")
    # Paired: two arms against Q8_0 on the SAME positions. The per-request mean difference removes what the texts
    # share (hard text is hard for both), so its interval is much tighter than comparing two separate intervals.
    arms = [n for n in done if n != REF and "@2" not in n]
    pairs = [(x, y) for i, x in enumerate(arms) for y in arms[i + 1:]]
    if pairs:
        lines += ["## Paired differences (row arm minus column arm, same positions)", "",
                  "KL diff > 0: the first arm is further from Q8_0. Interval: 95% bootstrap over requests. "
                  "Requests worse: share of requests where the first arm's mean KL is higher. Decisive top-1: "
                  "difference in agreement with Q8_0 at decisive positions, percentage points (> 0: the first arm agrees "
                  "more often).", ""]
        for gname, pred in groups:
            lines += [f"### {gname}", "", "| first - second | requests | mean KL diff [95% CI] | requests worse | "
                      "decisive top-1 diff (pp) |", "|---|---:|---:|---:|---:|"]
            for x, y in pairs:
                keys = [k for k in comps[x] if k in comps[y] and pred(reqs[k[0]])]
                if not keys:
                    continue
                per = {}
                for k in keys:
                    per.setdefault(k[0], []).append((comps[x][k][0] - comps[y][k][0],
                                                     comps[x][k][2] >= 0.5,
                                                     float(comps[x][k][1]) - float(comps[y][k][1])))
                ids = list(per)
                sums = np.array([sum(v[0] for v in per[i]) for i in ids])
                cnts = np.array([len(per[i]) for i in ids])
                boots = []
                for _ in range(4000):
                    j = rng.integers(0, len(ids), len(ids))
                    boots.append(sums[j].sum() / cnts[j].sum())
                lo, hi = np.percentile(boots, [2.5, 97.5])
                ci = f"[{lo:+.4f}, {hi:+.4f}]" if len(ids) >= 5 else "(CI needs 5+ requests)"
                worse = np.mean([s_ / c_ > 0 for s_, c_ in zip(sums, cnts)])
                dec = [v[2] for i in ids for v in per[i] if v[1]]
                lines.append(f"| {x} - {y} | {len(ids)} | {sums.sum() / cnts.sum():+.4f} {ci} | {100 * worse:.0f}% | "
                             f"{100 * np.mean(dec) if dec else float('nan'):+.2f} |")
                summary.setdefault("paired", {}).setdefault(gname, {})[f"{x} - {y}"] = {
                    "requests": len(ids), "kl_diff": float(sums.sum() / cnts.sum()), "ci95": [float(lo), float(hi)],
                    "requests_worse": float(worse), "decisive_top1_diff": float(np.mean(dec)) if dec else None}
            lines.append("")
    lines += ["## Notes", "",
              "- Every arm runs on the same engine build; the Unsloth packs (Q8_0 too) have their Q8_0 hyper-connection "
              "projections rounded to BF16 (`--compat-bf16`), so the reference is Q8_0 as Strata runs it.",
              "- Experts held in VRAM run GPU kernels and the others CPU kernels; the split differs per arm (expert "
              "cache size), which is part of what is measured. `--adapt-every 100000` keeps it fixed within a run.",
              "- A token Q8_0 ranks in its top 256 but the arm does not gets a share of the arm's leftover probability "
              "(capped at its 256th), so KL can only be underestimated where the distributions differ a lot.",
              "- `x@2` is a second run of `x` (same everything): its row is the run-to-run floor, expected 0. "
              "`x@ecN` is `x` with `--expert-cache N`: the same weights with another GPU/CPU split of the experts, "
              "i.e. the floor from engine numerics alone. A quant's KL means something only above that floor."]
    for n in done:
        if "@" in n and n.split("@")[0] in data:
            rows = compare(data[n.split("@")[0]], data[n])
            kl = np.array([v[0] for v in rows.values()])
            lines.append(f"- {n} vs {n.split('@')[0]}: {len(kl):,} positions, mean KL {kl.mean():.6f}, "
                         f"top-1 same {100 * np.mean([v[1] for v in rows.values()]):.2f}%.")
    for n in done:
        d = json.loads((a.out / "arms" / n / "done.json").read_text())
        lines.append(f"- {n}: engine {d['engine'].get('version')}, expert slots {d['engine'].get('expert_slots')}, "
                     f"load {d['load_s']:.0f}s, arm total {d['total_s'] / 60:.0f} min.")
    (a.out / "REPORT.md").write_text("\n".join(lines) + "\n")
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("\n".join(lines))
    log(f"report: {a.out / 'REPORT.md'}")


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["prepare", "run", "report"])
    ap.add_argument("--engine-root", type=Path, default=HOME / "src" / "Strata-0.1.40.3", help="the test build")
    ap.add_argument("--out", type=Path, default=HOME / "strata-quality")
    ap.add_argument("--arms", default="q8_0,q8_0@ec2400,iq3_s,ud-q4_k_xl,ud-q5_k_xl,iq3_s@2",
                    help="comma list, reference first; name@2 = a repeat run, name@ecN = with --expert-cache N (numerics floor)")
    ap.add_argument("--plan", default="short:6,8192:2,32768:1",
                    help="per source: context:requests (short = the scored text alone)")
    ap.add_argument("--port", type=int, default=8080, help="production port (stopped during the run)")
    ap.add_argument("--prod-config", type=Path, default=HOME / "src" / "Strata" / "strata-iq3_s.json",
                    help="used for lib_dirs, --mtp and decode settings when production is not running")
    ap.add_argument("--bench-port", type=int, default=8096)
    ap.add_argument("--unsloth-dir", type=Path, default=HOME / "models" / "unsloth" / "Qwen3.8-Flash-Next-GGUF")
    ap.add_argument("--ista-dir", type=Path, default=HOME / "models" / "strata" / "models" / "IQ3_S")
    ap.add_argument("--budget-gib", type=int, default=56, help="--resident-budget-gib for the Unsloth arms")
    ap.add_argument("--max-context", type=int, default=49152)
    ap.add_argument("--no-mtp", action="store_true", help="run every arm without the MTP draft layer")
    ap.add_argument("--load-timeout", type=float, default=1800)
    ap.add_argument("--request-timeout", type=float, default=5400)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--exe", type=Path, help="engine binary instead of <engine-root>/build/strata")
    ap.add_argument("--no-padded", action="store_true", help="use the original Unsloth shard 1 even if a -padded copy exists")
    a = ap.parse_args()
    root = a.engine_root
    py = root / ".venv" / "bin" / "python"
    if os.environ.get("STRATA_QUALITY_REEXEC") != "1" and py.exists() and \
            os.path.abspath(sys.executable) != os.path.abspath(str(py)):
        os.environ["STRATA_QUALITY_REEXEC"] = "1"
        os.execv(str(py), [str(py), os.path.abspath(__file__)] + sys.argv[1:])
    if a.command == "prepare":
        a.out.mkdir(parents=True, exist_ok=True)
        prepare(a, root)
    elif a.command == "run":
        cmd_run(a, root)
    else:
        cmd_report(a)


if __name__ == "__main__":
    main()
