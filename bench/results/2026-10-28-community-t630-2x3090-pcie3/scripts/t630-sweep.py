"""t630-sweep.py - unattended one-change-at-a-time Strata sweep on the T630 (Linux, 1x RTX 3090).

Each arm is the production run config (~/.config/strata/t630/strata-iq3_s.json) with one change:
engine args added or removed, or environment variables. For each arm the script stops strata.service,
serves the arm in a transient systemd unit on :8080, waits for a token, runs speed-checks.py REPEAT
times, records the engine log's decode hit rates, and appends one JSON line per arm. When the sweep
ends (or fails) it restarts strata.service on production, so the box serves in the morning.

Usage: python3 -I t630-sweep.py OUT_DIR ARMS_JSON REPEAT SPEED_CHECKS_PY   (SWEEP_BASE=<run config> to start from another config)
"""
import json, os, re, subprocess, sys, time, urllib.request, shlex

HOME = os.path.expanduser("~")
PROD = os.environ.get("SWEEP_BASE", f"{HOME}/.config/strata/t630/strata-iq3_s.json")
STRATA = f"{HOME}/.local/bin/strata"
SPEED = sys.argv[4] if len(sys.argv) > 4 else None
OUT, ARMS = sys.argv[1], json.load(open(sys.argv[2]))
REPEAT = int(sys.argv[3]) if len(sys.argv) > 3 else 2
UNIT = "strata-sweep"
os.makedirs(OUT, exist_ok=True)
prod = json.load(open(PROD))
KEY = prod["api_key"]


def sh(cmd, check=False):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(f"{OUT}/sweep.log", "a") as f:
        f.write(line + "\n")


def stop_all():
    sh(f"sudo systemctl stop {UNIT}.service strata.service")
    sh(f"sudo systemctl reset-failed {UNIT}.service")
    for _ in range(90):
        if ":8080 " not in sh("ss -ltn").stdout:
            break
        time.sleep(2)
    else:
        raise RuntimeError("port 8080 still busy")
    # Each arm starts from the same memory state: the page cache had filled NUMA node 1 overnight, so the
    # resident copy's per-node placement failed and later arms ran slower (2026-10-24, round 1).
    sh("sync; echo 3 | sudo tee /proc/sys/vm/drop_caches")
    log("after cache drop: " + ", ".join(re.findall(r"node \d free: \d+ MB", sh("numactl -H").stdout)))


def wait_ready(limit_s=3600):
    body = json.dumps({"model": prod["model_name"], "max_tokens": 1,
                       "messages": [{"role": "user", "content": "hi"}]}).encode()
    t0 = time.time()
    while time.time() - t0 < limit_s:
        if sh(f"systemctl is-active {UNIT}.service").stdout.strip() not in ("active", "activating"):
            raise RuntimeError(f"server exited: {sh(f'journalctl -u {UNIT} -n 3 --no-pager -o cat').stdout.strip()}")
        try:
            req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", body,
                                         {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
            urllib.request.urlopen(req, timeout=30).read()
            return time.time() - t0
        except Exception:
            time.sleep(10)
    raise RuntimeError("server never answered")


def arm_config(arm):
    c = json.loads(json.dumps(prod))
    args = list(c["args"])
    for flag in arm.get("remove", []):        # remove a flag and, when it takes one, its value
        if flag in args:
            i = args.index(flag)
            n = 2 if i + 1 < len(args) and not args[i + 1].startswith("--") else 1
            del args[i:i + n]
    args += arm.get("add", [])
    c["args"] = args
    c["env"] = {**c.get("env", {}), **arm.get("env", {})}
    c["log"] = f"{OUT}/engine-{arm['name']}.log"
    for k, v in arm.get("config", {}).items():
        c[k] = v
    # The run config carries the server's API key: serve from a private copy, keep a redacted one with the results.
    private = f"{HOME}/.config/strata/t630/sweep"
    os.makedirs(private, exist_ok=True)
    path = f"{private}/config-{arm['name']}.json"
    json.dump(c, open(path, "w"), indent=1)
    os.chmod(path, 0o600)
    json.dump({**c, "api_key": "<redacted>"}, open(f"{OUT}/config-{arm['name']}.json", "w"), indent=1)
    return path, c


def parse_speed(text):
    r = {}
    m = re.search(r"thinking off: ([\d., ]+) tok/s", text); r["off"] = [float(x) for x in m.group(1).split(",")] if m else None
    m = re.search(r"thinking on: ([\d., ]+) tok/s", text); r["on"] = [float(x) for x in m.group(1).split(",")] if m else None
    m = re.search(r"prompt read (\d+) tok/s \(TTFT ([\d.]+)s\), generation ([\d.]+)", text)
    if m:
        r["read32k"], r["ttft32k"], r["gen32k"] = int(m.group(1)), float(m.group(2)), float(m.group(3))
    m = re.search(r"prefix cache\): TTFT ([\d.]+)s, generation ([\d.]+)", text)
    if m:
        r["cached_ttft"], r["cached_gen"] = float(m.group(1)), float(m.group(2))
    # agent-turns.py: per-turn decode tok/s and TTFT on coding-agent-shaped turns
    for name, n, rate, ttft in re.findall(r"agent (\S+): (\d+) tokens, decode ([\d.]+) tok/s, ttft ([\d.]+)s", text):
        r.setdefault("agent", {})[name] = float(rate)
        r.setdefault("agent_tokens", {})[name] = int(n)
        r.setdefault("agent_ttft", {})[name] = float(ttft)
    return r


def hit_rates(logfile):
    try:
        txt = open(logfile, errors="replace").read()
    except OSError:
        return {}
    rates = [float(x) for x in re.findall(r"decode expert cache hit rate: ([\d.]+)%", txt)]
    slots = re.findall(r"expert cache (\d+) slots, ([\d.]+) GiB", txt)
    pinned = re.findall(r"pinned ([\d.]+) GiB", txt)
    # The start-up PCIe probe: ~11.6 GB/s when the pinned buffers land on the GPU's node (0), ~4.7 when on
    # node 1, and the engine then runs slower for the whole session (2026-10-24, round 3).
    pcie = re.findall(r"PCIe probe: ([\d.]+) GB/s", txt)
    return {"hit_rates": rates[-8:], "slots": slots[-1] if slots else None, "pinned": pinned[-1:] or None,
            "pcie_gbs": [float(x) for x in pcie]}


def run_arm(arm):
    path, c = arm_config(arm)
    stop_all()
    # "power_limit": the arm's GPU cap in watts; every other arm runs at production's 250 W.
    sh(f"sudo nvidia-smi -pl {arm.get('power_limit', 250)}")
    # "pre" / "post": shell commands around the arm (e.g. reserving hugepages and giving them back).
    if arm.get("pre"):
        log(f"{arm['name']} pre: {sh(arm['pre']).stdout.strip()[:200]}")
    env = " ".join(f"-E {shlex.quote(f'{k}={v}')}" for k, v in c["env"].items())   # values may hold ; or spaces
    cmd = (f"sudo systemd-run --unit={UNIT} --uid=<user> --gid=<user> -p LimitMEMLOCK=infinity -p LimitNOFILE=65536 "
           f"-p WorkingDirectory={HOME}/.local/share/strata/terminal/current {env} "
           f"{arm.get('prefix', '')} {STRATA} serve --profile {path} --host 127.0.0.1 --port 8080 "
           f"--parallel {arm.get('parallel', 1)}")
    out = sh(cmd)
    if out.returncode:
        raise RuntimeError(f"systemd-run failed: {out.stderr.strip()}")
    load_s = wait_ready()
    log(f"{arm['name']}: ready in {load_s:.0f}s")
    runs = []
    gpu = subprocess.Popen(["nvidia-smi", "--query-gpu=power.draw,temperature.gpu,fan.speed",
                            "--format=csv,noheader,nounits", "-lms", "1000"], stdout=subprocess.PIPE, text=True)
    chassis = []  # iDRAC temperatures (inlet, CPU 1, CPU 2) after each run, for the cooling question
    for i in range(REPEAT):
        p = sh(f"STRATA_KEY={KEY} SPEED_NONCE={arm['name']}-{i + 1}-{time.time():.0f} python3 -I {SPEED} http://127.0.0.1:8080")
        open(f"{OUT}/speed-{arm['name']}-{i + 1}.txt", "w").write(p.stdout + p.stderr)
        runs.append(parse_speed(p.stdout))
        chassis.append([int(t) for t in re.findall(r"\| (\d+) degrees C", sh("sudo ipmitool sdr type Temperature").stdout)])
        log(f"{arm['name']} run {i + 1}: {runs[-1]}")
    gpu.terminate()
    if arm.get("post"):
        sh(arm["post"])
    rows = [[float(x) for x in l.split(",")] for l in gpu.communicate()[0].splitlines() if l.count(",") == 2]
    busy = [r for r in rows if r[0] > 60] or rows
    power = {"gpu_w_avg": round(sum(r[0] for r in busy) / len(busy)), "gpu_w_max": round(max(r[0] for r in busy)),
             "gpu_c_max": max(r[1] for r in busy), "gpu_fan_avg": round(sum(r[2] for r in busy) / len(busy))} if busy else {}
    if chassis:
        power["chassis_c_max"] = [max(col) for col in zip(*chassis)]
    log(f"{arm['name']} gpu: {power}")
    rec = {"arm": arm["name"], "change": arm, "load_s": round(load_s), "runs": runs, **power,
           **hit_rates(c["log"]), "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(f"{OUT}/sweep.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


try:
    for arm in ARMS:
        try:
            run_arm(arm)
        except Exception as e:  # noqa: BLE001 - one bad arm must not end the night
            log(f"{arm['name']}: FAILED {e!r}")
            with open(f"{OUT}/sweep.jsonl", "a") as f:
                f.write(json.dumps({"arm": arm["name"], "change": arm, "error": repr(e)}) + "\n")
finally:
    log("restoring production strata.service")
    stop_all()
    sh("sudo nvidia-smi -pl 250")
    sh("sudo sysctl -q vm.nr_hugepages=0")  # in case a hugepage arm failed before its "post"
    sh("sudo systemctl start strata.service")
    log("done")
