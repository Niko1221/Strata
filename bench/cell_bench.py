"""Measure decode tok/s for one (quant, context) cell of the matrix.

Generation speed is the headline, so that is what this reports. It is measured on a
SHORT prompt on purpose: a long window costs you decode throughput even when you are
not using the window, because the KV cache it holds is VRAM the expert cache cannot
have. The long-prompt prefill number is reported too, but it is a separate cost.

Every cell: reload the model at that setting, let it settle, then take the median of
three 160-token generations. Median rather than mean because one request can land on a
cache-miss storm or a checkpoint, and that is noise, not the setting's speed.
"""
import json, random, statistics, string, subprocess, sys, time, urllib.request, pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
PORT = 8090
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

def http_json(path, payload=None, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}",
                                 data=json.dumps(payload).encode() if payload is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def stop_engine():
    subprocess.run(["powershell.exe", "-NoProfile", "-Command",
                    "Get-Process strata -ErrorAction SilentlyContinue | Stop-Process -Force"],
                   capture_output=True, timeout=120)
    # the server.py parent lingers after its engine dies; free the port too
    out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if f":{PORT} " in line and "LISTENING" in line:
            pid = line.split()[-1]
            if pid.isdigit():
                subprocess.run(["powershell.exe", "-NoProfile", "-Command",
                                f"Stop-Process -Id {pid} -Force -ErrorAction SilentlyContinue"],
                               capture_output=True, timeout=60)
    time.sleep(3)

def start_engine(cfg_path):
    p = subprocess.Popen([str(REPO / ".venv/Scripts/python.exe"), str(REPO / "serve/server.py"),
                          "--engine", "strata", "--config", str(cfg_path), "--port", str(PORT)],
                         cwd=str(REPO), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 900
    while time.time() < deadline:
        try:
            if http_json("/health", timeout=5).get("loaded"):
                return p
        except Exception:
            pass
        if p.poll() is not None:
            raise RuntimeError("server exited during load")
        time.sleep(10)
    raise RuntimeError("load timed out")

def decode_tok_s(trials=3, max_tokens=160):
    # warm the path first: the first request pays graph setup, not throughput
    http_json("/v1/chat/completions", {"model": "x", "max_tokens": 4, "temperature": 0,
                                       "reasoning_effort": "none",
                                       "messages": [{"role": "user", "content": "warm"}]}, timeout=600)
    rates = []
    for _ in range(trials):
        # A fresh prompt per trial. The server keeps a conversation cache, so repeating one
        # prompt verbatim measures the cache rather than generation - that mistake first showed
        # up as an impossible 160 tok/s. Reject any trial the server reports as cached.
        tag = "".join(random.choice(string.ascii_letters + string.digits) for _ in range(48))
        t0 = time.time()
        d = http_json("/v1/chat/completions",
                      {"model": "x", "max_tokens": max_tokens, "temperature": 0,
                       "reasoning_effort": "none",
                       "messages": [{"role": "user",
                                     "content": f"[ref {tag}] Count from 1 to 80 separated by "
                                                f"commas. Output only the list."}]})
        dt = time.time() - t0
        ct = d["usage"]["completion_tokens"]
        cached = (d["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        if ct > 40 and not cached:
            rates.append(ct / dt)
    return statistics.median(rates) if rates else 0.0

def log_facts(log):
    """Pull the expert-cache slot count and the final free VRAM out of the engine log.

    The free-VRAM line reads "strata serve: 662 MiB of VRAM free with everything loaded",
    so the number is a MIDDLE field - take the first token that parses as a number rather
    than assuming a position. (Assuming split()[0] made this raise ValueError on 'strata',
    which cost a full matrix run.)
    """
    facts = {}
    try:
        for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "expert cache" in line and "slots" in line and "auto" not in line:
                for tok in line.split():
                    if tok.isdigit():
                        facts["slots"] = int(tok)
                        break
            if "of VRAM free with everything loaded" in line:
                for tok in line.split():
                    try:
                        facts["free_mib"] = round(float(tok))
                        break
                    except ValueError:
                        continue
    except OSError:
        pass
    return facts

CONFIG_FOR_QUANT = {
    "IQ2_XS": "strata-iq2_xs.json",
    "IQ3_XXS": "strata-iq3_xxs.json",
}

def main():
    quant, ctx = sys.argv[1], int(sys.argv[2])
    kv = sys.argv[3] if len(sys.argv) > 3 else "int8"
    print(f"=== {quant} @ {ctx//1024}K, kv={kv} ===", flush=True)

    cfg_name = CONFIG_FOR_QUANT.get(quant)
    if not cfg_name or not (REPO / cfg_name).exists():
        raise SystemExit(f"no config for {quant} (expected {cfg_name})")
    log_name = cfg_name.replace(".json", ".log")

    stop_engine()
    cfg_path = REPO / cfg_name
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    a = cfg["args"]
    if "--max-context" in a:
        i = a.index("--max-context"); a[i+1] = str(ctx)
    if "--kv" in a:
        i = a.index("--kv"); a[i+1] = kv
    cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    (REPO / log_name).write_text("", encoding="utf-8")

    proc = start_engine(cfg_path)
    try:
        tps = decode_tok_s()
        f = log_facts(REPO / log_name)
        print(f"RESULT quant={quant} ctx={ctx} kv={kv} tok_s={tps:.1f} "
              f"slots={f.get('slots','?')} free_mib={f.get('free_mib','?')}", flush=True)
    finally:
        stop_engine()
        if proc.poll() is None:
            proc.terminate()

if __name__ == "__main__":
    main()
