"""Cold-prompt read speed A/B for the batched prompt path (PR #6813).

    python bench_prefill_ab.py run TAG --config PROFILE.json [--exe EXE] [--tokens 120000] [--seed N]
                                     [--env KEY=VALUE ...] [--arg --flag=value] [--out DIR]
    python bench_prefill_ab.py compare TAG [TAG ...]        # reads <out>/TAG.json

One run = one fresh engine started from a Strata launcher profile, one cold prompt of the target size,
max_tokens=8.  The authoritative number is the engine's own log line

    prompt N tokens = R reused + F read in T ms (X tok/s)

not the wall clock (the wall clock carries tokenization and the response).  The profile's learned expert
table is copied to a throwaway file and both --expert-profile and expert_profile_save point at the copy, so
synthetic routing never writes the table the real service learns into.

Paths are not hardcoded: --config points at any launcher profile, strata.py is taken from --serve,
$STRATA_SERVE, or next to the profile.  Windows and POSIX both work; the process stop falls back to SIGTERM.
"""
import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.request

PROMPT_RE = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\)")
RANGE_RE = re.compile(r"(\d+) ranges registered \((\d+) MiB\)(.{0,60})")
COMP_RE = re.compile(r"complement ready: resident ([\d.]+) GiB, pinned ([\d.]+) GiB")
PORT = 8080

TEMPLATES = [
    "Clause {i}.1 specifies a tolerance of {a:.3f} millimeters under ambient load scenario {g}.",
    "Section {i}.2 records calibration constant {b} measured at {c} degrees Celsius.",
    "Paragraph {i}.3 assigns inspector {d} to verify bracket assembly {e} before shipment.",
    "Item {i}.4 notes that sample {f} passed the vibration sweep at {h} cycles per second.",
    "Note {i}.5 requires the logbook entry for unit {j} to be countersigned by the duty engineer.",
]


def para(rng, i):
    """One paragraph of five sentences of plausible-looking filler: stable token density, no repeated block."""
    return " ".join(
        TEMPLATES[rng.randrange(5)].format(
            i=i, g=rng.choice("abcdefg"), a=rng.uniform(0.1, 9.9), b=rng.randint(1000, 9999),
            c=rng.randint(-40, 120), d=rng.randint(10, 99), e=rng.randint(100, 999),
            f=rng.randint(10000, 99999), h=rng.randint(5, 500), j=rng.randint(1000, 9999))
        for _ in range(12))


def build_prompt(seed, tokens, cpt):
    rng = random.Random(seed)
    out, total, i, want = [], 0, 0, int(tokens * cpt)
    while total < want:
        p = para(rng, i)
        out.append(p)
        total += len(p) + 2
        i += 1
    return "\n\n".join(out)[:int(tokens * 1.01 * cpt)]


def http(path, body=None, timeout=3600):
    url = f"http://127.0.0.1:{PORT}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def free_ram_gib():
    """Free physical RAM, so a run only starts once the previous engine has given its memory back."""
    if os.name == "nt":
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB"],
                             capture_output=True, text=True).stdout.strip()
    else:
        out = ""
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                out = str(int(line.split()[1]) / 1048576)
    try:
        return float(out)
    except ValueError:
        return -1.0


def wait_health(timeout=1200):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            http("/health", timeout=5)
            return True
        except Exception:
            time.sleep(5)
    return False


def stop(proc):
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        proc.terminate()
        proc.wait(timeout=60)


def read_facts(log):
    """The engine's own numbers: the biggest prompt line, the RAM-residency line, the cache size."""
    try:
        lines = open(log, encoding="utf-8", errors="replace").read().splitlines()
    except OSError:
        return {}
    prompts = [m for m in (PROMPT_RE.search(l) for l in lines) if m]
    out = {}
    if prompts:
        m = max(prompts, key=lambda m: int(m.group(1)))
        out.update(prompt_tokens=int(m.group(1)), reused=int(m.group(2)), fresh=int(m.group(3)),
                   read_ms=int(m.group(4)), tok_s=float(m.group(5)))
    for l in reversed(lines):
        m = RANGE_RE.search(l)
        if m:
            note = m.group(3).strip()
            out["ram_pin"] = {"kind": "ranges", "ranges": int(m.group(1)), "mib": int(m.group(2)),
                              "note": note, "healthy": "refused" not in note or "cap was reached" in note}
            break
        m = COMP_RE.search(l)
        if m:
            res, pnd = float(m.group(1)), float(m.group(2))
            out["ram_pin"] = {"kind": "complement", "resident_gib": res, "pinned_gib": pnd,
                              "healthy": pnd >= res * 0.95}
            break
    return out


def run(tag, a):
    cfg = json.load(open(a.config, encoding="utf-8"))
    out_dir = a.out or os.path.join(os.path.dirname(os.path.abspath(a.config)), "prefill-ab")
    os.makedirs(out_dir, exist_ok=True)
    if a.exe:
        cfg["exe"] = a.exe
    log = os.path.join(out_dir, f"{tag}.engine.log")
    for k, v in (e.split("=", 1) for e in a.env):
        cfg.setdefault("env", {})[k] = v
    if a.arg:
        cfg.setdefault("args", []).extend(a.arg)
    # The learned table goes to a copy: a synthetic prompt must not teach the real profile anything.
    args = cfg.get("args", [])
    src = args[args.index("--expert-profile") + 1] if "--expert-profile" in args else None
    tmp = os.path.join(out_dir, f"{tag}.profile.bin")
    if src and os.path.exists(src):
        shutil.copy(src, tmp)
        args[args.index("--expert-profile") + 1] = tmp
        cfg["expert_profile_save"] = tmp
    cfg["log"] = log
    cfg_path = os.path.join(out_dir, f"{tag}.json")
    json.dump(cfg, open(cfg_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    serve = a.serve or os.environ.get("STRATA_SERVE") or os.path.join(os.path.dirname(a.config), "strata.py")
    py = a.python or sys.executable
    for _ in range(90):
        if free_ram_gib() > a.min_ram_gib:
            break
        time.sleep(5)
    if os.path.exists(log):
        os.remove(log)
    lf = open(os.path.join(out_dir, f"{tag}.serve.log"), "w", encoding="utf-8")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    proc = subprocess.Popen([py, serve, "serve", "--engine", "strata", "--config", cfg_path,
                             "--port", str(PORT)], cwd=cfg.get("cwd"), stdout=lf,
                            stderr=subprocess.STDOUT, creationflags=flags)
    res = {"tag": tag, "pack": cfg.get("model_name"), "exe": cfg.get("exe"), "env": cfg.get("env"),
           "prefill": args[args.index("--prefill") + 1] if "--prefill" in args else None,
           "max_context": args[args.index("--max-context") + 1] if "--max-context" in args else None}
    try:
        if not wait_health():
            res["error"] = "engine never became healthy"
        else:
            model = http("/v1/models")["data"][0]["id"]
            probe = " ".join(para(random.Random(99), i) for i in range(40))
            u = http({"model": model, "messages": [{"role": "user", "content": probe}],
                      "max_tokens": 4})["usage"]
            prompt = build_prompt(a.seed, a.tokens, len(probe) / u["prompt_tokens"])
            t0 = time.time()
            r = http({"model": model, "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": a.max_tokens})
            res["wall_s"] = round(time.time() - t0, 1)
            res["response_tokens"] = r["usage"]["completion_tokens"]
            time.sleep(2)
    except Exception as e:
        res["error"] = str(e)
    finally:
        stop(proc)
        lf.close()
    res.update(read_facts(log))
    pin = res.get("ram_pin") or {}
    res["valid"] = bool(pin.get("healthy")) and "error" not in res and "tok_s" in res
    json.dump(res, open(os.path.join(out_dir, f"{tag}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return res


def compare(a, out_dir):
    rows = []
    for tag in a.names:
        p = os.path.join(out_dir, f"{tag}.json")
        rows.append(json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {"tag": tag, "error": "no result"})
    print(f"{'tag':16s} {'tokens':>7s} {'ms':>7s} {'tok/s':>7s} {'ram pin':26s} valid")
    for r in rows:
        pin = r.get("ram_pin") or {}
        ptxt = (f"ranges {pin.get('ranges')}/{pin.get('mib')} MiB" if pin.get("kind") == "ranges"
                else f"pinned {pin.get('pinned_gib')}/{pin.get('resident_gib')} GiB") if pin else "-"
        print(f"{r['tag']:16s} {str(r.get('prompt_tokens', '-')):>7s} {str(r.get('read_ms', '-')):>7s} "
              f"{str(r.get('tok_s', '-')):>7s} {ptxt:26s} {'yes' if r.get('valid') else 'NO'}")
    good = [r["tok_s"] for r in rows if r.get("valid")]
    if len(good) >= 2:
        print(f"valid runs: min {min(good)}, max {max(good)}, mean {sum(good) / len(good):.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["run", "compare"])
    ap.add_argument("names", nargs="+", help="the run name in run mode, the run names to line up in compare mode")
    ap.add_argument("--config", help="launcher profile json (run mode)")
    ap.add_argument("--serve", help="path to strata.py (default $STRATA_SERVE, or next to --config)")
    ap.add_argument("--python", help="interpreter for strata.py (default: this one)")
    ap.add_argument("--exe", help="override the profile's engine binary")
    ap.add_argument("--env", action="append", default=[], metavar="K=V", help="engine env var, repeatable")
    ap.add_argument("--arg", action="append", default=[], metavar="--flag=value", help="extra engine arg")
    ap.add_argument("--tokens", type=int, default=120000)
    ap.add_argument("--seed", type=int, default=20261123)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--min-ram-gib", type=float, default=48.0)
    ap.add_argument("--out", help="where results go (default: prefill-ab next to the profile)")
    a = ap.parse_args()
    if a.mode == "run":
        if not a.config:
            ap.error("run needs --config")
        run(a.names[0], a)
    else:
        base = a.out or os.path.join(os.path.dirname(os.path.abspath(a.config or ".")), "prefill-ab")
        compare(a, base)


if __name__ == "__main__":
    main()
