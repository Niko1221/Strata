#!/usr/bin/env python3
"""perf_matrix.py: the benchy matrix (sycl/benchy.sh runs it) - every model x prompt size, each with its serve config.

    sudo python3 sycl/tools/perf_matrix.py [--configs strata-a.json,...] [--sizes 20,2185,...] [--warm] [--out DIR]

Benchy v1, fixed so that every card runs the same bench:

- **Prompts:** sycl/bench/v1. `short.ids` is the 20-token Fibonacci prompt. `long.ids` is 2,185 tokens of code and
  prose. The longer sizes are long.ids' first 2,000 tokens repeated to length. Every model in the family shares the
  tokenizer, so the ids are the same everywhere.
- **Sizes:** 20, 2,185, 8,000, 40,000, 128,000 and 256,000 tokens. A size that does not fit the config's
  `--max-context` with the new tokens is skipped and listed.
- **Each run:** a fresh engine process with the config's args verbatim, plus `--tokens-file`,
  `--max-new 256 --greedy --stats`, and strata-sycl.sh's environment. The page cache is dropped before each run,
  so every run is a cold start.
- **Configs:** every `strata-*.json` with `"backend": "sycl"` next to the checkout (what `sycl/setup_intel.py`
  writes), except steering variants (`--control-vector*`).

Root reads the VRAM (xe reports it per client in /proc/*/fdinfo, root only) and drops the page cache. Without root
(`--warm`), VRAM comes from /run/gpustat.json if a sampler writes it, and the runs are warm; the table says so.

Out (--out, default sycl/benchy-results/v1-<date>; it must sit under the data root, which the container mounts): `matrix.md` (the machine and the table - post it),
`config-<name>.txt` (each config's args - post them too), `matrix.jsonl` (raw numbers) and one log per run.
"""
import argparse, glob, json, os, re, shlex, subprocess, sys, time
from pathlib import Path

VERSION = "v1"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = Path(os.environ.get("STRATA_SYCL_ROOT", REPO.parent))      # mounted at /work, as in strata-sycl.sh
IMAGE = os.environ.get("STRATA_SYCL_IMAGE", "strata-sycl-dev")
BIN = os.environ.get("STRATA_SYCL_BIN", "build-sycl-aot/strata")
# docker or podman (the strata-sycl-dev image; rootless podman runs as the user who ran sudo, with that user's
# images), or distrobox:<name> - a toolbox with oneAPI where ROOT is /work, for hosts such as Fedora Silverblue that
# keep the toolchain in a container
RUNNER = os.environ.get("STRATA_SYCL_RUNNER", "docker")
BOX = RUNNER.split(":", 1)[1] if RUNNER.startswith("distrobox:") else None
PROMPTS = REPO / "sycl" / "bench" / VERSION
SIZES = [20, 2185, 8000, 40000, 128000, 256000]
NEW = 256
ENV = ["STRATA_VERIFY_DEVICE_PLAN=1", "STRATA_VERIFY_NO_HOST=1", "STRATA_STAGER_THREADS=12"]   # as strata-sycl.sh
CONTAINER = "strata-benchy"
sys.path.insert(0, str(HERE))
import gpustat as gs   # noqa: E402  (sysfs/fdinfo readers: PCIe link, hwmon, per-client VRAM)


# ---------------------------------------------------------------- the machine
def rd(p, default=None):
    return gs.rd(p, default)


def meminfo(key="MemAvailable"):
    for line in open("/proc/meminfo"):
        if line.startswith(key + ":"):
            return int(line.split()[1]) * 1024
    return 0


def intel_cards():
    """every Intel GPU on xe: [{pci, name, vram_gb, link, link_max}]"""
    out = []
    for d in sorted(glob.glob("/sys/class/drm/card[0-9]*/device")):
        if rd(f"{d}/vendor") != "0x8086" or not glob.glob(f"{d}/tile*/gt*"):
            continue
        gs.PCI = os.path.basename(os.path.realpath(d))
        link = gs.pcie_link() or {}
        cur, mx = link.get("cur") or {}, link.get("card_max") or {}
        out.append({"pci": gs.PCI, "name": gs.card_name((rd(f"{d}/device", "0x0") or "0x0")[2:]),
                    "vram_gb": round(gs.vram_total_mb() / 1024), "link": f"Gen{cur.get('gen', '?')} x{cur.get('width', '?')}",
                    "link_max": f"Gen{mx.get('gen', '?')} x{mx.get('width', '?')}"})
    return out


def xe_energy_uj():
    """card energy summed over every xe hwmon (energy1 = the whole card)"""
    tot = 0
    for n in glob.glob("/sys/class/hwmon/hwmon*/name"):
        if rd(n) == "xe":
            tot += int(rd(os.path.join(os.path.dirname(n), "energy1_input"), "0") or 0)
    return tot


def vram_used_mb(root):
    if root:
        return sum(c["vram_kb"] for c in gs.clients().values()) / 1024
    try:
        return json.loads(Path("/run/gpustat.json").read_text())["vram_used_mb"]
    except (OSError, ValueError, KeyError):
        return None


def vram_by_card_mb():
    """{pci: resident VRAM of every client on that card} (root: fdinfo)"""
    out = {}
    for c in gs.clients().values():
        out[c["pdev"]] = out.get(c["pdev"], 0) + c["vram_kb"] / 1024
    return out


def disk_of(path):
    src = subprocess.run(["findmnt", "-no", "SOURCE", "--target", str(path)], capture_output=True, text=True).stdout.strip()
    name = os.path.basename(src.split("[")[0])
    return re.sub(r"p\d+$", "", name) if name.startswith("nvme") else re.sub(r"\d+$", "", name)


def disk_read_bytes(dev):
    for line in open("/proc/diskstats"):
        f = line.split()
        if f[2] == dev:
            return int(f[5]) * 512
    return 0


def machine(dev, cold, root):
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?")
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()
    if BOX:
        icpx = subprocess.run(box_cmd(". /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; icpx --version | head -1"),
                              capture_output=True, text=True).stdout.strip().splitlines()
        image = f"distrobox {BOX}, {icpx[-1] if icpx else 'oneAPI ?'}"
    else:
        img = subprocess.run(ctr("image", "inspect", "-f", "{{.Id}}", IMAGE), capture_output=True, text=True).stdout.strip()
        img = img[7:] if img.startswith("sha256:") else img
        image = f"{IMAGE} {img[:12]}" + (" (podman)" if RUNNER == "podman" else "")
    return {
        "bench": f"benchy {VERSION}", "date": time.strftime("%Y-%m-%d"),
        "cards": intel_cards(), "cpu": cpu, "threads": os.cpu_count(), "ram_gib": round(meminfo("MemTotal") / 2**30, 1),
        "ssd": f"{dev} ({rd(f'/sys/block/{dev}/device/model', '?')})", "kernel": os.uname().release,
        "os": next((l.split("=", 1)[1].strip().strip('"') for l in open("/etc/os-release") if l.startswith("PRETTY_NAME")), "?"),
        "commit": (git("rev-parse", "--short=12", "HEAD") or "?") + (" + local changes" if git("status", "--porcelain", "--untracked-files=no") else ""),
        "image": image, "binary": BIN, "cold_cache": cold, "vram_source": "fdinfo" if root else "gpustat.json",
    }


# ---------------------------------------------------------------- one run


def as_user(cmd):
    """cmd as the user who ran sudo: a rootless container (podman, distrobox) is that user's, not root's"""
    uid = os.environ.get("SUDO_UID")
    if os.geteuid() == 0 and uid:
        user = os.environ.get("SUDO_USER") or uid
        cmd = ["runuser", "-u", user, "--", "env", f"XDG_RUNTIME_DIR=/run/user/{uid}", f"HOME={Path('~' + user).expanduser()}"] + cmd
    return cmd


def ctr(*args):
    return as_user(["podman", *args]) if RUNNER == "podman" else ["docker", *args]


def box_cmd(script):
    """a command line that runs script in the distrobox (benchy runs as root via sudo)"""
    return as_user(["distrobox", "enter", BOX, "--", "bash", "-c", script])


def engine_cmd(args, sel):
    run = f"cd /work/{REPO.name} && exec {BIN} " + " ".join(shlex.quote(a) for a in args)
    if BOX:   # the image's environment: its single-card selector, oneAPI's libraries, the OOM killer's first pick
        env = ENV + [f"ONEAPI_DEVICE_SELECTOR={sel or 'level_zero:0'}"]
        return box_cmd("echo 1000 > /proc/self/oom_score_adj; . /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; export "
                       + " ".join(shlex.quote(e) for e in env) + "; " + run)
    subprocess.run(ctr("rm", "-f", CONTAINER), capture_output=True)
    # podman: the data disk may carry no SELinux labels, which a confined container cannot read
    return ctr("run", "--rm", "--name", CONTAINER, "--device", "/dev/dri", "--oom-score-adj", "1000",
               *(["--security-opt", "label=disable"] if RUNNER == "podman" else []),
               "-v", f"{ROOT}:/work") + sum((["-e", e] for e in ENV), []) + (["-e", f"ONEAPI_DEVICE_SELECTOR={sel}"] if sel else []) + \
           [IMAGE, run]


def kill_engine():
    if BOX:   # the engine is the container's child, not ours
        subprocess.run(["pkill", "-KILL", "-x", Path(BIN).name[:15]] + (["-U", os.environ["SUDO_UID"]] if os.environ.get("SUDO_UID") else []))
    else:
        subprocess.run(ctr("kill", CONTAINER), capture_output=True)
def prompt_file(n, outdir):
    """the v1 prompt of n tokens, written under the data root so the container sees it"""
    short = (PROMPTS / "short.ids").read_text().strip().split(",")
    long_ = (PROMPTS / "long.ids").read_text().strip().split(",")
    if n == len(short):
        ids = short
    elif n == len(long_):
        ids = long_
    else:
        ids = (long_[:2000] * (n // 2000 + 1))[:n]
    p = outdir / "prompts" / f"{n}.ids"
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text(",".join(ids))
    return p


def to_container(p: Path):
    return "/work/" + str(p.resolve().relative_to(ROOT.resolve()))


def to_host(p: str):
    return ROOT / p[len("/work/"):] if p.startswith("/work/") else Path(p)


def parse(log):
    g = lambda pat, t=float: (lambda m: t(m.group(1)) if m else None)(re.search(pat, log, re.M))
    return {
        "engine": g(r"session is up \(engine ([^)]+)\)", str),
        "pp": g(r"^prefill\s+\d+ tokens in [\d.]+ ms\s+->\s+([\d.]+) tok/s"),
        "ttft_s": (g(r"time to first token ([\d.]+) ms") or 0) / 1000 or None,
        "decode": g(r"^decode\s+\d+ tokens in [\d.]+ ms\s+->\s+([\d.]+) tok/s"),
        "decoded": g(r"^decode\s+(\d+) tokens in", int),
        "accept": g(r"drafts accepted \d+ of \d+ \(([\d.]+)\)"),
        "cache_slots": g(r"expert cache (\d+) slots", int),
        "cache_gib": g(r"expert cache \d+ slots, ([\d.]+) GiB"),
        "mirror_experts": g(r"(\d+) of \d+ experts missing from VRAM mirrored", int) or 0,
        "mirror_gib": g(r"experts missing from VRAM mirrored in pinned host memory \(([\d.]+) GiB") or 0.0,
        "lent_slots": g(r"prompt path borrows (\d+) cache slots", int) or 0,
        "lend_mirror_s": g(r"lendable slots mirrored in pinned host memory \([\d.]+ GiB, ([\d.]+) s\)") or 0.0,
        "ple_ssd_mb": g(r"SSD reads \(([\d.]+) MB\)"),
        "exit": g(r"ENGINE EXIT (-?\d+)", int),
    }


def run_one(name, cfg, n, outdir, cold, root, dev, timeout):
    args0 = cfg["args"]
    ctx = int(args0[args0.index("--max-context") + 1]) if "--max-context" in args0 else 0
    if ctx and n + NEW > ctx:
        return {"config": name, "prompt": n, "ctx": ctx, "skipped": f"needs --max-context {n + NEW}"}
    args = args0 + ["--tokens-file", to_container(prompt_file(n, outdir)), "--max-new", str(NEW), "--greedy", "--stats"]
    if cold:
        subprocess.run(["sync"]); Path("/proc/sys/vm/drop_caches").write_text("3\n"); time.sleep(2)
    log_path = outdir / f"{name}-{n}.log"
    sel = os.environ.get("ONEAPI_DEVICE_SELECTOR") or ("level_zero:gpu" if "--layer-split" in args0 else None)   # as strata-sycl.sh
    cmd = engine_cmd(args, sel)
    base_avail, base_disk, base_vram = meminfo(), disk_read_bytes(dev), vram_used_mb(root) or 0
    t0 = time.time(); peak_vram = 0.0; min_avail = base_avail
    base_cards = vram_by_card_mb() if root else {}; peak_cards = {}   # which card the engine ran on
    load_s = load_disk = t_up = e_up = t_end = e_end = None
    with open(log_path, "w") as lf:
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        while p.poll() is None:
            time.sleep(1)
            v = vram_used_mb(root)
            if v is not None:
                peak_vram = max(peak_vram, v - base_vram)
            for pci, mb in (vram_by_card_mb() if root else {}).items():
                peak_cards[pci] = max(peak_cards.get(pci, 0.0), mb - base_cards.get(pci, 0.0))
            min_avail = min(min_avail, meminfo())
            text = log_path.read_text(errors="replace")
            if load_s is None and "session is up" in text:
                load_s, load_disk, t_up, e_up = time.time() - t0, disk_read_bytes(dev) - base_disk, time.time(), xe_energy_uj()
            if t_end is None and re.search(r"^decode\s+\d+ tokens", text, re.M):
                t_end, e_end = time.time(), xe_energy_uj()
            if time.time() - t0 > timeout:   # a hang, not a slow run: the timeout is several times the expected time
                kill_engine()
                lf.write("\nBENCHY TIMEOUT\n")
                break
        p.wait()
        lf.write(f"\nENGINE EXIT {p.returncode}\n")
    total_disk = disk_read_bytes(dev) - base_disk
    log = log_path.read_text(errors="replace")
    if t_end is None:
        t_end, e_end = time.time(), xe_energy_uj()
    avg_w = (e_end - e_up) / 1e6 / (t_end - t_up) if t_up and t_end > t_up + 1 else None
    parsed = parse(log)
    if load_s is not None:   # start-up work the engine does after "session is up" (serve mode: before READY)
        load_s += parsed["lend_mirror_s"]
    return {"config": name, "prompt": n, "new": NEW, "ctx": ctx, **parsed,
            "cards_used": sorted(c for c, mb in peak_cards.items() if mb > 1024),
            "peak_vram_gb": peak_vram / 1024 if (root or peak_vram) else None, "ram_gb": (base_avail - min_avail) / 2**30,
            "avg_power_w": avg_w, "load_s": load_s, "ssd_load_gb": (load_disk or 0) / 1e9,
            "ssd_request_gb": (total_disk - (load_disk or 0)) / 1e9, "wall_s": time.time() - t0,
            "timeout": "BENCHY TIMEOUT" in log}


# ---------------------------------------------------------------- the table
def fmt(v, spec):
    return "-" if v is None else format(v, spec)


def table(rows, m):
    """two tables (speed in the layout of gresstant's B580 report on the upstream PR, then resources), the system and
    build, and the caveats"""
    done = [r for r in rows if "skipped" not in r]
    ok = lambda r: r.get("exit") == 0 and not r.get("timeout") and r.get("decode") is not None
    out = [f"**benchy {VERSION}** ({m['date']})", "",
           "| Model / configuration | Input tokens | PP (tok/s) | TTFT (s) | TG (tok/s) | Drafts accepted | Actual output tokens | Completed requests |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in done:
        out.append(f"| {r['config']}, {r['ctx'] // 1024}K context | {r['prompt']:,} | {fmt(r['pp'], ',.2f')} | {fmt(r['ttft_s'], '.1f')} "
                   f"| {fmt(r['decode'], '.2f')} | {fmt(r['accept'], '.0%')} | {fmt(r.get('decoded'), ',')} | {1 if ok(r) else 0}/1 |")
    out += ["", "| Model / configuration | Input tokens | Experts in VRAM | Offloaded to the RAM mirror | Slots lent to the prompt "
                "| Peak VRAM (GB) | RAM (GB) | SSD read at load (GB) | SSD read, request (GB) | PLE rows from SSD (MB) | Load (s) | Avg power (W) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in done:
        out.append(f"| {r['config']} | {r['prompt']:,} | {fmt(r['cache_slots'], ',')} ({fmt(r['cache_gib'], '.1f')} GiB) "
                   f"| {r['mirror_experts']:,} ({r['mirror_gib']:.1f} GiB) | {r['lent_slots']:,} | {fmt(r['peak_vram_gb'], '.1f')} "
                   f"| {r['ram_gb']:.1f} | {r['ssd_load_gb']:.1f} | {r['ssd_request_gb']:.2f} | {fmt(r['ple_ssd_mb'], ',.0f')} "
                   f"| {fmt(r['load_s'], '.0f')} | {fmt(r['avg_power_w'], '.0f')} |")
    failed = [f"{r['config']} {r['prompt']:,} (exit {r.get('exit')}{', timeout' if r.get('timeout') else ''})" for r in done if not ok(r)]
    skipped = [f"{r['config']} {r['prompt']:,} ({r['skipped']})" for r in rows if "skipped" in r]
    cards = "; ".join(f"{c['name']}, {c['vram_gb']} GB, PCIe {c['link']} (card max {c['link_max']})" for c in m["cards"]) or "?"
    names = {c["pci"]: c["name"] for c in m["cards"]}
    used = sorted({p for r in done for p in r.get("cards_used", [])})
    out += ["", "System and build:",
            f"- {cards}.",
            *([f"- Ran on: {', '.join(f'{names.get(p, p)} ({p})' for p in used)} (the card(s) holding the engine's VRAM)."] if used else []),
            f"- {m['cpu']} ({m['threads']} threads), {m['ram_gib']} GiB RAM; models on {m['ssd']}.",
            f"- {m['os']}, kernel {m['kernel']}.",
            f"- Engine {next((r['engine'] for r in rows if r.get('engine')), '?')}, commit `{m['commit']}`, image `{m['image']}`, "
            f"`{m['binary']}`. Each configuration's engine args: `config-<name>.txt`.",
            "", "Caveats:",
            f"- One run per row, greedy, {NEW} new tokens, a fresh engine each time; "
            + ("the page cache dropped before every run (cold start)." if m["cold_cache"] else "a WARM page cache (not run as root)."),
            "- TG includes speculative decoding (the MTP draft layer): it depends on the text, and on how often drafts are accepted.",
            "- RAM is the drop in the host's available memory (pinned memory included); VRAM is "
            + ("every client's resident VRAM from fdinfo, less what was in use before." if m["vram_source"] == "fdinfo" else "from /run/gpustat.json."),
            *(["- Avg power sums every Intel card's energy counter, so an idle second card's draw is included."] if len(m["cards"]) > 1 else [])]
    if failed:
        out.append("- Did not complete: " + ", ".join(failed) + ".")
    if skipped:
        out.append("- Skipped (context too small): " + ", ".join(skipped) + ".")
    return "\n".join(out) + "\n"


def default_configs():
    out = []
    for p in sorted(REPO.glob("strata-*.json")):
        try:
            c = json.loads(p.read_text())
        except ValueError:
            continue
        if c.get("backend") == "sycl" and not any(a.startswith("--control-vector") for a in c.get("args", [])):
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser(description=f"benchy {VERSION}: the Intel SYCL performance matrix")
    ap.add_argument("--configs", help="comma-separated serve configs (default: every sycl strata-*.json next to the checkout)")
    ap.add_argument("--sizes", default=",".join(map(str, SIZES)), help="prompt sizes (default: v1's)")
    ap.add_argument("--warm", action="store_true", help="do not drop the page cache (implied without root)")
    ap.add_argument("--force", action="store_true", help="run even if another process holds VRAM")
    ap.add_argument("--out", help="output directory under the data root (default sycl/benchy-results/v1-<date>)")
    a = ap.parse_args()
    root = os.geteuid() == 0
    cold = root and not a.warm
    cfgs = [Path(c) if os.path.isabs(c) else REPO / c for c in a.configs.split(",")] if a.configs else default_configs()
    if not cfgs:
        sys.exit("benchy: no serve config found - run sycl/setup_intel.py for a model first")
    if not (REPO / BIN).exists():
        sys.exit(f"benchy: {BIN} is not built (sycl/tools/build.sh)")
    held = vram_used_mb(root)
    if held and held > 2048 and not a.force:
        sys.exit(f"benchy: {held / 1024:.1f} GB of VRAM is in use - stop the served model first (or --force)")
    outdir = Path(a.out) if a.out else REPO / "sycl" / "benchy-results" / f"{VERSION}-{time.strftime('%Y%m%d-%H%M')}"
    outdir.mkdir(parents=True, exist_ok=True)
    first = json.loads(cfgs[0].read_text())["args"]
    dev = disk_of(to_host(first[first.index("--native") + 1]) if "--native" in first else ROOT)
    m = machine(dev, cold, root)
    (outdir / "machine.json").write_text(json.dumps(m, indent=1))
    rows = []
    with open(outdir / "matrix.jsonl", "a") as jl:
        for cp in cfgs:
            cfg = json.loads(cp.read_text())
            name = cp.stem[len("strata-"):] if cp.stem.startswith("strata-") else cp.stem
            (outdir / f"config-{name}.txt").write_text(" ".join(shlex.quote(x) for x in cfg["args"]) + "\n")
            for n in map(int, a.sizes.split(",")):
                r = run_one(name, cfg, n, outdir, cold, root, dev, timeout=300 + n * 0.02)
                rows.append(r); jl.write(json.dumps(r) + "\n"); jl.flush()
                print(json.dumps(r), flush=True)
    md = table(rows, m)
    (outdir / "matrix.md").write_text(md)
    print("\n" + md + f"\nwritten to {outdir}: post matrix.md and the config-*.txt files")


if __name__ == "__main__":
    main()
