#!/usr/bin/env python3
"""GPU telemetry sampler for Intel Arc cards on the xe driver (runs as root: sycl/tools/gpustat.service).
Writes /run/gpustat.json every INTERVAL s; Strata's web app (serve/telemetry.py) shows it in the Monitor tab:
  name, vram_used_mb / vram_total_mb (sum of drm-resident-vram0 over unique drm
  clients from /proc/*/fdinfo - the only VRAM accounting xe exposes; root-only),
  busy_pct (compute/render cycle deltas), temp_pkg/temp_vram (hwmon 'xe'),
  power_w / pkg_power_w (energy counter deltas over a POWER_WINDOW s window: the counters update in
  bursts, so a 3 s delta swings 0.6 <-> 48 W), power_cap_w, fan_rpm, freq_mhz, the PCIe link the card
  trained at (the first upstream port wider than x1 - the card's own functions sit behind an internal
  x1 switch), the hottest VRAM channel, and the host's 1-minute load. Device totals only.
Without it the Monitor tab still shows temperature, power and the PCIe link (sysfs), but not load or VRAM: xe
reports those per client in /proc/*/fdinfo, which only root can read.
Install:  sudo install -m 755 sycl/tools/gpustat.py /usr/local/sbin/
          sudo install -m 644 sycl/tools/gpustat.service /etc/systemd/system/ && sudo systemctl enable --now gpustat
Env: GPUSTAT_INTERVAL (s), GPUSTAT_POWER_WINDOW (s), GPUSTAT_PCI (sample only this card, e.g. 0000:03:00.0;
default every Intel GPU on xe), GPUSTAT_VRAM_MB (the total; default the card's largest PCI BAR, which maps
all of VRAM with resizable BAR - a little above what the runtime reports as usable; when more than one card
is sampled it applies to each of them), GPUSTAT_OUT (the JSON path; default /run/gpustat.json).

Every Intel xe card is sampled. The document holds one entry per card under "cards", keyed by PCI address,
and the first card's fields are also written at the top level so a reader written before this still works.
"""
import collections, glob, json, os, shlex, subprocess, time

OUT = os.environ.get("GPUSTAT_OUT") or "/run/gpustat.json"
INTERVAL = float(os.environ.get("GPUSTAT_INTERVAL", "3"))   # each sample scans /proc/*/fd; keep it modest
POWER_WINDOW = float(os.environ.get("GPUSTAT_POWER_WINDOW", "15"))
NAMES = {"e223": "Intel(R) Arc(TM) Pro B70 Graphics"}


def find_cards():
    """Every Intel GPU on the xe driver (it has tile*/gt*), in sysfs order, as PCI addresses."""
    cards = []
    for d in sorted(glob.glob("/sys/class/drm/card[0-9]*/device")):
        try:
            if open(f"{d}/vendor").read().strip() == "0x8086" and glob.glob(f"{d}/tile*/gt*"):
                pci = os.path.basename(os.path.realpath(d))
                if pci not in cards:
                    cards.append(pci)
        except OSError:
            continue
    return cards


PCIS = ([os.environ["GPUSTAT_PCI"]] if os.environ.get("GPUSTAT_PCI")
        else find_cards() or ["0000:00:00.0"])


def vram_total_mb(pci):
    if os.environ.get("GPUSTAT_VRAM_MB"):
        return int(os.environ["GPUSTAT_VRAM_MB"])
    try:                                   # with resizable BAR, BAR2 maps all of VRAM
        # The file lists the card's BARs and its ROM at indices 0-6, then the bridge windows
        # it sits behind at 7-12.  A bridge window is sized for everything behind it and can
        # be larger than the card, so a max() over every line reports the bridge, not the card.
        bar = [l.split() for l in open(f"/sys/bus/pci/devices/{pci}/resource") if l.strip()][:7]
        sizes = [int(e, 16) - int(s, 16) + 1 for s, e, _ in bar]
        return max(sizes) // 2**20
    except (OSError, ValueError):
        return 0


def card_name(dev_id, pci):
    if dev_id in NAMES:
        return NAMES[dev_id]
    try:
        # lspci -mm quotes each field: slot, class, vendor, device, then -r/-p and the
        # subsystem.  Splitting on '" "' mangles the device name and swallows the fields
        # after it, so the fields are tokenised properly instead.
        out = subprocess.run(["lspci", "-mm", "-s", pci], capture_output=True, text=True, timeout=5).stdout
        parts = shlex.split(out)
        if len(parts) > 3 and parts[3]:
            return parts[3]
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return f"Intel GPU 8086:{dev_id}"


GEN = {"2.5": 1, "5.0": 2, "8.0": 3, "16.0": 4, "32.0": 5, "64.0": 6}


def rd(p, default=None):
    try:
        return open(p).read().strip()
    except Exception:
        return default


def pcie_link(pci):
    """The link the card trained at: walk up from the GPU function to the first port wider than x1."""
    d = os.path.realpath(f"/sys/bus/pci/devices/{pci}")
    chain = []
    while d.startswith("/sys/devices/pci") and os.path.exists(f"{d}/current_link_speed"):
        chain.append(d); d = os.path.dirname(d)
    def link(p, kind):
        sp = (rd(f"{p}/{kind}_link_speed", "") or "").split(" ")[0]
        return {"gen": GEN.get(sp), "gts": sp, "width": int(rd(f"{p}/{kind}_link_width", "0") or 0)}
    for i, p in enumerate(chain):
        cur = link(p, "current")
        if cur["width"] > 1:
            root = chain[-1]                                  # the CPU's root port: what the slot can do
            return {"cur": cur, "card_max": link(p, "max"), "slot_max": link(root, "max")}
    return None


def hwmon_dir(pci):
    """The 'xe' hwmon directory belonging to THIS card.  There is one per card, so the device
    symlink is matched against the card's sysfs path instead of taking the first 'xe' found."""
    want = os.path.realpath(f"/sys/bus/pci/devices/{pci}")
    for n in sorted(glob.glob("/sys/class/hwmon/hwmon*/name")):
        if (rd(n) or "") != "xe":
            continue
        d = os.path.dirname(n)
        if os.path.realpath(f"{d}/device") == want:
            return d
    return None


def hw_temps(h):
    t = {}
    for lab in glob.glob(f"{h}/temp*_label"):
        name = rd(lab); v = rd(lab.replace("_label", "_input"))
        if name and v: t[name] = int(v) / 1000
    return t


def clients_all():
    """{pci: {client_id: {pid, comm, vram_kb, cycles, total}}} in ONE pass over /proc, plus the key
    None for clients that name no card (an older kernel does not report drm-pdev).  One pass matters:
    scanning per card ran the whole of /proc once for every card fitted."""
    groups = {}
    for pdir in glob.glob("/proc/[0-9]*"):
        try:
            fds = os.listdir(f"{pdir}/fd")
        except Exception:
            continue
        for fd in fds:
            try:
                if not os.readlink(f"{pdir}/fd/{fd}").startswith("/dev/dri/"):
                    continue
                info = dict(l.split(":\t", 1) for l in open(f"{pdir}/fdinfo/{fd}").read().splitlines() if l.startswith("drm-"))
            except Exception:
                continue
            cid = info.get("drm-client-id")
            group = groups.setdefault(info.get("drm-pdev"), {})
            if not cid or cid in group:
                continue
            kb = lambda k: int(info.get(k, "0").split()[0] or 0)
            group[cid] = {"pid": int(pdir[6:]), "comm": rd(f"{pdir}/comm", "?"),
                          "cg": rd(f"{pdir}/cgroup", ""), "vram_kb": kb("drm-resident-vram0"),
                          "cycles": max(kb("drm-cycles-rcs"), kb("drm-cycles-ccs")),
                          "total": max(kb("drm-total-cycles-rcs"), kb("drm-total-cycles-ccs"))}
    return groups


def clients(pci, groups=None):
    """The clients of one card: those that name it, plus those that name no card.  A client that
    names no card cannot be attributed to one, so it counts for the first card sampled and is not
    counted again on the others.  Pass groups=clients_all() to reuse one scan for several cards."""
    groups = clients_all() if groups is None else groups
    got = dict(groups.get(pci) or {})
    if pci == PCIS[0]:
        got.update(groups.get(None) or {})
    return got


def card_state(pci):
    """The values and accumulators that belong to one card."""
    dev_id = (rd(f"/sys/bus/pci/devices/{pci}/device", "0x0") or "0x0")[2:]
    return {"hwmon": hwmon_dir(pci), "name": card_name(dev_id, pci), "total_mb": vram_total_mb(pci),
            "prev": {}, "ring": collections.deque(), "link": None, "link_t": 0.0}


def sample(pci, st, now, cl):
    """One sample of one card, in the flat shape the readers already understand."""
    h = st["hwmon"]
    busy = 0.0
    for cid, c in cl.items():
        p = st["prev"].get(cid)
        if p and c["total"] > p["total"]:
            busy += (c["cycles"] - p["cycles"]) / (c["total"] - p["total"]) * 100
    e = int(rd(f"{h}/energy1_input", "0") or 0) if h else 0
    e2 = int(rd(f"{h}/energy2_input", "0") or 0) if h else 0
    ring = st["ring"]
    ring.append((now, e, e2))
    while len(ring) > 2 and now - ring[1][0] >= POWER_WINDOW:
        ring.popleft()
    t0, ea, eb = ring[0]
    span = now - t0
    power = (e - ea) / 1e6 / span if span >= 2.5 and e >= ea else None
    pkg_power = (e2 - eb) / 1e6 / span if span >= 2.5 and e2 >= eb else None
    if now - st["link_t"] > 30:                   # the link can retrain (power saving, errors): re-read it
        st["link"] = pcie_link(pci); st["link_t"] = now
    temps = hw_temps(h) if h else {}
    vram_ch = [v for k, v in temps.items() if k.startswith("vram_ch_")]
    cap = rd(f"{h}/power1_cap") if h else None
    used_mb = sum(c["vram_kb"] for c in cl.values()) // 1024
    freq = glob.glob(f"/sys/bus/pci/devices/{pci}/tile0/gt0/freq0/act_freq")
    out = {
        "name": st["name"], "pci": pci, "ts": now,
        "vram_used_mb": used_mb, "vram_total_mb": st["total_mb"],
        "busy_pct": round(min(busy, 100), 1),
        "temp_pkg": temps.get("pkg"), "temp_vram": temps.get("vram"),
        "power_w": None if power is None else round(power, 1),
        "pkg_power_w": None if pkg_power is None else round(pkg_power, 1),
        "power_cap_w": round(int(cap) / 1e6) if cap and cap.isdigit() and int(cap) > 0 else None,
        "temp_vram_max": max(vram_ch) if vram_ch else None,
        "temp_pcie": temps.get("pcie"), "temp_mctrl": temps.get("mctrl"),
        "pcie": st["link"],
        "host_load1": round(os.getloadavg()[0], 2), "host_cpus": os.cpu_count(),
        "energy_j": round(e / 1e6, 1),           # cumulative card energy (hwmon energy1 = "card", uJ)
        "fan_rpm": int(rd(f"{h}/fan1_input", "0") or 0) if h else None,
        "freq_mhz": int(rd(freq[0], "0") or 0) if freq else None,
    }
    st["prev"] = cl
    return out


def collect(states, now=None):
    """One pass over every card.  /proc is scanned once for all of them, not once per card, and the
    client set is passed in.  Split out from main() so it can be exercised without the loop."""
    now = time.time() if now is None else now
    groups = clients_all()
    return {pci: sample(pci, states[pci], now, clients(pci, groups)) for pci in PCIS}


def document(cards):
    """The JSON that gets written: every card under "cards" keyed by PCI address, plus the first
    card's fields at the top level so a reader written before this still finds what it expects."""
    first = cards[PCIS[0]]
    doc = {"ts": first.get("ts"), "cards": cards}
    doc.update(first)
    return doc


def main():
    states = {pci: card_state(pci) for pci in PCIS}
    while True:
        cards = collect(states)
        out = document(cards)
        tmp = OUT + ".tmp"
        with open(tmp, "w") as f: json.dump(out, f)
        os.chmod(tmp, 0o644); os.replace(tmp, OUT)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
