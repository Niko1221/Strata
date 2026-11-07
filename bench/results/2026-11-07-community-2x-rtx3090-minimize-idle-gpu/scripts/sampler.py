#!/usr/bin/env python3
"""One row per --interval into a CSV on the machine being measured: the 1 Hz load capture of § 1.1 / § 5.1.

Run (on the box whose GPUs are measured - stdlib only, read-only, it never sets a clock or a limit):
    python3 sampler.py --out l28-1hz.csv --interval 1                 # stop with ^C
    python3 sampler.py --out l28-1hz.csv --interval 5 --limit 720     # 1 h of 5 s rows then stop

Columns are named once, in the header line the script writes: epoch and ISO stamp, then per card the
utilisation, VRAM, power.draw, edge temperature, SM/memory clocks, fan, PCIe link gen/width and the
`clocks_event_reasons` bit field, plus rx/tx MB/s from a standing `nvidia-smi dmon -s t` reader (a fresh
dmon call per row would restart its counters), then /proc deltas for the disks listed in `DISKS` at the
top — on this box a RAID0 (`md0`), its four strips and the root partition, which is what to change for
another machine. Also the CPU pool, the engine's own CPU %/RSS (found by its command line, `find_pids`)
and MemAvailable. The rows are the input of `resource_summary.py`, which joins them onto a sweep's JSONL
by timestamp.

The working copy of this script is started over ssh with `setsid nohup` by a driver that also fetches it
back, so it survives pod restarts and a dropped connection; that plumbing is not part of a measurement
and is not published here.
"""
import argparse, os, re, subprocess, sys, threading, time

DISKS = ["md0", "nvme0n1", "nvme1n1", "nvme2n1p2", "nvme3n1", "nvme4n1"]
GPU_Q = ("utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem,"
         "fan.speed,pcie.link.gen.current,pcie.link.width.current,clocks_event_reasons.active")
# 11 fields + the throttle hex makes 11 columns per card; keep in step with HEADER below.
CPU_KEYS = ("user", "nice", "system", "idle", "io_wait", "irq", "softirq", "steal", "guest", "guest_nice")
MEM_KEYS = ("MemAvailable", "Dirty", "Writeback", "SwapTotal", "SwapFree", "Shmem", "Committed_AS")


def gpu_rows():
    """nvidia-smi as one call: [[field, ...], ...] in card order, [] when the driver does not answer."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=" + GPU_Q, "--format=csv,noheader,nounits"],
                             capture_output=True, timeout=20).stdout.decode("utf8", "replace")
    except Exception:
        return []
    rows = []
    for line in out.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) >= 11:
            rows.append(f)
    return rows


class Pcie(threading.Thread):
    """Standing `nvidia-smi dmon -s t` reader: rxpci/txpci MB/s per card from the newest complete row."""

    def __init__(self):
        super().__init__(daemon=True)
        self.vals, self.n = {}, 0

    def run(self):
        while True:
            try:
                p = subprocess.Popen(["nvidia-smi", "dmon", "-s", "t", "-d", "1"],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            except Exception:
                return
            cur = {}
            for line in p.stdout:
                if line.startswith("#") or not line.strip():
                    continue
                f = line.split()
                if len(f) < 3:
                    continue
                try:
                    idx, rx, tx = int(f[0]), float(f[1]), float(f[2])
                except ValueError:
                    continue
                if idx == 0 and cur:                       # the card counter wrapped: one round done
                    self.vals = dict(cur)
                cur[idx] = (rx, tx)
                if idx == 0:
                    self.vals = dict(cur)                  # fresh round: publish rx/tx of card 0 at once
            time.sleep(2)                                  # dmon died (driver busy?) - start it again


def proc_stat():
    with open("/proc/stat") as fh:
        for line in fh:
            if line.startswith("cpu "):
                f = line.split()[1:]
                return {k: float(v) for k, v in zip(CPU_KEYS, f)}
    return {}


def disk_stats(want):
    out = {}
    with open("/proc/diskstats") as fh:
        for line in fh:
            f = line.split()
            if len(f) >= 14 and f[2] in want:
                out[f[2]] = (int(f[5]), int(f[9]), int(f[6]), int(f[10]), int(f[12]), int(f[13]), int(f[8]))
    return out


def mem_info():
    out = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            k = line.split(":")[0]
            if k in MEM_KEYS:
                out[k] = int(line.split()[1])
    return out


def find_pids():
    """(engine pids, server pids) - the strata process tree, by its command line."""
    eng, srv = [], []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open("/proc/%s/cmdline" % pid, "rb") as fh:
                cmd = fh.read().replace(b"\0", b" ").decode("utf8", "replace")
        except OSError:
            continue
        if "engine/strata" in cmd and "--serve" in cmd:
            eng.append(pid)
        elif "serve/server.py" in cmd:
            srv.append(pid)
    return eng, srv


def cpu_of(pids):
    ticks, rss = 0.0, 0
    for pid in pids:
        try:
            with open("/proc/%s/stat" % pid) as fh:
                f = fh.read().rsplit(")", 1)[1].split()
            ticks += float(f[11]) + float(f[12]) + float(f[13]) + float(f[14])
            rss += int(f[21]) * 4096
        except (OSError, IndexError, ValueError):
            pass
    return ticks, rss


HEADER = (["epoch", "iso"] +
          ["gpu%d_%s" % (i, s) for i in (0, 1) for s in
           ("util", "vram_mib", "vram_total", "power_w", "temp_c", "sm_mhz", "mem_mhz",
            "fan_pct", "pcie_gen", "pcie_width", "throttle")] +
          ["gpu%d_rxpci_mbs" % i for i in (0, 1)] + ["gpu%d_txpci_mbs" % i for i in (0, 1)] +
          ["cpu_busy_pct", "cpu_user_pct", "cpu_sys_pct", "cpu_iowait_pct", "cpu_irq_pct",
           "load1", "load5", "load15"] +
          ["eng_cpu_pct", "eng_rss_mib", "srv_cpu_pct", "srv_rss_mib"] +
          ["ram_available_gib", "ram_dirty_mib", "swap_used_gib", "shmem_gib"] +
          sum([["%s_r_mbs" % d, "%s_w_mbs" % d, "%s_riops" % d, "%s_wiops" % d,
                "%s_inflight" % d, "%s_util_pct" % d] for d in DISKS], []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0, help="stop after N rows (0 = forever)")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    new = not os.path.exists(a.out) or os.path.getsize(a.out) == 0
    fh = open(a.out, "a", buffering=1)
    if new:
        fh.write(",".join(HEADER) + "\n")
    pc = Pcie()
    pc.start()
    want = set(DISKS)
    hz = os.sysconf("SC_CLK_TCK")
    pids, p_clock = find_pids(), time.time()
    eng, srv = cpu_of(pids[0]), cpu_of(pids[1])
    prev = dict(t=time.time(), cpu=proc_stat(), disk=disk_stats(want), eng=eng[0], srv=srv[0])
    n = 0
    while True:
        time.sleep(a.interval)
        now = time.time()
        dts = now - prev["t"]
        ps, ds, mi = proc_stat(), disk_stats(want), mem_info()
        c0, c1 = prev["cpu"], ps
        dtot = sum(c1.values()) - sum(c0.values())
        if dtot <= 0 or dts <= 0.2:
            continue
        busy = 100.0 * (dtot - (c1["idle"] - c0["idle"]) - (c1["io_wait"] - c0["io_wait"])) / dtot
        parts = [100.0 * (c1[k] - c0[k]) / dtot for k in ("user", "system", "io_wait", "irq")]
        with open("/proc/loadavg") as fl:
            la = fl.read().split()
        gpus = gpu_rows()
        cols = [str(int(now)), time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now))]
        for i in (0, 1):
            cols += (gpus[i] if i < len(gpus) else [""] * 11)[:11]
        for i in (0, 1):
            cols.append("%.0f" % pc.vals[i][0] if i in pc.vals else "")
        for i in (0, 1):
            cols.append("%.0f" % pc.vals[i][1] if i in pc.vals else "")
        cols += ["%.1f" % busy] + ["%.1f" % x for x in parts] + la[:3]
        if now - p_clock > 20:                      # follow an engine restart, but never twice counted
            pids, p_clock = find_pids(), now
        eng, srv = cpu_of(pids[0]), cpu_of(pids[1])
        eg = max(0.0, eng[0] - prev["eng"]) if eng[0] >= prev["eng"] else eng[0]
        sg = max(0.0, srv[0] - prev["srv"]) if srv[0] >= prev["srv"] else srv[0]
        cols += ["%.0f" % (100.0 * eg / hz / dts), "%.0f" % (eng[1] / 1048576.0),
                 "%.0f" % (100.0 * sg / hz / dts), "%.0f" % (srv[1] / 1048576.0)]
        cols += ["%.1f" % (mi.get("MemAvailable", 0) / 1048576.0), "%.0f" % (mi.get("Dirty", 0) / 1024.0),
                 "%.1f" % ((mi.get("SwapTotal", 0) - mi.get("SwapFree", 0)) / 1048576.0),
                 "%.1f" % (mi.get("Shmem", 0) / 1048576.0)]
        d0 = prev["disk"]
        for d in DISKS:
            if d in ds and d in d0:
                r, w, rc, wc, infl, util, _ = ds[d]
                p_r, p_w, p_rc, p_wc, _, p_util, _ = d0[d]
                cols += ["%.2f" % ((r - p_r) * 512 / 1048576.0 / dts),
                         "%.2f" % ((w - p_w) * 512 / 1048576.0 / dts),
                         "%.1f" % ((rc - p_rc) / dts), "%.1f" % ((wc - p_wc) / dts),
                         str(infl), "%.1f" % ((util - p_util) * 10.0 / dts)]
            else:
                cols += ["", "", "", "", "", ""]
        fh.write(",".join(str(x) for x in cols) + "\n")
        prev = dict(t=now, cpu=ps, disk=ds, eng=eng[0], srv=srv[0])
        n += 1
        if a.limit and n >= a.limit:
            break
    fh.close()


main()
