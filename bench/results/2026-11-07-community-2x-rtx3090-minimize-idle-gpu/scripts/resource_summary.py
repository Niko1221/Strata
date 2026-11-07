#!/usr/bin/env python3
"""Join a load-capture CSV onto a sweep's JSONL: what the box was doing while each arm ran.

Run (from the folder above; `sampler.py` writes the CSV, published gzipped - unpack it first):
    zcat raw/l28-350w.csv.gz > /tmp/l28.csv
    python3 scripts/resource_summary.py --csv /tmp/l28.csv \
        --jsonl raw/p0-ist-350w.jsonl raw/p6-final-350w.jsonl --md
    ... --idle            # only the phases between the requests (no JSONL needed)
    ... --span T0 T1      # only the resource rows of a time window (node clock, epochs)

Every request in the JSONL carries `t_start`/`t_end` on the clock of the machine that served it (the sweep
driver probed that offset before measuring), `sampler.py` writes one row per second on the same clock - so
a request is a window of CSV rows and an arm is a set of windows. Empty windows (a request shorter than
the sample interval, or the sampler down) are counted in `rows` and reported, not hidden. The per-card
utilisation and wattage of § 1.1, and the 131 k-prompt row of § 5.1, are what this prints.

Columns (median over the request, then median over the arm's requests unless `--mean`):
    util0/1 %   the two cards' GPU utilisation;  utilΣ = util0 + util1 (how much of the PAIR works -
                a serial layer split leaves this near 100, the pipeline is what lifts it)
    W0/W1       power.draw per card - the 350 W question of this measurement
    SM0/1 MHz   SM clock;  M-MHz the memory clock (the +1000 MHz offset only shows up under load)
    °C          edge temperature of the hotter card;  thr = share of samples with a throttle bit set
    pci MB/s    rxpci + txpci (nvidia-smi dmon) - the expert streaming over PCIe, both cards summed
    md0 MB/s    read throughput of the RAID0 that holds the model (its four members are in the CSV too)
    cpu %       whole-pool busy (48 threads), eng % = the strata engine alone
    free GiB    MemAvailable

`--md` prints the same rows as a markdown table, `--csv` as machine-readable lines.
"""
import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict

THROTTLE = {0x01: "idle", 0x02: "app-clocks", 0x04: "sm-clocks", 0x08: "sync-boost",
            0x10: "sw-thermal", 0x20: "hw-thermal", 0x40: "hw-power-brake", 0x80: "sw-power-cap"}
FIELDS = ["t0", "iso"]


def f(x, d=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def load_csv(path):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                r["epoch"] = f(r["epoch"])
            except KeyError:
                continue
            rows.append(r)
    rows.sort(key=lambda r: r["epoch"])
    return rows


def decode_throttle(txt):
    try:
        bits = int(txt, 16)
    except (TypeError, ValueError):
        return set()
    return {name for bit, name in THROTTLE.items() if bits & bit}


def window(rows, t0, t1):
    lo, hi = 0, len(rows)
    while lo < hi:                                     # rows are sorted: binary bounds keep 8 h cheap
        mid = (lo + hi) // 2
        if rows[mid]["epoch"] < t0:
            lo = mid + 1
        else:
            hi = mid
    out = []
    for r in rows[lo:]:
        if r["epoch"] > t1:
            break
        out.append(r)
    return out


def agg(rows, key):
    xs = [f(r.get(key)) for r in rows if r.get(key) not in (None, "")]
    return statistics.median(xs) if xs else 0.0


def res_stats(rows):
    """One dict of readable resource numbers from a run of CSV rows."""
    if not rows:
        return {}
    u0, u1 = agg(rows, "gpu0_util"), agg(rows, "gpu1_util")
    thr = sum(1 for r in rows for b in (decode_throttle(r.get("gpu0_throttle")) |
                                        decode_throttle(r.get("gpu1_throttle")))
              if b not in ("idle",))
    stalling = sum(1 for r in rows if decode_throttle(r.get("gpu0_throttle")) &
                   {"sw-thermal", "hw-thermal", "hw-power-brake", "sw-power-cap"})
    return dict(rows=len(rows),
                util0=u0, util1=u1, utilsum=u0 + u1,
                w0=agg(rows, "gpu0_power_w"), w1=agg(rows, "gpu1_power_w"),
                sm0=agg(rows, "gpu0_sm_mhz"), sm1=agg(rows, "gpu1_sm_mhz"),
                mem=agg(rows, "gpu0_mem_mhz"), mem1=agg(rows, "gpu1_mem_mhz"),
                temp=max(agg(rows, "gpu0_temp_c"), agg(rows, "gpu1_temp_c")),
                fan=max(agg(rows, "gpu0_fan_pct"), agg(rows, "gpu1_fan_pct")),
                thr_pct=100.0 * thr / len(rows), stall_pct=100.0 * stalling / len(rows),
                pci=(agg(rows, "gpu0_rxpci_mbs") + agg(rows, "gpu1_rxpci_mbs") +
                     agg(rows, "gpu0_txpci_mbs") + agg(rows, "gpu1_txpci_mbs")),
                md0_r=agg(rows, "md0_r_mbs"), md0_w=agg(rows, "md0_w_mbs"),
                root_w=agg(rows, "nvme2n1p2_w_mbs"),
                cpu=agg(rows, "cpu_busy_pct"), load=agg(rows, "load1"),
                eng=agg(rows, "eng_cpu_pct"), srv=agg(rows, "srv_cpu_pct"),
                free=agg(rows, "ram_available_gib"))


COLS = [("tok_s", "tok/s", 1), ("read_tok_s", "read", 0), ("acc_pct", "acc%", 1),
        ("util0", "u0%", 0), ("util1", "u1%", 0), ("utilsum", "uΣ%", 0),
        ("w0", "W0", 0), ("w1", "W1", 0), ("sm0", "SM0", 0), ("mem", "M-MHz", 0),
        ("temp", "°C", 0), ("fan", "fan%", 0), ("stall_pct", "stall%", 0),
        ("pci", "pci MB/s", 0), ("md0_r", "md0 MB/s", 1), ("cpu", "cpu%", 0),
        ("eng", "eng%", 0), ("free", "free GiB", 1), ("rows", "rows", 0)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="the monitor CSV (fetched by strata_l28_monitor.py)")
    ap.add_argument("--jsonl", nargs="*", default=[], help="sweep results to attribute the rows to")
    ap.add_argument("--min-rows", type=int, default=1, help="drop records with fewer resource rows")
    ap.add_argument("--mean", action="store_true", help="mean over requests instead of median")
    ap.add_argument("--md", action="store_true", help="markdown table")
    ap.add_argument("--csv-out", default="", help="also write the joined table here")
    ap.add_argument("--idle", action="store_true", help="summarise the CSV outside every request window")
    ap.add_argument("--span", nargs=2, type=float, default=None, metavar=("T0", "T1"))
    a = ap.parse_args()

    rows = load_csv(a.csv)
    if not rows:
        raise SystemExit(f"{a.csv} holds no rows")
    print(f"# {a.csv}: {len(rows)} rows, "
          f"{rows[0]['iso']} .. {rows[-1]['iso']}", file=sys.stderr)

    if a.span:
        w = window(rows, a.span[0], a.span[1])
        print(json.dumps(res_stats(w), indent=2))
        return

    if a.idle or not a.jsonl:
        busy = set()
        for path in a.jsonl:
            for line in open(path):
                r = json.loads(line)
                for t in range(int(f(r.get("t_start"))), int(f(r.get("t_end"))) + 1):
                    busy.add(t)
        sel = [r for r in rows if (not a.idle) or int(r["epoch"]) not in busy]
        out = res_stats(sel)
        out["label"] = ("idle-between-requests" if a.idle else "whole-capture")
        print(json.dumps(out, indent=2))
        return

    per = defaultdict(list)                            # (group, arm, kind) -> resource dicts
    seen, hit = 0, 0
    for path in a.jsonl:
        for line in open(path):
            r = json.loads(line)
            if r.get("failed") or not r.get("t_start"):
                continue
            seen += 1
            w = window(rows, r["t_start"], r["t_end"])
            if len(w) < a.min_rows:
                continue
            hit += 1
            st = res_stats(w)
            for k in ("tok_s", "read_tok_s", "acc_pct", "windows", "ms_window", "tok_window"):
                st[k] = f(r.get(k))
            st["n"] = 1
            per[(r.get("group", "?"), r.get("arm", "?"), r.get("kind", "?"))].append(st)

    def comb(xs, key):
        vals = [x[key] for x in xs if isinstance(x.get(key), (int, float))]
        if not vals:
            return 0.0
        return (sum(vals) / len(vals)) if a.mean else statistics.median(vals)

    hdr = ["group", "arm", "kind", "n"] + [h for _, h, _ in COLS[3:]]
    lines = []
    for key in sorted(per):
        xs = per[key]
        row = {"group": key[0], "arm": key[1], "kind": key[2], "n": len(xs)}
        for k, _, _ in COLS:
            row[k] = comb(xs, k)
        lines.append(row)
    lines.sort(key=lambda r: -r["tok_s"])
    print(f"# {hit}/{seen} records fell inside the capture", file=sys.stderr)

    if a.md:
        print("| " + " | ".join(hdr) + " |")
        print("|" + "|".join(["---"] * len(hdr)) + "|")
        for r in lines:
            cells = [r["group"], r["arm"], r["kind"], str(r["n"])]
            for k, _, dec in COLS[3:]:
                cells.append(f"{r[k]:.{dec}f}")
            print("| " + " | ".join(cells) + " |")
    else:
        print("\t".join(hdr))
        for r in lines:
            cells = [r["group"], r["arm"], r["kind"], str(r["n"])]
            for k, _, dec in COLS[3:]:
                cells.append(f"{r[k]:.{dec}f}")
            print("\t".join(cells))
    if not lines:
        raise SystemExit("no record fell inside the CSV's time range - fetch the monitor first "
                         f"(CSV ends at {rows[-1]['iso']})")
    if a.csv_out:
        with open(a.csv_out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(lines[0].keys()))
            w.writeheader()
            w.writerows(lines)
        print(f"# wrote {a.csv_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
