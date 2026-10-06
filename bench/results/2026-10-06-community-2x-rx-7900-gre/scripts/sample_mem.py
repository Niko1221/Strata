#!/usr/bin/env python3
"""Sample VRAM and host RAM into a CSV until told to stop.

Reports what the machine used while Strata served requests: VRAM from the amdgpu
sysfs counters of every card (column named with the card's PCI slot), host RAM as
MemTotal - MemAvailable, and optionally one process's RSS. Both memory figures are
whole-machine numbers, not the engine's own accounting.

Copy of bench/quants/sample_mem.py with --interval (a longer step for a soak) and
the engine RSS column added.
"""
import argparse
import csv
import os
import time
from pathlib import Path


def cards():
    return sorted(Path('/sys/class/drm').glob('card*/device/mem_info_vram_total'))


def slot(dev):
    try:
        for line in (dev / 'uevent').read_text().splitlines():
            if line.startswith('PCI_SLOT_NAME='):
                return line.split('=', 1)[1]
    except OSError:
        pass
    return dev.name


def read(path):
    try:
        return int(path.read_text())
    except OSError:
        return -1


def meminfo():
    total = avail = 0
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemTotal:'):
            total = int(line.split()[1])
        elif line.startswith('MemAvailable:'):
            avail = int(line.split()[1])
    return total, avail


def engine_pids():
    pids = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            if b'engine/strata' in (p / 'cmdline').read_bytes():
                pids.append(int(p.name))
        except OSError:
            continue
    return pids


def rss_mib(pid):
    try:
        for line in (Path('/proc') / str(pid) / 'status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        pass
    return -1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=3600)
    p.add_argument('--interval', type=float, default=1.0, help='seconds between samples')
    p.add_argument('--engine-rss', action='store_true', help='add an engine_rss_mib column')
    p.add_argument('--stop-file', type=Path)
    args = p.parse_args()
    paths = cards()
    names = [slot(c.parent) for c in paths]
    total, _ = meminfo()
    end = time.monotonic() + args.seconds
    header = ['unix_s'] + [f'{n}_vram_used' for n in names] + ['ram_used_mib']
    if args.engine_rss:
        header += ['engine_rss_mib']
    with args.out.open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        while time.monotonic() < end:
            if args.stop_file and args.stop_file.exists():
                break
            used = total - meminfo()[1]
            row = [round(time.time(), 2)] + [read(c.with_name('mem_info_vram_used')) for c in paths] + [round(used / 1024, 1)]
            if args.engine_rss:
                row += [max((rss_mib(p) for p in engine_pids()), default=-1)]
            w.writerow(row)
            fh.flush()
            time.sleep(args.interval)


if __name__ == '__main__':
    os.nice(10)
    main()
