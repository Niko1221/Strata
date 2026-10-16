"""Drop the model files from the OS page cache without sudo (posix_fadvise DONTNEED), as the E5-2673 v3 community
report did before every engine start. Only pages no running process maps are dropped, so run it with Strata stopped.

    python3 drop_cache.py FILE_LIST     (one path per line, relative to <WORKDIR> or absolute)
Prints the page-cache size (Cached in /proc/meminfo) before and after.
"""
import os
import sys

TOP = "<WORKDIR>"


def cached_gib() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("Cached:"):
            return int(line.split()[1]) / 2**20
    return float("nan")


before = cached_gib()
n = 0
for line in open(sys.argv[1]):
    p = line.strip()
    if not p:
        continue
    p = p.split("  ", 1)[-1]          # also accepts sha256sum lines
    p = p if os.path.isabs(p) else os.path.join(TOP, p)
    try:
        fd = os.open(p, os.O_RDONLY)
    except OSError as e:
        print("skip", p, e)
        continue
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        n += 1
    finally:
        os.close(fd)
print(f"page cache dropped for {n} files: Cached {before:.1f} GiB -> {cached_gib():.1f} GiB")
