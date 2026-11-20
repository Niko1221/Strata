#!/usr/bin/env python3
"""Index page for docs/bench-results/:  docs/bench-results/README.md

    python tools/bench_results_index.py            write the index
    python tools/bench_results_index.py --check    fail when the index is out of date (a new result folder must be listed)

One row per result folder: its date (from the folder name), its own title (the first heading of its README or of its first
Markdown file), and how many files it holds.  The results themselves stay where they are.  Standard library only.
"""
import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import doc_pages  # noqa: E402

ROOT = doc_pages.ROOT
RESULTS = os.path.join(ROOT, "docs", "bench-results")
OUT = os.path.join(RESULTS, "README.md")


def published_files():
    """Index committed/staged results; private ignored runs must not change CI docs.

    An exported source archive has no Git index, so all of its files are public.
    """
    if os.path.exists(os.path.join(ROOT, ".git")):
        result = subprocess.run(["git", "-C", ROOT, "ls-files", "-z", "--", "bench/results"],
                                capture_output=True, check=True)
        prefix = "bench/results/"
        return {name[len(prefix):] for name in result.stdout.decode("utf-8").split("\0")
                if name.startswith(prefix) and os.path.isfile(os.path.join(ROOT, name))}
    return {os.path.relpath(os.path.join(base, name), RESULTS).replace(os.sep, "/")
            for base, _, names in os.walk(RESULTS) for name in names}


def title_of(folder, published=None):
    names = sorted(n for n in os.listdir(folder) if n.lower().endswith(".md")
                   and (published is None or os.path.relpath(os.path.join(folder, n), RESULTS).replace(os.sep, "/") in published))
    names.sort(key=lambda n: (n.lower() != "readme.md", n))
    for name in names:
        try:
            with open(os.path.join(folder, name), encoding="utf-8", errors="replace") as f:
                for line in f:
                    m = re.match(r"^#{1,3}\s+(.*\S)", line)
                    if m:
                        t = re.sub(r"[`*]", "", m.group(1)).replace("|", "/")
                        return t if len(t) <= 150 else t[:147].rstrip() + "..."
        except OSError:
            pass
    return "(no Markdown page)"


def count_files(folder, published):
    prefix = os.path.relpath(folder, RESULTS).replace(os.sep, "/") + "/"
    return sum(name.startswith(prefix) for name in published)


def build():
    published = published_files() - {"README.md"}
    folders = sorted({name.split("/", 1)[0] for name in published if "/" in name})
    loose = sorted(name for name in published if "/" not in name)
    total = len(published)
    lines = ["# Benchmark results", "",
             "Raw results and short reports from measurement runs, one folder per run, named by date. They stay here as they were written: each states",
             "the Strata version, hardware and method of its own day, and none is a statement about a later release. Current numbers are in the",
             "[docs](../README.md) and the [README](../../README.md); how to measure is in [BENCHMARK_PROTOCOL.md](../BENCHMARK_PROTOCOL.md).",
             "", "This page is built by `python tools/bench_results_index.py` (`--check` fails when a folder is missing from it). "
             "%d folders and %d loose files, %d files in all." % (len(folders), len(loose), total), ""]
    month = None
    for f in folders:
        m = re.match(r"^(\d{4}-\d{2})-(\d{2})", f)
        key = m.group(1) if m else "undated"
        if key != month:
            month = key
            lines += ["", "## " + key, "", "| Folder | What | Files |", "| --- | --- | ---: |"]
        lines.append("| [%s](%s/) | %s | %d |" % (f, f, title_of(os.path.join(RESULTS, f), published), count_files(os.path.join(RESULTS, f), published)))
    if loose:
        lines += ["", "## Loose files", "", "| File |", "| --- |"] + ["| [%s](%s) |" % (n, n) for n in loose]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    want = build()
    if args.check:
        have = open(OUT, encoding="utf-8", newline="").read().replace("\r\n", "\n") if os.path.exists(OUT) else ""
        if have != want:
            print("FAIL docs/bench-results/README.md is out of date (run python tools/bench_results_index.py)")
            return 1
        print("bench results index: ok")
        return 0
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(want)
    print("wrote %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
