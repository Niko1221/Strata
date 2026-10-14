#!/usr/bin/env python3
"""Compare the replies of two bench_concurrency.py runs uid by uid (the production-path exactness check).

bench_concurrency.py writes one line per level into <out>.requests.jsonl with every request's reply sha256.  The prompt
depends only on the uid and the prompt kind, so two arms that ran the same levels/rounds/prompt kind sent the same
requests; with --temperature 0 an exact engine change must give identical replies for every uid.  Meaningless with
sampling (temperature > 0): then only the length columns are comparable.

  python3 compare_replies.py results/sweep.requests.jsonl ab-x-T16-142fcb-short ab-x-T16-142fcd-short
  python3 compare_replies.py FILE LABEL_A LABEL_B --users 16 --show 5

If a label occurs several times in the file (re-run with the same label), the last line per level wins.
Exit code: 0 all common uids identical, 1 some differ, 2 nothing to compare.
"""
import argparse
import json
import sys


def load(path, label, users):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("label") != label or (users and row.get("users") != users):
                continue
            fields = row["fields"]
            for rec in row["reqs"]:
                r = dict(zip(fields, rec))
                out[r["uid"]] = r
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("file", help="the .requests.jsonl sidecar (e.g. results/sweep.requests.jsonl)")
    ap.add_argument("label_a")
    ap.add_argument("label_b")
    ap.add_argument("--file-b", default="", help="read label B from another sidecar file")
    ap.add_argument("--users", type=int, default=0, help="only this concurrency level (default: all levels)")
    ap.add_argument("--show", type=int, default=10, help="list up to N differing uids")
    a = ap.parse_args()
    ra = load(a.file, a.label_a, a.users)
    rb = load(a.file_b or a.file, a.label_b, a.users)
    if not ra or not rb:
        print(f"no requests for {'A' if not ra else 'B'} ({a.label_a if not ra else a.label_b}) in "
              f"{a.file if not ra else (a.file_b or a.file)}")
        return 2
    common = sorted(set(ra) & set(rb))
    only_a, only_b = len(set(ra) - set(rb)), len(set(rb) - set(ra))
    same, diff, errs = [], [], []
    for u in common:
        x, y = ra[u], rb[u]
        if x.get("err") or y.get("err"):
            errs.append(u)
        elif x["sha256"] == y["sha256"]:
            same.append(u)
        else:
            diff.append(u)
    mt = sum(1 for u in common if ra[u]["max_tok"] != rb[u]["max_tok"])
    print(f"{a.label_a} vs {a.label_b}: {len(common)} common uids - identical {len(same)}, different {len(diff)}"
          + (f", errored {len(errs)}" if errs else "")
          + (f"; only in A {only_a}, only in B {only_b}" if only_a or only_b else ""))
    if mt:
        print(f"WARNING: max_tokens differs for {mt} common uids (different --max-tok/--max-tok-min/--rounds?)")
    for u in diff[:a.show]:
        x, y = ra[u], rb[u]
        print(f"  uid {u}: tokens {x['gen']} vs {y['gen']} (max {x['max_tok']}), "
              f"sha {x['sha256'][:12]} vs {y['sha256'][:12]}")
    if not common:
        return 2
    return 1 if diff else 0


if __name__ == "__main__":
    sys.exit(main())
