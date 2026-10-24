"""Compare the configs 0.2.0-beta's setup wrote with the 0.1.40.3 report's, then build the five arms.

setup-written (configs-setup/A,B,D.json) vs the old run configs (the old orchestrator's full configs, with paths) and
the old report's configs/ (args only, placeholders).  Every difference is listed; then the arms are written to
configs-arms/: the setup config + the hand-added conversation cache (as in 0.1.40.3); C = B with --kv-resident 20480;
B-share-off = B with env STRATA_PREFILL_CPU_SHARE=0."""
import json
import pathlib
import re

W = pathlib.Path(__file__).resolve().parent
OLD_REPORT = pathlib.Path(r"<0.1.40.3-repo>\bench\results\2026-10-07-community-rtx3070ti-8gb\configs")
OLD_FULL = pathlib.Path(r"<scratch>")
CACHE = ["--conversation-cache-mib", "4096", "--conversation-cache-slots", "4"]


def ph(s):
    s = str(s).replace("<data-dir>", "<data-dir>")
    s = s.replace("<repo>", "<repo>").replace("<0.1.40.3-repo>", "<repo>")
    return s


def norm_args(a):
    return [ph(x) for x in a]


report = ["# Configs written by 0.2.0-beta's setup, compared with the 0.1.40.3 report", ""]
pairs = {"A": "1-A", "B": "2-B", "D": "4-D"}
for new, old in pairs.items():
    n = json.loads((W / "configs-setup" / f"{new}.json").read_text(encoding="utf-8"))
    o_args = json.loads((OLD_REPORT / f"{old}.json").read_text(encoding="utf-8"))["args"]
    o_full = json.loads((OLD_FULL / old / "config.json").read_text(encoding="utf-8"))
    na = norm_args(n["args"]) + CACHE                        # the arm as run: setup's args + the cache
    report.append(f"## {new} (setup) vs {old} (0.1.40.3 report)")
    if na == o_args:
        report.append("- args: identical once the hand-added conversation cache is appended (setup wrote the same "
                      "engine arguments as 0.1.40.3's setup)")
    else:
        import difflib
        report.append("- args differ: " + " | ".join(l for l in difflib.unified_diff(o_args, na, lineterm="", n=0)
                                                     if l[:1] in "+-" and not l.startswith(("+++", "---"))))
    for k in sorted(set(n) | set(o_full)):
        if k in ("args", "log"):
            continue
        a, b = ph(json.dumps(o_full.get(k))), ph(json.dumps(n.get(k)))
        if a != b:
            report.append(f"- `{k}`: 0.1.40.3 {a} -> 0.2.0-beta {b}")
    report.append("")
(W / "config-diff.md").write_text("\n".join(report) + "\n", encoding="utf-8")
print("\n".join(report))

out = W / "configs-arms"
out.mkdir(exist_ok=True)
base = {k: json.loads((W / "configs-setup" / f"{k}.json").read_text(encoding="utf-8")) for k in ("A", "B", "D")}


def arm(src, window=None, env=None):
    c = json.loads(json.dumps(base[src]))
    a = c["args"]
    if window:
        a[a.index("--kv-resident") + 1] = str(window)
    a += CACHE
    if env:
        c["env"] = dict(env)
    return c


arms = {"A": arm("A"), "B": arm("B"), "C": arm("B", window=20480), "D": arm("D"),
        "B-share-off": arm("B", env={"STRATA_PREFILL_CPU_SHARE": "0"})}
for name, c in arms.items():
    (out / f"{name}.json").write_text(json.dumps(c, indent=1), encoding="utf-8")
    print(f"{name}: " + " ".join(norm_args(c['args'][c['args'].index('--max-context'):]))
          + (f"  env {c['env']}" if c.get("env") else ""))
