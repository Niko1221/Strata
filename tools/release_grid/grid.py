#!/usr/bin/env python3
"""grid.py: the release test grid, driven from the owner PC (it starts nothing heavy here; every run goes through jobq on a test box).

    grid.py plan    --cand v0.2.20 [--prev v0.2.19] [--boxes a,b] [--models m] [--tier core|full]
    grid.py planfile --cand v0.2.20 --boxes github-b70                                     a plan file for a box without jobq (the Intel B70 runner)
    grid.py run     --cand v0.2.20 [--prev v0.2.19] --boxes p620,p1006 [--timeout 6h]      push the scripts, write the plan, submit one jobq job per box
    grid.py stage   --cand v0.2.20 --boxes p620                                             copy a published release's files to a box without DNS (p620, l620)
    grid.py fetch   --boxes vast0221-3060 [--models qwen-pack-q2]                           jobq job per box: download + SHA-256 check of the cells' `fetch` models
    grid.py collect --work DIR [--boxes ...] --cand v0.2.20                                 copy the result files back from the boxes
    grid.py summary --work DIR --cand v0.2.20 [--prev v0.2.19]                              table (markdown + JSON) with pass / fail / skip and the numbers
    grid.py boxes                                                                          the cells.json box list

A release label is a tag (v0.2.20: the GitHub release folder is its feed) or LABEL=FEED where FEED is a URL, or a folder on the box that holds a
test-signed feed (add --trust FILE-ON-BOX for the test key). --prev is the previous release: its cells run first on the same box, in the same job,
and the speed and greedy-fingerprint rules compare against it (docs/RELEASE_GRID.md).
Jobs: `python jobq.py wait <id>` is run by the caller (the job ids are printed and kept in WORK/jobs.json).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grid_lib as gl  # noqa: E402

SCRIPTS = ["grid_lib.py", "cell_runner.py", "fetch_models.py", "vram_hog.cu", "usage/usage_suite.py", "models.lock.json", "cells.json"]
JOBQ = os.environ.get("GRID_JOBQ") or r"C:\Users\AI-Server\Desktop\Strata\phaseA-tests\jobq\jobq.py"
GH_DOWNLOAD = "https://github.com/Niko1221/Strata/releases/download/%s"


# ---------------------------------------------------------------------------------------------------------------- plan
def unmsys(path):
    """Git Bash rewrites an argument like /workspace/x into C:/Program Files/Git/workspace/x before Python sees it: undo that (set MSYS_NO_PATHCONV=1 to avoid it)."""
    m = re.match(r"^[A-Za-z]:/Program Files/Git(/.*)$", path or "")
    return m.group(1) if m else path


def parse_release(text: str):
    """'v0.2.20' -> (label, feed url); 'main-abc=/data/feeds/main' -> (label, folder or url)."""
    if "=" in text:
        label, feed = text.split("=", 1)
        return label.strip(), unmsys(feed.strip())
    if re.fullmatch(r"v\d+\.\d+\.\d+(\.\d+)?", text):
        return text, GH_DOWNLOAD % text
    raise SystemExit("release %r: use a tag like v0.2.20 or LABEL=FEED" % text)


def merged_profile(data, box, model_id, entry):
    prof = copy.deepcopy(data["defaults"]["profile"])
    if box.get("profile_name"):
        prof.update(data["profiles"][box["profile_name"]])
    prof.update(box.get("profile") or {})
    prof.update(entry.get("profile") or {})
    return prof


def model_spec(data, box, model_id, entry):
    spec = {"launch_args": list(data["models"][model_id].get("launch_args") or [])}
    if entry.get("files"):
        spec["files"] = entry["files"]
        lock = json.loads(gl.LOCK_FILE.read_text(encoding="utf-8"))["models"]
        pinned = {f["name"]: f["bytes"] for f in lock.get(data["models"][model_id]["pins"], {}).get("files", [])}
        spec["sizes"] = {os.path.basename(f): pinned[os.path.basename(f)] for f in entry["files"] if os.path.basename(f) in pinned}
    elif entry.get("path"):
        spec["path"] = entry["path"]
    elif entry.get("fetch"):
        key = data["models"][model_id]["pins"]
        lock = json.loads(gl.LOCK_FILE.read_text(encoding="utf-8"))["models"]
        spec["path"] = "%s/models/%s" % (box["grid_dir"].rstrip("/"), lock[key]["folder"])
    if entry.get("vision_files"):
        spec["vision_files"] = entry["vision_files"]
    spec["launch_args"] += list(entry.get("launch_args") or [])
    return spec


def build_specs(data, releases, boxes=None, models=None, tier=None, drafters=None, trust=None):
    """releases: [(label, feed), ...], previous first. -> {box id: [spec, ...]} with the cells of each release interleaved per cell."""
    out = {}
    for c in gl.expand(data, boxes, models, tier, drafters):
        box = data["boxes"][c["box"]]
        entry = box["models"][c["model"]]
        for label, feed in releases:
            if box.get("offline") and feed.startswith(GH_DOWNLOAD % ""):
                feed = "%s/feeds/%s" % (box["grid_dir"].rstrip("/"), label)     # no DNS on the box: grid.py stage copied the release files there
            spec = {
                "cell": c["id"], "label": label, "box": c["box"], "model": c["model"], "drafter": c["drafter"], "family": c["family"],
                "feed": feed, "trust": (None if feed.startswith(GH_DOWNLOAD % "") or "/feeds/v" in feed else trust), "grid_dir": box["grid_dir"], "port": data["defaults"]["port"],
                "env": box.get("env") or {}, "vram_class_gib": box.get("vram_class_gib"), "ram_cap_gib": box.get("ram_cap_gib"), "launch_extra": list(box.get("launch_extra") or []) + list(entry.get("launch_extra") or []),
                "profile": merged_profile(data, box, c["model"], entry),
            }
            if c["skip"]:
                spec["skip"] = c["skip"]
            else:
                spec["model_spec"] = model_spec(data, box, c["model"], entry)
            out.setdefault(c["box"], []).append(spec)
    return out


# ---------------------------------------------------------------------------------------------------------------- remote
def sh(cmd, check=True, timeout=300):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise SystemExit("command failed (%s): %s\n%s" % (r.returncode, " ".join(cmd[:4]), (r.stdout + r.stderr)[-600:]))
    return r


def remote_rel(path: str) -> str:
    return path[2:] if path.startswith("~/") else path


def script_hash() -> str:
    h = hashlib.sha256()
    for n in SCRIPTS:
        h.update((HERE / n).read_bytes())
    return h.hexdigest()[:8]


def push(box: dict) -> str:
    """Copy the scripts to <grid_dir>/bin-<hash>/ (a new folder per version: a running job never sees its script change). Returns the folder."""
    ssh = box["ssh"]
    folder = "%s/bin-%s" % (box["grid_dir"].rstrip("/"), script_hash())
    sh(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", ssh, "mkdir -p %s/plans %s/results" % (box["grid_dir"], box["grid_dir"]) + " %s" % folder])
    sh(["scp", "-o", "BatchMode=yes", "-q"] + [str(HERE / n) for n in SCRIPTS] + ["%s:%s/" % (ssh, remote_rel(folder))], timeout=600)
    return folder


def jobq(*args, check=True):
    env = dict(os.environ, MSYS_NO_PATHCONV="1")
    r = subprocess.run([sys.executable, JOBQ] + list(args), capture_output=True, text=True, env=env, timeout=600)
    if check and r.returncode != 0:
        raise SystemExit("jobq %s failed: %s" % (args[0], (r.stdout + r.stderr)[-500:]))
    return r.stdout.strip()


def submit(box_id, box, name, cmd, timeout, locks):
    args = ["submit", box["jobq"], "--name", name, "--timeout", timeout]
    for lk in (locks if locks else ["none"]):
        args += ["--lock", lk]
    args += ["--", cmd]
    out = jobq(*args)
    m = re.search(r"\b(%s-\d{8}-\d{4}-[0-9a-f]{4})\b" % re.escape(box["jobq"]), out)
    return m.group(1) if m else out.splitlines()[-1]


def runner_cmd(box, folder, plan_remote, results_remote, only=None, minutes=360):
    py = box.get("python", "python3")
    cmd = "%s %s/cell_runner.py --plan %s --out %s" % (py, folder, plan_remote, results_remote)
    if only:
        cmd += " --only " + shlex.quote(",".join(only))
    if box.get("wrapper"):
        w = [x.replace("{minutes}", str(minutes)) for x in box["wrapper"]]
        cmd = " ".join(w) + " " + cmd
    return cmd


# ---------------------------------------------------------------------------------------------------------------- commands
def cmd_boxes(a, data):
    for bid, b in data["boxes"].items():
        n = len(b.get("models") or {})
        print("%-16s %-52s jobq=%-16s models=%d" % (bid, b["class"], b.get("jobq"), n))


def cmd_plan(a, data):
    rel = [parse_release(r) for r in ([a.prev] if a.prev else []) + [a.cand]]
    specs = build_specs(data, rel, _csv(a.boxes), _csv(a.models), a.tier, _csv(a.drafters))
    total = skip = 0
    for bid, lst in specs.items():
        for s in lst:
            if s["label"] != rel[-1][0]:
                continue
            total += 1
            skip += 1 if s.get("skip") else 0
            print("%-48s %s" % (s["cell"], ("SKIP: " + s["skip"][:90]) if s.get("skip") else "run"))
    print("%d cells (%d to run, %d skipped) for %s%s" % (total, total - skip, skip, rel[-1][0], (" against " + rel[0][0]) if a.prev else ""))


def cmd_run(a, data):
    rel = [parse_release(r) for r in ([a.prev] if a.prev else []) + [a.cand]]
    boxes = _csv(a.boxes)
    specs = build_specs(data, rel, boxes, _csv(a.models), a.tier, _csv(a.drafters), unmsys(a.trust))
    if a.launch_parallel or a.tag:
        for lst in specs.values():
            for sp in lst:
                if a.launch_parallel:
                    sp["launch_extra"] = list(sp.get("launch_extra") or []) + ["--parallel", str(a.launch_parallel)]
                    sp["parallel_flag"] = a.launch_parallel
                if a.tag:
                    sp["cell"] = sp["cell"] + "@" + a.tag                 # a separate result file for this variant of the cell
    if a.usage:
        for lst in specs.values():
            for sp in lst:
                sp["profile"].update({"usage_suite": True, "vision": "on", "context": "64k", "cell_budget_s": 14400, "request_timeout_s": 1800, "ready_timeout_s": max(3000, sp["profile"].get("ready_timeout_s", 0))})
                if a.usage == "only":
                    sp["profile"]["skip_standard"] = True
                if a.usage_cases:
                    sp["profile"]["usage_cases"] = _csv(a.usage_cases)
                if a.usage_long:
                    sp["profile"]["usage_long"] = a.usage_long
                    sp["profile"]["context"] = "%d" % (max(int(x) for x in a.usage_long.split(",")) + 4096)
                if a.usage_parallel:
                    sp["profile"]["usage_parallel"] = a.usage_parallel
                if a.usage_turns:
                    sp["profile"]["usage_agent_turns"] = a.usage_turns
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    jobs_file = work / "jobs.json"
    jobs = json.loads(jobs_file.read_text()) if jobs_file.exists() else []
    stamp = time.strftime("%m%d-%H%M%S")
    for bid, lst in specs.items():
        box = data["boxes"][bid]
        if not box.get("jobq"):
            print("%s: no jobq box (%s)" % (bid, "runs through its own runner"))
            continue
        runnable = [s for s in lst if not s.get("skip")]
        # skipped cells need no box: their result files are written locally by `summary` from cells.json
        if not runnable:
            continue
        folder = push(box)
        plan_name = "plan-%s-%s-%s.json" % (rel[-1][0], bid, stamp)
        local = work / plan_name
        local.write_text(json.dumps({"cells": runnable, "made": stamp}, indent=1), encoding="utf-8")
        sh(["scp", "-o", "BatchMode=yes", "-q", str(local), "%s:%s/plans/" % (box["ssh"], remote_rel(box["grid_dir"].rstrip("/")))])
        ids = []
        for cid in dict.fromkeys(s["cell"] for s in runnable):          # one job per cell (both releases): the box lock is free between cells
            cmd = runner_cmd(box, folder, "%s/plans/%s" % (box["grid_dir"].rstrip("/"), plan_name), "%s/results" % box["grid_dir"].rstrip("/"),
                             only=[cid], minutes=int(re.sub(r"\D", "", a.timeout) or 180) * (60 if a.timeout.endswith("h") else 1))
            jid = submit(bid, box, "grid-%s-%s" % (rel[-1][0], cid.split("__", 1)[1]), cmd, a.timeout, box.get("locks") or [])
            ids.append(jid)
            jobs.append({"box": bid, "cell": cid, "job": jid, "plan": plan_name, "cand": rel[-1][0], "prev": rel[0][0] if a.prev else None, "bin": folder})
        print("%s: submitted %d jobs: %s" % (bid, len(ids), ", ".join(ids)))
    jobs_file.write_text(json.dumps(jobs, indent=1), encoding="utf-8")


def cmd_planfile(a, data):
    """The plan for a box that has no jobq (the Intel B70 runner): copy the file to the runner and run cell_runner.py on it."""
    rel = [parse_release(r) for r in ([a.prev] if a.prev else []) + [a.cand]]
    specs = build_specs(data, rel, _csv(a.boxes), _csv(a.models), a.tier, _csv(a.drafters), a.trust)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    for bid, lst in specs.items():
        out = work / ("plan-%s-%s.json" % (rel[-1][0], bid))
        out.write_text(json.dumps({"cells": [s for s in lst if not s.get("skip")], "made": time.strftime("%m%d-%H%M%S")}, indent=1), encoding="utf-8")
        print("wrote", out)


def stage_assets(tag: str, variants, cache: Path) -> list:
    """Download (read-only, gh) the files of a release that a Linux x86_64 box with these engine variants needs; returns the paths."""
    dest = cache / tag
    dest.mkdir(parents=True, exist_ok=True)
    feed_files = ["install.sh", "install-manifest.txt", "strata-feed.json", "strata-feed.json.sig", "strata-control-*-linux-x86_64.tar.gz"]
    pats = feed_files + ["strata-*-linux-x86_64-*%s.tar.gz" % v for v in variants]
    for pat in pats:
        sh(["gh", "release", "download", tag, "-R", "Niko1221/Strata", "-p", pat, "-D", str(dest), "--clobber"], timeout=3600)
    return sorted(dest.iterdir())


def cmd_stage(a, data):
    """A box without DNS (p620, l620) cannot reach the release folder: copy the files it needs to <grid_dir>/feeds/<label>/ and serve them there."""
    label, feed = parse_release(a.cand)
    if not feed.startswith(GH_DOWNLOAD % ""):
        raise SystemExit("stage only copies a published release; a local feed is already a folder on the box")
    cache = Path(a.assets)
    for bid in _csv(a.boxes) or []:
        box = data["boxes"][bid]
        files = stage_assets(label, box.get("feed_variants") or [], cache)
        dest = "%s/feeds/%s" % (box["grid_dir"].rstrip("/"), label)
        sh(["ssh", "-o", "BatchMode=yes", box["ssh"], "mkdir -p %s" % dest])
        sh(["scp", "-o", "BatchMode=yes", "-q"] + [str(f) for f in files] + ["%s:%s/" % (box["ssh"], remote_rel(dest))], timeout=3600)
        print("%s: staged %d files (%.0f MB) in %s" % (bid, len(files), sum(f.stat().st_size for f in files) / 1e6, dest))


def cmd_fetch(a, data):
    lock = json.loads(gl.LOCK_FILE.read_text(encoding="utf-8"))["models"]
    for bid in _csv(a.boxes):
        box = data["boxes"][bid]
        keys = list(_csv(a.keys) or [])
        for mid, entry in ([] if keys else box["models"].items()):
            if entry.get("fetch") and (not a.models or mid in _csv(a.models)):
                if entry.get("skip") and not a.include_skipped:
                    continue
                if entry.get("tier") == "full" and a.tier == "core":
                    continue
                keys.append(data["models"][mid]["pins"])
        if not keys:
            print("%s: nothing to fetch" % bid)
            continue
        folder = push(box)
        py = box.get("python", "python3")
        dest = "%s/models" % box["grid_dir"].rstrip("/")
        cmd = " && ".join("%s %s/fetch_models.py fetch %s --dest %s --lock %s/models.lock.json --floor-gb %s" % (py, folder, k, dest, folder, a.floor_gb) for k in keys)
        need = sum(sum(f["bytes"] for f in lock[k]["files"]) for k in keys) / 1e9
        jid = submit(bid, box, "grid-fetch-%s" % bid, cmd, a.timeout, [])
        print("%s: fetch %s (%.0f GB) job %s" % (bid, ", ".join(keys), need, jid))


def cmd_collect(a, data):
    work = Path(a.work)
    labels = [parse_release(r)[0] for r in ([a.prev] if a.prev else []) + [a.cand]]
    for bid in _csv(a.boxes) or [b for b, v in data["boxes"].items() if v.get("jobq")]:
        box = data["boxes"][bid]
        for label in labels:
            dest = work / "results" / label
            dest.mkdir(parents=True, exist_ok=True)
            r = sh(["scp", "-o", "BatchMode=yes", "-q", "%s:%s/results/%s/%s__*" % (box["ssh"], remote_rel(box["grid_dir"].rstrip("/")), label, bid), str(dest)], check=False, timeout=600)
            print("%s %s: %s" % (bid, label, "ok" if r.returncode == 0 else "nothing yet"))


def cmd_summary(a, data):
    work = Path(a.work)
    cand = parse_release(a.cand)[0]
    prev = parse_release(a.prev)[0] if a.prev else None
    results = gl.load_results(work / "results" / cand)
    prev_results = gl.load_results(work / "results" / prev) if prev else {}
    declared = json.loads(Path(a.declared).read_text()) if a.declared and Path(a.declared).exists() else {}
    expected = gl.expand(data, _csv(a.boxes), _csv(a.models), a.tier, _csv(a.drafters))
    if not a.boxes:                                   # without --boxes: the boxes that ran, plus every cell that has a skip reason
        ran = {r.get("box") for r in results.values()}
        expected = [c for c in expected if c["box"] in ran or c.get("skip")]
    refs = {}
    for bid, b in data["boxes"].items():
        names = ([b["reference"]] if b.get("reference") else []) + list(b.get("targets") or [])
        got = [dict(data["references"][n], label=n) for n in names if n in data.get("references", {})]
        if got:
            refs[bid] = got
    summary = gl.summarize(results, prev_results, declared, expected, refs)
    gl.write_summary(summary, results, work / ("summary-%s" % cand), title="Release grid: %s" % cand, release=cand, prev_label=prev)
    c = summary["counts"]
    print("%d pass, %d fail, %d skip -> %s" % (c["pass"], c["fail"], c["skip"], work / ("summary-%s.md" % cand)))


def _csv(text):
    return [x for x in (text or "").split(",") if x] or None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["boxes", "plan", "planfile", "stage", "run", "fetch", "collect", "summary"])
    ap.add_argument("--cells", default=None)
    ap.add_argument("--cand")
    ap.add_argument("--prev")
    ap.add_argument("--boxes")
    ap.add_argument("--models")
    ap.add_argument("--drafters")
    ap.add_argument("--tier", choices=["core", "full"], default="full")
    ap.add_argument("--work", default=str(gl.default_results_dir()))
    ap.add_argument("--timeout", default="6h")
    ap.add_argument("--trust")
    ap.add_argument("--declared", help="declared_changes.json: {cell-id glob: reason}")
    ap.add_argument("--floor-gb", default="25")
    ap.add_argument("--include-skipped", action="store_true")
    ap.add_argument("--launch-parallel", type=int, help="start the engine with --parallel N (the Terminal accepts 1 to 8); without it the Terminal's own default (fit to the card) is used")
    ap.add_argument("--tag", help="suffix for the cell id (a variant run of the same cell keeps its own result file)")
    ap.add_argument("--keys", help="fetch: explicit pin keys (ink/Vision, ink/mmproj-BF16, ...) instead of the cells' fetch models")
    ap.add_argument("--usage-cases", help="comma list of usage cases (coding,agent,vision,longdoc,chat,parallel)")
    ap.add_argument("--usage-long", help="document sizes in tokens for the longdoc case, e.g. 8000 or 30000,60000")
    ap.add_argument("--usage-parallel", help="client counts, e.g. 1,2,4")
    ap.add_argument("--usage-turns", type=int)
    ap.add_argument("--usage", choices=["also", "only"], help="run the usage suite (usage/usage_suite.py) in each cell, after the standard checks (also) or instead of them (only)")
    ap.add_argument("--assets", default="C:/w-0221-grid-scratch/assets", help="owner-PC cache of downloaded release files (stage)")
    a = ap.parse_args(argv)
    data = gl.load_cells(a.cells)
    if a.cmd != "boxes" and a.cmd != "fetch" and not a.cand:
        raise SystemExit("--cand is required")
    {"boxes": cmd_boxes, "plan": cmd_plan, "planfile": cmd_planfile, "stage": cmd_stage, "run": cmd_run, "fetch": cmd_fetch, "collect": cmd_collect, "summary": cmd_summary}[a.cmd](a, data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
