"""Shared logic of the release test grid: the cell list, the pass / fail rules and the summary table.

Pure standard library (Python 3.8+), no network, no processes: the runner on the test boxes (cell_runner.py), the local
driver (grid.py) and the tests (tools/test_release_grid.py) all import this file.
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
CELLS_FILE = HERE / "cells.json"
LOCK_FILE = HERE / "models.lock.json"

PASS, FAIL, SKIP = "pass", "fail", "skip"
DRAFTERS = ("on", "off", "default")

# The rule from the owner (2026-11-23): a candidate fails when its decode or prefill rate is more than 5% below the
# previous release's on the same box, beyond the spread of the two runs.
SPEED_TOLERANCE = 0.05


def load_cells(path=None) -> dict:
    with open(path or CELLS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    problems = check_cells(data)
    if problems:
        raise ValueError("cells.json: " + "; ".join(problems))
    return data


def check_cells(data: dict) -> list:
    """Problems in the cell data (an unknown model, a drafter value that does not exist, a box without a folder)."""
    out = []
    boxes, models = data.get("boxes", {}), data.get("models", {})
    for bid, box in boxes.items():
        for key in ("backend", "class", "grid_dir", "os"):
            if not box.get(key):
                out.append("box %s has no %s" % (bid, key))
        for mid, entry in (box.get("models") or {}).items():
            if mid not in models:
                out.append("box %s lists the unknown model %s" % (bid, mid))
            if not isinstance(entry, dict):
                out.append("box %s model %s: the entry must be an object" % (bid, mid))
                continue
            if entry.get("drafters") and not set(entry["drafters"]) <= set(DRAFTERS):
                out.append("box %s model %s: drafters must be among %s" % (bid, mid, "/".join(DRAFTERS)))
            if not entry.get("skip") and not entry.get("files") and not entry.get("path") and not entry.get("fetch"):
                out.append("box %s model %s: needs path, files, fetch or skip" % (bid, mid))
    for mid, model in models.items():
        for key in ("family", "title"):
            if not model.get(key):
                out.append("model %s has no %s" % (mid, key))
    return out


def cell_id(box: str, model: str, drafter: str) -> str:
    return "%s__%s__%s" % (box, model, drafter)


def expand(data: dict, boxes=None, models=None, tier=None, drafters=None) -> list:
    """The flat cell list: one dict per box x model x drafter. A cell with a `skip` reason stays in the list (the table shows it)."""
    cells = []
    for bid, box in data["boxes"].items():
        if boxes and bid not in boxes:
            continue
        for mid, entry in (box.get("models") or {}).items():
            if models and mid not in models:
                continue
            model = data["models"][mid]
            if tier == "core" and entry.get("tier", model.get("tier", "core")) != "core":
                continue
            has_drafter = model.get("drafter", True)
            opts = entry.get("drafters") or (["on", "off"] if has_drafter else ["default"])
            for dr in opts:
                if drafters and dr not in drafters:
                    continue
                cells.append({
                    "id": cell_id(bid, mid, dr), "box": bid, "model": mid, "drafter": dr,
                    "family": model["family"], "skip": entry.get("skip"),
                    "tier": entry.get("tier", model.get("tier", "core")),
                    "priority": model.get("priority", 10),
                })
    cells.sort(key=lambda c: c["priority"])               # stable: the owner's priority models first, then the file order
    return cells


def sha256_file(path, block=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def text_fingerprint(texts) -> str:
    """Short hash of a list of greedy replies; two runs with equal output give the same string."""
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def spread(values) -> float:
    """(max - min) / median of a list of rates, 0 for fewer than two values."""
    vals = [v for v in values if v is not None and v > 0]
    if len(vals) < 2:
        return 0.0
    med = statistics.median(vals)
    return (max(vals) - min(vals)) / med if med else 0.0


def median(values):
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


def compare_speed(prev: dict, cand: dict, tolerance: float = SPEED_TOLERANCE) -> dict:
    """prev / cand: {"decode": [tok/s...], "prefill": [tok/s...]}.

    Fails a rate when candidate_median is lower than previous_median by more than `tolerance` plus the larger of the two
    spreads (a run that scatters by 8% cannot show a 5% loss). Returns {"decode": {...}, "prefill": {...}, "verdict": pass|fail|skip}.
    """
    out, verdict = {}, SKIP
    for key in ("decode", "prefill"):
        p, c = (prev or {}).get(key) or [], (cand or {}).get(key) or []
        pm, cm = median(p), median(c)
        if not pm or not cm:
            out[key] = {"verdict": SKIP, "prev": pm, "cand": cm}
            continue
        sp = max(spread(p), spread(c))
        change = (cm - pm) / pm
        bad = change < -(tolerance + sp)
        out[key] = {"verdict": FAIL if bad else PASS, "prev": round(pm, 2), "cand": round(cm, 2),
                    "change_pct": round(change * 100, 1), "spread_pct": round(sp * 100, 1)}
        if bad:
            verdict = FAIL
        elif verdict != FAIL:
            verdict = PASS
    out["verdict"] = verdict
    return out


def compare_fingerprint(prev, cand, declared=None) -> dict:
    """Greedy fingerprints of two releases for one cell. `declared` is a reason string when the change was announced."""
    if not prev or not cand:
        return {"verdict": SKIP, "note": "no fingerprint on one side"}
    if prev == cand:
        return {"verdict": PASS, "note": "identical"}
    if declared:
        return {"verdict": PASS, "note": "changed, declared: " + declared}
    return {"verdict": FAIL, "note": "greedy output differs from the previous release (%s -> %s)" % (prev, cand)}


def declared_for(declared: dict, cid: str):
    """declared_changes.json: {"<glob on the cell id>": "reason"}; the first match wins."""
    import fnmatch
    for pat, why in (declared or {}).items():
        if fnmatch.fnmatch(cid, pat):
            return why
    return None


def load_results(folder) -> dict:
    """{cell_id: result} from every *.json in a results folder."""
    out = {}
    folder = Path(folder)
    if folder.is_dir():
        for p in sorted(folder.glob("*.json")):
            try:
                r = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(r, dict) and r.get("cell"):
                out[r["cell"]] = r
    return out


def _step(result, name):
    return ((result or {}).get("steps") or {}).get(name) or {}


def judge(result: dict, prev: dict = None, declared=None, tolerance: float = SPEED_TOLERANCE) -> dict:
    """One row of the summary table: the cell's verdict with the numbers and the reasons.

    A result with verdict "skip" stays a skip. Otherwise the runner's own steps decide (install, launch, first chat, prompts,
    agent session); with a previous release's result the speed and greedy fingerprint rules are added.
    """
    cid = result.get("cell", "?")
    row = {"cell": cid, "box": result.get("box"), "model": result.get("model"), "drafter": result.get("drafter"),
           "release": result.get("release"), "reasons": [], "notes": []}
    if result.get("verdict") == SKIP:
        row.update(verdict=SKIP, reasons=[result.get("skip_reason") or "skipped"])
        return row
    verdict = PASS
    for name, step in ((result.get("steps") or {}).items()):
        st = step.get("status")
        if st == FAIL:
            verdict = FAIL
            row["reasons"].append("%s: %s" % (name, step.get("reason") or "failed"))
        elif st == SKIP:
            row["notes"].append("%s skipped: %s" % (name, step.get("reason") or ""))
    num = result.get("numbers") or {}
    row["numbers"] = num
    if prev:
        sp = compare_speed(prev.get("numbers") or {}, num, tolerance)
        row["speed_vs_prev"] = sp
        if sp["verdict"] == FAIL:
            verdict = FAIL
            for key in ("decode", "prefill"):
                if sp[key].get("verdict") == FAIL:
                    row["reasons"].append("%s %.1f -> %.1f tok/s (%+.1f%%, spread %.1f%%)" % (
                        key, sp[key]["prev"], sp[key]["cand"], sp[key]["change_pct"], sp[key]["spread_pct"]))
        fp = compare_fingerprint(_step(prev, "fingerprint").get("value"), _step(result, "fingerprint").get("value"), declared_for(declared, cid))
        row["fingerprint_vs_prev"] = fp
        if fp["verdict"] == FAIL:
            verdict = FAIL
            row["reasons"].append("fingerprint: " + fp["note"])
        elif fp["verdict"] == PASS and fp["note"] != "identical":
            row["notes"].append(fp["note"])
    elif _step(result, "launch").get("status") == PASS:
        row["notes"].append("no previous-release result: speed and fingerprint not compared")
    row["verdict"] = verdict
    return row


def usage_levels(result: dict) -> dict:
    """{clients: level dict} of the usage suite's parallel case in a cell result, else {}."""
    return (((result or {}).get("usage") or {}).get("cases") or {}).get("parallel", {}).get("levels") or {}


def add_reference(row: dict, targets, result: dict = None):
    """Distance from published or owner target lines: percent of each target's decode, prefill (4K) and C64 total rate.

    Not a pass / fail rule yet: the row only carries the percentages, the table shows them. A target with `model_glob` applies to the matching models only.
    """
    import fnmatch
    num = row.get("numbers") or {}
    out = []
    for t in targets:
        pat = t.get("model_glob") or t.get("model")
        if pat and not fnmatch.fnmatch(row.get("model") or "", pat):
            continue
        d = {"target": t.get("label") or t.get("source") or "target"}
        if num.get("decode_tps") and t.get("decode_tps"):
            d["decode_pct"] = round(100.0 * num["decode_tps"] / t["decode_tps"], 1)
        pre = num.get("prefill_4k_tps") or num.get("prefill_tps")
        if pre and t.get("prefill_tps") or (pre and t.get("prefill_4k_tps")):
            d["prefill_pct"] = round(100.0 * pre / (t.get("prefill_tps") or t.get("prefill_4k_tps")), 1)
        lv = usage_levels(result)
        c64 = (lv.get("64") or {}).get("burst_tps") or (lv.get("64") or {}).get("throughput_tps")
        if c64 and t.get("c64_total_tps"):
            d["c64_pct"] = round(100.0 * c64 / t["c64_total_tps"], 1)
        d["goal"] = ", ".join(k.replace("_tps", "").replace("_", " ") + " " + str(t[k]) for k in ("decode_tps", "prefill_tps", "prefill_4k_tps", "c64_total_tps") if t.get(k)) or "record only"
        out.append(d)
    if out:
        row["targets"] = out


def summarize(results: dict, prev_results: dict = None, declared=None, expected_cells=None, references=None) -> dict:
    """references: {box id: reference dict} (cells.json: the box's reference entry) for the boxes that have one; applies to the reference's own model."""
    """Rows for every cell: judged results, plus a `missing` row for an expected cell that has no result file."""
    rows = []
    for cid, r in sorted(results.items()):
        row = judge(r, (prev_results or {}).get(cid), declared)
        refs = (references or {}).get(r.get("box"))
        if refs and r.get("verdict") != SKIP:
            add_reference(row, refs, r)
        rows.append(row)
    for c in (expected_cells or []):
        if c["id"] not in results:
            if c.get("skip"):
                rows.append({"cell": c["id"], "box": c["box"], "model": c["model"], "drafter": c["drafter"],
                             "verdict": SKIP, "reasons": [c["skip"]], "notes": []})
            else:
                rows.append({"cell": c["id"], "box": c["box"], "model": c["model"], "drafter": c["drafter"],
                             "verdict": SKIP, "reasons": ["not run in this pass"], "notes": []})
    rows.sort(key=lambda r: r["cell"])
    counts = {PASS: 0, FAIL: 0, SKIP: 0}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    return {"rows": rows, "counts": counts}


def _fmt(v, digits=1):
    return "-" if v is None else ("%.*f" % (digits, v))


def step_cell(result, name) -> str:
    st = _step(result, name)
    s = st.get("status")
    return {PASS: "ok", FAIL: "FAIL", SKIP: "-"}.get(s, "-")


def render_markdown(summary: dict, results: dict, title="Release grid", release=None, prev_label=None) -> str:
    c = summary["counts"]
    lines = ["# %s" % title, ""]
    if release:
        lines.append("Release under test: **%s**%s" % (release, (", compared with **%s**" % prev_label) if prev_label else ""))
        lines.append("")
    lines.append("**%d pass, %d fail, %d skip** (%d cells)" % (c[PASS], c[FAIL], c[SKIP], len(summary["rows"])))
    lines.append("")
    lines.append("| cell | verdict | install | start | chat | 4K | 32K | agent | decode tok/s | prefill tok/s | vs previous | why |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in summary["rows"]:
        res = results.get(r["cell"], {})
        num = r.get("numbers") or {}
        sp = r.get("speed_vs_prev") or {}
        vs = "-"
        if sp:
            parts = []
            for key, tag in (("decode", "dec"), ("prefill", "pre")):
                if sp.get(key, {}).get("change_pct") is not None:
                    parts.append("%s %+.1f%%" % (tag, sp[key]["change_pct"]))
            fp = r.get("fingerprint_vs_prev") or {}
            if fp.get("verdict"):
                parts.append("fp " + ("same" if fp["note"] == "identical" else fp["verdict"]))
            vs = ", ".join(parts) or "-"
        why = "; ".join(r.get("reasons") or [])
        if r.get("targets") and not why:
            why = "; ".join("%s: %s" % (t["target"], ", ".join(x for x in (
                ("decode %s%%" % t["decode_pct"]) if "decode_pct" in t else "", ("prefill %s%%" % t["prefill_pct"]) if "prefill_pct" in t else "",
                ("C64 %s%%" % t["c64_pct"]) if "c64_pct" in t else "") if x) or "recorded, no number yet") + " of (%s)" % t["goal"] for t in r["targets"])
        if not why and r.get("notes"):
            why = "(" + "; ".join(r["notes"][:2]) + ")"
        why = why.replace("|", "/").replace("\n", " ")
        if len(why) > 220:
            why = why[:217] + "..."
        lines.append("| %s | **%s** | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["cell"].replace("__", " / "), r["verdict"].upper(),
            step_cell(res, "install"), step_cell(res, "launch"), step_cell(res, "first_chat"),
            step_cell(res, "prompt_4k"), step_cell(res, "prompt_32k"), step_cell(res, "agent"),
            _fmt(num.get("decode_tps")), _fmt(num.get("prefill_tps")), vs, why))
    lines.append("")
    return "\n".join(lines)


def write_summary(summary: dict, results: dict, out_prefix, **kw):
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(str(out_prefix) + ".md").write_text(render_markdown(summary, results, **kw), encoding="utf-8")
    payload = {"counts": summary["counts"], "rows": summary["rows"], "release": kw.get("release"), "previous": kw.get("prev_label")}
    Path(str(out_prefix) + ".json").write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")


def have_room(free_gb: float, need_gb: float, floor_gb: float) -> bool:
    return free_gb - need_gb >= floor_gb


def default_results_dir() -> Path:
    return Path(os.environ.get("GRID_RESULTS") or (HERE.parents[1] / "grid-results"))
