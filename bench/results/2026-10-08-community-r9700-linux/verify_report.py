#!/usr/bin/env python3
"""Recompute the published measurements without a GPU or model downloads."""
import argparse
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parent
WORK = ("input_ids_sha256", "output_ids", "generated", "finish", "reused",
        "prompt_read", "drafts_accepted", "drafts_offered")
METRICS = ("prompt_ms", "decode_ms", "ttft_s", "wall_s")


def read(path):
    return json.loads((ROOT / path).read_text())


def same(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)


def measured(record, incremental=False):
    assert record["generated"] == len(record["output_ids"]) == 256
    assert record["finish"] == "length"
    assert all(math.isfinite(record[k]) and record[k] > 0 for k in METRICS)
    expected_read = record["increment"] if incremental else record["input_tokens"]
    expected_reuse = record["base_tokens"] if incremental else 0
    assert record["prompt_read"] == expected_read
    assert record["reused"] == expected_reuse
    assert record["prompt_tokens"] == record["input_tokens"] == expected_read + expected_reuse


def compare_summary(groups, summary):
    assert len(groups) == len(summary["rows"])
    for row in summary["rows"]:
        key = (row["history_or_input_tokens"], row["increment"])
        pairs = groups[key]
        assert len(pairs) == len(row["pairs"])
        assert row["identical_work"]
        for saved in row["pairs"]:
            pair = pairs[saved["index"]]
            a, b = pair["baseline"], pair["candidate"]
            assert all(a[k] == b[k] for k in WORK)
            assert saved["differing_work_fields"] == []
            for metric in METRICS:
                same(saved[metric]["baseline"], a[metric])
                same(saved[metric]["candidate"], b[metric])
                same(saved[metric]["time_reduction_pct"], 100 * (1 - b[metric] / a[metric]))
        for metric in METRICS:
            reductions = [100 * (1 - p["candidate"][metric] / p["baseline"][metric]) for p in pairs.values()]
            expected = {
                "median_paired_time_reduction_pct": statistics.median(reductions),
                "min_paired_time_reduction_pct": min(reductions),
                "max_paired_time_reduction_pct": max(reductions),
                "faster_pairs": sum(x > 0 for x in reductions),
                "regressions_over_2pct": sum(x < -2 for x in reductions),
                **{arm + "_median": statistics.median(p[arm][metric] for p in pairs.values())
                   for arm in ("baseline", "candidate")},
            }
            for name, value in expected.items():
                same(row[metric][name], value)


def check_matrix(name, expected_pairs, expected_count):
    directory = Path("evidence") / name
    matrix = read(directory / "matrix.json")
    assert matrix["complete"] and matrix["pairs"] == expected_pairs
    assert len(matrix["sessions"]) == expected_pairs * 2
    incremental = matrix["mode"] == "incremental"
    shapes = {(n, inc) for n in matrix["lengths"] for inc in matrix["increments"]} if incremental else {(n, 0) for n in matrix["lengths"]}
    groups, count = {}, 0
    for session in matrix["sessions"]:
        assert session["status"] == "passed" and session["returncode"] == 0
        result = read(directory / session["name"] / "results.json")
        assert result["binary_sha256"] == matrix["configs"][session["arm"]]["binary_sha256"]
        assert result["fixtures_sha256"] == matrix["fixtures_sha256"]
        device = read(directory / (session["name"] + "-launch.json"))["device"]
        assert device["hip_device_count"] == 1 and device["bdf"] == "0000:63:00.0"
        formal = [r for r in result["records"] if r["phase"] == "measurement"]
        warmups = [r for r in result["records"] if r["phase"] == "shape_warmup"]
        assert len(formal) == len(warmups) == len(shapes)
        actual = set()
        for r in formal:
            measured(r, incremental)
            key = (r["base_tokens"], r["increment"]) if incremental else (r["input_tokens"], 0)
            assert key not in actual
            actual.add(key)
            pair = groups.setdefault(key, {}).setdefault(session["pair"], {})
            assert session["arm"] not in pair
            pair[session["arm"]] = r
        assert actual == shapes
        count += len(formal)
    assert count == expected_count
    compare_summary(groups, read(directory / "summary.json"))
    return groups, count


def check_sources():
    layout = read("source-layout.json")
    repo = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=ROOT, text=True).strip()
    archive = subprocess.check_output(["git", "archive", layout["public_base_commit"],
                                       "CMakeLists.txt", "src", "include", "tests"], cwd=repo)
    with tempfile.TemporaryDirectory(prefix="r9700-report-source-") as tmp:
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(tmp, filter="data")
        for arm, patch in (("baseline", "measured-baseline.patch"), ("candidate", "pr1107-candidate.patch")):
            subprocess.run(["git", "apply", "--unidiff-zero", str(ROOT / "patches" / patch)], cwd=tmp, check=True)
            for name, expected in layout["source_sha256"][arm].items():
                assert hashlib.sha256((Path(tmp) / name).read_bytes()).hexdigest() == expected, (arm, name)
    print("Archived patches reproduce all changed measured engine sources.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-sources", action="store_true", help="also reconstruct source in a temporary directory using the local git object database")
    args = parser.parse_args()
    exports = read("export-manifest.json")
    for item in exports["files"]:
        assert hashlib.sha256((ROOT / item["file"]).read_bytes()).hexdigest() == item["published_sha256"], item["file"]
    current = read("current-speed/iq3_s-session/results.json")
    summary = read("current-speed/summary.json")
    formal = [r for r in current["records"] if r["phase"] == "measurement"]
    assert len(formal) == 12
    assert len([r for r in current["records"] if r["phase"] == "shape_warmup"]) == 4
    fixtures = json.loads(gzip.decompress((ROOT / "current-speed/fixtures.json.gz").read_bytes()))
    hashes = {f["tokens"]: f["ids_sha256"] for f in fixtures["fixtures"]}
    for row in summary["rows"]:
        records = [r for r in formal if r["input_tokens"] == row["input_tokens"]]
        assert sorted(r["repeat"] for r in records) == [0, 1, 2]
        for r in records:
            measured(r)
            assert r["input_ids_sha256"] == hashes[r["input_tokens"]]
        for key, values in {
            "prefill_tokens_per_s": [r["prompt_read"] * 1000 / r["prompt_ms"] for r in records],
            "decode_tokens_per_s": [r["generated"] * 1000 / r["decode_ms"] for r in records],
            "ttft_s": [r["ttft_s"] for r in records], "wall_s": [r["wall_s"] for r in records],
        }.items():
            for stat, value in (("median", statistics.median(values)), ("min", min(values)), ("max", max(values))):
                same(row[key][stat], value)
    first, n1 = check_matrix("pr1107-incremental-screen-iq3s", 2, 20)
    confirm, n2 = check_matrix("pr1107-incremental-confirm-iq3s", 3, 30)
    for key, pairs in confirm.items():
        first[key].update({index + 2: pair for index, pair in pairs.items()})
    compare_summary(first, read("evidence/pr1107-incremental-five-pairs-iq3s/summary.json"))
    _, n3 = check_matrix("pr1107-incremental-screen-iq2xs", 2, 20)
    _, n4 = check_matrix("pr1107-fresh-screen-iq3s", 2, 12)
    _, n5 = check_matrix("pr1107-fresh-screen-iq2xs", 2, 12)
    checks = 0
    for quant in ("iq3s", "iq2xs"):
        for arm in ("baseline", "candidate"):
            result = read(f"evidence/pr1107-product-http-{quant}/{arm}/results.json")
            assert result["passed"] and result["complete"] and len(result["records"]) == 8
            assert all(r["passed"] for r in result["records"])
            checks += len(result["records"])
    coverage = {"scope": "Recomputed archived evidence, not new GPU measurements or automatic performance acceptance",
                "paired_formal_requests": n1+n2+n3+n4+n5, "current_speed_formal_requests": len(formal),
                "http_task_checks_passed": checks, "exported_artifact_hashes_verified": len(exports["files"]),
                "all_workload_performance_acceptance": False}
    assert coverage["paired_formal_requests"] == 94 and checks == 32
    if args.check_sources:
        check_sources()
    saved = ROOT / "coverage.json"
    if saved.exists():
        assert json.loads(saved.read_text()) == coverage
    else:
        saved.write_text(json.dumps(coverage, indent=2) + "\n")
    print(json.dumps(coverage, indent=2))


if __name__ == "__main__":
    main()
