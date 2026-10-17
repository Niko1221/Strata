"""Verify the published measurements and reproduce their statistics (standard library only)."""

import csv
import json
import math
from pathlib import Path
from collections import Counter
from analyze import analyze
from run_ab import compare, digest

ROOT = Path(__file__).resolve().parents[1]
x = json.loads((ROOT / "data/results.json").read_text())
assert x["completed"] and x["restored"]
assert len(x["arms"]) == 28 and len(x["comparisons"]) == 14
assert x["fixture_sha256"] == digest(ROOT / "data/fixtures.json")
assert x["schedule"] == json.loads((ROOT / "data/schedule.json").read_text())
assert Counter(v["order"][0] for v in x["schedule"] if v["kind"] == "ab") == {
    "on": 6,
    "auto": 6,
}
assert sum(len(v["requests"]) for v in x["arms"]) == 84
plan = json.loads((ROOT / "config.example.json").read_text())["reference_plan"]
for a in x["arms"]:
    assert a["engine_arm"] == ("on" if a["kind"] == "aa" else a["arm"])
    info = dict(v.split("=", 1) for v in a["info"][-1].split()[1:] if "=" in v)
    assert all(info.get(k) == str(v) for k, v in plan.items())
    assert all(v["available_bytes"] >= 20 * 1024**3 for v in a["memory_settle"][-3:])
    for q in a["requests"]:
        d = q["raw_done"].split()
        assert q["produced"] == len(q["output_ids"]) == int(d[1]) == 600
        assert q["decode_file_blobs"] == int(d[12]) == 0
        assert q["process_cpu_s"] > 0 and q["decode_ms"] > 0 and q["wall_s"] > 0
for c in x["comparisons"]:
    arms = {a["arm"]: a for a in x["arms"] if a["pair"] == c["pair"]}
    history = True
    checks = []
    for i, (on, auto) in enumerate(
        zip(arms["on"]["requests"], arms["auto"]["requests"])
    ):
        same = on["output_ids"] == auto["output_ids"] and (
            on["draft_accepted"],
            on["draft_offered"],
        ) == (auto["draft_accepted"], auto["draft_offered"])
        if i:
            q = compare(on, auto, 600)
            if not history:
                q["reasons"].append("preceding output or draft work differed")
            if (on["draft_accepted"], on["draft_offered"]) != (
                auto["draft_accepted"],
                auto["draft_offered"],
            ):
                q["reasons"].append("different MTP draft work")
            q["eligible"] = not q["reasons"]
            checks.append(q)
        history = history and same
    assert checks == c["requests"]
expected = json.loads((ROOT / "data/analysis.json").read_text())
actual = analyze(ROOT / "data/results.json")


def equivalent(a, b):
    if isinstance(a, float):
        return isinstance(b, (float, int)) and math.isclose(
            a, b, rel_tol=1e-10, abs_tol=1e-10
        )
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equivalent(v, b[k]) for k, v in a.items())
    if isinstance(a, list):
        return len(a) == len(b) and all(equivalent(v, w) for v, w in zip(a, b))
    return a == b


assert equivalent(expected, actual)
assert actual["summary"]["english"]["eligible_pairs"] == 11
assert actual["summary"]["code"]["eligible_pairs"] == 4
assert actual["summary"]["english"]["tier_counts_equal_pairs"] == 11
assert len(list(csv.DictReader((ROOT / "data/requests.csv").open()))) == 84
old = json.loads((ROOT / "data/earlier-results.json").read_text())
assert (
    old["candidate_sha256"] == x["candidate_sha256"]
    and old["limits"]["warmup_tokens"] == 150
)
print(
    "Verified: 28 starts, 84 requests, 50,400 tokens; every eligibility decision and published statistic."
)
