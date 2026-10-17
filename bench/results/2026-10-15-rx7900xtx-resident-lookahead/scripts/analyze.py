import argparse, json, math, random, statistics
from pathlib import Path


def interval(values, transform):
    if not values:
        return None
    if len(values) == 1:
        return {"estimate_pct": transform(values[0]), "ci95_pct": None, "pairs": 1}
    rng = random.Random(1922476)
    n = len(values)
    draws = sorted(
        transform(sum(rng.choice(values) for _ in range(n)) / n) for _ in range(50000)
    )
    return {
        "estimate_pct": transform(statistics.mean(values)),
        "ci95_pct": [draws[1249], draws[48749]],
        "pairs": n,
    }


def analyze(path):
    r = json.loads(Path(path).read_text())
    rows = []
    for block in r.get("comparisons", []):
        arms = {a["arm"]: a for a in r["arms"] if a["pair"] == block["pair"]}
        for c in block["requests"]:
            x = next(z for z in arms["on"]["requests"] if z["name"] == c["name"])
            y = next(z for z in arms["auto"]["requests"] if z["name"] == c["name"])

            def tiers(q):
                d = q["raw_done"].split()
                return {
                    name: int(d[i])
                    for name, i in [
                        ("gpu_hits", 9),
                        ("lookup_entries", 10),
                        ("ram_blobs", 11),
                        ("offloaded", 15),
                    ]
                }

            def busy(q):
                return sum(
                    q["host_cpu_after"][k] - q["host_cpu_before"][k]
                    for k in ("user", "system")
                )

            def faults(q):
                return q["memory_after"].get("num_page_faults", 0) - q[
                    "memory_before"
                ].get("num_page_faults", 0)

            rows.append(
                {
                    "pair": block["pair"],
                    "kind": block["kind"],
                    "order": block["order"],
                    "name": c["name"],
                    "eligible": c["eligible"],
                    "reasons": c["reasons"],
                    "tier_counts_on": tiers(x),
                    "tier_counts_auto": tiers(y),
                    "tier_counts_equal": tiers(x) == tiers(y),
                    "on_ms": x["decode_ms"],
                    "auto_ms": y["decode_ms"],
                    "on_cpu_s": x["process_cpu_s"],
                    "auto_cpu_s": y["process_cpu_s"],
                    "on_wall_s": x["wall_s"],
                    "auto_wall_s": y["wall_s"],
                    "on_tokens": x["produced"],
                    "auto_tokens": y["produced"],
                    "on_ttft_s": x["ttft_s"],
                    "auto_ttft_s": y["ttft_s"],
                    "on_prompt_ms": x["prompt_ms"],
                    "auto_prompt_ms": y["prompt_ms"],
                    "throughput_change_pct": 100
                    * (x["decode_ms"] / y["decode_ms"] - 1),
                    "all_pair_rate_change_pct": 100
                    * (
                        y["produced"]
                        * x["decode_ms"]
                        / (x["produced"] * y["decode_ms"])
                        - 1
                    )
                    if x["produced"] > 0 and y["produced"] > 0
                    else None,
                    "cpu_reduction_pct": 100
                    * (1 - y["process_cpu_s"] / x["process_cpu_s"]),
                    "wall_reduction_pct": 100 * (1 - y["wall_s"] / x["wall_s"]),
                    "on_host_cpu_s": busy(x),
                    "auto_host_cpu_s": busy(y),
                    "on_other_cpu_s_estimate": busy(x) - x["process_cpu_s"],
                    "auto_other_cpu_s_estimate": busy(y) - y["process_cpu_s"],
                    "on_page_faults": faults(x),
                    "auto_page_faults": faults(y),
                    "on_available_min": min(
                        x["system_available_before"], x["system_available_after"]
                    ),
                    "auto_available_min": min(
                        y["system_available_before"], y["system_available_after"]
                    ),
                }
            )
    summary = {}
    for name in ("english", "code"):
        allrows = [q for q in rows if q["kind"] == "ab" and q["name"] == name]
        ok = [q for q in allrows if q["eligible"]]
        logperf = [math.log(q["on_ms"] / q["auto_ms"]) for q in ok]
        logcpu = [math.log(q["on_cpu_s"] / q["auto_cpu_s"]) for q in ok]
        logwall = [math.log(q["on_wall_s"] / q["auto_wall_s"]) for q in ok]
        perf = interval(logperf, lambda x: 100 * math.expm1(x))
        classification = "insufficient pairs"
        if perf and len(ok) >= 8:
            lo, hi = perf["ci95_pct"]
            classification = (
                "gain" if lo > 0 else "loss" if hi < 0 else "direction unresolved"
            )
            if lo >= -2 and hi <= 2:
                classification += "; within practical +/-2% band"
            else:
                classification += "; outside practical band remains possible"
        summary[name] = {
            "eligible_pairs": len(ok),
            "total_ab_pairs": len(allrows),
            "primary_for_this_workload": perf,
            "cpu_reduction": interval(logcpu, lambda x: 100 * (-math.expm1(-x))),
            "wall_reduction": interval(logwall, lambda x: 100 * (-math.expm1(-x))),
            "ttft_median_ms": {
                "on": 1000 * statistics.median(q["on_ttft_s"] for q in ok),
                "auto": 1000 * statistics.median(q["auto_ttft_s"] for q in ok),
            }
            if ok
            else None,
            "median_throughput_pct": statistics.median(
                [q["throughput_change_pct"] for q in ok]
            )
            if ok
            else None,
            "faster_pairs": sum(q["throughput_change_pct"] > 0 for q in ok),
            "cpu_lower_pairs": sum(q["cpu_reduction_pct"] > 0 for q in ok),
            "tier_counts_equal_pairs": sum(q["tier_counts_equal"] for q in ok),
            "classification": classification,
            "all_pair_descriptive_geomean_pct": 100
            * math.expm1(
                statistics.mean(
                    math.log1p(q["all_pair_rate_change_pct"] / 100)
                    for q in allrows
                    if q["all_pair_rate_change_pct"] is not None
                )
            )
            if any(q["all_pair_rate_change_pct"] is not None for q in allrows)
            else None,
            "order_groups": {
                order: {
                    "n": len(g),
                    "geomean_pct": 100
                    * math.expm1(
                        statistics.mean(math.log(q["on_ms"] / q["auto_ms"]) for q in g)
                    )
                    if g
                    else None,
                }
                for order in ("on", "auto")
                for g in [[q for q in ok if q["order"][0] == order]]
            },
            "aa_controls": [q for q in rows if q["kind"] == "aa" and q["name"] == name],
        }
    return {
        "completed": r["completed"],
        "restored": r["restored"],
        "error": r.get("error"),
        "candidate_sha256": r["candidate_sha256"],
        "summary": summary,
        "rows": rows,
        "all_expert_file_reads_zero": all(
            q["decode_file_blobs"] == 0 for a in r["arms"] for q in a["requests"]
        ),
        "source": "Same final binary; fixed schedule of 12 AB and 2 AA pairs. CPU is whole request; host/page-fault fields descriptive.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("results")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    x = analyze(a.results)
    Path(a.out).write_text(json.dumps(x, indent=2))
    print(json.dumps({k: v for k, v in x.items() if k != "rows"}, indent=2))
