"""Run the recorded schedule on an idle GPU, managing only its benchmark children."""

import argparse
import json
import sys
import time
from pathlib import Path
import psutil
from run_ab import Engine, compare, digest

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fixtures", default=str(ROOT / "data/fixtures.json"))
    parser.add_argument("--schedule", default=str(ROOT / "data/schedule.json"))
    parser.add_argument("--total-seconds", type=int, default=5400)
    args = parser.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    fixtures = json.loads(Path(args.fixtures).read_text(encoding="utf-8"))
    schedule = json.loads(Path(args.schedule).read_text(encoding="utf-8"))
    if [p["name"] for p in fixtures] != ["warmup", "english", "code"]:
        parser.error("fixtures must contain warmup, english and code, in that order")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    record = dict(
        completed=False,
        restored=None,
        arms=[],
        comparisons=[],
        schedule=schedule,
        candidate_sha256=digest(cfg["exe"]),
        test_args=cfg["args"],
        fixture_sha256=digest(args.fixtures),
        config_sha256=digest(args.config),
    )

    def save():
        (out / "results.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )

    save()
    deadline = time.monotonic() + args.total_seconds
    try:
        for pair, item in enumerate(schedule):
            pair_results = {}
            for arm in item["order"]:
                folder = out / f"{pair:02d}-{arm}"
                folder.mkdir()
                actual = "on" if item["kind"] == "aa" else arm
                entry = dict(
                    pair=pair,
                    kind=item["kind"],
                    arm=arm,
                    engine_arm=actual,
                    requests=[],
                )
                record["arms"].append(entry)
                until = min(deadline, time.monotonic() + 90)
                stable = 0
                entry["memory_settle"] = []
                while time.monotonic() < until:
                    available = psutil.virtual_memory().available
                    entry["memory_settle"].append(dict(available_bytes=available))
                    stable = (
                        stable + 1
                        if available >= cfg.get("min_available_gib", 20) * 1024**3
                        else 0
                    )
                    if stable >= 3:
                        break
                    time.sleep(2)
                if stable < 3:
                    raise RuntimeError(
                        "available RAM did not reach the configured admission threshold"
                    )
                started = time.monotonic()
                engine = Engine(cfg, actual, folder, deadline)
                try:
                    entry["startup_s"] = time.monotonic() - started
                    entry["info"] = [
                        v
                        for v in (folder / "stdout.log").read_text().splitlines()
                        if v.startswith("INFO ")
                    ]
                    if not entry["info"]:
                        raise RuntimeError("missing startup INFO")
                    plan = dict(
                        v.split("=", 1)
                        for v in entry["info"][-1].split()[1:]
                        if "=" in v
                    )
                    if any(
                        plan.get(k) != str(v) for k, v in cfg["reference_plan"].items()
                    ):
                        raise RuntimeError(
                            "startup allocation/execution plan differs from reference_plan"
                        )
                    proc = psutil.Process(engine.p.pid)
                    for prompt in fixtures:
                        before = proc.cpu_times()
                        memory = proc.memory_info()
                        available = psutil.virtual_memory().available
                        host_before = psutil.cpu_times()._asdict()
                        result = engine.generate(prompt, 600, 1922, 180)
                        host_after = psutil.cpu_times()._asdict()
                        after = proc.cpu_times()
                        result.update(
                            process_cpu_s=after.user
                            + after.system
                            - before.user
                            - before.system,
                            memory_before=memory._asdict(),
                            memory_after=proc.memory_info()._asdict(),
                            system_available_before=available,
                            system_available_after=psutil.virtual_memory().available,
                            host_cpu_before=host_before,
                            host_cpu_after=host_after,
                        )
                        entry["requests"].append(result)
                        save()
                        if result["decode_file_blobs"] or result["finish"] not in (
                            "length",
                            "stop",
                        ):
                            raise RuntimeError(
                                "expert-file read or abnormal generation finish"
                            )
                finally:
                    engine.close()
                pair_results[arm] = entry["requests"]
            history_ok = True
            comparisons = []
            for i, (x, y) in enumerate(zip(pair_results["on"], pair_results["auto"])):
                same = x["output_ids"] == y["output_ids"] and (
                    x["draft_accepted"],
                    x["draft_offered"],
                ) == (y["draft_accepted"], y["draft_offered"])
                if i:
                    c = compare(x, y, 600)
                    if not history_ok:
                        c["reasons"].append("preceding output or draft work differed")
                    if (x["draft_accepted"], x["draft_offered"]) != (
                        y["draft_accepted"],
                        y["draft_offered"],
                    ):
                        c["reasons"].append("different MTP draft work")
                    c["eligible"] = not c["reasons"]
                    comparisons.append(c)
                history_ok = history_ok and same
            record["comparisons"].append(
                dict(
                    pair=pair,
                    kind=item["kind"],
                    order=item["order"],
                    requests=comparisons,
                )
            )
            save()
        record["completed"] = True
    except BaseException as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        save()
    return 0 if record["completed"] else 1


if __name__ == "__main__":
    sys.exit(main())
