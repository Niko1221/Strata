"""Validate a candidate expert profile on independent prompts and roll back regressions.

    python tools/validate_expert_profile.py --profile data/expert-profile.bin \
        --candidate data/expert-profile-candidate.bin --tasks tasks.jsonl \
        --server http://127.0.0.1:8080/v1 --model MODEL_ID --report report.json

The current server must already use --profile. Candidate evaluation happens after
copying --candidate to --profile and running --restart-command, when supplied.
The backup is restored when any task group loses more than --tolerance of its
median decode throughput. No default model or engine setting is changed.
"""
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import urllib.request
from collections import defaultdict
from pathlib import Path


def read_tasks(path):
    tasks = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"task {number}: prompt must be a nonempty string")
            group = row.get("group", "general")
            if not isinstance(group, str) or not group:
                raise ValueError(f"task {number}: group must be a nonempty string")
            max_tokens = row.get("max_tokens", 128)
            if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
                raise ValueError(f"task {number}: max_tokens must be a positive integer")
            tasks.append({"prompt": prompt, "group": group, "max_tokens": max_tokens})
    if not tasks:
        raise ValueError("no validation tasks")
    return tasks


def median(values):
    values = sorted(values)
    middle = len(values) // 2
    if not values:
        raise ValueError("median of an empty sequence")
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def group_medians(results):
    groups = defaultdict(list)
    for result in results:
        groups[result["group"]].append(result["timings"]["predicted_per_second"])
    return {group: median(values) for group, values in groups.items()}


def regressions(baseline, candidate, tolerance):
    failed = []
    for group, old in baseline.items():
        new = candidate[group]
        loss = (old - new) / old if old else 0.0
        if loss > tolerance:
            failed.append({"group": group, "baseline": old, "candidate": new, "loss": loss})
    return failed


def evaluate_http(server, model, tasks):
    results = []
    for task in tasks:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": task["prompt"]}],
            "max_tokens": task["max_tokens"],
            "temperature": 0,
        }
        request = urllib.request.Request(
            server.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            answer = json.load(response)
        results.append({"group": task["group"], "timings": answer.get("timings") or {}})
    return results


def restart(command):
    if callable(command):
        command()
    elif command:
        subprocess.run(shlex.split(command), check=True)


def validate(profile, candidate, tasks, evaluate, restart_command=None, tolerance=0.10):
    profile, candidate = Path(profile), Path(candidate)
    backup = profile.with_name(profile.name + ".validation-backup")
    if backup.exists():
        raise FileExistsError(backup)
    baseline = evaluate(tasks)
    baseline_medians = group_medians(baseline)
    shutil.copyfile(profile, backup)
    try:
        shutil.copyfile(candidate, profile)
        restart(restart_command)
        current = evaluate(tasks)
        current_medians = group_medians(current)
        failed = regressions(baseline_medians, current_medians, tolerance)
        if failed:
            shutil.copyfile(backup, profile)
            restart(restart_command)
            status = "rolled_back"
        else:
            status = "accepted"
    except Exception:
        shutil.copyfile(backup, profile)
        restart(restart_command)
        raise
    return {
        "status": status,
        "baseline": baseline_medians,
        "candidate": current_medians,
        "regressions": failed,
        "backup": str(backup),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--profile", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--restart-command")
    parser.add_argument("--tolerance", type=float, default=0.10)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    if not 0 <= args.tolerance < 1:
        parser.error("--tolerance must be between 0 and 1")
    tasks = read_tasks(args.tasks)
    evaluate = lambda rows: evaluate_http(args.server, args.model, rows)
    report = validate(args.profile, args.candidate, tasks, evaluate, args.restart_command, args.tolerance)
    report["server"] = args.server
    report["model"] = args.model
    report["tolerance"] = args.tolerance
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
