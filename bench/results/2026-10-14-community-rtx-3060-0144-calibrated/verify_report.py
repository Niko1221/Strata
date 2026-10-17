"""Verify the saved report and optionally unpack its complete evidence; no inference."""
import argparse
import base64
import csv
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import statistics
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
MODEL_ORDER = {"q2_0": 0, "iq2_xs": 1, "iq3_xxs": 2}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative_path(root, name):
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts and "\\" not in name
            and ":" not in name and bool(path.parts), f"Unsafe evidence path: {name}")
    target = root.joinpath(*path.parts)
    require(target.resolve().is_relative_to(root.resolve()), f"Escaping evidence path: {name}")
    return target


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_file_hashes():
    for line in (ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        actual = hashlib.sha256(relative_path(ROOT, name).read_bytes()).hexdigest()
        require(actual == expected, f"Published file hash differs: {name}")


def unpack(bundle, destination):
    require(bundle["schema"] == 1, "Unsupported evidence schema")
    destination.mkdir(parents=True, exist_ok=False)
    for section, directory in (("files", "current"), ("historical_files", "historical")):
        for name, record in bundle[section].items():
            body = base64.b64decode(record["data"], validate=True)
            require(len(body) == record["bytes"], f"Evidence size differs: {name}")
            require(hashlib.sha256(body).hexdigest() == record["sha256"], f"Evidence hash differs: {name}")
            blob = b"blob " + str(len(body)).encode() + b"\0" + body
            require(hashlib.sha1(blob).hexdigest() == record["git_blob_sha"], f"Source blob differs: {name}")
            target = relative_path(destination / directory, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)


def order(row):
    return MODEL_ORDER[row["model"]], row["context"], row["target"]


def request_row(row):
    engine = row["engine"]
    return dict(number=row["number"], model=row["model"], context_limit=row["context"],
                group=row["group"], target_input=row["target"], repeat=row["repetition"], ok=row["ok"],
                input_tokens=engine["prompt_tokens"], reused_tokens=engine["reused"],
                fresh_tokens=engine["prompt_read"], generated_tokens=engine["engine_generated"],
                prompt_ms=engine["prompt_ms"], decode_ms=engine["decode_ms"],
                pp_tok_s=(engine["prompt_tokens"] - engine["reused"]) * 1000 / engine["prompt_ms"],
                decode_tok_s=engine["decode_tok_s"], ttft_s=row["ttft_s"],
                latency_s=row["client_elapsed_s"], drafts_offered=engine["drafts_offered"],
                drafts_accepted=engine["drafts_accepted"], request_sha256=row["request_sha256"],
                finish=engine["finish"])


def exports(current, historical, summary):
    rows = read_json(current / "per-request.json")
    speed = []
    for group in sorted(summary["speed"], key=order):
        require(len(set(group["prompt_tokens"])) == 1 and group["generated_tokens"] == [512, 512, 512],
                "The published speed counts require three equal-input, 512-token runs")
        item = dict(model=group["model"], context_limit=group["context"], target_input=group["target"],
                    runs=group["successful"], input_tokens=group["prompt_tokens"][0],
                    generated_tokens=group["generated_tokens"][0])
        for metric in ("pp_tok_s", "decode_tok_s", "ttft_s", "latency_s"):
            item.update({metric + "_" + stat: group[metric][stat] for stat in ("median", "min", "max")})
        speed.append(item)
    previous = []
    for source in ("per-request.json", "long-context/per-request.json"):
        previous += [dict(row, source_file=source) for row in read_json(historical / source)]
    comparison = []
    retained = []
    for group in sorted(summary["speed"], key=order):
        key = group["model"], group["context"], group["target"]
        prior = [row for row in previous if row["group"] == "speed" and row["ok"]
                 and (row["model"], row["context"], row["target"]) == key]
        fresh = [row for row in rows if row["group"] == "speed"
                 and (row["model"], row["context"], row["target"]) == key]
        require(len(prior) == len(fresh) == 3, f"Comparison coverage differs: {key}")
        for row in sorted(prior, key=lambda row: row["repetition"]):
            matched = next(item for item in fresh if item["repetition"] == row["repetition"])
            require(row["request_sha256"] == matched["request_sha256"], f"Comparison input differs: {key}")
            retained.append(dict(source_file=row["source_file"], **request_row(row)))
        old_pp = statistics.median(row["pp_tok_s"] for row in prior)
        old_decode = statistics.median(row["decode_tok_s"] for row in prior)
        new_decode = group["decode_tok_s"]["median"]
        comparison.append(dict(model=key[0], context_limit=key[1], target_input=key[2], runs=3,
                               old_pp_tok_s=old_pp, new_pp_tok_s=group["pp_tok_s"]["median"],
                               old_decode_tok_s=old_decode, new_decode_tok_s=new_decode,
                               decode_change_percent=(new_decode / old_decode - 1) * 100,
                               all_request_hashes_match=True))
    resources = []
    for session, data in sorted(summary["resources"].items(),
                                key=lambda item: (MODEL_ORDER[item[0].split("-")[0]], item[0])):
        resources.append(dict(session=session, min_available_ram_gib=data["available_ram_bytes"]["min"] / 1024**3,
                              peak_commit_fraction=data["commit_fraction"]["max"],
                              peak_vram_gib=data["vram_mib"]["max"] / 1024,
                              peak_gpu_temperature_c=data["temperature_c"]["max"]))
    return {"requests.csv": [request_row(row) for row in rows], "speed.csv": speed,
            "comparison-0143.csv": comparison, "historical-requests.csv": retained,
            "resources.csv": resources}


def check_csv(name, rows):
    with (ROOT / name).open(encoding="utf-8", newline="") as stream:
        saved = list(csv.DictReader(stream))
    expected = [{key: "" if value is None else str(value) for key, value in row.items()} for row in rows]
    require(saved == expected, f"CSV differs from complete records: {name}")


def check_sources(current):
    for name in ("protocol.json", "hardware.json", "model-provenance.json", "runtime-artifacts.json"):
        require((ROOT / name).read_bytes() == (current / name).read_bytes(), f"Published source differs: {name}")
    for name in ("community_bench_0144.py", "reproduce.py", "analyze.py", "validate.py"):
        require((ROOT / "harness" / name).read_bytes() == (current / name).read_bytes(), f"Harness differs: {name}")
    for model in MODEL_ORDER:
        name = f"fixed-configs/{model}.json"
        require((ROOT / name).read_bytes() == (current / name).read_bytes(), f"Fixed configuration differs: {model}")
    identity = read_json(current / "runner-identity.json")
    measured = gzip.decompress((current / identity["measured_runner_file"]).read_bytes())
    require(hashlib.sha256(measured).hexdigest() == identity["runner_sha256"], "Measured runner hash differs")
    require(hashlib.sha256((current / identity["published_runner_file"]).read_bytes()).hexdigest()
            == identity["published_runner_sha256"], "Published runner hash differs")
    require(hashlib.sha256((current / "previous-runner.py").read_bytes()).hexdigest()
            == identity["base_runner_sha256"], "Base runner hash differs")
    provenance = read_json(current / "calibration-provenance.json")
    require(hashlib.sha256((current / "calibrate-official.py").read_bytes()).hexdigest()
            == provenance["iq2_xs"]["calibrator_sha256"], "Official tuner hash differs")
    for row in read_json(current / "per-request.json"):
        name = row["output_file"].removesuffix(".response.json") + ".metrics.json.gz"
        metrics = json.loads(gzip.decompress((current / "requests" / name).read_bytes()))
        require(metrics["requests"][0] == row["engine"], f"Saved engine metrics differ: {row['number']}")
    settings = {model: read_json(current / f"{model}-calibration-result.json")["settings"] for model in MODEL_ORDER}
    require(read_json(ROOT / "calibration-settings.json") == settings, "Calibration export differs")
    for model, selected in settings.items():
        flags = read_json(current / "fixed-configs" / f"{model}.json")["args"]
        for flag, value in selected.items():
            require(flags[flags.index(flag) + 1] == str(value), f"Selected setting differs: {model} {flag}")
        for context in (131072, 262144):
            loaded = read_json(current / "servers" / f"{model}-{context}-loaded.json")["metrics"]["engine"]
            require(float(loaded["pcie_frac"]) == float(selected["--pcie-frac"])
                    and float(loaded["spec_min_p"]) == float(selected["--spec-min-p"]),
                    f"Loaded calibration values differ: {model}")


def verify(bundle, destination):
    unpack(bundle, destination)
    current = destination / "current"
    check_sources(current)
    validation = subprocess.run([sys.executable, str(current / "validate.py"), str(current)],
                                capture_output=True, text=True, encoding="utf-8", check=True)
    analysis = destination / "analysis"
    subprocess.run([sys.executable, str(current / "analyze.py"), str(current), "--out", str(analysis)],
                   capture_output=True, text=True, encoding="utf-8", check=True)
    summary = read_json(analysis / "summary.json")
    require(summary == read_json(current / "summary.json"), "Recomputed summary differs")
    for name, rows in exports(current, destination / "historical", summary).items():
        check_csv(name, rows)
    result = json.loads(validation.stdout)
    result.update(comparison_requests=36, complete_metrics_checked=70, csv_and_summary_verified=True,
                  source_commit=bundle["source_commit"], historical_source_commit=bundle["historical_source_commit"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", type=Path, help="New directory for verified current/historical evidence")
    args = parser.parse_args()
    if args.extract:
        require(not args.extract.exists(), "Preserve existing files; extraction destination must not exist")
    check_file_hashes()
    bundle = json.loads(gzip.decompress((ROOT / "measurements.json.gz").read_bytes()))
    with tempfile.TemporaryDirectory(prefix="strata-report-") as work:
        result = verify(bundle, Path(work) / "evidence")
    if args.extract:
        unpack(bundle, args.extract.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
