"""Explicit idle original-profile component upgrade; default is an inert plan.

Only supported import/use APIs are called. The persistent app and models are
never started, stopped or adopted by this controller's short-lived owned Job.
"""

from __future__ import annotations

import argparse
import http.client
import importlib.util
import json
import logging
from pathlib import Path
import re
import sys
import time
from typing import Callable

sys.dont_write_bytecode = True
HERE = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location("publication_acceptance", HERE / "acceptance.py")
acceptance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acceptance)
delivery, feed, native, release = (
    acceptance.delivery,
    acceptance.feed,
    acceptance.native,
    acceptance.release,
)
normal = release.module("publication_normal", HERE / "normal-start.py")
normal.delivery, normal.native, normal.release = delivery, native, release
require, artifact = release.require, acceptance.artifact
HELPERS = (
    "original-engine.py",
    "normal-start.py",
    "acceptance.py",
    "delivery.py",
    "components-feed.py",
    "native-input.py",
    "release.py",
)
SCHEMA = "strata.original-engine.v1"
INPUTS = ("install", "component", "original")
FILETIME_EPOCH = 116444736000000000


def option(command: list[str], name: str) -> str:
    """Read exactly one bound option, including argparse's equals form."""
    values = [command[i + 1] for i, v in enumerate(command[:-1]) if v == name]
    values += [v.split("=", 1)[1] for v in command[3:] if v.startswith(name + "=")]
    require(len(values) == 1, "Missing or duplicate bound option: " + name)
    return values[0]


def owners(command: list[str]) -> dict:
    return {
        name: option(command, "--" + name)
        for name in ("shell-pid", "shell-created", "host-pid", "host-created", "port")
    }


def helper_inputs(names: tuple[str, ...]) -> dict:
    return {name: artifact(HERE / name) for name in names}


def proof(path: Path, expected: str, phase: str, schema: str, commit: str) -> dict:
    """Shared small post-drain envelope, retaining the existing raw Job contract."""
    record = delivery.pinned(path, expected)
    require(
        path.name == phase + ".json" and record.get("schema") == schema,
        "Wrong typed publication proof",
    )
    raw = delivery.receipt(path.parent / "receipt.json", record["receiptSHA256"], phase, commit)
    require(
        {
            k: v
            for k, v in record.items()
            if k not in ("receiptSHA256", "ownedJobSHA256", "workerLogSHA256", "drainedAtFiletime")
        }
        == raw,
        "Typed proof differs from drained receipt",
    )
    require("ramPolicy" in raw, "Current operation provenance required")
    artifact(path.parent / "owned-job.json", record["ownedJobSHA256"])
    artifact(path.parent / "worker.log", record["workerLogSHA256"])
    job = release.load(path.parent / "owned-job.json")
    acceptance.successful_job(job, raw, path.parent)
    require(
        type(record.get("drainedAtFiletime")) is int
        and record["drainedAtFiletime"] >= job["root_creation_filetime"],
        "Missing post-drain observation time",
    )
    script = "publish.py" if phase.startswith("publication-") else phase + ".py"
    require(
        Path(job["operationExecution"]["command"][2]) == HERE / script, "Wrong operation adapter"
    )
    acceptance.check_artifacts(record["inputs"])
    acceptance.check_artifacts(record["results"])
    artifact(path, expected)
    return record


def finalize(
    work: Path, phase: str, schema: str, commit: str, validate: Callable[[Path, str, str], dict]
) -> None:
    raw = release.load(work / "receipt.json")
    acceptance.successful_job(release.load(work / "owned-job.json"), raw, work)
    require(raw.get("schema") == schema, "Missing typed worker result")
    target = work / (phase + ".json")
    release.save(
        target,
        {
            **raw,
            "receiptSHA256": release.sha(work / "receipt.json"),
            "ownedJobSHA256": release.sha(work / "owned-job.json"),
            "workerLogSHA256": release.sha(work / "worker.log"),
            "drainedAtFiletime": time.time_ns() // 100 + FILETIME_EPOCH,
        },
    )
    validate(target, release.sha(target), commit)


def require_helpers(record: dict, names: tuple[str, ...]) -> None:
    for name in names:
        row = record["inputs"][name]
        require(Path(row["path"]) == HERE / name, "Foreign helper input")
        artifact(HERE / name, row["sha256"])


def check_idle(base: str) -> None:
    for route in ("/api/native", "/api/runtime"):
        value = normal.read_api(base, route)
        require(value.get("instances") == [], "Live runtime requires separate owner handling")
        require(
            not any(j.get("state") in ("running", "cancelling") for j in value.get("jobs", [])),
            "Active native operation requires separate handling",
        )
    require(normal.read_api(base, "/api/chat/active").get("turns") == [], "Active turn")
    engine = normal.read_api(base, "/api/health")["engine"]
    require(
        engine.get("status") == "stopped" and engine.get("running") is False,
        "Model must be unloaded",
    )


def stable(args: argparse.Namespace, original: dict) -> None:
    base = f"http://127.0.0.1:{args.port}"
    normal.owned_processes(args)
    require(
        normal.read_api(base, "/api/app/build") == {"host": "0.6.0", "shell": "0.6.0"},
        "Wrong persistent installed build",
    )
    require(
        normal.profile_identity(args, base) == original["profileIdentity"],
        "Original profile changed",
    )
    require(
        normal.choices_digest(normal.choices(base)) == original["choicesSHA256"],
        "Original choices or skills changed",
    )
    check_idle(base)
    normal.owned_processes(args)


def call(
    args: argparse.Namespace, route: str, *, archive: Path | None = None, body: dict | None = None
) -> dict:
    """Stream only the verified archive to a fixed owned loopback port, without redirects."""
    require(route in ("/api/engine/from-file", "/api/engines/use"), "Unsupported mutation")
    connection = http.client.HTTPConnection("127.0.0.1", args.port, timeout=180)
    headers = {"Origin": f"http://127.0.0.1:{args.port}"}
    try:
        if archive is not None:
            require(
                route == "/api/engine/from-file"
                and re.fullmatch(r"[A-Za-z0-9._-]+\.tar\.gz", archive.name),
                "Safe verified native archive required",
            )
            size = archive.stat().st_size
            headers.update(
                {
                    "Content-Length": str(size),
                    "x-file-name": archive.name,
                    "Content-Type": "application/octet-stream",
                }
            )
            connection.putrequest("POST", route)
            for name, value in headers.items():
                connection.putheader(name, value)
            connection.endheaders()
            with archive.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    connection.send(block)
        else:
            require(route == "/api/engines/use", "Wrong JSON mutation")
            raw = json.dumps(body).encode()
            headers.update({"Content-Type": "application/json", "Content-Length": str(len(raw))})
            connection.request("POST", route, raw, headers)
        response = connection.getresponse()
        require(response.status == (202 if archive else 200), "Component API refused operation")
        raw = response.read(2 * 1024 * 1024 + 1)
        require(len(raw) <= 2 * 1024 * 1024, "Oversized API result")
        return json.loads(raw)
    finally:
        connection.close()


def wait_job(base: str, job_id: str, *, timeout: float = 300) -> dict:
    require(isinstance(job_id, str) and bool(job_id), "Import must return a job ID")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = normal.read_api(base, "/api/native")
        matches = [j for j in state.get("jobs", []) if j.get("id") == job_id]
        require(
            len(matches) == 1 and matches[0].get("kind") == "install-engine-file",
            "Import job replaced/missing",
        )
        job = matches[0]
        require(job.get("state") in ("running", "done"), "Import failed or cancelled")
        if job["state"] == "done":
            return {k: job.get(k) for k in ("id", "kind", "state", "startedAt", "finishedAt")}
        time.sleep(0.5)
    raise RuntimeError("Exact import job deadline exceeded; app left running")


def entries(listing: dict) -> list[dict]:
    require(
        listing.get("available") is True and listing.get("serving") is False,
        "Engine library unavailable/busy",
    )
    return [e for f in listing["families"] for e in f["engines"]]


def installed_component(path: Path, signed: Path, files: dict, components: Path) -> dict:
    require(
        path == components / "engine" / release.NAME, "Installed component escaped original profile"
    )
    require(
        release.inventory(signed) == files and release.inventory(path) == files,
        "Installed component bytes differ",
    )
    sys.path.insert(0, str(native.PRIMARY))
    from strata_control import engine_component, signing

    item = engine_component.read_manifest(path)
    manifest = release.load(path / "engine-component.json")
    signing.verify_bytes(
        signing.load_trusted_keys(environ={}),
        engine_component.signing_payload(manifest),
        manifest.get("signature"),
    )
    engine_component.verify(item, full=True)
    return {
        "path": str(path),
        "manifestSHA256": release.sha(path / "engine-component.json"),
        "files": files,
        "version": item.version,
        "buildId": item.build_id,
    }


def verified_selection(
    listing: dict,
    signed: Path,
    files: dict,
    components: Path,
    commit: str,
    *,
    active: bool = False,
) -> tuple[dict, dict]:
    """Verify the imported component before changing the persistent selection."""
    selected = [
        row
        for row in entries(listing)
        if row.get("family") == "qwen" and row.get("folder") == release.NAME
    ]
    require(len(selected) == 1, "Imported component absent/ambiguous")
    row = selected[0]
    require(
        all(row.get(k) is True for k in ("signed", "fits"))
        and (not active or row.get("active") is True)
        and row.get("version") == "0.6.0"
        and row.get("backend") == "cuda"
        and row.get("variant") == "cuda13"
        and row.get("platform") == "windows-x86_64"
        and row.get("build_id") == commit + "-hakua-runtimes-v1",
        "Wrong imported native identity",
    )
    identity = installed_component(Path(row["path"]), signed, files, components)
    return row, identity


def phase(args: argparse.Namespace, work: Path) -> dict:
    original = normal.original_baseline(args)
    installed = delivery.prior(args, "install")
    require(
        installed.get("installed") is True and installed.get("exactPayloadMatch") is True,
        "Protected installed payload required",
    )
    require(delivery.payload(delivery.APP) == installed["payload"], "Installed app bytes changed")
    component = feed.receipt_assets(
        args.component_receipt, args.component_receipt_sha256, "component", args.reviewed_commit
    )
    component_raw = release.load(args.component_receipt)
    signed = Path(component_raw["signedComponent"])
    archive = Path(component["assets"]) / component["entry"]["url"]
    base = f"http://127.0.0.1:{args.port}"
    stable(args, original)
    components = Path(original["profileIdentity"]["effectiveHost"]["components"]["physical"])
    before = entries(normal.read_api(base, "/api/engines"))
    old = {}
    for row in before:
        folder = Path(row["path"])
        require(folder.is_relative_to(components), "Foreign existing component root")
        old[str(folder)] = release.inventory(folder)
    stable(args, original)
    release.save(
        work / "import-intent.json",
        {
            "archive": artifact(archive, component["entry"]["sha256"]),
            "contentLength": archive.stat().st_size,
            "fileName": archive.name,
            "origin": base,
            "owners": owners([sys.executable, "-B", *sys.argv]),
        },
    )
    operation = call(args, "/api/engine/from-file", archive=archive)
    release.save(
        work / "import-response.json", {k: operation.get(k) for k in ("id", "kind", "state")}
    )
    require(operation.get("kind") == "install-engine-file", "Wrong returned import operation")
    job = wait_job(base, operation.get("id"))
    stable(args, original)
    _, before_selection = verified_selection(
        normal.read_api(base, "/api/engines"),
        signed,
        component_raw["files"],
        components,
        args.reviewed_commit,
    )
    stable(args, original)
    call(args, "/api/engines/use", body={"family": "qwen", "folder": release.NAME})
    stable(args, original)
    row, identity = verified_selection(
        normal.read_api(base, "/api/engines"),
        signed,
        component_raw["files"],
        components,
        args.reviewed_commit,
        active=True,
    )
    require(identity == before_selection, "Imported component changed during selection")
    require(all(release.inventory(Path(p)) == v for p, v in old.items()), "Old components changed")
    require(delivery.payload(delivery.APP) == installed["payload"], "App payload changed")
    stable(args, original)
    inputs = helper_inputs(HELPERS)
    inputs.update(
        {
            n: artifact(getattr(args, n + "_receipt"), getattr(args, n + "_receipt_sha256"))
            for n in INPUTS
        }
    )
    inputs["archive"] = artifact(archive, component["entry"]["sha256"])
    observation = {
        "owners": owners([sys.executable, "-B", *sys.argv]),
        "job": job,
        "selected": {
            k: row[k]
            for k in (
                "family",
                "folder",
                "version",
                "backend",
                "variant",
                "platform",
                "signed",
                "fits",
                "active",
                "build_id",
            )
        },
        "installed": identity,
        "oldComponents": old,
        "profileIdentitySHA256": original["profileIdentitySHA256"],
        "choicesSHA256": args.choices_sha256,
        "modelUnloaded": True,
        "activeTurns": 0,
    }
    release.save(work / "observation.json", observation)
    return {
        "schema": SCHEMA,
        "inputs": inputs,
        "results": {
            "observation": artifact(work / "observation.json"),
            "importIntent": artifact(work / "import-intent.json"),
            "importResponse": artifact(work / "import-response.json"),
        },
        "installerSHA256": installed["installerSHA256"],
        "componentArchiveSHA256": component["entry"]["sha256"],
        "profileIdentitySHA256": original["profileIdentitySHA256"],
        "choicesSHA256": args.choices_sha256,
        "modelUnloaded": True,
        "persistentAppStartedByThisController": False,
    }


def validate_original_engine(path: Path, expected: str, commit: str) -> dict:
    record = proof(path, expected, "original-engine", SCHEMA, commit)
    require_helpers(record, HELPERS)
    job = release.load(path.parent / "owned-job.json")
    command = job["operationExecution"]["command"]
    observation_path = acceptance.evidence(
        record, "results", "observation", path.parent / "observation.json"
    )
    observation = release.load(observation_path)
    require(observation["owners"] == owners(command), "Observed owners differ from command")
    for n in INPUTS:
        row = record["inputs"][n]
        require(
            option(command, "--" + n + "-receipt") == row["path"]
            and option(command, "--" + n + "-receipt-sha256") == row["sha256"],
            "Receipt not bound to command",
        )
    install = record["inputs"]["install"]
    installed = delivery.receipt(Path(install["path"]), install["sha256"], "install", commit)
    comp = record["inputs"]["component"]
    component = feed.receipt_assets(Path(comp["path"]), comp["sha256"], "component", commit)
    raw = release.load(Path(comp["path"]))
    orig = record["inputs"]["original"]
    baseline_args = argparse.Namespace(
        reviewed_commit=commit,
        original_receipt=Path(orig["path"]),
        original_receipt_sha256=orig["sha256"],
        choices_sha256=record["choicesSHA256"],
    )
    original = normal.original_baseline(baseline_args)
    require(
        record["installerSHA256"] == installed["installerSHA256"]
        and installed.get("installed") is True
        and installed.get("exactPayloadMatch") is True
        and delivery.payload(delivery.APP) == installed["payload"],
        "Wrong installed source",
    )
    require(
        record["componentArchiveSHA256"] == component["entry"]["sha256"]
        and record["inputs"]["archive"]
        == artifact(Path(component["assets"]) / component["entry"]["url"]),
        "Archive proof mismatch",
    )
    require(
        observation["job"].get("kind") == "install-engine-file"
        and observation["job"].get("state") == "done"
        and observation["job"].get("id"),
        "No successful exact import job",
    )
    intent = release.load(
        acceptance.evidence(record, "results", "importIntent", path.parent / "import-intent.json")
    )
    response = release.load(
        acceptance.evidence(
            record, "results", "importResponse", path.parent / "import-response.json"
        )
    )
    archive_path = Path(component["assets"]) / component["entry"]["url"]
    require(
        intent
        == {
            "archive": record["inputs"]["archive"],
            "contentLength": archive_path.stat().st_size,
            "fileName": archive_path.name,
            "origin": "http://127.0.0.1:" + owners(command)["port"],
            "owners": owners(command),
        }
        and response.get("id") == observation["job"]["id"]
        and response.get("kind") == "install-engine-file"
        and response.get("state") in ("running", "done"),
        "Import request/response identity differs",
    )
    selected = observation["selected"]
    require(
        selected
        == {
            "family": "qwen",
            "folder": release.NAME,
            "version": "0.6.0",
            "backend": "cuda",
            "variant": "cuda13",
            "platform": "windows-x86_64",
            "signed": True,
            "fits": True,
            "active": True,
            "build_id": commit + "-hakua-runtimes-v1",
        },
        "Wrong selected component proof",
    )
    components = Path(original["profileIdentity"]["effectiveHost"]["components"]["physical"])
    require(
        observation["installed"]
        == installed_component(
            Path(observation["installed"]["path"]),
            Path(raw["signedComponent"]),
            raw["files"],
            components,
        ),
        "Installed identity changed",
    )
    require(
        all(
            Path(p).is_relative_to(components) and release.inventory(Path(p)) == rows
            for p, rows in observation["oldComponents"].items()
        ),
        "Old component preservation differs",
    )
    require(
        record["profileIdentitySHA256"]
        == observation["profileIdentitySHA256"]
        == original["profileIdentitySHA256"]
        and record["choicesSHA256"]
        == observation["choicesSHA256"]
        == option(command, "--choices-sha256")
        and record.get("modelUnloaded") is observation.get("modelUnloaded") is True
        and observation.get("activeTurns") == 0
        and record.get("persistentAppStartedByThisController") is False,
        "Original identity or final idle proof differs",
    )
    return record


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cli = native.parser(__doc__)
    for name in INPUTS:
        cli.add_argument("--" + name + "-receipt", type=Path)
        cli.add_argument("--" + name + "-receipt-sha256")
    cli.add_argument("--choices-sha256")
    for name, kind in (
        ("shell-pid", int),
        ("shell-created", float),
        ("host-pid", int),
        ("host-created", float),
        ("port", int),
    ):
        cli.add_argument("--" + name, type=kind)
    args = cli.parse_args(argv)
    args.phase = "original-engine"
    if args.execute:
        require(
            "--attempt" in argv and re.fullmatch(r"[a-f0-9]{64}", args.choices_sha256 or ""),
            "Explicit attempt and choices required",
        )
        require(
            all(
                v is not None and v > 0
                for v in (args.shell_pid, args.host_pid, args.shell_created, args.host_created)
            )
            and args.port is not None
            and 1 <= args.port <= 65535,
            "Explicit owner identity required",
        )
        for name in INPUTS:
            require(
                getattr(args, name + "_receipt") is not None
                and re.fullmatch(r"[a-f0-9]{64}", getattr(args, name + "_receipt_sha256") or ""),
                "Pinned input required",
            )
    result = native.run_controller(args, HERE / "original-engine.py", phase, argv)
    if args.execute and not args.worker:
        finalize(
            release.OUT / (args.phase + "-" + args.attempt),
            args.phase,
            SCHEMA,
            args.reviewed_commit,
            validate_original_engine,
        )
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        raise SystemExit(main())
    except Exception as error:
        logging.error(
            "Original-engine refused (%s); raw error suppressed; app left running",
            type(error).__name__,
        )
        raise SystemExit(1) from None
