"""Hakua 0.6.0 guarded preparation. Default is a read-only plan; no downloads."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace
from typing import Any
import uuid

HERE = Path(__file__).absolute().parent
ROOT = HERE.parents[1]
CHECKOUT_BRANCHES = {
    Path("C:/Users/downl/Documents/ChatGPT/Strata"): "main",
    Path("H:/Strata/worktrees/upstream-sync-20261116"): "codex/upstream-sync-20261116",
}
OUT = Path("H:/Strata/releases/stable-060-20261116-codex")
RES = ROOT / "apps/desktop/src-tauri/resources"
PRIMARY_RES = Path("C:/Users/downl/Documents/ChatGPT/Strata/apps/desktop/src-tauri/resources")
NODE = RES / "node/node.exe"
VERSION = "0.6.0"
NAME = VERSION + "-windows-x86_64-cuda-cuda13"
GIB = 1024**3
LOG = logging.getLogger("release060")


def require(condition: Any, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def no_links(path: Path) -> None:
    for part in (path, *path.parents):
        if part.exists() or part.is_symlink():
            require(
                not part.is_symlink() and not getattr(part, "is_junction", lambda: False)(),
                "Linked path refused",
            )
            require(not (getattr(part.lstat(), "st_file_attributes", 0) & 0x400), "Reparse path refused")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path: Path) -> Any:
    no_links(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value: Any) -> None:
    no_links(path)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def inventory(root: Path) -> dict[str, str]:
    no_links(root)
    require(root.is_dir(), "Inventory root missing")
    result: dict[str, str] = {}
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in dirs + names:
            path = Path(directory) / name
            no_links(path)
            if name in names:
                require(path.is_file(), "Non-regular input")
                result[path.relative_to(root).as_posix()] = sha(path)
    require(len({name.casefold() for name in result}) == len(result), "Case-colliding inputs")
    return dict(sorted(result.items()))


def copy_tree(source: Path, target: Path) -> dict[str, str]:
    before = inventory(source)
    no_links(target)
    require(not target.exists(), "Fresh copy destination required")
    require(
        shutil.disk_usage("H:/").free > sum((source / n).stat().st_size for n in before) + 10 * GIB,
        "Copy disk reserve",
    )
    shutil.copytree(source, target)
    require(inventory(target) == before == inventory(source), "Copied input changed")
    return before


def stage(source: Path, target: Path) -> dict[str, str]:
    """Fill only a fresh resource directory or its tracked scaffolding, never replace payload."""
    no_links(target)
    before = inventory(target) if target.exists() else {}
    require(
        set(before) <= {".gitignore", "README.txt", "README.md"},
        "Existing staged payload must be preserved by controller",
    )
    incoming = inventory(source)
    require(not any(n in before and before[n] != h for n, h in incoming.items()), "Scaffolding conflicts")
    target.mkdir(parents=True, exist_ok=True)
    for name, checksum in incoming.items():
        path = target / name
        if name not in before:
            path.parent.mkdir(parents=True, exist_ok=True)
            with (source / name).open("rb") as src, path.open("xb") as dst:
                shutil.copyfileobj(src, dst)
        require(sha(path) == checksum == sha(source / name), "Staging mismatch")
    require(inventory(target) == dict(sorted((before | incoming).items())), "Stage membership drift")
    return inventory(target)


def command(args: list[str | Path], *, cwd: Path = ROOT, quiet: bool = False, timeout: int = 300) -> str:
    run = subprocess.run(
        list(map(str, args)),
        cwd=cwd,
        check=False,
        text=True,
        encoding="utf-8",
        stdout=subprocess.DEVNULL if quiet else subprocess.PIPE,
        stderr=subprocess.DEVNULL if quiet else subprocess.PIPE,
        timeout=timeout,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    require(
        run.returncode == 0,
        (
            "Child failed; raw output suppressed"
            if quiet
            else "Command failed: " + str(args[0]) + "\n" + (run.stderr or "")[-3000:]
        ),
    )
    if not quiet and run.stderr:
        LOG.info(run.stderr[-3000:])
    return "" if quiet else run.stdout.strip()


def git(*args: str) -> str:
    return command(["git", "-C", ROOT, *args], timeout=90)


def validate_checkout(root: Path, branch: str) -> None:
    """Accept only the named physical checkout and its corresponding branch."""
    no_links(root)
    require(root in CHECKOUT_BRANCHES and root.resolve(strict=True) == root, "Unapproved checkout or alias")
    require(branch == CHECKOUT_BRANCHES[root], "Wrong branch for approved checkout")


def signing_key_path(key: Path) -> Path:
    """Validate key location without reading its contents or moving it."""
    require(key.is_absolute(), "Absolute key path required")
    no_links(key)
    resolved = key.resolve(strict=True)
    require(resolved.is_file(), "Existing regular signing key required")
    for tree in (*CHECKOUT_BRANCHES, ROOT, Path("H:/Strata/worktrees"), Path("H:/Strata/releases"), OUT):
        no_links(tree)
        require(not resolved.is_relative_to(tree.resolve()), "Key must be outside source and release trees")
    return resolved


def reviewed(commit: str) -> dict[str, str]:
    require(bool(re.fullmatch("[a-f0-9]{40}", commit or "")), "Explicit full reviewed commit required")
    no_links(HERE / "release.py")
    validate_checkout(ROOT, git("branch", "--show-current"))
    require(
        git("rev-parse", "HEAD") == commit and not git("status", "--porcelain", "--untracked-files=all"),
        "Clean reviewed checkout HEAD required",
    )
    for name in ("VERSION", "CONTROL_VERSION"):
        require((ROOT / name).read_text().strip() == VERSION, "Version mismatch")
    require(
        sha(NODE) == "86fc4785327db5f5c7558dc6156c543b433c313e813a222e088086222ab4afa6",
        "Pinned Node executable differs",
    )
    require(
        sha(RES / "node/LICENSE") == "ed34dd8e3f0a78dbaf00d0444ce8e285b015b765379c2e17880455f70370f8e9",
        "Pinned Node licence differs",
    )
    return {"sourceCommit": commit, "sourceRoot": str(ROOT), "tree": git("rev-parse", "HEAD^{tree}")}


def module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec and spec.loader, "Module missing")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def sample(_root: Any = None) -> dict[str, int]:
    sys.path.insert(0, str(ROOT / "tools"))
    from native_model_resource_guard import available_memory

    memory = available_memory()
    return {
        "available_ram": memory["available_physical_bytes"],
        "commit_headroom": memory["available_commit_bytes"],
        "H_free": shutil.disk_usage("H:/").free,
        "C_free": shutil.disk_usage("C:/").free,
    }


def ram_policy(allow_low_ram: bool) -> dict:
    """Select this invocation's operation floor; never consult inherited settings."""
    require(type(allow_low_ram) is bool, "Explicit boolean RAM exception required")
    return {
        "allowLowRam": allow_low_ram,
        "nominalFloorBytes": 3 * GIB,
        "effectiveFloorBytes": 0 if allow_low_ram else 3 * GIB,
        "authorization": "user-explicit-2026-11-16-operation-only" if allow_low_ram else None,
    }


def observe_ram(policy: dict, values: dict, observations: list[int] | None = None) -> dict:
    """Keep actual minima and nominal breaches even when the exception is selected."""
    available = values["available_ram"]
    require(type(available) is int and available >= 0, "Invalid RAM observation")
    policy["samples"] += 1
    policy["nominalBreachSamples"] += int(available < 3 * GIB)
    previous = policy["minAvailableBytes"]
    policy["minAvailableBytes"] = available if previous is None else min(previous, available)
    if observations is not None:
        observations.append(available)
    return values


def admit(*, allow_low_ram: bool = False) -> dict:
    policy = ram_policy(allow_low_ram)
    values = sample()
    for key, floor in (
        ("available_ram", policy["effectiveFloorBytes"]),
        ("commit_headroom", 4 * GIB),
        ("H_free", 10 * GIB),
        ("C_free", 5 * GIB),
    ):
        require(values[key] >= floor, "Resource floor refused: " + key)
    return values


def publication_admissions(work: Path) -> list[dict]:
    """Read a bounded, worker-bound stream of successful readback admissions."""
    path = work / "publication-admissions.jsonl"
    no_links(path)
    require(
        path.is_file() and path.stat().st_size <= 256 * 1024 * 1024,
        "Publication admission evidence missing or oversized",
    )
    with path.open("r", encoding="utf-8") as stream:
        header = json.loads(stream.readline())
        require(
            header == {"workerContextSHA256": sha(work / "operation-worker.json")},
            "Publication admission worker binding differs",
        )
        rows = []
        for line in stream:
            require(len(rows) < 1000000 and len(line) <= 1024, "Publication admission evidence exceeds limit")
            rows.append(json.loads(line))
    require(rows, "Publication admission evidence empty")
    return rows


def receipt_ram_floor(record: dict, work: Path, job: dict) -> int:
    """Lower a consumer floor only with the observed command and owned worker proof."""
    policy = record.get("ramPolicy")
    if "ramPolicy" not in record:
        require(
            "ramPolicy" not in job and "operationExecution" not in job, "Missing receipt RAM policy"
        )
        return 3 * GIB
    require(isinstance(policy, dict), "Malformed RAM policy")
    selected = ram_policy(policy.get("allowLowRam"))
    require(
        set(policy) == set(selected) | {"samples", "nominalBreachSamples", "minAvailableBytes"}
        and all(type(policy.get(k)) is type(v) and policy.get(k) == v for k, v in selected.items()),
        "RAM policy selection differs",
    )
    count, breaches, minimum = (
        policy[k] for k in ("samples", "nominalBreachSamples", "minAvailableBytes")
    )
    require(
        type(count) is int
        and count > 1
        and type(breaches) is int
        and 0 <= breaches <= count
        and type(minimum) is int
        and minimum >= selected["effectiveFloorBytes"]
        and (minimum < 3 * GIB) == (breaches > 0),
        "Invalid RAM policy observations",
    )
    raw_policy = job.get("ramPolicy")
    require(
        isinstance(raw_policy, dict)
        and set(raw_policy) == set(policy)
        and all(type(raw_policy[k]) is type(v) and raw_policy[k] == v for k, v in policy.items()),
        "Receipt/raw Job RAM policy differs",
    )
    observations = job.get("operationRAMSamples")
    require(
        isinstance(observations, list)
        and len(observations) == count
        and all(type(v) is int and v >= 0 for v in observations)
        and min(observations) == minimum
        and sum(v < 3 * GIB for v in observations) == breaches,
        "Raw RAM sample/breach counts differ",
    )
    reservation = load(work / "reservation.json")
    identity = {k: record[k] for k in ("sourceCommit", "sourceRoot", "tree")}
    execution = job.get("operationExecution", {})
    command = execution.get("command", [])
    require(
        isinstance(command, list)
        and len(command) > 3
        and all(isinstance(v, str) for v in command)
        and Path(command[0]) == Path(sys.executable)
        and command[1] == "-B"
        and Path(command[2]).parent == HERE
        and Path(command[2]).name
        in {
            "release.py",
            "native-input.py",
            "components-feed.py",
            "delivery.py",
            "acceptance.py",
            "real-model.py",
            "normal-start.py",
            "bundle.py",
            "publish.py",
            "original-engine.py",
        },
        "Foreign operation worker command",
    )

    def option(name: str) -> str:
        values = [command[i + 1] for i, v in enumerate(command[:-1]) if v == name]
        values += [v.split("=", 1)[1] for v in command[3:] if v.startswith(name + "=")]
        require(len(values) == 1, "Missing/duplicate operation argument: " + name)
        return values[0]

    require(
        command.count("--allow-low-ram") == int(selected["allowLowRam"])
        and not any(v.startswith("--allow-low-ram=") for v in command)
        and command.count("--execute") == 1
        and command.count("--worker") == 1
        and option("--reviewed-commit") == record["sourceCommit"]
        and work == OUT / (record["phase"] + "-" + option("--attempt")),
        "Operation policy argv differs",
    )
    require(
        reservation.get("identity") == identity
        and reservation.get("ramPolicy") == selected
        and reservation.get("command") == command,
        "Operation reservation policy/source differs",
    )
    helpers = reservation.get("helpers", {})
    require(
        set(helpers) == {"release.py", "verify.mjs", Path(command[2]).name},
        "Reserved operation helpers differ",
    )
    for path in (HERE / name for name in helpers):
        require(helpers.get(path.name) == sha(path), "Reserved operation source changed")
    context = load(work / "operation-worker.json")
    admissions = job.get("operationAdmissions")
    install = record["phase"] == "install"
    publication = record["phase"] == "publication-readback"
    chunk_path = work / "publication-admissions.jsonl"
    chunks = publication_admissions(work) if publication else []
    require(publication or not chunk_path.exists(), "Unexpected publication admission evidence")
    require(
        isinstance(admissions, list)
        and len(admissions) == 1 + int(install) + len(chunks)
        and (not publication or admissions[1:] == chunks)
        and all(
            isinstance(v, dict)
            and set(v) == {"available_ram", "commit_headroom", "H_free", "C_free"}
            and all(type(n) is int and n >= 0 for n in v.values())
            for v in admissions
        ),
        "Missing/malformed worker or install admission observations",
    )
    install_sha = None
    if install:
        install_path = work / "operation-install-admission.json"
        require(
            load(install_path)
            == {
                "resources": admissions[1],
                "workerContextSHA256": sha(work / "operation-worker.json"),
            },
            "Install admission worker binding differs",
        )
        install_sha = sha(install_path)
    require(
        context
        == {
            "command": command,
            "identity": identity,
            "ramPolicy": selected,
            "pid": job.get("owned_root_pid"),
            "creationFiletime": job.get("root_creation_filetime"),
            "attemptRoot": str(work),
            "reservationSHA256": sha(work / "reservation.json"),
            "admission": admissions[0],
        }
        and type(context.get("pid")) is int
        and context["pid"] > 0
        and type(context.get("creationFiletime")) is int
        and context["creationFiletime"] > 0,
        "Operation worker identity/policy differs",
    )
    expected_execution = {
        "command": command,
        "cwd": record["sourceRoot"],
        "attemptRoot": str(work),
        "reservationSHA256": sha(work / "reservation.json"),
        "workerContextSHA256": sha(work / "operation-worker.json"),
        "installAdmissionSHA256": install_sha,
        "admission": execution.get("admission"),
    }
    if publication:
        expected_execution["publicationAdmissionsSHA256"] = sha(chunk_path)
    require(
        execution == expected_execution,
        "Unbound operation Job",
    )
    admission, diag = execution["admission"], job.get("memory_diagnostics", {})
    require(
        isinstance(admission, dict)
        and diag.get("status") == "AVAILABLE"
        and type(job.get("resource_samples")) is int
        and job["resource_samples"] > 0
        and diag.get("samples") == job["resource_samples"]
        and count == job["resource_samples"] + 1 + len(admissions),
        "Operation resource samples differ",
    )
    require(
        observations[0] == admission.get("available_ram")
        and min(observations[1 : 1 + job["resource_samples"]]) == diag.get("min_available_ram")
        and observations[1 + job["resource_samples"] :] == [v["available_ram"] for v in admissions]
        and minimum >= selected["effectiveFloorBytes"],
        "Operation measured RAM minimum differs",
    )
    for key, floor in (
        ("available_ram", selected["effectiveFloorBytes"]),
        ("commit_headroom", 4 * GIB),
        ("H_free", 10 * GIB),
        ("C_free", 5 * GIB),
    ):
        require(
            all(v.get(key, -1) >= floor for v in [admission, *admissions]),
            "Operation admission floor violated",
        )
    for key, floor in (("min_commit_headroom", 4 * GIB), ("min_H_free", 10 * GIB)):
        require(diag.get(key, -1) >= floor, "Operation observed non-RAM floor violated")
    require(
        job.get("last_resources", {}).get("C_free", -1) >= 5 * GIB
        and job.get("status") == "FINISHED"
        and job.get("exit_code") == 0
        and job.get("native_exit_code") == 0
        and job.get("contained_before_resume") is True
        and job.get("termination_confirmed") is True
        and job.get("job_active_after") == 0
        and job.get("observed_dropped_bytes") == 0
        and "admissionEvidenceError" not in job,
        "Successful sampled drained operation Job required",
    )
    return selected["effectiveFloorBytes"]


def verify(mode: str, path: Path, args: argparse.Namespace, receipt: Path | None = None) -> None:
    command(
        [
            NODE,
            HERE / "verify.mjs",
            mode,
            ROOT,
            path,
            args.reviewed_commit,
            str(receipt or "-"),
            str(args.public_assets or "-"),
            str(args.public_assets_receipt or "-"),
            args.public_assets_receipt_sha256 or "-",
        ]
    )


def public_asset_identity(args: argparse.Namespace) -> dict | None:
    """Bind explicit, physical prepared inputs; never infer historical provenance."""
    paths = (args.public_assets, args.public_assets_receipt)
    expected = args.public_assets_receipt_sha256
    if not any(paths) and not expected and args.phase not in {"dependencies", "component", "app"}:
        return None
    require(
        all(paths) and bool(re.fullmatch("[a-f0-9]{64}", expected or "")),
        "Explicit public assets, receipt and expected receipt SHA-256 required",
    )
    inputs = OUT / "inputs"
    no_links(inputs)
    for path in paths:
        require(path.is_absolute(), "Absolute public asset input required")
        no_links(path)
        require(
            path.resolve(strict=True) == path and path.is_relative_to(inputs),
            "Public asset inputs must be physical paths inside release inputs",
        )
    raw = args.public_assets_receipt.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, "Public asset receipt SHA-256 mismatch")
    record = json.loads(raw.decode("utf-8-sig"))
    require(
        record.get("lucide_commit") == "03f11e2d6adc319ea21e9a1d5820a99cd3767257",
        "Unreviewed public asset source",
    )
    files = record.get("files")
    require(isinstance(files, dict) and bool(files), "Public asset file inventory required")
    for name, checksum in files.items():
        require(
            isinstance(name, str)
            and not any(c in name for c in ("\\", ":"))
            and all(p not in {"", ".", ".."} for p in name.split("/"))
            and "gtwalsheim" not in name.lower()
            and isinstance(checksum, str)
            and bool(re.fullmatch("[a-f0-9]{64}", checksum)),
            "Unsafe public asset inventory member",
        )
    require(inventory(args.public_assets) == files, "Public asset membership or bytes changed")
    require(sha(args.public_assets_receipt) == expected, "Public asset receipt changed during verification")
    return {
        "root": str(args.public_assets),
        "receipt": str(args.public_assets_receipt),
        "receiptSHA256": expected,
        "lucideCommit": record["lucide_commit"],
        "files": files,
    }


def unchanged_public_assets(args: argparse.Namespace, expected: dict | None) -> None:
    require(public_asset_identity(args) == expected, "Reserved public asset inputs changed")


def prior_notice_identity(args: argparse.Namespace, work: Path) -> dict | None:
    """Verify preserved old notices independently of the fresh resource destination."""
    prior, expected = args.prior_notices, args.prior_notices_manifest_sha256
    if prior is None and expected is None and args.phase != "notices":
        return None
    require(
        prior is not None and bool(re.fullmatch("[a-f0-9]{64}", expected or "")),
        "Explicit prior notices and expected manifest SHA-256 required",
    )
    require(prior.is_absolute(), "Absolute prior notices input required")
    no_links(prior)
    inputs = OUT / "inputs"
    require(
        prior.resolve(strict=True) == prior and prior != inputs and prior.is_relative_to(inputs),
        "Prior notices must be a physical directory inside release inputs",
    )
    for other in (RES / "notices", work):
        no_links(other)
        physical = other.resolve()
        require(
            not prior.is_relative_to(physical) and not physical.is_relative_to(prior),
            "Prior notices overlap resource target or attempt",
        )
    manifest = prior / "NOTICES.json"
    no_links(manifest)
    raw = manifest.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, "Prior notices manifest SHA-256 mismatch")
    old = json.loads(raw.decode("utf-8-sig"))
    files = old.get("files")
    require(
        old.get("schema") == 1 and isinstance(files, dict) and bool(files),
        "Prior notices file inventory required",
    )
    for name, checksum in files.items():
        require(
            isinstance(name, str)
            and name != "NOTICES.json"
            and not any(c in name for c in ("\\", ":"))
            and all(p not in {"", ".", ".."} for p in name.split("/"))
            and isinstance(checksum, str)
            and bool(re.fullmatch("[a-f0-9]{64}", checksum)),
            "Unsafe prior notice inventory member",
        )
    actual = inventory(prior)
    require(
        set(actual) - set(files) <= {".gitignore", "NOTICES.json"}
        and actual.get("NOTICES.json") == expected
        and all(actual.get(name) == checksum for name, checksum in files.items()),
        "Prior notices membership or bytes changed",
    )
    vendor = load(ROOT / "packaging/microsoft-2022-notice-sources.json")
    for item in vendor:
        require(
            bool(re.fullmatch("[A-Za-z0-9.-]+", item["file"]))
            and files.get(item["file"]) == item["sha256"]
            and actual.get(item["file"]) == item["sha256"],
            "Prior vendor notice pin mismatch",
        )
    require(sha(manifest) == expected, "Prior notices manifest changed during verification")
    return {"root": str(prior), "manifestSHA256": expected, "files": actual, "microsoftSources": vendor}


def unchanged_prior_notices(args: argparse.Namespace, work: Path, expected: dict | None) -> None:
    require(prior_notice_identity(args, work) == expected, "Reserved prior notices changed")


def freeze(args: argparse.Namespace, work: Path) -> dict:
    identity = reviewed(args.reviewed_commit)
    files = {}
    for name in git("ls-files", "-z").split("\0"):
        if name:
            no_links(ROOT / name)
            files[name] = sha(ROOT / name)
    require(reviewed(args.reviewed_commit) == identity, "Source changed while freezing")
    return {**identity, "files": files, "releaseQualified": False, "mode": "integration exact clean commit"}


def dependencies(args: argparse.Namespace, work: Path) -> dict:
    verify("dependencies", PRIMARY_RES, args)
    no_links(PRIMARY_RES)
    no_links(RES)
    same = PRIMARY_RES == RES and PRIMARY_RES.resolve(strict=True) == RES.resolve(strict=True)
    before = {name: inventory(PRIMARY_RES / name) for name in ("harness", "local-backends")}
    staged = before if same else {name: stage(PRIMARY_RES / name, RES / name) for name in before}
    verify("dependencies", RES, args)
    require(all(inventory(PRIMARY_RES / n) == rows for n, rows in before.items()), "Source dependency drift")
    require(all(inventory(RES / n) == rows for n, rows in staged.items()), "Staged dependency drift")
    return {
        "mode": "verified-in-place" if same else "verified-copy",
        "copied": not same,
        "sourceRoot": str(PRIMARY_RES),
        "sourceFiles": before,
        "stagedFiles": staged,
        "publicAssetsReceiptSHA256": args.public_assets_receipt_sha256,
        "modelsOrProfilesCopied": False,
        "noticeStage": "fresh after Control",
    }


def control(args: argparse.Namespace, work: Path) -> dict:
    require(args.python_cache and args.wheelhouse, "Explicit verified Python cache and wheelhouse required")
    pt = module("release060_package_terminal", ROOT / "packaging/package_terminal.py")
    require(pt.PYTHON_VERSION == "3.12.15", "Unexpected CPython pin")
    pin = pt.PYTHON_PINS["windows-x86_64"]
    archive = args.python_cache / pin["file"].replace("+", "_")
    no_links(archive)
    require(sha(archive) == pin["sha256"], "CPython archive pin mismatch")
    lock = pt.lock_for("windows-x86_64")
    require(len(pt.lock_names(lock)) == 25, "Reviewed 25-wheel lock required")
    wheel_tree = inventory(args.wheelhouse)
    require(
        wheel_tree and all(n.endswith(".whl") and "/" not in n for n in wheel_tree),
        "Flat wheelhouse required",
    )
    approved = set(re.findall(r"--hash=sha256:([a-f0-9]{64})", lock.read_text()))
    require(all(h in approved for h in wheel_tree.values()), "Wheel not allowed by reviewed lock")
    cache = work / "python-cache"
    cache.mkdir()
    shutil.copy2(archive, cache / archive.name)
    require(sha(cache / archive.name) == pin["sha256"], "CPython cache copy mismatch")
    wheels = work / "wheels"
    copy_tree(args.wheelhouse, wheels)
    terminal = work / "terminal"
    command(
        [
            sys.executable,
            "-B",
            ROOT / "packaging/package_terminal.py",
            "--out",
            terminal,
            "--target",
            "windows-x86_64",
            "--version",
            VERSION,
            "--commit",
            args.reviewed_commit,
            "--keep-staging",
            "--cache",
            cache,
            "--wheelhouse",
            wheels,
        ],
        timeout=1200,
    )
    archives = list(terminal.glob("*.zip"))
    require(len(archives) == 1, "Expected one new Terminal archive")
    fresh = work / "control"
    require(
        not fresh.exists() and not fresh.with_name("control.incoming").exists(),
        "Fresh Control scratch required",
    )
    command(
        [
            sys.executable,
            "-B",
            ROOT / "apps/desktop/scripts/stage-control.py",
            "--archive",
            archives[0],
            "--out",
            fresh,
        ],
        timeout=600,
    )
    bundle = load(fresh / "BUNDLE.json")
    require(
        bundle["version"] == VERSION and bundle["python"]["version"] == pt.PYTHON_VERSION,
        "Control identity mismatch",
    )
    require(bundle["lock"]["sha256"] == sha(lock), "Control lock mismatch")
    require(
        all(bundle["installed"].get(n) == v for n, v in pt.lock_names(lock).items()),
        "Control installed packages mismatch",
    )
    require((fresh / "COMMIT").read_text().strip() == args.reviewed_commit, "Control source differs")
    tree = stage(fresh, RES / "control")
    command(
        [
            sys.executable,
            "-I",
            "-B",
            ROOT / "packaging/feed_tool.py",
            "pack-control",
            fresh,
            "--out",
            work / "assets",
            "--version",
            VERSION,
            "--platform",
            "windows-x86_64",
            "--format",
            "zip",
            "--url-prefix=",
        ]
    )
    return {
        "archive": str(archives[0]),
        "archiveSHA256": sha(archives[0]),
        "bundle": bundle,
        "controlFiles": tree,
        "pythonArchiveSHA256": pin["sha256"],
        "wheelFiles": wheel_tree,
        "assets": inventory(work / "assets"),
    }


def host_dependency_root() -> Path:
    """Permit only the reviewed cache through the primary dependency entrypoint."""
    host_deps = (ROOT / "apps/host/node_modules").resolve(strict=True)
    require(
        host_deps == Path("H:/Strata/evidence/upstream-sync-20261116/host-deps-060/node_modules"),
        "Unexpected Host dependency root",
    )
    no_links(host_deps)
    require(load(host_deps / "vite/package.json")["version"] == "6.4.6", "Reviewed Vite 6.4.6 required")
    return host_deps


def notices(args: argparse.Namespace, work: Path) -> dict:
    """Fresh notices bind current dependencies; never reuse the obsolete external ggml pin."""
    require(args.cuda_dir is not None, "Explicit CUDA toolkit directory required")
    prior_identity = prior_notice_identity(args, work)
    prior = args.prior_notices
    destination = work / "notices"
    destination.mkdir()
    entries: dict[str, str] = {}

    def copy(source: Path, name: str, expected: str | None = None) -> None:
        no_links(source)
        require(source.is_file() and source.stat().st_size < 8 * 1024**2, "Invalid notice")
        checksum = sha(source)
        require(expected is None or expected == checksum, "Vendor notice pin mismatch")
        target = destination / name
        require(
            target.resolve().is_relative_to(destination.resolve()) and name not in entries,
            "Unsafe notice name",
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        require(sha(target) == checksum == sha(source), "Notice copy mismatch")
        entries[name] = checksum

    for source, name in [
        (ROOT / "LICENSE", "Strata-MIT.txt"),
        (ROOT / "third_party/strata-ggml/LICENSE", "ggml-MIT.txt"),
        (ROOT / "third_party/libstrix/NOTICE.md", "libstrix-NOTICE.md"),
        (args.cuda_dir / "EULA.txt", "NVIDIA-EULA.txt"),
        (args.cuda_dir / "LICENSE", "NVIDIA-LICENSE.txt"),
        (RES / "node/LICENSE", "Node-LICENSE.txt"),
        (ROOT / "packaging/hakua-third-party-notices.md", "README.md"),
    ]:
        copy(source, name)
    vendor = load(ROOT / "packaging/microsoft-2022-notice-sources.json")
    for item in vendor:
        require(bool(re.fullmatch("[A-Za-z0-9.-]+", item["file"])), "Unsafe vendor name")
        copy(prior / item["file"], item["file"], item["sha256"])
    qwen = load(ROOT / "data/experimental-speed-projection/PROVENANCE.json")["license"]
    copy(
        ROOT / "data/experimental-speed-projection/LICENSE", "Qwen-Community-License-1.0.txt", qwen["sha256"]
    )
    host_deps = host_dependency_root()
    for name in ("react", "react-dom", "scheduler", "katex", "prismjs", "pdfjs-dist", "html-to-image"):
        folder = host_deps / name
        found = [
            p
            for p in folder.rglob("*")
            if p.is_file() and re.match("^(license|licence|notice|copying)", p.name, re.I)
        ]
        require(found, "GUI dependency notice missing")
        for path in found:
            copy(path, "gui/" + name + "/" + path.relative_to(folder).as_posix())
    py = RES / "control/python"
    found = [p for p in py.iterdir() if p.is_file() and p.name.upper().startswith("LICENSE")]
    require(found, "Python licence missing")
    for path in found:
        copy(path, "python/" + path.name)
    for info in sorted((py / "Lib/site-packages").glob("*.dist-info")):
        for path in sorted(info.rglob("*")):
            if path.is_file() and (
                re.match("^(license|licence|notice|copying)", path.name, re.I)
                or "licenses" in path.relative_to(info).parts
            ):
                copy(path, "python/distributions/" + info.name + "/" + path.relative_to(info).as_posix())
    save(
        destination / "NOTICES.json",
        {"schema": 1, "microsoftSources": vendor, "qwenSource": qwen, "files": entries},
    )
    unchanged_prior_notices(args, work, prior_identity)
    return {
        "noticeFiles": stage(destination, RES / "notices"),
        "ggmlMode": "vendored",
        "priorNotices": prior_identity,
    }


def component(args: argparse.Namespace, work: Path) -> dict:
    require(
        args.native_component and args.native_binding and args.key,
        "Native package, binding receipt and external key path required",
    )
    key = signing_key_path(args.key)
    verify("native", args.native_component, args, args.native_binding)
    require(
        not any(
            Path(n).suffix.lower() in (".gguf", ".q36epk", ".pack", ".safetensors", ".pt", ".pth", ".onnx")
            for n in inventory(args.native_component)
        ),
        "Model payload in native component",
    )
    verify("dependencies", RES, args)
    command(
        [
            NODE,
            ROOT / "apps/desktop/scripts/stage-engine-components.mjs",
            work / "unsigned",
            "--native-component",
            args.native_component,
            "--private-candidate",
        ]
    )
    signed = work / "signed" / NAME
    signed.parent.mkdir()
    unsigned_tree = copy_tree(work / "unsigned" / NAME, signed)
    command(
        [sys.executable, "-I", "-B", ROOT / "packaging/feed_tool.py", "sign-package", signed, "--key", key],
        quiet=True,
    )
    sys.path.insert(0, str(ROOT))
    from strata_control import engine_component, signing

    manifest = load(signed / "engine-component.json")
    signing.verify_bytes(
        signing.load_trusted_keys(environ={}),
        engine_component.signing_payload(manifest),
        manifest.get("signature"),
    )
    engine_component.verify(engine_component.read_manifest(signed), full=True)
    signed_tree = inventory(signed)
    require(
        {n: h for n, h in signed_tree.items() if n != "engine-component.json"}
        == {n: h for n, h in unsigned_tree.items() if n != "engine-component.json"},
        "Signing changed payload",
    )
    verify("native", signed, args, args.native_binding)
    deps = module("release060_binary_deps", ROOT / "packaging/binary_deps.py")
    for folder in (signed, signed / "local-backends/prism", signed / "local-backends/turboquant"):
        names = [p.name for p in folder.iterdir() if p.is_file()]
        require(not deps.uncovered(folder, names), "Uncovered PE dependencies")
    assets = work / "assets"
    command(
        [
            sys.executable,
            "-I",
            "-B",
            ROOT / "packaging/feed_tool.py",
            "pack-engine",
            signed,
            "--out",
            assets,
            "--format",
            "tar.gz",
            "--url-prefix=",
        ]
    )
    require(not (RES / "engine-bundle" / NAME).exists(), "Fresh staged component required")
    staged = stage(signed, RES / "engine-bundle" / NAME)
    require(staged == signed_tree, "Staged signed component differs")
    return {
        "signedComponent": str(signed),
        "manifestSHA256": sha(signed / "engine-component.json"),
        "files": signed_tree,
        "assets": inventory(assets),
        "signatureVerified": True,
        "nativeBindingSHA256": sha(args.native_binding),
        "releaseQualified": False,
    }


def app(args: argparse.Namespace, work: Path) -> dict:
    require(args.native_binding, "Fresh native binding receipt required")
    host_dependency_root()
    native = RES / "engine-bundle" / NAME
    verify("native", native, args, args.native_binding)
    target = work / "cargo-target"
    require(not target.exists(), "Fresh Cargo output required")
    os.environ.update(
        CARGO_TARGET_DIR=str(target),
        STRATA_PUBLIC_PACKAGE="1",
        STRATA_PUBLIC_ASSETS=str(args.public_assets),
        STRATA_PUBLIC_ASSETS_RECEIPT=str(args.public_assets_receipt),
        STRATA_NATIVE_SOURCE_ROOT=str(ROOT),
        STRATA_NATIVE_SOURCE_COMMIT=args.reviewed_commit,
        STRATA_NATIVE_MANIFEST_SHA256=sha(native / "engine-component.json"),
    )
    schema_dir = ROOT / "apps/desktop/src-tauri/gen/schemas"
    before = {p: p.read_bytes() for p in schema_dir.glob("*.json")}
    command(
        [
            NODE,
            "node_modules/@tauri-apps/cli/tauri.js",
            "build",
            "--bundles",
            "nsis",
            "--",
            "--locked",
            "--offline",
        ],
        cwd=ROOT / "apps/desktop",
        timeout=5400,
    )
    for path, content in before.items():
        require(json.loads(path.read_bytes()) == json.loads(content), "Generated schema changed semantically")
        path.write_bytes(content)
    reviewed(args.reviewed_commit)
    installer = target / f"release/bundle/nsis/Strata_{VERSION}_x64-setup.exe"
    public = work / f"Strata-{VERSION}-windows-x64-setup.exe"
    shutil.copy2(installer, public)
    require(sha(public) == sha(installer), "Installer copy mismatch")
    shell = target / "release/strata-desktop.exe"
    return {
        "installer": str(public),
        "installerSHA256": sha(public),
        "installerBytes": public.stat().st_size,
        "shell": str(shell),
        "shellSHA256": sha(shell),
        "compilerJobs": 5,
        "releaseQualified": False,
        "authenticodeSigned": False,
        "installed": False,
        "published": False,
    }


PHASES = {
    "freeze": freeze,
    "dependencies": dependencies,
    "control": control,
    "notices": notices,
    "component": component,
    "app": app,
}


def worker_environment(work: Path, token: str) -> dict[str, str]:
    """Construct the environment used at the owned production child boundary."""
    env = dict(os.environ)
    # Tauri also merges inherited JSON; unit-only resource overrides must not enter a release build.
    for name in (
        "PYTHONHOME",
        "PYTHONPATH",
        "RUSTC_WRAPPER",
        "RUSTC_WORKSPACE_WRAPPER",
        "NODE_OPTIONS",
        "TAURI_CONFIG",
    ):
        env.pop(name, None)
    env.update(
        HAKUA060_WORKER_TOKEN=token,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        UV_OFFLINE="1",
        UV_PYTHON_DOWNLOADS="never",
        PIP_NO_INDEX="1",
        CARGO_NET_OFFLINE="true",
        CARGO_BUILD_JOBS="5",
        CMAKE_BUILD_PARALLEL_LEVEL="5",
        CTEST_PARALLEL_LEVEL="1",
        NODE_OPTIONS="--max-old-space-size=1024",
        TEMP=str(work / "temp"),
        TMP=str(work / "temp"),
        PATH=str(NODE.parent) + os.pathsep + os.environ.get("PATH", ""),
    )
    return env


PATH_OPTIONS = (
    "python-cache",
    "wheelhouse",
    "cuda-dir",
    "native-component",
    "native-binding",
    "key",
    "public-assets",
    "public-assets-receipt",
    "prior-notices",
)


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=PHASES, nargs="?", default="freeze")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--allow-low-ram",
        action="store_true",
        help="Explicit operation-only RAM exception; retain commit/disk/Job guards",
    )
    parser.add_argument("--reviewed-commit")
    parser.add_argument("--attempt", default="r1")
    for name in PATH_OPTIONS:
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--public-assets-receipt-sha256")
    parser.add_argument("--prior-notices-manifest-sha256")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def worker_command(args: argparse.Namespace) -> list[str]:
    argv = [
        sys.executable,
        "-B",
        str(HERE / "release.py"),
        args.phase,
        "--execute",
        "--worker",
        "--reviewed-commit",
        args.reviewed_commit,
        "--attempt",
        args.attempt,
    ]
    if args.allow_low_ram:
        argv.append("--allow-low-ram")
    for name in (*PATH_OPTIONS, "public-assets-receipt-sha256", "prior-notices-manifest-sha256"):
        value = getattr(args, name.replace("-", "_"))
        if value is not None:
            argv.extend(["--" + name, str(value)])
    return argv


def main() -> int:
    args = arguments()
    selected_policy = ram_policy(args.allow_low_ram)
    require(bool(re.fullmatch("r[1-9][0-9]*", args.attempt)), "Attempt must be r1, r2, ...")
    work = OUT / (args.phase + "-" + args.attempt)
    if not args.execute:
        require(not args.worker, "Worker requires execute")
        LOG.info(
            json.dumps(
                {
                    "execute": False,
                    "phase": args.phase,
                    "root": str(ROOT),
                    "out": str(work),
                    "reviewedCommit": args.reviewed_commit,
                    "jobs": 5,
                    "testJobs": 1,
                    "floorsGiB": {"RAM": 3, "commit": 4, "H": 10, "C": 5},
                    "ramPolicy": selected_policy,
                    "releaseQualified": False,
                }
            )
        )
        return 0
    require(os.name == "nt" and sys.version_info >= (3, 12), "Windows Python 3.12+ required")
    identity = reviewed(args.reviewed_commit)
    public_assets = public_asset_identity(args)
    prior_notices = prior_notice_identity(args, work)
    admission = admit(allow_low_ram=args.allow_low_ram)
    command_argv = worker_command(args)
    helpers = {
        p.name: sha(p) for p in (HERE / "release.py", HERE / "verify.mjs", Path(command_argv[2]))
    }
    if args.worker:
        reservation = load(work / "reservation.json")
        require(
            os.environ.get("HAKUA060_WORKER_TOKEN") == reservation["token"]
            and reservation["identity"] == identity
            and reservation["ramPolicy"] == selected_policy
            and reservation["command"] == command_argv,
            "Owned worker token mismatch",
        )
        require(
            reservation["helpers"] == helpers,
            "Reserved helper bytes changed",
        )
        require(
            reservation["publicAssets"] == public_assets, "Reserved public asset inputs changed"
        )
        require(reservation["priorNotices"] == prior_notices, "Reserved prior notices changed")
        import ctypes
        from ctypes import wintypes
        from perf131k.windows_job import api

        present = wintypes.BOOL()
        require(
            api().IsProcessInJob(wintypes.HANDLE(-1), None, ctypes.byref(present))
            and present.value,
            "Owned Windows Job required",
        )
        times = [ctypes.c_ulonglong() for _ in range(4)]
        require(
            api().GetProcessTimes(wintypes.HANDLE(-1), *[ctypes.byref(t) for t in times]),
            "Cannot observe operation worker identity",
        )
        save(
            work / "operation-worker.json",
            {
                "command": [sys.executable, "-B", *sys.argv],
                "identity": identity,
                "ramPolicy": selected_policy,
                "pid": os.getpid(),
                "creationFiletime": times[0].value,
                "attemptRoot": str(work),
                "reservationSHA256": sha(work / "reservation.json"),
                "admission": admission,
            },
        )
        require(
            load(work / "operation-worker.json")["command"] == command_argv,
            "Actual worker argv differs",
        )
        result = PHASES[args.phase](args, work)
        unchanged_public_assets(args, public_assets)
        unchanged_prior_notices(args, work, prior_notices)
        require(reviewed(args.reviewed_commit) == identity, "Source changed during phase")
        require(
            reservation["helpers"] == helpers
            and load(work / "reservation.json") == reservation
            and ram_policy(args.allow_low_ram) == selected_policy,
            "Helper bytes changed during phase",
        )
        save(
            work / "pending.json",
            {
                "ok": True,
                **identity,
                "phase": args.phase,
                "releaseQualified": False,
                **result,
                "publicAssets": public_assets,
                "priorNotices": prior_notices,
            },
        )
        return 0
    no_links(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    require(not work.exists(), "Preserve previous attempt; choose a new attempt explicitly")
    work.mkdir()
    (work / "temp").mkdir()
    token = uuid.uuid4().hex
    save(
        work / "reservation.json",
        {
            "identity": identity,
            "token": token,
            "helpers": helpers,
            "command": command_argv,
            "ramPolicy": selected_policy,
            "publicAssets": public_assets,
            "priorNotices": prior_notices,
        },
    )
    env = worker_environment(work, token)
    reservation_sha = sha(work / "reservation.json")
    policy = {**selected_policy, "samples": 0, "nominalBreachSamples": 0, "minAvailableBytes": None}
    ram_samples: list[int] = []
    observe_ram(policy, admission, ram_samples)
    sys.path.insert(0, str(ROOT / "tools"))
    from perf131k.windows_job import run_owned

    budget = SimpleNamespace(
        job_commit_bytes=None,
        max_processes=64,
        min_available_ram=selected_policy["effectiveFloorBytes"],
        min_commit_headroom=4 * GIB,
        min_H_free=10 * GIB,
        timeout_s=6000,
        output_bytes=16 * 1024**2,
    )
    run = run_owned(
        command_argv,
        ROOT,
        env,
        work / "worker.log",
        budget,
        lambda root=None: observe_ram(policy, sample(root), ram_samples),
        memory_diagnostics=True,
        on_tick=lambda: "C_disk_floor" if shutil.disk_usage("C:/").free < 5 * GIB else None,
    )
    # Reconcile child-only admission samples after drain; never infer them from
    # parent/monitor minima, which may all be higher than these observations.
    admissions = []
    install_path = work / "operation-install-admission.json"
    chunk_path = work / "publication-admissions.jsonl"
    try:
        if (work / "operation-worker.json").is_file():
            admissions.append(load(work / "operation-worker.json")["admission"])
        if install_path.is_file():
            admissions.append(load(install_path)["resources"])
        if args.phase == "publication-readback":
            admissions.extend(publication_admissions(work))
        else:
            require(not chunk_path.exists(), "Unexpected publication admission evidence")
        for value in admissions:
            observe_ram(policy, value, ram_samples)
    except Exception as error:
        run["admissionEvidenceError"] = type(error).__name__
    run["ramPolicy"] = policy
    run["operationRAMSamples"] = ram_samples
    run["operationAdmissions"] = admissions
    run["operationExecution"] = {
        "command": command_argv,
        "cwd": str(ROOT),
        "attemptRoot": str(work),
        "reservationSHA256": reservation_sha,
        "admission": admission,
        "workerContextSHA256": sha(work / "operation-worker.json")
        if (work / "operation-worker.json").is_file()
        else None,
        "installAdmissionSHA256": sha(install_path) if install_path.is_file() else None,
    }
    if args.phase == "publication-readback" and chunk_path.is_file():
        run["operationExecution"]["publicationAdmissionsSHA256"] = sha(chunk_path)
    save(work / "owned-job.json", run)
    require(
        "admissionEvidenceError" not in run, "Worker admission evidence invalid; raw Job retained"
    )
    require(
        run["status"] == "FINISHED"
        and run["exit_code"] == 0
        and run["native_exit_code"] == 0
        and run["contained_before_resume"]
        and run["termination_confirmed"]
        and run["job_active_after"] == 0,
        "Owned phase failed; preserve evidence, no automatic retry",
    )
    require(reviewed(args.reviewed_commit) == identity, "Source changed after worker exit")
    require(
        sha(work / "reservation.json") == reservation_sha
        and ram_policy(args.allow_low_ram) == selected_policy,
        "Reserved operation policy changed after worker exit",
    )
    unchanged_public_assets(args, public_assets)
    unchanged_prior_notices(args, work, prior_notices)
    result = load(work / "pending.json")
    require(result["publicAssets"] == public_assets, "Worker public asset identity differs")
    require(result["priorNotices"] == prior_notices, "Worker prior notices identity differs")
    result["ramPolicy"] = policy
    receipt_ram_floor(result, work, run)
    save(work / "receipt.json", result)
    LOG.info(
        json.dumps({"ok": True, "receipt": str(work / "receipt.json"), "releaseQualified": False})
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        raise SystemExit(main())
    except Exception as error:
        LOG.error(
            "Release preparation failed (%s): %s",
            type(error).__name__,
            error if isinstance(error, RuntimeError) else "raw error suppressed",
        )
        raise SystemExit(1) from None
