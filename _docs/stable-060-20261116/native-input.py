"""Prepare one unsigned 0.6.0 native input using the established owned-Job runner.

Default is a read-only plan. Execution requires a clean reviewed C main and an
explicit tested driver floor. This does not qualify models or publish anything.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Callable

sys.dont_write_bytecode = True
HERE = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location("release060_native_runner", HERE / "release.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)
PRIMARY = Path("C:/Users/downl/Documents/ChatGPT/Strata")
BUILD = Path("H:/Strata/evidence/upstream-sync-20261116/cuda-build")
CUDA = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3")
Q38 = release.module("release060_q38_input", HERE / "q38-input.py")
PAYLOAD = ("strata-q36.exe", "strata-q38.exe", "native-build.json", "q38-build.json", "LICENSE-q38-formats.txt", "LICENSE-turbo-dialects.txt")


def validate_version(output: str) -> None:
    """Require the stable token for every runner, including the owned Q38."""
    release.require(re.search(r"(^|\s)0\.6\.0(\s|$)", output), "Native version probe failed")


def parser(description: str) -> argparse.ArgumentParser:
    """Common argv for two small adapters of release.py's guarded runner."""
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--execute", action="store_true")
    result.add_argument(
        "--allow-low-ram",
        action="store_true",
        help="Explicit operation-only RAM exception; keep other resource guards",
    )
    result.add_argument("--reviewed-commit")
    result.add_argument("--attempt", default="r1")
    result.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    result.set_defaults(
        public_assets=None,
        public_assets_receipt=None,
        public_assets_receipt_sha256=None,
        prior_notices=None,
        prior_notices_manifest_sha256=None,
    )
    return result


def explicit_attempt(argv: list[str], attempt: str) -> None:
    release.require(
        sum(v == "--attempt" or v.startswith("--attempt=") for v in argv) == 1
        and bool(re.fullmatch(r"r[1-9][0-9]*", attempt)),
        "Execution requires one explicit valid --attempt before controller admission",
    )


def environment(work: Path, token: str) -> dict[str, str]:
    """Use the existing guard's credential-free tool environment and five jobs."""
    guard = release.module("native060_environment", PRIMARY / "_docs/local-053-20261109/native/common.py")
    guard.NODE = release.NODE
    guard.BASE = work
    env = guard.environment()
    env.update(
        HAKUA060_WORKER_TOKEN=token,
        CARGO_BUILD_JOBS="5",
        CMAKE_BUILD_PARALLEL_LEVEL="5",
        CTEST_PARALLEL_LEVEL="1",
        CARGO_NET_OFFLINE="true",
        STRATA_TEST_CONCURRENCY="1",
        NODE_OPTIONS="--max-old-space-size=1024",
    )
    return env


def run_controller(
    args: argparse.Namespace, script: Path, phase: Callable[[argparse.Namespace, Path], dict], argv: list[str]
) -> int:
    """Reuse admission, reservation, containment, failure retention and receipts."""
    release.require(release.ROOT == PRIMARY, "Controller requires the physical C main checkout")
    # release.py also supports an integration checkout; these adapters deliberately do not.
    release.CHECKOUT_BRANCHES = {PRIMARY: "main"}
    release.PHASES = {args.phase: phase}
    release.arguments = lambda: args
    release.worker_environment = environment
    release.worker_command = lambda _args: [
        sys.executable,
        "-B",
        str(script),
        *[v for v in argv if v != "--worker"],
        "--worker",
    ]
    return release.main()


def logged(argv: list[str | Path], work: Path, name: str, timeout: int = 5400) -> dict:
    """Keep complete non-secret builder/packager output, including failed attempts."""
    vector = list(map(str, argv))
    with (work / (name + ".log")).open("xb") as stream:
        proc = subprocess.run(
            vector,
            cwd=PRIMARY,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    result = {
        "command": vector,
        "exitCode": proc.returncode,
        "logSHA256": release.sha(work / (name + ".log")),
    }
    release.save(work / (name + ".json"), result)
    release.require(proc.returncode == 0, "Native command failed; retained log: " + name)
    return result


def configured_inputs(build: Path) -> dict:
    """Reject foreign/nonportable caches and freeze the actual compiler input set."""
    release.require(build == BUILD, "Only the configured integration build is approved")
    release.no_links(build)
    cache = build / "CMakeCache.txt"
    release.no_links(cache)
    values = cache_values(cache.read_text())
    release.require(
        Path(values.get("CMAKE_HOME_DIRECTORY", "")) == PRIMARY,
        "Configured build belongs to a different source",
    )
    for name, value in {
        "STRATA_PORTABLE": "ON",
        "STRATA_ENABLE_CUDA": "ON",
        "CMAKE_CUDA_ARCHITECTURES": "120",
    }.items():
        release.require(values.get(name) == value, "Unexpected cache setting: " + name)
    vendor = PRIMARY / "third_party/strata-ggml"
    release.require(
        not values.get("STRATA_GGML_DIR") or Path(values["STRATA_GGML_DIR"]) == vendor,
        "Current vendored ggml required",
    )
    baseline = Path(values.get("STRATA_CUDA_SPILL_BASELINE", ""))
    release.require(
        baseline.is_absolute() and baseline.is_relative_to(PRIMARY),
        "Source-bound CUDA resource profile required",
    )
    release.no_links(baseline)
    expression = (
        "import{pathToFileURL}from'node:url';"
        "const root=process.argv[1];"
        "const m=await import(pathToFileURL(root+'/apps/host/scripts/release-integrity.mjs'));"
        "console.log(JSON.stringify({sources:m.nativeSources(root),ggml:m.ggmlIdentity(root)}));"
    )
    native = json.loads(release.command([release.NODE, "--input-type=module", "-e", expression, PRIMARY]))
    release.require(native["ggml"]["mode"] == "vendored", "Vendored identity required")
    return {
        **native,
        "cacheSHA256": release.sha(cache),
        "baseline": {"path": baseline.relative_to(PRIMARY).as_posix(), "sha256": release.sha(baseline)},
    }


def cache_values(text: str) -> dict[str, str]:
    """CMake records are single lines; comments and blank lines cannot consume keys."""
    values = {}
    for line in text.splitlines():
        match = re.fullmatch(r"([^/#:=][^:=]*):[^=]+=(.*)", line)
        if match:
            values[match[1]] = match[2]
    return values


def package_command(args: argparse.Namespace, work: Path) -> list[str | Path]:
    """Use the production selector and only the approved CUDA library root."""
    release.require(
        bool(re.fullmatch(r"nvidia=\d+(?:\.\d+){1,3}", args.min_driver or "")),
        "Explicit measured NVIDIA minimum driver required",
    )
    release.require(args.cuda_search == CUDA / "bin/x64", "Only the CUDA 13.3 library root is approved")
    release.no_links(args.cuda_search)
    return [
        sys.executable,
        "-I",
        "-B",
        PRIMARY / "packaging/package_engine.py",
        "--src",
        work / "package-input",
        "--backend",
        "cuda",
        "--version",
        "0.6.0",
        "--platform",
        "windows-x86_64",
        "--variant",
        "cuda13",
        "--arch",
        "sm_120",
        "--build-id",
        args.reviewed_commit,
        "--min-driver",
        args.min_driver,
        *[v for name in PAYLOAD for v in ("--include", name)],
        "--search",
        args.cuda_search,
        "--out",
        work / "unsigned",
    ]


def capture_outputs(work: Path, label: str) -> dict[str, str]:
    """Preserve both executables and provenance even when the builder fails."""
    target = work / label
    target.mkdir()
    for name in ("strata.exe", "strata-q36.exe", "native-build.json"):
        source = BUILD / "Release" / name
        release.no_links(source)
        if source.is_file():
            shutil.copy2(source, target / name)
            release.require(release.sha(source) == release.sha(target / name), "Snapshot copy differs")
    return release.inventory(target)


def toolkit_libraries(search: Path, source: Path | None = None) -> dict[str, str]:
    """Inventory installed CUDA sources referenced by the production selection."""
    sys.path[:0] = [str(PRIMARY), str(PRIMARY / "packaging")]
    packager = release.module("native060_packager", PRIMARY / "packaging/package_engine.py")
    deps = packager.binary_deps
    source = source or BUILD / "Release"
    selected = packager.select(
        source, "strata.exe", tree=False, include=list(PAYLOAD)
    )
    installed = {p.name.lower(): p for p in search.glob("*.dll")}
    pending = [source / name for name in selected if Path(name).suffix.lower() in (".exe", ".dll")]
    used: dict[str, str] = {}
    while pending:
        path = pending.pop()
        release.no_links(path)
        names = [path.name, *deps.pe_imports(path)]
        for name in names:
            hit = installed.get(name.lower())
            if hit is not None and hit.name not in used:
                release.no_links(hit)
                used[hit.name] = release.sha(hit)
                pending.append(hit)
    return dict(sorted(used.items()))


def native(args: argparse.Namespace, work: Path) -> dict:
    """Build, probe, package and validate; signing and composition belong to release.py."""
    package = package_command(args, work)
    for folder in ("cache", "cache/cuda", "cache/uv", "cache/pip"):
        (work / folder).mkdir(parents=True, exist_ok=True)
    before = configured_inputs(args.build)
    release.save(work / "inputs-before.json", before)
    release.save(work / "outputs-before.json", capture_outputs(work, "build-before"))
    try:
        builder = logged(
            [release.NODE, PRIMARY / "tools/build_hakua_engine.mjs", args.build, "5"], work, "builder"
        )
    finally:
        release.save(work / "outputs-after.json", capture_outputs(work, "build-after"))
    after = configured_inputs(args.build)
    release.save(work / "inputs-after.json", after)
    release.require(before == after, "Native inputs changed during build")
    build_record = release.load(BUILD / "Release/native-build.json")
    release.require(
        build_record["sources"] == before["sources"]
        and build_record["recipe"]["ggml"] == before["ggml"]
        and build_record["recipe"].get("cudaSpillBaseline") == before["baseline"],
        "Builder did not bind and qualify the configured inputs",
    )
    # Stage a bounded selection, preserving the incremental cache and all old evidence.
    source = work / "package-input"
    source.mkdir()
    sys.path[:0] = [str(PRIMARY), str(PRIMARY / "packaging")]
    packager = release.module("native060_stage_packager", PRIMARY / "packaging/package_engine.py")
    for name in packager.select(BUILD / "Release", "strata.exe", tree=False, include=["strata-q36.exe", "native-build.json"]):
        shutil.copy2(BUILD / "Release" / name, source / name)
    q38 = Q38.stage(release, PRIMARY, source)
    probes = {}
    for binary in ("strata.exe", "strata-q36.exe", "strata-q38.exe"):
        probes[binary] = release.command([source / binary, "--version"], timeout=20)
        validate_version(probes[binary])
    release.save(work / "version-probes.json", probes)
    toolkit = release.load(CUDA / "version.json")
    release.require(toolkit.get("cuda", {}).get("version") == "13.3.1", "Unexpected CUDA toolkit version")
    toolkit_version_sha = release.sha(CUDA / "version.json")
    # Record actual library bytes before packaging; the production packager owns PE closure.
    libraries = toolkit_libraries(args.cuda_search, source)
    release.save(
        work / "toolkit-inputs.json",
        {"versionSHA256": toolkit_version_sha, "libraries": libraries},
    )
    logged(package, work, "package", 600)
    component = work / "unsigned" / release.NAME
    for path in component.glob("*.dll"):
        if path.name in libraries:
            release.require(
                release.sha(path) == libraries[path.name] == release.sha(args.cuda_search / path.name),
                "CUDA DLL changed during packaging",
            )
    release.require(toolkit_libraries(args.cuda_search, source) == libraries, "CUDA toolkit inputs changed")
    release.require(
        release.sha(CUDA / "version.json") == toolkit_version_sha, "CUDA version metadata changed"
    )
    binding = {
        "sourceRoot": str(PRIMARY),
        "sourceCommit": args.reviewed_commit,
        "releaseQualified": False,
        "nativeBuildSHA256": release.sha(component / "native-build.json"),
        "q38BuildSHA256": release.sha(component / "q38-build.json"),
        "q38CompiledSourceCommit": q38["sourceCommit"],
        "binaries": {n: release.sha(component / n) for n in probes},
        "builder": builder,
        "inputsBeforeSHA256": release.sha(work / "inputs-before.json"),
        "inputsAfterSHA256": release.sha(work / "inputs-after.json"),
        "toolkitInputsSHA256": release.sha(work / "toolkit-inputs.json"),
    }
    release.save(work / "native-binding.pending.json", binding)
    release.verify("native", component, args, work / "native-binding.pending.json")
    logged(
        [sys.executable, "-I", "-B", PRIMARY / "packaging/package_engine.py", "--verify", component],
        work,
        "package-verify",
        300,
    )
    release.require(configured_inputs(args.build) == before, "Native inputs changed during packaging")
    return {
        "nativeComponent": str(component),
        "files": release.inventory(component),
        "bindingPendingSHA256": release.sha(work / "native-binding.pending.json"),
        "builder": builder,
        "signed": False,
        "published": False,
    }


def finalize_binding(work: Path) -> None:
    """Publish the external binding only once the owned build Job has drained."""
    receipt = release.load(work / "receipt.json")
    job = release.load(work / "owned-job.json")
    release.require(
        receipt.get("ok") is True
        and receipt.get("phase") == "native-input"
        and job.get("status") == "FINISHED"
        and job.get("exit_code") == 0
        and job.get("native_exit_code") == 0
        and job.get("contained_before_resume") is True
        and job.get("termination_confirmed") is True
        and job.get("job_active_after") == 0,
        "Successful drained native owned Job required",
    )
    pending = work / "native-binding.pending.json"
    release.require(release.sha(pending) == receipt["bindingPendingSHA256"], "Pending binding changed")
    component = Path(receipt["nativeComponent"])
    release.require(release.inventory(component) == receipt["files"], "Native package changed after worker")
    binding = release.load(pending)
    release.require(binding["sourceCommit"] == receipt["sourceCommit"], "Native source receipt differs")
    release.save(
        work / "native-binding.json",
        {
            **binding,
            "ownedJob": str(work / "owned-job.json"),
            "ownedJobSHA256": release.sha(work / "owned-job.json"),
            "receiptSHA256": release.sha(work / "receipt.json"),
        },
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cli = parser(__doc__)
    cli.add_argument("--build", type=Path, default=BUILD)
    cli.add_argument("--cuda-search", type=Path, default=CUDA / "bin/x64")
    cli.add_argument("--min-driver")
    args = cli.parse_args(argv)
    args.phase = "native-input"
    if args.execute:
        explicit_attempt(argv, args.attempt)
        package_command(args, release.OUT / (args.phase + "-" + args.attempt))
    result = run_controller(args, HERE / "native-input.py", native, argv)
    if args.execute and not args.worker:
        finalize_binding(release.OUT / (args.phase + "-" + args.attempt))
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        raise SystemExit(main())
    except Exception as error:
        logging.error(
            "Native preparation failed (%s); retained attempt; raw error suppressed", type(error).__name__
        )
        raise SystemExit(1) from None
