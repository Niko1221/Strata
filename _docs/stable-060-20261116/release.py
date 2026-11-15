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

HERE = Path(__file__).resolve().parent
ROOT = Path("H:/Strata/worktrees/upstream-sync-20261116")
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


def reviewed(commit: str) -> dict[str, str]:
    require(bool(re.fullmatch("[a-f0-9]{40}", commit or "")), "Explicit full reviewed commit required")
    no_links(ROOT)
    require(
        git("rev-parse", "HEAD") == commit and not git("status", "--porcelain", "--untracked-files=all"),
        "Clean reviewed integration HEAD required",
    )
    require(git("branch", "--show-current") == "codex/upstream-sync-20261116", "Wrong integration branch")
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


def admit() -> None:
    values = sample()
    for key, floor in (("available_ram", 3), ("commit_headroom", 4), ("H_free", 10), ("C_free", 5)):
        require(values[key] >= floor * GIB, "Resource floor refused: " + key)


def verify(mode: str, path: Path, args: argparse.Namespace, receipt: Path | None = None) -> None:
    command([NODE, HERE / "verify.mjs", mode, ROOT, path, args.reviewed_commit, str(receipt or "-")])


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
    before = {name: inventory(PRIMARY_RES / name) for name in ("harness", "local-backends")}
    staged = {name: stage(PRIMARY_RES / name, RES / name) for name in before}
    verify("dependencies", RES, args)
    require(all(inventory(PRIMARY_RES / n) == rows for n, rows in before.items()), "Source dependency drift")
    return {
        "sourceRoot": str(PRIMARY_RES),
        "sourceFiles": before,
        "stagedFiles": staged,
        "publicAssetsReceiptSHA256": sha(Path("H:/Strata/releases/hakua033-public-assets.receipt.json")),
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


def notices(args: argparse.Namespace, work: Path) -> dict:
    """Fresh notices bind current dependencies; never reuse the obsolete external ggml pin."""
    require(args.cuda_dir is not None, "Explicit CUDA toolkit directory required")
    prior = PRIMARY_RES / "notices"
    old = load(prior / "NOTICES.json")
    require(
        all(sha(prior / n) == h for n, h in old["files"].items()), "Prior vendor notice inventory differs"
    )
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
    # The integration's reviewed dependency junction is allowed only for this explicit source read.
    host_deps = (ROOT / "apps/host/node_modules").resolve(strict=True)
    require(
        host_deps == Path("H:/Strata/evidence/upstream-sync-20261116/host-deps-060").resolve(),
        "Unexpected Host dependency root",
    )
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
    return {"noticeFiles": stage(destination, RES / "notices"), "ggmlMode": "vendored"}


def component(args: argparse.Namespace, work: Path) -> dict:
    require(
        args.native_component and args.native_binding and args.key,
        "Native package, binding receipt and external key path required",
    )
    key = args.key
    require(key.is_absolute(), "Absolute key path required")
    no_links(key)
    require(
        not key.resolve().is_relative_to(Path("H:/Strata").resolve())
        and not key.resolve().is_relative_to(Path("C:/Users/downl/Documents").resolve()),
        "Key must be outside source and release trees",
    )
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
    native = RES / "engine-bundle" / NAME
    verify("native", native, args, args.native_binding)
    target = work / "cargo-target"
    require(not target.exists(), "Fresh Cargo output required")
    os.environ.update(
        CARGO_TARGET_DIR=str(target),
        STRATA_PUBLIC_PACKAGE="1",
        STRATA_PUBLIC_ASSETS="H:/Strata/releases/hakua033-public-assets",
        STRATA_PUBLIC_ASSETS_RECEIPT="H:/Strata/releases/hakua033-public-assets.receipt.json",
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=PHASES, nargs="?", default="freeze")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--reviewed-commit")
    parser.add_argument("--attempt", default="r1")
    for name in ("python-cache", "wheelhouse", "cuda-dir", "native-component", "native-binding", "key"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
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
                    "releaseQualified": False,
                }
            )
        )
        return 0
    require(os.name == "nt" and sys.version_info >= (3, 12), "Windows Python 3.12+ required")
    identity = reviewed(args.reviewed_commit)
    admit()
    if args.worker:
        reservation = load(work / "reservation.json")
        require(
            os.environ.get("HAKUA060_WORKER_TOKEN") == reservation["token"]
            and reservation["identity"] == identity,
            "Owned worker token mismatch",
        )
        require(
            reservation["helpers"] == {p.name: sha(p) for p in (HERE / "release.py", HERE / "verify.mjs")},
            "Reserved helper bytes changed",
        )
        import ctypes
        from ctypes import wintypes
        from perf131k.windows_job import api

        present = wintypes.BOOL()
        require(
            api().IsProcessInJob(wintypes.HANDLE(-1), None, ctypes.byref(present)) and present.value,
            "Owned Windows Job required",
        )
        result = PHASES[args.phase](args, work)
        require(reviewed(args.reviewed_commit) == identity, "Source changed during phase")
        require(
            reservation["helpers"] == {p.name: sha(p) for p in (HERE / "release.py", HERE / "verify.mjs")},
            "Helper bytes changed during phase",
        )
        save(
            work / "pending.json",
            {"ok": True, **identity, "phase": args.phase, "releaseQualified": False, **result},
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
            "helpers": {p.name: sha(p) for p in (HERE / "release.py", HERE / "verify.mjs")},
        },
    )
    env = worker_environment(work, token)
    sys.path.insert(0, str(ROOT / "tools"))
    from perf131k.windows_job import run_owned

    budget = SimpleNamespace(
        job_commit_bytes=None,
        max_processes=64,
        min_available_ram=3 * GIB,
        min_commit_headroom=4 * GIB,
        min_H_free=10 * GIB,
        timeout_s=6000,
        output_bytes=16 * 1024**2,
    )
    run = run_owned(
        [sys.executable, "-B", str(HERE / "release.py"), *sys.argv[1:], "--worker"],
        ROOT,
        env,
        work / "worker.log",
        budget,
        sample,
        memory_diagnostics=True,
        on_tick=lambda: "C_disk_floor" if shutil.disk_usage("C:/").free < 5 * GIB else None,
    )
    save(work / "owned-job.json", run)
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
    result = load(work / "pending.json")
    save(work / "receipt.json", result)
    LOG.info(json.dumps({"ok": True, "receipt": str(work / "receipt.json"), "releaseQualified": False}))
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
