"""Protected 0.6.0 extraction and NSIS overwrite; default is a read-only plan.

Every mutation runs through the existing release controller and owned Windows
Job. No process stop, model unload, setting change or publication is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location("delivery060_feed", HERE / "components-feed.py")
feed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feed)
native, release = feed.native, feed.release
APP = Path("C:/Users/downl/AppData/Local/Programs/Strata")
PRIVATE = Path("H:/Strata/private-backups/stable-060-20261116-codex")
SEVEN = Path("C:/Program Files/7-Zip/7z.exe")
USER_ROOTS = (
    Path("C:/Users/downl/.strata"),
    Path("C:/Users/downl/AppData/Roaming/Strata"),
    Path("C:/Users/downl/AppData/Local/Strata"),
    Path("C:/Users/downl/Documents/Strata/Workspace"),
)
MANAGED = ("control", "host", "node", "harness", "engine-bundle", "local-backends", "notices")
WEIGHTS = {".gguf", ".safetensors", ".pack", ".weights", ".q36epk"}
INSTALLER = "Strata-0.6.0-windows-x64-setup.exe"
QA_CHECKS = {
    "installed-host-shell-renderer-identity",
    "btw-typed-during-main-stream",
    "main-not-cancelled-or-queued",
    "btw-real-child-tool-free-api",
    "separate-durable-aside",
    "deepresearch-slash-menu",
    "installed-rustxng-full-real-search",
    "aside-survives-page-reload",
    "learning-and-application-default-off-independent-ui-toggles",
    "real-harness-three-completed-reads",
    "same-model-bounded-tool-free-review",
    "private-skill-created-without-file-outputs",
    "next-workflow-receives-full-matching-skill",
    "ui-undo-disables-retains-document",
    "settings-and-undone-skill-survive-reload",
    "mobile-no-horizontal-overflow",
}
MODEL = Path("H:/Strata/models/新しいフォルダー/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-Q4_K_M.gguf")


def pinned(path: Path, expected: str) -> dict:
    """Never infer trust from a self-described receipt or a filename alone."""
    release.require(path is not None and path.is_absolute(), "Absolute receipt required")
    release.no_links(path)
    release.require(re.fullmatch(r"[a-f0-9]{64}", expected or ""), "Expected receipt SHA-256 required")
    release.require(release.sha(path) == expected, "Receipt digest mismatch")
    return release.load(path)


def receipt(path: Path, expected: str, phase: str, commit: str) -> dict:
    record = pinned(path, expected)
    release.require(
        path.name == "receipt.json"
        and path.parent.parent == release.OUT
        and re.fullmatch(re.escape(phase) + r"-r[1-9][0-9]*", path.parent.name),
        "Explicit current phase receipt required",
    )
    release.require(
        record.get("ok") is True
        and record.get("phase") == phase
        and record.get("sourceCommit") == commit
        and Path(record.get("sourceRoot", "")) == native.PRIMARY
        and record.get("releaseQualified") is False,
        "Receipt source or status mismatch",
    )
    feed.successful_job(path.parent / "owned-job.json")
    return record


def prior(args: argparse.Namespace, name: str, phase: str | None = None) -> dict:
    return receipt(
        getattr(args, name + "_receipt"),
        getattr(args, name + "_receipt_sha256"),
        phase or name,
        args.reviewed_commit,
    )


def app_build(args: argparse.Namespace) -> dict:
    build = prior(args, "app")
    folder = args.app_receipt.parent
    expected_paths = {
        "installer": folder / INSTALLER,
        "shell": folder / "cargo-target/release/strata-desktop.exe",
    }
    for name, path in expected_paths.items():
        release.require(Path(build.get(name, "")) == path, "App artifact escaped its successful attempt")
        release.no_links(path)
        release.require(release.sha(path) == build[name + "SHA256"], "App artifact changed")
    release.require(
        Path(build["installer"]).stat().st_size == build["installerBytes"], "Installer size differs"
    )
    release.require(
        build.get("installed") is False and build.get("published") is False, "Expected build-only receipt"
    )
    return build


def files(root: Path, *, omit_weights: bool = False) -> dict:
    release.no_links(root)
    release.require(root.is_dir(), "Inventory root absent")
    rows = {}
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in dirs + names:
            release.no_links(Path(directory) / name)
        for name in names:
            path = Path(directory) / name
            release.require(path.is_file(), "Nonregular inventory member")
            model_bin = path.suffix.lower() == ".bin" and (
                "models" in [part.casefold() for part in path.relative_to(root).parts[:-1]]
                or name.startswith("pytorch_model")
            )
            if omit_weights and (path.suffix.lower() in WEIGHTS or ".q36epk" in name.lower() or model_bin):
                continue
            rows[path.relative_to(root).as_posix()] = {
                "sha256": release.sha(path),
                "bytes": path.stat().st_size,
            }
    release.require(len({n.casefold() for n in rows}) == len(rows), "Case-colliding files")
    return dict(sorted(rows.items()))


def payload(root: Path) -> dict:
    release.no_links(root)
    shell = root / "strata-desktop.exe"
    release.no_links(shell)
    rows = {shell.name: {"sha256": release.sha(shell), "bytes": shell.stat().st_size}}
    for name in MANAGED:
        if (root / name).exists():
            rows.update({name + "/" + p: row for p, row in files(root / name).items()})
    release.require(
        all((root / name).is_dir() for name in ("control", "host", "node", "harness", "notices")),
        "Incomplete managed payload",
    )
    return dict(sorted(rows.items()))


def user_files() -> dict:
    return {str(root): files(root, omit_weights=True) if root.exists() else None for root in USER_ROOTS}


def app_user_files(rows: dict) -> dict:
    """Preserve unexpected user files beside the app; NSIS owns its uninstaller."""
    return {
        name: row
        for name, row in rows.items()
        if name not in ("strata-desktop.exe", "uninstall.exe") and name.split("/")[0] not in MANAGED
    }


def model_identity() -> dict:
    """Hash the selected original weight without copying it into the backup."""
    release.no_links(MODEL)
    release.require(MODEL.is_file() and MODEL.stat().st_size == 5629111584, "Selected original model differs")
    return {"bytes": MODEL.stat().st_size, "sha256": release.sha(MODEL)}


def quiescent() -> None:
    """Refuse live or uninspectable relevant processes; never stop any process."""
    import psutil

    names = {
        "strata-desktop.exe",
        "strata.exe",
        "strata-q36.exe",
        "llama-server.exe",
        "prism-server.exe",
        "node.exe",
        "python.exe",
        "pythonw.exe",
    }
    roots = [str(p).replace("\\", "/").casefold() + "/" for p in (APP, *USER_ROOTS)]
    for process in psutil.process_iter(["name"]):
        if process.pid == os.getpid():
            continue
        try:
            name = (process.info.get("name") or "").casefold()
            exe = process.exe().replace("\\", "/").casefold()
            cmd = " ".join(process.cmdline()).replace("\\", "/").casefold()
            release.require(
                not any(exe.startswith(root) or root in cmd for root in roots),
                "Installed app or profile still in use; require fresh normal idle quit",
            )
            release.require(
                name
                not in {
                    "strata-desktop.exe",
                    "strata.exe",
                    "strata-q36.exe",
                    "llama-server.exe",
                    "prism-server.exe",
                },
                "A model process remains; establish its ownership separately",
            )
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied as error:
            if name in names:
                raise RuntimeError("Cannot establish relevant process quiescence") from error


def private_directory(path: Path) -> None:
    release.no_links(path)
    release.require(not path.exists(), "Preserve private backup; fresh attempt required")
    path.mkdir(parents=True)
    who = release.command(["whoami.exe", "/user", "/fo", "csv", "/nh"])
    sid = next(csv.reader(io.StringIO(who)))[1]
    release.require(re.fullmatch(r"S-1-\d+(?:-\d+)+", sid), "Invalid current owner SID")
    release.command(
        ["icacls.exe", path, "/inheritance:r", "/grant:r", f"*{sid}:(OI)(CI)F", "*S-1-5-18:(OI)(CI)F"],
        quiet=True,
    )
    literal = str(path).replace("'", "''")
    script = (
        "$a=Get-Acl -LiteralPath '" + literal + "';"
        "$rules=@($a.Access|ForEach-Object{[pscustomobject]@{"
        "sid=$_.IdentityReference.Translate([System.Security.Principal.SecurityIdentifier]).Value;"
        "allow=($_.AccessControlType -eq 'Allow');inherited=$_.IsInherited;"
        "rights=[int]$_.FileSystemRights;inheritance=[int]$_.InheritanceFlags}});"
        "[pscustomobject]@{protected=$a.AreAccessRulesProtected;rules=$rules}|ConvertTo-Json -Depth 4 -Compress"
    )
    powershell = Path(os.environ["SYSTEMROOT"]) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    acl = json.loads(release.command([powershell, "-NoProfile", "-NonInteractive", "-Command", script]))
    release.require(
        acl.get("protected") is True
        and len(acl.get("rules", [])) == 2
        and {row["sid"] for row in acl["rules"]} == {sid, "S-1-5-18"}
        and all(
            row["allow"] and not row["inherited"] and row["rights"] == 2032127 and row["inheritance"] == 3
            for row in acl["rules"]
        ),
        "Private backup ACL verification failed; no private bytes copied",
    )


def app_weight_name(name: str) -> bool:
    """Conservatively reject binary model candidates inside an app backup."""
    parts = Path(name).parts
    return any(
        Path(part).suffix.lower() in WEIGHTS | {".bin"}
        or re.search(r"\.(gguf|safetensors|q36epk)(?:\.(part|partial))?$", part, re.I)
        for part in parts
    )


def reject_app_weights(root: Path) -> None:
    """Metadata-only admission: never read, copy, move or delete weight bytes."""
    release.no_links(root)
    release.require(root.is_dir(), "Installed app absent")
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in dirs + names:
            path = Path(directory) / name
            release.no_links(path)
            release.require(
                not app_weight_name(path.relative_to(root).as_posix()),
                "App-resident model or pack weights refused before backup; originals preserved",
            )


def copy_rows(source: Path, target: Path, rows: dict, *, reject_weights: bool = False) -> None:
    release.no_links(source)
    release.no_links(target)
    release.require(not target.exists(), "Fresh copy required")
    for name in rows:
        relative = Path(name)
        release.require(
            not relative.anchor and ".." not in relative.parts and ":" not in name, "Unsafe inventory member"
        )
        release.require(
            not reject_weights or not app_weight_name(name), "Model weights refused before app copy"
        )
    target.mkdir(parents=True)
    for name, row in rows.items():
        relative = Path(name)
        src, dst = source / relative, target / relative
        release.no_links(src)
        release.no_links(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with src.open("rb") as incoming, dst.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
        release.require(
            release.sha(dst) == row["sha256"] == release.sha(src) and dst.stat().st_size == row["bytes"],
            "Backup copy differs",
        )
    release.require(files(target) == rows, "Backup membership differs")


def extraction(args: argparse.Namespace) -> tuple[dict, dict, Path]:
    build, record = app_build(args), prior(args, "extract")
    root = args.extract_receipt.parent / "extracted"
    release.require(record.get("installerSHA256") == build["installerSHA256"], "Extraction installer differs")
    release.require(record.get("payload") == payload(root), "Extracted payload changed")
    release.require(
        record.get("selfTest", {}).get("ok") is True or record.get("selfTest", {}).get("passed") is True,
        "Packaged self-test required",
    )
    return build, record, root


def isolated_environment(home: Path) -> dict:
    env = native.environment(home, os.environ.get("HAKUA060_WORKER_TOKEN", ""))
    for name, folder in {
        "STRATA_HOME": "",
        "STRATA_COMPONENTS_DIR": "components",
        "STRATA_SETTINGS_FILE": "settings.json",
        "STRATA_INSTALL_DIR": "install",
        "STRATA_DESKTOP_DATA_DIR": "desktop",
        "STRATA_PROFILE_DIR": "profiles",
        "STRATA_RUNTIME_DIR": "runtime",
        "STRATA_WORKSPACE": "workspace",
        "STRATA_MODELS_DIR": "models",
        "APPDATA": "Roaming",
        "LOCALAPPDATA": "Local",
        "DSH_HOME": "harness",
        "TEMP": "temp",
        "TMP": "temp",
    }.items():
        env[name] = str(home / folder)
    env.update(
        STRATA_NO_REPAIR_ACCESS="1", STRATA_NO_DIALOG="1", STRATA_NO_BUNDLED_ENGINE="1", STRATA_NO_ACQUIRE="1"
    )
    return env


def extract(args: argparse.Namespace, work: Path) -> dict:
    build = app_build(args)
    target = work / "extracted"
    release.require(not target.exists(), "Fresh extraction required")
    native.logged([SEVEN, "x", build["installer"], "-o" + str(target), "-y"], work, "extract", 600)
    expected = payload(target)
    unbundled = Path(build["shell"]).read_bytes()
    marker = b"__TAURI_BUNDLE_TYPE_VAR_UNK"
    release.require(unbundled.count(marker) == 1, "Single official Tauri bundle marker required")
    nsis = unbundled.replace(marker, b"__TAURI_BUNDLE_TYPE_VAR_NSS", 1)
    release.require(
        (target / "strata-desktop.exe").read_bytes() == nsis, "NSIS shell differs beyond official marker"
    )
    release.require(
        not (target / "engine-bundle").exists() and not (target / "local-backends").exists(),
        "GPU runtime must remain separate",
    )
    release.require(
        release.load(target / "harness/rustxng/SOURCE.json").get("profile") == "full", "RustXNG Full required"
    )
    home = work / "self-test-home"
    home.mkdir()
    for name in ("workspace", "Roaming", "Local", "temp"):
        (home / name).mkdir()
    report = work / "self-test.json"
    subprocess.run(
        [str(target / "strata-desktop.exe"), "--self-test=" + str(report)],
        cwd=target,
        env=isolated_environment(home),
        check=True,
        timeout=180,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    result = release.load(report)
    release.require(result.get("ok") is True or result.get("passed") is True, "Packaged self-test failed")
    release.require(
        app_build(args) == build and payload(target) == expected, "Extraction changed during self-test"
    )
    return {
        "installerSHA256": build["installerSHA256"],
        "payload": expected,
        "selfTest": result,
        "nsisShellSHA256": hashlib.sha256(nsis).hexdigest(),
        "onlyOfficialNSISMarkerChanged": True,
    }


def backup(args: argparse.Namespace, work: Path) -> dict:
    build, extracted, _ = extraction(args)
    quiescent()
    reject_app_weights(APP)
    release.require(
        release.load(APP / "host/package.json").get("version") == "0.5.9",
        "Fresh 0.5.9 baseline observation required",
    )
    root = PRIVATE / ("backup-" + args.attempt)
    private_directory(root)
    users, old, model = user_files(), files(APP), model_identity()
    size = sum(
        row["bytes"] for rows in [old, *[v for v in users.values() if v is not None]] for row in rows.values()
    )
    release.require(shutil.disk_usage(root).free > size + 10 * release.GIB, "Backup disk reserve")
    release.save(root / "source-inventory.json", {"user": users, "installed": old})
    release.require(
        not any(app_weight_name(name) for name in old), "App weight inventory changed; no payload copied"
    )
    copy_rows(APP, root / "installed-app", old, reject_weights=True)
    for index, (source, rows) in enumerate(users.items()):
        if rows is not None:
            copy_rows(Path(source), root / f"user-{index}", rows)
    quiescent()
    release.require(
        user_files() == users and files(APP) == old and model_identity() == model,
        "Source changed during backup",
    )
    release.command(
        [
            "reg.exe",
            "export",
            r"HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\Strata",
            root / "uninstall.reg",
        ],
        quiet=True,
    )
    release.save(root / "inventory.json", {"user": users, "installed": old, "selectedModel": model})
    release.require(extraction(args)[1] == extracted, "Candidate changed during backup")
    return {
        "backup": str(root),
        "inventorySHA256": release.sha(root / "inventory.json"),
        "installerSHA256": build["installerSHA256"],
        "modelsCopied": False,
        "rollbackRetained": True,
        "observedBaselineVersion": "0.5.9",
    }


def backup_identity(args: argparse.Namespace, build: dict) -> tuple[dict, dict]:
    record = prior(args, "backup")
    root = Path(record.get("backup", ""))
    release.require(
        root.parent == PRIVATE and re.fullmatch(r"backup-r[1-9][0-9]*", root.name),
        "Approved private backup required",
    )
    inventory = pinned(root / "inventory.json", record.get("inventorySHA256"))
    release.require(record.get("installerSHA256") == build["installerSHA256"], "Backup candidate differs")
    release.require(files(root / "installed-app") == inventory["installed"], "Rollback app changed")
    release.require(model_identity() == inventory["selectedModel"], "Selected original model changed")
    for index, rows in enumerate(inventory["user"].values()):
        if rows is not None:
            release.require(files(root / f"user-{index}") == rows, "Rollback user backup changed")
    return record, inventory


def install(args: argparse.Namespace, work: Path) -> dict:
    build, extracted, _ = extraction(args)
    candidate = prior(args, "candidate_qa", "candidate-qa")
    release.require(
        candidate.get("installerSHA256") == build["installerSHA256"], "Candidate QA installer differs"
    )
    qa_report = pinned(
        args.candidate_qa_receipt.parent / "qa-home/result.json", candidate.get("resultSHA256")
    )
    validate_qa(qa_report, "candidate", args.reviewed_commit, build["installerSHA256"])
    backup_record, before = backup_identity(args, build)
    quiescent()
    release.require(
        user_files() == before["user"] and files(APP) == before["installed"], "State changed since backup"
    )
    # Official Tauri per-user NSIS upgrade, with the existing literal destination.
    release.no_links(APP)
    release.admit()
    quiescent()
    native.logged([build["installer"], "/S", "/D=" + str(APP)], work, "nsis-install", 600)
    quiescent()
    release.require(payload(APP) == extracted["payload"], "Installed managed payload differs")
    release.require(
        app_user_files(files(APP)) == app_user_files(before["installed"]),
        "User files beside the installed app changed",
    )
    release.require(user_files() == before["user"], "User bytes or membership changed during install")
    release.require(
        model_identity() == before["selectedModel"], "Original selected model changed during install"
    )
    release.require(app_build(args) == build, "Installer changed during install")
    return {
        "installerSHA256": build["installerSHA256"],
        "product": "0.6.0",
        "exactPayloadMatch": True,
        "payload": extracted["payload"],
        "userFilesUnchanged": True,
        "backup": backup_record,
        "installed": True,
        "published": False,
    }


def validate_qa(report: dict, target: str, commit: str, installer_sha: str) -> None:
    release.require(
        report.get("ok") is True
        and report.get("mockModelOnly") is True
        and report.get("realRustxng") is True
        and report.get("consoleErrors") == 0
        and report.get("normalQuit") is True
        and report.get("actualWebView2") is True
        and report.get("isolatedHome") is True
        and report.get("target") == target
        and report.get("actualInstalledExe") is (target == "installed")
        and report.get("sourceCommit") == commit
        and report.get("installerSHA256") == installer_sha
        and len(report.get("checks", [])) == len(QA_CHECKS)
        and set(report.get("checks", [])) == QA_CHECKS
        and report.get("rustxngSources")
        and report.get("errors") == [],
        "Complete source-bound WebView2 fixture checks required",
    )


def qa(args: argparse.Namespace, work: Path) -> dict:
    build, extracted, candidate = extraction(args)
    # The Windows single-instance mutex is shared across profile directories.
    # A second launch would activate the user's existing shell, not this fixture.
    quiescent()
    target = "installed" if args.phase == "installed-qa" else "candidate"
    app = APP if target == "installed" else candidate
    if target == "installed":
        installed = prior(args, "install")
        release.require(
            installed.get("installerSHA256") == build["installerSHA256"]
            and installed.get("exactPayloadMatch") is True,
            "Current installed receipt required",
        )
    release.require(payload(app) == extracted["payload"], "QA target differs from extracted installer")
    contract = {
        "target": target,
        "app": str(app),
        "work": str(work),
        "ownerPID": os.getpid(),
        "sourceCommit": args.reviewed_commit,
        "installerSHA256": build["installerSHA256"],
        "nsisShellSHA256": extracted["nsisShellSHA256"],
    }
    release.save(work / "qa-input.json", contract)
    native.logged(
        [
            release.NODE,
            HERE / "qa-installed.mjs",
            "--execute",
            work / "qa-input.json",
            release.sha(work / "qa-input.json"),
        ],
        work,
        "webview-qa",
        600,
    )
    report = release.load(work / "qa-home/result.json")
    validate_qa(report, target, args.reviewed_commit, build["installerSHA256"])
    release.require(payload(app) == extracted["payload"] and app_build(args) == build, "QA payload changed")
    quiescent()
    return {
        "installerSHA256": build["installerSHA256"],
        "target": target,
        "resultSHA256": release.sha(work / "qa-home/result.json"),
        "qa": report,
    }


PHASES = {"extract": extract, "backup": backup, "install": install, "candidate-qa": qa, "installed-qa": qa}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cli = native.parser(__doc__)
    cli.add_argument("phase", choices=PHASES)
    for name in ("app", "extract", "backup", "install", "candidate-qa"):
        cli.add_argument("--" + name + "-receipt", type=Path)
        cli.add_argument("--" + name + "-receipt-sha256")
    args = cli.parse_args(argv)
    if args.execute:
        release.require("--attempt" in argv, "Explicit fresh attempt required")
        app_build(args)
    return native.run_controller(args, HERE / "delivery.py", PHASES[args.phase], argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        raise SystemExit(main())
    except Exception as error:
        logging.error(
            "Delivery failed (%s); preserve attempts and backups; raw error suppressed", type(error).__name__
        )
        raise SystemExit(1) from None
