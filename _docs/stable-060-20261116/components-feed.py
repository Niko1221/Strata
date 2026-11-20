"""Prepare a signed component-only feed from successful final-source receipts.

No publication or app update metadata. Default is a read-only plan. An explicit
live predecessor comparison and --no-dsv41-carry are required for execution.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import re
import shutil
import sys
import urllib.request

sys.dont_write_bytecode = True
HERE = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location("release060_native_adapter", HERE / "native-input.py")
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
release = native.release
KEY = Path("H:/Strata/private-keys/hakua-feed-20261101/feed-key.json")
CURRENT = "https://github.com/zapabob/Strata/releases/latest/download/strata-feed.json"


def successful_job(path: Path, record: dict | None = None) -> dict:
    """A phase receipt is consumable only with its drained successful owned Job."""
    job = release.load(path)
    if record is not None:
        release.receipt_ram_floor(record, path.parent, job)
    release.require(
        job.get("status") == "FINISHED"
        and job.get("exit_code") == 0
        and job.get("native_exit_code") == 0
        and job.get("contained_before_resume") is True
        and job.get("termination_confirmed") is True
        and job.get("job_active_after") == 0,
        "Successful drained component/control Job required",
    )
    return job


def receipt_assets(path: Path, expected: str, phase: str, commit: str) -> dict:
    """Check explicit local receipts and every referenced archive and entry byte."""
    release.require(
        path.is_absolute()
        and path.name == "receipt.json"
        and path.parent.parent == release.OUT
        and re.fullmatch(phase + r"-r[1-9][0-9]*", path.parent.name),
        "Explicit phase receipt inside the approved release root required",
    )
    release.no_links(path)
    release.require(
        re.fullmatch(r"[a-f0-9]{64}", expected or "") and release.sha(path) == expected,
        "Receipt digest mismatch",
    )
    record = release.load(path)
    release.require(
        record.get("ok") is True
        and record.get("phase") == phase
        and record.get("sourceCommit") == commit
        and Path(record.get("sourceRoot", "")) == native.PRIMARY
        and record.get("releaseQualified") is False,
        "Phase source or status mismatch",
    )
    successful_job(path.parent / "owned-job.json", record)
    assets = path.parent / "assets"
    files = release.inventory(assets)
    release.require(
        files and files == record.get("assets"), "Phase asset bytes or membership changed"
    )
    entries = [name for name in files if name.endswith(".entry.json")]
    release.require(len(entries) == 1, "Exactly one current Windows entry required per phase")
    entry = release.load(assets / entries[0])
    name = entry.get("url", "")
    release.require(
        isinstance(name, str)
        and bool(name)
        and Path(name).name == name
        and "/" not in name
        and "\\" not in name
        and ":" not in name,
        "Flat local archive URL required",
    )
    release.require(set(files) == {entries[0], name}, "Unreferenced phase asset")
    archive = assets / name
    release.require(
        entry.get("version") == "0.6.0"
        and entry.get("platform") == "windows-x86_64"
        and entry.get("sha256") == release.sha(archive)
        and entry.get("size") == archive.stat().st_size,
        "Entry/archive identity differs",
    )
    if phase == "control":
        original = Path(record.get("archive", ""))
        release.require(
            original.is_relative_to(path.parent / "terminal") and original.is_absolute(),
            "Control original archive must belong to the same attempt",
        )
        release.no_links(original)
        release.require(
            release.sha(original) == record.get("archiveSHA256"), "Original control archive changed"
        )
        staged = release.inventory(release.RES / "control")
        fresh = release.inventory(path.parent / "control")
        release.require(staged == record.get("controlFiles"), "Staged Control payload changed")
        # release.stage returns the staged inventory, including retained tracked scaffolding.
        scaffold = "apps/desktop/src-tauri/resources/control/.gitignore"
        release.require(
            release.git("ls-files", "--error-unmatch", "--", scaffold) == scaffold,
            "Current Control scaffold must be tracked",
        )
        release.require(
            all(staged.get(name) == checksum for name, checksum in fresh.items())
            and set(staged) - set(fresh) <= {".gitignore"},
            "Fresh Control payload differs from staged receipt",
        )
    else:
        signed = Path(record.get("signedComponent", ""))
        release.require(signed == path.parent / "signed" / release.NAME, "Foreign signed component")
        release.require(
            release.inventory(signed) == record.get("files")
            and release.sha(signed / "engine-component.json") == record.get("manifestSHA256")
            and entry.get("manifest_sha256") == record.get("manifestSHA256")
            and entry.get("backend") == "cuda"
            and entry.get("variant") == "cuda13"
            and record.get("signatureVerified") is True,
            "Signed component changed",
        )
        sys.path.insert(0, str(native.PRIMARY))
        from terminal.strata_control import engine_component, signing

        manifest = release.load(signed / "engine-component.json")
        release.require(
            manifest["build_id"] == commit + "-hakua-runtimes-v1", "Component build source differs"
        )
        signing.verify_bytes(
            signing.load_trusted_keys(environ={}),
            engine_component.signing_payload(manifest),
            manifest.get("signature"),
        )
        engine_component.verify(engine_component.read_manifest(signed), full=True)
    release.require(release.sha(path) == expected, "Receipt changed during verification")
    return {
        "receipt": str(path),
        "receiptSHA256": expected,
        "assets": str(assets),
        "files": files,
        "entry": entry,
        "entryName": entries[0],
        "ownedJobSHA256": release.sha(path.parent / "owned-job.json"),
    }


def read_current(url: str) -> bytes:
    """Bounded anonymous read of the canonical fork Latest feed or its signature."""
    release.require(url in (CURRENT, CURRENT + ".sig"), "Only canonical current fork feed allowed")
    request = urllib.request.Request(url, headers={"User-Agent": "Hakua-060-release-preflight"})
    with urllib.request.urlopen(request, timeout=45) as response:
        data = response.read(2 * 1024**2 + 1)
    release.require(len(data) <= 2 * 1024**2, "Feed response exceeds bound")
    return data


def predecessor(args: argparse.Namespace) -> dict:
    """Require a pinned local copy to equal the currently published signed feed."""
    release.require(args.previous_source == CURRENT, "Current fork source identity required")
    release.require(
        args.previous_feed
        and args.previous_feed.is_absolute()
        and re.fullmatch(r"[a-f0-9]{64}", args.previous_feed_sha256 or ""),
        "Explicit previous feed path and digest required",
    )
    release.no_links(args.previous_feed)
    local = args.previous_feed.read_bytes()
    release.require(
        hashlib.sha256(local).hexdigest() == args.previous_feed_sha256, "Pinned previous feed changed"
    )
    remote = read_current(CURRENT)
    signature = read_current(CURRENT + ".sig")
    release.require(local == remote, "Previous feed is not the current source-side predecessor")
    sys.path.insert(0, str(native.PRIMARY))
    from terminal.strata_control import feed, signing

    verified = feed.verify_and_parse(
        remote, signature, source=CURRENT, trusted=signing.load_trusted_keys(environ={})
    )
    document = json.loads(remote)
    release.require(
        verified.channel == "stable"
        and document.get("feed") == "strata-updates"
        and type(args.sequence) is int
        and args.sequence > verified.sequence,
        "Stable successor sequence must exceed the current source-side feed",
    )
    release.require(
        not document.get("dsv41"), "Unexpected predecessor DeepSeek member; scope review required"
    )
    return {
        "source": CURRENT,
        "sha256": hashlib.sha256(remote).hexdigest(),
        "signatureSHA256": hashlib.sha256(signature).hexdigest(),
        "sequence": verified.sequence,
        "keyId": verified.key_id,
    }


def feed_phase(args: argparse.Namespace, work: Path) -> dict:
    """Rehash inputs, copy fresh assets/entries, sign quietly and verify public trust."""
    release.require(args.no_dsv41_carry, "Explicit --no-dsv41-carry required")
    release.require(args.key == KEY, "Only the canonical external signing key is approved")
    key = release.signing_key_path(args.key)  # Metadata only; the quiet production signer reads it.
    prior = predecessor(args)
    inputs = {
        phase: receipt_assets(
            getattr(args, phase + "_receipt"),
            getattr(args, phase + "_receipt_sha256"),
            phase,
            args.reviewed_commit,
        )
        for phase in ("control", "component")
    }
    release.save(work / "inputs-before.json", {"predecessor": prior, "phases": inputs})
    assets = work / "assets"
    assets.mkdir()
    for phase, identity in inputs.items():
        entries = work / (phase + "-entries")
        entries.mkdir()
        source = Path(identity["assets"])
        for name, checksum in identity["files"].items():
            target = entries / name if name.endswith(".entry.json") else assets / name
            release.require(not target.exists(), "Feed asset collision; preserve previous bytes")
            with (source / name).open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst)
            release.require(
                release.sha(target) == checksum == release.sha(source / name), "Feed copy differs"
            )
    release.command(
        [
            sys.executable,
            "-I",
            "-B",
            native.PRIMARY / "packaging/feed_tool.py",
            "build-feed",
            "--out",
            assets,
            "--key",
            key,
            "--sequence",
            str(args.sequence),
            "--channel",
            "stable",
            "--engine",
            work / "component-entries",
            "--control",
            work / "control-entries",
            "--no-dsv41-carry",
        ],
        quiet=True,
    )
    release.command(
        [
            sys.executable,
            "-I",
            "-B",
            native.PRIMARY / "packaging/feed_tool.py",
            "verify-feed",
            assets / "strata-feed.json",
            "--pub",
            native.PRIMARY / "strata_control/update_keys.json",
        ]
    )
    document = release.load(assets / "strata-feed.json")
    release.require(
        document.get("sequence") == args.sequence and not document.get("dsv41") and not document.get("app"),
        "Unexpected feed scope",
    )
    expected_control = {k: v for k, v in inputs["control"]["entry"].items() if k != "version"}
    release.require(
        document.get("engine", {}).get("packages") == [inputs["component"]["entry"]]
        and document.get("control") == {"version": "0.6.0", "packages": [expected_control]},
        "Feed entries differ from verified receipt inputs",
    )
    for phase in inputs:
        release.require(
            receipt_assets(
                getattr(args, phase + "_receipt"),
                getattr(args, phase + "_receipt_sha256"),
                phase,
                args.reviewed_commit,
            )
            == inputs[phase],
            "Feed source changed",
        )
        for name, checksum in inputs[phase]["files"].items():
            copied = work / (phase + "-entries") / name if name.endswith(".entry.json") else assets / name
            release.require(release.sha(copied) == checksum, "Copied feed input changed")
    release.require(predecessor(args) == prior, "Published predecessor changed during feed preparation")
    return {
        "assets": release.inventory(assets),
        "predecessor": prior,
        "inputs": inputs,
        "sequence": args.sequence,
        "signatureVerified": True,
        "published": False,
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cli = native.parser(__doc__)
    for phase in ("control", "component"):
        cli.add_argument("--" + phase + "-receipt", type=Path)
        cli.add_argument("--" + phase + "-receipt-sha256")
    cli.add_argument("--previous-feed", type=Path)
    cli.add_argument("--previous-feed-sha256")
    cli.add_argument("--previous-source")
    cli.add_argument("--sequence", type=int)
    cli.add_argument("--no-dsv41-carry", action="store_true")
    cli.add_argument("--key", type=Path)
    args = cli.parse_args(argv)
    args.phase = "feed"
    if args.execute:
        native.explicit_attempt(argv, args.attempt)
        release.require(
            args.no_dsv41_carry
            and args.key == KEY
            and args.previous_source == CURRENT
            and args.control_receipt
            and args.component_receipt
            and args.previous_feed
            and args.sequence is not None,
            "Explicit feed inputs and scope required",
        )
    return native.run_controller(args, HERE / "components-feed.py", feed_phase, argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        raise SystemExit(main())
    except Exception as error:
        logging.error(
            "Feed preparation failed (%s); retained attempt; raw error suppressed", type(error).__name__
        )
        raise SystemExit(1) from None
