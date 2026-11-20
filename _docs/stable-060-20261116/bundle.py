"""Curated 0.6.0 public bundle from pinned current proofs; default is inert."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
from typing import BinaryIO

sys.dont_write_bytecode = True
HERE = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location("bundle_original", HERE / "original-engine.py")
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
acceptance, delivery, feed, native, release = (
    original.acceptance,
    original.delivery,
    original.feed,
    original.native,
    original.release,
)
require, artifact = release.require, acceptance.artifact
real = release.module("bundle_real", HERE / "real-model.py")
SCHEMA = "strata.bundle.v1"
TAG = "v0.6.0"
ENGINE = "strata-engine-" + release.NAME + ".tar.gz"
CONTROL = "strata-control-0.6.0-windows-x86_64.zip"
ARCHIVE = "Strata-0.6.0-windows-x64.tar.7z"
TAR = "Strata-0.6.0-windows-x64.tar"
FEED_NAMES = {ENGINE, CONTROL, "strata-feed.json", "strata-feed.json.sig", "install-manifest.txt"}
BASE_NAMES = FEED_NAMES | {
    delivery.INSTALLER,
    "install.ps1",
    "install.sh",
    "RustXNG-0.31.1-source.tar.gz",
    "RustXNG-LICENSE.txt",
    "RustXNG-NOTICE.txt",
    "SOURCE.json",
    "QUALITY.json",
    "INSTALL.md",
}
PUBLIC_NAMES = BASE_NAMES | {"BUNDLE-SHA256SUMS.txt", "SHA256SUMS.txt", ARCHIVE}
INPUTS = (
    "app",
    "extract",
    "install",
    "candidate_qa",
    "installed_qa",
    "control",
    "component",
    "feed",
    "normal_start",
    "original_engine",
    "snapshot",
    "ci",
    "harness",
    "nsis",
    "final_native",
    "real_model",
)
HELPERS = original.HELPERS + ("bundle.py", "real-model.py", "verify.mjs")
QUALIFICATION_ROOT = Path("H:/Strata/private-backups")
SELF_TESTS = {
    "resources-packaged",
    "host-entry-present",
    "ui-dist-present",
    "private-node-runtime",
    "shell-contained-in-kill-on-close-job",
    "host-spawned",
    "host-healthy",
    "ui-served",
    "api-reachable-on-same-origin",
    "control-component-present",
    "control-runs-on-bundled-python",
    "engine-missing-is-a-notice-not-a-download",
    "host-sees-control-and-forbids-acquiring",
}


def table(folder: Path) -> dict:
    require(folder.is_dir(), "Asset directory absent")
    rows = {}
    for path in sorted(folder.iterdir()):
        release.no_links(path)
        require(
            path.is_file() and path.name not in rows and 0 < path.stat().st_size < 2 * release.GIB,
            "Nonregular, empty or oversized public asset",
        )
        rows[path.name] = {"bytes": path.stat().st_size, "sha256": release.sha(path)}
    return rows


def sums(rows: dict) -> bytes:
    return "".join(f"{rows[name]['sha256']}  {name}\n" for name in sorted(rows)).encode()


def safe_public(value: object) -> None:
    text = json.dumps(value, ensure_ascii=False)
    require(
        not re.search(
            r"[A-Za-z]:[/\\]|/Users/|/home/|Bearer |api[_-]?key|access[_-]?token", text, re.I
        ),
        "Private path or credential-shaped public metadata",
    )


def job_of(path: Path) -> dict:
    return release.load(path.parent / "owned-job.json")


def proof_created(path: Path) -> int:
    """Windows birth time of a proof emitted by its producer only after drain."""
    stat = path.stat()
    return getattr(stat, "st_birthtime_ns", stat.st_ctime_ns) // 100 + original.FILETIME_EPOCH


def validate_snapshot(path: Path, expected: str, commit: str) -> dict:
    record = original.proof(path, expected, "publication-snapshot", "strata.publication.v1", commit)
    original.require_helpers(record, ("publish.py", "bundle.py", "original-engine.py"))
    value = release.load(
        acceptance.evidence(record, "results", "snapshot", path.parent / "snapshot.json")
    )
    require(
        value["stable"]["tag_name"] == "v0.5.9"
        and value["latest"] == value["stable"]
        and value["stable"]["draft"] is False
        and value["stable"]["prerelease"] is False,
        "Snapshot must observe the actual prior stable and canonical Latest",
    )
    require(
        value["predecessor"]["sha256"] == record["results"]["previousFeed"]["sha256"]
        and value["predecessor"]["signatureSHA256"]
        == record["results"]["previousSignature"]["sha256"],
        "Snapshot predecessor bytes differ",
    )
    verify_feed_bytes(
        Path(record["results"]["previousFeed"]["path"]),
        Path(record["results"]["previousSignature"]["path"]),
        value["predecessor"]["sequence"],
    )
    return record


def verify_feed_bytes(path: Path, signature: Path, sequence: int) -> dict:
    sys.path.insert(0, str(native.PRIMARY))
    from terminal.strata_control import feed as parser, signing

    verified = parser.verify_and_parse(
        path.read_bytes(),
        signature.read_bytes(),
        source=feed.CURRENT,
        trusted=signing.load_trusted_keys(environ={}),
    )
    require(
        verified.channel == "stable" and verified.sequence == sequence,
        "Feed sequence/channel differs",
    )
    # Verify again through the actual packaged interpreter and its own bundled
    # production trust, rather than accepting a source-checkout-only result.
    code = (
        "import sys;sys.path.insert(0,sys.argv[1]);from pathlib import Path;"
        "from terminal.strata_control import feed,signing;"
        "v=feed.verify_and_parse(Path(sys.argv[2]).read_bytes(),Path(sys.argv[3]).read_bytes(),"
        "source=sys.argv[4],trusted=signing.load_trusted_keys(environ={}));"
        "assert v.channel=='stable' and v.sequence==int(sys.argv[5]);print('verified')"
    )
    result = release.command(
        [
            delivery.APP / "control/python/python.exe",
            "-I",
            "-B",
            "-c",
            code,
            delivery.APP / "control",
            path,
            signature,
            feed.CURRENT,
            str(sequence),
        ]
    )
    require(result.strip() == "verified", "Packaged Control feed verification failed")
    return release.load(path)


def validate_feed(
    path: Path, expected: str, commit: str, snapshot: dict, control: dict, component: dict
) -> dict:
    record = delivery.receipt(path, expected, "feed", commit)
    acceptance.successful_job(job_of(path), record, path.parent)
    assets = path.parent / "assets"
    require(
        set(record["assets"]) == FEED_NAMES and release.inventory(assets) == record["assets"],
        "Feed public membership or bytes differ",
    )
    require(
        record.get("signatureVerified") is True and record.get("published") is False,
        "Expected signed unpublished feed",
    )
    prior = release.load(Path(snapshot["results"]["snapshot"]["path"]))["predecessor"]
    require(
        record["predecessor"] == prior and record["sequence"] > prior["sequence"],
        "Changed predecessor",
    )
    require(
        record["inputs"] == {"control": control, "component": component},
        "Feed source inputs differ",
    )
    command = job_of(path)["operationExecution"]["command"]
    require(
        original.option(command, "--previous-feed") == snapshot["results"]["previousFeed"]["path"]
        and original.option(command, "--previous-feed-sha256") == prior["sha256"],
        "Unbound predecessor snapshot",
    )
    document = verify_feed_bytes(
        assets / "strata-feed.json", assets / "strata-feed.json.sig", record["sequence"]
    )
    require(
        document.get("engine", {}).get("packages") == [component["entry"]]
        and document.get("control")
        == {
            "version": "0.6.0",
            "packages": [{k: v for k, v in control["entry"].items() if k != "version"}],
        }
        and not any(
            document.get(k) for k in ("app", "dsv41", "models", "shape_engines", "drafters")
        ),
        "Feed scope or entries differ",
    )
    require(
        (assets / "install-manifest.txt").read_text()
        == f"windows-x86_64 0.6.0 {CONTROL} {control['entry']['sha256']} {control['entry']['size']}\n",
        "Terminal install manifest differs",
    )
    return record


def script_input(name: str, commit: str) -> tuple[bytes, dict]:
    require(name in ("install.ps1", "install.sh"), "Unapproved source script")
    relative = "packaging/" + name
    path = native.PRIMARY / relative
    release.no_links(path)
    blob = release.git("rev-parse", commit + ":" + relative)
    require(re.fullmatch(r"[a-f0-9]{40}", blob), "Reviewed script blob absent")
    canonical = delivery.canonical_blob(blob)
    physical = path.read_bytes()
    # Git checkout CRLF is permitted only as complete LF->CRLF conversion.
    require(
        physical in (canonical, canonical.replace(b"\n", b"\r\n")) and b"\r" not in canonical,
        "Source script differs beyond documented checkout EOL conversion",
    )
    return canonical, {
        "source": relative,
        "gitBlob": blob,
        "canonicalSHA256": hashlib.sha256(canonical).hexdigest(),
        "physicalSHA256": release.sha(path),
        "checkoutEOL": "LF" if physical == canonical else "CRLF",
    }


def validate_self_test(report: dict) -> None:
    rows = report.get("steps", [])
    require(
        report.get("schemaVersion") == 1
        and report.get("kind") == "strata-desktop-self-test"
        and report.get("carrier") == "tauri"
        and report.get("ok") is True
        and report.get("error") == ""
        and len(rows) == len(SELF_TESTS)
        and {r.get("name") for r in rows} == SELF_TESTS
        and all(r.get("ok") is True for r in rows),
        "Incomplete named packaged self-test",
    )


def normal_pair(record: dict, path: Path, engine: dict, engine_path: Path, commit: str) -> None:
    acceptance.successful_job(job_of(path), record, path.parent)
    normal_command = job_of(path)["operationExecution"]["command"]
    engine_command = job_of(engine_path)["operationExecution"]["command"]
    require(
        Path(normal_command[2]) == HERE / "normal-start.py"
        and original.owners(normal_command) == original.owners(engine_command),
        "Final normal owner differs",
    )
    require(
        job_of(path)["root_creation_filetime"] > engine["drainedAtFiletime"],
        "Normal observation must follow original-engine drain",
    )
    for key in ("installerSHA256", "choicesSHA256", "profileIdentitySHA256"):
        require(record.get(key) == engine.get(key), "Normal/original-engine identity differs")
    for key in (
        "normalStartup",
        "originalChoicesAndSkillsUnchanged",
        "originalEffectiveProfileUnchanged",
        "modelUnloaded",
        "observationOnly",
    ):
        require(record.get(key) is True, "Missing final normal observation")
    require(
        record.get("settingsWritten") is False
        and record.get("persistentAppStartedByThisController") is False
        and record.get("activeTurns") == 0,
        "Unexpected normal observation action",
    )
    require(
        str(record["shellPID"]) == original.option(normal_command, "--shell-pid")
        and str(record["hostPID"]) == original.option(normal_command, "--host-pid"),
        "Normal PID mismatch",
    )
    require(
        original.option(normal_command, "--choices-sha256") == engine["choicesSHA256"],
        "Normal choices command differs",
    )


def qualification(
    actual: dict, real_sha: str, commit: str, path: Path | None, expected: str | None
) -> dict | None:
    """Only root's pinned, narrow model-declined reuse adjudication may be admitted."""
    if actual["outcome"] == "MECHANICS_PASSED":
        require(
            all(v == "PASS" for v in actual["checks"].values())
            and path is None
            and expected is None,
            "Complete mechanics require no qualification override",
        )
        return None
    require(
        actual["outcome"] == "PARTIAL_NOT_REUSABLE"
        and set(actual["checks"]) == set(real.CHECKS)
        and all(
            v == ("NOT_REUSABLE" if k in ("learning", "autoApply") else "PASS")
            for k, v in actual["checks"].items()
        ),
        "Unapproved partial/FAIL/NOT_RUN model outcome",
    )
    require(
        path is not None and expected is not None and path.is_relative_to(QUALIFICATION_ROOT),
        "Explicit private root qualification required",
    )
    value = delivery.pinned(path, expected)
    require(
        set(value) == {"schema", "sourceCommit", "realModelSHA256", "decision", "rationale"}
        and value["schema"] == "strata.publication-qualification.v1"
        and value["sourceCommit"] == commit
        and value["realModelSHA256"] == real_sha
        and value["decision"] == "ALLOW_LIMITED_PUBLICATION"
        and isinstance(value["rationale"], str)
        and 0 < len(value["rationale"].strip()) <= 1000,
        "Stale, mismatched or malformed root qualification",
    )
    return artifact(path, expected)


def admission(args: argparse.Namespace) -> dict:
    commit = args.reviewed_commit
    records = {
        n: artifact(getattr(args, n + "_receipt"), getattr(args, n + "_receipt_sha256"))
        for n in INPUTS
    }
    app = delivery.app_build(args)
    require(
        app.get("compilerJobs") == 5 and app.get("authenticodeSigned") is False,
        "App compiler/signing scope differs",
    )
    raw = {
        n: delivery.receipt(
            Path(records[n]["path"]), records[n]["sha256"], n.replace("_", "-"), commit
        )
        for n in ("extract", "install", "candidate_qa", "installed_qa", "normal_start")
    }
    for n, value in raw.items():
        acceptance.successful_job(
            job_of(Path(records[n]["path"])), value, Path(records[n]["path"]).parent
        )
        require(
            value.get("installerSHA256") == app["installerSHA256"], "Mismatched installer receipt"
        )
    extract = raw["extract"]
    validate_self_test(extract["selfTest"])
    extracted_root = Path(records["extract"]["path"]).parent / "extracted"
    require(
        delivery.payload(extracted_root)
        == extract["payload"]
        == raw["install"]["payload"]
        == delivery.payload(delivery.APP)
        and extract.get("onlyOfficialNSISMarkerChanged") is True,
        "Extraction/installed payload differs",
    )
    unbundled = Path(app["shell"]).read_bytes()
    marker = b"__TAURI_BUNDLE_TYPE_VAR_UNK"
    require(
        unbundled.count(marker) == 1
        and (extracted_root / "strata-desktop.exe").read_bytes()
        == unbundled.replace(marker, b"__TAURI_BUNDLE_TYPE_VAR_NSS", 1)
        and release.sha(extracted_root / "strata-desktop.exe") == extract["nsisShellSHA256"],
        "NSIS marker binding differs",
    )
    require(
        raw["install"].get("installed") is True
        and raw["install"].get("exactPayloadMatch") is True
        and raw["install"].get("userFilesUnchanged") is True,
        "Incomplete protected installation",
    )
    for name, target in (("candidate_qa", "candidate"), ("installed_qa", "installed")):
        value = raw[name]
        report = delivery.pinned(
            Path(records[name]["path"]).parent / "qa-home/result.json", value["resultSHA256"]
        )
        require(value["qa"] == report and value["target"] == target, "WebView2 proof body differs")
        delivery.validate_qa(report, target, commit, app["installerSHA256"])
    components = {
        n: feed.receipt_assets(Path(records[n]["path"]), records[n]["sha256"], n, commit)
        for n in ("control", "component")
    }
    snap = validate_snapshot(
        Path(records["snapshot"]["path"]), records["snapshot"]["sha256"], commit
    )
    snapshot_created = snap["drainedAtFiletime"]
    require(
        snapshot_created < job_of(Path(records["component"]["path"]))["root_creation_filetime"],
        "Snapshot must precede signing",
    )
    feed_record = validate_feed(
        Path(records["feed"]["path"]),
        records["feed"]["sha256"],
        commit,
        snap,
        components["control"],
        components["component"],
    )
    proofs = {
        n: acceptance.validate_acceptance(
            Path(records[n]["path"]), records[n]["sha256"], n.replace("_", "-"), commit
        )
        for n in ("ci", "harness", "nsis", "final_native")
    }
    actual = real.validate_real_model(
        Path(records["real_model"]["path"]), records["real_model"]["sha256"], commit
    )
    require(
        actual["inputs"]["install"] == records["install"]
        and actual["inputs"]["component"] == records["component"],
        "Genuine-model input lineage differs",
    )
    allowance = qualification(
        actual,
        records["real_model"]["sha256"],
        commit,
        args.qualification_receipt,
        args.qualification_receipt_sha256,
    )
    if allowance is not None:
        records["qualification"] = allowance
    engine_path = Path(records["original_engine"]["path"])
    engine = original.validate_original_engine(
        engine_path, records["original_engine"]["sha256"], commit
    )
    require(
        engine["inputs"]["install"] == records["install"]
        and engine["inputs"]["component"] == records["component"],
        "Original engine input lineage differs",
    )
    normal_pair(
        raw["normal_start"], Path(records["normal_start"]["path"]), engine, engine_path, commit
    )
    require(
        job_of(engine_path)["root_creation_filetime"]
        > proof_created(Path(records["real_model"]["path"])),
        "Original engine must follow genuine isolated QA drain",
    )
    for name in ("app", "extract", "installed_qa"):
        require(actual["inputs"][name] == records[name], "Genuine installed source lineage differs")
    return {
        "pins": records,
        "app": app,
        "raw": raw,
        "components": components,
        "snapshot": snap,
        "feed": feed_record,
        "proofs": proofs,
        "real": actual,
        "engine": engine,
        "extracted": extracted_root,
    }


def stream_members(stream: BinaryIO, expected: dict) -> dict:
    """Hash every regular flat tar member; never extract filesystem paths."""
    seen = {}
    with tarfile.open(fileobj=stream, mode="r|") as archive:
        for info in archive:
            require(
                info.isfile()
                and info.name in expected
                and info.name not in seen
                and re.fullmatch(r"[A-Za-z0-9._-]+", info.name)
                and info.size == expected[info.name]["bytes"]
                and info.size < 2 * release.GIB
                and not info.pax_headers,
                "Unsafe, duplicate or unexpected tar member",
            )
            digest, count = hashlib.sha256(), 0
            with archive.extractfile(info) as source:
                while block := source.read(1024 * 1024):
                    digest.update(block)
                    count += len(block)
                    require(count <= info.size, "Tar member overflow")
            seen[info.name] = {"bytes": count, "sha256": digest.hexdigest()}
    require(seen == expected, "Tar member bytes/SHA or population differ")
    return seen


def verify_archive(path: Path, members: dict, work: Path) -> None:
    listing = release.command([delivery.SEVEN, "l", "-slt", "-ba", path])
    require(
        re.findall(r"^Path = (.+)$", listing, re.M) == [TAR]
        and re.findall(r"^Method = (.+)$", listing, re.M) == ["Copy"],
        "Expected one Copy tar container",
    )
    with (work / "decoder.log").open("ab") as errors:
        child = subprocess.Popen(
            [str(delivery.SEVEN), "x", "-so", str(path)],
            stdout=subprocess.PIPE,
            stderr=errors,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        try:
            stream_members(child.stdout, members)
            # Drain the container's zero tar padding; reject nonzero hidden data.
            while block := child.stdout.read(1024 * 1024):
                require(not any(block), "Unexpected trailing archive data")
            require(child.wait(timeout=60) == 0, "7z decoder failed")
        finally:
            child.stdout.close()
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=10)


def projection(state: dict, commit: str, scripts: dict) -> tuple[dict, dict]:
    native_proof = state["proofs"]["final_native"]
    equality = release.load(Path(native_proof["results"]["equality"]["path"]))
    native_build = release.load(Path(native_proof["inputs"]["nativeBuild"]["path"]))
    rust = release.load(state["extracted"] / "harness/rustxng/SOURCE.json")
    observations = release.load(Path(state["real"]["results"]["observations.json"]["path"]))
    require(
        rust.get("version") == "0.31.1" and rust.get("profile") == "full",
        "Exact RustXNG Full required",
    )
    source = {
        "schema": 1,
        "product": "0.6.0",
        "releaseTag": TAG,
        "sourceCommit": commit,
        "upstream": {
            "tag": "v0.2.15",
            "tagObject": "0a80d5e5ed2f3c7602efb9d697be71a6321abf62",
            "tagTarget": "c77572806996349b49995e4f98d41b5294d36a21",
            "frozenIntakeCutoff": "c77572806996349b49995e4f98d41b5294d36a21",
            "primaryObservationJST": "2026-11-16 18:31",
        },
        "installer": {
            "name": delivery.INSTALLER,
            "sha256": state["app"]["installerSHA256"],
            "unbundledShellSHA256": state["app"]["shellSHA256"],
            "nsisShellSHA256": state["raw"]["extract"]["nsisShellSHA256"],
            "onlyOfficialNSISMarkerChanged": True,
        },
        "native": {
            "version": "0.6.0",
            "sourceCommit": commit,
            "nativeBuildSHA256": native_proof["inputs"]["nativeBuild"]["sha256"],
            "sourceDigest": equality["currentSourceDigest"],
            "binaries": equality["binaries"],
            "ggmlIdentity": native_build["recipe"]["ggml"],
            "windowsProfile": {
                "name": acceptance.PROFILE,
                "sha256": native_proof["inputs"]["cudaProfile"]["sha256"],
            },
            "compilerKeySHA256": equality["profileBindings"]["compilerKey"],
            "manifestSHA256": state["components"]["component"]["entry"]["manifest_sha256"],
            "archiveSHA256": state["components"]["component"]["entry"]["sha256"],
            "productionSignatureVerified": True,
        },
        "control": {
            "version": "0.6.0",
            "sourceCommit": commit,
            "archiveSHA256": state["components"]["control"]["entry"]["sha256"],
        },
        "scripts": scripts,
        "nodeVersion": "24.22.3",
        "nodeSHA256": release.sha(release.NODE),
        "toolchainProfile": "sm120-cuda133-msvc1444",
        "compilerJobs": 5,
        "rustxng": {k: rust[k] for k in ("version", "profile", "sourceCommit")},
        "authenticodeSigned": False,
        "releaseQualified": False,
    }
    ci = state["proofs"]["ci"]
    workflows = {
        name: {
            "status": "PASS",
            "jobs": [j["name"] for j in release.load(Path(row["path"]))["jobs"]],
        }
        for name, row in ci["results"].items()
        if name in acceptance.WORKFLOWS
    }
    quality = {
        "schema": 1,
        "sourceCommit": commit,
        "releaseQualified": False,
        "authenticodeSigned": False,
        "componentAndFeedProductionSignatures": "VERIFIED",
        "ci": {
            "workflows": workflows,
            "limits": "Compile/host tests do not qualify hardware inference; Metal link is not Mac inference.",
        },
        "harness": {
            "counters": state["proofs"]["harness"]["counters"],
            "scope": acceptance.SCOPES["harness"],
        },
        "nsis": {"scenarios": sorted(acceptance.SCENARIOS), "status": "PASS"},
        "packagedSelfTests": sorted(SELF_TESTS),
        "webView2FixtureChecks": sorted(delivery.QA_CHECKS),
        "fixtureUI": "PASS; synthetic model responses and actual RustXNG Full searches",
        "native": {
            "scope": equality["claim"],
            "numericalChecks": sorted(acceptance.GPU_NAMES),
            "genericSpillDefault": "Two retained IQ2 failures",
            "explicitWindowsProfile": "PASS",
            "binariesEqual": equality["binaries"],
            "sourceDigest": equality["currentSourceDigest"],
        },
        "realModel": {
            "model": "Selected Qwen3.5 9B Q4_K_M reference",
            "runtime": "PrismML 0.4.1",
            "actualModel": True,
            "syntheticResponses": False,
            "checks": state["real"]["checks"],
            "outcome": state["real"]["outcome"],
            "quality": state["real"]["quality"],
            "auditCounts": {
                k: observations["audit"][k]
                for k in ("total_mismatches", "total_duplicates", "total_missing", "total_failed")
            },
            "researchSearches": len(observations["research"]["queries"]),
            "researchPageReads": len(observations["research"]["urls"]),
            "publicationQualification": (
                "Root permitted limited publication after the model declined reusable learning; learning and automatic application remain NOT_REUSABLE."
                if "qualification" in state["pins"]
                else "Bounded requested mechanics passed; general quality remains unqualified."
            ),
        },
        "originalProfile": {
            "nativeVersion": "0.6.0",
            "nativeSource": commit,
            "oldComponentsPreserved": True,
            "choicesSkillsAndEffectiveProfilePreserved": True,
            "normalNoDebugStartup": True,
            "modelUnloaded": True,
        },
        "generalModelQuality": "UNQUALIFIED",
        "fullWindow": "UNQUALIFIED",
        "gpuConcurrency": "UNQUALIFIED",
    }
    quality["operationRAM"] = {
        name: {
            k: value[k]
            for k in (
                "allowLowRam",
                "nominalFloorBytes",
                "effectiveFloorBytes",
                "minAvailableBytes",
                "nominalBreachSamples",
                "samples",
            )
        }
        for name, pin in state["pins"].items()
        if name != "qualification"
        for value in [release.load(Path(pin["path"]))["ramPolicy"]]
    }
    safe_public(source)
    safe_public(quality)
    return source, quality


def phase(args: argparse.Namespace, work: Path) -> dict:
    state = admission(args)
    assets = work / "assets"
    require(not assets.exists(), "Fresh public assets required")
    source_files = {
        name: Path(state["pins"]["feed"]["path"]).parent / "assets" / name for name in FEED_NAMES
    }
    source_files[delivery.INSTALLER] = Path(state["app"]["installer"])
    search = state["extracted"] / "harness/rustxng"
    rust = release.load(search / "SOURCE.json")
    for old, new in (
        ("corresponding-source.tar.gz", "RustXNG-0.31.1-source.tar.gz"),
        ("LICENSE", "RustXNG-LICENSE.txt"),
        ("NOTICE", "RustXNG-NOTICE.txt"),
    ):
        require(
            release.sha(search / old) == rust["files"][old]["sha256"],
            "RustXNG corresponding source/license changed",
        )
        source_files[new] = search / old
    size = sum(p.stat().st_size for p in source_files.values())
    require(
        shutil.disk_usage(work).free >= 10 * release.GIB + 3 * size + 64 * 1024**2,
        "Estimated copy/tar/Copy-7z space refused",
    )
    assets.mkdir()
    for name, source in source_files.items():
        release.no_links(source)
        with source.open("rb") as src, (assets / name).open("xb") as dst:
            shutil.copyfileobj(src, dst, 1024 * 1024)
        require(release.sha(assets / name) == release.sha(source), "Curated copy differs")
    scripts = {}
    for name in ("install.ps1", "install.sh"):
        content, scripts[name] = script_input(name, args.reviewed_commit)
        with (assets / name).open("xb") as stream:
            stream.write(content)
    source, quality = projection(state, args.reviewed_commit, scripts)
    release.save(assets / "SOURCE.json", source)
    release.save(assets / "QUALITY.json", quality)
    install = (
        "# Hakua 0.6.0 for Windows x64\n\n"
        "Run Strata-0.6.0-windows-x64-setup.exe for the per-user desktop application. "
        "Finish requests, unload models and close the app before an overwrite. Preserve a private backup.\n\n"
        "The identical installer is inside Strata-0.6.0-windows-x64.tar.7z. "
        "The signed CUDA engine is separate: import its tar.gz in the app, then choose it under Engines. "
        "The app installer has no Windows Authenticode signature. Component and feed production signatures are independently verified.\n\n"
        "install.ps1 installs the Windows Terminal from this release's signed feed. "
        "This release includes no Linux or Mac Terminal package; install.sh is supplied as corresponding installer source. "
        "RustXNG 0.31.1 Full corresponding source, LICENSE and NOTICE are included.\n\n"
        "Verify SHA256SUMS.txt for public files and BUNDLE-SHA256SUMS.txt for flat archive members. "
        "Read SOURCE.json and QUALITY.json for actual source and bounded evidence. "
        "General model quality, full-window inference and GPU concurrency remain unqualified; releaseQualified=false.\n"
    )
    with (assets / "INSTALL.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(install)
    members = table(assets)
    require(set(members) == BASE_NAMES, "Curated base membership differs")
    (assets / "BUNDLE-SHA256SUMS.txt").write_bytes(sums(members))
    members = table(assets)
    tar = work / TAR
    with tarfile.open(tar, "x", format=tarfile.USTAR_FORMAT) as archive:
        for name in sorted(members):
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = members[name]["bytes"], 0o644, 0
            with (assets / name).open("rb") as stream:
                archive.addfile(info, stream)
    native.logged(
        [delivery.SEVEN, "a", "-t7z", "-mx=0", "-mmt=1", assets / ARCHIVE, tar],
        work,
        "archive",
        600,
    )
    verify_archive(assets / ARCHIVE, members, work)
    (assets / "SHA256SUMS.txt").write_bytes(sums(table(assets)))
    require(set(table(assets)) == PUBLIC_NAMES, "Unexpected public release membership")
    require(admission(args)["pins"] == state["pins"], "Input proofs changed during bundle")
    inputs = {**original.helper_inputs(HELPERS), **state["pins"]}
    for name in ("install.ps1", "install.sh"):
        inputs["source:" + name] = artifact(
            native.PRIMARY / "packaging" / name, scripts[name]["physicalSHA256"]
        )
    return {
        "schema": SCHEMA,
        "inputs": inputs,
        "results": {name: artifact(assets / name) for name in PUBLIC_NAMES},
        "members": members,
        "assets": table(assets),
        "method": "USTAR + 7z Copy mx=0 mmt=1",
        "installerSHA256": state["app"]["installerSHA256"],
        "published": False,
    }


def validate_bundle(path: Path, expected: str, commit: str) -> dict:
    record = original.proof(path, expected, "bundle", SCHEMA, commit)
    original.require_helpers(record, HELPERS)
    require(
        record.get("published") is False
        and set(record["assets"]) == PUBLIC_NAMES
        and table(path.parent / "assets") == record["assets"],
        "Bundle assets changed",
    )
    command = job_of(path)["operationExecution"]["command"]
    require(Path(command[2]) == HERE / "bundle.py", "Wrong bundle worker")
    args = argparse.Namespace(
        reviewed_commit=commit, qualification_receipt=None, qualification_receipt_sha256=None
    )
    for name in INPUTS:
        row = record["inputs"][name]
        option_name = "--" + name.replace("_", "-") + "-receipt"
        require(
            original.option(command, option_name) == row["path"]
            and original.option(command, option_name + "-sha256") == row["sha256"],
            "Bundle input command differs",
        )
        setattr(args, name + "_receipt", Path(row["path"]))
        setattr(args, name + "_receipt_sha256", row["sha256"])
    if "qualification" in record["inputs"]:
        row = record["inputs"]["qualification"]
        require(
            original.option(command, "--qualification-receipt") == row["path"]
            and original.option(command, "--qualification-receipt-sha256") == row["sha256"],
            "Unbound root qualification",
        )
        args.qualification_receipt, args.qualification_receipt_sha256 = (
            Path(row["path"]),
            row["sha256"],
        )
    else:
        require(
            not any(v.startswith("--qualification-receipt") for v in command),
            "Unrecorded qualification input",
        )
    state = admission(args)
    source = release.load(path.parent / "assets/SOURCE.json")
    scripts = {name: script_input(name, commit)[1] for name in ("install.ps1", "install.sh")}
    require(
        (source, release.load(path.parent / "assets/QUALITY.json"))
        == projection(state, commit, scripts),
        "Public metadata projection changed",
    )
    require(
        record["installerSHA256"]
        == state["app"]["installerSHA256"]
        == record["assets"][delivery.INSTALLER]["sha256"]
        == record["members"][delivery.INSTALLER]["sha256"],
        "Inner/outer installer mismatch",
    )
    expected_members = {
        n: row for n, row in record["assets"].items() if n not in (ARCHIVE, "SHA256SUMS.txt")
    }
    require(record["members"] == expected_members, "Bundle member manifest differs")
    require(
        (path.parent / "assets/BUNDLE-SHA256SUMS.txt").read_bytes()
        == sums({n: r for n, r in expected_members.items() if n != "BUNDLE-SHA256SUMS.txt"})
        and (path.parent / "assets/SHA256SUMS.txt").read_bytes()
        == sums({n: r for n, r in record["assets"].items() if n != "SHA256SUMS.txt"}),
        "Checksum population differs",
    )
    return record


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cli = native.parser(__doc__)
    for name in INPUTS:
        cli.add_argument("--" + name.replace("_", "-") + "-receipt", type=Path)
        cli.add_argument("--" + name.replace("_", "-") + "-receipt-sha256")
    cli.add_argument("--qualification-receipt", type=Path)
    cli.add_argument("--qualification-receipt-sha256")
    args = cli.parse_args(argv)
    args.phase = "bundle"
    if args.execute:
        require("--attempt" in argv, "Explicit fresh attempt required")
    result = native.run_controller(args, HERE / "bundle.py", phase, argv)
    if args.execute and not args.worker:
        original.finalize(
            release.OUT / (args.phase + "-" + args.attempt),
            args.phase,
            SCHEMA,
            args.reviewed_commit,
            validate_bundle,
        )
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        raise SystemExit(main())
    except Exception as error:
        logging.error(
            "Bundle refused (%s); retained attempt; raw error suppressed", type(error).__name__
        )
        raise SystemExit(1) from None
