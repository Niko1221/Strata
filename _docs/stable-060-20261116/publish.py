"""Individually explicit guarded publication phases; default performs no I/O.

Normal gh credential-manager authentication is used only for metadata/mutations.
Public content readback uses a separate unauthenticated streaming HTTP client.
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
import subprocess
import sys
import urllib.parse
import urllib.request

sys.dont_write_bytecode = True
HERE = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location("publication_bundle", HERE / "bundle.py")
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)
original, acceptance, delivery, native, release = (
    bundle.original,
    bundle.acceptance,
    bundle.delivery,
    bundle.native,
    bundle.release,
)
require, artifact = release.require, acceptance.artifact
GH, REPO, TAG = acceptance.GH, "zapabob/Strata", bundle.TAG
SCHEMA = "strata.publication.v1"
ACTIONS = ("snapshot", "tag", "draft", "upload", "verify", "publish", "alias", "readback")
PREVIOUS = {
    "draft": "tag",
    "upload": "draft",
    "verify": "upload",
    "publish": "verify",
    "alias": "publish",
    "readback": "alias",
}
HELPERS = ("publish.py", "bundle.py", "original-engine.py")


def gh(*args: str) -> str:
    # release.command(quiet=True) deliberately discards stdout; metadata needs
    # captured output while credential-manager diagnostics must stay suppressed.
    result = subprocess.run(
        [str(GH), *args],
        cwd=native.PRIMARY,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        timeout=1500,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    require(
        result.returncode == 0 and len(result.stdout) <= 16 * 1024**2,
        "GitHub command failed or response exceeded bound; raw diagnostics suppressed",
    )
    return result.stdout.strip()


def api(path: str) -> object:
    require(not path.startswith(("/", "http")), "Repository-relative GitHub API path required")
    return json.loads(gh("api", "repos/" + REPO + "/" + path))


def git_mutate(argv: list[str]) -> None:
    require(
        release.git("remote", "get-url", "origin").strip().removesuffix(".git")
        == "https://github.com/" + REPO,
        "Publication remote identity differs",
    )
    release.command(["git", *argv], quiet=True, timeout=300)


def local_ref(name: str) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", "refs/tags/" + name],
        cwd=native.PRIMARY,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=30,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    require(result.returncode in (0, 1), "Cannot observe local tag")
    value = result.stdout.decode().strip() if result.returncode == 0 else None
    require(value is None or re.fullmatch(r"[a-f0-9]{40}", value), "Malformed local ref")
    return value


def tags() -> dict:
    rows = api("git/refs/tags")
    require(isinstance(rows, list) and len(rows) < 100, "Bounded explicit tag inventory required")
    require(len({r["ref"] for r in rows}) == len(rows), "Duplicate remote refs")
    return {r["ref"]: r["object"] for r in rows}


def releases() -> dict:
    rows = api("releases?per_page=100")
    require(
        isinstance(rows, list)
        and len(rows) < 100
        and len({r["tag_name"] for r in rows}) == len(rows),
        "Bounded unambiguous release inventory required",
    )
    return {r["tag_name"]: r for r in rows}


def identity(row: dict) -> dict:
    assets = row["assets"]
    require(len({r["name"] for r in assets}) == len(assets), "Duplicate server asset name")
    return {k: row[k] for k in ("id", "tag_name", "draft", "prerelease")} | {
        "assets": {
            r["name"]: {k: r.get(k) for k in ("id", "size", "digest", "state")} for r in assets
        }
    }


def remote_assets(row: dict, expected: dict, *, complete: bool = True) -> set[str]:
    observed = identity(row)["assets"]
    require(set(observed) <= set(expected), "Extra remote asset")
    if complete:
        require(set(observed) == set(expected), "Missing remote asset")
    for name, value in observed.items():
        require(
            value["state"] == "uploaded"
            and value["size"] == expected[name]["bytes"]
            and value["digest"] == "sha256:" + expected[name]["sha256"],
            "Existing/partial asset collision: " + name,
        )
    return set(expected) - set(observed)


def remote_tag(commit: str) -> dict:
    refs = tags()
    row = refs.get("refs/tags/" + TAG)
    require(row is not None and row.get("type") == "tag", "Immutable annotated remote tag required")
    tag = api("git/tags/" + row["sha"])
    require(
        tag.get("tag") == TAG
        and tag.get("object", {}).get("type") == "commit"
        and tag["object"]["sha"] == commit,
        "Wrong immutable tag source",
    )
    require(local_ref(TAG) == row["sha"], "Local/remote immutable tag differs")
    return {"object": row["sha"], "commit": commit}


class PublicRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self, req: object, fp: object, code: int, msg: str, headers: object, newurl: str
    ) -> object:
        public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def public_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    require(
        parsed.scheme == "https"
        and parsed.username is None
        and parsed.password is None
        and parsed.port in (None, 443)
        and (
            parsed.hostname == "github.com"
            or (parsed.hostname or "").endswith(".githubusercontent.com")
        ),
        "Foreign anonymous download destination",
    )


def public_open(url: str) -> object:
    public_url(url)
    # No environment proxy credentials, cookie jar, Authorization or gh download.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), PublicRedirect())
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Hakua-release-byte-verification", "Accept-Encoding": "identity"},
    )
    return opener.open(request, timeout=60)


def public_small(url: str) -> bytes:
    with public_open(url) as response:
        require(response.status == 200, "Anonymous response refused")
        value = response.read(1024 * 1024 + 1)
        require(len(value) <= 1024 * 1024, "Oversized public metadata")
        return value


def predecessor(work: Path) -> dict:
    payload = public_small(bundle.feed.CURRENT)
    signature = public_small(bundle.feed.CURRENT + ".sig")
    for name, value in (("previous-feed.json", payload), ("previous-feed.json.sig", signature)):
        with (work / name).open("xb") as stream:
            stream.write(value)
    document = json.loads(payload)
    bundle.verify_feed_bytes(
        work / "previous-feed.json", work / "previous-feed.json.sig", document["sequence"]
    )
    return {
        "source": bundle.feed.CURRENT,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "signatureSHA256": hashlib.sha256(signature).hexdigest(),
        "sequence": document["sequence"],
        "keyId": json.loads(signature)["key_id"],
    }


def unchanged(snapshot: dict, *, alias_changed: bool = False) -> None:
    before = release.load(Path(snapshot["results"]["snapshot"]["path"]))
    rows = releases()
    for tag, value in before["releases"].items():
        if alias_changed and tag == "latest":
            continue
        require(
            tag in rows and identity(rows[tag]) == value,
            "Historical release or asset changed: " + tag,
        )
    refs = tags()
    for ref, value in before["tags"].items():
        if alias_changed and ref == "refs/tags/latest":
            continue
        require(refs.get(ref) == value, "Historical tag changed")


def inputs(args: argparse.Namespace) -> tuple[dict, dict, dict | None, dict]:
    pins = original.helper_inputs(HELPERS)
    if args.action == "snapshot":
        return pins, {}, None, {}
    for name in ("bundle", "snapshot"):
        pins[name] = artifact(
            getattr(args, name + "_receipt"), getattr(args, name + "_receipt_sha256")
        )
    built = bundle.validate_bundle(
        args.bundle_receipt, args.bundle_receipt_sha256, args.reviewed_commit
    )
    snapshot = bundle.validate_snapshot(
        args.snapshot_receipt, args.snapshot_receipt_sha256, args.reviewed_commit
    )
    require(built["inputs"]["snapshot"] == pins["snapshot"], "Bundle used a different snapshot")
    prior = {}
    if args.action in PREVIOUS:
        pins["prior"] = artifact(args.prior_receipt, args.prior_receipt_sha256)
        prior = validate_publication(
            args.prior_receipt,
            args.prior_receipt_sha256,
            args.reviewed_commit,
            action=PREVIOUS[args.action],
            recursive=False,
        )
        require(
            prior["inputs"]["bundle"] == pins["bundle"]
            and prior["inputs"]["snapshot"] == pins["snapshot"],
            "Prior publication input lineage differs",
        )
    return pins, built, snapshot, prior


def release_row() -> dict:
    import time

    for _ in range(5):
        matches = releases()
        if TAG in matches:
            return matches[TAG]
        time.sleep(2)
    require(False, "Expected exact formal release")


def snapshot_phase(work: Path) -> dict:
    require(gh("api", "user", "--jq", ".login").strip() == "zapabob", "Wrong publication owner")
    rows, refs = releases(), tags()
    require(
        TAG not in rows and "refs/tags/" + TAG not in refs and local_ref(TAG) is None,
        "Existing immutable tag/release must be preserved",
    )
    require("v0.5.9" in rows and "refs/tags/latest" in refs, "Prior stable/alias absent")
    latest = api("releases/latest")
    require(
        identity(latest) == identity(rows["v0.5.9"]), "Canonical stable changed before snapshot"
    )
    previous = predecessor(work)
    value = {
        "releases": {t: identity(r) for t, r in rows.items()},
        "tags": refs,
        "latest": identity(latest),
        "stable": identity(rows["v0.5.9"]),
        "localAlias": local_ref("latest"),
        "predecessor": previous,
    }
    require(
        {t: identity(r) for t, r in releases().items()} == value["releases"]
        and tags() == refs
        and identity(api("releases/latest")) == value["latest"],
        "Snapshot changed during observation",
    )
    release.save(work / "snapshot.json", value)
    return {
        "snapshot": artifact(work / "snapshot.json"),
        "previousFeed": artifact(work / "previous-feed.json"),
        "previousSignature": artifact(work / "previous-feed.json.sig"),
    }


def tag_phase(args: argparse.Namespace, work: Path) -> dict:
    require(
        local_ref(TAG) is None and "refs/tags/" + TAG not in tags() and TAG not in releases(),
        "Immutable tag/release already exists",
    )
    message = work / "tag-message.txt"
    message.write_text(
        "Hakua 0.6.0\n\nReviewed source " + args.reviewed_commit + "\n",
        encoding="utf-8",
        newline="\n",
    )
    git_mutate(["tag", "-a", TAG, args.reviewed_commit, "-F", str(message)])
    git_mutate(["push", "origin", "refs/tags/" + TAG + ":refs/tags/" + TAG])
    return remote_tag(args.reviewed_commit)


def notes_input(args: argparse.Namespace) -> dict:
    require(
        args.notes is not None and args.notes == HERE / "RELEASE_NOTES.md",
        "Exact reviewed notes source required",
    )
    row = artifact(args.notes, args.notes_sha256)
    blob = release.git(
        "rev-parse", args.reviewed_commit + ":_docs/stable-060-20261116/RELEASE_NOTES.md"
    )
    raw = delivery.canonical_blob(blob)
    require(
        args.notes.read_bytes() in (raw, raw.replace(b"\n", b"\r\n")),
        "Reviewed release notes differ",
    )
    return row


def alias_phase(
    args: argparse.Namespace, work: Path, built: dict, snapshot: dict, row: dict
) -> dict:
    target = remote_tag(args.reviewed_commit)["object"]
    before = release.load(Path(snapshot["results"]["snapshot"]["path"]))
    old = before["tags"]["refs/tags/latest"]["sha"]
    require(tags().get("refs/tags/latest", {}).get("sha") == old, "Alias lease changed")
    require(local_ref("latest") == before["localAlias"], "Local alias changed")
    current = releases().get("latest")
    require(
        current is not None and identity(current) == before["releases"].get("latest"),
        "Unexpected alias release owner/state",
    )
    intent = {
        "oldObject": old,
        "oldLocalObject": before["localAlias"],
        "newObject": target,
        "aliasReleaseId": current["id"],
        "assets": built["assets"],
        "immutableReleaseId": row["id"],
    }
    release.save(work / "alias-intent.json", intent)
    # Only the separately snapshotted mutable alias release may lose old assets.
    for asset in current["assets"]:
        require(
            current["tag_name"] == "latest" and current["id"] != row["id"],
            "Immutable asset deletion refused",
        )
        gh("release", "delete-asset", "latest", asset["name"], "--yes", "--repo", REPO)
    assets = args.bundle_receipt.parent / "assets"
    for name in sorted(built["assets"]):
        gh("release", "upload", "latest", str(assets / name), "--repo", REPO)
    remote_assets(releases()["latest"], built["assets"])
    git_mutate(
        [
            "push",
            "--force-with-lease=refs/tags/latest:" + old,
            "origin",
            target + ":refs/tags/latest",
        ]
    )
    git_mutate(["update-ref", "refs/tags/latest", target, before["localAlias"] or "0" * 40])
    require(
        tags()["refs/tags/latest"]["sha"] == target and local_ref("latest") == target,
        "Alias ref readback differs",
    )
    gh(
        "release",
        "edit",
        "latest",
        "--latest=false",
        "--notes-file",
        str(args.notes),
        "--repo",
        REPO,
    )
    remote_assets(releases()["latest"], built["assets"])
    return intent


def download(args: argparse.Namespace, url: str, target: Path, expected: dict, admissions) -> dict:
    digest, count = hashlib.sha256(), 0
    release.no_links(target.parent)
    with public_open(url) as response, target.open("xb") as output:
        require(
            response.status == 200
            and response.headers.get("Content-Encoding", "identity") == "identity",
            "Unusable public content",
        )
        length = response.headers.get("Content-Length")
        require(length is None or int(length) == expected["bytes"], "Public Content-Length differs")
        while block := response.read(1024 * 1024):
            count += len(block)
            require(count <= expected["bytes"], "Public download exceeded expected bytes")
            observation = release.admit(allow_low_ram=args.allow_low_ram)
            admissions.write(json.dumps(observation, separators=(",", ":")) + "\n")
            require(
                admissions.tell() <= 256 * 1024 * 1024,
                "Publication admission evidence exceeds limit",
            )
            digest.update(block)
            output.write(block)
    result = {"bytes": count, "sha256": digest.hexdigest()}
    require(
        result == expected and release.sha(target) == expected["sha256"],
        "Anonymous content mismatch",
    )
    return {**result, "unauthenticated": True}


def phase(args: argparse.Namespace, work: Path) -> dict:
    pins, built, snapshot, prior = inputs(args)
    results, outcome = {}, {}
    if args.action == "snapshot":
        results = snapshot_phase(work)
    else:
        require(gh("api", "user", "--jq", ".login").strip() == "zapabob", "Wrong publication owner")
        unchanged(snapshot, alias_changed=args.action == "readback")
        if args.action == "tag":
            outcome = tag_phase(args, work)
        else:
            pins["notes"] = notes_input(args)
            outcome["tag"] = remote_tag(args.reviewed_commit)
            if args.action == "draft":
                require(TAG not in releases(), "Existing release must not be replaced")
                gh(
                    "release",
                    "create",
                    TAG,
                    "--draft",
                    "--latest=false",
                    "--verify-tag",
                    "--title",
                    "Hakua / Strata 0.6.0",
                    "--notes-file",
                    str(args.notes),
                    "--repo",
                    REPO,
                )
            row = release_row()
            require(row.get("prerelease") is False, "Prerelease refused")
            require(
                row.get("body", "").replace("\r\n", "\n").rstrip()
                == args.notes.read_text().replace("\r\n", "\n").rstrip(),
                "Release notes drifted",
            )
            outcome["notesSHA256"] = pins["notes"]["sha256"]
            if args.action in ("draft", "upload", "verify", "publish"):
                require(row["draft"] is True, "Explicit unpublished draft required")
            if args.action == "draft":
                require(row["assets"] == [], "Fresh draft unexpectedly has assets")
                require(
                    row.get("body", "").replace("\r\n", "\n").rstrip()
                    == args.notes.read_text().replace("\r\n", "\n").rstrip(),
                    "Draft notes differ",
                )
            else:
                require(row["id"] == prior["outcome"]["releaseId"], "Release owner changed")
            if args.action == "upload":
                missing = remote_assets(row, built["assets"], complete=False)
                release.save(work / "upload-before.json", identity(row))
                results["uploadBefore"] = artifact(work / "upload-before.json")
                for name in sorted(missing, key=lambda n: built["assets"][n]["bytes"]):
                    gh(
                        "release",
                        "upload",
                        TAG,
                        str(args.bundle_receipt.parent / "assets" / name),
                        "--repo",
                        REPO,
                    )
                remote_assets(release_row(), built["assets"])
            if args.action in ("verify", "publish", "alias", "readback"):
                remote_assets(row, built["assets"])
            if args.action == "publish":
                gh(
                    "release",
                    "edit",
                    TAG,
                    "--draft=false",
                    "--prerelease=false",
                    "--latest=true",
                    "--repo",
                    REPO,
                )
                row = release_row()
                require(
                    row["draft"] is False and identity(api("releases/latest")) == identity(row),
                    "Canonical Latest publication failed",
                )
                remote_assets(row, built["assets"])
            if args.action in ("alias", "readback"):
                require(
                    row["draft"] is False and identity(api("releases/latest")) == identity(row),
                    "Formal canonical release required",
                )
                pins["notes"] = notes_input(args)
            if args.action == "alias":
                outcome["alias"] = alias_phase(args, work, built, snapshot, row)
            if args.action == "readback":
                require(
                    tags()["refs/tags/latest"]["sha"] == outcome["tag"]["object"],
                    "Alias source differs",
                )
                remote_assets(releases()["latest"], built["assets"])
                require(
                    shutil.disk_usage(work).free
                    >= 10 * release.GIB + 2 * sum(r["bytes"] for r in built["assets"].values()),
                    "Anonymous download space refused",
                )
                observed = {}
                admission_path = work / "publication-admissions.jsonl"
                with admission_path.open("x", encoding="utf-8", newline="\n") as admissions:
                    admissions.write(
                        json.dumps(
                            {"workerContextSHA256": release.sha(work / "operation-worker.json")}
                        )
                        + "\n"
                    )
                for tag in (TAG, "latest"):
                    target = work / tag
                    target.mkdir()
                    with admission_path.open("a", encoding="utf-8", newline="\n") as admissions:
                        observed[tag] = {
                            name: download(
                                args,
                                "https://github.com/"
                                + REPO
                                + "/releases/download/"
                                + tag
                                + "/"
                                + name,
                                target / name,
                                expected,
                                admissions,
                            )
                            for name, expected in sorted(built["assets"].items())
                        }
                    results.update({tag + ":" + n: artifact(target / n) for n in built["assets"]})
                outcome["anonymousBytes"] = observed
                outcome["complete"] = True
            outcome["releaseId"] = row["id"]
            outcome["releaseIdentity"] = identity(release_row())
        unchanged(snapshot, alias_changed=args.action in ("alias", "readback"))
        if args.action != "tag":
            remote_tag(args.reviewed_commit)
    release.save(work / "outcome.json", outcome)
    results["outcome"] = artifact(work / "outcome.json")
    return {
        "schema": SCHEMA,
        "action": args.action,
        "inputs": pins,
        "results": results,
        "outcome": outcome,
    }


def validate_publication(
    path: Path, expected: str, commit: str, *, action: str | None = None, recursive: bool = True
) -> dict:
    record = original.proof(
        path,
        expected,
        "publication-" + (action or delivery.pinned(path, expected)["action"]),
        SCHEMA,
        commit,
    )
    original.require_helpers(record, HELPERS)
    require(
        record.get("action") in ACTIONS and (action is None or record["action"] == action),
        "Wrong publication action",
    )
    command = bundle.job_of(path)["operationExecution"]["command"]
    require(
        Path(command[2]) == HERE / "publish.py" and command[3] == record["action"],
        "Wrong publication worker action",
    )
    require(
        release.load(
            acceptance.evidence(record, "results", "outcome", path.parent / "outcome.json")
        )
        == record["outcome"],
        "Outcome proof differs",
    )
    if record["action"] == "snapshot":
        bundle.validate_snapshot(path, expected, commit)
    else:
        for name in ("bundle", "snapshot"):
            row = record["inputs"][name]
            require(
                original.option(command, "--" + name + "-receipt") == row["path"]
                and original.option(command, "--" + name + "-receipt-sha256") == row["sha256"],
                "Unbound publication input",
            )
        if recursive:
            row = record["inputs"]["bundle"]
            bundle.validate_bundle(Path(row["path"]), row["sha256"], commit)
        if record["action"] in PREVIOUS:
            row = record["inputs"]["prior"]
            require(
                original.option(command, "--prior-receipt") == row["path"]
                and original.option(command, "--prior-receipt-sha256") == row["sha256"],
                "Unbound prior proof",
            )
            if recursive:
                validate_publication(
                    Path(row["path"]),
                    row["sha256"],
                    commit,
                    action=PREVIOUS[record["action"]],
                    recursive=False,
                )
        if record["action"] != "tag":
            notes = record["inputs"]["notes"]
            require(
                Path(notes["path"]) == HERE / "RELEASE_NOTES.md"
                and original.option(command, "--notes") == notes["path"]
                and original.option(command, "--notes-sha256") == notes["sha256"]
                and record["outcome"].get("notesSHA256") == notes["sha256"],
                "Unbound final notes",
            )
        built_row = record["inputs"]["bundle"]
        built = delivery.pinned(Path(built_row["path"]), built_row["sha256"])
        snapshot_row = record["inputs"]["snapshot"]
        snapshot = bundle.validate_snapshot(
            Path(snapshot_row["path"]), snapshot_row["sha256"], commit
        )
        validate_outcome(record["action"], record["outcome"], built, snapshot, commit)
    return record


def validate_outcome(action: str, value: dict, built: dict, snapshot: dict, commit: str) -> None:
    """Do not admit a successful envelope with another source or incomplete stage."""
    tag = value if action == "tag" else value.get("tag", {})
    require(
        tag.get("commit") == commit and re.fullmatch(r"[a-f0-9]{40}", tag.get("object", "")),
        "Publication outcome tag/source differs",
    )
    if action == "tag":
        return
    row = value.get("releaseIdentity", {})
    require(
        row.get("tag_name") == TAG
        and row.get("id") == value.get("releaseId")
        and type(row.get("id")) is int
        and row["id"] > 0
        and row.get("prerelease") is False
        and row.get("draft") is (action in ("draft", "upload", "verify")),
        "Wrong release outcome",
    )
    expected = {} if action == "draft" else built["assets"]
    synthetic_row = {**row, "assets": [{"name": n, **r} for n, r in row["assets"].items()]}
    remote_assets(synthetic_row, expected)
    if action == "alias":
        before = release.load(Path(snapshot["results"]["snapshot"]["path"]))
        alias = value.get("alias", {})
        require(
            alias.get("oldObject") == before["tags"]["refs/tags/latest"]["sha"]
            and alias.get("oldLocalObject") == before["localAlias"]
            and alias.get("newObject") == tag["object"]
            and alias.get("assets") == built["assets"]
            and alias.get("immutableReleaseId") == row["id"]
            and alias.get("aliasReleaseId") == before["releases"]["latest"]["id"],
            "Alias outcome lease/identity differs",
        )
    if action == "readback":
        observed = value.get("anonymousBytes", {})
        require(
            value.get("complete") is True and set(observed) == {TAG, "latest"},
            "Incomplete public readback",
        )
        for tag_rows in observed.values():
            require(
                tag_rows == {n: {**r, "unauthenticated": True} for n, r in built["assets"].items()},
                "Anonymous content hashes differ",
            )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cli = native.parser(__doc__)
    cli.add_argument("action", choices=ACTIONS)
    for name in ("bundle", "snapshot", "prior"):
        cli.add_argument("--" + name + "-receipt", type=Path)
        cli.add_argument("--" + name + "-receipt-sha256")
    cli.add_argument("--notes", type=Path)
    cli.add_argument("--notes-sha256")
    args = cli.parse_args(argv)
    args.phase = "publication-" + args.action
    if args.execute:
        require("--attempt" in argv, "Explicit fresh attempt required")
    result = native.run_controller(args, HERE / "publish.py", phase, argv)
    if args.execute and not args.worker:
        original.finalize(
            release.OUT / (args.phase + "-" + args.attempt),
            args.phase,
            SCHEMA,
            args.reviewed_commit,
            validate_publication,
        )
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        raise SystemExit(main())
    except Exception as error:
        logging.error(
            "Publication refused (%s); retained attempt; raw error suppressed", type(error).__name__
        )
        raise SystemExit(1) from None
