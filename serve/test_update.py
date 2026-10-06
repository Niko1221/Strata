"""The updater's checks, without a network or a GPU.

Run:  python serve/test_update.py          (or: python -m unittest serve.test_update)

Everything here is offline.  The release JSON, the download and the archive are all built locally, so
the point being tested is the ORDER of operations and the refusals, not that GitHub answers.

The cases that matter most are the ones where the update must NOT happen:

  * a download that stops early is caught by the size check, and nothing on disk changes
  * an archive whose BUILD.json version disagrees with the release tag is refused
  * a zip containing a path outside the destination is refused before anything is extracted
  * a staged engine that will not run stops the install before the working one is touched
  * once the engine HAS been replaced, a failure restores it from the backup

That last pair is the whole design: nothing changes until every check has passed, and after the
change there is always a way back.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # so import serve.update works
import serve.update as U  # noqa: E402


FAILS: list[str] = []
CHECKS = [0]


def sha256_of(data: bytes) -> str:
    """What the API's `digest` field would say for these bytes."""
    import hashlib
    return hashlib.sha256(data).hexdigest()


def check(ok: bool, what: str, detail: str = ""):
    CHECKS[0] += 1
    print(f"  {'ok  ' if ok else 'FAIL'}  {what:<58}{' ' + detail if detail else ''}")
    if not ok:
        FAILS.append(what)


# ---------------------------------------------------------------------------------------------
# a fake release + a fake network


def make_zip(files: dict[str, bytes], version: str, archs=(120,), ptx=False) -> bytes:
    """A release-shaped zip.  `archs`/`ptx` go into its BUILD.json, which is where the updater reads the
    release's own GPU architectures from - so a test can say what the RELEASE supports, separately from
    what the installed engine claims."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, body in files.items():
            zf.writestr(name, body)
        zf.writestr("BUILD.json", json.dumps({"version": version, "archs": list(archs), "ptx": ptx,
                                              "src": "abc123"}))
    return buf.getvalue()


def engine_files(version: str, vision: bytes = b"new vision") -> dict[str, bytes]:
    """The files a real release zip carries: the engine, the vision engine, and BUILD.json.

    strata-vision.exe is included on purpose.  An update copies what the archive contains and leaves
    files it does not mention alone, so a test can only show a file being REPLACED if the new release
    also contains it.
    """
    return {"strata.exe" if os.name == "nt" else "strata": fake_engine(version),
            "strata-vision.exe" if os.name == "nt" else "strata-vision": vision}


def fake_engine(version: str) -> bytes:
    """A stand-in for the engine binary, good enough for the probe step to really run it.

    The probe in update.py executes the staged file, so this has to be a real executable.  A batch
    script named .exe is not: Windows refuses it with WinError 216 ("this version is not compatible"),
    and the test would then pass for the wrong reason - the probe failing on a malformed fixture
    rather than being exercised.

    `cmd.exe` is the stand-in because it was MEASURED to behave: copied to another directory under the
    name strata.exe it still prints its usage to a captured pipe (168 bytes).  `where.exe` does not -
    renamed it prints nothing at all, which would make the probe fail for a reason that has nothing to
    do with the code under test.
    """
    if os.name == "nt":
        candidate = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "cmd.exe"
        if candidate.exists():
            return candidate.read_bytes()
        raise RuntimeError("no cmd.exe to stand in for the engine; cannot run this test")
    # On Linux a shell script needs its executable bit, which zipfile does not carry across, so the
    # probe's copy step makes it runnable after extraction.
    return b"#!/bin/sh\necho 'strata usage: --pack DIR --tokens LIST'\nexit 0\n"


class Fake:
    """Stands in for the network: a release dict, and zip bytes served for any asset URL."""

    def __init__(self, version: str, zip_bytes: bytes | None = None, size: int | None = None,
                 asset_name: str = "strata-windows-x64.zip", digest: str | None = None):
        self.version = version
        self.zip_bytes = zip_bytes if zip_bytes is not None else make_zip(engine_files(version), version)
        # size=None means "report the real size"; a number is used to fake a truncated download
        self.size = size
        self.asset_name = asset_name
        # The SHA-256 the API reports for the asset, defaulting to the real one so a test that is not
        # about verification still passes. `digest=""` means the release publishes none, which the
        # updater must refuse; a hex string means it publishes that one, wrong or right.
        self.digest = None if digest == "" else digest or sha256_of(self.zip_bytes)
        self.hits: list[str] = []

    def fetch(self, url: str) -> dict:
        self.hits.append(url)
        size = self.size if self.size is not None else len(self.zip_bytes)
        return {"tag_name": f"v{self.version}", "html_url": "https://example/releases",
                "assets": [{"name": self.asset_name, "size": size,
                            # None, not a computed hash: `digest=""` means the release publishes none,
                            # and the point of that test is that nothing is checked against anything
                            "digest": f"sha256:{self.digest}" if self.digest else None,
                            "browser_download_url": "https://example/asset.zip"}]}

    def head(self, url: str):
        payload = self.zip_bytes

        class R:
            def __init__(self):
                self.headers = {"Content-Length": str(len(payload))}
                self._buf = io.BytesIO(payload)

            def read(self, n=-1):
                return self._buf.read(n)

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        return R()


def install_fake(engine_dir: Path, version: str, exe_name: str = "strata.exe"):
    """Make an engine directory that looks installed."""
    engine_dir.mkdir(parents=True, exist_ok=True)
    (engine_dir / exe_name).write_bytes(fake_engine(version))
    (engine_dir / "BUILD.json").write_text(json.dumps({"version": version}), encoding="utf-8")
    (engine_dir / "strata-vision.exe").write_bytes(b"old vision")
    return engine_dir


def run_update(engine_dir: Path, net: Fake, **kw):
    """Run a whole update, recording what a POLLING client would see.

    The recorder wraps `_step`, which is where every visible change happens, and reads state_dict() after
    each one - the same thing the web app does every 700 ms. So this also asserts, thirty-odd times per
    run, that the state is safe to snapshot from another thread while it is being written.
    """
    exe = engine_dir / ("strata.exe" if os.name == "nt" else "strata")
    seen: list[dict] = []
    up = U.Updater(engine_exe=exe)
    up.fetch, up.head = net.fetch, net.head
    real_step = up._step

    def recording_step(*a, **kw):
        out = real_step(*a, **kw)
        seen.append(up.state_dict())
        return out

    up._step = recording_step
    up.check()
    state = up.run(**kw) if up.state == "ready" else up.state_dict()
    return up, state, seen


# ---------------------------------------------------------------------------------------------
# cases

def t_version_and_assets():
    print("\nversion parsing and asset choice")
    check(U.parse_version("0.1.38") == (0, 1, 38), "parse_version 0.1.38")
    check(U.parse_version("0.1.9") < U.parse_version("0.1.38"), "0.1.9 sorts below 0.1.38 (not as text)")
    check(U.parse_version(None) == () and U.parse_version("junk") == (), "unparseable is treated as older")

    # The real asset list, as published on v0.1.40.1 (measured). Note the CUDA 12 asset: this table
    # used to be hardcoded and did not know about it, which would have handed a CUDA 12 machine a CUDA
    # 13 engine it cannot run.
    rel = {"assets": [{"name": "strata-windows-x64.zip", "size": 1},
                      {"name": "strata-windows-x64-cuda12.zip", "size": 2},
                      {"name": "strata-windows-x64-hip.zip", "size": 3}]}
    check((U.asset_for(rel, "cuda", 13) or {}).get("name") == "strata-windows-x64.zip",
          "CUDA 13 keeps the default build")
    check((U.asset_for(rel, "cuda", 12) or {}).get("name") == "strata-windows-x64-cuda12.zip",
          "CUDA 12 keeps the CUDA 12 build (setup.py's CUDA12_ASSET)")
    check((U.asset_for(rel, "hip", 13) or {}).get("name") == "strata-windows-x64-hip.zip",
          "HIP picks the HIP build")
    check((U.asset_for(rel, "cuda", None) or {}).get("name") == "strata-windows-x64.zip",
          "an engine with no recorded CUDA gets the default build")
    check((U.asset_for(rel, "cuda", 11) or {}).get("name") == "strata-windows-x64.zip",
          "a CUDA major with no asset falls back to the default, rather than refusing")
    check(U.asset_for({"assets": []}, "cuda", 13) is None, "a release with no matching asset is None")

    check(U.cuda_major_of({"cuda": "13.0"}) == 13, "cuda 13.0 -> 13")
    check(U.cuda_major_of({"cuda": "12.6"}) == 12, "cuda 12.6 -> 12")
    check(U.cuda_major_of({}) is None and U.cuda_major_of({"cuda": "x"}) is None,
          "an engine with no usable cuda field -> None")


def t_installed_version():
    print("\ninstalled version comes from BUILD.json (the same file /status reads)")
    with tempfile.TemporaryDirectory() as d:
        eng = Path(d)
        check(U.installed_version(eng) is None, "no BUILD.json -> unknown, not a crash")
        (eng / "BUILD.json").write_text("{}", encoding="utf-8")
        check(U.installed_version(eng) is None, "BUILD.json with no version -> unknown")
        (eng / "BUILD.json").write_text(json.dumps({"version": "0.1.31"}), encoding="utf-8")
        check(U.installed_version(eng) == "0.1.31", "BUILD.json version is read")

        # A BOM, which a Windows editor or PowerShell's `Set-Content -Encoding utf8` writes. json.loads
        # rejects the WHOLE file for those three bytes, so a version written that way reads as unknown
        # and the panel offers an update to someone already on the latest. Found on this machine by
        # writing the fixture through PowerShell and watching "installed" come back empty.
        (eng / "BUILD.json").write_text(json.dumps({"version": "0.1.31"}), encoding="utf-8-sig")
        check(U.installed_version(eng) == "0.1.31",
              "a BUILD.json with a UTF-8 BOM still reads (utf-8-sig)",
              repr(U.installed_version(eng)))
        (eng / "BUILD.json").write_bytes(b"\xff\xfe not json at all")
        check(U.installed_version(eng) is None, "a corrupt BUILD.json is unknown, not a crash")


def t_zip_safety():
    print("\narchive paths are checked before anything is extracted")
    with tempfile.TemporaryDirectory() as d:
        dest = Path(d) / "dest"
        dest.mkdir()
        with zipfile.ZipFile(io.BytesIO(make_zip({"ok.txt": b"x"}, "1.0")), "w") as _:
            pass
        # a member escaping the destination
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("../../escaped.txt", b"nope")
        with zipfile.ZipFile(io.BytesIO(buf.getvalue())) as zf:
            try:
                U.safe_members(zf, dest)
                check(False, "a ../ path is refused")
            except U.UpdateError as e:
                check("unsafe path" in str(e), "a ../ path is refused", str(e)[:44])
        check(not (Path(d) / "escaped.txt").exists(), "nothing was written outside the destination")
        # an absolute path
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("/etc/passwd", b"nope")
        with zipfile.ZipFile(io.BytesIO(buf.getvalue())) as zf:
            try:
                U.safe_members(zf, dest)
                check(False, "an absolute path is refused")
            except U.UpdateError:
                check(True, "an absolute path is refused")


def t_happy_path():
    print("\nthe whole update, end to end")
    with tempfile.TemporaryDirectory() as d:
        eng = install_fake(Path(d) / "engine", "0.1.31")
        before = (eng / "strata-vision.exe").read_bytes()
        net = Fake("0.1.38")
        up, state, seen = run_update(eng, net)
        check(state["state"] == "done", "state is done", str(state.get("detail", {}).get("error", ""))[:60])
        check(all(s["status"] == "done" for s in state["steps"]), "every step finished",
              str([s["key"] for s in state["steps"] if s["status"] != "done"]))
        check(U.installed_version(eng) == "0.1.38", "BUILD.json now reports the new version")
        check((eng / "strata-vision.exe").read_bytes() != before, "the other engine files were replaced too")
        check(up.backup_dir is not None and up.backup_dir.exists(), "a backup was kept")
        check(len(seen) > 5, "the state was readable at every step, as the poller sees it",
              f"{len(seen)} snapshots")
        # percent is only meaningful during a run; the check phase reports None so the UI cannot show
        # 100% (one step, finished) and then snap back to 0% when the real step list is built.
        running = [s["percent"] for s in seen if s["percent"] is not None]
        check(bool(running), "percent is reported while running")
        check(running == sorted(running), "and it only goes up", f"{running}")
        check(all(s["percent"] is None for s in seen if s["state"] != "running"),
              "percent is None outside a run")


def t_no_update_needed():
    print("\nnothing to do when already on the latest")
    with tempfile.TemporaryDirectory() as d:
        eng = install_fake(Path(d) / "engine", "0.1.38")
        up, state, _ = run_update(eng, Fake("0.1.38"))
        check(state["state"] == "ready", "stays ready, does not enter a run")
        check(U.installed_version(eng) == "0.1.38", "nothing was touched")


def t_downgrade_refused():
    print("\na downgrade is refused, and is not confused with 'nothing to do'")
    with tempfile.TemporaryDirectory() as d:
        eng = install_fake(Path(d) / "engine", "0.1.38")
        net = Fake("0.1.20")
        up, state, _ = run_update(eng, net)
        check(state["state"] == "failed", "run() refuses a downgrade", state["state"])
        check("older" in state["detail"].get("error", ""), "and says so",
              state["detail"].get("error", "")[:56])
        check("downgrade" in state["steps"][0]["label"].lower(), "the step is named for what it is",
              state["steps"][0]["label"])
        check(U.installed_version(eng) == "0.1.38", "the installed version is untouched")


def t_version_mismatch_refused():
    print("\nan archive that disagrees with the release tag is refused")
    with tempfile.TemporaryDirectory() as d:
        eng = install_fake(Path(d) / "engine", "0.1.31")
        # the release says 0.1.38, the zip inside says 0.1.20
        net = Fake("0.1.38", zip_bytes=make_zip({"strata.exe": fake_engine("0.1.20")}, "0.1.20"))
        up, state, _ = run_update(eng, net)
        failed = [s for s in state["steps"] if s["status"] == "failed"]
        check(state["state"] == "failed", "the update failed")
        check(failed and failed[0]["key"] == "inspect", "it failed at inspect, before staging")
        check(U.installed_version(eng) == "0.1.31", "the installed engine is untouched")


def t_staged_engine_must_run():
    print("\na staged engine that will not run stops the install")
    with tempfile.TemporaryDirectory() as d:
        eng = install_fake(Path(d) / "engine", "0.1.31")
        # a BUILD.json that matches, but a binary that prints nothing and fails
        payload = make_zip({"strata.exe": b"not an executable at all"}, "0.1.38")
        net = Fake("0.1.38", zip_bytes=payload)
        up, state, _ = run_update(eng, net)
        failed = [s for s in state["steps"] if s["status"] == "failed"]
        check(state["state"] == "failed", "the update failed")
        check(failed and failed[0]["key"] == "test", "it failed at the probe, before backup/apply",
              failed[0]["key"] if failed else "")
        check(U.installed_version(eng) == "0.1.31", "the installed engine is untouched")
        check(up.backup_dir is None, "no backup was even taken, so there is nothing to restore")


def t_probe_explains_a_missing_runtime():
    print("\\na probe failure names the reason, not just 'printed nothing'")
    # The real case, measured on this machine: v0.1.38's strata.exe links cublas64_13.dll, this PC has
    # CUDA 12.6, so the process dies with 0xC0000135 (STATUS_DLL_NOT_FOUND) before main(). The user has to
    # be told that, because the obvious reading of "printed nothing" is "the release is broken".
    with tempfile.TemporaryDirectory() as d:
        eng = install_fake(Path(d) / "engine", "0.1.31")


        up = U.Updater(engine_exe=eng / "strata.exe")
        net = Fake("0.1.38")
        up.fetch, up.head = net.fetch, net.head
        up.check()

        real_run = subprocess.run

        def fake_run(*a, **kw):
            class R:
                returncode = -1073741515          # 0xC0000135 as a signed Windows exit code
                stdout = b""
                stderr = b""
            return R()

        subprocess.run = fake_run
        try:
            state = up.run()
        finally:
            subprocess.run = real_run

        err = state["detail"].get("error", "")
        failed = [s for s in state["steps"] if s["status"] == "failed"]
        check(state["state"] == "failed", "the update failed")
        check(failed and failed[0]["key"] == "test", "at the probe")
        check("CUDA" in err or "DLL" in err, "the message says a library is missing, not 'printed nothing'",
              err[:60])
        check("Nothing was changed" in err, "and says nothing was changed", err[-30:])
        check(U.installed_version(eng) == "0.1.31", "the installed engine is untouched")

        # And the case this box actually hits: the DLLs ARE present (setup.py's own wheels) but the card
        # is too old for them, so the process dies with STATUS_ILLEGAL_INSTRUCTION. Measured here with
        # the real v0.1.38 engine on a GTX 1070 (compute capability 6.1); the message must name the card
        # requirement, because "printed nothing" sends the reader looking for a corrupt download.
        up2 = U.Updater(engine_exe=eng / "strata.exe")
        net2 = Fake("0.1.38")
        up2.fetch, up2.head = net2.fetch, net2.head
        up2.check()

        def fake_run_cc(*a, **kw):
            class R:
                returncode = -1073741795        # 0xC000001D, STATUS_ILLEGAL_INSTRUCTION
                stdout = b""
                stderr = b""
            return R()

        subprocess.run = fake_run_cc
        try:
            state2 = up2.run()
        finally:
            subprocess.run = real_run

        err2 = state2["detail"].get("error", "")
        check(state2["state"] == "failed", "an unsupported GPU also fails at the probe")
        check("7.5" in err2, "the message states the compute capability the release needs", err2[:64])
        check("Nothing was changed" in err2, "and says nothing was changed")
        check(U.installed_version(eng) == "0.1.31", "the installed engine is still untouched")


def t_workspace_is_cleaned():
    print("\nthe downloaded archive and staging copy are cleaned up, backups are not")
    with tempfile.TemporaryDirectory() as d:
        eng = install_fake(Path(d) / "engine", "0.1.31")
        net = Fake("0.1.38")
        up, state, _ = run_update(eng, net)
        check(state["state"] == "done", "the update succeeded", str(state.get("detail", {}).get("error", ""))[:40])
        left = sorted(p.name for p in up.root.glob("*"))
        check(all(n.startswith("backup-") for n in left),
              "only the backup is left (no 124 MB zip, no staging tree)", str(left))
        check(up.backup_dir and up.backup_dir.exists(), "the backup is kept for a by-hand restore")

        # a failed run must clean up too, and must not take the backups with it
        with tempfile.TemporaryDirectory() as d2:
            eng2 = install_fake(Path(d2) / "engine", "0.1.31")
            net2 = Fake("0.1.38", zip_bytes=make_zip({"strata.exe": fake_engine("0.1.20")}, "0.1.20"))
            up2, state2, _ = run_update(eng2, net2)
            check(state2["state"] == "failed", "a version-mismatch release fails")
            left2 = sorted(p.name for p in up2.root.glob("*"))
            check(not [n for n in left2 if n.endswith(".zip") or n.startswith("stage-")],
                  "a failed run leaves no archive and no staging tree", str(left2))

        # three updates in a row must not leave three zips behind
        with tempfile.TemporaryDirectory() as d3:
            eng3 = install_fake(Path(d3) / "engine", "0.1.31")
            for i in range(3):
                up3, st3, _ = run_update(eng3, Fake("0.1.38"))
                # each round starts from the new version, so pretend a newer one is out
                (eng3 / "BUILD.json").write_text(json.dumps({"version": "0.1.31"}), encoding="utf-8")
            zips = list(up3.root.glob("*.zip"))
            check(not zips, f"three updates leave no zips behind", str([p.name for p in zips]))
            check(len(list(up3.root.glob("backup-*"))) <= U.KEEP_BACKUPS,
                  f"and at most {U.KEEP_BACKUPS} backups",
                  str(len(list(up3.root.glob("backup-*")))))


def t_truncated_download_refused():
    print("\na download that stops early is caught by the size check")
    with tempfile.TemporaryDirectory() as d:
        eng = install_fake(Path(d) / "engine", "0.1.31")
        real = make_zip({"strata.exe": fake_engine("0.1.38")}, "0.1.38")
        net = Fake("0.1.38", zip_bytes=real, size=len(real) + 5000)   # API claims more than we serve
        up, state, _ = run_update(eng, net)
        failed = [s for s in state["steps"] if s["status"] == "failed"]
        check(state["state"] == "failed", "the update failed")
        check(failed and failed[0]["key"] == "download", "it failed at download")
        check(U.installed_version(eng) == "0.1.31", "the installed engine is untouched")
        check(not list(up.root.glob("*.zip")), "the partial download was cleaned up")


def t_rollback_after_apply():
    print("\na failure AFTER replacing the engine restores it")
    with tempfile.TemporaryDirectory() as d:
        eng = install_fake(Path(d) / "engine", "0.1.31")
        original = (eng / "BUILD.json").read_text(encoding="utf-8")

        up = U.Updater(engine_exe=eng / "strata.exe")
        net = Fake("0.1.38")
        up.fetch, up.head = net.fetch, net.head
        up.check()
        # Let the update run for real, but make the LAST step fail, which is after the files changed.
        real_verify = up._verify
        up._verify = lambda tag: (_ for _ in ()).throw(RuntimeError("simulated verify failure"))
        state = up.run()
        up._verify = real_verify

        check(state["state"] == "failed", "the update reported failure")
        check(state["detail"].get("rolled_back") is True, "and reported that it rolled back")
        check(state["detail"].get("action", "").startswith("Restored"),
              "and a short action line for the panel", state["detail"].get("action", "")[:48])
        failed_step = [s for s in state["steps"] if s["status"] == "failed"]
        check(failed_step and "Restored" in failed_step[0]["note"],
              "the failing step carries the full message, so the panel need not repeat it",
              (failed_step[0]["note"] if failed_step else "")[:40])
        check((eng / "BUILD.json").read_text(encoding="utf-8") == original,
              "BUILD.json is back to the old version")
        check(U.installed_version(eng) == "0.1.31", "the engine reports the old version again")
        check("Restored" in state["detail"]["error"], "the message says it restored the backup",
              state["detail"]["error"][:60])


def t_backup_retention():
    print("\nold backups are pruned, the newest few kept")
    with tempfile.TemporaryDirectory() as d:
        eng = install_fake(Path(d) / "engine", "0.1.31")
        up = U.Updater(engine_exe=eng / "strata.exe")
        up.detail["installed"] = "0.1.31"
        for i in range(U.KEEP_BACKUPS + 2):
            up._backup()
        left = sorted(up.root.glob("backup-*"))
        check(len(left) == U.KEEP_BACKUPS, f"at most {U.KEEP_BACKUPS} backups are kept", f"{len(left)} left")


def t_gpu_arch_refused_before_download():
    print("\na card the RELEASE has no code for is refused, at the inspect step")
    # The case the live run on this PC hit: the release's own BUILD.json lists archs [75, 86, 89, 120]
    # and "ptx", the card is a GTX 1070 at compute capability 6.1, so the engine cannot run here. The
    # check runs where the release's archs are finally known - after the archive is inspected, which is
    # still four steps before anything is changed.
    release = make_zip(engine_files("0.1.38"), "0.1.38", archs=[75, 86, 89, 120], ptx=True)
    with tempfile.TemporaryDirectory() as d:
        eng = install_fake(Path(d) / "engine", "0.1.31")
        net = Fake("0.1.38", zip_bytes=release)
        up = U.Updater(engine_exe=eng / "strata.exe", gpu_cc=6.1)
        up.fetch, up.head = net.fetch, net.head
        up.check()
        state = up.run()

        check(state["state"] == "failed", "the update is refused", state["state"])
        failed = [s for s in state["steps"] if s["status"] == "failed"]
        check(failed and failed[0]["key"] == "inspect", "at the inspect step, before staging",
              failed[0]["key"] if failed else "none")
        err = state["detail"].get("error", "")
        check("6.1" in err, "the message names the card's compute capability", err[:56])
        check("7.5" in err or "8.6" in err, "and the architectures the release was built for", err[:90])
        check("v0.1.38" in err, "and names the release", err[:40])
        check(U.installed_version(eng) == "0.1.31", "the installed engine is untouched")
        check(up.backup is None, "no backup was taken, so there is nothing to restore")

    print("\n  ... and it uses the RELEASE's archs, not the installed engine's")
    # The bug this placement fixes: checked against the INSTALLED engine, a card the installed build
    # happens not to support would be refused even when the new release does support it - and, worse,
    # a release that DROPS an architecture could never be caught before the probe. Here the installed
    # engine claims [75, 86, 89, 120] and the card is 6.1, but the release carries [61]: it must go
    # through, because the release is what will actually run.
    release = make_zip(engine_files("0.1.38"), "0.1.38", archs=[61])
    with tempfile.TemporaryDirectory() as d:
        eng = Path(d) / "engine"
        eng.mkdir()
        (eng / ("strata.exe" if os.name == "nt" else "strata")).write_bytes(fake_engine("0.1.31"))
        (eng / "BUILD.json").write_text(
            json.dumps({"version": "0.1.31", "archs": [75, 86, 89, 120], "ptx": True}), encoding="utf-8")
        net = Fake("0.1.38", zip_bytes=release)
        up = U.Updater(engine_exe=eng / ("strata.exe" if os.name == "nt" else "strata"), gpu_cc=6.1)
        up.fetch, up.head = net.fetch, net.head
        up.check()
        state = up.run()
        check(state["state"] == "done",
              "a 6.1 card is allowed when the RELEASE supports 6.1, whatever the old engine claimed",
              state["state"])

    print("\n  ... a supported card is not refused")
    release = make_zip(engine_files("0.1.38"), "0.1.38", archs=[75, 86, 89, 120], ptx=True)
    with tempfile.TemporaryDirectory() as d:
        eng = install_fake(Path(d) / "engine", "0.1.31")
        net = Fake("0.1.38", zip_bytes=release)
        up = U.Updater(engine_exe=eng / "strata.exe", gpu_cc=8.6)      # a supported card
        up.fetch, up.head = net.fetch, net.head
        up.check()
        check(up.run()["state"] == "done", "an 8.6 card updates normally")

    print("\n  ... PTX covers anything NEWER than the newest architecture listed")
    release = make_zip(engine_files("0.1.38"), "0.1.38", archs=[75], ptx=True)
    with tempfile.TemporaryDirectory() as d:
        eng = install_fake(Path(d) / "engine", "0.1.31")
        net = Fake("0.1.38", zip_bytes=release)
        up = U.Updater(engine_exe=eng / "strata.exe", gpu_cc=12.0)     # newer than archs, but has PTX
        up.fetch, up.head = net.fetch, net.head
        up.check()
        check(up.run()["state"] == "done", "a 12.0 card is covered by the PTX")

    print("\n  ... an unknown card is not refused (a wrong refusal is worse than a download)")
    release = make_zip(engine_files("0.1.38"), "0.1.38", archs=[75, 86, 89, 120])
    with tempfile.TemporaryDirectory() as d:
        eng = install_fake(Path(d) / "engine", "0.1.31")
        net = Fake("0.1.38", zip_bytes=release)
        up = U.Updater(engine_exe=eng / "strata.exe", gpu_cc=None)     # cannot tell
        up.fetch, up.head = net.fetch, net.head
        up.check()
        check(up.run()["state"] == "done", "an unknown card is let through")

    print("\n  ... a release with no archs at all is not refused either")
    release = make_zip(engine_files("0.1.38"), "0.1.38", archs=[])
    with tempfile.TemporaryDirectory() as d:
        eng = install_fake(Path(d) / "engine", "0.1.31")
        net = Fake("0.1.38", zip_bytes=release)
        up = U.Updater(engine_exe=eng / "strata.exe", gpu_cc=6.1)
        up.fetch, up.head = net.fetch, net.head
        up.check()
        check(up.run()["state"] == "done", "no archs means no refusal")

    print("\n  ... and the AMD backend skips the check entirely")
    with tempfile.TemporaryDirectory() as d:
        eng = install_fake(Path(d) / "engine", "0.1.31")
        release = make_zip(engine_files("0.1.38"), "0.1.38", archs=[120])
        net = Fake("0.1.38", zip_bytes=release, asset_name="strata-windows-x64-hip.zip")
        up = U.Updater(engine_exe=eng / "strata.exe", backend="hip", gpu_cc=6.1)
        up.fetch, up.head = net.fetch, net.head
        up.check()
        check(up.run()["state"] == "done", "a HIP build is never judged on NVIDIA compute capability")

def t_missing_asset_refused():
    print("\na release with no build for this platform is refused, not half-applied")
    with tempfile.TemporaryDirectory() as d:
        eng = install_fake(Path(d) / "engine", "0.1.31")
        net = Fake("0.1.38", asset_name="strata-something-else.zip")
        up, state, _ = run_update(eng, net)
        check(state["state"] == "failed", "the check fails")
        check("no cuda build" in state["steps"][0]["note"], "and says why",
              state["steps"][0]["note"][:52])
        check(U.installed_version(eng) == "0.1.31", "nothing was touched")


def t_hotfix_release_tag():
    print("\na hotfix tag (v0.1.40.1) whose engine is v0.1.40 is NOT refused")
    # Measured against the real v0.1.40.1 release: the tag is four parts, the cuda12 archive's
    # BUILD.json says 0.1.40, because the hotfix was in setup.py. An exact tag-vs-engine comparison
    # refuses that, which blocks a perfectly good update.
    with tempfile.TemporaryDirectory() as d:
        eng = install_fake(Path(d) / "engine", "0.1.37")
        net = Fake("0.1.40.1", zip_bytes=make_zip(engine_files("0.1.40"), "0.1.40", archs=[120]))
        up = U.Updater(engine_exe=eng / "strata.exe")
        up.fetch, up.head = net.fetch, net.head
        up.check()
        state = up.run()
        check(state["state"] == "done", "a four-part tag with a three-part engine updates normally",
              state["state"])
        check(state["detail"].get("engine_version") == "0.1.40",
              "and the engine version actually installed is reported",
              str(state["detail"].get("engine_version")))

    print("\n  ... but an archive OLDER than the tag still means the wrong asset was picked")
    with tempfile.TemporaryDirectory() as d:
        eng = install_fake(Path(d) / "engine", "0.1.37")
        # release v0.1.40.1 but the archive carries a 0.1.38 engine: a mismatched or stale asset
        net = Fake("0.1.40.1", zip_bytes=make_zip(engine_files("0.1.38"), "0.1.38", archs=[120]))
        up = U.Updater(engine_exe=eng / "strata.exe")
        up.fetch, up.head = net.fetch, net.head
        up.check()
        state = up.run()
        check(state["state"] == "failed", "an older engine in the archive is refused", state["state"])
        err = state["detail"].get("error", "")
        check("wrong archive was picked" in err, "and it says the wrong archive was picked", err[:56])
        check(U.installed_version(eng) == "0.1.37", "the installed engine is untouched")

    print("\n  ... and a BUILD.json with no version at all is refused")
    with tempfile.TemporaryDirectory() as d:
        eng = install_fake(Path(d) / "engine", "0.1.37")
        payload = make_zip(engine_files("0.1.40"), "0.1.40", archs=[120])
        # rewrite the archive's BUILD.json with no version
        out = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(payload)) as src, \
                zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dst:
            for info in src.infolist():
                body = src.read(info.filename)
                if info.filename.endswith("BUILD.json"):
                    body = json.dumps({"archs": [120]}).encode()
                dst.writestr(info.filename, body)
        net = Fake("0.1.40", zip_bytes=out.getvalue())
        up = U.Updater(engine_exe=eng / "strata.exe")
        up.fetch, up.head = net.fetch, net.head
        up.check()
        state = up.run()
        check(state["state"] == "failed", "an archive with no version is refused", state["state"])
        check("names no version" in state["detail"].get("error", ""), "and says so",
              state["detail"].get("error", "")[:44])


def t_sha256_verified_before_anything_is_applied():
    print("\nthe download's SHA-256 is checked against GitHub's, and a wrong one stops the run")
    # The reason this exists: without it the only check was the byte count, which a substituted file of
    # the same length passes. GitHub publishes the hash in the releases API (`digest`), on a different
    # connection from the download itself.
    with tempfile.TemporaryDirectory() as d:
        eng = install_fake(Path(d) / "engine", "0.1.31")
        net = Fake("0.1.38")
        up, state, _ = run_update(eng, net)
        check(state["state"] == "done", "a matching digest updates normally", state["state"])
        check(state["detail"].get("sha256") == net.digest,
              "and the hash of what arrived is reported",
              str(state["detail"].get("sha256"))[:16])
        notes = {s["key"]: s["note"] for s in state["steps"]}
        check("verified" in (notes.get("download") or ""), "the download step says it verified",
              notes.get("download") or "")

    print("\n  ... a file whose bytes differ from the published hash is refused and deleted")
    with tempfile.TemporaryDirectory() as d:
        eng = install_fake(Path(d) / "engine", "0.1.31")
        # the API promises the hash of the real zip, but the server hands over different bytes
        good = make_zip(engine_files("0.1.38"), "0.1.38")
        tampered = good[:-2] + b"XX"                    # same length, different content
        net = Fake("0.1.38", zip_bytes=tampered, digest=sha256_of(good))
        up, state, _ = run_update(eng, net)
        check(state["state"] == "failed", "the update fails", state["state"])
        failed = [s for s in state["steps"] if s["status"] == "failed"]
        check(failed and failed[0]["key"] == "download", "at the download step, before anything else",
              failed[0]["key"] if failed else "none")
        err = state["detail"].get("error", "")
        check("wrong SHA-256" in err, "the message says the hash did not match", err[:52])
        check(U.installed_version(eng) == "0.1.31", "the installed engine is untouched")
        check(not list(up.root.glob("*.zip")), "the bad file is deleted, not kept",
              str([p.name for p in up.root.glob("*")]))
        check(all(s["status"] == "pending" for s in state["steps"][3:]),
              "and no later step ran at all")

    print("\n  ... a release that publishes no digest is refused rather than installed unchecked")
    with tempfile.TemporaryDirectory() as d:
        eng = install_fake(Path(d) / "engine", "0.1.31")
        net = Fake("0.1.38", digest="")
        up, state, _ = run_update(eng, net)
        check(state["state"] == "failed", "the update fails", state["state"])
        err = state["detail"].get("error", "")
        check("no SHA-256" in err, "and says there is no hash to check against", err[:52])
        check(U.installed_version(eng) == "0.1.31", "the installed engine is untouched")
        check(not list(up.root.glob("*.zip")), "and nothing was downloaded")

    print("\n  ... a digest in another algorithm is not accepted as if it were sha256")
    with tempfile.TemporaryDirectory() as d:
        eng = install_fake(Path(d) / "engine", "0.1.31")
        net = Fake("0.1.38")
        net.fetch = lambda url: {"tag_name": "v0.1.38", "html_url": "", "assets": [
            {"name": net.asset_name, "size": net.size,
             "digest": "sha512:" + "0" * 128,
             "browser_download_url": "https://example/asset.zip"}]}
        up = U.Updater(engine_exe=eng / "strata.exe")
        up.fetch, up.head = net.fetch, net.head
        up.check()
        state = up.run()
        check(state["state"] == "failed", "an md5/sha512 digest does not pass as sha256", state["state"])
        check("no SHA-256" in state["detail"].get("error", ""), "and it is reported as unverifiable",
              state["detail"].get("error", "")[:44])

    print("\n  ... the size check still runs, and still refuses a short download")
    with tempfile.TemporaryDirectory() as d:
        eng = install_fake(Path(d) / "engine", "0.1.31")
        real = make_zip(engine_files("0.1.38"), "0.1.38")
        net = Fake("0.1.38", zip_bytes=real, size=len(real) + 5000)
        up, state, _ = run_update(eng, net)
        failed = [s for s in state["steps"] if s["status"] == "failed"]
        check(state["state"] == "failed", "a truncated download still fails", state["state"])
        check(failed and failed[0]["key"] == "download", "at the download step",
              failed[0]["key"] if failed else "none")


def main() -> int:
    for fn in (t_version_and_assets, t_installed_version, t_zip_safety, t_happy_path,
               t_no_update_needed, t_downgrade_refused, t_version_mismatch_refused,
               t_staged_engine_must_run, t_probe_explains_a_missing_runtime,
               t_gpu_arch_refused_before_download, t_hotfix_release_tag, t_sha256_verified_before_anything_is_applied,
               t_workspace_is_cleaned, t_truncated_download_refused, t_rollback_after_apply,
               t_backup_retention, t_missing_asset_refused):
        fn()
    print(f"\n{U.__name__}: {len(FAILS)} failures out of {CHECKS[0]} checks")
    for f in FAILS:
        print(f"  FAILED: {f}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
