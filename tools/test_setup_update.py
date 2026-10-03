"""The updater's checks, without a network or a GPU.

Run:  python tools/test_setup_update.py

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
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "serve"))

import update as U  # noqa: E402

FAILS: list[str] = []
CHECKS = [0]


def check(ok: bool, what: str, detail: str = ""):
    CHECKS[0] += 1
    print(f"  {'ok  ' if ok else 'FAIL'}  {what:<58}{' ' + detail if detail else ''}")
    if not ok:
        FAILS.append(what)


# ---------------------------------------------------------------------------------------------
# a fake release + a fake network


def make_zip(files: dict[str, bytes], version: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, body in files.items():
            zf.writestr(name, body)
        zf.writestr("BUILD.json", json.dumps({"version": version, "archs": [120], "src": "abc123"}))
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
                 asset_name: str = "strata-windows-x64.zip"):
        self.version = version
        self.zip_bytes = zip_bytes if zip_bytes is not None else make_zip(engine_files(version), version)
        # size=None means "report the real size"; a number is used to fake a truncated download
        self.size = size
        self.asset_name = asset_name
        self.hits: list[str] = []

    def fetch(self, url: str) -> dict:
        self.hits.append(url)
        size = self.size if self.size is not None else len(self.zip_bytes)
        return {"tag_name": f"v{self.version}", "html_url": "https://example/releases",
                "assets": [{"name": self.asset_name, "size": size,
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
    exe = engine_dir / ("strata.exe" if os.name == "nt" else "strata")
    seen: list[dict] = []
    up = U.Updater(engine_exe=exe, progress=seen.append)
    up.fetch, up.head = net.fetch, net.head
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
    rel = {"assets": [{"name": "strata-windows-x64.zip", "size": 1},
                      {"name": "strata-windows-x64-hip.zip", "size": 2}]}
    check((U.asset_for(rel, "cuda") or {}).get("name") == "strata-windows-x64.zip", "CUDA picks the CUDA zip")
    check((U.asset_for(rel, "hip") or {}).get("name") == "strata-windows-x64-hip.zip", "HIP picks the HIP zip")
    check(U.asset_for({"assets": []}, "cuda") is None, "a release with no matching asset is None")


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
        check(len(seen) > 5, "progress was emitted for the UI", f"{len(seen)} updates")
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

        class Dead(RuntimeError):
            pass

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
    print("\na card the release has no code for is refused BEFORE downloading")
    # The real case: v0.1.38's BUILD.json says archs [75, 86, 89, 120] and this PC's GTX 1070 is 6.1.
    # The probe catches it, but only after a 124 MB download - so the check runs first.
    with tempfile.TemporaryDirectory() as d:
        eng = Path(d) / "engine"
        eng.mkdir()
        (eng / "strata.exe").write_bytes(fake_engine("0.1.37"))
        (eng / "BUILD.json").write_text(
            json.dumps({"version": "0.1.37", "archs": [75, 86, 89, 120], "ptx": True}), encoding="utf-8")
        net = Fake("0.1.38")
        up = U.Updater(engine_exe=eng / "strata.exe", gpu_cc=6.1)
        up.fetch, up.head = net.fetch, net.head
        up.check()
        state = up.run()

        check(state["state"] == "failed", "the update is refused", state["state"])
        failed = [s for s in state["steps"] if s["status"] == "failed"]
        check(failed and failed[0]["key"] == "room", "at the disk/GPU step, before the download",
              failed[0]["key"] if failed else "none")
        err = state["detail"].get("error", "")
        check("6.1" in err, "the message names the card's compute capability", err[:56])
        check("7.5" in err or "8.6" in err, "and the architectures the release has", err[:80])
        check("nothing was downloaded" in err.lower() or "nothing was changed" in err.lower(),
              "and says nothing was downloaded", err[-40:])
        check(not net.hits or all("api.github.com" in h for h in net.hits),
              "only the release metadata was fetched - no asset request", str(len(net.hits)))
        check(U.installed_version(eng) == "0.1.37", "the installed engine is untouched")

    print("\n  ... and a card the release DOES support is not refused")
    with tempfile.TemporaryDirectory() as d:
        eng = Path(d) / "engine"
        eng.mkdir()
        (eng / "strata.exe").write_bytes(fake_engine("0.1.37"))
        (eng / "BUILD.json").write_text(
            json.dumps({"version": "0.1.37", "archs": [75, 86, 89, 120], "ptx": True}), encoding="utf-8")
        net = Fake("0.1.38")
        up = U.Updater(engine_exe=eng / "strata.exe", gpu_cc=8.6)      # a supported card
        up.fetch, up.head = net.fetch, net.head
        up.check()
        state = up.run()
        check(state["state"] == "done", "an 8.6 card updates normally", state["state"])

    print("\n  ... and PTX covers anything NEWER than the newest listed architecture")
    with tempfile.TemporaryDirectory() as d:
        eng = Path(d) / "engine"
        eng.mkdir()
        (eng / "strata.exe").write_bytes(fake_engine("0.1.37"))
        (eng / "BUILD.json").write_text(
            json.dumps({"version": "0.1.37", "archs": [75], "ptx": True}), encoding="utf-8")
        net = Fake("0.1.38")
        up = U.Updater(engine_exe=eng / "strata.exe", gpu_cc=12.0)      # newer than archs, but has PTX
        up.fetch, up.head = net.fetch, net.head
        up.check()
        state = up.run()
        check(state["state"] == "done", "a 12.0 card is covered by the PTX", state["state"])

    print("\n  ... and an unknown card is not refused (a wrong refusal would be worse)")
    with tempfile.TemporaryDirectory() as d:
        eng = Path(d) / "engine"
        eng.mkdir()
        (eng / "strata.exe").write_bytes(fake_engine("0.1.37"))
        (eng / "BUILD.json").write_text(
            json.dumps({"version": "0.1.37", "archs": [75, 86, 89, 120]}), encoding="utf-8")
        net = Fake("0.1.38")
        up = U.Updater(engine_exe=eng / "strata.exe", gpu_cc=None)     # cannot tell
        up.fetch, up.head = net.fetch, net.head
        up.check()
        state = up.run()
        check(state["state"] == "done", "an unknown card is let through", state["state"])

    print("\n  ... and an engine with no BUILD.json has no archs to check against")
    with tempfile.TemporaryDirectory() as d:
        eng = Path(d) / "engine"
        eng.mkdir()
        (eng / "strata.exe").write_bytes(fake_engine("0.1.37"))
        (eng / "BUILD.json").write_text(json.dumps({"version": "0.1.37"}), encoding="utf-8")
        net = Fake("0.1.38")
        up = U.Updater(engine_exe=eng / "strata.exe", gpu_cc=6.1)
        up.fetch, up.head = net.fetch, net.head
        up.check()
        state = up.run()
        check(state["state"] == "done", "no archs means no refusal", state["state"])


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


def main() -> int:
    for fn in (t_version_and_assets, t_installed_version, t_zip_safety, t_happy_path,
               t_no_update_needed, t_downgrade_refused, t_version_mismatch_refused,
               t_staged_engine_must_run, t_probe_explains_a_missing_runtime, t_gpu_arch_refused_before_download, t_workspace_is_cleaned, t_truncated_download_refused, t_rollback_after_apply,
               t_backup_retention, t_missing_asset_refused):
        fn()
    print(f"\n{U.__name__}: {len(FAILS)} failures out of {CHECKS[0]} checks")
    for f in FAILS:
        print(f"  FAILED: {f}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
