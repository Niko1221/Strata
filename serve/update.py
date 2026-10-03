"""Updating the engine from the project's GitHub release, without a hand-rolled download.

WHY THIS EXISTS.  The engine is a ready-made build downloaded by setup.py
(``releases/latest/download/strata-windows-x64.zip``), and the project ships a release most days.  So
"update Strata" has meant: find the release, download a 124 MB zip, and overwrite the files in the
engine directory by hand.  That is a fine procedure and a poor user interface, and it is easy to get
wrong - a half-extracted zip leaves an engine that will not start.

HOW IT IS DELIBERATE.  This module is cautious by construction, and the order of the steps is the
design:

  1. Nothing on disk changes until every download and every check has passed.  The zip is downloaded
     to a temp file, opened, inspected, extracted to a staging directory, and the STAGED binary is
     run - all before the installed engine is touched.
  2. The installed engine is copied to a backup directory first, and the backup is kept until the new
     engine has been started and has reported the expected version.  If anything fails from the
     apply step onward, the backup is restored and the install is left as it was.
  3. A downgrade is refused.  Getting a new engine is the point; silently going backwards is not, and
     a stale download URL or a wrong tag could otherwise do it without asking.
  4. Every step is recorded with its own status so the web UI can show what is happening rather than
     a spinner.  The failure note is the one line that matters, so it says what was restored.

WHAT IT DOES NOT DO.  It does not touch the Python checkout, the pinned packages, the model, or the
packs.  Those are a ``git pull`` and a ``pip install -r``; rewriting the code this server is running
from, in-process, is a different and much larger thing.  It does not update ``data/*.bin`` either -
see ``data_files`` below for why that is left out rather than guessed at.

MEASURED, not assumed: every step is verified against the artifact rather than trusted.  The zip's
member list is checked before extraction, ``BUILD.json`` is parsed and its version compared with the
release tag, and the staged binary is executed before the installed one is replaced.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import threading
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

# The project publishes one zip per platform/backend.  The name has to be matched exactly, so the
# mapping is written out rather than built from strings that could drift from what is published.
# Measured from the v0.1.38 release: strata-windows-x64.zip (124 MB) and strata-windows-x64-hip.zip
# (599 MB), both under releases/latest/download/.
ASSETS = {
    ("windows", "cuda"): "strata-windows-x64.zip",
    ("windows", "hip"): "strata-windows-x64-hip.zip",
    ("linux", "cuda"): "strata-linux-x64.zip",
    ("linux", "hip"): "strata-linux-x64-hip.zip",
}

API = "https://api.github.com/repos/{repo}/releases/latest"

# Never fetch more than this from a URL the server chooses.  The real asset is ~124 MB; the ceiling
# exists so a wrong or hostile Content-Length cannot fill the disk.
MAX_ASSET_BYTES = 2 << 30

# The GPU architectures Strata supports, as compute capability x10.  Same rule setup.py applies before it
# installs anything (engine_runs_on / the "older than the RTX 20 series" message): an engine carries code
# for a listed set of architectures, plus PTX for anything NEWER than the newest of them.
#
# Why the updater needs it: the released build has "archs": [75, 86, 89, 120], so it has no code for
# anything below 7.5.  Measured on this machine (a GTX 1070, compute capability 6.1): the staged v0.1.38
# engine dies at start with STATUS_ILLEGAL_INSTRUCTION, after a 124 MB download.  Checking first turns
# that into an instant, free refusal with an explanation.
MIN_CC = 7.5

# How many backups to keep.  One is enough to recover from a bad update; a few means a user who
# updates, regrets it, and updates again can still get back.
KEEP_BACKUPS = 3

# The staged binary is run with --help.  It must print something and exit 0 within this long, or the
# stage is treated as broken.  Measured on a Windows build: under a second.
PROBE_TIMEOUT_S = 30

# The nine steps, in order, and the method that carries each one out.  The ORDER is the safety property:
# nothing on disk changes until `backup`, and everything before it is a check that can refuse.  The
# web panel shows exactly this list, so it lives here once and the panel reads it from the state.
#
#   plan     what we are installing
#   room     there is space for the download, the staging tree and the backup
#   download the archive, to a temp file
#   inspect  member paths, CRCs, the version inside - then, now that the release's own architectures are
#            known, whether this card can run it (see _check_gpu for why the card is checked HERE and not
#            before the download)
#   stage    unpack to a staging directory, not over the install
#   test     run the STAGED engine, before the installed one is touched
#   backup   copy the installed files - the first step that changes anything
#   apply    copy the new files over
#   verify   read the new BUILD.json back
STEPS = (
    ("plan", "Check the release"),
    ("room", "Check the free disk space"),
    ("download", "Download the engine"),
    ("inspect", "Inspect the archive"),
    ("stage", "Unpack to a staging area"),
    ("test", "Run the staged engine"),
    ("backup", "Back up the installed engine"),
    ("apply", "Install the new engine"),
    ("verify", "Start it and check the version"),
)


class UpdateError(RuntimeError):
    """Any refusal or failure.  The message is shown to the user, so it says what to do next."""


def tag_word(tag: str | None) -> str:
    """A release tag for a message: 'v0.1.38', or 'the release' when there is none."""
    return tag or "the release"


def parse_version(text: str | None) -> tuple[int, ...]:
    """'0.1.38' -> (0, 1, 38).  Returns () for anything unparseable, which compares as older."""
    if not text:
        return ()
    out = []
    for part in str(text).strip().lstrip("vV").split("."):
        digits = "".join(c for c in part if c.isdigit())
        if not digits:
            break
        out.append(int(digits))
    return tuple(out)


def installed_version(engine_dir: Path) -> str | None:
    """The engine's own version, from the BUILD.json next to the binary.

    That is the same file the server already reads for /status, so the number shown in the UI and the
    number this compares against cannot disagree.
    """
    build = engine_dir / "BUILD.json"
    try:
        # utf-8-sig, not utf-8: a BUILD.json saved by a Windows text editor or a PowerShell
        # `Set-Content -Encoding utf8` starts with a BOM, and json.loads rejects the whole file because
        # of those three bytes - so the version silently reads as "unknown" and the panel offers an
        # update to someone already on the latest. utf-8-sig reads a file with or without the BOM.
        return json.loads(build.read_text(encoding="utf-8-sig")).get("version")
    except (OSError, ValueError):
        return None


def gpu_compute_capability() -> float | None:
    """The first NVIDIA card's compute capability, or None if it cannot be read.

    Asks nvidia-smi exactly the way setup.py does (`--query-gpu=compute_cap`), rather than adding a
    dependency the server does not have.  None - no nvidia-smi, an AMD card, an unreadable answer - is the
    normal case on a machine where this check does not apply, and the caller then skips the check.
    """
    if not sys.platform.startswith("win") and not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
                             capture_output=True, timeout=PROBE_TIMEOUT_S, text=True)
    except (OSError, subprocess.SubprocessError):
        return None
    for line in (out.stdout or "").splitlines():
        line = line.strip()
        if line and line.replace(".", "").isdigit():
            return int(line.replace(".", "")) / 10.0
    return None


def installed_archs(engine_dir: Path) -> tuple[list[int], bool]:
    """The INSTALLED engine's GPU architectures and PTX flag, from its BUILD.json.

    (`archs`, `ptx`) - the same two fields setup.py reads to decide whether an engine runs on a card.
    ([], False) when there is no BUILD.json or it has neither, which callers must treat as "unknown"
    rather than "no architectures".
    """
    try:
        build = json.loads((engine_dir / "BUILD.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return [], False
    archs = [int(a) for a in (build.get("archs") or []) if str(a).isdigit()]
    return archs, bool(build.get("ptx"))


def asset_for(release: dict, backend: str) -> dict | None:
    """The release asset for this PC, or None if the release has none (e.g. CUDA on a HIP box)."""
    key = (sys.platform.startswith("win") and "windows" or "linux", backend)
    name = ASSETS.get(key)
    if not name:
        return None
    for asset in release.get("assets") or []:
        if asset.get("name") == name:
            return asset
    return None


def safe_members(zf: zipfile.ZipFile, dest: Path) -> list[str]:
    """Check every member path before extracting anything.

    A zip is an archive of NAMES, so an entry called ``../../something`` writes outside ``dest``.  This
    rejects absolute paths, drive letters, and any name that does not stay under ``dest``.  Returns
    the names it accepted so a refusal can name the offending entry.
    """
    good = []
    for info in zf.infolist():
        name = info.filename
        if name.startswith(("/", "\\")) or ".." in Path(name).parts or ":" in name:
            raise UpdateError(f"the archive contains an unsafe path ({name!r}); not extracting anything")
        target = (dest / name).resolve()
        try:
            target.relative_to(dest.resolve())
        except ValueError:
            raise UpdateError(f"the archive contains an unsafe path ({name!r}); not extracting anything")
        good.append(name)
    return good


@dataclass
class Step:
    key: str
    label: str
    status: str = "pending"          # pending | active | done | failed | skipped
    note: str = ""
    started: float | None = None
    ended: float | None = None

    def as_dict(self) -> dict:
        return {"key": self.key, "label": self.label, "status": self.status, "note": self.note,
                "seconds": round((self.ended or time.time()) - self.started, 2) if self.started else None}


@dataclass
class Updater:
    """Runs the update.  Read with state_dict() from another thread; see _put() for the locking."""

    engine_exe: Path
    backend: str = "cuda"
    repo: str = "Niko1221/Strata"
    gpu_cc: float | None = None          # the card's compute capability, if the caller knows it
    fetch: object = None                           # url -> bytes iterator; injectable for tests
    head: object = None                            # url -> (status, headers); injectable for tests

    # Filled in by run(); the UI reads them.
    steps: list = field(default_factory=list)
    state: str = "idle"                            # idle | checking | ready | running | done | failed
    detail: dict = field(default_factory=dict)
    backup_dir: Path | None = None
    staging: Path | None = None           # set by _do_stage, cleared when the run ends
    backup: Path | None = None            # set by _do_backup; what a rollback restores from
    _changed: bool = False                         # has anything on disk been replaced yet?

    def __post_init__(self):
        self.engine_exe = Path(self.engine_exe)
        self.engine_dir = self.engine_exe.parent
        self.root = self.engine_dir.parent / ".strata-update"
        self._lock = threading.RLock()   # state_dict() is read from HTTP threads while a run writes

    # ---- plumbing ---------------------------------------------------------------------------------

    def _put(self, **fields):
        """Set `detail` fields under the lock (see state_dict() for why the lock is needed)."""
        with self._lock:
            self.detail.update(fields)

    def state_dict(self) -> dict:
        # The web app POLLS this every 700 ms on an HTTP thread while run() mutates the state on its own
        # thread. `dict(d)` while another thread inserts a key can raise "dictionary changed size during
        # iteration", so the snapshot is taken under the lock - see _put() for the writes.
        with self._lock:
            done = sum(1 for s in self.steps if s.status == "done")
            active = next((s for s in self.steps if s.status == "active"), None)
            steps = [s.as_dict() for s in self.steps]
            detail = dict(self.detail)
            state, backup = self.state, self.backup_dir
        # percent is only meaningful DURING a run.  The check phase has one step, so reporting
        # done/total there gives 100% the instant a release is found, and the UI would then snap back
        # to 0% when run() builds the real nine-step list.  Reporting None outside a run avoids that.
        percent = None
        if state == "running" and steps:
            percent = round(100.0 * done / len(steps))
        return {
            "state": state,
            "detail": detail,
            "steps": steps,
            "done": done,
            "total": len(steps),
            "percent": percent,
            "active": active.label if active else None,
            "active_key": active.key if active else None,   # the panel matches on this, not the label
            "backup": str(backup) if backup else None,
        }

    def _step(self, key: str, label: str, status: str, note: str = ""):
        with self._lock:
            for s in self.steps:
                if s.key == key:
                    s.status, s.note = status, note
                    if status == "active":
                        s.started = time.time()
                    if status in ("done", "failed", "skipped"):
                        s.ended = time.time()
                    return s
            raise UpdateError(f"internal: unknown step {key!r}")

    # ---- network ----------------------------------------------------------------------------------

    def _open(self, url: str):
        if callable(self.head):
            return self.head(url)
        req = urllib.request.Request(url, headers={"User-Agent": "Strata-updater"})
        return urllib.request.urlopen(req, timeout=30)

    def _release(self) -> dict:
        if callable(self.fetch):
            return self.fetch(API.format(repo=self.repo))
        try:
            with self._open(API.format(repo=self.repo)) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise UpdateError(f"GitHub answered {e.code} for the latest release; not changing anything")
        except Exception as e:
            raise UpdateError(f"could not reach GitHub ({type(e).__name__}: {e}); not changing anything")

    def _download(self, url: str, dest: Path, expected: int | None, label: str) -> None:
        """Download with progress, then check the size against what the API said.

        The size check is the only integrity check available: the release publishes no checksum and
        BUILD.json carries none either, so a short or oversized download is caught here rather than
        at extraction time.  This is a real limitation, not a verification - see the PR note.
        """
        total = expected or 0
        got = 0
        last = 0.0
        with self._open(url) as r, open(dest, "wb") as fh:
            if not total:
                try:
                    total = int(r.headers.get("Content-Length") or 0)
                except Exception:
                    total = 0
            if total > MAX_ASSET_BYTES:
                raise UpdateError(f"{label} is {total} bytes, over the {MAX_ASSET_BYTES} limit; refusing")
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                got += len(chunk)
                now = time.time()
                if now - last > 0.2:
                    last = now
                    pct = round(100.0 * got / total) if total else 0
                    self._put(percent=pct,
                              downloaded_mb=round(got / 1e6, 1),
                              total_mb=round(total / 1e6, 1) if total else None)
        if total and got != total:
            raise UpdateError(
                f"the download stopped early ({got:,} of {total:,} bytes); nothing on disk was changed")

    # ---- the steps --------------------------------------------------------------------------------

    def check(self) -> dict:
        """Compare the installed engine with the latest release.  Changes nothing."""
        # The step list has to exist BEFORE _step() is called: _step() only updates an existing entry,
        # so an empty list makes the very first call raise "unknown step".
        self.steps = [Step("plan", "Check the release")]
        self.state = "checking"
        self._step("plan", "Check the release", "active")
        release = self._release()
        tag = (release.get("tag_name") or "").strip()
        have = installed_version(self.engine_dir)
        asset = asset_for(release, self.backend)
        want = parse_version(tag)
        if not asset:
            self._step("plan", "Check the release", "failed",
                       f"{tag} publishes no {self.backend} build for this platform")
            self.state = "failed"
            self._put(latest=tag, installed=have)
            return self.state_dict()
        newer = bool(want) and (not parse_version(have) or want > parse_version(have))
        self._step("plan", "Check the release", "done",
                   f"installed {have or 'unknown'}, latest {tag}")
        self.state = "ready"
        # asset_url and asset_size are what run() downloads against; the expected size is the only
        # integrity check the release gives us, so it is carried through from the API rather than
        # re-derived from a HEAD request that could disagree with the listing.
        self._put(latest=tag,
                installed=have,
                newer=newer,
                asset=asset.get("name"),
                asset_url=asset.get("browser_download_url"),
                asset_size=asset.get("size"),
                asset_mb=round((asset.get("size") or 0) / 1e6, 1),
                release_url=release.get("html_url"))
        return self.state_dict()

    def run(self, allow_downgrade: bool = False) -> dict:
        """The whole update.  Any failure restores what was replaced; see `rollback`."""
        if self.state != "ready":
            raise UpdateError("check for updates first")

        have = self.detail.get("installed")
        tag = self.detail["latest"]
        want, havev = parse_version(tag), parse_version(have)

        # The downgrade test comes FIRST.  A strictly older release also has newer == False, so if the
        # "nothing to do" early return ran first it would answer an older tag with "already up to
        # date" - which is how a stale tag or a wrong mirror silently rolls an install backwards.
        if want and havev and want < havev and not allow_downgrade:
            self.steps = [Step("plan", "Refuse a downgrade", "failed",
                               f"the latest release is {tag}, older than the installed {have}")]
            self.state = "failed"
            self._put(error=f"{tag} is older than the installed {have}; refusing to downgrade")
            return self.state_dict()

        if not self.detail.get("newer") and not allow_downgrade:
            self.steps = [Step("plan", "Nothing to do", "skipped", "already on the latest release")]
            self.state = "ready"
            return self.state_dict()

        self.steps = [Step(k, lbl) for k, lbl in STEPS]
        self.state = "running"

        self.staging = self.backup = None
        try:
            for key, label in STEPS:
                self._step(key, label, "active")
                # each step is one method that returns the note to show when it finishes
                note = getattr(self, f"_do_{key}")()
                if note:
                    self._step(key, label, "done", note)
            # A successful run keeps the backup (that is the point of it) but not the 124 MB archive it
            # came from, and never the staging copy: those are what fill the disk when setup.py and the
            # updater both run on a machine nobody prunes.
            self._put(installed=tag)
            self._clean_workspace()
            self.state = "done"
        except Exception as e:
            note = str(e)
            failed = next((s for s in self.steps if s.status == "active"), None)
            if self._changed:
                try:
                    restored = self.rollback(self.backup)
                    note += f" Restored the previous engine from {restored.name}."
                except Exception as rb:
                    note += (f" ROLLBACK FAILED: {rb}. The engine is left as it is; restore "
                             f"{self.backup} by hand if it will not start.")
            # The step is marked failed AFTER the rollback above, so its note is the whole outcome and
            # not just the error. The panel shows that note on the step; detail["error"] is the same
            # text for anything reading the state as JSON, and detail["action"] is the short version.
            if failed:
                self._step(failed.key, failed.label, "failed", note)
            self.state = "failed"
            self._put(error=note, rolled_back=bool(self._changed and "Restored" in note))
            # A short, separate line for the panel: what was DONE about the failure. The full `error`
            # belongs to the step that failed and is shown there, so repeating it here would print the
            # same 300-character paragraph twice on one screen.
            self._put(action=(f"Restored the previous engine from {self.backup.name}."
                            if self._changed and self.backup and "Restored" in note
                            else "Nothing on this PC was changed."))
            # The backup is kept on failure as well as on success - it is the only way back. The
            # archive and the staging tree are not: see _clean_workspace.
            self._clean_workspace()
        finally:
            # _clean_workspace() also removes the staging tree and any partial download, so this is a
            # second, belt-and-braces sweep for the case where the run died before root/ existed.
            self._clean_workspace()
        return self.state_dict()

    def _clean_workspace(self):
        """Remove the staging tree, the downloaded archive and any partial download.

        Never touches a backup-<...> directory - that is the only way back, on success and on failure
        alike.  The archive is 124 MB and the staging tree another 221 MB, and a machine where setup.py
        has also run accumulates them; a release URL is stable for its tag, so re-running downloads it
        again and keeping it only grows the engine directory's parent.
        """
        for path in list(self.root.glob("stage-*")) + list(self.root.glob("*.zip")) + \
                list(self.root.glob("*.part")):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)

    # ---- the individual checks ---------------------------------------------------------------------

    def _check_gpu(self):
        """Refuse a release that carries no code for this card.

        This runs AFTER the archive is inspected, not before the download, and that placement is the whole
        point.  The release's own BUILD.json - the only honest source of what it can run - is inside the
        zip, so a check before the download can only compare the card against the *installed* engine's
        archs.  That is what setup.py's get_prebuilt() does too, and it has a consequence worth stating:
        it cannot see the case that actually bites, where a new release drops an architecture the installed
        one had.  Doing it here costs the download and removes the guesswork.

        Nothing on disk has changed when this runs: the apply step is four steps later.

        An unknown card is NOT refused.  A wrong refusal costs the user their update; a wrong install costs
        them an engine that will not start, and they can always update by hand.
        """
        if self.backend != "cuda" or self.gpu_cc is None:
            return
        archs, ptx = self.detail.get("release_archs") or ([], False)
        if not archs:
            return
        # BUILD.json lists architectures as compute capability x10 (86 = 8.6), the way setup.py turns
        # nvidia-smi's "compute_cap" into an int (`cc.replace(".", "")`). The card here is a float, so it
        # is converted the same way - comparing 8.6 against [75, 86, ...] directly never matches.
        cc = float(self.gpu_cc)
        cc10 = int(round(cc * 10))
        if cc10 in archs or (ptx and cc10 > max(archs)):
            return
        listed = ", ".join(f"{a / 10:g}" for a in sorted(archs))
        raise UpdateError(
            f"{tag_word(self.detail.get('latest'))} has no code for this graphics card (compute capability "
            f"{cc:g}; it was built for {listed}), so the engine would not start on it. This is the same "
            f"check setup.py makes before installing - see get_prebuilt() - and nothing has been changed. "
            f"Strata needs compute capability {MIN_CC:g} or newer (RTX 20 or later); cards below that "
            f"need the community CUDA 12.x build (STRATA_EXPERIMENTAL_SM60=1)")

    # ---- the nine steps ----------------------------------------------------------------------
    # Each returns the note to show when it finishes, and raises UpdateError to fail the run.  run()
    # does the state bookkeeping, so nothing here has to know about steps or the panel.

    def _do_plan(self) -> str:
        return f"installing {self.detail['latest']}"

    def _do_room(self) -> str:
        self._check_room()
        return f"{self.detail.get('free_gb')} GB free"

    def _do_download(self) -> str:
        tag = self.detail["latest"]
        self.root.mkdir(parents=True, exist_ok=True)
        part = self.root / f"download-{tag}.zip.part"
        self._download(self.detail["asset_url"], part, self.detail.get("asset_size"),
                       "the engine archive")
        part.replace(self.root / f"download-{tag}.zip")
        return f"{self.detail['asset_mb']} MB"

    def _do_inspect(self) -> str:
        self._inspect()
        self._check_gpu()          # after _inspect, which is where the release's archs come from
        return f"{self.detail['members']} files, BUILD.json says {self.detail['latest']}"

    def _do_stage(self) -> str:
        self.staging = Path(tempfile.mkdtemp(prefix="stage-", dir=str(self.root)))
        self._extract(self.staging)
        return self.staging.name

    def _do_test(self) -> str:
        self._probe(self.staging)
        return "it starts and prints its usage"

    def _do_backup(self) -> str:
        self.backup = self._backup()
        return self.backup.name

    def _do_apply(self) -> str:
        self._apply(self.staging)
        self._changed = True
        return "replaced"

    def _do_verify(self) -> str:
        self._verify(self.detail["latest"])
        return f"running {self.detail['verified_version']}"

    # ---- the individual checks ---------------------------------------------------------------------

    def _check_room(self):
        free = shutil.disk_usage(self.root if self.root.exists() else self.engine_dir).free
        self._put(free_gb=round(free / 1e9, 1))
        need = (self.detail.get("asset_size") or 0) * 3      # zip + staging + backup, roughly
        if free < need:
            raise UpdateError(
                f"only {free / 1e9:.1f} GB free where the update needs about {need / 1e9:.1f} GB; "
                f"nothing was changed")

    def _inspect(self):
        path = self.root / f"download-{self.detail['latest']}.zip"
        if not path.exists():
            raise UpdateError("the downloaded archive is missing; nothing on disk was changed")
        try:
            with zipfile.ZipFile(path) as zf:
                names = zf.namelist()
                if zf.testzip() is not None:
                    raise UpdateError("the archive is corrupt (a member fails its CRC); not installing it")
                safe = safe_members(zf, self.engine_dir)
                if not any(n.endswith("BUILD.json") for n in safe):
                    raise UpdateError("the archive has no BUILD.json, so its version cannot be checked")
                if not any(Path(n).name.startswith("strata") and n.endswith((".exe", "")) for n in safe):
                    raise UpdateError("the archive has no engine binary in it")
                build = json.loads(zf.read(next(n for n in safe if n.endswith("BUILD.json")))
                                   .decode("utf-8-sig"))    # utf-8-sig: a release zipped on Windows can carry a BOM
            self._put(release_archs=([int(a) for a in (build.get("archs") or []) if str(a).isdigit()],
                                     bool(build.get("ptx"))))
        except zipfile.BadZipFile:
            raise UpdateError("the download is not a valid zip; nothing on disk was changed")
        got = str(build.get("version") or "")
        if parse_version(got) != parse_version(self.detail["latest"]):
            raise UpdateError(
                f"the archive is v{got} but the release is {self.detail['latest']}; refusing to install it")
        self._put(members=len(safe))

    def _extract(self, staging: Path):
        path = self.root / f"download-{self.detail['latest']}.zip"
        with zipfile.ZipFile(path) as zf:
            zf.extractall(staging)

    @staticmethod
    def _find_engine(where: Path):
        """The engine binary inside a staged tree, or None.

        A plain generator expression is the trap here: ``next((p for p in it), None)`` finds the first
        match, while ``next((p for p in it, None))`` builds a TUPLE containing a generator and never
        matches anything.  Both spellings look right and only one works.
        """
        names = ["strata.exe", "strata"]        # the release builds Windows and Linux names
        for name in names:
            found = next((p for p in where.rglob(name) if p.is_file()), None)
            if found:
                return found
        return None

    def _probe(self, staging: Path):
        """Run the STAGED binary.  This is the check that most protects an install: the replacement is
        proven to start before the working one is touched."""
        exe = self._find_engine(staging)
        if exe is None:
            raise UpdateError("the staged engine has no strata binary; not installing it")
        try:
            r = subprocess.run([str(exe), "--help"], capture_output=True, timeout=PROBE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            raise UpdateError("the staged engine did not answer --help within "
                              f"{PROBE_TIMEOUT_S} s; not installing it")
        except OSError as e:
            raise UpdateError(f"the staged engine could not be started ({e}); not installing it")
        out = (r.stdout or b"").decode("utf-8", "replace") + (r.stderr or b"").decode("utf-8", "replace")
        if out.strip():
            return
        # It produced nothing, so say WHY rather than leaving "printed nothing" to be read as a broken
        # release.  Measured on this machine: v0.1.38's strata.exe links cublas64_13.dll, this PC has
        # CUDA 12.6, and the process dies before main() with exit code 0xC0000135.  That is a working
        # release that cannot run HERE, which is exactly the case the probe exists to catch - but the
        # fix belongs to the reader (install the CUDA runtime setup.py provides), so the message names it.
        code = r.returncode & 0xFFFFFFFF
        if code == 0xC0000135:
            raise UpdateError(
                "the new engine needs a CUDA library this PC does not have (it stops before it starts: "
                "DLL not found). The release is usually built against a newer CUDA than the runtime "
                "installed here; run setup.py to install the runtime it needs, then check again. "
                "Nothing was changed")
        if code == 0xC000001D:
            # Measured on this machine with the real v0.1.38 release: cublas64_13.dll and
            # cublasLt64_13.dll installed from setup.py's own wheels (nvidia-cublas==13.0.2.14), and the
            # process still died here with STATUS_ILLEGAL_INSTRUCTION. This PC has a GTX 1070, compute
            # capability 6.1; CUDA 13 dropped support below 7.5, so the library's instructions do not
            # exist for this card. docs/MULTI_GPU.md states the 7.5 requirement, and the release runs
            # here only via the community CUDA 12.x build (#295).
            raise UpdateError(
                "the new engine cannot run on this graphics card (it stops before it starts: the "
                "instructions are not supported by this GPU). Strata needs compute capability 7.5 or "
                "newer (RTX 20 or later); this release's CUDA libraries do not cover older cards, which "
                "need the community CUDA 12.x build. Nothing was changed")
        if code in (0xC0000139, 0xC0000005, 0xC000007B):
            raise UpdateError(f"the new engine stopped before it started (exit code {hex(code)}), which "
                              "usually means a runtime or driver mismatch. Nothing was changed")
        raise UpdateError("the staged engine printed nothing for --help "
                          f"(exit code {code if code < 0x80000000 else hex(code)}); not installing it")

    def _backup(self) -> Path:
        # A second-resolution stamp collides when two backups happen in the same second - the second
        # call then writes into the first's directory and the pruning below deletes the wrong thing.
        # A counter suffix keeps every backup distinct.
        stamp = time.strftime("%Y%m%d-%H%M%S")
        dest = self.root / f"backup-{stamp}"
        n = 1
        while dest.exists():
            dest = self.root / f"backup-{stamp}-{n}"
            n += 1
        dest.mkdir(parents=True, exist_ok=True)
        for item in self.engine_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, dest / item.name)
        (dest / "_version.txt").write_text(str(self.detail.get("installed")), encoding="utf-8")
        # Keep the newest few; the rest are dead weight once a newer update has succeeded.
        old = sorted(self.root.glob("backup-*"), reverse=True)[KEEP_BACKUPS:]
        for d in old:
            shutil.rmtree(d, ignore_errors=True)
        self.backup_dir = dest
        return dest

    def _apply(self, staging: Path):
        """Copy the staged files over the installed ones.

        Kept deliberately simple - copy, do not move - so the backup remains the only thing that can
        undo this, and so a crash halfway leaves files that are either old or new rather than absent.

        The executable bit is carried across explicitly: zipfile does not preserve it on Linux, and an
        engine that loses it will not start after the update, which would look like a bad release.
        """
        for src in staging.rglob("*"):
            if not src.is_file():
                continue
            dst = self.engine_dir / src.relative_to(staging)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            if os.name != "nt" and src.suffix != ".json" and not src.suffix:
                dst.chmod(src.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    def _verify(self, tag: str):
        """The installed BUILD.json is the engine's own report of what it is; check it, and that the
        files landed where the config expects them."""
        got = installed_version(self.engine_dir)
        if parse_version(got) != parse_version(tag):
            raise UpdateError(f"after installing, the engine still reports {got or 'nothing'}")
        if not self.engine_exe.exists():
            raise UpdateError(f"after installing, {self.engine_exe} is missing")
        self._put(verified_version=got)

    def rollback(self, backup: Path | None) -> Path:
        if not backup or not backup.exists():
            raise UpdateError("no backup to restore from")
        for src in backup.iterdir():
            if src.name == "_version.txt" or not src.is_file():
                continue
            shutil.copy2(src, self.engine_dir / src.name)
        return backup
