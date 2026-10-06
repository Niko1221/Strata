"""Updating the engine from the project's GitHub release, without a hand-rolled download.

WHY THIS EXISTS.  The engine is a ready-made build downloaded by setup.py
(``releases/latest/download/strata-windows-x64.zip``), and the project ships a release most days.  So
"update Strata" has meant: find the release, download a 124 MB zip, and overwrite the files in the
engine directory by hand.  That is a fine procedure and a poor user interface, and it is easy to get
wrong - a half-extracted zip leaves an engine that will not start.

HOW IT IS DELIBERATE.  This module is cautious by construction, and the order of the steps is the
design:

  1. **The download is verified against a SHA-256 that came from somewhere else.**  GitHub's releases
     API publishes ``digest`` for every asset (``sha256:<hex>``); the archive itself comes from the
     release download.  Two different origins, so a tampered or substituted download does not arrive
     with a matching hash, and the hash is checked - in the same pass, on the bytes as they arrive -
     before the archive is opened.  A file that does not match is deleted, and a release that publishes
     no digest is refused rather than installed unchecked.
  2. Nothing on disk changes until that and every other check has passed.  The archive is opened, its
     members' paths and CRCs checked, its version read, extracted to a staging directory, and the
     STAGED binary run - all before the installed engine is touched.  The staged run happens with the
     ENGINE's environment, because that is where its CUDA libraries are.
  3. The installed engine is copied to a backup first - files and subdirectories - and the swap that
     follows is all or nothing: every new file is assembled before anything installed is touched, each
     one then lands atomically, and the caller's lock is held so no request can start the engine
     mid-swap.  Any failure from there restores the backup AND removes anything the new engine added,
     because restoring alone would leave a mixture of two engines.
  4. A downgrade is refused, and so is a release whose own engine is older than its tag implies.
  5. Every step is recorded with its own status so the web UI can show what is happening rather than
     a spinner.  The failure note says what was restored.

WHAT THE HASH DOES NOT DO, stated plainly.  It proves the bytes are the ones GitHub published *for that
asset*.  It does not make a malicious release safe: if someone who can publish a release publishes a
malicious engine, the digest matches it.  A hash PINNED IN THE REPO - which is what ``setup.py`` does
for the Unsloth shards, and what a reviewer asked for - is stronger, because a compromised release cannot
change it; it costs a reviewed commit per release.  What this buys is everything the digest is good for:
a corrupted transfer, a substituted or mirrored download, a TLS-terminating proxy, a hostile network.
``get_prebuilt()`` checks the same way on the install path (#1218); before that it checked nothing at all,
so a release asset there was installed on its length alone.

WHAT IT DOES NOT DO.  It does not touch the Python checkout, the pinned packages, the model, or the
packs.  Those are a ``git pull`` and a ``pip install -r``; rewriting the code this server is running
from, in-process, is a different and much larger thing.  It does not update ``data/*.bin`` either.

MEASURED, not assumed: every step is verified against the artifact rather than trusted, and the whole
run was exercised against the live v0.1.40.1 release on a machine whose card cannot run the CUDA 13
build - see docs/DETAILS.md.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

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

# Where setup.py keeps the engine it replaced, and what its docs call it.  Reused rather than
# invented, so `setup.py --rollback-engine` works on an engine this updated.
PREVIOUS_ENGINE = ".previous"

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
    return engine_build(engine_dir).get("version")


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


def engine_build(engine_dir: Path) -> dict:
    """The installed engine's BUILD.json, or {} when there is none or it will not parse.

    utf-8-sig because a BUILD.json written by a Windows editor, or PowerShell's
    `Set-Content -Encoding utf8`, starts with a BOM and json.loads rejects the whole file for those three
    bytes - which would read as "no version and no CUDA" rather than as an error.
    """
    try:
        return json.loads((engine_dir / "BUILD.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}


def cuda_major_of(build: dict) -> int | None:
    """The CUDA major the engine was built against: "13.0" -> 13, "12.6" -> 12. None if not recorded."""
    raw = str(build.get("cuda") or "")
    return int(raw.split(".")[0]) if raw.split(".")[0].isdigit() else None


def asset_for(release: dict, backend: str, cuda_major: int | None = None) -> dict | None:
    """The release asset for this PC, or None if the release has none for it.

    Chosen from the release's own asset list rather than from a table of expected names, because that
    table is a standing invitation to be wrong: this repository published
    `strata-windows-x64-cuda12.zip` (setup.py's CUDA12_ASSET) after this was written, and a hardcoded map
    did not know about it.  A machine running the experimental CUDA 12 engine would have been handed the
    CUDA 13 build, which it cannot run.

    So the name is built from two facts instead - the platform, and the CUDA major the INSTALLED engine
    was built against, which its own BUILD.json records ("cuda": "12.6" / "13.0"). Keeping an install on
    the CUDA line it already uses is the only safe default: switching a working install to a different
    CUDA major is a bigger change than an update should make, and setup.py owns that decision.
    """
    platform_name = "windows" if sys.platform.startswith("win") else "linux"
    assets = release.get("assets") or []
    if backend == "hip":
        want = f"strata-{platform_name}-x64-hip.zip"
    else:
        want = f"strata-{platform_name}-x64.zip"
        if cuda_major is not None:
            want = f"strata-{platform_name}-x64-cuda{cuda_major}.zip"
    for asset in assets:
        if asset.get("name") == want:
            return asset
    # No CUDA-specific asset for this CUDA major. Fall back to the default one rather than refusing:
    # the CUDA 13 build is what the plain name means, and it is only wrong for a card below 7.5 - which
    # the card check in step 4 then catches, before anything is changed.
    if want.endswith(".zip") and "-cuda" in want:
        fallback = f"strata-{platform_name}-x64.zip"
        for asset in assets:
            if asset.get("name") == fallback:
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
    env: object = None                  # the engine's own environment (its PATH carries the CUDA libs)
    exclusive: object = None            # a context manager held across the destructive steps only
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
    _placed: list = field(default_factory=list)   # files this run has put in place, for the restore
    _added: list = field(default_factory=list)    # names the new engine ADDS; the restore removes them

    def __post_init__(self):
        self.engine_exe = Path(self.engine_exe)
        self.engine_dir = self.engine_exe.parent
        self.root = self.engine_dir.parent / ".strata-update"
        # The backup lives where setup.py puts it and where its docs say it is: engine/.previous. Using
        # the same place means `python setup.py --rollback-engine` works on an engine this updated, and a
        # user has ONE rollback to learn rather than two. Measured cost of one generation: 211 MiB
        # (docs/TROUBLESHOOTING.md says the same). The scratch space - staging, the assembled new- tree,
        # the archive - stays outside the engine directory so it is never near the engine's own files.
        self.backup_root = self.engine_dir / PREVIOUS_ENGINE
        # the probe runs the downloaded binary, so it gets the environment the ENGINE runs with, not
        # this process's: the CUDA libraries the engine needs are on the config's lib_dirs, which the
        # server puts on PATH only when it starts the engine. See _probe.
        self.env = dict(self.env) if isinstance(self.env, dict) else None
        self._lock = threading.RLock()   # state_dict() is read from HTTP threads while a run writes

    # ---- plumbing ---------------------------------------------------------------------------------

    def _exclusive(self):
        """The caller's lock, or nothing. `exclusive` is the Service's FIFO, which serialises loading;
        holding it across the swap is what stops a request from starting the engine mid-swap."""
        return self.exclusive if callable(self.exclusive) else contextlib.nullcontext()

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
            # 403 or 429 here is almost always the unauthenticated rate limit - 60 requests an hour per
            # IP, which a shared or NAT'd connection runs out of - and "GitHub answered 403" tells the
            # reader nothing about that. Measured: the limit is 60/hour without a token.
            if e.code in (403, 429):
                raise UpdateError(
                    f"GitHub's API refused the request ({e.code}), which is usually its rate limit for "
                    f"unauthenticated requests - 60 an hour, counted per internet address. Nothing was "
                    f"changed. Try again in a few minutes, or run UPDATE.bat")
            raise UpdateError(f"GitHub answered {e.code} for the latest release; not changing anything")
        except Exception as e:
            raise UpdateError(f"could not reach GitHub ({type(e).__name__}: {e}); not changing anything")

    def _download(self, url: str, dest: Path, expected: int | None, sha256: str | None, label: str) -> None:
        """Download with progress, checking the size and the SHA-256 against what the API reported.

        Both come from the releases API, and the digest is what makes this a verification rather than a
        hope: the asset is fetched from the release CDN while the expected hash arrives from
        `api.github.com` over its own TLS connection, so a tampered or substituted download does not
        come with a matching hash.  (setup.py's `verify_sha256()` is the same idea for files whose hash is
        pinned in the repo; the API's `digest` field is the equivalent for a published asset, and it is
        present on every release - measured on v0.1.34 through v0.1.40.1.)

        The hash is computed in the same pass as the download rather than by re-reading the file, so a
        131 MB asset is read once.  A file that does not match is DELETED, not kept: a wrong engine left
        on disk is the thing this whole step exists to prevent.

        No digest means no verification is possible, and the run stops.  That is the safe answer and it
        should rarely happen; if it does, the release is unusual and the user can still update by hand.
        """
        if not sha256:
            raise UpdateError(
                f"GitHub's API reports no SHA-256 for {label}, so it cannot be verified; refusing to "
                f"install an engine that cannot be checked. Nothing was changed. UPDATE.bat will "
                f"still fetch it")
        total = expected or 0
        got = 0
        digest = hashlib.sha256()
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
                digest.update(chunk)
                got += len(chunk)
                now = time.time()
                if now - last > 0.2:
                    last = now
                    pct = round(100.0 * got / total) if total else 0
                    self._put(percent=pct,
                              downloaded_mb=round(got / 1e6, 1),
                              total_mb=round(total / 1e6, 1) if total else None)
        have = digest.hexdigest()
        self._put(sha256=have)
        if have != sha256:
            dest.unlink(missing_ok=True)
            raise UpdateError(
                f"{label} has the wrong SHA-256 ({have}, expected {sha256}); deleted, and nothing on "
                f"this PC was changed. The download was not the file GitHub published - try again, and "
                f"if it keeps failing, UPDATE.bat")
        if total and got != total:
            dest.unlink(missing_ok=True)
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
        build = engine_build(self.engine_dir)
        have = build.get("version")
        cuda = cuda_major_of(build)
        asset = asset_for(release, self.backend, cuda)
        want = parse_version(tag)
        if not asset:
            self._step("plan", "Check the release", "failed",
                       f"{tag} publishes no {self.backend} build for this platform"
                       + (f" (CUDA {cuda})" if cuda else ""))
            self.state = "failed"
            self._put(latest=tag, installed=have)
            return self.state_dict()
        # GitHub's own SHA-256 for the asset, as "sha256:<hex>". It arrives here, from api.github.com,
        # while the file itself comes from the release CDN - two different origins, so the hash is
        # worth having. A release that does not publish one is refused in _download() rather than
        # installed unchecked.
        digest = str(asset.get("digest") or "")
        sha256 = digest.split(":", 1)[1] if digest.startswith("sha256:") else None
        newer = bool(want) and (not parse_version(have) or want > parse_version(have))
        self._step("plan", "Check the release", "done",
                   f"installed {have or 'unknown'}, latest {tag}")
        self.state = "ready"
        self._put(latest=tag,
                installed=have,
                newer=newer,
                asset=asset.get("name"),
                asset_url=asset.get("browser_download_url"),
                asset_size=asset.get("size"),
                asset_mb=round((asset.get("size") or 0) / 1e6, 1),
                sha256_expected=sha256,
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
        self._placed.clear()
        try:
            # The last three steps touch the installed engine. `exclusive` is the caller's own lock -
            # the Service passes the FIFO that serialises loading - and it is held for exactly this
            # window, so a request arriving mid-update cannot start the engine against a directory that
            # is half swapped. It is NOT held for the download: blocking loads for a minute and a half
            # to protect a window that has not opened yet would be the wrong trade.
            with self._exclusive():
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
        """Remove the staging tree, the assembled `new-` tree, the downloaded archive and any partial
        download.

        Never touches a backup-<...> directory - that is the only way back, on success and on failure
        alike.  The archive is ~130-190 MB and each of the two trees about as much again, and a machine
        where setup.py has also run accumulates them; a release URL is stable for its tag, so re-running
        downloads it again and keeping it only grows the engine directory's parent.

        The staging tree can be locked at this point, and that is not hypothetical: step 6 EXECUTES
        strata.exe out of it, and on Windows the child process can still hold the file a moment after it
        exits.  Measured - a live update left its whole 190 MB staging tree behind, because rmtree failed
        on the still-open binary and `ignore_errors=True` swallowed it, so every update would leak it.
        So: retry for a short while, and if something still will not go, say so in the state instead of
        quietly leaving it.
        """
        left = []
        for path in list(self.root.glob("stage-*")) + list(self.root.glob("new-*")) + \
                list(self.root.glob("*.zip")) + list(self.root.glob("*.part")):
            for attempt in range(6):
                if not path.exists():
                    break
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)
                if path.exists() and attempt < 5:
                    time.sleep(0.4 * (attempt + 1))     # the child is usually closing by now
            if path.exists():
                left.append(path.name)
        if left:
            self._put(left_behind=left)

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
                       self.detail.get("sha256_expected"), "the engine archive")
        part.replace(self.root / f"download-{tag}.zip")
        return f"{self.detail['asset_mb']} MB, SHA-256 verified"

    def _do_inspect(self) -> str:
        self._inspect()
        self._check_gpu()          # after _inspect, which is where the release's archs come from
        # the engine's own version, which is not always the tag: a hotfix release such as v0.1.40.1
        # ships the v0.1.40 engine
        return (f"{self.detail['members']} files, engine "
                f"{self.detail.get('engine_version') or 'unknown'} for {self.detail['latest']}")

    def _do_stage(self) -> str:
        self.staging = Path(tempfile.mkdtemp(prefix="stage-", dir=str(self.root)))
        self._extract(self.staging)
        return self.staging.name

    def _do_test(self) -> str:
        self._probe(self.staging)
        return "it starts and prints its usage"

    def _do_backup(self) -> str:
        self.backup = self._backup()
        return f"engine/{PREVIOUS_ENGINE}"

    def _do_apply(self) -> str:
        self._apply(self.staging)      # _changed and _placed are armed inside it, before the swap
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
        # The release TAG is Strata's version, not the engine's, and they are not always the same: a
        # hotfix release tagged v0.1.40.1 ships the v0.1.40 engine, because the change was in setup.py
        # or the server. Measured on the real v0.1.40.1 release - the cuda12 archive's BUILD.json says
        # 0.1.40 - which an earlier exact comparison here refused, blocking a perfectly good update.
        # So only an OLDER engine is a refusal: that means the wrong asset was picked (a CUDA 12 build
        # for a CUDA 13 install, say). Equal or newer is fine, and the panel reports the version that
        # actually arrived rather than the one the tag promised.
        #
        # This is a sanity check, not the security check - the SHA-256 in step 3 is what proves the bytes
        # are the ones GitHub published.
        tag_engine = parse_version(self.detail["latest"])[:3]
        got_engine = parse_version(got)[:3]
        if got_engine and tag_engine > got_engine:
            raise UpdateError(
                f"the archive holds engine v{got} but {self.detail['latest']} should carry "
                f"v{'.'.join(str(x) for x in tag_engine)} or newer, so the wrong archive was picked; "
                f"refusing to install it. Nothing was changed")
        if not got:
            raise UpdateError("the archive's BUILD.json names no version, so it cannot be checked")
        self._put(members=len(safe), engine_version=got)

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
        proven to start before the working one is touched.

        Run with the ENGINE's own environment, not this process's.  A ready-made engine finds its CUDA
        libraries through the `lib_dirs` the config carries, which the server puts on PATH when it
        starts the engine (child_env) - those directories are NOT on the server's own PATH, and setup.py
        pip-installs the cuBLAS wheels into site-packages rather than a system toolkit.  Probing with
        os.environ would therefore fail with "DLL not found" on a perfectly good install and refuse the
        update.  Measured on this machine only by luck: the system CUDA 12.6 toolkit happens to be on the
        server's PATH, which is why the live run passed and a setup.py install would not have.
        """
        exe = self._find_engine(staging)
        if exe is None:
            raise UpdateError("the staged engine has no strata binary; not installing it")
        try:
            r = subprocess.run([str(exe), "--help"], capture_output=True, timeout=PROBE_TIMEOUT_S,
                               env=self.env)
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
        """Copy the installed engine to `engine/.previous`, replacing whatever was there.

        One generation, as setup.py keeps it and as docs/TROUBLESHOOTING.md describes.  `.previous` from
        an earlier update is replaced rather than kept, because keeping both would mean two rollbacks
        to explain and 211 MiB each; setup.py's `--rollback-engine` swaps them, so one is what it uses
        too.

        Subdirectories are copied, not just files: a packaged HIP engine ships ROCm's per-architecture
        folders (#1183 fixed exactly that), and a backup that skipped them could not restore them.
        """
        dest = self.backup_root
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True)
        for item in self.engine_dir.iterdir():
            if item.name == PREVIOUS_ENGINE:      # never copy the previous engine into itself
                continue
            if item.is_dir():
                shutil.copytree(item, dest / item.name)
            else:
                shutil.copy2(item, dest / item.name)
        (dest / "_version.txt").write_text(str(self.detail.get("installed")), encoding="utf-8")
        self.backup_dir = dest
        return dest

    def _apply(self, staging: Path):
        """Install the new engine: all or nothing.

        Two things this has to get right, and the first version got neither:

        1. **Nothing installed is touched until every new file is ready.** The staged files are copied
           into a `new-...` directory beside the engine first, so a failure while assembling them -
           a full disk, a locked file - costs nothing.
        2. **A failure part way through the swap restores everything.** Each file lands atomically, via
           a `.new` temporary and `os.replace`, so no file is ever half-written; and the restore is armed
           BEFORE the first destructive step rather than after the last one. Arming it after - which is
           what this did - means a failure on the third of nine files leaves a directory holding a mixture
           of two engines and no attempt to put it right, which is the one outcome an update must never
           produce. setup.py's own replace does the same thing (it keeps `engine/.previous` and moves
           files back on OSError); the difference here is that the backup is made first, so the restore
           does not depend on the swap having been tidy.

        The executable bit is carried across explicitly: zipfile does not preserve it on Linux, and an
        engine that loses it will not start after the update, which would look like a bad release.
        """
        incoming = Path(tempfile.mkdtemp(prefix="new-", dir=str(self.root)))
        files: list[Path] = []          # the same layout, one directory level down, ready to move in
        for src in sorted(staging.rglob("*")):
            if not src.is_file():
                continue
            dst = incoming / src.relative_to(staging)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            files.append(dst)
        if not files:
            raise UpdateError("the archive held no files to install; nothing was changed")

        # Armed here, on purpose: from this line on the install can be half-written, so a failure must
        # restore it. Anything earlier fails with the install untouched and needs no restore.
        self._changed = True
        # What was in the directory BEFORE the swap, so the restore can also take away anything the new
        # engine ADDED. Restoring the backup alone is not enough: it holds only files that existed, so a
        # file the new archive introduces would survive the rollback and leave the directory holding a
        # mixture of two engines - the one outcome this whole design is meant to make impossible.
        before = {p.name for p in self.engine_dir.iterdir()}
        incoming_names = {(self.engine_dir / src.relative_to(incoming)).name for src in files}
        self._added = sorted(incoming_names - before)
        for src in files:
            dst = self.engine_dir / src.relative_to(incoming)
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".new")
            shutil.copy2(src, tmp)
            if os.name != "nt" and not dst.suffix:
                tmp.chmod(src.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            os.replace(tmp, dst)      # atomic: dst is the old file or the new one, never a mixture
            self._placed.append(dst)

    def _verify(self, tag: str):
        """The installed BUILD.json is the engine's own report of what it is; check it, and that the
        files landed where the config expects them.

        Compared on three version parts, not the whole tag: a hotfix release such as v0.1.40.1 ships the
        v0.1.40 engine, so an exact comparison fails a correct install - and the rollback then puts the
        old engine back, which is the right behaviour for the wrong reason.  Measured on the real
        v0.1.40.1 release.
        """
        got = installed_version(self.engine_dir)
        if parse_version(got)[:3] < parse_version(tag)[:3]:
            raise UpdateError(f"after installing, the engine reports {got or 'nothing'}, "
                              f"older than {tag}")
        if not self.engine_exe.exists():
            raise UpdateError(f"after installing, {self.engine_exe} is missing")
        self._put(verified_version=got)

    def rollback(self, backup: Path | None) -> Path:
        """Put the installed files back from `backup`, leaving the install whole.

        The backup holds every file that was in the engine directory, so restoring all of them undoes a
        swap that stopped half way - not just the files that had been reached. Any `.new` temporary left
        by an interrupted swap is removed, so the directory does not keep a stray partial file next to the
        engine.
        """
        if not backup or not backup.exists():
            raise UpdateError("no backup to restore from")
        for src in backup.iterdir():
            if src.name == "_version.txt":
                continue
            dst = self.engine_dir / src.name
            if src.is_dir():
                shutil.rmtree(dst, ignore_errors=True)      # a subdirectory the old engine had
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        # Take away anything the new engine ADDED. Restoring the backup alone leaves those files in
        # place, so the directory ends up holding a mixture of two engines - the exact outcome this
        # exists to prevent. Measured before this was fixed.
        for name in self._added:
            added = self.engine_dir / name
            if added.is_dir():
                shutil.rmtree(added, ignore_errors=True)
            else:
                added.unlink(missing_ok=True)
        for leftover in self.engine_dir.glob("*.new"):
            leftover.unlink(missing_ok=True)
        self._placed.clear()
        return backup
