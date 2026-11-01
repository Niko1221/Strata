"""Getting the DFlash2 drafter onto this computer: a pinned download, checked against the repository's own SHA-256, packed for the engine.

    strata drafting plan  [--data-dir DIR]      what it would download (nothing is changed)
    strata drafting fetch [--data-dir DIR]      download, check, pack; progress in plain words

Only an explicit command reaches the network (strata_control/acquisition.py): a start never downloads. The drafter is
incoai/Qwen3.8-Flash-Next-DFlash2 at one revision (strata_control/drafting.py has the revision, the file sizes and hashes). The 1.2 GB
safetensors file is packed (tools/dflash2_pack.py, numpy only) into the engine's 0.76 GB runtime file and then removed.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from strata_control import acquisition, drafting, transfer

Progress = Callable[[str, int, int], None]    # (step in plain words, bytes done, bytes total)


class DraftingError(RuntimeError):
    """The drafter could not be fetched or packed. The message is plain words and says what to do."""


def plan(data_dir: str | Path) -> dict:
    """What fetching would do, in numbers and words. ``gb`` is 0 when the drafter is already here."""
    ready = drafting.pack_ready(data_dir)
    gb = 0.0 if ready else drafting.DOWNLOAD_GB
    if ready:
        text = "The DFlash2 drafter is already on this computer."
    else:
        text = (f"The DFlash2 drafter is {drafting.DOWNLOAD_GB:.1f} GB to download (it is packed to {drafting.PACK_GB:.2f} GB for the engine and the "
                f"download is removed), from {drafting.REPO}, {drafting.LICENSE}. It needs about {drafting.VRAM_GB:.1f} GB of graphics memory; "
                "if it does not fit, the model starts with the MTP instead.")
    return {"ready": ready, "gb": gb, "pack": str(drafting.pack_path(data_dir)), "repo": drafting.REPO, "revision": drafting.REVISION, "text": text}


def _packer():
    """tools/dflash2_pack.py as a module: beside strata_control in a source checkout and in the packaged Control (TOOLS_SUBSET)."""
    path = Path(__file__).resolve().parents[1] / "tools" / "dflash2_pack.py"
    if not path.is_file():
        raise DraftingError("The packing tool is missing from this installation. Update Strata, then try again.")
    spec = importlib.util.spec_from_file_location("strata_dflash2_pack", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("strata_dflash2_pack", module)
    spec.loader.exec_module(module)
    return module


REACH_SECONDS = 6.0


def online(url: str, timeout: float = REACH_SECONDS) -> bool:
    """Does the download address answer at all? One short HEAD request, no retry: with no internet (or a connection that hangs) the person gets a plain
    sentence in a few seconds instead of a stalled progress bar. Any answer from the server counts, even an error page."""
    try:
        transfer.check_url_policy(url)
        request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "strata-drafting/1"})
        with transfer._opener.open(request, timeout=timeout):     # the download's own opener: same certificates, same redirect policy, same proxy settings
            return True
    except urllib.error.HTTPError:
        return True
    except (urllib.error.URLError, OSError, TimeoutError, ValueError, transfer.TransferError) as exc:
        from strata_control import tls
        return bool(tls.is_certificate_error(exc))        # a certificate problem is an answer: the download explains it in its own words


def _hub_snapshots() -> list[Path]:
    """Where the Hugging Face tools keep this repository's pinned revision, if the person already has it from them (nothing is created or changed)."""
    roots: list[Path] = []
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(name):
            roots.append(Path(os.environ[name]))
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    return [r / ("models--" + drafting.REPO.replace("/", "--")) / "snapshots" / drafting.REVISION for r in roots]


def _present(folders: list[Path], spec) -> dict[str, Path]:
    """The pinned files that are already on this computer, found in these folders and checked against the repository's size and SHA-256."""
    found: dict[str, Path] = {}
    for name, size, sha in spec:
        for folder in folders:
            candidate = folder / name
            try:
                if candidate.is_file() and candidate.stat().st_size == size and transfer.sha256_file(candidate) == sha:
                    found[name] = candidate
                    break
            except OSError:
                continue
    return found


def ensure(data_dir: str | Path, *, progress: Progress | None = None, cancel: threading.Event | None = None,
           base_url: str | None = None, files=None) -> Path:
    """Make the packed drafter exist under ``data_dir`` and return its path. A pack that is there is kept. Resumable: an interrupted
    download is continued, a file whose hash differs is deleted and the person is told. ``base_url`` / ``files`` are for tests."""
    report = progress or (lambda *_: None)
    out = drafting.pack_path(data_dir)
    if drafting.pack_ready(data_dir):
        report("The DFlash2 drafter is ready.", 1, 1)
        return out
    acquisition.require("the DFlash2 drafter", "strata drafting fetch")
    folder = out.parent
    src = folder / "src"
    spec = tuple(files if files is not None else drafting.FILES)
    have = _present([src, *(_hub_snapshots() if files is None else [])], spec)       # already here and checked: used where it is, never fetched again
    if len(have) < len(spec) and not online(base_url or drafting.BASE_URL):
        raise DraftingError("There is no internet connection, so the DFlash2 drafter can't be downloaded now.")
    src.mkdir(parents=True, exist_ok=True)
    total = sum(size for _n, size, _h in spec)
    done_before = 0
    for name, size, sha in spec:
        if name not in have:
            base = done_before
            partial = src / (name + ".part")
            try:
                transfer.download((base_url or drafting.BASE_URL) + name, partial, size=size, sha256=sha, cancel=cancel,
                                  progress=lambda d, _s, base=base: report("Downloading the DFlash2 drafter…", base + d, total))
            except transfer.ChecksumMismatch:
                raise DraftingError("The downloaded DFlash2 file did not pass the safety check, so it was deleted. Try again; if it keeps "
                                    "happening, use MTP for now.") from None
            except transfer.TransferError as exc:
                raise DraftingError(f"The DFlash2 download did not finish: {exc}. Try again; it continues where it stopped.") from None
            os.replace(partial, src / name)
            have[name] = src / name
        done_before += size
    report("Packing the DFlash2 drafter for the engine…", total, total)
    tmp = out.with_suffix(".tmp")
    try:
        _packer().pack(str(have["model.safetensors"]), str(have["config.json"]), str(tmp), say=lambda *_: None)
        os.replace(tmp, out)
    except DraftingError:
        raise
    except Exception as exc:                         # noqa: BLE001 - a plain sentence for any packing failure; the cause stays in the chain
        tmp.unlink(missing_ok=True)
        raise DraftingError("The DFlash2 drafter could not be prepared on this computer. The model keeps using the MTP.") from exc
    for name, _size, _sha in spec:                   # the source files are 1.2 GB of nothing the engine reads (a copy kept by the Hugging Face tools is theirs: left alone)
        if have[name].parent == src:
            have[name].unlink(missing_ok=True)
    try:
        src.rmdir()
    except OSError:
        pass
    report("The DFlash2 drafter is ready.", total, total)
    return out
