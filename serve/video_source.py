"""Owned video spools, cumulative disk accounting and bounded source I/O."""
from __future__ import annotations

import base64
import contextlib
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
import time
import urllib.parse

from .media_process import OwnedMediaProcess
from .video import ClipInfo, VideoCancelled, VideoError, VideoLimitError


def network_path(path: str) -> bool:
    p = path.replace("/", "\\")
    return p.startswith("\\\\") or p.startswith("\\??\\")


def local_video_path(source):
    scheme = urllib.parse.urlsplit(source).scheme
    if scheme and scheme != "file" and not re.match(r"^[A-Za-z]:[\\\\/]", source):
        raise VideoError("video source must be a local path, file URI, data URL, or http(s) URL")
    path = source
    if source.startswith("file://"):
        # Accept drive-letter spellings as the existing image loader does, but
        # refuse an authority/encoded UNC BEFORE anything queries the filesystem.
        rest = source[7:]
        host = re.split(r"[/\\]", rest, maxsplit=1)[0]
        if host and not re.fullmatch(r"[A-Za-z]:|localhost", host, re.IGNORECASE):
            raise VideoError("video files on another computer are not read")
        if host.lower() == "localhost":
            rest = rest[len(host):]
        path = urllib.parse.unquote(rest)
        if os.name == "nt" and re.match(r"^/[A-Za-z]:/", path):
            path = path[1:]
    if network_path(path):
        raise VideoError("video files on another computer are not read (network paths are refused)")
    if not path or any(ord(c) < 32 or 0xD800 <= ord(c) <= 0xDFFF for c in path) or len(path) > 8192:
        raise VideoError("invalid local video path")
    return os.path.abspath(path)


class DiskQuota:
    """Includes active spools, request artifacts AND completed video cache files."""
    def __init__(self, limit):
        self.limit, self.used, self.lock = limit, 0, threading.Lock()

    def claim(self, n):
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise VideoError("invalid video disk reservation")
        with self.lock:
            if self.used + n > self.limit:
                raise VideoLimitError("video disk budget is busy or exceeded; try later or use a smaller clip")
            self.used += n

    def release(self, n):
        with self.lock:
            if not 0 <= n <= self.used:
                raise RuntimeError("video disk reservation ownership mismatch")
            self.used -= n


class OwnedMediaFile:
    def __init__(self, root, quota, maximum, suffix):
        quota.claim(maximum)
        try:
            fd, name = tempfile.mkstemp(prefix="video-", suffix=suffix, dir=root)
            os.close(fd)
        except BaseException:
            quota.release(maximum)
            raise
        self.path, self.quota, self.reserved = Path(name), quota, maximum
        self.closed = False

    def finish(self, size):
        if self.closed or not 0 <= size <= self.reserved or self.path.stat().st_size != size:
            raise VideoError("video artifact exceeds its disk reservation or is incomplete")
        self.quota.release(self.reserved - size)
        self.reserved = size
        return self

    def close(self):
        if not self.closed:
            # Native publication uses this additional owned name. It is never an
            # arbitrary prefix glob, and child processes are reaped before here.
            Path(str(self.path) + ".partial").unlink(missing_ok=True)
            if os.name == "nt" and self.path.exists():
                self.path.chmod(0o600)  # staged sources were read-only; Windows refuses unlink otherwise
            self.path.unlink(missing_ok=True)
            self.quota.release(self.reserved)
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def worker(mode, source, out, budget, maximum=0, info=None):
    budget.check()
    remaining = budget.deadline - time.monotonic()
    policy = {**asdict(budget.policy), "deadline_s": max(0.01, remaining)}
    args = [sys.executable, "-I", str(Path(__file__).with_name("media_worker.py")), mode, str(source), str(out),
            "--policy", json.dumps(policy, separators=(",", ":")), "--maximum", str(maximum)]
    if info is not None:
        args += ["--info", json.dumps(asdict(info), separators=(",", ":"))]
    with OwnedMediaProcess(args, budget, max_stdout=65536, worker_gate=True, check_exit=False) as process:
        raw = process.read()
    try:
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise ValueError()
        if "error" in result:
            error = result["error"]
            code, message = error.get("code"), error.get("message", "video worker failed")
            if code == "worker_error":
                raise RuntimeError(message)
            failure = VideoCancelled if code == "cancelled" else VideoLimitError if code == "limit" else VideoError
            raise failure(message)
        if process.proc.returncode:
            raise RuntimeError("video worker exited without a structured error")
        return result
    except VideoError:
        raise
    except (ValueError, UnicodeError):
        raise VideoError("video worker returned invalid metadata") from None


def stage_video(source, root, quota, budget, *, kind="video", url_maximum=None):
    """Return (owned local source, content SHA256). Downloads stay outside the FIFO."""
    maximum = budget.policy.max_source_bytes - budget.source_bytes
    if url_maximum is not None and source.startswith(("http://", "https://")):
        maximum = min(maximum, url_maximum)
    if maximum <= 0:
        raise VideoLimitError("video source byte budget exceeded")
    owned = OwnedMediaFile(root, quota, maximum, ".source")
    try:
        if source.startswith("data:"):
            header, sep, payload = source.partition(",")
            if not sep or not re.fullmatch(r"data:" + kind + r"/[A-Za-z0-9.+-]+;base64", header):
                raise VideoError(f"{kind} data URL must contain base64 with a {kind}/* media type")
            if not payload or len(payload) % 4:
                raise VideoError("invalid video base64")
            estimated = len(payload) // 4 * 3 - (2 if payload.endswith("==") else 1 if payload.endswith("=") else 0)
            if estimated > maximum:
                raise VideoLimitError("video base64 exceeds its source byte budget")
            digest, size = hashlib.sha256(), 0
            with owned.path.open("wb") as out:
                for i in range(0, len(payload), 65536):
                    budget.check()
                    chunk = payload[i:i + 65536]
                    if "=" in chunk and i + len(chunk) < len(payload):
                        raise VideoError("invalid video base64 padding")
                    try:
                        data = base64.b64decode(chunk, validate=True)
                    except (ValueError, UnicodeError):
                        raise VideoError("invalid video base64") from None
                    size += len(data)
                    if size > maximum:
                        raise VideoLimitError("video source byte budget exceeded")
                    digest.update(data)
                    out.write(data)
            sha = digest.hexdigest()
        else:
            remote = source.startswith(("http://", "https://"))
            if remote and (len(source) > 8192 or any(c in source for c in "\x00\r\n")):
                raise VideoError("invalid video URL")
            result = worker("download" if remote else "copy", source if remote else local_video_path(source),
                            owned.path, budget, maximum)
            size, sha = result.get("bytes"), result.get("sha256")
            if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= maximum or \
                    not isinstance(sha, str) or not re.fullmatch("[0-9a-f]{64}", sha):
                raise VideoError("video source worker returned invalid byte count/hash")
        if size == 0:
            raise VideoError("video source is empty")
        budget.charge(source_bytes=size)
        owned.finish(size)
        owned.path.chmod(0o400)
        return owned, sha
    except BaseException:
        owned.close()
        raise


def decode_video(source, root, quota, budget):
    # Probe first, then reserve the PRECISE finite RGB spool before decode. The
    # source is staged/private and unchanged between the two owned worker jobs.
    result = worker("probe", source, "unused", budget)
    info = result.get("info")
    if not isinstance(info, dict):
        raise VideoError("video decoder returned invalid clip metadata")
    try:
        info = ClipInfo(**{**info, "indices": tuple(info["indices"]), "times": tuple(info["times"])})
    except (TypeError, KeyError):
        raise VideoError("video decoder returned invalid clip metadata") from None
    p = budget.policy
    if not (1 <= info.frames <= p.max_source_frames and 1 <= len(info.indices) <= p.max_frames and
            all(isinstance(i, int) and not isinstance(i, bool) and 0 <= i < info.frames for i in info.indices) and
            info.indices[0] == 0 and
            all(a < b for a, b in zip(info.indices, info.indices[1:])) and
            len(info.times) == len(info.indices) and info.times[0] == 0 and
            all(0 <= t <= info.duration_s for t in info.times) and
            all(a <= b for a, b in zip(info.times, info.times[1:])) and
            0 < info.duration_s <= p.max_duration_s and 0 < info.source_fps <= 240 and
            0 < info.width <= p.max_source_side and 0 < info.height <= p.max_source_side and
            info.width * info.height <= p.max_source_pixels and
            p.min_group_tokens <= info.resized_width // 32 * (info.resized_height // 32) <= p.max_group_tokens and
            info.rows <= p.max_tokens and info.rgb_bytes <= p.max_rgb_bytes and
            info.resized_width % 32 == info.resized_height % 32 == 0):
        raise VideoError("video decoder metadata exceeds its policy")
    info.charge(budget)
    owned = OwnedMediaFile(root, quota, info.packet_bytes, ".svf")
    try:
        result = worker("decode", source, owned.path, budget, info=info)
        info = result.get("info")
        if not isinstance(info, dict):
            raise VideoError("video decoder returned invalid clip metadata")
        info = ClipInfo(**{**info, "indices": tuple(info["indices"]), "times": tuple(info["times"])})
        # The worker is owned/trusted; nevertheless validate its result against
        # policy before cache accounting or native encoder submission.
        p = budget.policy
        if not (1 <= len(info.indices) <= p.max_frames and 0 < info.duration_s <= p.max_duration_s and
                p.min_group_tokens <= info.resized_width // 32 * (info.resized_height // 32) <= p.max_group_tokens and
                info.rows <= p.max_tokens and info.rgb_bytes <= p.max_rgb_bytes and
                info.resized_width % 32 == info.resized_height % 32 == 0):
            raise VideoError("video decoder metadata exceeds its policy")
        owned.finish(info.packet_bytes)
        return owned, info
    except BaseException:
        owned.close()
        raise
