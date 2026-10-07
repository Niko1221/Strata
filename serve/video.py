"""Finite-video policy and the Qwen 16x2x2 frame contract (no model imports).

Sampling follows the pinned Qwen3VL processor. Resolution is an explicit serving
policy, not a claim that mtmd's image defaults are the checkpoint's video defaults.
"""
from __future__ import annotations

import math
import struct
import time
from dataclasses import asdict, dataclass, fields


class VideoError(ValueError):
    pass


class VideoCancelled(VideoError):
    pass


class VideoLimitError(VideoError):
    pass


# RGB frame spool: header, then (float64 seconds, tightly packed RGB24) per frame.
# Files are owned intermediates, NOT another public/API upload format.
FRAME_HEADER = struct.Struct("<4sIIIIdQ")
FRAME_TIME = struct.Struct("<d")
VIDEO_PROFILE = "qwen4_exp_16x2x2_2560_v1"
PREPROCESS_VERSION = 1


@dataclass(frozen=True)
class VideoPolicy:
    fps: float = 2.0
    min_group_tokens: int = 8
    max_group_tokens: int = 256
    max_source_bytes: int = 64 << 20
    max_duration_s: float = 60.0
    max_frames: int = 128
    max_source_frames: int = 8192
    max_source_side: int = 8192
    max_source_pixels: int = 16 << 20
    max_rgb_bytes: int = 256 << 20
    max_tokens: int = 16384
    max_embedding_bytes: int = 256 << 20
    max_disk_bytes: int = 512 << 20
    cache_bytes: int = 256 << 20
    deadline_s: float = 120.0
    decoder_memory_bytes: int = 2 << 30
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"

    @classmethod
    def from_config(cls, config, image_min=0, image_max=0):
        if config is None or config is False:
            return None
        if not isinstance(config, dict):
            raise VideoError('"vision.video" must be an object')
        enabled = config.get("enabled", False)
        if not isinstance(enabled, bool):
            raise VideoError('"vision.video.enabled" must be true or false')
        if not enabled:
            return None
        names = {f.name for f in fields(cls)}
        if set(config) - names - {"enabled"}:
            raise VideoError("unknown vision.video setting: " + ", ".join(sorted(set(config) - names - {"enabled"})))
        values = {k: v for k, v in config.items() if k in names}
        values.setdefault("min_group_tokens", max(8, image_min or 0))
        p = cls(**values)
        for f in fields(p):
            v = getattr(p, f.name)
            if f.name in ("ffmpeg", "ffprobe"):
                if not isinstance(v, str) or not v or any(c in v for c in "\x00\r\n"):
                    raise VideoError(f"vision.video.{f.name} must name an executable")
            elif f.name in ("fps", "deadline_s", "max_duration_s"):
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                    raise VideoError(f"vision.video.{f.name} must be finite and positive")
            elif isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                raise VideoError(f"vision.video.{f.name} must be a positive integer")
        if not (p.fps <= 10 and p.max_duration_s <= 3600 and p.deadline_s <= 3600):
            raise VideoError("video FPS/duration/deadline exceeds the supported ceiling")
        if not (max(8, image_min or 0) <= p.min_group_tokens <= p.max_group_tokens <= min(1024, image_max or 4096)):
            raise VideoError("video group tokens must fit the existing image encoder's min/max token policy (8..1024)")
        if p.max_frames > 128 or p.max_tokens > 16384 or p.max_source_frames > 65536:
            raise VideoError("video frames/tokens exceed the supported transport ceiling")
        if p.max_rgb_bytes > 256 << 20 or p.max_embedding_bytes > 256 << 20:
            raise VideoError("video RGB/embedding budgets must be at most 256 MiB")
        if p.cache_bytes > p.max_disk_bytes or p.max_source_bytes > p.max_disk_bytes:
            raise VideoError("video source/cache budget exceeds the disk budget")
        if p.max_source_side > 16384 or p.max_source_pixels > 64 << 20 or p.decoder_memory_bytes > 8 << 30:
            raise VideoError("video dimensions/decoder memory exceed the supported ceiling")
        return p

    def identity(self):
        return {"version": PREPROCESS_VERSION, "profile": VIDEO_PROFILE, **asdict(self)}


class VideoRequestBudget:
    """Cumulative logical costs; even cache hits and repeated clips consume them."""
    def __init__(self, policy: VideoPolicy, cancel=None):
        self.policy, self.cancel = policy, cancel
        self.deadline = time.monotonic() + policy.deadline_s
        self.source_bytes = self.frames = self.rgb_bytes = self.tokens = self.embedding_bytes = 0
        self.duration_s = 0.0

    def check(self):
        if self.cancel is not None and self.cancel.is_set():
            raise VideoCancelled("video request cancelled")
        if time.monotonic() >= self.deadline:
            raise VideoLimitError("video decode/encode deadline exceeded")

    def charge(self, **costs):
        self.check()
        for key, n in costs.items():
            if key not in ("source_bytes", "frames", "rgb_bytes", "tokens", "embedding_bytes", "duration_s"):
                raise VideoError("unknown video budget cost")
            if isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) or n < 0:
                raise VideoError("invalid video budget cost")
            value = getattr(self, key) + n
            if value > getattr(self.policy, "max_" + key):
                raise VideoLimitError(f"video request exceeds its {key} budget")
            setattr(self, key, value)


def sample_indices(total_frames: int, source_fps: float, target_fps: float = 2.0) -> tuple[int, ...]:
    """The pinned sample_frames method: linspace, ties-to-even, min 4/max 768.

    Serving limits reject a too-large result; they never silently change sampling.
    """
    if isinstance(total_frames, bool) or not isinstance(total_frames, int) or total_frames < 1:
        raise VideoError("video has no finite frame count")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0
           for x in (source_fps, target_fps)):
        raise VideoError("video has no finite positive FPS")
    count = min(total_frames, max(4, min(768, int(total_frames / source_fps * target_fps))))
    if count == 1:
        return (0,)
    step = (total_frames - 1) / (count - 1)
    return tuple(round(i * step) if i < count - 1 else total_frames - 1 for i in range(count))


def resize_shape(height: int, width: int, frames: int, policy: VideoPolicy) -> tuple[int, int]:
    """Pinned smart_resize with explicit per-group min/max resolution budgets.

    A one-frame input is explicitly duplicated for temporal patch size 2 BEFORE
    resize; larger odd inputs are padded AFTER resize, as in the processor.
    """
    if min(height, width, frames) < 1:
        raise VideoError("invalid video dimensions/frame count")
    factor = 32
    frames = max(2, frames)
    if height < factor or width < factor:
        scale = max(factor / height, factor / width)
        height, width = int(height * scale), int(width * scale)
    if max(height, width) / min(height, width) > 200:
        raise VideoError("video aspect ratio exceeds 200")
    h, w = round(height / factor) * factor, round(width / factor) * factor
    temporal = round(frames / 2) * 2
    minimum = temporal * policy.min_group_tokens * factor * factor
    maximum = temporal * policy.max_group_tokens * factor * factor
    if temporal * h * w > maximum:
        beta = math.sqrt(frames * height * width / maximum)
        h = max(factor, math.floor(height / beta / factor) * factor)
        w = max(factor, math.floor(width / beta / factor) * factor)
    elif temporal * h * w < minimum:
        beta = math.sqrt(minimum / (frames * height * width))
        h, w = math.ceil(height * beta / factor) * factor, math.ceil(width * beta / factor) * factor
    rows = h // factor * (w // factor)
    if not policy.min_group_tokens <= rows <= policy.max_group_tokens:
        raise VideoError("reference resize cannot fit this aspect ratio in the configured group token budget")
    return h, w


@dataclass(frozen=True)
class ClipInfo:
    frames: int
    source_fps: float
    width: int
    height: int
    duration_s: float
    indices: tuple[int, ...]
    resized_width: int
    resized_height: int
    rotation: int = 0

    @property
    def rows(self):
        return ((len(self.indices) + 1) // 2) * (self.resized_width // 32) * (self.resized_height // 32)

    @property
    def rgb_bytes(self):
        return len(self.indices) * self.resized_width * self.resized_height * 3

    @property
    def packet_bytes(self):
        return FRAME_HEADER.size + len(self.indices) * FRAME_TIME.size + self.rgb_bytes

    def charge(self, budget):
        budget.charge(duration_s=self.duration_s, frames=len(self.indices), rgb_bytes=self.rgb_bytes)


def probe_info(data: dict, policy: VideoPolicy) -> ClipInfo:
    """Untrusted ffprobe JSON. First implementation accepts constant-rate video.

    Every decoded frame's PTS/dimensions is checked, not only the format's claims.
    Audio is ignored. Playlists, variable rates and changing resolution are refused.
    """
    try:
        streams = data["streams"]
        if not isinstance(streams, list) or len(streams) != 1:
            raise VideoError("video must have a first video stream")
        s = streams[0]
        w, h = s["width"], s["height"]
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 or x > policy.max_source_side for x in (w, h)):
            raise VideoError("video source dimensions exceed the configured limit")
        if w * h > policy.max_source_pixels:
            raise VideoError("video source pixels exceed the configured limit")
        if max(w, h) / min(w, h) > 200:
            raise VideoError("video source aspect ratio exceeds 200")
        a, b = s["avg_frame_rate"].split("/")
        fps = int(a) / int(b)
        if not math.isfinite(fps) or not 0.1 <= fps <= 240:
            raise VideoError("video source FPS must be finite (0.1..240)")
        records = data["frames"]
        if not isinstance(records, list) or not 1 <= len(records) <= policy.max_source_frames:
            raise VideoError("video source frame count exceeds the configured limit")
        times = []
        for frame in records:
            if frame.get("width") != w or frame.get("height") != h:
                raise VideoError("videos with changing resolution are not supported")
            t = float(frame["best_effort_timestamp_time"])
            if not math.isfinite(t):
                raise VideoError("video frame timestamps must be finite")
            times.append(t)
        # ffprobe prints microseconds. Permit that quantization, not VFR or guessed timestamps.
        if any(abs((t - times[0]) - i / fps) > 3e-6 for i, t in enumerate(times)):
            raise VideoError("variable-frame-rate videos are not supported; use a constant-frame-rate clip")
        duration = len(times) / fps
        if duration > policy.max_duration_s:
            raise VideoError("video duration exceeds the configured limit")
        rotation = 0
        for side in s.get("side_data_list", []):
            if "rotation" in side:
                r = float(side["rotation"])
                if not math.isfinite(r) or r % 90:
                    raise VideoError("video rotation must be a multiple of 90 degrees")
                rotation = int(r) % 360
        if rotation % 180:
            w, h = h, w
        indices = sample_indices(len(times), fps, policy.fps)
        if len(indices) > policy.max_frames:
            raise VideoError("video sampling exceeds the frame budget; no frames were silently dropped")
        rh, rw = resize_shape(h, w, len(indices), policy)
        info = ClipInfo(len(times), fps, w, h, duration, indices, rw, rh, rotation)
        if info.rows > policy.max_tokens or info.rgb_bytes > policy.max_rgb_bytes:
            raise VideoError("video exceeds its visual token/RGB budget")
        return info
    except VideoError:
        raise
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        raise VideoError("video probe returned invalid or incomplete metadata") from None
