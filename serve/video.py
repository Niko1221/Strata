"""Finite-video policy and the Qwen 16x2x2 frame contract (no model imports).

Frames are chosen on a fixed time grid by actual PTS (sample_times); the pinned Qwen3VL
processor's index-linspace rule is kept as sample_indices for the trace fixtures. Resolution is
an explicit serving policy, not a claim that mtmd's image defaults are the checkpoint's video
defaults.
"""
from __future__ import annotations

import math
import statistics
import struct
import time
from dataclasses import asdict, dataclass, fields

from .media import HEADER as MEDIA_HEADER, SPAN as MEDIA_SPAN


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
PREPROCESS_VERSION = 4


@dataclass(frozen=True)
class VideoPolicy:
    fps: float = 2.0
    min_group_tokens: int = 8
    max_group_tokens: int = 256
    max_source_bytes: int = 256 << 20
    max_duration_s: float = 600.0
    max_frames: int = 1024
    max_source_frames: int = 32768
    max_source_side: int = 8192
    max_source_pixels: int = 16 << 20
    max_rgb_bytes: int = 256 << 20
    max_decoded_bytes: int = 4 << 30
    max_tokens: int = 65536
    max_embedding_bytes: int = 512 << 20
    max_disk_bytes: int = 2 << 30
    cache_bytes: int = 256 << 20
    deadline_s: float = 600.0
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
        for name, ceiling in (("max_frames", 4096), ("max_tokens", 65536), ("max_source_frames", 65536),
                              ("max_rgb_bytes", 1 << 30), ("max_embedding_bytes", 768 << 20),
                              ("max_decoded_bytes", 16 << 30), ("max_source_bytes", 1 << 30),
                              ("max_disk_bytes", 8 << 30), ("cache_bytes", 1 << 30)):
            if getattr(p, name) > ceiling:
                raise VideoError(f"vision.video.{name} exceeds the supported ceiling ({ceiling})")
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
        self.source_bytes = self.frames = self.rgb_bytes = self.decoded_bytes = self.tokens = self.embedding_bytes = 0
        self.duration_s = 0.0

    def check(self):
        if self.cancel is not None and self.cancel.is_set():
            raise VideoCancelled("video request cancelled")
        if time.monotonic() >= self.deadline:
            raise VideoLimitError("video decode/encode deadline exceeded")

    def charge(self, **costs):
        self.check()
        projected = {}
        for key, n in costs.items():
            if key not in ("source_bytes", "frames", "rgb_bytes", "decoded_bytes", "tokens", "embedding_bytes", "duration_s"):
                raise VideoError("unknown video budget cost")
            if isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) or n < 0:
                raise VideoError("invalid video budget cost")
            value = getattr(self, key) + n
            if value > getattr(self.policy, "max_" + key):
                raise VideoLimitError(f"video request exceeds its {key} budget")
            projected[key] = value
        for key, value in projected.items():
            setattr(self, key, value)


def sample_indices(total_frames: int, source_fps: float, target_fps: float = 2.0) -> tuple[int, ...]:
    """The pinned HF processor sample_frames method: index linspace, ties-to-even, min 4/max 768.

    Kept as the reference the trace fixtures were produced with. Serving samples by PTS with
    sample_times() instead, which agrees with this on the frame count for constant-rate input
    but chooses frames by time, so it stays correct when the frame spacing is not constant.
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


def sample_times(times: list, target_fps: float, max_frames: int) -> tuple[tuple, tuple]:
    """Sample on a fixed time grid, choosing each frame by its actual PTS.

    Grid points are 0, 1/target_fps, 2/target_fps ... seconds from the first frame. For each
    point the source frame with the nearest PTS is used (the earlier frame wins an exact tie),
    so constant- and variable-frame-rate clips take the same path and no choice depends on a
    reported frame rate. A grid point that would repeat the previous frame is dropped rather
    than decoded twice. Returns (source_indices, seconds_of_each_emitted_frame), measured from
    the first frame, so a label never claims a time the frame does not show.
    """
    if isinstance(target_fps, bool) or not isinstance(target_fps, (int, float)) or \
            not math.isfinite(target_fps) or target_fps <= 0:
        raise VideoError("video has no finite positive FPS")
    if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 1:
        raise VideoError("video frame limit must be a positive integer")
    if not times or any(isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) for t in times) or \
            any(b <= a for a, b in zip(times, times[1:])):
        raise VideoError("video frame timestamps must be finite and strictly increasing")
    start, span = times[0], times[-1] - times[0]
    grid = span * target_fps
    if not math.isfinite(grid) or grid > 36000:
        raise VideoLimitError("video sampling grid exceeds the supported duration/FPS ceiling")
    count = int(grid) + 1
    step = 1 / target_fps
    indices, seconds, chosen = [], [], 0
    for i in range(count):
        t = start + i * step
        while chosen + 1 < len(times) and abs(times[chosen + 1] - t) < abs(times[chosen] - t):
            chosen += 1
        if indices and indices[-1] == chosen:
            continue
        indices.append(chosen)
        if len(indices) > max_frames:
            raise VideoLimitError("video selected frames exceed the frame budget; lower FPS or raise max_frames")
        seconds.append(round(times[chosen] - start, 6))
    return tuple(indices), tuple(seconds)


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
    times: tuple[float, ...]
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
    def decoded_bytes(self):
        return len(self.indices) * self.width * self.height * 4

    @property
    def packet_bytes(self):
        return FRAME_HEADER.size + len(self.indices) * FRAME_TIME.size + self.rgb_bytes

    @property
    def spool_reservation(self):
        return self.packet_bytes + 64 * len(self.indices)

    @property
    def max_wire_bytes(self):
        groups = (len(self.indices) + 1) // 2
        return (MEDIA_HEADER.size + (self.rows + groups * 512) * 4 + groups * MEDIA_SPAN.size +
                self.rows * (12 + 2560 * 4))

    def charge(self, budget):
        if self.max_wire_bytes > budget.policy.max_embedding_bytes:
            raise VideoLimitError("video wire payload exceeds its embedding/transport byte budget")
        budget.charge(duration_s=self.duration_s, frames=len(self.indices), rgb_bytes=self.rgb_bytes,
                      decoded_bytes=self.decoded_bytes, tokens=self.rows, embedding_bytes=self.rows * 2560 * 4)


def probe_info(data: dict, policy: VideoPolicy) -> ClipInfo:
    """Validate untrusted ffprobe JSON and choose frames by PTS on a fixed time grid.

    Every frame's PTS and dimensions are checked, not only the stream's claims, and the
    sample grid is wall-clock time, so variable-frame-rate sources need no special case and no
    transcode. Audio is ignored; playlists and changing resolution are refused.
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
        declared_fps = None
        try:
            a, b = s["avg_frame_rate"].split("/")
            candidate = int(a) / int(b)
            if math.isfinite(candidate) and 0.1 <= candidate <= 240:
                declared_fps = candidate
        except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
            pass
        records = data["frames"]
        if not isinstance(records, list) or not 1 <= len(records) <= policy.max_source_frames:
            raise VideoError("video source frame count exceeds the configured limit")
        times, durations = [], []
        for frame in records:
            if frame.get("width") != w or frame.get("height") != h:
                raise VideoError("videos with changing resolution are not supported")
            t = float(frame["best_effort_timestamp_time"])
            if not math.isfinite(t):
                raise VideoError("video frame timestamp must be finite")
            times.append(t)
            try:
                d = float(frame.get("pkt_duration_time", "nan"))
            except (TypeError, ValueError, OverflowError):
                d = math.nan
            durations.append(d)
        if any(b <= a for a, b in zip(times, times[1:])):
            raise VideoError("video frame timestamps must be strictly increasing")
        span = times[-1] - times[0]
        intervals = [b - a for a, b in zip(times, times[1:])]
        median = statistics.median(intervals) if intervals else \
            (1 / declared_fps if declared_fps else 0.0)
        end = durations[-1] if math.isfinite(durations[-1]) and durations[-1] > 0 else median
        duration = span + end
        if not math.isfinite(duration) or duration <= 0 or duration > policy.max_duration_s:
            raise VideoError("video duration exceeds the configured limit")
        # Reported only; sampling never uses it. Fall back to the observed PTS spacing.
        fps = declared_fps or (1 / median if median > 0 else 0)
        rotation = 0
        for side in s.get("side_data_list", []):
            if "rotation" in side:
                r = float(side["rotation"])
                if not math.isfinite(r) or r % 90:
                    raise VideoError("video rotation must be a multiple of 90 degrees")
                rotation = int(r) % 360
        if rotation % 180:
            w, h = h, w
        indices, seconds = sample_times(times, policy.fps, policy.max_frames)
        rh, rw = resize_shape(h, w, len(indices), policy)
        info = ClipInfo(len(times), fps, w, h, duration, indices, seconds, rw, rh, rotation)
        for name, cost, ceiling in (("visual rows", info.rows, policy.max_tokens),
                                    ("RGB", info.rgb_bytes, policy.max_rgb_bytes),
                                    ("decoder-output", info.decoded_bytes, policy.max_decoded_bytes),
                                    ("wire", info.max_wire_bytes, policy.max_embedding_bytes)):
            if cost > ceiling:
                raise VideoLimitError(f"video {name} budget exceeded ({cost} > {ceiling})")
        return info
    except VideoError:
        raise
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        raise VideoError("video probe returned invalid or incomplete metadata") from None
