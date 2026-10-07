"""Thin resident-encoder adapter and byte-bounded video artifact cache.

No generation loop. The caller stages/decode sources outside its inference FIFO,
then calls encode under that FIFO and the existing vision process lock.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import threading

from .media import (HEADER, SPAN, MediaLimits, read_bundle, validate_qwen4, write_bundle)
from .video import VIDEO_PROFILE, VideoError, VideoPolicy
from .video_source import DiskQuota, OwnedMediaFile


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class VideoEncoder:
    def __init__(self, cfg, directory):
        self.directory, self.cfg = Path(directory), cfg
        self.enabled = isinstance(cfg.get("video"), dict) and cfg["video"].get("enabled") is True
        self.policy, self.reason, self.identity = None, "video is disabled (vision.video.enabled is not true)", None
        self.cache, self.cache_lock, self.cache_used = OrderedDict(), threading.Lock(), 0
        try:
            self.policy = VideoPolicy.from_config(cfg.get("video"), cfg.get("min_tokens", 0), cfg.get("max_tokens", 0))
        except VideoError as e:
            # Fail closed for video, not for the otherwise usable image encoder.
            self.reason = str(e)
        self.quota = DiskQuota(self.policy.max_disk_bytes) if self.policy is not None else None
        if self.policy is not None:
            self.reason = "video encoder capabilities have not been negotiated"

    def configure(self, caps, exchange, budget):
        if self.policy is None:
            return
        if " " in str(self.directory):
            self.reason = "video temporary directory has spaces; set TEMP/TMP to a space-free local directory"
            return
        self.reason = "the encoder supports images only (no SVE2 video capability)"
        if not caps.startswith("CAPS "):
            return
        facts = dict(field.split("=", 1) for field in caps.split()[1:] if "=" in field)
        wanted = {"media": "2", "profile": VIDEO_PROFILE, "width": "2560", "image_pad": "248056",
                  "video_pad": "248057", "vision_start": "248053", "vision_end": "248054"}
        if any(facts.get(k) != v for k, v in wanted.items()):
            self.reason = "the encoder/projector/tokenizer do not match the supported Qwen4 video profile"
            return
        ffmpeg, ffprobe = shutil.which(self.policy.ffmpeg), shutil.which(self.policy.ffprobe)
        if ffmpeg is None or ffprobe is None:
            self.reason = "video requires ffmpeg and ffprobe (images remain available)"
            return
        try:
            import PIL
        except ImportError:
            self.reason = "video RGB preprocessing requires Pillow (images remain available)"
            return
        p = self.policy = replace(self.policy, ffmpeg=str(Path(ffmpeg).resolve()), ffprobe=str(Path(ffprobe).resolve()))
        command = f"VSET {p.max_group_tokens} {p.max_frames} {p.max_tokens} {p.max_rgb_bytes} {p.max_embedding_bytes} {p.max_duration_s}"
        answer = exchange(command, budget)
        if answer != "VOK":
            self.reason = "video encoder configuration failed: " + answer[:500]
            return
        # Content, not filenames/mtime alone. Computed only for opt-in video.
        identities = {}
        for key in ("exe", "model", "mmproj"):
            path = self.cfg[key]
            found = shutil.which(path) if key == "exe" and not Path(path).is_file() else path
            identities[key] = file_hash(found)
        identities["ffmpeg"] = file_hash(p.ffmpeg)
        identities["ffprobe"] = file_hash(p.ffprobe)
        identities["pillow"] = PIL.__version__
        self.identity = hashlib.sha256(json.dumps({"policy": p.identity(), "files": identities}, sort_keys=True).encode()).hexdigest()
        self.reason = None

    @property
    def available(self):
        return self.policy is not None and self.reason is None and self.identity is not None

    def limits(self, *, tokens=1 << 20, position=(1 << 31) - 1, vocab=(1 << 31)):
        p = self.policy
        return MediaLimits(expected_width=2560, max_tokens=min(tokens, 1 << 20), max_rows=p.max_tokens,
                           max_bytes=p.max_embedding_bytes, max_position=position, vocab_size=vocab,
                           allowed_pad_ids=(248056, 248057))

    def key(self, source_hash):
        return hashlib.sha256((self.identity + source_hash).encode()).hexdigest()

    def evict_for(self, maximum):
        with self.cache_lock:
            while self.cache and self.quota.used + maximum > self.quota.limit:
                _, (owned, _) = self.cache.popitem(last=False)
                self.cache_used -= owned.reserved
                owned.close()

    @staticmethod
    def costs(bundle, info, budget, *, frames=True):
        if frames:
            info.charge(budget)
        rows = sum(len(span.positions) for span in bundle.spans)
        budget.charge(tokens=rows, embedding_bytes=rows * bundle.width * 4)

    @staticmethod
    def check_tokens(bundle, info, tokenizer):
        if tokenizer is None:
            return
        rows = info.resized_width // 32 * (info.resized_height // 32)
        text = []
        for i in range(0, len(info.indices), 2):
            a, b = info.indices[i], info.indices[min(i + 1, len(info.indices) - 1)]
            seconds = (a / info.source_fps + b / info.source_fps) / 2
            text.append(f"<{seconds:.1f} seconds><|vision_start|>" + "<|video_pad|>" * rows + "<|vision_end|>")
        if tuple(tokenizer.encode("".join(text), parse_special=True)) != bundle.tokens:
            raise VideoError("video encoder/tokenizer disagree on the processor's timestamp/control/pad tokens")

    def cached(self, source_hash, budget, tokenizer=None):
        key = self.key(source_hash)
        with self.cache_lock:
            entry = self.cache.get(key)
            if entry is None:
                return None
            owned, info = entry
            # Pinned until completely read into immutable payload storage; eviction
            # cannot remove a file under an active reader. No movable row pointers.
            with owned.path.open("rb") as source:
                bundle = read_bundle(source, self.limits(), qwen4=True)
            self.check_tokens(bundle, info, tokenizer)
            self.costs(bundle, info, budget)
            self.cache.move_to_end(key)
            return bundle

    def encode(self, packet, source_hash, info, budget, exchange, tokenizer=None):
        if not self.available:
            raise VideoError("video unavailable: " + (self.reason or "encoder is not ready"))
        budget.check()
        # A pair has <=256 wrapper tokens (checked by the native exporter). The
        # precise completed wire size is checked before cache publication.
        groups, rows = (len(info.indices) + 1) // 2, info.rows
        maximum = HEADER.size + (rows + groups * 512) * 4 + groups * SPAN.size + rows * (12 + 2560 * 4)
        if maximum > self.policy.max_embedding_bytes:
            raise VideoError("video wire payload exceeds its embedding/transport byte budget")
        self.evict_for(maximum)
        owned = OwnedMediaFile(self.directory, self.quota, maximum, ".sve2")
        try:
            command = f"ENCV {Path(packet).relative_to(self.directory).as_posix()} {owned.path.relative_to(self.directory).as_posix()}"
            answer = exchange(command, budget)
            if not answer.startswith("VOK "):
                raise VideoError("video encode failed: " + answer[:1000])
            with owned.path.open("rb") as source:
                bundle = read_bundle(source, self.limits(), qwen4=True)
            if (len(bundle.spans) != groups or sum(len(s.positions) for s in bundle.spans) != rows or
                    any(int(s.kind) != 2 for s in bundle.spans)):
                raise VideoError("video encoder did not preserve all temporal groups/rows")
            self.check_tokens(bundle, info, tokenizer)
            self.costs(bundle, info, budget, frames=False)  # decoder already charged frames/RGB/duration
            owned.finish(owned.path.stat().st_size)
            if owned.reserved <= self.policy.cache_bytes:
                with self.cache_lock:
                    # Another request can have prepared the same clip concurrently.
                    key = self.key(source_hash)
                    if key in self.cache:
                        previous = self.cache.pop(key)[0]
                        self.cache_used -= previous.reserved
                        previous.close()
                    while self.cache and self.cache_used + owned.reserved > self.policy.cache_bytes:
                        _, (old, _) = self.cache.popitem(last=False)
                        self.cache_used -= old.reserved
                        old.close()
                    self.cache[key] = (owned, info)
                    self.cache_used += owned.reserved
                    owned = None  # cache takes ownership only after full validation
            return bundle
        finally:
            if owned is not None:
                owned.close()

    def request_artifact(self, bundle, limits):
        validate_qwen4(bundle, limits)
        rows = sum(len(s.positions) for s in bundle.spans)
        size = HEADER.size + len(bundle.tokens) * 4 + len(bundle.spans) * SPAN.size + rows * (12 + 2560 * 4)
        self.evict_for(size)
        owned = OwnedMediaFile(self.directory, self.quota, size, ".request.sve2")
        try:
            with Path(str(owned.path) + ".partial").open("wb") as out:
                write_bundle(out, bundle, limits)
            Path(str(owned.path) + ".partial").replace(owned.path)
            return owned.finish(size)
        except BaseException:
            owned.close()
            raise

    def close(self):
        with self.cache_lock:
            for owned, _ in self.cache.values():
                owned.close()
            self.cache.clear()
            self.cache_used = 0
