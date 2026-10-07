"""Private CPU-only media I/O worker. Launched by path with python -I.

Staged containers never reach a decoder as URLs. A dedicated process group/job
allows blocked reads and decoder descendants to be stopped without touching Strata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import sys
import threading
import urllib.parse
import urllib.request

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from serve.media_process import OwnedMediaProcess
from serve.video import (FRAME_HEADER, FRAME_TIME, VideoCancelled, VideoError, VideoLimitError,
                         VideoPolicy, VideoRequestBudget, probe_info)


FORMATS = "mov,matroska,webm,avi,gif"


class _HttpOnlyRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme not in ("http", "https"):
            raise VideoError("a video URL may redirect only to http(s)")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def stage(source, output, maximum, budget, download=False):
    h, count = hashlib.sha256(), 0
    if download:
        request = urllib.request.Request(source, headers={"User-Agent": "strata", "Accept-Encoding": "identity"})
        stream = urllib.request.build_opener(_HttpOnlyRedirect()).open(request, timeout=min(60, budget.policy.deadline_s))
        claimed = stream.headers.get("Content-Length", "").strip()
        if claimed.isdigit() and int(claimed) > maximum:
            stream.close()
            raise VideoLimitError("video source exceeds its byte budget")
    else:
        fd = os.open(source, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            os.close(fd)
            raise VideoError("video source must be a regular local file")
        if info.st_size > maximum:
            os.close(fd)
            raise VideoLimitError("video source exceeds its byte budget")
        stream = os.fdopen(fd, "rb")
    with stream, open(output, "wb") as out:
        while True:
            budget.check()
            data = stream.read(min(65536, maximum - count + 1))
            if not data:
                break
            count += len(data)
            if count > maximum:
                raise VideoLimitError("video source exceeds its byte budget")
            h.update(data)
            out.write(data)
    if count == 0:
        raise VideoError("video source is empty")
    return {"bytes": count, "sha256": h.hexdigest()}


def decoder_args(exe, source):
    return [exe, "-v", "error", "-threads", "1", "-protocol_whitelist", "file,pipe",
            "-format_whitelist", FORMATS, "-max_alloc", str(64 << 20), "-i", source]


def probe(source, budget):
    # All frame PTS/dimensions: reject variable rates/resolution instead of inventing a time base.
    args = decoder_args(budget.policy.ffprobe, source)
    args[1:1] = ["-select_streams", "v:0", "-show_frames", "-show_streams", "-of", "json", "-show_entries",
                 "frame=best_effort_timestamp_time,width,height:stream=width,height,avg_frame_rate:stream_side_data=rotation"]
    with OwnedMediaProcess(args, budget, max_stdout=8 << 20, inherit_group=True) as process:
        raw = process.read()
    try:
        return probe_info(json.loads(raw), budget.policy)
    except (UnicodeError, json.JSONDecodeError):
        raise VideoError("video probe returned invalid JSON") from None


def decode(source, output, budget, info=None):
    from PIL import Image
    info = info or probe(source, budget)
    p = budget.policy
    # Fail before allocating RGB. These products use validated positive dimensions/counts.
    if info.rgb_bytes > p.max_rgb_bytes or info.packet_bytes > p.max_disk_bytes:
        raise VideoError("video frame spool exceeds its RGB/disk budget")
    wanted = "+".join(f"eq(n\\,{i})" for i in info.indices)
    args = decoder_args(p.ffmpeg, source)
    # No implicit hardware decoder/device selection, console input, audio, subtitles or data streams.
    args[1:1] = ["-nostdin", "-hwaccel", "none"]
    args += ["-map", "0:v:0", "-an", "-sn", "-dn", "-vf", "select=" + wanted,
             "-fps_mode", "passthrough", "-frames:v", str(len(info.indices)), "-threads", "1",
             "-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1"]
    original_bytes = info.width * info.height * 4
    pending, written = bytearray(), 0
    with open(output, "wb") as out, OwnedMediaProcess(args, budget,
            max_stdout=original_bytes * len(info.indices), inherit_group=True) as process:
        out.write(FRAME_HEADER.pack(b"SVF1", 0, len(info.indices), info.resized_width, info.resized_height,
                                   info.duration_s, info.rgb_bytes))
        for chunk in process:
            pending.extend(chunk)
            while len(pending) >= original_bytes:
                budget.check()
                if written >= len(info.indices):
                    raise VideoError("video decoder returned extra frames")
                rgb = bytes(pending[:original_bytes])
                del pending[:original_bytes]
                im = Image.frombytes("RGBA", (info.width, info.height), rgb)
                if im.getextrema()[3] != (255, 255):
                    raise VideoError("transparent video frames are not supported; supply an opaque RGB clip")
                im = im.convert("RGB")
                if im.size != (info.resized_width, info.resized_height):
                    im = im.resize((info.resized_width, info.resized_height), Image.Resampling.BICUBIC)
                out.write(FRAME_TIME.pack(info.times[written]))
                out.write(im.tobytes())
                written += 1
        if pending or written != len(info.indices):
            raise VideoError("video decoder returned a short RGB frame stream")
    return {"bytes": info.packet_bytes,
            "info": {**info.__dict__, "indices": list(info.indices), "times": list(info.times)}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("copy", "download", "probe", "decode", "image"))
    ap.add_argument("source")
    ap.add_argument("output")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--maximum", type=int, default=0)
    ap.add_argument("--info")
    args = ap.parse_args()
    # Parent installs containment before allowing any source I/O or decoder launch.
    if sys.stdin.buffer.readline(4) != b"GO\n":
        raise VideoError("media worker was not released by its owner")
    p = VideoPolicy(**json.loads(args.policy))
    cancel = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: cancel.set())
    if os.name != "nt":
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (p.decoder_memory_bytes, p.decoder_memory_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (p.max_disk_bytes, p.max_disk_bytes))
        cpu = math.ceil(p.deadline_s) + 2
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    budget = VideoRequestBudget(p, cancel)
    if args.mode == "probe":
        result = {"info": probe(args.source, budget).__dict__}
    elif args.mode == "decode":
        from serve.video import ClipInfo
        info = json.loads(args.info) if args.info else None
        info = ClipInfo(**{**info, "indices": tuple(info["indices"]), "times": tuple(info["times"])}) if info else None
        result = decode(args.source, args.output, budget, info)
    elif args.mode == "image":
        from PIL import Image
        with Image.open(args.source) as image:
            w, h = image.size
        if w < 1 or h < 1 or max(w, h) > p.max_source_side or w * h > p.max_source_pixels:
            raise VideoError("mixed-request image dimensions exceed the video source pixel budget")
        result = {"width": w, "height": h}
    else:
        result = stage(args.source, args.output, args.maximum, budget, args.mode == "download")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        code = "cancelled" if isinstance(e, VideoCancelled) else "limit" if isinstance(e, (VideoLimitError, MemoryError)) else \
            "invalid_video" if isinstance(e, (VideoError, OSError, ValueError)) else "worker_error"
        print(json.dumps({"error": {"code": code, "message": str(e)[:1000]}}), flush=True)
        sys.exit(2)
