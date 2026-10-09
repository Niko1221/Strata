"""Source/spool tests; STRATA_FFMPEG/STRATA_FFPROBE enable actual CPU decode."""
from dataclasses import replace
import base64
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from serve.video import FRAME_HEADER, FRAME_TIME, VideoError, VideoLimitError, VideoRequestBudget
from serve.media_worker import select_filter
from serve.video_source import DiskQuota, OwnedMediaFile, decode_video, local_video_path, stage_video
from serve.test_video_policy import policy


class SourceTests(unittest.TestCase):
    def test_network_authority_device_and_encoded_paths_rejected(self):
        for path in (r"\\host\share\clip.mp4", r"\\?\UNC\host\clip.mp4", "//host/share/clip",
                     "file://evil/clip", "file:////evil/clip", "file:///%5C%5Cevil/clip", "file:///clip%0A.mp4",
                     "rtsp://host/live", "ftp://host/movie", "pipe:0"):
            with self.subTest(path=path), self.assertRaises(VideoError):
                local_video_path(path)
        self.assertTrue(local_video_path("file:///tmp/local.mp4").endswith("/tmp/local.mp4"))

    def test_quota_partial_exact_ownership_and_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            quota=DiskQuota(20)
            sentinel=Path(d)/"keep";sentinel.write_bytes(b"keep")
            with OwnedMediaFile(d,quota,15,".test") as f:
                Path(str(f.path)+".partial").write_bytes(b"partial")
                Path(str(f.path)+".select").write_text("select=1")
                with self.assertRaises(VideoLimitError):
                    OwnedMediaFile(d,quota,10,".test")
            self.assertEqual(quota.used,0)
            self.assertEqual(list(Path(d).iterdir()),[sentinel])

    def test_large_select_expression_has_bounded_depth(self):
        expression = select_filter(range(4096))
        self.assertEqual(expression.count("eq(n\\,"), 4096)
        depth = peak = 0
        for ch in expression:
            depth += (ch == "(") - (ch == ")")
            peak = max(peak, depth)
        self.assertEqual(depth, 0)
        self.assertLessEqual(peak, 13)

    def test_regular_local_and_base64_hashes_cumulative_source_limit(self):
        with tempfile.TemporaryDirectory() as d:
            original=Path(d)/"original";original.write_bytes(b"opaque source bytes")
            quota=DiskQuota(1000);b=VideoRequestBudget(policy(max_source_bytes=40))
            with stage_video(str(original),d,quota,b)[0] as f:
                self.assertEqual(f.path.read_bytes(),original.read_bytes())
            src="data:video/mp4;base64,"+base64.b64encode(original.read_bytes()).decode()
            file,digest=stage_video(src,d,quota,b)
            with file:
                self.assertEqual(digest,hashlib.sha256(original.read_bytes()).hexdigest())
            with self.assertRaises(VideoLimitError):
                stage_video(src,d,quota,b)
            self.assertEqual(quota.used,0)
            self.assertEqual(list(Path(d).iterdir()),[original])

    def test_malformed_base64_and_special_files_leave_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            quota=DiskQuota(10000)
            for src in ("data:image/png;base64,eA==", "data:video/mp4;base64,!abc", "data:video/mp4;base64,eA=",d):
                with self.subTest(src=src),self.assertRaises(VideoError):
                    stage_video(src,d,quota,VideoRequestBudget(policy(max_source_bytes=100)))
                self.assertEqual(quota.used,0)
            if os.name != "nt":
                fifo=Path(d)/"fifo";os.mkfifo(fifo)
                with self.assertRaises(VideoError):
                    stage_video(str(fifo),d,quota,VideoRequestBudget(policy(max_source_bytes=100)))
                fifo.unlink()
            self.assertEqual(list(Path(d).iterdir()),[])


@unittest.skipUnless(os.environ.get("STRATA_FFMPEG") and os.environ.get("STRATA_FFPROBE"),
                     "real CPU video decoder requires explicit STRATA_FFMPEG/STRATA_FFPROBE")
class DecoderTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.p=policy(ffmpeg=os.environ["STRATA_FFMPEG"],ffprobe=os.environ["STRATA_FFPROBE"])
        self.q=DiskQuota(self.p.max_disk_bytes)
        self.source=self.root/"five.mkv"
        subprocess.run([self.p.ffmpeg,"-v","error","-f","lavfi","-i","testsrc2=size=128x64:rate=2",
                        "-frames:v","5","-c:v","ffv1","-y",str(self.source)],check=True,timeout=15,
                       env={**os.environ,"CUDA_VISIBLE_DEVICES":"","HIP_VISIBLE_DEVICES":""})

    def test_real_cfr_sampling_spool_timestamps_and_cleanup(self):
        b=VideoRequestBudget(self.p)
        with stage_video(str(self.source),self.root,self.q,b)[0] as source:
            packet,info=decode_video(source.path,self.root,self.q,b)
            with packet:
                raw=packet.path.read_bytes()
                self.assertEqual(len(raw),info.packet_bytes)
                magic,flags,n,w,h,duration,rgb=FRAME_HEADER.unpack_from(raw)
                self.assertEqual((magic,flags,n,w,h,duration,rgb),(b"SVF1",0,5,128,64,2.5,122880))
                per=w*h*3;offset=FRAME_HEADER.size
                for index, when in zip(info.indices, info.times):
                    self.assertEqual(FRAME_TIME.unpack_from(raw,offset)[0],when)
                    offset+=FRAME_TIME.size+per
                self.assertEqual(offset,len(raw))
                self.assertEqual((b.frames,b.duration_s,b.rgb_bytes),(5,2.5,122880))
        self.assertEqual(self.q.used,0)
        self.assertEqual(list(self.root.iterdir()),[self.source])

    def test_vfr_samples_by_pts_and_labels_with_real_frame_times(self):
        """Frames come from the time grid, not from index/reported-FPS arithmetic."""
        source = self.root / "gap.mkv"
        # 12 frames: 0.25 s apart up to 2.0 s, then a 1 s gap to 3.0 s. Reported rate stays 4/1,
        # so index sampling would pick frames 4,5,7 where the grid picks 4,6,8 and drops 2.5 s.
        subprocess.run([self.p.ffmpeg, "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=128x64:rate=4",
                        "-frames:v", "12",
                        "-vf", "setpts='if(lt(N,9),N*0.25,3.0+(N-9)*0.25)/TB'", "-fps_mode", "vfr",
                        "-c:v", "ffv1", "-y", str(source)], check=True, timeout=15,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""})
        b = VideoRequestBudget(self.p)
        with stage_video(str(source), self.root, self.q, b)[0] as staged:
            packet, info = decode_video(staged.path, self.root, self.q, b)
            with packet:
                self.assertEqual(info.indices, (0, 2, 4, 6, 8, 9, 11))
                self.assertEqual(info.times, (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 3.5))
                raw = packet.path.read_bytes()
                per = info.resized_width * info.resized_height * 3
                for slot, (index, when) in enumerate(zip(info.indices, info.times)):
                    offset = FRAME_HEADER.size + slot * (FRAME_TIME.size + per)
                    self.assertEqual(FRAME_TIME.unpack_from(raw, offset)[0], when)
                    offset += FRAME_TIME.size
                    self.assertEqual(self.source_frame(source, index), raw[offset:offset + per])
        self.assertEqual(self.q.used, 0)

    def source_frame(self, path, index):
        """One source frame through the same RGBA-to-RGB path the decoder worker uses."""
        from PIL import Image
        rgba = subprocess.run([self.p.ffmpeg, "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{index})",
                               "-fps_mode", "passthrough", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgba",
                               "-"], capture_output=True, check=True, timeout=15,
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""}).stdout
        return Image.frombytes("RGBA", (128, 64), rgba).convert("RGB").tobytes()

    def test_corrupt_clip_is_error_not_empty_success(self):
        self.source.write_bytes(b"not a video")
        b=VideoRequestBudget(self.p)
        with stage_video(str(self.source),self.root,self.q,b)[0] as source:
            with self.assertRaises(VideoError):
                decode_video(source.path,self.root,self.q,b)
        self.assertEqual(self.q.used,0)
        self.assertEqual(list(self.root.iterdir()),[self.source])

    def test_sample_budget_rejects_without_resampling(self):
        b=VideoRequestBudget(replace(self.p,max_frames=4))
        with stage_video(str(self.source),self.root,self.q,b)[0] as source:
            with self.assertRaisesRegex(VideoError,"budget|limit"):
                decode_video(source.path,self.root,self.q,b)
        self.assertEqual(self.q.used,0)


if __name__ == "__main__":
    unittest.main()
