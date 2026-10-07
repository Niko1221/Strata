"""Host-only video policy/sampling tests; no decoder or model."""
from dataclasses import replace
import copy
import threading
import time
import unittest

from serve.video import (VideoCancelled, VideoError, VideoLimitError, VideoPolicy, VideoRequestBudget,
                         probe_info, resize_shape, sample_indices, sample_times)


def policy(**options):
    return replace(VideoPolicy(), **options)


def probe_data(frames=5, fps=2):
    return {"streams": [{"width": 128, "height": 64, "avg_frame_rate": f"{fps}/1"}],
            "frames": [{"width": 128, "height": 64, "best_effort_timestamp_time": str(i / fps)}
                       for i in range(frames)]}


class VideoPolicyTests(unittest.TestCase):
    def test_opt_in_and_validation(self):
        for cfg in (None, False, {}, {"enabled": False}):
            self.assertIsNone(VideoPolicy.from_config(cfg))
        self.assertEqual(VideoPolicy.from_config({"enabled": True}), VideoPolicy())
        for cfg in (True, {"enabled": "yes"}, {"enabled": True, "typo": 1},
                    {"enabled": True, "fps": float("nan")}, {"enabled": True, "max_frames": 129},
                    {"enabled": True, "max_tokens": False}, {"enabled": True, "max_disk_bytes": 1}):
            with self.subTest(cfg=cfg), self.assertRaises(VideoError):
                VideoPolicy.from_config(cfg)
        with self.assertRaises(VideoError):
            VideoPolicy.from_config({"enabled": True, "max_group_tokens": 301}, 8, 300)

    def test_reference_sampling_and_ties_to_even(self):
        import numpy as np
        for frames in (1, 2, 3, 5, 11, 100, 300, 8192):
            for fps in (2., 24., 30., 240.):
                n = min(frames, max(4, min(768, int(frames / fps * 2))))
                expected = tuple(np.linspace(0, frames - 1, n).round().astype(int).tolist())
                self.assertEqual(sample_indices(frames, fps, 2), expected)
        self.assertEqual(sample_indices(5, 2, 2), (0,1,2,3,4))
        with self.assertRaises(VideoError):
            sample_indices(True, 30, 2)

    def test_resize_alignment_budgets_and_one_frame_extension(self):
        for h, w, frames in ((64,128,5), (720,1280,20), (1,2,1), (1080,1920,120)):
            rh, rw = resize_shape(h, w, frames, policy())
            self.assertEqual((rh % 32, rw % 32), (0,0))
            self.assertTrue(8 <= rh // 32 * (rw // 32) <= 256)
        self.assertEqual(resize_shape(64,128,1,policy()), resize_shape(64,128,2,policy()))
        with self.assertRaises(VideoError):
            resize_shape(1,1000,5,policy())

    def test_all_pts_and_rotation_not_container_duration(self):
        raw = probe_data()
        raw["streams"][0]["duration"] = "999999"
        info = probe_info(raw, policy())
        self.assertEqual((info.duration_s, info.indices, info.rows), (2.5, (0,1,2,3,4), 24))
        self.assertEqual(info.times, (0.0, 0.5, 1.0, 1.5, 2.0))
        raw["streams"][0]["side_data_list"] = [{"rotation": 90}]
        info = probe_info(raw,policy())
        self.assertEqual((info.width,info.height,info.rotation), (64,128,90))
        # A variable frame spacing is sampled by PTS, not rejected, and a missing
        # reported rate does not matter because sampling never uses it.
        vfr = copy.deepcopy(probe_data())
        vfr["frames"][3]["best_effort_timestamp_time"] = "1.6"
        info = probe_info(vfr, policy())
        self.assertEqual((info.indices, info.times), ((0, 1, 2, 3, 4), (0.0, 0.5, 1.0, 1.6, 2.0)))
        vfr["streams"][0]["avg_frame_rate"] = "0/0"
        self.assertEqual(probe_info(vfr, policy()).indices, (0, 1, 2, 3, 4))
        for change in ("resolution", "nan", "rotation", "nonmonotonic", "equal"):
            bad = copy.deepcopy(probe_data())
            if change == "resolution": bad["frames"][3]["width"] = 64
            if change == "nan": bad["frames"][3]["best_effort_timestamp_time"] = "nan"
            if change == "rotation": bad["streams"][0]["side_data_list"] = [{"rotation": 45}]
            if change == "nonmonotonic": bad["frames"][3]["best_effort_timestamp_time"] = "0.4"
            if change == "equal": bad["frames"][3]["best_effort_timestamp_time"] = "1.0"
            with self.subTest(change=change), self.assertRaises(VideoError):
                probe_info(bad,policy())

    def test_sample_times_picks_the_frame_nearest_each_grid_point(self):
        # An irregular spacing is read as its real times: a 0.65 s clip has two 2 FPS grid
        # points, and the one at 0.5 s takes the frame nearest 0.5 s (0.650 is nearer than 0.319).
        pts = [0.000, 0.034, 0.068, 0.101, 0.285, 0.319, 0.650]
        self.assertEqual(sample_times(pts, 2.0, 128), ((0, 6), (0.0, 0.65)))
        # A long irregular clip: every 0.5 s point takes the nearest frame, and a point that
        # would repeat the previous frame is dropped rather than decoded twice.
        # A long irregular clip on the fixed 2 FPS grid: each point takes the nearest frame, and
        # points inside the 2.3 s -> 5.0 s gap collapse to one frame instead of repeating it.
        pts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 1.6, 2.0, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3,
                   5.0, 5.05, 5.1, 5.15, 5.2, 5.25]
        self.assertEqual(sample_times(pts, 2.0, 128), ((0, 5, 6, 7, 13, 14), (0.0, 0.45, 1.6, 2.0, 2.3, 5.0)))
        # Constant-rate input keeps the pinned count and lands on every other frame.
        self.assertEqual(sample_times([i / 30 for i in range(300)], 2.0, 128)[0],
                         (0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285))
        # Fewer frames than grid points spreads the grid instead of repeating a frame.
        self.assertEqual(sample_times([0.0, 1.0, 2.0], 2.0, 128), ((0, 1, 2), (0.0, 1.0, 2.0)))
        self.assertEqual(sample_times([0.0], 2.0, 128), ((0,), (0.0,)))
        with self.assertRaises(VideoError):
            sample_times([i / 2 for i in range(300)], 2.0, 128)   # 300 points over the 128-frame cap
        with self.assertRaises(VideoError):
            sample_times([0.0, 0.5, 0.5], 2.0, 128)
        with self.assertRaises(VideoError):
            sample_times([0.0, 0.5], 0, 128)

    def test_cumulative_budget_repeat_costs_cancel_and_deadline(self):
        b = VideoRequestBudget(policy(max_frames=6, max_duration_s=4))
        info = probe_info(probe_data(), b.policy)
        info.charge(b)
        with self.assertRaises(VideoLimitError):
            info.charge(b)
        cancel=threading.Event(); cancel.set()
        with self.assertRaises(VideoCancelled):
            VideoRequestBudget(policy(),cancel).check()
        b=VideoRequestBudget(policy()); b.deadline=time.monotonic()-1
        with self.assertRaises(VideoLimitError):
            b.check()



if __name__ == "__main__":
    unittest.main()
