"""#4891: after `strata update` removes the control folder a run config was written for, the engine start dies before
it says anything (Windows: NotADirectoryError, WinError 267), and the launcher shows the vision reader's CUDA line
instead, so the person reads "no CUDA-capable device is detected". The config's control paths follow the control in use.

    python -m serve.test_control_paths_follow_control
"""
import shutil
import tempfile
import unittest
from pathlib import Path

import serve.server as server


class ControlPathsFollowControl(unittest.TestCase):
    def make(self, root: Path, *folders: str):
        for name in folders:
            d = root / "components" / "control" / name
            (d / "data").mkdir(parents=True)
            (d / "data" / "expert-profile.bin").write_bytes(b"x")
        return root / "components" / "control"

    def test_a_pruned_cwd_and_expert_profile_move_to_the_control_in_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            control = self.make(Path(tmp), "0.1.47", "0.1.55")
            shutil.rmtree(control / "0.1.47")     # what `strata update` does
            cfg = {"cwd": str(control / "0.1.47"),
                   "args": ["--expert-profile", str(control / "0.1.47" / "data" / "expert-profile.bin"), "--spec", "4"]}
            fixed = server.with_current_control_paths(cfg, control / "0.1.55")
            self.assertEqual(fixed["cwd"], str(control / "0.1.55"))
            self.assertEqual(fixed["args"][1], str(control / "0.1.55" / "data" / "expert-profile.bin"))
            self.assertEqual(fixed["args"][2:], ["--spec", "4"])

    def test_paths_already_in_the_control_in_use_stay(self):
        with tempfile.TemporaryDirectory() as tmp:
            control = self.make(Path(tmp), "0.1.55")
            cfg = {"cwd": str(control / "0.1.55"),
                   "args": ["--expert-profile", str(control / "0.1.55" / "data" / "expert-profile.bin")]}
            self.assertIs(server.with_current_control_paths(cfg, control / "0.1.55"), cfg)

    def test_a_cwd_outside_the_component_folders_is_the_persons_choice_and_stays(self):
        with tempfile.TemporaryDirectory() as tmp:
            control = self.make(Path(tmp), "0.1.55")
            mine = Path(tmp) / "mycontrol"
            mine.mkdir()
            cfg = {"cwd": str(mine)}
            self.assertIs(server.with_current_control_paths(cfg, control / "0.1.55"), cfg)

    def test_an_expert_profile_outside_the_component_folders_stays(self):
        with tempfile.TemporaryDirectory() as tmp:
            control = self.make(Path(tmp), "0.1.55")
            mine = Path(tmp) / "data"
            mine.mkdir()
            profile = mine / "expert-profile.bin"
            profile.write_bytes(b"x")
            cfg = {"args": ["--expert-profile", str(profile)]}
            self.assertIs(server.with_current_control_paths(cfg, control / "0.1.55"), cfg)

    def test_a_pruned_expert_profile_not_shipped_by_the_control_in_use_stays(self):
        with tempfile.TemporaryDirectory() as tmp:
            control = self.make(Path(tmp), "0.1.47", "0.1.55")
            (control / "0.1.47" / "data" / "my-profile.bin").write_bytes(b"x")   # only the older control had it
            shutil.rmtree(control / "0.1.47")
            cfg = {"args": ["--expert-profile", str(control / "0.1.47" / "data" / "my-profile.bin")]}
            self.assertIs(server.with_current_control_paths(cfg, control / "0.1.55"), cfg)


if __name__ == "__main__":
    unittest.main()
