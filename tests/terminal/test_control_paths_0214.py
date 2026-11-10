"""A run config that names files in a Control folder that is gone (the AppImage's /tmp/.mount_<random>) is pointed at this Control (QAGUI-LIN 0213)."""
import json
from pathlib import Path

from strata_control import control_paths


def _control(root: Path, name: str) -> Path:
    here = root / name / "usr" / "lib" / "Strata" / "control"
    (here / "data").mkdir(parents=True)
    (here / "data" / "expert-profile.bin").write_bytes(b"x")
    return here


def test_a_gone_mount_path_is_pointed_at_this_control(tmp_path):
    here = _control(tmp_path, ".mount_StrataNEW")
    gone = "/tmp/.mount_StratappjGco/usr/lib/Strata/control"
    cfg = {"cwd": gone, "args": ["--native", "/data/m/a.gguf", "--expert-profile", gone + "/data/expert-profile.bin", "--max-context", "32768"],
           "model_name": "Qwen"}
    out = control_paths.repoint(cfg, here)
    assert out["cwd"] == str(here)
    assert out["args"][3] == str(here / "data" / "expert-profile.bin")
    assert out["args"][:3] == cfg["args"][:3] and out["args"][4:] == cfg["args"][4:]
    assert cfg["cwd"] == gone, "the input is not changed"


def test_paths_that_exist_or_have_no_counterpart_stay(tmp_path):
    here = _control(tmp_path, "new")
    other = _control(tmp_path, "old")
    kept = str(other / "data" / "expert-profile.bin")
    assert control_paths.repoint_value(kept, here) == kept, "an existing file is left alone"
    missing = "/tmp/.mount_X/usr/lib/Strata/control/data/not-in-this-control.bin"
    assert control_paths.repoint_value(missing, here) == missing, "no counterpart: unchanged (the start reports it)"
    assert control_paths.repoint_value("/home/u/controller/x", here) == "/home/u/controller/x"
    assert control_paths.repoint_value(32768, here) == 32768


def test_the_update_test_start_uses_the_repointed_config():
    src = (Path(control_paths.__file__).parent / "updater.py").read_text(encoding="utf-8")
    at = src.index('cfg = json.loads(source.read_text(encoding="utf-8-sig"))')
    assert "control_paths.repoint(cfg)" in src[at:at + 400]
    json.dumps(control_paths.repoint({"cwd": "x"}))
