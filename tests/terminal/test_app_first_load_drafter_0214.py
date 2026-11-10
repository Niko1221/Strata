"""The app's setup, the app's first load and the Terminal's start pick the same drafter for Auto (lane APP-LIN 0.2.14 item 7).

Before: the app prepared every DFlash2-capable model for DFlash2 (the host's own rule), so on a card where Control's Auto means the MTP the setup skipped the MTP layer and the
first load ran undrafted. Now the app asks `strata drafting plan` at the install (no run config yet: --family/--model) and at the load (--config). This pins that both answers
equal the start's own decision (drafting.apply, what serve/server.py runs) on the card classes where they differ: a 12 GB CUDA card streaming IQ3_S (MTP) and a 24 GB one (DFlash2).
"""
import argparse
import json
from pathlib import Path

import pytest

from strata_cli import drafting_cmd
from strata_control import drafting


def _plan(capsys, **kw) -> dict:
    a = argparse.Namespace(drafting_command="plan", data_dir=None, config=None, family=None, model=None, mode=None, no_split=False,
                           engines_unloading=False, output="json")
    for k, v in kw.items():
        setattr(a, k, v)
    assert drafting_cmd.run(a) == 0
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


@pytest.mark.parametrize("vram_mib,expected", [(12288, "mtp"), (24576, "dflash2")])
def test_setup_first_load_and_start_agree(tmp_path, monkeypatch, capsys, vram_mib, expected):
    monkeypatch.setattr(drafting_cmd, "_hardware", lambda: ("cuda", ["sm_86"]))
    monkeypatch.setattr(drafting, "card_vram_mib", lambda *a, **k: float(vram_mib))
    monkeypatch.setattr(drafting_cmd, "_split_expected", lambda *a, **k: False)
    monkeypatch.setattr(drafting_cmd, "_installed_dflash2_configs", lambda a: [])
    data = tmp_path / "data"
    (data / "mtp" / "rt").mkdir(parents=True)
    # 1. the app's setup: no run config yet
    at_install = _plan(capsys, family="qwen", model="IQ3_S", mode="auto", data_dir=str(data))
    assert at_install["choice"] == expected
    # 2. the app's first load: the run config setup wrote (Auto: no "drafting" key)
    mtp = data / "mtp" / "rt"
    (mtp / "experts.bin").write_bytes(b"x" * 2048)
    cfg = {"model_name": "qwen3.8-flash-next-iq3_s", "backend": "cuda",
           "args": ["--pack", str(data / "packs" / "iq3_s"), "--native", "/m/IQ3_S-00001-of-00002.gguf", "--mtp", str(mtp), "--max-context", "131072"]}
    path = tmp_path / "strata-iq3_s.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    at_load = _plan(capsys, config=str(path))
    assert at_load["choice"] == expected
    # 3. the start (serve/server.py runs drafting.apply on the same config): the same choice
    want = drafting.choice_of("auto", model_name=cfg["model_name"], backend="cuda", archs=["sm_86"], args=cfg["args"], vram_mib=float(vram_mib))
    assert want == expected
