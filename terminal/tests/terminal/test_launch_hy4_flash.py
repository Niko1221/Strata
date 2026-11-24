"""hy4-flash local launch, without a GPU, downloads or an inference process."""
import argparse
import json
from unittest import mock

import pytest

from strata_cli.launch import settings as S
from strata_cli.launch.catalog import Computer
from strata_cli.launch.cmd import _cfg_context, add_parsers, settings_from_args
from strata_cli.launch.runner import Runner, apply_choices, read_config, files_present
from strata_control import hy4_flash, models
from tests.terminal.launch_support import Home, catalog
from tests.terminal.test_models import gguf_fixture


def computer():
    return Computer(ram_gb=128, engine_vendor="nvidia", gpus=[
        {"index": 0, "vendor": "nvidia", "usable": True, "vram_gb": 96},
        {"index": 1, "vendor": "nvidia", "usable": True, "vram_gb": 96}])


def test_single_gpu_and_slot_defaults_do_not_use_all_cards():
    s = S.LaunchSettings(family="hy4-flash", model="Q4_K_M")
    hy4_flash.defaults(s, computer())
    assert s.gpus == [0] and s.parallel == 1 and s.context == 4096
    assert s.vision == s.mtp == s.drafting == s.split_engine == "off"
    assert 0 < s.expert_vram < 96


@pytest.mark.parametrize("choices", [
    {"parallel": 2}, {"vision": "cpu"}, {"gpus": [0, 1]}, {"gpus": [8]},
    {"draft_tokens": 0}, {"draft_tokens": 8}, {"mtp": "on", "drafting": "off"},
    {"drafting": "dflash2"}, {"kv": "q4_0"}, {"kv": "k8v4"}, {"tune": "quick"},
    {"expert_vram": float("nan")}, {"expert_vram": -1}, {"split_engine": "on"},
    {"extra": {"--batch": "4"}},
])
def test_unsupported_choices_fail_before_start(choices):
    with pytest.raises(ValueError):
        hy4_flash.defaults(S.LaunchSettings(family="hy4-flash", **choices), computer())


def test_vision_requires_a_compiled_encoder_and_shard(tmp_path, monkeypatch):
    exe = tmp_path / "strata"
    shard = tmp_path / "mmproj-Hy4-flash-bf16.gguf"
    monkeypatch.setenv("STRATA_HY4_FLASH_VISION_SHARD", str(shard))
    exe.write_bytes(b"HY4_FLASH_VISION encoded pictures")
    exe.chmod(0o755)
    def settings(vision):
        return S.LaunchSettings(family="hy4-flash", engine_path=str(exe), vision=vision)
    with pytest.raises(ValueError, match="engine built"):
        hy4_flash.defaults(settings("on"), computer())
    exe.write_bytes(b"hy4-flash requires a hy_v4_flash mmproj GGUF")
    with pytest.raises(ValueError, match="Missing hy4-flash mmproj GGUF"):
        hy4_flash.defaults(settings("on"), computer())
    shard.touch()
    picture, text = settings("on"), settings(None)
    for s in (picture, text):
        hy4_flash.defaults(s, computer())
    assert picture.vision == "on" and text.vision == "off"
    assert text.expert_vram - picture.expert_vram == pytest.approx(2)


def test_mmproj_discovery_and_override(tmp_path, monkeypatch):
    monkeypatch.delenv("STRATA_HY4_FLASH_VISION_SHARD", raising=False)
    (tmp_path / "language.gguf").touch()
    (tmp_path / "hy4-flash-Flash-Vision.strata").touch()
    mmproj = tmp_path / "mmproj-Hy4-flash-bf16.gguf"
    s = S.LaunchSettings(family="hy4-flash", model_path=str(tmp_path))
    with pytest.raises(ValueError, match="Missing hy4-flash mmproj GGUF"):
        hy4_flash.vision_shard(s)
    mmproj.touch()
    assert hy4_flash.vision_shard(s) == mmproj
    (tmp_path / "mmproj-other.gguf").touch()
    with pytest.raises(ValueError, match="Multiple mmproj"):
        hy4_flash.vision_shard(s)
    monkeypatch.setenv("STRATA_HY4_FLASH_VISION_SHARD", str(mmproj))
    assert hy4_flash.vision_shard(s) == mmproj
    monkeypatch.setenv("STRATA_HY4_FLASH_VISION_SHARD", str(tmp_path / "hy4-flash-Flash-Vision.strata"))
    with pytest.raises(ValueError, match="require a mmproj GGUF"):
        hy4_flash.vision_shard(s)


def test_kv_format_follows_the_engine_and_the_user(tmp_path):
    old = tmp_path / "old"
    old.write_bytes(b"\0HY4_FLASH_GPU_SAMPLE enabled\0")
    new = tmp_path / "new"
    new.write_bytes(b"\0HY4_FLASH_KV_Q8 enabled\0")
    for exe in (old, new):
        exe.chmod(0o700)
    budgets = {}
    for name, exe, kv, want in (("old", old, None, "fp16"), ("new", new, None, "int8"),
                                ("new-fp16", new, "fp16", "fp16")):
        s = S.LaunchSettings(family="hy4-flash", engine_path=str(exe), kv=kv, context=196608)
        hy4_flash.defaults(s, computer())
        assert s.kv == want, name
        budgets[name] = s.expert_vram
    # The int8 latent's rows (672 B a token a block against 1152) go to the expert pool.
    saved = 196608 * hy4_flash.KV_SAVED_PER_TOKEN["int8"] / 2**30
    assert abs(budgets["new"] - budgets["new-fp16"] - saved) < 0.02
    assert budgets["old"] == budgets["new-fp16"]
    s = S.LaunchSettings(family="hy4-flash", engine_path=str(new), extra={"STRATA_HY4_FLASH_KV": "f16"})
    hy4_flash.defaults(s, computer())
    assert s.kv == "fp16"
    with pytest.raises(ValueError, match="does not support Q8_0"):
        hy4_flash.defaults(S.LaunchSettings(family="hy4-flash", engine_path=str(old), kv="int8"), computer())


def test_q8_pool_savings_with_192k_mtp_and_prefill_loans(tmp_path):
    exe = tmp_path / "build-4090" / "strata"
    exe.parent.mkdir()
    exe.write_bytes(b"HY4_FLASH_KV_Q8 enabled\0prefill borrow: verify window drain failed\0STRATA_HY4_FLASH_PREFILL_LM_R")
    exe.chmod(0o755)
    machine = computer()
    machine.gpus[0].update(name="NVIDIA GeForce RTX 4090", vram_gb=48)
    for kv, expected_kv, expected_pool in [(None, "int8", 23.87), ("fp16", "fp16", 20.09)]:
        s = S.LaunchSettings(family="hy4-flash", model_path="model.gguf", gpus=[0], context=196608, kv=kv)
        with mock.patch.object(hy4_flash, "available_vram_gib", return_value=48508/1024), \
             mock.patch.object(hy4_flash, "mtp_model_layout", return_value=(True, 512)), \
             mock.patch.object(hy4_flash, "model_file", return_value=tmp_path), \
             mock.patch.object(hy4_flash, "mtp_weight_bytes", return_value=3980025984), \
             mock.patch.dict(hy4_flash.os.environ, {"STRATA_BIN": ""}):
            hy4_flash.defaults(s, machine, src=tmp_path)
            cfg = {"backend": "cuda", "exe": str(exe), "env": hy4_flash.serving_environment(True),
                   "mtp_resident_bytes": 3980025984}
            hy4_flash.apply_runtime_defaults(cfg, s, machine)
        assert (s.kv, s.mtp, s.draft_tokens, s.expert_vram) == (expected_kv, "on", 3, expected_pool)
        assert cfg["env"]["STRATA_HY4_FLASH_PREFILL_BORROW"] == "1"
        assert cfg["env"]["STRATA_HY4_FLASH_PREFILL_LM_R"] == "vram"


def test_import_uses_architecture_and_expert_format(tmp_path):
    path = tmp_path / "renamed.gguf"
    path.write_bytes(gguf_fixture(architecture="hy_v4_flash", expert_type=12))
    plan = models.plan_import(str(path))
    assert not plan["issues"]
    assert plan["candidate_selection"]["family"] == "hy4-flash"
    path.write_bytes(gguf_fixture(architecture="hy_v4_flash", expert_type=42))
    assert models.plan_import(str(path))["issues"]
    path.write_bytes(gguf_fixture(architecture="unknown", expert_type=12))
    assert "unsupported_or_missing_architecture" in models.plan_import(str(path))["issues"]


def test_local_runner_writes_real_config_and_plan_is_read_only():
    with Home() as home:
        cat = catalog()
        assert cat.resolve("q4_k_m", "hy4-flash") == ("hy4-flash", "Q4_K_M")
        model = home.base / "model.gguf"
        model.write_bytes(gguf_fixture(architecture="hy_v4_flash", expert_type=12))
        pack = home.base / "pack"
        for name in hy4_flash.PACK_FILES:
            p = pack / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("fixture")
        exe = home.base / "strata"
        exe.write_text("#!/bin/sh\nexit 0\n")
        exe.chmod(0o700)
        s = S.LaunchSettings(name="hy4-flash-test", model_path=str(model), engine_path=str(exe),
                             pack_path=str(pack), context=8192, expert_vram=24, resident_experts=True)
        runner = Runner(s, catalog=cat)
        with mock.patch.object(cat, "computer", return_value=computer()):
            before = set(home.base.rglob("*"))
            plan = runner.plan()
            assert set(home.base.rglob("*")) == before
            assert plan["download"]["gb"] == 0 and plan["download"]["engine_usable"]
            assert s.family == "hy4-flash" and s.model_path == str(model)
            # No subprocess should be started for a local engine and a complete pack.
            with mock.patch.object(runner, "_spawn", side_effect=AssertionError("unexpected process")):
                runner.ensure_engine()
                base = runner.ensure_model()
            cfg_path = runner.build_config(base)
            runner.ensure_profile(cfg_path)
        cfg = read_config(cfg_path)
        assert cfg["exe"] == str(exe) and cfg["arch"] == "hy4-flash"
        assert "--resident-experts" in cfg["args"] and "--expert-host" not in cfg["args"]
        assert cfg["gpu"] == 0 and cfg["parallel"] == 1
        assert _cfg_context(cfg) == 8192
        assert all(flag not in cfg["args"] for flag in ("--batch", "--max-context", "--prompt-cache-spread"))
        changed = apply_choices(cfg, S.LaunchSettings(context=4096, vision="off"))
        assert _cfg_context(changed) == 4096
        profile = json.loads((home.base / "config/profiles/launch-hy4-flash-test.json").read_text())
        assert profile["source_config"] == str(cfg_path)
        assert S.LaunchSettings.from_dict(s.to_dict()).engine_path == str(exe)
        assert "--engine-path" in S.flags_text(s)


def test_native_sq_and_local_gguf_route_to_their_own_engines():
    with Home() as home:
        cat = catalog()
        native = home.base / "Hy4-Flash-SQ-64.strata"
        native.touch()
        gguf = home.base / "model.gguf"
        gguf.write_bytes(gguf_fixture(architecture="hy_v4_flash", expert_type=12))
        for path, model, shape in [(native, "SQ-64", "hy4"), (gguf, "Q4_K_M", None)]:
            s = S.LaunchSettings(model_path=str(path))
            runner = Runner(s, catalog=cat)
            runner.check_own_files()
            assert (s.family, s.model) == ("hy4-flash", model)
            assert runner.model_shape() == shape
            assert runner.is_hy4() == (path == native)
        assert Runner(S.LaunchSettings(family="hy4-flash"), catalog=cat).model_shape() == "hy4"


def test_missing_engine_does_not_fall_back_to_download(tmp_path):
    with pytest.raises(ValueError, match="local engine build"):
        hy4_flash.engine_path(S.LaunchSettings(engine_path=str(tmp_path / "missing")), tmp_path)


def test_shard_directory_and_second_shard_resolve_to_first(tmp_path):
    first = tmp_path / "model-00001-of-00002.gguf"
    second = tmp_path / "model-00002-of-00002.gguf"
    first.touch()
    second.touch()
    assert hy4_flash.model_file(str(tmp_path)) == first
    assert hy4_flash.model_file(str(second)) == first
    second.unlink()
    with pytest.raises(ValueError, match="shard is missing"):
        hy4_flash.model_file(str(first))


def test_dfly_selects_external_worker_and_budgets_its_weights(tmp_path):
    draft = tmp_path / "draft.gguf"
    draft.write_bytes(gguf_fixture(architecture="hy_v4_next_dfly", expert_type=8))
    exe = tmp_path / "strata"
    exe.write_bytes(b"--draft dfly requires F16 or Q8_0 KV / HY4_FLASH_KV_Q8 enabled / HY4_FLASH_WINDOW_SUPPLY initialized / HY4_FLASH_WINDOW_CPU enabled")
    exe.chmod(0o755)
    s = S.LaunchSettings(family="hy4-flash", engine_path=str(exe), dfly_model=str(draft))
    plain = S.LaunchSettings(family="hy4-flash", engine_path=str(exe), mtp="off")
    hy4_flash.defaults(plain, computer())
    hy4_flash.defaults(s, computer())
    assert (s.mtp, s.drafting, s.draft_tokens, s.kv) == ("off", "dfly", 3, "int8")
    assert abs(plain.expert_vram - s.expert_vram - hy4_flash.dfly_reserve_gib(s)) < 0.01
    with mock.patch.object(hy4_flash, "model_file", return_value=draft), mock.patch.object(hy4_flash, "missing_pack", return_value=[]):
        cfg = hy4_flash.config(s, tmp_path)
    hy4_flash.apply_runtime_defaults(cfg, s, computer())
    assert cfg["args"][-6:] == ["--draft", "dfly", "--draft-model", str(draft), "--draft-max", "3"]
    assert cfg["env"]["STRATA_HY4_FLASH_WINDOW"] == "1" and cfg["env"]["STRATA_HY4_FLASH_MTP_GRAPH"] == "0"
    assert S.LaunchSettings.from_dict(s.to_dict()).dfly_model == str(draft)
    s.draft_tokens = 5
    with pytest.raises(ValueError, match="Rebuild"):
        hy4_flash.defaults(s, computer())
    wide = tmp_path / "wide-strata"
    wide.write_bytes(exe.read_bytes() + b" dfly: --draft-max exceeds the GGUF block size minus the anchor")
    wide.chmod(0o755)
    s.engine_path = str(wide)
    hy4_flash.defaults(s, computer())
    s.draft_tokens = 6
    with pytest.raises(ValueError, match="1..5"):
        hy4_flash.defaults(s, computer())


def test_explicit_dfly_needs_weights_and_can_switch_a_saved_preset():
    parser = argparse.ArgumentParser()
    add_parsers(parser.add_subparsers())
    args = parser.parse_args(["launch", "--family", "hy4-flash", "--drafting", "dfly"])
    settings, _ = settings_from_args(args)
    with pytest.raises(ValueError, match="requires --dfly-model"):
        hy4_flash.defaults(settings, computer())

    for mode in ("dfly", "mtp"):
        saved = S.LaunchSettings(family="hy4-flash", drafting="dfly", mtp="off",
                                 dfly_model="draft.gguf", draft_tokens=4)
        args = parser.parse_args(["launch", "--preset", "saved", "--drafting", mode])
        with mock.patch("strata_cli.launch.store.load", return_value=saved):
            settings, _ = settings_from_args(args)
        if mode == "dfly":
            assert (settings.dfly_model, settings.draft_tokens) == ("draft.gguf", 4)
            assert "--drafting dfly" in S.flags_text(settings)
        else:
            assert (settings.dfly_model, settings.mtp, settings.draft_tokens) == (None, None, None)
        assert settings.drafting == mode


def test_mtp_residency_is_subtracted_from_automatic_expert_budget(tmp_path):
    model = tmp_path / "model.gguf"
    model.touch()
    plain = S.LaunchSettings(family="hy4-flash", model_path=str(model), mtp="off")
    draft = S.LaunchSettings(family="hy4-flash", model_path=str(model), mtp="on", draft_tokens=3)
    with mock.patch.object(hy4_flash, "mtp_weight_bytes", return_value=4 * 2**30):
        hy4_flash.defaults(plain, computer())
        hy4_flash.defaults(draft, computer())
    assert draft.expert_vram < plain.expert_vram - 4
    assert draft.mtp == "on" and draft.drafting == "mtp" and draft.draft_tokens == 3
    assert S.LaunchSettings.from_dict(draft.to_dict()).draft_tokens == 3
    assert '--draft-tokens "3"' in S.flags_text(draft)


@pytest.mark.parametrize("free_gib,requested,expected,host", [
    (16, None, "off", False), (24, None, "on", False),
    (16, "on", "on", True), (48, "off", "off", False),
])
def test_default_mtp_requires_space_for_gpu_draft_and_trunk(free_gib, requested, expected, host):
    s = S.LaunchSettings(family="hy4-flash", model_path="fixture", context=33024, mtp=requested)
    with mock.patch.object(hy4_flash, "available_vram_gib", return_value=free_gib), \
         mock.patch.object(hy4_flash, "mtp_model_layout", return_value=(True, 512)), \
         mock.patch.object(hy4_flash, "model_file", return_value="fixture"), \
         mock.patch.object(hy4_flash, "mtp_weight_bytes", side_effect=lambda _, host_experts=False:
                           int((0.12 if host_experts else 3.71) * 2**30)):
        hy4_flash.defaults(s, computer())
    assert s.mtp == expected
    assert hy4_flash.mtp_host_experts(s) == host


def test_host_mapped_draft_budget_accounts_for_experts_on_both_shards(tmp_path):
    from types import SimpleNamespace

    first = tmp_path / "model-00001-of-00002.gguf"
    last = tmp_path / "model-00002-of-00002.gguf"

    def tensor(name, size):
        return SimpleNamespace(name=name, expected_bytes=lambda: size)

    files = {
        first: SimpleNamespace(metadata={"hy_v4_flash.block_count": 43,
                                         "hy_v4_flash.nextn_predict_layers": 1},
                               tensors=[tensor("blk.41.ffn_gate_exps.weight", 100),
                                        tensor("blk.42.ffn_gate_exps.weight", 10),
                                        tensor("blk.42.nextn.eh_proj.weight", 4)]),
        last: SimpleNamespace(tensors=[tensor("blk.42.ffn_up_exps.weight", 11),
                                       tensor("blk.42.ffn_down_exps.weight", 12)]),
    }
    with mock.patch("tools.gguf_reader.GGUFFile", side_effect=files.__getitem__):
        assert hy4_flash.mtp_weight_bytes(first) == 37
        assert hy4_flash.mtp_weight_bytes(first, host_experts=True) == 4


def test_saved_family_and_expert_settings_migrate_together():
    saved = {"family": "hy4next", "extra": {"STRATA_HY4_CPU_SPLIT": "fixed",
             "STRATA_HY4_FLASH_CPU_SPLIT": "auto", "HY4_DIRECT_CONTEXT": "8192"}}
    settings = S.LaunchSettings.from_dict(saved)
    assert settings.family == "hy4-flash"
    assert settings.extra == {"STRATA_HY4_FLASH_CPU_SPLIT": "auto", "HY4_FLASH_DIRECT_CONTEXT": "8192"}
    assert S.LaunchSettings.from_dict(settings.to_dict()).extra == settings.extra


def test_old_host_prefetch_preset_requires_an_explicit_storage_mode(tmp_path):
    old = {"family": "hy4-flash", "model": "Q4_K_M", "expert_host": 50, "expert_vram": 30}
    with pytest.raises(ValueError, match="--no-resident-experts"):
        hy4_flash.defaults(S.LaunchSettings.from_dict(old), computer())

    parser = argparse.ArgumentParser()
    add_parsers(parser.add_subparsers())
    for flag, split in (("--resident-experts", True), ("--no-resident-experts", False)):
        args = parser.parse_args(["launch", "--preset", "old-hy4-flash", flag])
        with mock.patch("strata_cli.launch.store.load", return_value=S.LaunchSettings.from_dict(old)):
            settings, _ = settings_from_args(args, catalog(root=tmp_path))
        assert settings.resident_experts is split
        assert "--expert-host" not in settings.extra


def test_mtp_boolean_flag_is_not_a_missing_pack(tmp_path):
    cfg={"arch":"hy4-flash","args":["--pack",str(tmp_path),"--mtp","--draft-tokens","3"]}
    assert files_present(cfg)==[]


def h20_computer():
    c = computer()
    c.gpus[0]["name"] = "NVIDIA H20"
    c.gpus[1]["name"] = "NVIDIA RTX 4090"
    return c


def test_measured_runtime_defaults_and_recomputation():
    s = S.LaunchSettings(family="hy4-flash", mtp="on", gpus=[0], model_path="fixture")
    cfg = {"exe": "fixture", "backend": "cuda", "env": hy4_flash.serving_environment(True)}
    with mock.patch.object(hy4_flash, "runtime_capabilities", return_value={"sample", "window", "draft", "conditional"}), \
         mock.patch.object(hy4_flash, "mtp_model_layout", return_value=(True, 512)):
        hy4_flash.apply_runtime_defaults(cfg, s, h20_computer())
        assert all(cfg["env"][k] == "1" for k in hy4_flash.RUNTIME_KEYS)
        hy4_flash.apply_runtime_defaults(cfg, s, h20_computer())
        assert all(cfg["env"][k] == "1" for k in hy4_flash.RUNTIME_KEYS)
        # Changing a card recomputes card-specific defaults: SKIP_EMPTY is an H20 tuning.
        s.gpus = [1]
        hy4_flash.apply_runtime_defaults(cfg, s, h20_computer())
        assert cfg["env"]["STRATA_HY4_FLASH_WINDOW_SKIP_EMPTY"] == "0"
        assert all(cfg["env"][k] == "1" for k in hy4_flash.RUNTIME_KEYS if k != "STRATA_HY4_FLASH_WINDOW_SKIP_EMPTY")
        # A card without CUDA engine features keeps none of them.
        cfg["backend"] = "vulkan"
        hy4_flash.apply_runtime_defaults(cfg, s, h20_computer())
        assert all(cfg["env"][k] == "0" for k in hy4_flash.RUNTIME_KEYS)


@pytest.mark.parametrize("mtp,extra,model_ok,caps,expected", [
    ("off", {}, True, {"sample", "draft", "window", "conditional"}, {"GPU_SAMPLE"}),
    ("on", {"STRATA_HY4_FLASH_CPU_MISS_MAX": "8"}, True, {"sample", "draft", "window", "conditional"},
     {"GPU_SAMPLE", "MTP_GRAPH", "MTP_CACHE_ONLY"}),
    # The default CPU cap yields to an engine without CPU verify windows.
    ("on", {}, True, {"sample", "draft", "window", "conditional"},
     {"GPU_SAMPLE", "MTP_GRAPH", "MTP_CACHE_ONLY", "WINDOW", "WINDOW_GRAPH", "WINDOW_COMPACT_ROWS", "WINDOW_SKIP_EMPTY"}),
    ("on", {"STRATA_HY4_FLASH_CPU_MISS_MAX": "8"}, True, {"sample", "draft", "window", "conditional", "window_cpu"},
     {"GPU_SAMPLE", "MTP_GRAPH", "MTP_CACHE_ONLY", "WINDOW", "WINDOW_GRAPH", "WINDOW_COMPACT_ROWS", "WINDOW_SKIP_EMPTY"}),
    ("on", {"STRATA_HY4_FLASH_PREFILL_BORROW": "1"}, True, {"window", "draft", "window_cpu"},
     {"MTP_GRAPH", "MTP_CACHE_ONLY"}),
    ("on", {"STRATA_HY4_FLASH_PREFILL_BORROW": "1"}, True, {"window", "draft", "window_cpu", "window_borrow"},
     {"MTP_GRAPH", "MTP_CACHE_ONLY", "WINDOW", "WINDOW_GRAPH", "WINDOW_COMPACT_ROWS"}),
    ("on", {"STRATA_HY4_FLASH_GPU_SAMPLE": "0", "STRATA_HY4_FLASH_WINDOW": "0"}, True,
     {"sample", "draft", "window", "conditional"}, {"MTP_GRAPH", "MTP_CACHE_ONLY"}),
    ("on", {}, False, {"sample", "draft", "window", "conditional"}, {"GPU_SAMPLE"}),
    ("on", {}, True, set(), set()),
    ("on", {}, True, {"sample", "draft", "window"},
     {"GPU_SAMPLE", "MTP_GRAPH", "MTP_CACHE_ONLY", "WINDOW", "WINDOW_GRAPH", "WINDOW_COMPACT_ROWS"}),
])
def test_runtime_capability_and_explicit_overrides(mtp, extra, model_ok, caps, expected):
    s = S.LaunchSettings(family="hy4-flash", mtp=mtp, gpus=[0], model_path="fixture", extra=extra)
    cfg = {"exe": "fixture", "backend": "cuda", "env": {**hy4_flash.serving_environment(mtp == "on"), **extra}}
    with mock.patch.object(hy4_flash, "runtime_capabilities", return_value=caps), \
         mock.patch.object(hy4_flash, "mtp_model_layout", return_value=(model_ok, 512)):
        hy4_flash.apply_runtime_defaults(cfg, s, h20_computer())
    actual = {k.removeprefix("STRATA_HY4_FLASH_") for k in hy4_flash.RUNTIME_KEYS if cfg["env"][k] == "1"}
    assert actual == expected
    for k, v in extra.items():
        assert cfg["env"][k] == v


def test_feature_detection_does_not_execute_engine(tmp_path):
    exe = tmp_path / "engine"
    exe.write_bytes(b"HY4_FLASH_GPU_SAMPLE enabled\0HY4_FLASH_WINDOW_SUPPLY initialized\0HY4_FLASH_WINDOW_CPU enabled\0HY4_FLASH_MTP_GRAPH captured\0")
    assert hy4_flash.runtime_capabilities(str(exe)) == {"sample", "window", "draft", "window_cpu"}
    assert hy4_flash.runtime_capabilities(str(tmp_path / "missing")) == set()


def test_runner_default_config_matches_plan_and_keeps_overrides():
    with Home() as home:
        cat = catalog()
        model = home.base / "model.gguf"
        model.write_bytes(gguf_fixture(architecture="hy_v4_flash", expert_type=12))
        pack = home.base / "pack"
        for name in hy4_flash.PACK_FILES:
            p = pack / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("fixture")
        exe = home.base / "strata"
        exe.write_bytes(b"HY4_FLASH_GPU_SAMPLE enabled\0HY4_FLASH_WINDOW_SUPPLY initialized\0"
                        b"HY4_FLASH_MTP_GRAPH captured\0window conditional body capture failed\0")
        exe.chmod(0o700)
        s = S.LaunchSettings(name="hy4-flash-defaults", family="hy4-flash", model_path=str(model),
            pack_path=str(pack), engine_path=str(exe), mtp="on", context=8192,
            expert_vram=48, extra={"STRATA_HY4_FLASH_WINDOW_SKIP_EMPTY": "0"})
        runner = Runner(s, catalog=cat)
        with mock.patch.object(cat, "computer", return_value=h20_computer()), \
             mock.patch.object(hy4_flash, "mtp_weight_bytes", return_value=4 * 2**30), \
             mock.patch.object(hy4_flash, "mtp_model_layout", return_value=(True, 512)), \
             mock.patch.object(runner, "_spawn", side_effect=AssertionError("no execution")):
            before = set(home.base.rglob("*"))
            plan = runner.plan()
            assert set(home.base.rglob("*")) == before
            path = runner.build_config(runner.ensure_model())
            cfg = read_config(path)
        assert "--resident-experts" not in cfg["args"] and "--expert-host" not in cfg["args"]
        assert plan["hy4_flash_runtime"] == {k: cfg["env"][k] for k in hy4_flash.RUNTIME_KEYS}
        assert cfg["env"]["STRATA_HY4_FLASH_WINDOW_SKIP_EMPTY"] == "0"
        assert all(cfg["env"][k] == "1" for k in hy4_flash.RUNTIME_KEYS if k != "STRATA_HY4_FLASH_WINDOW_SKIP_EMPTY")
        assert cfg["env"].get("STRATA_HY4_FLASH_WINDOW_GROUPED", "0") == "0"


@pytest.mark.parametrize("available,requested,explicit,expected_mode,expected_depth", [
    (12, None, None, "off", None), (16, None, None, "off", None),
    (24, None, None, "on", 1), (32, None, None, "on", 1), (48, None, None, "on", 1),
    (24, "off", None, "off", None), (16, "on", None, "on", 1), (24, "on", 3, "on", 3),
])
def test_reap25_selects_depth_after_placement(available, requested, explicit, expected_mode, expected_depth):
    settings = S.LaunchSettings(family="hy4-flash", model_path="renamed.gguf", mtp=requested,
                                draft_tokens=explicit, context=33024)
    with mock.patch.object(hy4_flash, "available_vram_gib", return_value=available), \
         mock.patch.object(hy4_flash, "mtp_model_layout", return_value=(True, 384)), \
         mock.patch.object(hy4_flash, "model_file", return_value="renamed.gguf"), \
         mock.patch.object(hy4_flash, "mtp_weight_bytes", side_effect=lambda _, host_experts=False:
                           int((0.12 if host_experts else 2.82) * 2**30)):
        hy4_flash.defaults(settings, computer())
    assert (settings.mtp, settings.draft_tokens) == (expected_mode, expected_depth)


def test_48g_4090_uses_three_drafts_with_cpu_windows_and_respects_explicit_width():
    c = computer()
    c.gpus[0].update(name="NVIDIA GeForce RTX 4090", vram_gb=48)
    caps = {"sample", "draft", "window", "conditional", "window_cpu"}
    with mock.patch.object(hy4_flash, "runtime_capabilities", return_value=caps), \
         mock.patch.object(hy4_flash, "mtp_model_layout", return_value=(True, 512)):
        for cap, explicit, expected in (("0", None, 1), ("0", 3, 3), ("8", None, 3), ("8", 1, 1), (None, None, 3)):
            extra = {} if cap is None else {"STRATA_HY4_FLASH_CPU_MISS_MAX": cap}
            s = S.LaunchSettings(family="hy4-flash", mtp="on", model_path="fixture", expert_vram=30, gpus=[0],
                                 resident_experts=True, draft_tokens=explicit, extra=extra)
            hy4_flash.defaults(s,c)
            cfg = {"backend":"cuda", "mtp_resident_bytes": 3980025984,
                   "env":{**hy4_flash.serving_environment(True), **s.extra}}
            hy4_flash.apply_runtime_defaults(cfg,s,c)
            assert s.draft_tokens == expected
            assert cfg["env"]["STRATA_HY4_FLASH_GPU_SAMPLE"] == "1"
            assert cfg["env"]["STRATA_HY4_FLASH_WINDOW_GRAPH"] == "1"
            assert cfg["env"]["STRATA_HY4_FLASH_WINDOW_SKIP_EMPTY"] == "0"
            assert cfg["env"]["STRATA_HY4_FLASH_PREFILL_CHUNK"] == "1024"
            s.extra["STRATA_HY4_FLASH_PREFILL_CHUNK"] = "512"
            cfg["env"]["STRATA_HY4_FLASH_PREFILL_CHUNK"] = "512"
            hy4_flash.apply_runtime_defaults(cfg,s,c)
            assert cfg["env"]["STRATA_HY4_FLASH_PREFILL_CHUNK"] == "512"


def test_default_cpu_experts_use_every_slot_with_q8k_activations():
    plain, draft = hy4_flash.serving_environment(False), hy4_flash.serving_environment(True)
    for env in (plain, draft):
        assert env["STRATA_HY4_FLASH_CPU_MISS_MAX"] == "16" and env["STRATA_HY4_FLASH_CPU_SPLIT"] == "auto"
        assert env["STRATA_HY4_FLASH_CPU_NATIVE_Q8K"] == "1"


@pytest.mark.parametrize("mtp", ["off", "on"])
def test_layer_major_prefill_at_minimum_pool_and_explicit_overrides(mtp):
    s = S.LaunchSettings(family="hy4-flash", mtp=mtp, expert_vram=0, gpus=[0])
    cfg = {"backend": "cuda", "env": hy4_flash.serving_environment(mtp == "on")}
    hy4_flash.apply_runtime_defaults(cfg, s, computer())
    assert cfg["env"]["STRATA_HY4_FLASH_PREFILL_LAYER_MAJOR"] == "1"
    assert cfg["env"]["STRATA_HY4_FLASH_PREFILL_CHUNK"] == "256"
    assert cfg["env"]["STRATA_HY4_FLASH_PREFILL_SUPERCHUNK"] == "8192"
    assert cfg["env"]["STRATA_HY4_FLASH_PREFILL_BORROW"] == "0"
    s.extra.update(STRATA_HY4_FLASH_PREFILL_LAYER_MAJOR="0", STRATA_HY4_FLASH_PREFILL_CHUNK="512")
    cfg["env"].update(s.extra)
    hy4_flash.apply_runtime_defaults(cfg, s, computer())
    assert cfg["env"]["STRATA_HY4_FLASH_PREFILL_LAYER_MAJOR"] == "0"
    assert cfg["env"]["STRATA_HY4_FLASH_PREFILL_CHUNK"] == "512"


@pytest.mark.parametrize("pool_gib,host,mtp,expected", [
    (3.58, "1", "on", "512"), (2.0, "1", "on", "512"),
    (1.99, "1", "on", "256"), (1.34, "1", "on", "256"),
    (3.58, "0", "on", "512"), (3.58, "1", "off", "512"),
])
def test_prefill_chunk_tracks_draft_placement_and_trunk_pool(pool_gib, host, mtp, expected):
    key = "STRATA_HY4_FLASH_PREFILL_CHUNK"
    s = S.LaunchSettings(family="hy4-flash", mtp=mtp, expert_vram=pool_gib, gpus=[0],
                         extra={"STRATA_HY4_FLASH_MTP_HOST_EXPERTS": host})
    cfg = {"backend": "cuda", "env": {**hy4_flash.serving_environment(mtp == "on"), **s.extra}}
    with mock.patch.object(hy4_flash, "available_vram_gib", return_value=16):
        hy4_flash.apply_runtime_defaults(cfg, s, computer())
    assert cfg["env"][key] == expected
    s.extra[key] = cfg["env"][key] = "1024"
    hy4_flash.apply_runtime_defaults(cfg, s, computer())
    assert cfg["env"][key] == "1024"


@pytest.mark.parametrize("mtp,free_mib,pool_gib,draft_bytes", [
    ("off", 16383, 4.44, 0),
    ("on", 24575, 8.0, 3980025984),
])
def test_prefill_counts_context_expert_pool_and_gpu_draft_against_free_memory(mtp, free_mib, pool_gib, draft_bytes):
    key = "STRATA_HY4_FLASH_PREFILL_CHUNK"
    s = S.LaunchSettings(family="hy4-flash", mtp=mtp, expert_vram=pool_gib, context=33024, gpus=[0])
    cfg = {"backend": "cuda", "env": hy4_flash.serving_environment(mtp == "on"), "mtp_resident_bytes": draft_bytes}
    with mock.patch.object(hy4_flash, "available_vram_gib", return_value=free_mib / 1024):
        hy4_flash.apply_runtime_defaults(cfg, s, computer())
        assert cfg["env"][key] == "512"
        s.context = 8192
        hy4_flash.apply_runtime_defaults(cfg, s, computer())
        assert cfg["env"][key] == ("512" if free_mib <= 16384 else "1024")
        s.context, s.expert_vram = 33024, pool_gib - 1.44
        hy4_flash.apply_runtime_defaults(cfg, s, computer())
        assert cfg["env"][key] == ("512" if free_mib <= 16384 else "1024")
        s.expert_vram = pool_gib
        s.extra[key] = cfg["env"][key] = "256"
        hy4_flash.apply_runtime_defaults(cfg, s, computer())
        assert cfg["env"][key] == "256"


@pytest.mark.parametrize("free_gib,requested_chunk,expected_chunk,expected_pool", [
    (16, None, "512", 4.44),
    (24575 / 1024, None, "1024", 7.00),
    (24575 / 1024, "512", "512", 8.00),
    (24575 / 1024, "2048", "2048", 4.50),
    (32767 / 1024, None, "2048", 12.50),
    (31.375, None, "2048", 11.87),
    (32767 / 1024, "1024", "1024", 15.00),
    (48508 / 1024, None, "2048", 27.87),
    (48508 / 1024, "1024", "1024", 30.37),
])
def test_automatic_pool_leaves_room_for_selected_chunk(free_gib, requested_chunk, expected_chunk, expected_pool):
    key = "STRATA_HY4_FLASH_PREFILL_CHUNK"
    s = S.LaunchSettings(family="hy4-flash", model_path="model.gguf", context=33024, kv="fp16",
                         extra={key: requested_chunk} if requested_chunk else {})
    gpu_draft_bytes = 3980025984
    with mock.patch.object(hy4_flash, "available_vram_gib", return_value=free_gib), \
         mock.patch.object(hy4_flash, "mtp_model_layout", return_value=(True, 512)), \
         mock.patch.object(hy4_flash, "model_file", return_value=None), \
         mock.patch.object(hy4_flash, "mtp_weight_bytes", return_value=gpu_draft_bytes):
        hy4_flash.defaults(s, computer())
        mtp = s.mtp == "on"
        cfg = {"env": {**hy4_flash.serving_environment(mtp), **s.extra},
               "mtp_resident_bytes": gpu_draft_bytes if mtp else 0}
        hy4_flash.apply_runtime_defaults(cfg, s, computer())
    assert s.expert_vram == expected_pool
    assert cfg["env"][key] == expected_chunk


@pytest.mark.parametrize("extra,capable,exclusive,expected_pool,expected_borrow", [
    ({}, True, False, 31.37, "1"),
    ({"STRATA_HY4_FLASH_PREFILL_LM_R": "host"}, True, False, 31.37, "1"),
    ({"STRATA_HY4_FLASH_PREFILL_SUPERCHUNK": "16384"}, True, False, 31.37, "1"),
    ({"STRATA_HY4_FLASH_PREFILL_BORROW": "0"}, True, False, 27.87, "0"),
    ({"STRATA_HY4_FLASH_PREFILL_LAYER_MAJOR": "0"}, True, False, 27.87, "0"),
    ({}, True, True, 27.87, "0"),
    ({}, False, False, 27.87, "0"),
])
def test_48g_4k_loan_budget_matches_runtime(tmp_path, extra, capable, exclusive, expected_pool, expected_borrow):
    exe = tmp_path / "strata"
    exe.write_bytes(b"prefill borrow: verify window drain failed STRATA_HY4_FLASH_PREFILL_LM_R" if capable else b"old engine")
    exe.chmod(0o755)
    machine = Computer(ram_gb=128, engine_vendor="nvidia", gpus=[
        {"index": 0, "vendor": "nvidia", "name": "NVIDIA GeForce RTX 4090", "vram_gb": 48, "usable": True}])
    s = S.LaunchSettings(family="hy4-flash", model_path="model.gguf", engine_path=str(exe), context=33024,
                         resident_experts=exclusive, extra={"STRATA_HY4_FLASH_PREFILL_CHUNK": "4096", **extra})
    with mock.patch.object(hy4_flash, "available_vram_gib", return_value=48508/1024), \
         mock.patch.object(hy4_flash, "mtp_model_layout", return_value=(True, 512)), \
         mock.patch.object(hy4_flash, "model_file", return_value=tmp_path), \
         mock.patch.object(hy4_flash, "mtp_weight_bytes", return_value=3980025984):
        hy4_flash.defaults(s, machine)
        cfg = {"backend": "cuda", "exe": str(exe), "env": {**hy4_flash.serving_environment(True), **s.extra}}
        hy4_flash.apply_runtime_defaults(cfg, s, machine)
    assert s.expert_vram == expected_pool
    assert cfg["env"]["STRATA_HY4_FLASH_PREFILL_BORROW"] == expected_borrow
    expect_device = expected_borrow == "1" and not extra
    assert cfg["env"].get("STRATA_HY4_FLASH_PREFILL_LM_R", "host") == ("vram" if expect_device else "host")


@pytest.mark.parametrize("free_mib,chunk,pool,borrow", [
    (12287, "512", 0.44, "0"),
    (16383, "1024", 5.25, "1"),
    (24575, "2048", 8.00, "1"),
    (32767, "2048", 16.00, "1"),
    (48508, "4096", 31.37, "1"),
])
def test_4090_default_prefill_loans(tmp_path, free_mib, chunk, pool, borrow):
    exe = tmp_path / "strata"
    exe.write_bytes(b"prefill borrow: verify window drain failed STRATA_HY4_FLASH_PREFILL_LM_R")
    exe.chmod(0o755)
    machine = computer()
    machine.gpus[0].update(name="NVIDIA GeForce RTX 4090", vram_gb=48)
    settings = S.LaunchSettings(family="hy4-flash", model_path="model.gguf", context=33024,
                                engine_path=str(exe), gpus=[0])
    with mock.patch.object(hy4_flash, "available_vram_gib", return_value=free_mib/1024), \
         mock.patch.object(hy4_flash, "mtp_model_layout", return_value=(True, 512)), \
         mock.patch.object(hy4_flash, "model_file", return_value=tmp_path), \
         mock.patch.object(hy4_flash, "mtp_weight_bytes", return_value=3980025984):
        hy4_flash.defaults(settings, machine)
        mtp = settings.mtp == "on"
        cfg = {"backend": "cuda", "exe": str(exe), "env": hy4_flash.serving_environment(mtp),
               "mtp_resident_bytes": 3980025984 if mtp else 0}
        hy4_flash.apply_runtime_defaults(cfg, settings, machine)
    assert mtp == (free_mib > 16384)
    assert settings.expert_vram == pool
    assert cfg["env"]["STRATA_HY4_FLASH_PREFILL_CHUNK"] == chunk
    assert cfg["env"]["STRATA_HY4_FLASH_PREFILL_BORROW"] == borrow
    assert cfg["env"].get("STRATA_HY4_FLASH_PREFILL_LM_R", "host") == ("vram" if borrow == "1" else "host")


@pytest.mark.parametrize("vram_gb,mtp,expected,host", [
    (11.99, "off", 1.42, None), (11.99, "on", 1.16, "1"),
    (23.99, "off", 12.42, None), (23.99, "on", 8.57, None),
    (47.99, "off", 33.92, None), (47.99, "on", 30.07, None),
])
def test_fixed_allocation_expert_budget_fills_measured_4090_tiers(vram_gb, mtp, expected, host):
    c = computer()
    c.gpus[0].update(name="NVIDIA GeForce RTX 4090", vram_gb=vram_gb)
    s = S.LaunchSettings(family="hy4-flash", model_path="fixture", mtp=mtp, context=1024, gpus=[0], kv="fp16",
                         extra={"STRATA_HY4_FLASH_PREFILL_BORROW": "0"})
    def draft(path, host_experts=False):
        return int(0.121 * 2**30) if host_experts else 3980025984
    with mock.patch.object(hy4_flash, "mtp_weight_bytes", side_effect=draft), \
         mock.patch.object(hy4_flash, "model_file", side_effect=lambda p: p), \
         mock.patch.object(hy4_flash, "mtp_model_layout", return_value=(True, 512)), \
         mock.patch.dict(hy4_flash.os.environ, {}, clear=False) as env:
        env.pop("STRATA_HY4_FLASH_MTP_HOST_EXPERTS", None)
        hy4_flash.defaults(s, c)
    assert s.expert_vram == pytest.approx(expected, abs=0.02)
    assert s.extra.get("STRATA_HY4_FLASH_MTP_HOST_EXPERTS") == host


@pytest.mark.parametrize("name,vram_gb", [("NVIDIA GeForce RTX 4090", 24), ("NVIDIA GeForce RTX 3090", 24),
                                           ("NVIDIA RTX A6000", 48)])
def test_any_cuda_card_mtp_enables_cpu_windows(name, vram_gb):
    c = computer()
    c.gpus[0].update(name=name, vram_gb=vram_gb)
    caps = {"sample", "draft", "window", "conditional", "window_cpu"}
    s = S.LaunchSettings(family="hy4-flash", mtp="on", model_path="fixture", expert_vram=9, gpus=[0])
    with mock.patch.object(hy4_flash, "runtime_capabilities", return_value=caps), \
         mock.patch.object(hy4_flash, "mtp_model_layout", return_value=(True, 512)):
        hy4_flash.defaults(s, c)
        cfg = {"backend": "cuda", "env": hy4_flash.serving_environment(True)}
        hy4_flash.apply_runtime_defaults(cfg, s, c)
    assert s.draft_tokens == 3
    assert cfg["env"]["STRATA_HY4_FLASH_CPU_MISS_MAX"] == "16" and cfg["env"]["STRATA_HY4_FLASH_WINDOW"] == "1"


def test_hy4_flash_uses_free_memory_without_changing_physical_gpu_identity():
    card = dict(index=0, vendor="nvidia", name="NVIDIA GeForce RTX 4090", vram_gb=48, driver="580")
    with mock.patch.object(hy4_flash.subprocess, "check_output", return_value="12280\n"):
        assert hy4_flash.available_vram_gib(card) == 12280 / 1024
    assert hy4_flash.is_48g_4090(card)
