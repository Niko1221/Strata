"""Owned dense runner contract; preview is never a successful native load."""
from __future__ import annotations

import hashlib
from pathlib import Path

ARCH = "qwen38_dense"
KV = ("auto", "f16", "k8v4", "spin8_k8v4", "f16_turbo2", "f16_turbo3", "f16_turbo4")
LANGUAGES = ("ja", "model-default")
CODECS = {"F32", "F16", "BF16", "Q8_0", "Q4_K", "Q5_K", "Q6_K", "IQ2_S", "IQ3_S", "IQ4_XS", "PQ2_0"}

def preview(metadata: dict, files: int) -> tuple[dict | None, str | None]:
    if metadata.get("architecture") != "qwen35":
        return None, None
    dims = metadata.get("dense_dimensions", {})
    expected = {"embedding_length": 5120, "feed_forward_length": 17408, "attention.head_count": 24,
                "attention.head_count_kv": 4, "attention.key_length": 256, "attention.value_length": 256,
                "ssm.state_size": 128, "ssm.group_count": 16, "ssm.time_step_rank": 48, "ssm.inner_size": 6144}
    if files != 1 or metadata.get("split"):
        return None, "dense native runner requires one unsplit GGUF"
    if dims.get("expert_count", 0) or metadata.get("expert_tensor_types"):
        return None, "dense native runner does not support experts"
    nextn = dims.get("nextn_predict_layers", 0)
    if nextn not in (0, 1) or dims.get("block_count") != 64 + nextn or any(dims.get(k) != v for k, v in expected.items()):
        return None, "incompatible dense qwen35 geometry; requires Qwen3.8-27B trunk"
    unknown = set(metadata.get("tensor_types", {})) - CODECS
    if unknown:
        return None, "unsupported dense weight codecs: " + ", ".join(sorted(unknown))
    if metadata.get("tokenizer_model") != "gpt2" or not all(metadata.get(k) for k in ("tokenizer_tokens_present", "tokenizer_merges_present")):
        return None, "dense native requires original GPT-2 tokenizer metadata"
    return {"family": ARCH, "model": "local", "basis": "dense_qwen35_geometry", "expected_expert_types": [],
            "expected_tensor_count": metadata.get("tensor_count_scanned"), "native_load": "unverified"}, None

def arguments(cfg: dict) -> list[str]:
    """Validate argv and reconcile effective scalar reports after user-owned options.

    Without args_written, raw args are explicit. With it, unchanged generated
    options use scalar settings; hand edits and user_args keep their authority.
    Mutate the effective fields only after all validation, for normal writers.
    """
    from strata_control import user_args
    unsupported = ("cache_type_k", "cache_type_v", "q36_memory", "q36_run", "spin8_mode", "spin8_native_config",
                   "spin8_native_sha256", "layer_split", "replicas", "split_fallback", "slot_save_path", "gpu_layers",
                   "refusal_direction_removal", "ds_experts")
    for key in unsupported:
        if cfg.get(key) not in (None, "", False):
            raise ValueError(f"dense native does not support {key}")
    owned = (user_args.capture(cfg) if isinstance(cfg.get("args_written"), list)
             else user_args.merge(cfg.get("args", []), cfg.get("user_args", [])))
    args = user_args.merge(cfg.get("args", []), owned)
    explicit = {user_args.flag_of(group) for group in user_args.groups(owned)}
    accepted = {"--model", "--ctx-size", "--kv", "--spin8-view", "--spin8-profile", "--output-language"}
    values = {}
    if len(args) % 2:
        raise ValueError("dense native options require flag/value pairs")
    for flag, value in zip(args[::2], args[1::2]):
        if flag not in accepted or flag in values or not isinstance(value, str) or not value or any(ord(c) < 32 for c in value):
            raise ValueError("invalid, repeated or unsupported dense native option")
        values[flag] = value
    if not Path(values.get("--model", "")).is_absolute():
        raise ValueError("dense native requires an absolute --model GGUF path")
    context = values["--ctx-size"] if "--ctx-size" in explicit else cfg.get("max_context", values.get("--ctx-size", 4096))
    if isinstance(context, bool) or not str(context).isascii() or not str(context).isdecimal() or not 1 <= int(context) <= 262144:
        raise ValueError("dense native context must be 1..262144")
    values["--ctx-size"] = str(context)
    if "--kv" not in explicit:
        values["--kv"] = cfg.get("dense_kv", values.get("--kv", "auto"))
    if "--output-language" not in explicit:
        values["--output-language"] = cfg.get("output_language", values.get("--output-language", "ja"))
    if values["--kv"] not in KV or values["--output-language"] not in LANGUAGES:
        raise ValueError("invalid dense native KV or output language")
    if values.get("--spin8-view", "vector") not in ("vector", "plus", "minus"):
        raise ValueError("invalid dense native Spin8 view")
    if cfg.get("parallel", 1) != 1 or cfg.get("vision") or cfg.get("drafting", "off") != "off" or cfg.get("mtp", False):
        raise ValueError("dense native supports one text session with drafting off")
    cfg.update(max_context=int(context), dense_kv=values["--kv"], output_language=values["--output-language"])
    if owned:
        cfg["user_args"] = owned
    return [part for pair in values.items() for part in pair]

def register(path: str, *, directory: Path | None = None, context: int | None = None, kv: str | None = None, language: str | None = None) -> dict:
    from strata_control import models
    from strata_cli.launch.runner import write_private
    plan = models.plan_import(path)
    if plan.get("issues") or (plan.get("candidate_selection") or {}).get("family") != ARCH:
        raise ValueError("not a plausible dense native model: " + "; ".join(plan.get("issues") or ["unsupported family"]))
    model = str(Path(plan["files"][0]["path"]).resolve())
    ident = "q38-dense-" + hashlib.sha256(model.encode("utf-8")).hexdigest()[:16]
    directory = directory or models.resolve_install()[0]
    target = directory / ("strata-" + ident + ".json")
    cfg = models._read(target)[1] if target.exists() or target.is_symlink() else {
        "arch": ARCH, "exe": "component:strata-engine", "native_runner": ARCH, "backend": "cuda",
        "model_name": Path(model).stem, "args": ["--model", model], "tokenizer": model,
        "output_language": "ja", "dense_kv": "auto", "max_context": 4096, "parallel": 1,
        "drafting": "off", "mtp": False, "split_engine": "off"}
    if cfg.get("arch") != ARCH:
        raise ValueError("registered dense model identity conflicts")
    for key, value in (("max_context", context), ("dense_kv", kv), ("output_language", language)):
        if value is not None:
            cfg[key] = value
    arguments(cfg)
    write_private(target, cfg)
    return {"model": ident, "config": str(target), "execution_compatibility": "unverified", "acquires": False}
