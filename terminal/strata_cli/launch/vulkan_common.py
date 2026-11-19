"""Shared lifecycle plumbing for the two Vulkan research launches.

Both ``vulkan_native`` and ``vulkan_bridge`` write a Strata run config and then
need to behave like any other ``strata launch``: a saved ``launch-vulkan-*``
profile, a runtime owner host, a presets file, and an address ``strata launch
--status`` / ``--stop`` / ``strata connect --check`` can find. Keeping that in
one place avoids two copies drifting apart.

Nothing here changes the normal CUDA/HIP/SYCL/Metal flow. ``runtime.start`` is
the same call the Runner makes: it spawns the owner host, which loads the model,
watches the engine and answers the runtime RPC. That is what makes status,
stop and show-key work without Vulkan-specific cases elsewhere.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[3]
TERMINAL = REPO / "terminal"

# A fixed profile per implementation keeps --status/--stop simple and lets a
# second launch replace the first the way a normal preset restart does.
PROFILE = {"native": "launch-vulkan", "bridge": "launch-vulkan-bridge"}


def profile_name(impl: str) -> str:
    return PROFILE.get(impl, PROFILE["native"])


def _host_for_access(args) -> str:
    from strata_cli.launch import access
    kind = getattr(args, "access", None) or "local"   # the parser leaves it None when the flag is absent
    return access.host_for(kind, getattr(args, "listen", None))


def ensure_profile(config: pathlib.Path, impl: str, args) -> str:
    """Create or update the shared serving profile for this Vulkan launch.

    The run config written by setup_bridge.py / _native_config passes
    ``profiles.read_run_config`` (it has ``exe``, ``tokenizer`` and an ``args``
    array). Overrides carry host and port, exactly as Runner.ensure_profile
    does. Returns the profile id.
    """
    from strata_control import profiles

    name = profile_name(impl)
    host = _host_for_access(args)
    port = int(getattr(args, "port", 8080) or 8080)
    want = {"host": host, "port": port}
    try:
        current, _rev = profiles.load(name)
    except profiles.ProfileError:
        profiles.create(name, str(config))
    else:
        if current.get("source_config") != str(config.resolve()):
            proposed, rev = profiles.propose_source(name, str(config))
            profiles.save(name, proposed, expected_revision=rev)
    current, rev = profiles.load(name)
    if current.get("overrides") != want:
        profiles.save(name, {**current, "overrides": want}, expected_revision=rev)
    return name


def _settings(args, impl: str, gguf: pathlib.Path, config: pathlib.Path):
    """Build a LaunchSettings for --save-preset with the research path's choices."""
    from strata_cli.launch.settings import LaunchSettings

    name = str(getattr(args, "save_preset", None) or (getattr(args, "preset", None) or "vulkan")).strip()
    s = LaunchSettings(name=name, model_path=str(gguf),
                       context=int(getattr(args, "context", None) or 32768),
                       port=int(getattr(args, "port", 8080) or 8080),
                       access=str(getattr(args, "access", "local") or "local"),
                       data_dir=str(config.parent))
    s.extra["backend"] = "vulkan"
    s.extra["vulkan_impl"] = impl
    s.extra["model_path"] = str(gguf)
    return s


def save_preset(args, impl: str, gguf: pathlib.Path, config: pathlib.Path):
    """Save the whole answer set when --save-preset was asked for."""
    if not getattr(args, "save_preset", None):
        return None
    from strata_cli.launch import store
    saved = store.save(_settings(args, impl, gguf, config))
    store.touch_used(saved.name)
    return saved


def _print_result(impl: str, name: str, result: dict, machine: bool, args) -> int:
    from strata_cli.launch import access

    url = access.host_url(_host_for_access(args), int(getattr(args, "port", 8080) or 8080))
    urls = [url]
    key = str(os.environ.get("STRATA_API_KEY") or "").strip() or None
    if machine:
        print(json.dumps({"schema_version": 1, "ok": True, "backend": "vulkan", "impl": impl,
                          "profile": name, "urls": urls,
                          "api_key": "set" if key else None}, ensure_ascii=False, allow_nan=False))
        return 0
    print(f"Strata is serving ({impl} Vulkan engine).")
    print(f"  {'This computer':<14} {urls[0]}/v1")
    if key:
        print(f"  {'API key':<14} {access.mask_key(key)}   (show it: strata launch --show-key --preset {name})")
    print(f"  {'Profile':<14} {name}   (status: strata launch --status; stop: strata launch --stop --preset {name.removeprefix('launch-')})")
    print(f"  Watch it:    strata monitor --url {urls[0]}")
    return 0


def start(args, impl: str, gguf: pathlib.Path, config: pathlib.Path, machine: bool = False) -> int:
    """Start the saved Vulkan config through runtime, like every other launch.

    Returns the CLI exit code. A failed start is reported in the normal
    ``strata:`` voice and leaves the profile in place so ``strata launch
    --preset`` can retry it.
    """
    from strata_control import profiles, runtime

    name = ensure_profile(config, impl, args)
    try:
        result = runtime.start(name, wait=300)
    except Exception as exc:                                  # noqa: BLE001 - a start error is reported, not raised
        if machine:
            print(json.dumps({"schema_version": 1, "ok": False, "error": str(exc)}, ensure_ascii=False))
        else:
            print(f"strata: the Vulkan server could not start: {exc}", file=sys.stderr)
        return 1
    if not result.get("ok") or result.get("state") != "ready":
        detail = result.get("error") or result.get("state") or "unknown state"
        if machine:
            print(json.dumps({"schema_version": 1, "ok": False, "error": detail}, ensure_ascii=False))
        else:
            print(f"strata: the Vulkan server did not become ready ({detail}); see strata logs", file=sys.stderr)
        return 1
    saved = save_preset(args, impl, gguf, config)
    if saved is not None and not machine:
        print(f"Your choices were saved as the launch preset {saved.name}; "
              f"start it again with `strata launch --preset {saved.name}`.")
    return _print_result(impl, name, result, machine, args)
