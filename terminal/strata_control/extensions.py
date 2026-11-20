"""strata_control/extensions.py - Fork extension sidecar resolution and snapshot manager.

Provides:
- ExtensionOptions: Raw user/profile requested extension options.
- EffectiveExtension: Resolved, immutable snapshot of active extensions.
- resolve_extension(): Validates options, detects CLI argument collisions,
  and compiles an immutable snapshot with a cryptographic hash.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any


@dataclass(frozen=True)
class ExtensionOptions:
    schema_version: str = "1.0"
    spin8_mode: str = "off"  # off, cpu, cuda, triality_full
    dialect: str = "standard"  # standard, tq4_1s, tq3_0s, pq2_0
    steering_enabled: bool = False
    steering_profile: str = ""
    steering_strength: float = 0.0
    cli_args: list[str] = field(default_factory=list)
    raw_config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EffectiveExtension:
    schema_version: str
    clean_off: bool
    spin8_mode: str
    dialect: str
    steering_enabled: bool
    extra_args: list[str]
    snapshot_hash: str
    metadata: dict[str, Any]


def resolve_extension(options: ExtensionOptions) -> EffectiveExtension:
    """Resolve raw extension options into a canonical, immutable EffectiveExtension snapshot."""
    # Check for CLI argument collisions
    cli_spin8 = None
    if "--spin8-mode" in options.cli_args:
        idx = options.cli_args.index("--spin8-mode")
        if idx + 1 < len(options.cli_args):
            cli_spin8 = options.cli_args[idx + 1]

    if cli_spin8 is not None and options.spin8_mode != "off" and cli_spin8 != options.spin8_mode:
        raise ValueError(
            f"Conflicting extension settings: cli requested --spin8-mode {cli_spin8} "
            f"but options specified {options.spin8_mode}"
        )

    effective_spin8 = cli_spin8 if cli_spin8 is not None else options.spin8_mode
    is_off = (
        effective_spin8 == "off"
        and options.dialect == "standard"
        and not options.steering_enabled
    )

    extra_args: list[str] = []
    if not is_off:
        if effective_spin8 != "off":
            extra_args.extend(["--spin8-mode", effective_spin8])
        if options.dialect != "standard":
            extra_args.extend(["--kv", options.dialect])
        if options.steering_enabled and options.steering_profile:
            extra_args.extend(["--steering-profile", options.steering_profile])

    canonical_repr = {
        "schema_version": options.schema_version,
        "clean_off": is_off,
        "spin8_mode": effective_spin8,
        "dialect": options.dialect,
        "steering_enabled": options.steering_enabled,
        "steering_profile": options.steering_profile if options.steering_enabled else "",
        "steering_strength": round(options.steering_strength, 4) if options.steering_enabled else 0.0,
        "extra_args": extra_args,
    }
    json_bytes = json.dumps(canonical_repr, sort_keys=True, separators=(",", ":")).encode("utf-8")
    snapshot_hash = hashlib.sha256(json_bytes).hexdigest()

    return EffectiveExtension(
        schema_version=options.schema_version,
        clean_off=is_off,
        spin8_mode=effective_spin8,
        dialect=options.dialect,
        steering_enabled=options.steering_enabled,
        extra_args=extra_args,
        snapshot_hash=snapshot_hash,
        metadata=canonical_repr,
    )
