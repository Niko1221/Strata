"""strata_control/backends/dsv41.py - Adapter for strata-dsv41 DeepSeek engine and Maya packing tools."""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from terminal.strata_control.childenv import quiet_crash_env
from terminal.strata_control.bundle_identity import VerifiedBundle
from terminal.strata_control.backend_contracts import BackendCapabilities
from terminal.strata_control.model_bundle import inspect_gguf


@dataclass(frozen=True)
class CompatibilityReport:
    decision: str  # QUALIFIED, INCOMPATIBLE, MISSING_TENSORS
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    required_dense_tensors: list[str] = field(default_factory=list)
    qualification_status: str = "IN_PROGRESS"


@dataclass(frozen=True)
class PrepareRequest:
    output_dir: Path
    allow_dense_override: bool = False
    context_tokens: int = 4096


@dataclass(frozen=True)
class PreparedModel:
    model_set_id: str
    pack_manifest_path: str
    dense_path: str
    engram_path: str | None = None
    is_ready: bool = True
    error_message: str | None = None


# Canonical dense tensor groups required for DeepSeek execution (each tuple represents accepted aliases)
REQUIRED_DEEPSEEK_DENSE_ALIASES: tuple[tuple[str, ...], ...] = (
    ("model.embed_tokens.weight", "token_embd.weight"),
    ("model.norm.weight", "output_norm.weight"),
    ("lm_head.weight", "output.weight"),
)

REQUIRED_DEEPSEEK_DENSE_TENSORS = [aliases[0] for aliases in REQUIRED_DEEPSEEK_DENSE_ALIASES]


def validate_dsv41(
    bundle: VerifiedBundle,
    caps: BackendCapabilities | None = None,
) -> CompatibilityReport:
    """Validate that bundle meets DeepSeek geometry, real required tensors, and dialect specifications."""
    errors: list[str] = []
    warnings: list[str] = []

    # Architecture validation
    arch = bundle.bundle.architecture.lower()
    if arch not in ("dsv41", "deepseek", "deepseek2", "deepseek41"):
        errors.append(f"Architecture '{arch}' is not supported by strata-dsv41")

    # Real dense tensor verification across all shards
    observed_tensor_names: set[str] = set()
    for fm in bundle.verified_files:
        try:
            inspected = inspect_gguf(Path(fm.path))
            observed_tensor_names.update(t.name for t in inspected.tensors)
        except Exception as exc:
            errors.append(f"Failed to inspect tensors in shard '{Path(fm.path).name}': {exc}")

    missing: list[str] = []
    for alias_group in REQUIRED_DEEPSEEK_DENSE_ALIASES:
        if not any(alias in observed_tensor_names for alias in alias_group):
            missing.append(alias_group[0])

    if missing:
        errors.append(f"Missing required DeepSeek dense tensors: {missing}")

    decision = "QUALIFIED" if not errors else "INCOMPATIBLE"
    return CompatibilityReport(
        decision=decision,
        errors=errors,
        warnings=warnings,
        required_dense_tensors=REQUIRED_DEEPSEEK_DENSE_TENSORS,
        qualification_status="PASS" if not errors else "FAIL",
    )


def prepare_dsv41(
    bundle: VerifiedBundle,
    request: PrepareRequest,
) -> PreparedModel:
    """Prepare DeepSeek pack manifest and staging area with real packer verification."""
    report = validate_dsv41(bundle)
    if report.decision != "QUALIFIED":
        raise ValueError(f"Cannot prepare incompatible DeepSeek bundle: {report.errors}")

    request.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_p = request.output_dir / "ds4_pack.json"
    
    # Check for pack_gguf tool
    root = Path(__file__).resolve().parents[2]
    pack_tool = root / "tools" / "dsv41" / "pack_gguf.py"
    
    is_ready = False
    error_msg = None
    
    if pack_tool.exists() and bundle.verified_files:
        first_shard = Path(bundle.verified_files[0].path)
        model_dir = first_shard.parent
        ds4_txt = request.output_dir / "ds4.txt"
        
        # If output files already exist or packer successfully generates them
        if ds4_txt.exists():
            is_ready = True
        else:
            try:
                cmd = [
                    sys.executable,
                    str(pack_tool),
                    "--model",
                    str(model_dir),
                    "--out",
                    str(request.output_dir),
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                                      env=quiet_crash_env(dict(os.environ)))
                if proc.returncode == 0 and ds4_txt.exists():
                    is_ready = True
                else:
                    error_msg = f"DeepSeek packer did not produce ds4.txt: {proc.stderr[:200]}"
            except Exception as exc:
                error_msg = f"DeepSeek packing execution error: {exc}"

    manifest_p.write_text(
        f'{{"model_set_id": "{bundle.model_set_id}", "is_ready": {str(is_ready).lower()}}}\n',
        encoding="utf-8",
    )

    return PreparedModel(
        model_set_id=bundle.model_set_id,
        pack_manifest_path=str(manifest_p),
        dense_path=str(bundle.verified_files[0].path),
        is_ready=is_ready,
        error_message=error_msg,
    )
