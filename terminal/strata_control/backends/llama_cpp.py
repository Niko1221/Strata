"""strata_control/backends/llama_cpp.py - Generic llama.cpp backend adapter."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from terminal.strata_control.backend_contracts import BackendCapabilities
from terminal.strata_control.bundle_identity import VerifiedBundle, compute_file_sha256


@dataclass(frozen=True)
class LlamaCppLaunchPlan:
    backend_id: str = "llama-cpp"
    binary_path: str = "llama-server"
    args: tuple[str, ...] = field(default_factory=tuple)
    is_dense_only: bool = True
    verified_sha256: str = ""


def build_llama_cpp_plan(
    bundle: VerifiedBundle,
    caps: BackendCapabilities | None = None,
    port: int = 18502,
) -> LlamaCppLaunchPlan:
    arch = bundle.bundle.architecture.lower()
    if arch in ("dsv41", "deepseek2", "deepseek", "deepseek41"):
        raise ValueError(f"Generic llama.cpp backend does not support DeepSeek MoE architecture '{arch}'")

    bin_path = caps.binary_path if caps else "llama-server"
    verified_sha = ""
    if caps and Path(caps.binary_path).exists():
        actual_sha = compute_file_sha256(Path(caps.binary_path))
        if caps.binary_sha256 and actual_sha != caps.binary_sha256:
            raise ValueError(
                f"Binary hash mismatch for {caps.binary_path}: expected {caps.binary_sha256}, actual {actual_sha}"
            )
        verified_sha = actual_sha

    args = (
        "-m", str(bundle.verified_files[0].path),
        "--port", str(port),
        "--host", "127.0.0.1",
    )
    return LlamaCppLaunchPlan(
        backend_id="llama-cpp",
        binary_path=bin_path,
        args=args,
        verified_sha256=verified_sha,
    )
