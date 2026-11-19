"""strata_control/backend_registry.py - Backend registration and deterministic routing engine."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from strata_control.bundle_identity import VerifiedBundle
from strata_control.backend_contracts import (
    BackendCapabilities,
    LoadRequest,
    ResolvedBackend,
)
from strata_control.backend_probe import probe_backend_binary


class BackendRegistry:
    def __init__(self) -> None:
        self._backends: dict[str, BackendCapabilities] = {}

    def register_backend(self, caps: BackendCapabilities) -> None:
        self._backends[caps.backend_id] = caps

    def get(self, backend_id: str) -> BackendCapabilities | None:
        return self._backends.get(backend_id)

    def list_backends(self) -> list[str]:
        return list(self._backends.keys())

    def select_backend(
        self,
        bundle: VerifiedBundle,
        request: LoadRequest,
    ) -> ResolvedBackend:
        """Deterministically route a verified bundle to an approved local backend without fallback."""
        arch = bundle.bundle.architecture.lower()
        dialect = bundle.bundle.dialect.lower()

        # If a specific backend was explicitly requested by user
        if request.preferred_backend:
            target = self.get(request.preferred_backend)
            if not target:
                raise ValueError(f"Requested backend '{request.preferred_backend}' is not registered")
            
            # Strict validation: do not substitute or cloud-fallback!
            if arch not in [a.lower() for a in target.supported_architectures]:
                raise ValueError(
                    f"Manual backend '{target.backend_id}' does not support architecture '{arch}'. Substitution forbidden."
                )
            if dialect not in [d.lower() for d in target.supported_dialects]:
                raise ValueError(
                    f"Manual backend '{target.backend_id}' does not support dialect '{dialect}'."
                )
            return ResolvedBackend(
                backend_id=target.backend_id,
                capabilities=target,
                decision_reason=f"Manually selected backend '{target.backend_id}' matches architecture and dialect",
            )

        # Automatic deterministic selection based on geometry and dialect
        # Priority 1: DeepSeek architecture -> strata-dsv41
        if arch in ("dsv41", "deepseek", "deepseek2", "deepseek41") and "strata-dsv41" in self._backends:
            ds_backend = self._backends["strata-dsv41"]
            return ResolvedBackend(
                backend_id="strata-dsv41",
                capabilities=ds_backend,
                decision_reason="Selected dedicated strata-dsv41 backend for DeepSeek architecture",
            )

        # Priority 2: Prism dialects (PQ2_0, PTQ1_0) -> prism
        if dialect in ("pq2_0", "ptq1_0") and "prism" in self._backends:
            prism_backend = self._backends["prism"]
            return ResolvedBackend(
                backend_id="prism",
                capabilities=prism_backend,
                decision_reason="Selected prism backend for PQ2_0 dialect",
            )

        # Priority 3: Native Qwen / Spin8 / Swift -> strata-native
        if arch in ("qwen4exp", "qwen2", "qwen", "qwen35", "qwen35moe") and "strata-native" in self._backends:
            native_backend = self._backends["strata-native"]
            return ResolvedBackend(
                backend_id="strata-native",
                capabilities=native_backend,
                decision_reason="Selected strata-native backend for Qwen/Swift architecture",
            )

        # Priority 4: Standard Dense GGUF -> llama-cpp
        if "llama-cpp" in self._backends:
            llama_backend = self._backends["llama-cpp"]
            if arch in [a.lower() for a in llama_backend.supported_architectures]:
                return ResolvedBackend(
                    backend_id="llama-cpp",
                    capabilities=llama_backend,
                    decision_reason="Selected llama-cpp backend for standard dense architecture",
                )

        raise ValueError(
            f"No suitable local backend found for architecture '{arch}' and dialect '{dialect}'. Cloud fallback is forbidden."
        )


def create_default_registry(root_dir: Path | None = None) -> BackendRegistry:
    """Create registry populated with standard Strata backends, cryptographically probed if root_dir is given."""
    reg = BackendRegistry()
    bin_base = Path(root_dir) / "bin" if root_dir else Path("bin")

    reg.register_backend(
        probe_backend_binary(
            binary_path=bin_base / "strata-native",
            backend_id="strata-native",
            supported_architectures=("qwen4exp", "qwen2", "qwen", "qwen35", "qwen35moe"),
            supported_dialects=("standard", "tq4_1s", "tq3_0s"),
            supports_cuda=True,
            supports_cpu=True,
            supports_spin8=True,
            supports_steering=True,
        )
    )
    reg.register_backend(
        probe_backend_binary(
            binary_path=bin_base / "strata-dsv41",
            backend_id="strata-dsv41",
            supported_architectures=("dsv41", "deepseek", "deepseek2", "deepseek41"),
            supported_dialects=("standard", "iq4_nl"),
            supports_cuda=True,
            supports_cpu=True,
            supports_steering=True,
        )
    )
    reg.register_backend(
        probe_backend_binary(
            binary_path=bin_base / "prism",
            backend_id="prism",
            supported_architectures=("qwen2", "qwen", "qwen35", "prism", "llama"),
            supported_dialects=("pq2_0", "ptq1_0"),
            supports_cuda=True,
            supports_cpu=True,
        )
    )
    reg.register_backend(
        probe_backend_binary(
            binary_path=bin_base / "llama-server",
            backend_id="llama-cpp",
            supported_architectures=("llama", "mistral", "phi3", "qwen2"),
            supported_dialects=("standard", "q4_k_m"),
            supports_cuda=True,
            supports_cpu=True,
        )
    )
    return reg
