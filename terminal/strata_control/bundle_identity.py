"""strata_control/bundle_identity.py - Streaming cryptographic hash identity and immutability verification for ModelBundles."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

from strata_control.bundle_resolver import ModelBundle, FileMember


@dataclass(frozen=True)
class VerifyPolicy:
    full_sha256: bool = True
    allow_symlinks: bool = True
    chunk_size: int = 1024 * 1024  # 1 MiB


@dataclass(frozen=True)
class VerifiedBundle:
    model_set_id: str
    canonical_hash: str
    bundle: ModelBundle
    verified_files: list[FileMember]
    policy: VerifyPolicy


def compute_file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute streaming SHA-256 for a single file."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_bundle(bundle: ModelBundle, policy: VerifyPolicy | None = None) -> VerifiedBundle:
    """Verify that all files in a bundle exist, have stable content, and compute the canonical bundle hash."""
    policy = policy or VerifyPolicy()
    verified_files: list[FileMember] = []
    bundle_hasher = hashlib.sha256()

    for fm in bundle.files:
        p = Path(fm.path)
        if not p.exists():
            raise FileNotFoundError(f"Bundle file missing during verification: {p}")

        if not policy.allow_symlinks and p.is_symlink():
            raise ValueError(f"Symlinks not permitted by verify policy: {p}")

        # Compute or check sha256
        if policy.full_sha256:
            sha = compute_file_sha256(p, chunk_size=policy.chunk_size)
        else:
            if not fm.sha256:
                raise ValueError(
                    f"Cannot verify bundle file {p}: full_sha256 is False and no trusted precomputed sha256 was provided"
                )
            sha = fm.sha256

        if fm.sha256 and sha != fm.sha256:
            raise ValueError(
                f"Content drift detected in {p}: expected {fm.sha256}, actual {sha}"
            )

        verified_fm = FileMember(
            role=fm.role,
            ordinal=fm.ordinal,
            path=str(p.resolve() if policy.allow_symlinks else p),
            file_size=p.stat().st_size,
            sha256=sha,
        )
        verified_files.append(verified_fm)
        bundle_hasher.update(f"{verified_fm.role}:{verified_fm.ordinal}:{verified_fm.sha256}".encode("utf-8"))

    canonical_hash = bundle_hasher.hexdigest()
    return VerifiedBundle(
        model_set_id=bundle.model_set_id,
        canonical_hash=canonical_hash,
        bundle=bundle,
        verified_files=verified_files,
        policy=policy,
    )
