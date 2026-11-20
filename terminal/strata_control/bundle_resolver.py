"""strata_control/bundle_resolver.py - ModelBundle resolution, shard assembly, and replacement recipes."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import re
from typing import Any


@dataclass(frozen=True)
class FileMember:
    role: str  # shard, metadata, engram, adapter, tokenizer
    ordinal: int
    path: str
    file_size: int
    sha256: str = ""


@dataclass(frozen=True)
class ReplacementRecipe:
    parent_model_set_id: str
    slot: int
    replacement_path: str
    expected_parent_sha256: str = ""
    replacement_sha256: str = ""


@dataclass(frozen=True)
class ImportSelection:
    primary_path: Path
    candidate_paths: list[Path] = field(default_factory=list)
    artifact_kind: str = "model"
    architecture: str = "qwen2"
    dialect: str = "standard"


@dataclass(frozen=True)
class ModelBundle:
    schema_version: str
    model_set_id: str
    artifact_kind: str
    architecture: str
    dialect: str
    split_total: int
    files: list[FileMember]
    recipe: ReplacementRecipe | None = None


def resolve_bundle(
    selection: ImportSelection,
    recipe: ReplacementRecipe | None = None,
) -> ModelBundle:
    """Resolve an import selection and optional replacement recipe into a cohesive ModelBundle."""
    p = selection.primary_path
    if not p.exists():
        raise FileNotFoundError(f"Selected model path does not exist: {p}")

    # Check for adapter-only artifacts
    is_adapter = False
    p_name_lower = p.name.lower()
    if selection.architecture in ("adapter", "controlvector") or selection.artifact_kind in ("adapter", "controlvector"):
        is_adapter = True
    elif "mmproj" in p_name_lower or p_name_lower.endswith((".lora.gguf", ".adapter.gguf", "-adapter.gguf", "_adapter.gguf")):
        is_adapter = True
    else:
        try:
            from strata_control.model_bundle import inspect_gguf
            meta = inspect_gguf(p)
            if meta.architecture in ("adapter", "controlvector", "clip"):
                is_adapter = True
        except Exception:
            pass

    if is_adapter and selection.artifact_kind == "model":
        raise ValueError(f"Artifact '{p.name}' is an auxiliary adapter, not a runnable base model")

    # Detect shard pattern: e.g. model-00002-of-00006.gguf or model.gguf-02-of-04
    split_pattern = re.search(r"(\d+)-of-(\d+)", p.name)
    files: list[FileMember] = []
    split_total = 1

    if split_pattern:
        curr_idx = int(split_pattern.group(1))
        total_shards = int(split_pattern.group(2))
        split_total = total_shards

        # Find sibling shards in the same directory
        parent_dir = p.parent
        prefix = p.name[: split_pattern.start()]
        ext = p.name[split_pattern.end() :]

        for i in range(1, total_shards + 1):
            expected_num_str = f"{i:05d}" if len(split_pattern.group(1)) == 5 else f"{i:02d}"
            target_name = f"{prefix}{expected_num_str}-of-{split_pattern.group(2)}{ext}"
            sibling = parent_dir / target_name
            if not sibling.exists():
                raise FileNotFoundError(f"Missing required shard {i}/{total_shards}: {sibling}")
            
            # Check for mixed shard configurations (e.g. 4-part mixed with 6-part)
            if sibling.stat().st_size == 0:
                raise ValueError(f"Shard {sibling} is corrupt or zero-byte")

            files.append(
                FileMember(
                    role="shard",
                    ordinal=i - 1,
                    path=str(sibling),
                    file_size=sibling.stat().st_size,
                )
            )
    else:
        # Single-file bundle
        files.append(
            FileMember(
                role="shard",
                ordinal=0,
                path=str(p),
                file_size=p.stat().st_size,
            )
        )

    # Apply replacement recipe if provided
    if recipe is not None:
        rec_path = Path(recipe.replacement_path)
        if not rec_path.exists():
            raise FileNotFoundError(f"Replacement shard does not exist: {rec_path}")
        if recipe.slot < 0 or recipe.slot >= len(files):
            raise IndexError(f"Replacement slot {recipe.slot} out of range ({len(files)} shards)")
        
        # Replace the target slot without modifying original file
        orig = files[recipe.slot]
        files[recipe.slot] = FileMember(
            role="shard",
            ordinal=orig.ordinal,
            path=str(rec_path),
            file_size=rec_path.stat().st_size,
            sha256=recipe.replacement_sha256,
        )

    # Compute stable model_set_id from canonical file members
    hasher = hashlib.sha256()
    for fm in files:
        hasher.update(f"{fm.role}:{fm.ordinal}:{Path(fm.path).name}:{fm.file_size}".encode("utf-8"))
    model_set_id = f"bundle-{hasher.hexdigest()[:16]}"

    return ModelBundle(
        schema_version="1.0",
        model_set_id=model_set_id,
        artifact_kind=selection.artifact_kind,
        architecture=selection.architecture,
        dialect=selection.dialect,
        split_total=split_total,
        files=files,
        recipe=recipe,
    )
