"""Explicit local preview overlay; preserve prior engines and H rollback files.

Call apply only with an independently verified package and named install roots.
There is no download, automatic launch, service control or recursive deletion.
"""
from pathlib import Path
import hashlib
import json
import shutil
import uuid

from native_preview import no_reparse, relative_name
from strata_control import engine_component


def sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def record(path, data):
    with path.open("x", encoding="utf-8") as target:
        json.dump(data, target, indent=2)


def copy_checked(source, target, digest):
    no_reparse(source)
    no_reparse(target)
    if sha(source) != digest:
        raise ValueError("input drift")
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, target.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, 1024**2)
    shutil.copymode(source, target, follow_symlinks=False)
    if sha(target) != digest:
        raise ValueError("copied file differs")


def publish(source, target):
    if target.exists():
        raise ValueError("immutable preview destination already exists")
    source.rename(target)


def apply(stage, roots, control, backup, expected_control):
    stage, control, backup = Path(stage), Path(control), Path(backup)
    roots = [Path(root) for root in roots]
    for path in [stage, control, backup, *roots]:
        no_reparse(path)
    if len(roots) != 2 or roots[0].resolve() == roots[1].resolve():
        raise ValueError("two distinct explicit component roots required")
    if not all(root.is_dir() for root in roots):
        raise ValueError("existing component roots required")
    manifest_path = stage / "engine-component.json"
    with manifest_path.open("rb") as source:
        manifest_bytes = source.read(128 * 1024 + 1)
    if len(manifest_bytes) > 128 * 1024:
        raise ValueError("component manifest exceeds bound")
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    component = engine_component.read_manifest(stage)
    if component.raw != json.loads(manifest_bytes):
        raise ValueError("manifest changed while being parsed")
    engine_component.verify(component, full=True)
    signature_status = engine_component.verify_signature(component)
    if sha(manifest_path) != manifest_digest:
        raise ValueError("manifest changed after initial verification")

    def verify_copy(folder):
        if sha(folder / "engine-component.json") != manifest_digest:
            raise ValueError("installed manifest differs from verified bytes")
        copied = engine_component.read_manifest(folder)
        if copied.raw != component.raw or engine_component.verify_signature(copied) != signature_status:
            raise ValueError("copied manifest/signature differs from verified input")
        engine_component.verify(copied, full=True)
    version = relative_name(component.version)
    if "/" in version or component.platform != "windows-x86_64" or component.backend != "cuda":
        raise ValueError("Windows CUDA preview required")
    folder = f"{version}-windows-x86_64-cuda-cuda13"
    finals = [root / folder for root in roots]
    if any(path.exists() for path in finals) or backup.exists():
        raise ValueError("fresh preview and backup destinations required")
    # Publish the dependency before its consumer; an interrupted process must
    # not leave a new server importing a helper that has not been copied yet.
    names = ("memory_harness.py", "perf131k_e2e.py", "server.py")
    if set(expected_control) != set(names):
        raise ValueError("exact three-file Control overlay required")
    payload = {name: (digest, size) for name, digest, size in component.files}
    original = {}
    for name in names:
        target = control / "serve" / name
        no_reparse(target)
        observed = sha(target) if target.is_file() else None
        if observed != expected_control[name] or f"control/serve/{name}" not in payload:
            raise ValueError("Control drift or missing overlay input")
        original[name] = observed
    backup.mkdir()
    for name, digest in original.items():
        if digest is not None:
            copy_checked(control / "serve" / name, backup / name, digest)
    record(backup / "plan.json", {"components": list(map(str, finals)), "control": str(control),
                                  "original_control": original, "stage": str(stage)})
    prepared, published, changed = [], [], []
    try:
        for root in roots:
            temporary = root / f".staging-{folder}-{uuid.uuid4().hex}"
            temporary.mkdir()
            prepared.append(temporary)
            for name, (digest, _) in payload.items():
                relative_name(name)
                copy_checked(stage / name, temporary / name, digest)
            copy_checked(manifest_path, temporary / "engine-component.json", manifest_digest)
            verify_copy(temporary)
        for name, expected in original.items():
            target = control / "serve" / name
            if (sha(target) if target.is_file() else None) != expected:
                raise ValueError("Control changed after backup")
            digest = payload[f"control/serve/{name}"][0]
            temporary = target.with_name(f".{name}.{uuid.uuid4().hex}.new")
            copy_checked(stage / "control/serve" / name, temporary, digest)
            changed.append(name)  # Include uncertain replace failures in rollback responsibility.
            temporary.replace(target)
        for temporary, final in zip(prepared, finals):
            publish(temporary, final)
            published.append(final)
        for final in finals:
            verify_copy(final)
        for name in names:
            if sha(control / "serve" / name) != payload[f"control/serve/{name}"][0]:
                raise ValueError("installed Control hash mismatch")
        result = {"status": "INSTALLED_VERIFIED_FILES", "components": list(map(str, finals)),
                  "control_sha256": {name: sha(control / "serve" / name) for name in names},
                  "manifest_sha256": manifest_digest, "signature_status": signature_status,
                  "model_inference": "NOT_RUN", "GUI": "UNCHANGED"}
        record(backup / "result.json", result)
        return result
    except Exception:
        errors, retained = [], []
        for name in reversed(changed):
            try:
                target = control / "serve" / name
                if original[name] is None:
                    retained.append(str(target))  # Newly created inactive helpers are preserved, never deleted.
                else:
                    temporary = target.with_name(f".{name}.{uuid.uuid4().hex}.restore")
                    copy_checked(backup / name, temporary, original[name])
                    temporary.replace(target)
            except Exception as error:
                errors.append(type(error).__name__)
        for final in published:
            try:
                no_reparse(final)
                final.rename(final.with_name(f".failed-{folder}-{uuid.uuid4().hex}"))
            except Exception as error:
                errors.append(type(error).__name__)
        record(backup / "result.json", {"status": "ROLLBACK_UNCONFIRMED" if errors else "FAILED_ROLLED_BACK_EXISTING_FILES",
                                       "errors": errors, "retained_new_files": retained})
        raise
