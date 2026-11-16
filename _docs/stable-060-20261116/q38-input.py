"""Reuse the frozen owned Q38 build without relabelling its compiled commit.

Only portable provenance, the executable and notices enter the package. Full
compiler/SDK inventories and guarded receipts stay in the private evidence tree.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
import shutil
import sys
from typing import Any

EVIDENCE = Path("H:/Strata/evidence/native-dense-own-cuda13-20261117")
REDIST = Path(
    "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Redist/MSVC/14.44.35112/x64/Microsoft.VC143.CRT"
)
COMMIT = "6ee425434c98277555332c7934e1ac51436eb198"
EXECUTABLE_SHA256 = "706867da2b9ad1a1debf09b0ac2f02ce3921412382e167ff4d1211efb7012af7"
MANIFESTS = {
    "language-build-manifest.json": "7204b8c66d84ebd7fff932fd4b71fdd1a1d6ef4cf2acddf9f4f3f6433324a0b1",
    "BUILD_MANIFEST-language-full-r1.json": "40851d465307d73a8c3844222112be5bfab87ca23c070923539c3bab53be4fcb",
}
NOTICES = {
    "src/qwen38/LICENSE-formats.txt": "LICENSE-q38-formats.txt",
    "tqcodec/DIALECT_LICENSE.txt": "LICENSE-turbo-dialects.txt",
}
CODECS = {
    "PrismML-0.4.1-PQ2_0": "1965755ceee6738b9a04ceaf5d30e6305d6ccd00",
    "TurboQuant-private-tables": "643956891dfdf51aecdbd05ca508dd21e21e8195",
    "Prism-linear-ternary": "73bfe6cc485554822f80c473a3c0c4f25e80f41b",
    "Strata-ggml-format-reference": "3cf03257f219afbe7334045ff7c6a06ac68c627d",
}


def stage(release: Any, root: Path, destination: Path) -> dict:
    """Validate immutable evidence and current bytes before copying any payload."""
    for name, expected in MANIFESTS.items():
        release.no_links(EVIDENCE / name)
        release.require(release.sha(EVIDENCE / name) == expected, "Frozen Q38 manifest changed")
    language = release.load(EVIDENCE / "language-build-manifest.json")
    full = release.load(EVIDENCE / "BUILD_MANIFEST-language-full-r1.json")
    release.require(
        language["sourceCommit"] == full["source_commit"] == COMMIT
        and full["clean_worktree"] is True
        and not full["missing_observed_paths"]
        and full["compiled_read_dependency_count"] == 624,
        "Q38 compiler source closure is not the reviewed build",
    )
    source_hashes = {}
    for entry in language["nativeSourceClosure"]:
        name = entry["path"]
        release.require(
            isinstance(name, str)
            and not Path(name).is_absolute()
            and "\\" not in name
            and not any(p in ("", ".", "..") for p in name.split("/"))
            and name not in source_hashes,
            "Unsafe Q38 source path",
        )
        path = root / name
        release.no_links(path)
        release.require(
            path.stat().st_size == entry["bytes"] and release.sha(path) == entry["sha256"],
            "Q38 source bytes changed: " + name,
        )
        source_hashes[name] = entry["sha256"]
    release.require(len(source_hashes) == 385, "Incomplete Q38 native closure")
    for entry in language["buildConfiguration"]:
        path = Path(entry["path"])
        release.no_links(path)
        release.require(release.sha(path) == entry["sha256"], "Frozen Q38 configuration changed")
    # Verify the successful native build/fixtures, retaining unrelated failed UI attempts.
    for name in ("q38-language-build-final", "q38-language-ctest-final", "q38-language-cli-final"):
        row = next(r for r in language["receipts"] if r["name"] == name)
        path = Path(row["path"])
        release.no_links(path)
        release.require(
            row["passed"] is True and release.sha(path) == row["sha256"],
            "Q38 guard receipt changed",
        )
        receipt = release.load(path)
        release.require(
            receipt["passed"] is True and receipt["run"]["job_active_after"] == 0,
            "Q38 guard did not drain",
        )
    executable = Path(language["executable"]["path"])
    release.no_links(executable)
    release.require(
        release.sha(executable)
        == language["executable"]["sha256"]
        == full["exe"]["sha256"]
        == EXECUTABLE_SHA256,
        "Frozen Q38 executable changed",
    )
    sys.path.insert(0, str(root / "packaging"))
    import binary_deps

    imports = binary_deps.pe_imports(executable)
    release.require(
        not any("llama" in n.lower() or "ggml" in n.lower() for n in imports),
        "Unexpected Q38 runtime dependency",
    )
    required = [
        n
        for n in imports
        if not any(
            fnmatch.fnmatchcase(n.lower(), p.lower())
            for p in (*binary_deps.WINDOWS_OS, *binary_deps.WINDOWS_DRIVER)
        )
    ]
    runtime_hashes = {}
    for name in required:
        path = REDIST / name
        release.no_links(path)
        release.require(path.is_file(), "Required Q38 redistributable is missing: " + name)
        runtime_hashes[name] = release.sha(path)
    record = {
        "schema": 1,
        "version": "0.6.0",
        "entrypoint": "strata-q38.exe",
        "sourceCommit": COMMIT,
        "sha256": EXECUTABLE_SHA256,
        "sources": dict(sorted(source_hashes.items())),
        "releaseQualified": False,
        "recipe": {
            "project": "src/qwen38/CMakeLists.txt",
            "configuration": "Release",
            "toolchain": full["toolchain"],
            "cudaArchs": "120",
            "cudaRuntime": "static",
            "compilerOptions": [
                "c++17",
                "/MD",
                "/O2",
                "/fp:strict (Spin8)",
                "--fmad=false",
                "--ftz=false",
                "--prec-div=true",
                "--prec-sqrt=true",
            ],
            "embeddedProfileSHA256": source_hashes["data/spin8/reference.profile"],
            "generatedProfileSHA256": language["buildConfiguration"][1]["sha256"],
            "codecSourcePins": CODECS,
            "runtimeImplementation": "Strata-owned",
            "compilerManifestSHA256": MANIFESTS["BUILD_MANIFEST-language-full-r1.json"],
            "languageManifestSHA256": MANIFESTS["language-build-manifest.json"],
        },
        "imports": imports,
        "requiredLibraries": required,
        "runtimeLibraries": runtime_hashes,
        "notices": {name: release.sha(root / source) for source, name in NOTICES.items()},
    }
    shutil.copy2(executable, destination / "strata-q38.exe")
    release.require(
        release.sha(destination / "strata-q38.exe") == EXECUTABLE_SHA256, "Q38 copy differs"
    )
    for source, name in NOTICES.items():
        release.no_links(root / source)
        shutil.copy2(root / source, destination / name)
    for name, expected in runtime_hashes.items():
        shutil.copy2(REDIST / name, destination / name)
        release.require(
            release.sha(destination / name) == expected == release.sha(REDIST / name),
            "Q38 CRT copy differs",
        )
    release.save(destination / "q38-build.json", record)
    return record
