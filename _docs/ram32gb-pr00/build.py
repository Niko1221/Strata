"""Compile changed Q36 units, linking unchanged source-bound 0.5.9 libraries.

Run only through check.py's owned Job. This never writes to the old build,
frozen checkout, installed engine or release assets. It is an incremental
candidate, not a clean full rebuild or a release.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("H:/Strata/evidence/ram32gb-pr00-20261115/candidate-r4")
PRIOR = Path("H:/Strata/releases/stable-059-20261113-codex/native-build")
SOURCE = "db640451e677ba2c43ebfb555a505e4c037e0bd1"
CUDA = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3")
NS = {"m": "http://schemas.microsoft.com/developer/msbuild/2003"}


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    # The only native source changes may be the two rebuilt units and tracing
    # headers. The remaining library sources must match the frozen build source.
    changed = subprocess.check_output(
        [
            "git",
            "diff",
            "--name-only",
            SOURCE,
            "--",
            "src",
            "include",
            "spin8",
            "tqcodec",
            "cmake",
            "third_party",
        ],
        cwd=ROOT,
        text=True,
    ).splitlines()
    allowed = {
        "src/qwen35/q36_engine.cu",
        "src/qwen35/expert_store.cpp",
        "include/strata/qwen35/phase_trace.hpp",
    }
    if set(changed) - allowed:
        raise RuntimeError("unchanged-library source identity no longer holds")
    project = ET.parse(PRIOR / "strata_qwen35.vcxproj")
    groups = project.findall("m:ItemDefinitionGroup", NS)
    release = next(
        g for g in groups if "'Release|x64'" in g.attrib.get("Condition", "")
    )
    definitions = release.find("m:ClCompile/m:PreprocessorDefinitions", NS).text.split(
        ";"
    )
    definitions = [
        d
        for d in definitions
        if d and "%(" not in d and not d.startswith("CMAKE_INTDIR=")
    ]
    includes = release.find(
        "m:ClCompile/m:AdditionalIncludeDirectories", NS
    ).text.split(";")
    includes = [p for p in includes if p and "%(" not in p]
    exe_project = ET.parse(PRIOR / "strata-q36.vcxproj")
    group = next(
        g
        for g in exe_project.findall("m:ItemDefinitionGroup", NS)
        if "'Release|x64'" in g.attrib.get("Condition", "")
    )
    dependencies = group.find("m:Link/m:AdditionalDependencies", NS).text.split(";")
    libraries = []
    for value in dependencies:
        if value == r"Release\strata_qwen35.lib":
            continue
        path = Path(value)
        libraries.append(
            str((PRIOR / path).resolve())
            if not path.is_absolute() and (PRIOR / path).is_file()
            else value
        )
    OUT.mkdir()  # A second attempt must use a new reviewed directory.
    source_files = [
        ROOT / p
        for p in (
            "src/qwen35/q36_engine.cu",
            "src/qwen35/expert_store.cpp",
            "src/qwen35/q36_main.cpp",
            "include/strata/qwen35/phase_trace.hpp",
            "include/strata/qwen35/startup_trace.hpp",
        )
    ]
    sources = {str(path.relative_to(ROOT)): sha(path) for path in source_files}
    library_hashes = {
        value: sha(Path(value)) for value in libraries if Path(value).is_absolute()
    }
    env_result = subprocess.run(
        [
            os.environ["COMSPEC"],
            "/d",
            "/u",
            "/c",
            str(Path(__file__).with_name("vs-env.cmd")),
        ],
        capture_output=True,
        check=True,
    )
    env = dict(os.environ)
    env.update(
        {
            key.upper(): value
            for key, value in (
                line.split("=", 1)
                for line in env_result.stdout.decode("utf-16le").splitlines()
                if "=" in line and not line.startswith("=")
            )
        }
    )
    commands = []

    def run(args: list[str]) -> None:
        if not Path(args[0]).is_absolute():
            args[0] = shutil.which(args[0], path=env["PATH"]) or args[0]
        commands.append(args)
        subprocess.run(args, cwd=OUT, env=env, check=True)

    flags = ["-D" + d for d in definitions] + ["-I" + p for p in includes]
    run(
        [
            str(CUDA / "bin/nvcc.exe"),
            "-c",
            str(ROOT / "src/qwen35/q36_engine.cu"),
            "-o",
            str(OUT / "q36_engine.obj"),
            "-std=c++20",
            "--threads",
            "5",
            "-O3",
            "-Xcompiler=/MT,/O2,/EHsc,/utf-8",
            "-D_WINDOWS",
            "--generate-code=arch=compute_120,code=[compute_120,sm_120]",
            *flags,
        ]
    )
    for source in ("expert_store.cpp", "q36_main.cpp"):
        run(
            [
                "cl.exe",
                "/nologo",
                "/c",
                "/std:c++20",
                "/EHsc",
                "/MT",
                "/O2",
                "/W4",
                "/utf-8",
                *["/D" + d for d in definitions],
                *["/I" + p for p in includes],
                str(ROOT / "src/qwen35" / source),
                "/Fo" + str(OUT / (source + ".obj")),
            ]
        )
    run(
        [
            "link.exe",
            "/nologo",
            "/MACHINE:X64",
            "/OUT:" + str(OUT / "strata-q36.exe"),
            "/LIBPATH:" + str(CUDA / "lib/x64"),
            str(OUT / "q36_engine.obj"),
            str(OUT / "expert_store.cpp.obj"),
            str(OUT / "q36_main.cpp.obj"),
            *libraries,
        ]
    )
    if sources != {str(path.relative_to(ROOT)): sha(path) for path in source_files}:
        raise RuntimeError("source changed during candidate compilation")
    if library_hashes != {value: sha(Path(value)) for value in library_hashes}:
        raise RuntimeError("reused library changed during candidate compilation")
    receipt = {
        "kind": "incremental_source_bound_candidate",
        "unchanged_library_source_sha": SOURCE,
        "source_files": sources,
        "reused_library_sha256": library_hashes,
        "requested_compiler_jobs": 5,
        "actual_translation_units_parallel": 1,
        "commands": commands,
        "binary_sha256": sha(OUT / "strata-q36.exe"),
        "qualification": "NOT_RUN",
        "releaseQualified": False,
    }
    with (OUT / "build-receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)


if __name__ == "__main__":
    main()
