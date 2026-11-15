"""Compile and execute the focused CPU contracts under check.py's owned Job."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("H:/Strata/evidence/ram32gb-pr00-20261115/cpu-final")


def main() -> None:
    OUT.mkdir()
    result = subprocess.run(
        [
            os.environ["COMSPEC"],
            "/d",
            "/u",
            "/c",
            str(Path(__file__).with_name("vs-env.cmd")),
        ],
        check=True,
        capture_output=True,
    )
    env = dict(os.environ)
    env.update(
        {
            key.upper(): value
            for key, value in (
                line.split("=", 1)
                for line in result.stdout.decode("utf-16le").splitlines()
                if "=" in line and not line.startswith("=")
            )
        }
    )
    compiler = shutil.which("cl.exe", path=env["PATH"])
    if not compiler:
        raise RuntimeError("MSVC not available in the explicit developer environment")
    with (OUT / "compiler-environment.json").open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "command": [compiler],
                "environment": {k: env[k] for k in ("INCLUDE", "LIB")},
            },
            stream,
            indent=2,
        )
    for name in (
        "q36_phase_trace_test",
        "q36_expert_store_test",
        "q36_transfer_lifetime_test",
        "q36_bounded_experts_test",
        "q36_kv_layout_test",
    ):
        sources = [str(ROOT / "tests/qwen35" / (name + ".cpp"))]
        libs = []
        if name in ("q36_expert_store_test", "q36_transfer_lifetime_test"):
            sources.append(str(ROOT / "src/qwen35/expert_store.cpp"))
            libs = ["bcrypt.lib", "crypt32.lib"]
        subprocess.run(
            [
                compiler,
                "/nologo",
                "/std:c++20",
                "/EHsc",
                "/MT",
                "/O2",
                "/W4",
                "/I" + str(ROOT / "include"),
                *sources,
                "/Fo" + str(OUT) + "\\",
                "/Fe" + str(OUT / (name + ".exe")),
                *libs,
            ],
            cwd=OUT,
            env=env,
            check=True,
        )
        subprocess.run([str(OUT / (name + ".exe"))], cwd=OUT, env=env, check=True)


if __name__ == "__main__":
    main()
