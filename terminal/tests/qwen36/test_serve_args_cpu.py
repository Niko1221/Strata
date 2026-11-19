"""Compile and execute Q36's verbatim argument parser without model/GPU code.

This is a CPU parser contract, not a build of q36_engine.cu or engine runtime
qualification. On Windows set Q36_CPU_COMPILE_RECEIPT to a local MSVC receipt
containing command[0] and environment INCLUDE/LIB, or use a developer prompt.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def parser_exe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    source = (ROOT / "src/qwen35/q36_engine.cu").read_text(encoding="utf-8")
    run = source[source.index("int strata::qwen35::run(int argc, char** argv) {"):]
    # Keep the parser and its post-parse guards intact. Stop before environment
    # setup/model loading; the fixture reports values that would reach the engine.
    parser = run[run.index("    std::string model, tokens,"):run.index("#if !defined(_WIN32)")]
    folder = tmp_path_factory.mktemp("q36-parser-cpu")
    cpp = folder / "parser.cpp"
    cpp.write_text(
        "#include <string>\n#include <cstdio>\n#include <cstdlib>\n"
        "#include <cerrno>\n#include <cctype>\n#include <limits>\n"
        '#include "strata/qwen35/bounded_experts.hpp"\n'
        "using strata::qwen35::q36_explicit_slots;\n"
        "using strata::qwen35::q36_memory_mode_error;\n"
        "int main(int argc, char** argv) {\n" + parser +
        'std::printf("CPU_PARSER k=%s v=%s expert_vram=%.1f\\n", '
        "cache_type_k.c_str(), cache_type_v.c_str(), expert_vram);\nreturn 0;\n}\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    receipt_path = env.get("Q36_CPU_COMPILE_RECEIPT")
    if receipt_path:
        receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
        compiler = receipt["command"][0]
        env.update({key: receipt["environment"][key] for key in ("INCLUDE", "LIB")})
        env["PATH"] = str(Path(compiler).parent) + os.pathsep + env.get("PATH", "")
    else:
        compiler = env.get("CXX") or shutil.which("cl") or shutil.which("c++")
    if not compiler:
        pytest.skip("CPU C++ compiler is not configured")
    exe = folder / ("parser.exe" if os.name == "nt" else "parser")
    if Path(compiler).name.lower() in ("cl", "cl.exe"):
        command = [compiler, "/nologo", "/std:c++17", "/EHsc", "/W4", "/I" + str(ROOT / "include"), str(cpp),
                   "/Fe:" + str(exe), "/Fo:" + str(folder / "parser.obj"), "/link", "/INCREMENTAL:NO"]
    else:
        command = [compiler, "-std=c++17", "-Wall", "-Wextra", "-I" + str(ROOT / "include"), str(cpp), "-o", str(exe)]
    (folder / "command.json").write_text(json.dumps({
        "command": command, "scope": "CPU verbatim parser only; no native engine/model/GPU",
        "parser_sha256": hashlib.sha256(parser.encode()).hexdigest(),
    }, indent=2), encoding="utf-8")
    result = subprocess.run(command, cwd=folder, env=env, capture_output=True, timeout=60)
    (folder / "compile.txt").write_bytes(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout.decode(errors="replace") + result.stderr.decode(errors="replace")
    return exe


def parse(exe: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(exe), "--model", "fixture-never-opened.gguf", *args],
                          capture_output=True, text=True, timeout=5)


@pytest.mark.parametrize("flag", [
    ["--vram-reserve-mib", "1024"], ["--vram-reserve-mib=1024"],
    ["--vram-reserve-mib"], ["--vram-reserve-mib="],
])
@pytest.mark.parametrize("position", ["before-serve", "after-serve", "no-serve"])
def test_unsupported_reserve_is_never_silently_ignored(parser_exe: Path, flag: list[str], position: str) -> None:
    args = flag + ["--serve"] if position == "before-serve" else ["--serve", *flag] if position == "after-serve" else flag
    result = parse(parser_exe, args)
    assert result.returncode == 2
    assert "It was not applied" in result.stderr
    assert "--expert-vram" in result.stderr
    assert "ignoring" not in result.stderr
    assert "CPU_PARSER" not in result.stdout


@pytest.mark.parametrize("key,value", [("f32", "f32"), ("f16", "f16"), ("f32", "f16"), ("f16", "f32")])
@pytest.mark.parametrize("serve_first", [True, False])
def test_supported_kv_formats_still_reach_engine_settings(parser_exe: Path, key: str, value: str, serve_first: bool) -> None:
    options = ["--cache-type-k", key, "--cache-type-v", value, "--expert-vram", "3.5"]
    result = parse(parser_exe, ["--serve", *options] if serve_first else [*options, "--serve"])
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"CPU_PARSER k={key} v={value} expert_vram=3.5"
    assert not result.stderr


def test_defaults_and_compatibility_fallback_stay_intact(parser_exe: Path) -> None:
    result = parse(parser_exe, ["--serve", "--compat-fixture", "value"])
    assert result.returncode == 0
    assert "k=f32 v=f32" in result.stdout
    assert "ignoring --compat-fixture value" in result.stderr


@pytest.mark.parametrize("flag", ["--cache-type-k", "--cache-type-v"])
def test_unsupported_kv_format_still_fails(parser_exe: Path, flag: str) -> None:
    result = parse(parser_exe, ["--serve", flag, "q4_0"])
    assert result.returncode == 2
    assert "accepts f32 or f16" in result.stderr


def test_reserve_refusal_precedes_model_requirement(parser_exe: Path) -> None:
    result = subprocess.run([str(parser_exe), "--serve", "--vram-reserve-mib", "1024"],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 2
    assert "It was not applied" in result.stderr
    assert "--model is required" not in result.stderr
