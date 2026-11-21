"""Check gfx90a configuration errors or the generated HIP compile commands."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1] / "engine"


class Gfx90aConfigure(unittest.TestCase):
    def test_reject_unsupported_configurations(self):
        cases = [
            ({"STRATA_HIP_GFX906": "ON"}, "STRATA_HIP_GFX906"),
            ({"STRATA_ENABLE_HIP": "ON"}, "STRATA_ENABLE_HIP"),
            ({"STRATA_ENABLE_CUDA": "ON"}, "STRATA_ENABLE_CUDA"),
            ({"STRATA_ENABLE_SYCL": "ON"}, "STRATA_ENABLE_SYCL"),
            ({"STRATA_BUILD_SPLIT": "ON"}, "STRATA_BUILD_SPLIT"),
            ({"STRATA_NATIVE_EXPERTS": "OFF"}, "STRATA_NATIVE_EXPERTS"),
            ({"CMAKE_HIP_ARCHITECTURES": "gfx906"}, "exactly one gfx90a target"),
            ({"CMAKE_HIP_ARCHITECTURES": "gfx90a;gfx906"}, "exactly one gfx90a target"),
            ({"CMAKE_HIP_ARCHITECTURES": ""}, "exactly one gfx90a target"),
            ({"CMAKE_HIP_ARCHITECTURES": "gfx90abc"}, "exactly one gfx90a target"),
        ]
        for options, diagnostic in cases:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as build:
                result = subprocess.run(
                    ["cmake", "-S", str(ROOT), "-B", build, "-DSTRATA_HIP_GFX90A=ON",
                     *(f"-D{k}={v}" for k, v in options.items())],
                    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                )
                self.assertNotEqual(result.returncode, 0, result.stdout)
                output = " ".join(result.stdout.split())
                self.assertRegex(output, r"CMake Error .*STRATA_HIP_GFX90A")
                self.assertIn(diagnostic, output)


def check_compile_commands(path):
    required = {
        "src/core/device.cu",
        "src/kernels/cuda/native_mmvq.cu",
        "src/kernels/cuda/iq_kernels.cu",
        "src/prefill/moe_fused_iq.cu",
        "src/prefill/moe_f16_kq.cu",
        "src/prefill/moe_p1a.cu",
    }
    seen = set()
    for entry in json.loads(path.read_text()):
        source = Path(entry["directory"], entry["file"]).resolve()
        if not source.is_relative_to(ROOT) or source.suffix != ".cu":
            continue
        relative = source.relative_to(ROOT).as_posix()
        args = entry.get("arguments") or shlex.split(entry["command"])
        arches = [flag for flag in args if flag.startswith("--offload-arch=")]
        defines = {flag[2:].split("=", 1)[0] for flag in args if flag.startswith("-D")}
        if arches != ["--offload-arch=gfx90a"] or not {"STRATA_HIP_GFX90A", "STRATA_HIP_WAVE64"} <= defines:
            raise SystemExit(f"{relative}: expected gfx90a Wave64 device compilation, got {arches}")
        if {"STRATA_HIP_GFX906", "STRATA_USE_HIP", "STRATA_WMMA_GFX11"} & defines:
            raise SystemExit(f"{relative}: an incompatible backend macro is defined")
        if any(flag.startswith(("-use_fast_math", "--use_fast_math", "--extended-lambda", "-Xptxas", "--generate-code"))
               for flag in args):
            raise SystemExit(f"{relative}: found NVIDIA-specific compiler options")
        seen.add(relative)
    if required - seen:
        raise SystemExit(f"Missing gfx90a build sources: {', '.join(sorted(required - seen))}")
    print(f"Checked {len(seen)} gfx90a HIP source commands including own-prompt helpers")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-commands", type=Path)
    args, unittest_args = parser.parse_known_args()
    if args.compile_commands:
        check_compile_commands(args.compile_commands)
    else:
        unittest.main(argv=[__file__, *unittest_args])
