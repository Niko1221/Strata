"""CPU-only proof and architecture-path comparison; never calls CUDA runtime."""
import argparse
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
BASELINE = "82f46a8c8f475f001ad76d92f58f4a4f8ffb0253"
parser = argparse.ArgumentParser()
parser.add_argument("--cxx", default="clang++")
parser.add_argument("--baseline", default=BASELINE, help="Pristine reference commit for architecture-path checks")
args = parser.parse_args()

with tempfile.TemporaryDirectory(prefix="p100-dp4a-offline-") as folder:
    temp = pathlib.Path(folder)
    binary = temp / "dp4a-model"
    subprocess.run([args.cxx, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                    "-fsanitize=undefined", "-fno-sanitize-recover=undefined", str(ROOT / "tests/p100/dp4a_model.cpp"),
                    "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
    original = subprocess.check_output(["git", "show", f"{args.baseline}:include/strata/kernels/dp4a.hpp"], cwd=ROOT)
    candidate = (ROOT / "include/strata/kernels/dp4a.hpp").read_bytes()
    pristine_header = temp / "pristine.hpp"
    pristine_header.write_bytes(original)
    scalar_binary = temp / "scalar-reference"
    subprocess.run([args.cxx, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                    "-fsanitize=undefined", "-fno-sanitize-recover=undefined", "-D__CUDA_ARCH__=500", "-D__device__=",
                    "-D__forceinline__=inline", "-DP100_SCALAR_REFERENCE=1", "-include", str(pristine_header),
                    str(ROOT / "tests/p100/dp4a_model.cpp"), "-o", str(scalar_binary)], check=True)
    subprocess.run([str(scalar_binary)], check=True)
    def preprocess(source, arch):
        header = temp / "probe.hpp"
        header.write_bytes(source)
        wrapper = temp / "wrapper.cpp"
        wrapper.write_text('#include "probe.hpp"\n')
        command = [args.cxx, "-E", "-P", "-x", "c++", str(wrapper)]
        if arch is not None:
            command.append(f"-D__CUDA_ARCH__={arch}")
        return subprocess.check_output(command)
    for arch in (None, 500, 601, 609, 610, 700, 750, 800, 900, 1200):
        if preprocess(original, arch) != preprocess(candidate, arch):
            raise SystemExit(f"Unexpected source-path change for architecture {arch}")
    sm60 = preprocess(candidate, 600).decode()
    if sm60.count("vmad.s32.s32.s32") != 4 or '"=r"(result)' not in sm60:
        raise SystemExit("GP100 source does not select the four VMAD instructions")
    print("PASS: all tested non-sm_60 preprocessed paths identical; sm_60 selects VMAD")
