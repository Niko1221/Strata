"""Build the optional Hy4-Flash reference operators against a configured CUDA llama.cpp build."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("llama_build", type=Path, help="build directory with compile_commands.json")
    parser.add_argument("output", type=Path, help="path to the output shared library")
    args = parser.parse_args()
    build = args.llama_build.resolve()
    output = args.output.resolve()
    source = Path(__file__).with_name("ggml_reference_ops.cu").resolve()
    entries = json.loads((build / "compile_commands.json").read_text())
    entry = next((row for row in entries if Path(row["file"]).name == "fattn.cu"), None)
    if not entry:
        parser.error("the compilation database must contain the ggml CUDA Flash Attention target")
    command = entry.get("arguments") or shlex.split(entry["command"])
    compiler = command[0]
    command[command.index("-c") + 1] = str(source)
    with tempfile.TemporaryDirectory() as tmp:
        obj = Path(tmp) / "reference_ops.o"
        command[command.index("-o") + 1] = str(obj)
        subprocess.run(command, cwd=entry["directory"], check=True)
        lib = build / "bin"
        subprocess.run([compiler, "-shared", "--cudart=shared", "-o", output, obj, f"-L{lib}", "-Xlinker", "-rpath",
                        "-Xlinker", str(lib), "-lggml-cuda", "-lggml-base", "-lcublas"], check=True)


if __name__ == "__main__":
    main()
