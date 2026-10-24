"""Build the engine from another checkout (a worktree) with a chosen CUDA toolkit.

usage: build_in_tree.py <root> <nvcc.exe> [expected major.minor]
"""
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
NVCC = sys.argv[2]
WANT = tuple(int(x) for x in sys.argv[3].split(".")) if len(sys.argv) > 3 else None

os.environ["STRATA_NVCC"] = NVCC
sys.path.insert(0, str(ROOT))

import setup as S  # noqa: E402

gpu = S.gpu_info()
nvcc, cuda_v = S.find_nvcc()
print(f"tree: {ROOT}  (engine version {S.source_version()})", flush=True)
print(f"nvcc: {nvcc}  CUDA {cuda_v}  build {S.nvcc_build(nvcc)}", flush=True)
if WANT and cuda_v != WANT:
    raise SystemExit(f"expected CUDA {WANT}, found {cuda_v}")

for folder in ("build", "build-vision", "build-cuda12", "build-vision-cuda12"):
    path = ROOT / folder
    if path.exists():
        print(f"removing {folder} (another compiler was configured there)", flush=True)
        shutil.rmtree(path, ignore_errors=True)

llama = S.get_llama_cpp()
eng = S.build_engine(gpu, "gpu", True, llama, toolkit=13)
print("engine folder:", eng, flush=True)
print("BUILD.json:", (eng / "BUILD.json").read_text(encoding="utf-8"), flush=True)
print("engine version:", S.engine_version_of(eng), flush=True)
print("DONE", flush=True)
