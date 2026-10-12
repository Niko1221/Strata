#!/opt/strata/.venv/bin/python

import json, os, pathlib, shutil
import setup

CUDA_MAJOR_VERSION = os.environ.get("CUDA_VERSION").split('.')[0]

CUDA_ARCHITECTURES_DEFAULT="60,61,70" if CUDA_MAJOR_VERSION == "12" else "75;80;86;89;120"
STRATA_EXPERIMENTAL_SM60 = f"-DSTRATA_EXPERIMENTAL_SM60=1" if CUDA_MAJOR_VERSION == "12" else ""
STRATA_ENGINE="engine" if CUDA_MAJOR_VERSION != "12" else "engine-cuda12"

llama = setup.get_llama_cpp()
nvcc, _ = setup.find_nvcc()
arch = os.environ.get("CUDA_ARCHITECTURES",CUDA_ARCHITECTURES_DEFAULT).strip().strip('"').replace(",", ";")
vision = "gpu" if os.environ.get("BUILD_VISION", "1") == "1" else "none"

setup.cmake_build(setup.ROOT, setup.ROOT / "build", "strata",
    [STRATA_EXPERIMENTAL_SM60, "-DSTRATA_ENABLE_CUDA=ON", "-DSTRATA_BUILD_TESTS=OFF",
     f"-DCMAKE_CUDA_ARCHITECTURES={arch}", f"-DCMAKE_CUDA_COMPILER={nvcc}",
     f"-DSTRATA_GGML_DIR={llama}"], None, "build-strata.bat")
if vision != "none":
    setup.cmake_build(setup.ROOT / "tools" / "vision", setup.ROOT / "build-vision", "strata-vision",
        [STRATA_EXPERIMENTAL_SM60, f"-DLLAMA_DIR={llama}", "-DSTRATA_VISION_CUDA=ON", "-DSTRATA_PORTABLE=OFF",
         f"-DCMAKE_CUDA_ARCHITECTURES={arch}", f"-DCMAKE_CUDA_COMPILER={nvcc}"], None, "build-vision.bat")

eng = setup.ROOT / STRATA_ENGINE

eng.mkdir(exist_ok=True)
shutil.copy2(setup.ROOT / "build" / setup.EXE, eng / setup.EXE)
if vision != "none":
    shutil.copy2(setup.ROOT / "build-vision" / "bin" / setup.VEXE, eng / setup.VEXE)
bindir = pathlib.Path(nvcc).parent
meta = {"source": "local", "version": setup.source_version(),
        "archs": [int(a.split("-")[0]) for a in arch.split(";") if a.split("-")[0].isdigit()], "vision": vision,
        "cuda_dirs": [str(d) for d in (bindir, bindir / "x64", bindir.parent / "lib64") if d.is_dir()],
        "src": setup.source_hash(setup.ENGINE_SOURCES),
        "vision_src": setup.source_hash(setup.VISION_SOURCES) if vision != "none" else None}
(eng / "BUILD.json").write_text(json.dumps(meta, indent=1))
