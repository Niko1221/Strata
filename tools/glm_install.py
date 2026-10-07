"""GLM's advanced installer path; reuses setup.py's resumable downloader and build helpers."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile

from tools import glm_setup

REPO = "Justvugg/GLM-5.3-colibri-int4-g64"
MIN_RAM_GIB = 60  # a nominal 64 GB machine reports about 63 GiB after hardware reservations


def model_files(metadata):
    """Use one immutable revision and accept only model assets at the repository root."""
    revision = metadata.get("sha", "")
    if not re.fullmatch(r"[0-9a-f]{40,64}", revision):
        raise ValueError("model metadata has no immutable revision")
    files = []
    for entry in metadata.get("siblings", []):
        name = entry.get("rfilename", "")
        if "/" in name or "\\" in name or Path(name).name != name:
            continue
        if name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "LICENSE") or name.endswith(".safetensors"):
            size = entry.get("size") or entry.get("lfs", {}).get("size")
            if not isinstance(size, int) or size <= 0:
                raise ValueError(f"missing file size for {name}")
            files.append((name, size))
    names = {f for f, _ in files}
    if not {"config.json", "tokenizer.json"} <= names or not any(n.endswith(".safetensors") for n in names):
        raise ValueError("model repository is missing configuration, tokenizer or weights")
    return revision, files


def bytes_needed(folder, files):
    needed = 0
    for name, size in files:
        final, part = folder / name, folder / (name + ".part")
        if final.is_file() and final.stat().st_size == size:
            continue
        have = min(part.stat().st_size, size) if part.is_file() else 0
        needed += size - have
    return needed


def check_resources(ram, free_bytes, needed):
    if ram < MIN_RAM_GIB:
        raise ValueError("GLM-5.3 setup requires a PC with 64 GB of RAM")
    if free_bytes < needed + (1 << 30):
        raise ValueError(f"GLM-5.3 needs {needed / 1e9:.1f} GB more model space plus 1 GB of working space; "
                         f"{free_bytes / 1e9:.1f} GB is free")


def fetch_model(folder, setup):
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / ".strata-download.json"
    if manifest.exists():
        meta = json.loads(manifest.read_text(encoding="utf-8"))
    else:
        with urllib.request.urlopen(f"https://huggingface.co/api/models/{REPO}?blobs=true", timeout=60) as r:
            meta = json.load(r)
    revision, files = model_files(meta)
    check_resources(setup.ram_gb(), shutil.disk_usage(folder).free, bytes_needed(folder, files))
    manifest.write_text(json.dumps(meta), encoding="utf-8")
    for name, size in files:
        dst = folder / name
        if not dst.is_file() or dst.stat().st_size != size:
            # A stale completion marker must never make an incomplete shard look finished.
            dst.with_name(dst.name + ".done").unlink(missing_ok=True)
            setup.download(f"https://huggingface.co/{REPO}/resolve/{revision}/{name}", dst)
        if dst.stat().st_size != size:
            raise ValueError(f"{name}: downloaded size does not match the pinned model revision")
    glm_setup.check_model(folder)
    return folder


def cublas_root(nvcc, version, setup):
    """An optional compiler-only Windows CUDA install can use NVIDIA's checksum-verified cuBLAS component."""
    given = os.environ.get("STRATA_CUBLAS_ROOT")
    if given:
        return Path(given).resolve()
    toolkit = Path(nvcc).resolve().parent.parent
    if (toolkit / "include/cublas_v2.h").exists() or os.name != "nt":
        return None
    base = "https://developer.download.nvidia.com/compute/cuda/redist/"
    with urllib.request.urlopen(base + f"redistrib_{version[0]}.{version[1]}.0.json", timeout=60) as r:
        component = json.load(r)["libcublas"]["windows-x86_64"]
    folder = setup.ROOT / "third_party/glm-cublas"
    archive = folder / Path(component["relative_path"]).name
    setup.download(base + component["relative_path"], archive, "NVIDIA cuBLAS build and runtime libraries")
    digest = hashlib.sha256()
    with archive.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != component["sha256"]:
        raise ValueError("NVIDIA cuBLAS checksum mismatch")
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            target = (folder / name).resolve()
            if not target.is_relative_to(folder.resolve()):
                raise ValueError("invalid cuBLAS archive path")
        z.extractall(folder)
    return next(folder.glob("*/include/cublas_v2.h")).parent.parent


def build_engine(setup, cuda, force=False):
    try:
        existing = glm_setup.find_exe(None)
    except SystemExit:
        existing = None
    if existing and not force:
        try:
            features = json.loads(subprocess.check_output([str(existing), "--features"], text=True, stderr=subprocess.DEVNULL))
            if not cuda or features.get("cuda"):
                return existing
        except (OSError, ValueError, subprocess.CalledProcessError):
            pass
    defs = ["-DSTRATA_NATIVE_EXPERTS=OFF", "-DSTRATA_BUILD_TESTS=OFF", f"-DSTRATA_ENABLE_CUDA={'ON' if cuda else 'OFF'}"]
    vcvars = setup.find_vcvars() if setup.WIN else None
    libroot = None
    if cuda:
        nvcc, version = setup.find_nvcc()
        if not nvcc:
            raise ValueError("CUDA build requested but nvcc was not found; install the CUDA Toolkit or use --backend cpu")
        libroot = cublas_root(nvcc, version, setup)
        defs += [f"-DCMAKE_CUDA_COMPILER={nvcc}", "-DCMAKE_CUDA_ARCHITECTURES=native"]
        if libroot:
            defs += [f"-DCMAKE_PREFIX_PATH={libroot}"]
    bdir = setup.ROOT / ("build-glm-cuda" if cuda else "build-glm-cpu")
    setup.cmake_build(setup.ROOT, bdir, "strata-glm", defs, vcvars, "build-glm.bat")
    exe = bdir / ("strata-glm.exe" if setup.WIN else "strata-glm")
    if cuda and setup.WIN:
        roots = [libroot] if libroot else [Path(nvcc).resolve().parent.parent]
        for root in roots:
            for dll in root.rglob("cublas*64_*.dll"):
                shutil.copy2(dll, bdir / dll.name)
    return exe


def install(a, setup):
    if a.gpus or a.parallel not in (None, 1) or a.vision not in (None, "no", "none"):
        raise ValueError("GLM-5.3 supports one GPU, one request at a time and text input")
    ram = setup.ram_gb()
    if ram < MIN_RAM_GIB:
        raise ValueError("GLM-5.3 setup requires a PC with 64 GB of RAM")
    if a.backend not in (None, "cpu", "cuda"):
        raise ValueError("GLM-5.3 supports CPU and CUDA backends")
    cuda = a.backend == "cuda" or a.gpu is not None
    if a.backend == "cpu":
        cuda = False
    setup.say("GLM-5.3 (advanced): 419 GB of weights on an NVMe SSD; CPU with optional CUDA acceleration")
    if a.check:
        setup.say(f"RAM: {ram:.1f} GiB; backend: {'CUDA' if cuda else 'CPU'}")
        return 0
    setup.pip_install(setup.requirement_lines(), "Strata Python dependencies")
    exe = build_engine(setup, cuda, force=a.build)
    if a.glm_model_dir:
        model = Path(a.glm_model_dir).resolve()
        glm_setup.check_model(model)
    else:
        data, _ = setup.data_folder(a.data_dir)
        model = Path(a.models_dir) if a.models_dir else data / "models"
        model = fetch_model(model / "GLM-5.3-colibri-int4-g64", setup)
    cfg, script = glm_setup.prepare(model, exe, max_context=a.context or 8192, port=a.port or 8080,
                                    kv=a.glm_kv, gpu=int(a.gpu or 0) if cuda else None,
                                    host=a.host or "127.0.0.1", api_key=a.api_key,
                                    open_browser=a.browser is not False)
    setup.say(f"Ready: {script.name}; http://127.0.0.1:{a.port or 8080}")
    return 0 if a.no_start else glm_setup.start_config(cfg)
