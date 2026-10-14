#!/usr/bin/env python3
"""Package the Windows HIP engine as the ready-made zip setup.py downloads (strata-windows-x64-hip.zip).

    python tools/hip/package_windows.py --build build-hip-win --rocm <rocm-sdk path --root> \\
        --archs "gfx1100;gfx1201" --rocm-version 10.2.0a20260930 --out dist

tools/hip/build_windows.bat runs it after the build.  The zip holds:

    strata.exe, strata-device.exe, BUILD.json      (backend "hip", the archs, the ROCm and hipBLASLt versions)
    amdhip64_7.dll, amd_comgr.dll + what they import   the HIP runtime beside the exes too (#468 #461: before System32)
    rocm/bin/*.dll                                 the ROCm DLLs the two programs load (their import tables, followed
                                                   through the ROCm DLLs, + amd_comgr.dll, which the HIP runtime loads
                                                   by name) and the Microsoft C++ runtime they import
    rocm/bin/rocblas/library, rocm/bin/hipblaslt/library   the GEMM kernels, for these archs only
    rocm/.kpack/                                   rocBLAS's own device code for these archs (rocblas.dll finds it at
                                                   ../.kpack from its folder)
    rocm/licenses/                                 the shipped components' licenses, and NOTICE.txt

setup.py puts rocm/bin first on the engine's PATH (the config's lib_dirs).  The AMD display driver (AMD Software:
Adrenalin Edition) is the only thing the user installs: TheRock's HIP runtime runs on top of it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSET = "strata-windows-x64-hip.zip"
PROGRAMS = ("strata.exe", "strata-device.exe")
DYNAMIC = ("amd_comgr.dll",)                     # LoadLibrary'd by amdhip64_7.dll: not in any import table
CRT = ("msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "vcruntime140.dll", "vcruntime140_1.dll")
KPACKS = ("blas_lib",)       # rocblas.dll's own device code; rocSOLVER's (27 MB per arch) is never launched by Strata
LICENSES = ("rocblas", "rocsolver", "hipblas", "hipblas-common", "hipblaslt", "amd_comgr", "rocm-core")
NOTICE = """Strata's Windows HIP engine ships these AMD ROCm components unmodified, from AMD's TheRock Python wheels
(ROCm {rocm}, https://nightly.repo.amd.com/rocm/whl-next/, built from https://github.com/ROCm/TheRock):

{dlls}

Licenses: rocBLAS, hipBLAS, hipBLAS-common, hipBLASLt (and its TensileLite host and origami libraries), rocm-core: MIT;
rocSOLVER: BSD 2-clause; amd_comgr: Apache 2.0 with LLVM exceptions (files in this folder).  The HIP runtime
(amdhip64_7.dll, https://github.com/ROCm/rocm-systems/tree/develop/projects/clr) and rocm_kpack.dll (TheRock) are MIT
licensed; their wheels carry no license file.  The Microsoft Visual C++ runtime DLLs (msvcp140*.dll,
vcruntime140*.dll) are redistributable under the Visual Studio license (Distributable Code).
"""


def imports(objdump: Path, dll: Path) -> list[str]:
    text = subprocess.run([str(objdump), "-p", str(dll)], capture_output=True, text=True, check=True).stdout
    return [m.group(1) for m in re.finditer(r"DLL Name:\s*(\S+)", text)]


def crt_dir() -> Path | None:
    """The MSVC redist folder with the C++ runtime DLLs (vcvars sets VCToolsRedistDir)."""
    base = os.environ.get("VCToolsRedistDir")
    if not base:
        return None
    hits = sorted(Path(base, "x64").glob("Microsoft.VC*.CRT"))
    return hits[-1] if hits else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True, type=Path)
    ap.add_argument("--rocm", required=True, type=Path, help="the ROCm root (rocm-sdk path --root)")
    ap.add_argument("--archs", required=True, help="the archs the engine was compiled for, ';'-separated")
    ap.add_argument("--rocm-version", required=True)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    archs = [x for x in re.split(r"[;, ]+", a.archs) if x]
    rbin = a.rocm / "bin"
    objdump = a.rocm / "lib" / "llvm" / "bin" / "llvm-objdump.exe"
    stage = a.out / "strata-windows-x64-hip"
    shutil.rmtree(stage, ignore_errors=True)
    (stage / "rocm" / "bin").mkdir(parents=True)

    for p in PROGRAMS:
        shutil.copy2(a.build / p, stage / p)

    # the ROCm DLLs: the programs' imports, followed through the ROCm DLLs themselves
    rocm_dlls = {p.name.lower(): p for p in rbin.glob("*.dll")}
    need, todo, crt = [], [stage / p for p in PROGRAMS] + [rbin / d for d in DYNAMIC], set()
    while todo:
        f = todo.pop()
        for name in imports(objdump, f):
            low = name.lower()
            if low in CRT:
                crt.add(low)
            elif low in rocm_dlls and low not in need:
                need.append(low)
                todo.append(rocm_dlls[low])
    for d in DYNAMIC:
        if d not in need:
            need.append(d)
    for d in sorted(need):
        shutil.copy2(rocm_dlls[d], stage / "rocm" / "bin" / rocm_dlls[d].name)
    cdir = crt_dir()
    if cdir is None:
        sys.exit("VCToolsRedistDir is not set: run this from a Visual Studio x64 developer prompt (vcvars64.bat)")
    for d in sorted(crt):
        shutil.copy2(cdir / d, stage / "rocm" / "bin" / d)
    # #468 #461: the HIP runtime (and the compiler it loads by name) next to the exes as well - Windows searches the
    # exe's folder before System32, where an AMD driver may install its own amdhip64_7.dll (found before PATH's
    # rocm/bin); rocBLAS/hipBLASLt stay in rocm/bin, which they resolve their kernel libraries and ../.kpack from
    # #461: and everything those two import (amdhip64_7.dll needs rocm_kpack.dll and the C++ runtime, or its full-path
    # load fails with error 126 and Windows falls back to System32's copy): the closure of their PE import tables
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dll_closure import dll_closure
    roots = sorted(f.name for pat in ("amdhip64_*.dll", "amd_comgr*.dll") for f in (stage / "rocm" / "bin").glob(pat))
    for name in dll_closure(stage / "rocm" / "bin", roots):
        shutil.copy2(stage / "rocm" / "bin" / name, stage / name)

    # the GEMM kernels and the libraries' own device code, for these archs
    lib = stage / "rocm" / "bin" / "rocblas" / "library"
    lib.mkdir(parents=True)
    for f in (rbin / "rocblas" / "library").iterdir():
        m = re.search(r"gfx[0-9a-f]+", f.name)
        if m is None or m.group() in archs:
            if f.is_dir():                                # the gfx1151 wheel keeps its rocBLAS kernels in library/gfx1151/
                shutil.copytree(f, lib / f.name)
            else:
                shutil.copy2(f, lib / f.name)
    for arch in archs:
        src = rbin / "hipblaslt" / "library" / arch
        if src.is_dir():                                  # hipBLASLt has no RDNA2 (gfx1030) kernels
            shutil.copytree(src, stage / "rocm" / "bin" / "hipblaslt" / "library" / arch)
    kp = stage / "rocm" / ".kpack"
    kp.mkdir()
    kpack_src = a.rocm / ".kpack"
    for arch in archs:
        for k in KPACKS:
            f = kpack_src / f"{k}_{arch}.kpack"
            if not f.exists():
                sys.exit(f"missing {f}: install the rocm-sdk-device-{arch} wheel and run rocm-sdk init")
            shutil.copy2(f, kp / f.name)

    lic = stage / "rocm" / "licenses"
    lic.mkdir()
    for comp in LICENSES:
        for f in (a.rocm / "share" / "doc" / comp).glob("LICENSE*"):
            shutil.copy2(f, lic / f"{comp}-{f.name}")
    shipped = sorted(p.name for p in (stage / "rocm" / "bin").glob("*.dll"))
    (lic / "NOTICE.txt").write_text(NOTICE.format(rocm=a.rocm_version, dlls="\n".join("  " + d for d in shipped)),
                                    encoding="utf-8")

    version = re.search(r"project\(strata VERSION ([\d.]+)", (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")).group(1)
    hl = (a.rocm / "include" / "hipblaslt" / "hipblaslt-version.h").read_text(encoding="utf-8")
    hlv = [int(re.search(rf"#define\s+HIPBLASLT_VERSION_{k}\s+(\d+)", hl).group(1)) for k in ("MAJOR", "MINOR", "PATCH")]
    meta = {"source": "prebuilt", "backend": "hip", "platform": "windows-x64", "version": version, "archs": archs,
            "rocm": a.rocm_version, "hipblaslt_version": hlv[0] * 100000 + hlv[1] * 100 + hlv[2], "vision": "none",
            "lib_dirs": ["rocm/bin"]}
    (stage / "BUILD.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    z = a.out / ASSET
    z.unlink(missing_ok=True)
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as f:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                f.write(p, p.relative_to(stage).as_posix())
    size = sum(p.stat().st_size for p in stage.rglob("*") if p.is_file())
    print(f"{z}: {z.stat().st_size / 2**20:.0f} MiB ({size / 2**20:.0f} MiB unpacked), engine {version}, "
          f"{', '.join(archs)}, ROCm {a.rocm_version}, DLLs: {', '.join(shipped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
