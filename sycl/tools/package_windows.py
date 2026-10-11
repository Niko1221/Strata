#!/usr/bin/env python3
"""Put the Windows SYCL engine into engine\\: the programs, the oneAPI runtime DLLs beside them, and BUILD.json.

    python sycl\\tools\\package_windows.py --build build-sycl-win [--oneapi "C:\\Program Files (x86)\\Intel\\oneAPI"]
        [--engine engine] [--aot bmg-g31] [--bundle-ze-loader]

sycl\\tools\\build_windows.bat runs it after the build, from the vcvars + setvars environment (icx and dumpbin on PATH).
engine\\ then holds:

    strata.exe, strata-device.exe              the engine and the device probe (strata-device --list-devices)
    doorbell_probe.exe alloc_probe.exe
    bw_probe.exe graph_probe.exe               the sycl\\probe programs, those that were built
    sycl9.dll, ur_*.dll, mkl_*.dll, ...        the oneAPI runtime the programs load: the set llama.cpp copies beside its
                                               Windows SYCL binaries (.github/workflows/release.yml, windows-sycl, oneAPI
                                               2026.1.1) with oneMKL's sequential threading layer in place of the TBB one,
                                               plus whatever else their import tables name from the oneAPI folders;
                                               ze_loader.dll only with --bundle-ze-loader (the Arc driver 32.0.101.9034+
                                               ships Level Zero loader 1.32 itself)
    BUILD.json                                 backend "sycl", source "local-sycl", the version, the AOT device, the
                                               compiler version, and the source hash setup.py compares on start (#31)

Windows searches a program's own folder before PATH and System32, so these DLLs are the ones that load whatever else is
installed.  setup.py reads BUILD.json ("backend": "sycl"); the server puts engine\\ first on the engine's PATH.
Exit code 1 with the reason when a program or one of the listed DLLs is not there.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROGRAMS = ("strata.exe", "strata-device.exe")
PROBES = ("doorbell_probe.exe", "alloc_probe.exe", "bw_probe.exe", "graph_probe.exe")
# llama.cpp's windows-sycl release job (.github/workflows/release.yml, oneAPI 2026.1.1), plus mkl_sequential.3.dll because the
# engine links the sequential oneMKL (-qmkl=sequential / /Qmkl:sequential).  mkl_tbb_thread.3.dll stays on the list:
# mkl_sycl_blas.6.dll imports it by name on oneAPI 2026.0 (the B70 PC's first build, 2026-10-04, found it through the
# import-table walk below; a run without dumpbin would have missed it).  dnnl.dll and sycl-ls.exe left out (not used).
# mkl_def / mkl_avx2 are loaded by mkl_core by name, not imported, so the import tables below cannot find them: they
# have to be listed.
RUNTIME_DLLS = (
    "sycl9.dll", "ur_loader.dll", "ur_adapter_level_zero.dll", "ur_adapter_level_zero_v2.dll", "ur_adapter_opencl.dll",
    "ur_win_proxy_loader.dll",
    "mkl_sycl_blas.6.dll", "mkl_core.3.dll", "mkl_def.3.dll", "mkl_avx2.3.dll", "mkl_sequential.3.dll",
    "mkl_tbb_thread.3.dll",
    "svml_dispmd.dll", "libmmd.dll", "libiomp5md.dll", "tbb12.dll", "tcm.dll", "umf.dll", "libhwloc-15.dll",
)
ZE_LOADER = "ze_loader.dll"
# where the install keeps them ("latest" is a junction to the installed version; resolved so the messages name the
# real folder)
DLL_DIRS = (("compiler", "latest", "bin"), ("mkl", "latest", "bin"), ("tbb", "latest", "bin"), ("umf", "latest", "bin"),
            ("tcm", "latest", "bin"))
# the SYCL architecture the engine checks at start (info::device::architecture) for the ocloc device the code was
# built for; a JIT build ("") is for the same card
ARCH_OF = {"": "intel_gpu_bmg_g31", "bmg-g31": "intel_gpu_bmg_g31", "bmg-g21": "intel_gpu_bmg_g21",
           "bmg": "intel_gpu_bmg_g21"}
# imports that are Windows' own, the MSVC runtime, or the graphics driver's (OpenCL.dll, ze_loader.dll): not ours to ship.
# Name prefixes, matched without case (d3d covers d3d11 / d3d12 / d3dcompiler, api-ms- the api-ms-win-* set).  The first
# packaging on the B70 PC (2026-10-04) warned that imagehlp.dll, imported by libiomp5md.dll, was "neither Windows' own
# nor in the oneAPI folders": it is Windows' own (System32), this list just did not name it.  Added with it, the other
# system DLLs a dumpbin walk of the oneAPI runtime can name: ncrypt, propsys, uxtheme, dwmapi, opengl32.
SYSTEM_DLLS = re.compile(r"^(api-ms-|ext-ms-|kernel32|kernelbase|user32|advapi32|ole32|oleaut32|shell32|shlwapi|ws2_32|"
                         r"bcrypt|ncrypt|crypt32|msvcp|vcruntime|ucrtbase|msvcrt|ntdll|rpcrt4|gdi32|dxgi|d3d|opengl32|"
                         r"setupapi|cfgmgr32|version|iphlpapi|psapi|dbghelp|imagehlp|userenv|winmm|imm32|comdlg32|"
                         r"comctl32|wtsapi32|secur32|netapi32|powrprof|propsys|uxtheme|dwmapi|synchronization|wintrust|"
                         r"opencl|ze_loader)", re.I)


def find_exe(build: Path, name: str) -> Path | None:
    """Ninja puts a target's exe in its directory's build folder: the root for strata and strata-device, probe\\ for the
    probes (unless sycl/probe sets RUNTIME_OUTPUT_DIRECTORY); look there, then anywhere under the build tree."""
    for d in (build, build / "probe", build / "bin"):
        if (d / name).is_file():
            return d / name
    return next(build.rglob(name), None)


def oneapi_root(given: Path) -> Path:
    """--oneapi / ONEAPI_ROOT when it is an install; else the install the icx on PATH belongs to
    (<root>\\compiler\\<version>\\bin\\icx.exe)."""
    if (given / "compiler").is_dir():
        return given
    icx = shutil.which("icx")
    if icx:
        try:
            root = Path(icx).resolve().parents[3]
        except IndexError:
            return given
        if (root / "compiler").is_dir():
            return root
    return given


def imports(dumpbin: str, f: Path) -> list[str]:
    """The DLLs a PE file's import table names (dumpbin /dependents, from the Build Tools; delay-loaded ones included)."""
    r = subprocess.run([dumpbin, "/nologo", "/dependents", str(f)], capture_output=True, text=True)
    return re.findall(r"^\s+(\S+\.dll)\s*$", r.stdout, re.M | re.I)


def icx_version(root: Path) -> str:
    """The first line of `icx --version`, e.g. "Intel(R) oneAPI DPC++/C++ Compiler 2026.1.1 (2026.1.1.20260xxx)"."""
    exe = shutil.which("icx") or str(root / "compiler" / "latest" / "bin" / "icx.exe")
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=120).stdout
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return next((ln.strip() for ln in out.splitlines() if ln.strip()), "unknown")


def copy(src: Path, dst: Path) -> None:
    try:
        shutil.copy2(src, dst)
    except PermissionError:
        sys.exit(f"cannot write {dst}: a running engine or server still has it open; stop it and run this again")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True, type=Path, help="the CMake build tree (build-sycl-win)")
    ap.add_argument("--oneapi", type=Path,
                    default=Path(os.environ.get("ONEAPI_ROOT") or r"C:\Program Files (x86)\Intel\oneAPI"),
                    help="the oneAPI install (default: ONEAPI_ROOT, else the usual folder, else the icx on PATH's)")
    ap.add_argument("--engine", type=Path, default=ROOT / "engine", help="where the engine goes (default: <repo>\\engine)")
    ap.add_argument("--aot", default="bmg-g31",
                    help='the STRATA_SYCL_AOT device the build was configured with ("" for a JIT build)')
    ap.add_argument("--bundle-ze-loader", action="store_true",
                    help="also copy oneAPI's ze_loader.dll (for a graphics driver older than 32.0.101.9034)")
    a = ap.parse_args()

    root = oneapi_root(a.oneapi)
    dirs = [d.resolve() for d in (root.joinpath(*p) for p in DLL_DIRS) if d.is_dir()]
    if not dirs:
        sys.exit(f"no oneAPI runtime folders under {root} (expected compiler\\latest\\bin, mkl\\latest\\bin, ...): install "
                 "the oneAPI Base Toolkit 2026.0.0 or newer, or point ONEAPI_ROOT / --oneapi at the install")
    available: dict[str, Path] = {}
    for d in dirs:
        for f in d.glob("*.dll"):
            available.setdefault(f.name.lower(), f)

    a.engine.mkdir(parents=True, exist_ok=True)
    programs, not_built = [], []
    for p in PROGRAMS + PROBES:
        src = find_exe(a.build, p)
        if src is None:
            if p in PROGRAMS:
                sys.exit(f"{p} not found under {a.build}: build the targets strata and strata-device first "
                         "(sycl\\tools\\build_windows.bat)")
            not_built.append(p)
            continue
        copy(src, a.engine / p)
        programs.append(p)

    # the runtime: the fixed list first (an error when the install lacks one), then what the import tables add
    want = list(RUNTIME_DLLS) + ([ZE_LOADER] if a.bundle_ze_loader else [])
    missing = [n for n in want if n.lower() not in available]
    if missing:
        sys.exit("oneAPI runtime DLLs not found in " + ", ".join(str(d) for d in dirs) + ": " + ", ".join(missing)
                 + " - the oneAPI Base Toolkit 2026.0.0+ (DPC++/C++ compiler, oneMKL, TBB) has them; is ONEAPI_ROOT the "
                 "install they belong to?")
    dlls: list[str] = []
    for n in want:
        copy(available[n.lower()], a.engine / available[n.lower()].name)
        dlls.append(available[n.lower()].name)
    extra, unresolved = [], []
    dumpbin = shutil.which("dumpbin")
    if dumpbin:
        seen = {n.lower() for n in dlls}
        todo = [a.engine / n for n in programs + dlls]
        while todo:
            f = todo.pop()
            for name in imports(dumpbin, f):
                low = name.lower()
                if low in seen or SYSTEM_DLLS.match(low):
                    continue
                seen.add(low)
                if low in available:
                    copy(available[low], a.engine / available[low].name)
                    dlls.append(available[low].name)
                    extra.append(f"{available[low].name} (imported by {f.name})")
                    todo.append(a.engine / available[low].name)
                else:
                    unresolved.append(f"{name} (imported by {f.name})")

    # BUILD.json, the keys setup.py reads for a ready-made engine, with "backend": "sycl"
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "sycl"))
    import setup  # noqa: E402  (setup.py: source_hash)
    import setup_intel  # noqa: E402  (sycl/setup_intel.py: sycl_sources(), the parts the hash covers - the list setup reads back with)
    # the version is the top-level project's (sycl/CMakeLists.txt reads it from ../CMakeLists.txt), written plain so
    # that setup.py's engine_version / MIN_ENGINE comparison reads it as it reads every other engine's
    version = setup_intel.sycl_version()
    meta = {"backend": "sycl", "source": "local-sycl", "version": version, "aot": a.aot,
            "archs": [ARCH_OF.get(a.aot, "intel_gpu_" + a.aot.replace("-", "_"))], "oneapi": icx_version(root),
            "src": setup.source_hash(setup_intel.sycl_sources()), "lib_dirs": [],
            "built": datetime.date.today().isoformat()}
    (a.engine / "BUILD.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    print(f"{a.engine}: {', '.join(programs)}")
    print(f"  oneAPI runtime from {', '.join(str(d) for d in dirs)}: {', '.join(dlls)}")
    for e in extra:
        print(f"  also copied, named in an import table: {e}")
    if not a.bundle_ze_loader:
        print("  ze_loader.dll: the graphics driver's (32.0.101.9034 and newer carry Level Zero 1.32); "
              "--bundle-ze-loader copies oneAPI's instead")
    if not dumpbin:
        print("  (dumpbin is not on PATH, so the import tables were not checked; a Visual Studio x64 developer prompt has it)")
    for u in unresolved:
        print(f"  WARNING: {u} is neither Windows' own nor in the oneAPI folders; the program may not start without it")
    if not_built:
        print(f"  not built (sycl\\probe, STRATA_SYCL_PROBES): {', '.join(not_built)}")
    print(f"  BUILD.json: {meta['version']}, aot {meta['aot'] or '(none: JIT)'}, {meta['oneapi']}, src {meta['src']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
