#!/usr/bin/env python3
"""add_notices.py: write licenses/ for the vendor runtime libraries that are already in a staging folder (the folder that goes to package_engine.py --src).

For a package whose libraries were collected before the notices existed (the 0.2.22 SYCL Inkling and Hy4 packages, the Windows CUDA Inkling package) and
that is packed again without a rebuild:

    python tools/packaging/add_notices.py STAGE [--oneapi /opt/intel/oneapi]
    python tools/packaging/package_engine.py --src STAGE ... --include 'licenses/*'

  - Intel oneAPI runtime libraries found in STAGE (libiomp5, libsycl, libur_*, libmkl_*, libimf, libsvml, libirng, libintlc, ...): licenses/intel-oneapi.txt,
    and with --oneapi (the installation they came from) its license and third-party files.
  - Microsoft Visual C++ runtime DLLs found in STAGE (VCOMP140, VCRUNTIME140*, MSVCP140* ...): licenses/microsoft-vc-runtime.txt.
Nothing else is touched; a folder with none of them gets no licenses/.  Prints what it wrote.  (A build gets the same files from collect_libs.py and
package_engine.py themselves: this is only the repack path.)
"""
from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect_libs  # noqa: E402

INTEL_LIBS = ("libiomp5.so*", "libsycl.so*", "libur_*.so*", "libmkl_*.so*", "libimf.so*", "libsvml.so*", "libirng.so*", "libintlc.so*",
              "libsvml*.so*", "libpi_*.so*","libiomp5md.dll", "libmmd.dll", "svml_dispmd.dll", "sycl*.dll", "ur_*.dll", "mkl_*.dll")
VC_RUNTIME = ("vcomp140*.dll", "vcruntime140*.dll", "msvcp140*.dll", "concrt140*.dll", "vcamp140*.dll", "vccorlib140*.dll")


def held(stage: Path, patterns: tuple[str, ...]) -> list[str]:
    return sorted(p.name for p in stage.iterdir() if p.is_file() and any(fnmatch.fnmatch(p.name.lower(), pat.lower()) for pat in patterns))


def add(stage: Path, oneapi: list[str]) -> list[str]:
    wrote: list[str] = []
    intel = held(stage, INTEL_LIBS)
    if intel:
        where = oneapi[0] if oneapi else "an Intel oneAPI installation of the build machine"
        collect_libs.write_intel_notice({n: f"(bundled in this package; from {where})" for n in intel}, oneapi, stage)
        wrote += sorted(str(p.relative_to(stage)).replace("\\", "/") for p in (stage / "licenses").glob("intel-oneapi*"))
    vc = held(stage, VC_RUNTIME)
    if vc:
        source = collect_libs.NOTICES / "microsoft-vc-runtime.txt"
        dest = stage / "licenses" / "microsoft-vc-runtime.txt"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(source.read_text(encoding="utf-8") + "".join(f"  {n}\n" for n in vc), encoding="utf-8")
        wrote.append("licenses/microsoft-vc-runtime.txt")
    return wrote


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", type=Path)
    ap.add_argument("--oneapi", action="append", default=[], help="the oneAPI installation the Intel libraries came from (repeatable)")
    a = ap.parse_args(argv)
    if not a.stage.is_dir():
        ap.error(f"{a.stage} is not a folder")
    wrote = add(a.stage, a.oneapi)
    for w in wrote:
        print("wrote", w)
    if not wrote:
        print("nothing to add: no Intel oneAPI or Visual C++ runtime library in", a.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
