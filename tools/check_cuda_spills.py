#!/usr/bin/env python3
"""Check that no release CUDA kernel spills registers or keeps a per-thread array in local memory.

A kernel that needs more than its registers puts the rest in local memory - ptxas spills under register pressure, and
a per-thread array it cannot promote to registers (indexed dynamically, or too large) - and every access there is a
round trip to device memory through L1, in the hottest loops the engine has.  Nothing on the build machine says so:
ptxas reports it only with -Xptxas -v, and a GPU is needed for a profiler to.  The compiler does write it into every
object, though: `cuobjdump --dump-resource-usage` lists, per entry function and per SASS architecture, the registers,
the stack frame (STACK: with CUDA 13 spills and local arrays both live there), the shared memory and the .local bytes
(LOCAL), so this check reads the libraries that were built anyway and needs no GPU.

Every kernel must have STACK 0 and LOCAL 0 on every sm_ in the library, unless ALLOWED below carries an entry for it
with the measured allowance: the kernels that keep a per-thread array on purpose, one line each saying what the array
is.  A byte over the allowance fails; a kernel that is not listed fails with any of either.

  check_cuda_spills.py libstrata_kernels.a [libstrata_core.a ...]   (needs cuobjdump, and cu++filt or c++filt)
  check_cuda_spills.py --list LIB ...   the full table (library, object, sm, REG, STACK, LOCAL, SHARED, kernel) as
                                        CSV, every kernel on every sm_, nothing judged: this is how the numbers in
                                        ALLOWED were produced
  --cuobjdump PATH                      the toolkit's cuobjdump when bin/ is not on PATH (CMake passes it)

Exit 0 = clean, 1 = a violation, 77 = cannot run here (cuobjdump missing, a library missing; ctest SKIP_RETURN_CODE).
"""
import argparse
import csv
import os
import re
import shutil
import subprocess
import sys

# Kernels that keep a per-thread array in local memory on purpose (or spill) at 296b59d2, measured with CUDA 13.1
# (V13.1.80) for sm_86 and sm_120.  Key: the demangled kernel name without its template arguments and parameter list
# (namespace kept; matched exactly), so every instantiation of a template kernel shares one entry.  Value: per sm_,
# the allowed (STACK, LOCAL) bytes - the measured numbers, so any growth fails.  One line each: what the array is.
ALLOWED = {
}


def kernel_key(demangled):
    """The demangled name without parameter list, template arguments and return type: the ALLOWED key."""
    # c++filt prints an anonymous namespace as "(anonymous namespace)", cu++filt as "<unnamed>": one spelling, so the
    # key is the same whichever demangler is found
    s = demangled.replace("(anonymous namespace)", "{anonymous}").replace("<unnamed>", "{anonymous}")
    depth, cut = 0, len(s)
    for i, c in enumerate(s):                       # the parameter list starts at the first '(' outside <...>
        if c == "<":
            depth += 1
        elif c == ">":
            depth -= 1
        elif c == "(" and depth == 0:
            cut = i
            break
    out, depth = [], 0
    for c in s[:cut]:                               # drop every <...> (template arguments, nested ones too)
        if c == "<":
            depth += 1
        elif c == ">":
            depth -= 1
        elif depth == 0:
            out.append(c)
    s = "".join(out).strip()
    return s.split()[-1] if s else demangled        # a template kernel is printed with its return type: "void ns::k"


def find_tools(cuobjdump_hint):
    """cuobjdump from --cuobjdump or PATH; the demangler beside it (cu++filt), on PATH, or binutils' c++filt."""
    exe = ".exe" if os.name == "nt" else ""
    cuobjdump = cuobjdump_hint if cuobjdump_hint and os.access(cuobjdump_hint, os.X_OK) else shutil.which("cuobjdump")
    if cuobjdump is None:
        return None, None
    for cand in (os.path.join(os.path.dirname(cuobjdump), "cu++filt" + exe), shutil.which("cu++filt"),
                 shutil.which("c++filt")):
        if cand and os.access(cand, os.X_OK):
            return cuobjdump, cand
    return cuobjdump, None


def resource_usage(cuobjdump, path):
    """[(object, sm, mangled, {REG: n, STACK: n, SHARED: n, LOCAL: n, ...})] for every entry function of every SASS
    image in an archive, an object, a shared library or an executable."""
    run = subprocess.run([cuobjdump, "--dump-resource-usage", path], capture_output=True, text=True)
    if run.returncode != 0:
        print("check_cuda_spills: cuobjdump failed on %s:\n%s" % (path, (run.stderr or run.stdout).strip()[:2000]))
        return None
    rows, obj, sm, elf, func = [], os.path.basename(path), None, False, None
    for line in run.stdout.splitlines():
        s = line.strip()
        m = re.match(r"^member .*:([^:]+):$", s)      # an archive: "member /path/libx.a:gdn.cu.o:"
        if m:
            obj, sm, elf, func = m.group(1), None, False, None
            continue
        if s.startswith("Fatbin elf code:"):
            elf = True
            continue
        if s.startswith("Fatbin ptx code:"):            # the PTX image has an "arch =" line too, and no resources
            elf = False
            continue
        m = re.match(r"^arch = (\S+)$", s)
        if m:
            sm = m.group(1)
            continue
        m = re.match(r"^Function (\S+):$", s)
        if m and elf:
            func = m.group(1)
            continue
        if func is not None and s.startswith("REG:"):
            cols = {k: int(v) for k, v in re.findall(r"([A-Z]+(?:\[\d+\])?):(\d+)", s)}
            rows.append((obj, sm, func, cols))
            func = None
    return rows


def demangle(demangler, names):
    if demangler is None or not names:
        return {n: n for n in names}
    run = subprocess.run([demangler], input="\n".join(names) + "\n", capture_output=True, text=True)
    out = run.stdout.splitlines()
    if run.returncode != 0 or len(out) != len(names):
        return {n: n for n in names}
    return dict(zip(names, out))


def check(path, rows, names):
    archs = sorted({sm for _, sm, _, _ in rows})
    held, bad = 0, []
    for obj, sm, func, cols in rows:
        stack, local = cols.get("STACK", 0), cols.get("LOCAL", 0)
        if stack == 0 and local == 0:
            continue
        held += 1
        allow = ALLOWED.get(kernel_key(names[func]), {}).get(sm)
        if allow is None:
            why = "not in ALLOWED"
        elif stack > allow[0] or local > allow[1]:
            why = "over its allowance STACK %d LOCAL %d" % allow
        else:
            continue
        bad.append((names[func], sm, stack, local, why, obj))
    print("check_cuda_spills: %s: %d kernels on %s (%d entries), %d use stack / local memory, %d not allowed"
          % (path, len({f for _, _, f, _ in rows}), ", ".join(archs), len(rows), held, len(bad)))
    for name, sm, stack, local, why, obj in bad:
        print("  VIOLATION %s %s: STACK %d LOCAL %d (%s) [%s]" % (name[:160], sm, stack, local, why, obj))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("library", nargs="+", help="a static library, object, shared library or executable with SASS")
    ap.add_argument("--list", action="store_true", help="print every kernel on every sm_ as CSV, judge nothing")
    ap.add_argument("--cuobjdump", help="the toolkit's cuobjdump (default: on PATH)")
    a = ap.parse_args()
    cuobjdump, demangler = find_tools(a.cuobjdump)
    if cuobjdump is None:
        print("check_cuda_spills: cuobjdump not found (put the CUDA toolkit's bin/ on PATH, or pass --cuobjdump)")
        return 77
    for lib in a.library:
        if not os.path.exists(lib):
            print("check_cuda_spills: %s does not exist" % lib)
            return 77
    out = csv.writer(sys.stdout, lineterminator="\n") if a.list else None
    if out:
        out.writerow(["library", "object", "sm", "reg", "stack", "local", "shared", "kernel", "mangled"])
    rc = 0
    for lib in a.library:
        rows = resource_usage(cuobjdump, lib)
        if rows is None:
            return 77
        if not rows:
            # a library without one SASS entry function is not one of the kernel libraries, or cuobjdump's output
            # changed shape: either way this check did not run, and a pass with nothing counted is not a pass
            print("check_cuda_spills: %s: no SASS entry function found (not a CUDA library, or cuobjdump's format "
                  "changed)" % lib)
            return 77
        names = demangle(demangler, sorted({f for _, _, f, _ in rows}))
        if out:
            for obj, sm, func, cols in rows:
                out.writerow([os.path.basename(lib), obj, sm, cols.get("REG", 0), cols.get("STACK", 0),
                              cols.get("LOCAL", 0), cols.get("SHARED", 0), names[func], func])
        else:
            rc = max(rc, check(lib, rows, names))
    return rc


if __name__ == "__main__":
    sys.exit(main())
