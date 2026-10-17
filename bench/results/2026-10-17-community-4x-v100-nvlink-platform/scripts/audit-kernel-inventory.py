#!/usr/bin/env python3
"""
audit-kernel-inventory.py — Этап II (п. 5.1): инвентарь CUDA-ядер рабочей сборки Strata.

Только чтение бинарника (cuobjdump из приватного тулкита 12.8) и дерева исходников.
Ничего не собирает, не устанавливает и не запускает на GPU.

Вход:  --engine <путь к engine-cuda12/strata>  --tree <путь к дереву исходников>
Выход: --out  <каталог>  →  kernel-inventory.csv (по ярам), kernel-tu-summary.csv (по файлам компиляции)

Имя translation unit восстанавливается из mangling анонимного пространства имён
(_ZN6strata4core53_GLOBAL__N__<hash>_<len>_<file>_cu_<hash>13<kernel>...); для ядер вне
анонимного пространства имя ищется в дереве по объявлению __global__.
"""
import argparse, csv, os, re, subprocess, sys, collections, shutil, hashlib, datetime

def cuobjdump(tool, binary, flag):
    r = subprocess.run([tool, flag, binary], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"cuobjdump {flag} не удался: {r.stderr[:400]}")
    return r.stdout

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--cuobjdump", default=os.path.expanduser("~/strata185/cuda-12.8/bin/cuobjdump"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    res = cuobjdump(a.cuobjdump, a.engine, "--dump-resource-usage")
    open(os.path.join(a.out, "cuobjdump-resource-usage.txt"), "w").write(res)
    open(os.path.join(a.out, "cuobjdump-list-elf.txt"), "w").write(cuobjdump(a.cuobjdump, a.engine, "--list-elf"))

    pat = re.compile(r"Function (_Z[^:]+):\s*\n\s*(REG:\d+ STACK:\d+ SHARED:\d+ LOCAL:\d+ CONSTANT\[\d+\]:\d+)[^\n]*")
    kernels = pat.findall(res)
    archs = sorted(set(re.findall(r"arch = (sm_\d+)", res)))

    # demangle пакетно
    names = [k[0] for k in kernels]
    p = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True)
    dem = p.stdout.splitlines() if p.returncode == 0 else []
    if len(dem) != len(names):
        dem = names

    # индекс __global__-объявлений в дереве: базовое имя -> файл
    idx = collections.defaultdict(list)
    for root, dirs, files in os.walk(a.tree):
        dirs[:] = [d for d in dirs if d not in ("build", "build-cuda12", "build-vision-cuda12", ".git")]
        for fn in files:
            if not fn.endswith((".cu", ".cuh", ".hpp", ".h")):
                continue
            path = os.path.join(root, fn)
            try:
                src = open(path, errors="replace").read()
            except OSError:
                continue
            for m in re.finditer(r"__global__[^;{]*?\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", src):
                idx[m.group(1)].append(os.path.relpath(path, a.tree))

    TU_RE = re.compile(r"GLOBAL__N__([0-9a-f]+)_(\d+)_([A-Za-z0-9_]+?)_cu_([0-9a-f]+)")
    rows = []
    for (mangled, resline), human in zip(kernels, dem):
        t = TU_RE.search(mangled)
        tu = t.group(3) + ".cu" if t else ""
        # базовое имя ядра: первый идентификатор из demangled-имени, который встречается в дереве
        # как __global__-объявление (шаблонные ядра дают имена вида void name<...>(...), поэтому
        # брать «последний идентификатор перед (» нельзя).
        base = ""
        for cand in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", human):
            if cand in idx:
                base = cand
                break
        if not base:
            bm = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(?:<[^<>]*>)?\s*\(", human)
            base = bm.group(1) if bm else human[:80]
        if not tu:
            cand = idx.get(base) or []
            tu = cand[0] if cand else "?"
        regs = int(re.search(r"REG:(\d+)", resline).group(1))
        shared = int(re.search(r"SHARED:(\d+)", resline).group(1))
        local = int(re.search(r"LOCAL:(\d+)", resline).group(1))
        stack = int(re.search(r"STACK:(\d+)", resline).group(1))
        rows.append({
            "kernel_mangled": mangled,
            "kernel": base or human[:120],
            "translation_unit": tu,
            "arch": ",".join(archs),
            "regs": regs, "shared_bytes": shared, "local_bytes": local, "stack_bytes": stack,
            "source_decl_files": ";".join(sorted(set(idx.get(base, [])))[:3]),
            "source_found": "yes" if base in idx else "unknown",
        })

    with open(os.path.join(a.out, "kernel-inventory.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    per = collections.Counter(r["translation_unit"] for r in rows)
    maxr = collections.defaultdict(int); maxs = collections.defaultdict(int)
    for r in rows:
        maxr[r["translation_unit"]] = max(maxr[r["translation_unit"]], r["regs"])
        maxs[r["translation_unit"]] = max(maxs[r["translation_unit"]], r["shared_bytes"])
    with open(os.path.join(a.out, "kernel-tu-summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["translation_unit", "kernels_compiled", "max_regs", "max_shared_bytes", "arch"])
        for tu, n in per.most_common():
            w.writerow([tu, n, maxr[tu], maxs[tu], ",".join(archs)])

    print(f"ядер: {len(rows)}; translation units: {len(per)}; arch: {archs}")
    print(f"артефакты: {a.out}/kernel-inventory.csv, {a.out}/kernel-tu-summary.csv")

if __name__ == "__main__":
    main()
