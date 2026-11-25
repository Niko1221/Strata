#!/usr/bin/env python3
"""Keep the debug symbols of a release build (owner decision 2026-11-25, 0.2.22): one symbol file per shipped engine binary, stored OUTSIDE the
package and never uploaded with a release.

  keep_symbols.py elf    --out DIR [--require] FILE...      split the debug information off an ELF binary (objcopy --only-keep-debug), strip it from
                                                           FILE (--strip-debug only: the function symbols stay, as before) and link the two with a
                                                           .gnu_debuglink.  Run it on the STAGE copy of the binary, BEFORE package_engine.py hashes it.
  keep_symbols.py pe     --out DIR [--pdb-dir DIR] [--require] FILE...   copy the PDB of each exe / dll (found by the name the exe's CodeView record
                                                           names, in --pdb-dir or next to the file) after checking the GUID and age; needs a build made
                                                           with -DSTRATA_KEEP_SYMBOLS=ON (/Z7 + /DEBUG, same code as without it)
  keep_symbols.py macho  --out DIR [--require] FILE...      dsymutil FILE -o DIR/<name>.dSYM, then strip -S FILE (run before the owner signs)
  keep_symbols.py check  A B                                compare the code sections of two builds of the same source (one with
                                                           STRATA_KEEP_SYMBOLS, one without): .text, .nv_fatbin / .nv_fatb, .rodata / .rdata are not
                                                           compared when they hold build paths.  Exit 0 when .text (and the CUDA fatbin) are equal.
  keep_symbols.py show   FILE                               the id the symbol file must match (ELF build-id, PE CodeView GUID+age, Mach-O UUID)

Every command appends to DIR/symbols.json: {package file, kind, id, sha256 of the shipped binary, symbol file, its sha256}.
The store is C:\\r0222\\symbols\\<package>\\ on the owner PC (docs/RELEASING.md); a crash address plus the build id from the user's report
finds the file.  Nothing here changes the machine code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------- ELF
def elf_sections(path: Path):
    """[(name, offset, size, type)] of a 64-bit little-endian ELF, or None when the file is not one."""
    with open(path, "rb") as f:
        head = f.read(64)
        if head[:4] != b"\x7fELF" or head[4] != 2 or head[5] != 1:
            return None
        shoff, = struct.unpack_from("<Q", head, 0x28)
        shentsize, shnum, shstrndx = struct.unpack_from("<HHH", head, 0x3A)
        f.seek(shoff)
        raw = f.read(shentsize * shnum)
        secs = []
        for i in range(shnum):
            name, typ, _flags, _addr, off, size = struct.unpack_from("<IIQQQQ", raw, i * shentsize)
            secs.append([name, off, size, typ])
        stroff, strsize = secs[shstrndx][1], secs[shstrndx][2]
        f.seek(stroff)
        strtab = f.read(strsize)
    out = []
    for name, off, size, typ in secs:
        end = strtab.find(b"\0", name)
        out.append((strtab[name:end].decode("ascii", "replace"), off, size, typ))
    return out


def elf_section_bytes(path: Path, wanted: str):
    secs = elf_sections(path) or []
    for name, off, size, typ in secs:
        if name == wanted and typ != 8:   # 8 = NOBITS
            h = hashlib.sha256()
            with open(path, "rb") as f:
                f.seek(off)
                left = size
                while left:
                    block = f.read(min(left, 1 << 20))
                    h.update(block)
                    left -= len(block)
            return size, h.hexdigest()
    return None


def elf_build_id(path: Path) -> str:
    sec = None
    for name, off, size, _typ in elf_sections(path) or []:
        if name == ".note.gnu.build-id":
            sec = (off, size)
    if not sec:
        return ""
    with open(path, "rb") as f:
        f.seek(sec[0])
        raw = f.read(sec[1])
    namesz, descsz, _typ = struct.unpack_from("<III", raw, 0)
    start = 12 + ((namesz + 3) & ~3)
    return raw[start:start + descsz].hex()


def tool(name: str) -> str:
    exe = shutil.which(name)
    if not exe:
        raise SystemExit("keep_symbols.py: %s not found on PATH" % name)
    return exe


def do_elf(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    objcopy = tool("objcopy")
    rc = 0
    for file in map(Path, a.files):
        secs = elf_sections(file)
        if secs is None:
            print("skip (not an ELF): %s" % file)
            continue
        if not any(n == ".debug_info" for n, *_ in secs):
            print("%s: no debug information (built without -DSTRATA_KEEP_SYMBOLS=ON)" % file)
            if a.require:
                rc = 1
            continue
        bid = elf_build_id(file)
        dbg = out / (file.name + ".debug")
        before = {s: elf_section_bytes(file, s) for s in (".text", ".nv_fatbin")}
        subprocess.run([objcopy, "--only-keep-debug", str(file), str(dbg)], check=True)
        subprocess.run([objcopy, "--strip-debug", str(file)], check=True)
        subprocess.run([objcopy, "--add-gnu-debuglink=" + str(dbg), str(file)], check=True)
        after = {s: elf_section_bytes(file, s) for s in before}
        if before != after:
            raise SystemExit("keep_symbols.py: %s: .text / .nv_fatbin changed by the strip: %s != %s" % (file, before, after))
        record(out, {"file": file.name, "kind": "elf", "id": bid, "sha256": sha256(file), "symbols": dbg.name, "symbols_sha256": sha256(dbg),
                     "symbols_bytes": dbg.stat().st_size})
        print("kept %s -> %s (%.1f MiB, build-id %s); shipped binary %.1f MiB" % (file.name, dbg, dbg.stat().st_size / 1048576, bid, file.stat().st_size / 1048576))
    return rc


# ---------------------------------------------------------------------------------------------------- PE
def pe_info(path: Path):
    """(sections {name: (raw offset, raw size)}, codeview (guid hex, age, pdb path) or None) of a PE file, or None."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:2] != b"MZ":
        return None
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe:pe + 4] != b"PE\0\0":
        return None
    nsec, = struct.unpack_from("<H", data, pe + 6)
    optsize, = struct.unpack_from("<H", data, pe + 20)
    opt = pe + 24
    magic, = struct.unpack_from("<H", data, opt)
    ddir = opt + (112 if magic == 0x20B else 96)
    dbg_rva, dbg_size = struct.unpack_from("<II", data, ddir + 6 * 8)
    sec0 = opt + optsize
    sections, rva_map = {}, []
    for i in range(nsec):
        base = sec0 + i * 40
        name = data[base:base + 8].rstrip(b"\0").decode("ascii", "replace")
        vsize, vaddr, rsize, roff = struct.unpack_from("<IIII", data, base + 8)
        sections[name] = (roff, rsize)
        rva_map.append((vaddr, max(vsize, rsize), roff))
    cv = None
    if dbg_rva:
        off = next((roff + (dbg_rva - va) for va, sz, roff in rva_map if va <= dbg_rva < va + sz), None)
        for i in range(dbg_size // 28 if off is not None else 0):
            ent = off + i * 28
            typ, size, _addr, ptr = struct.unpack_from("<IIII", data, ent + 12)
            if typ == 2 and data[ptr:ptr + 4] == b"RSDS":
                guid = data[ptr + 4:ptr + 20]
                age, = struct.unpack_from("<I", data, ptr + 20)
                pdb = data[ptr + 24:ptr + size].split(b"\0")[0].decode("utf-8", "replace")
                d1, d2, d3 = struct.unpack_from("<IHH", guid, 0)
                cv = ("%08X%04X%04X%s" % (d1, d2, d3, guid[8:].hex().upper()), age, pdb)
    return sections, cv, data


def pe_section_hash(path: Path, name: str):
    info = pe_info(path)
    if not info or name not in info[0]:
        return None
    roff, rsize = info[0][name]
    return rsize, hashlib.sha256(info[2][roff:roff + rsize]).hexdigest()


def do_pe(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rc = 0
    for file in map(Path, a.files):
        info = pe_info(file)
        if info is None:
            print("skip (not a PE file): %s" % file)
            continue
        cv = info[1]
        if cv is None:
            print("%s: no CodeView record (built without -DSTRATA_KEEP_SYMBOLS=ON / /DEBUG)" % file)
            if a.require:
                rc = 1
            continue
        guid, age, pdb_path = cv
        name = Path(pdb_path.replace("\\", "/")).name
        found = None
        for d in ([Path(a.pdb_dir)] if a.pdb_dir else []) + [file.parent]:
            if (d / name).is_file():
                found = d / name
                break
        if found is None:
            print("%s: the PDB %s is not in %s" % (file, name, a.pdb_dir or file.parent))
            rc = 1
            continue
        raw = found.read_bytes()
        if not raw.startswith(b"Microsoft C/C++ MSF 7.00"):
            print("%s: %s is not an MSF 7.00 PDB" % (file, found))
            rc = 1
            continue
        # the PDB's own GUID sits in the PDB info stream; checking it needs an MSF reader, so check the one thing that is cheap and reliable: the
        # file was written after the exe's link started and carries the exe's GUID bytes somewhere in its info stream
        d1, d2, d3 = int(guid[:8], 16), int(guid[8:12], 16), int(guid[12:16], 16)
        gbytes = struct.pack("<IHH", d1, d2, d3) + bytes.fromhex(guid[16:])
        if gbytes not in raw:
            print("%s: %s does not hold the exe's GUID %s: it is the PDB of another link" % (file, found, guid))
            rc = 1
            continue
        dst = out / (file.name + ".pdb" if name.lower() != file.stem.lower() + ".pdb" else name)
        shutil.copy2(found, dst)
        record(out, {"file": file.name, "kind": "pe", "id": "%s-%d" % (guid, age), "sha256": sha256(file), "symbols": dst.name,
                     "symbols_sha256": sha256(dst), "symbols_bytes": dst.stat().st_size})
        print("kept %s -> %s (%.1f MiB, GUID %s age %d)" % (file.name, dst, dst.stat().st_size / 1048576, guid, age))
    return rc


# ---------------------------------------------------------------------------------------------------- Mach-O
def macho_uuid(path: Path) -> str:
    with open(path, "rb") as f:
        data = f.read(1 << 16)
    magic, = struct.unpack_from("<I", data, 0)
    if magic != 0xFEEDFACF:
        return ""
    ncmds, = struct.unpack_from("<I", data, 16)
    off = 32
    for _ in range(ncmds):
        cmd, size = struct.unpack_from("<II", data, off)
        if cmd == 0x1B:
            return data[off + 8:off + 24].hex()
        off += size
    return ""


def do_macho(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dsymutil, strip = tool("dsymutil"), tool("strip")
    rc = 0
    for file in map(Path, a.files):
        uid = macho_uuid(file)
        if not uid:
            print("skip (not a 64-bit Mach-O): %s" % file)
            continue
        dsym = out / (file.name + ".dSYM")
        if dsym.exists():
            shutil.rmtree(dsym)
        r = subprocess.run([dsymutil, str(file), "-o", str(dsym)], capture_output=True, text=True)
        dwarf = dsym / "Contents" / "Resources" / "DWARF" / file.name
        if r.returncode != 0 or not dwarf.is_file():
            print("%s: dsymutil produced no DWARF (built without -g?): %s" % (file, (r.stderr or r.stdout).strip()[:300]))
            if a.require:
                rc = 1
            shutil.rmtree(dsym, ignore_errors=True)
            continue
        subprocess.run([strip, "-S", str(file)], check=True)
        record(out, {"file": file.name, "kind": "macho", "id": uid, "sha256": sha256(file), "symbols": dsym.name,
                     "symbols_sha256": sha256(dwarf), "symbols_bytes": dwarf.stat().st_size})
        print("kept %s -> %s (%.1f MiB DWARF, UUID %s)" % (file.name, dsym, dwarf.stat().st_size / 1048576, uid))
    return rc


# ---------------------------------------------------------------------------------------------------- shared
def record(out: Path, entry: dict) -> None:
    path = out / "symbols.json"
    items = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    items = [e for e in items if e.get("file") != entry["file"]] + [entry]
    path.write_text(json.dumps(items, indent=1) + "\n", encoding="utf-8")


def pe_text_shift(pa: Path, pb: Path):
    """For two PE files whose .text have the same size but different bytes: (number of differing 32-bit fields, the one delta they all share) when EVERY
    differing byte lies in a 32-bit field that differs by the same constant (rip-relative displacements to a data section that moved because the debug
    directory was added in front of it); None when anything else differs (another instruction)."""
    ia, ib = pe_info(pa), pe_info(pb)
    if not ia or not ib or ".text" not in ia[0] or ".text" not in ib[0]:
        return None
    (oa, sa), (ob, sb) = ia[0][".text"], ib[0][".text"]
    if sa != sb:
        return None
    a, b = ia[2][oa:oa + sa], ib[2][ob:ob + sb]
    diff = [i for i in range(sa) if a[i] != b[i]]
    delta, fields, covered = None, 0, -1
    for i in diff:
        if i <= covered:
            continue
        cands = []
        for st in range(max(0, i - 3), i + 1):
            if st + 4 > sa:
                continue
            d = (int.from_bytes(a[st:st + 4], "little") - int.from_bytes(b[st:st + 4], "little")) & 0xFFFFFFFF
            small = d if d < 0x80000000 else (1 << 32) - d     # the size of the move, either way
            if d != 0 and small < 0x100000 and (delta is None or d == delta):
                cands.append((small, st, d))
        if not cands:
            return None
        _small, st, d = min(cands)                              # the narrowest reading: the field that really moved
        delta = d if delta is None else delta
        covered = st + 3
        fields += 1
    return fields, (delta if delta is not None and delta < 0x80000000 else (delta or 0) - (1 << 32))


def do_check(a) -> int:
    pa, pb = Path(a.a), Path(a.b)
    if elf_sections(pa) is not None:
        names = (".text", ".nv_fatbin", ".rodata")
        get = elf_section_bytes
    elif pe_info(pa) is not None:
        names = (".text", ".nv_fatb", ".rdata")
        get = pe_section_hash
    else:
        raise SystemExit("keep_symbols.py check: %s is neither ELF nor PE" % pa)
    bad = 0
    for n in names:
        x, y = get(pa, n), get(pb, n)
        same = x == y
        must = n in (".text", ".nv_fatbin", ".nv_fatb")
        print("%-10s %s  A=%s  B=%s%s" % (n, "EQUAL " if same else "DIFFER", x, y, "" if must else "  (informational: may hold paths and the debug directory)"))
        if n == ".text" and not same and pe_info(pa) is not None:
            shift = pe_text_shift(pa, pb)
            if shift is not None:
                print("           .text differs only in %d rip-relative displacement fields, all moved by %d bytes (the debug directory in .rdata): same instructions" % shift)
                same = True
        if must and not same and not (x is None and y is None):
            bad += 1
    print("check: %s" % ("code sections equal" if not bad else "CODE DIFFERS"))
    return 1 if bad else 0


def do_show(a) -> int:
    f = Path(a.file)
    if elf_sections(f) is not None:
        print("elf build-id", elf_build_id(f))
    elif (info := pe_info(f)) is not None:
        print("pe", info[1])
    else:
        print("macho uuid", macho_uuid(f))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("elf", do_elf), ("pe", do_pe), ("macho", do_macho)):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--require", action="store_true", help="fail when a file has no debug information")
        if name == "pe":
            p.add_argument("--pdb-dir")
        p.add_argument("files", nargs="+")
        p.set_defaults(fn=fn)
    p = sub.add_parser("check")
    p.add_argument("a")
    p.add_argument("b")
    p.set_defaults(fn=do_check)
    p = sub.add_parser("show")
    p.add_argument("file")
    p.set_defaults(fn=do_show)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
