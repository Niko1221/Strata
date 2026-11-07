#!/usr/bin/env python3
# Rename HIP globals so they can sit beside the CUDA objects compiled from the
# same sources. A defined symbol is renamed in every object that mentions it,
# including undefined references, so a call from one HIP object to another
# follows the prefix. A symbol the HIP objects never define stays unprefixed
# and binds to the CUDA object.
import re
import shutil
import subprocess
import sys
from pathlib import Path


def skip_name(name):
    if name.startswith(("hipdual_", "hip", "__hip")):
        return True
    return "@" in name


def parse_nm(line):
    # BSD nm: "0000 T name" / "     U name". POSIX nm: "name T [value]".
    matched = re.match(r"^[0-9a-fA-F]+\s+([A-Za-z])\s+(\S+)\s*$", line)
    if matched:
        return matched.group(2), matched.group(1)
    matched = re.match(r"^\s+([A-Za-z])\s+(\S+)\s*$", line)
    if matched:
        return matched.group(2), matched.group(1)
    matched = re.match(r"^(\S+)\s+([A-Za-z])(?:\s|$)", line)
    if matched:
        return matched.group(1), matched.group(2)
    return None


def nm_symbols(path):
    result = subprocess.run(["nm", "-g", path], check=True, capture_output=True, text=True)
    rows = []
    for line in result.stdout.splitlines():
        parsed = parse_nm(line)
        if parsed is None or skip_name(parsed[0]):
            continue
        rows.append(parsed)
    return rows


def defined_type(type_):
    # U is undefined. Every other global type nm -g prints is a definition.
    return len(type_) == 1 and type_.isupper() and type_ != "U"


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: prefix-hip.py OUT LISTFILE")
    out = Path(sys.argv[1])
    objs = [line for line in Path(sys.argv[2]).read_text().splitlines() if line]
    if not objs:
        sys.exit("no HIP objects")

    defined = set()
    present = []
    for src in objs:
        names = {}
        for name, type_ in nm_symbols(src):
            names[name] = True
            if defined_type(type_):
                defined.add(name)
        present.append(names)

    directory = Path(str(out) + ".objs")
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir()

    renamed = 0
    for index, src in enumerate(objs):
        dst = directory / f"{index:05d}.o"
        shutil.copy2(src, dst)
        mapping = []
        for name in sorted(present[index]):
            if name not in defined:
                continue
            mapping.append(f"{name} hipdual_{name}\n")
            renamed += 1
        if not mapping:
            continue
        map_path = Path(str(dst) + ".map")
        map_path.write_text("".join(mapping))
        subprocess.run(["objcopy", f"--redefine-syms={map_path}", dst], check=True)

    prefixed = sorted(directory.glob("*.o"))
    if not prefixed:
        sys.exit("no prefixed objects")
    out.unlink(missing_ok=True)
    subprocess.run(["ar", "rcs", out, *prefixed], check=True)
    print(f"prefix-hip: {len(objs)} objects, {renamed} symbol mentions renamed", file=sys.stderr)


if __name__ == "__main__":
    main()
