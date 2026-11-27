"""Read-only look at a .strata file's header and Section 0 (its manifest), to tell which model family a file of any name is.

A .strata v1.0 / v1.1 file starts with the 8 bytes STRATA\\0\\1 and a 4 KiB header frame with six 64-byte section entries: at 0x10 in the 4 KiB
layout, at 0x20 in the 16 KiB-aligned layout (tools/inkling_tokenizer._section0 decides which, the same rule as the engine's reader). Section 0 is
a zstd-compressed JSON manifest with `arch` and `config` (the model's own config: `model_type`). Nothing is written and no weights are read.

The Python Strata ships has no zstd module, so a manifest written with real compression cannot always be unpacked here. Then the family comes
from the file's name or, failing that, from its size (see `family_from_hints`), and the result says so (`manifest_read` False) and the caller words it as
a guess for the person to check. A manifest in a zstd frame of stored blocks (what the tests write) is read without any module.
"""

from __future__ import annotations

import json
import os
import re
import stat
import struct
from pathlib import Path

MAGIC = b"STRATA\x00\x01"
MAX_MANIFEST = 64 * 1024 * 1024
INKLING = "inkling-2-mini"
HY4 = "hy4-flash"
_INKLING_WORDS = re.compile(r"inkling")
_HY4_WORDS = re.compile(r"hy[-_ ]?4|hy4|hunyuan")
#: no zstd module and no hint in the name: a file this big is HY-4 Flash (its smallest build is 62 GB), a file this small is Inkling-2-Mini (its biggest is 20 GB)
HY4_MIN_BYTES = 45 * 10 ** 9
INKLING_MAX_BYTES = 30 * 10 ** 9


class NeedsZstd(RuntimeError):
    """Section 0 is compressed and this Python has no zstd module."""


def _raw_frame(raw: bytes) -> bytes:
    """A zstd frame whose blocks are all stored (raw or run-length): decoded here with no module. Anything compressed raises NeedsZstd."""
    if len(raw) < 6 or struct.unpack_from("<I", raw, 0)[0] != 0xFD2FB528:
        raise ValueError("Section 0 is not a zstd frame")
    d = raw[4]
    pos = 5
    single = bool(d >> 5 & 1)
    if not single:
        pos += 1
    pos += (0, 1, 2, 4)[d & 3]
    pos += {0: 1 if single else 0, 1: 2, 2: 4, 3: 8}[d >> 6]
    out = bytearray()
    while True:
        if pos + 3 > len(raw):
            raise ValueError("Section 0 is cut short")
        h = raw[pos] | raw[pos + 1] << 8 | raw[pos + 2] << 16
        pos += 3
        last, kind, size = h & 1, h >> 1 & 3, h >> 3
        if kind == 0:
            out += raw[pos:pos + size]
            pos += size
        elif kind == 1:
            out += raw[pos:pos + 1] * size
            pos += 1
        else:
            raise NeedsZstd("Section 0 is compressed and this Python has no zstd module")
        if len(out) > MAX_MANIFEST:
            raise ValueError("Section 0 is larger than Strata reads")
        if last:
            return bytes(out)


def _decompress(raw: bytes) -> bytes:
    try:
        from compression import zstd as module                      # Python 3.14
        return module.decompress(raw)
    except ImportError:
        pass
    try:
        import zstandard
    except ImportError:
        return _raw_frame(raw)
    return zstandard.ZstdDecompressor().decompress(raw, max_output_size=MAX_MANIFEST)


def read_header(path: Path) -> dict:
    """{"size", "layout": "4k"|"16k", "section0": (offset, length), "profile"} of one .strata file; ValueError says in words what is wrong with it
    (no magic, a section table that points nowhere or outside the file, a file cut short)."""
    from tools import inkling_tokenizer
    real = os.path.realpath(path)
    with open(real, "rb") as stream:
        details = os.fstat(stream.fileno())
        if not stat.S_ISREG(details.st_mode):
            raise ValueError("the .strata file is not a regular file")
        frame = stream.read(4096)
    if frame[:8] != MAGIC:
        raise ValueError("the file does not begin with a STRATA v1.0/v1.1 header")
    try:
        offset, length = inkling_tokenizer._section0(frame, details.st_size)       # noqa: SLF001 - the one rule both layouts share
    except ValueError as exc:
        raise ValueError(f"the .strata header is not valid: {exc}") from None
    layout = "16k" if _is_16k(frame, details.st_size) else "4k"
    return {"size": details.st_size, "layout": layout, "section0": (offset, length), "profile": struct.unpack_from("<H", frame, 12)[0]}


def _is_16k(frame: bytes, size: int) -> bool:
    """Section 0 entry at 0x20 valid and 16 KiB aligned, the one at 0x10 not (the engine's rule, same as _section0 picks)."""
    entry = struct.Struct("<QQ32sIIQ")

    def aligned(base: int, page: int) -> bool:
        try:
            ents = [entry.unpack_from(frame, base + 64 * i) for i in range(6)]
        except struct.error:
            return False
        live = [e for e in ents if e[1]]
        return bool(live) and all(e[3] == i for i, e in enumerate(ents)) and all(e[0] % page == 0 and e[0] >= 4096 and e[0] + e[1] <= size for e in live)
    return aligned(0x20, 16384) and not aligned(0x10, 4096)


def read_manifest(path: Path, header: dict | None = None) -> dict:
    """Section 0 as a dict. ValueError: the file is not a valid .strata. NeedsZstd: valid, but its manifest cannot be unpacked here."""
    head = header or read_header(path)
    offset, length = head["section0"]
    if length > MAX_MANIFEST:
        raise ValueError("the manifest is larger than Strata reads")
    with open(os.path.realpath(path), "rb") as stream:
        stream.seek(offset)
        raw = stream.read(length)
    try:
        text = _decompress(raw)
    except NeedsZstd:
        raise
    except Exception as exc:                                         # noqa: BLE001 - a bad frame is a damaged file
        raise ValueError(f"the .strata manifest cannot be read: {exc}") from None
    try:
        value = json.loads(text.decode("utf-8"))
    except (UnicodeError, ValueError):
        raise ValueError("the .strata manifest is not valid JSON") from None
    if not isinstance(value, dict):
        raise ValueError("the .strata manifest is not a JSON object")
    return value


def family_of_manifest(manifest: dict) -> str | None:
    """The setup family a manifest's own words name ("inkling-2-mini", "hy4-flash"), None when it names none Strata's engines run."""
    config = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
    words = " ".join(str(v).lower() for v in (manifest.get("arch"), manifest.get("model_type"), config.get("model_type"), text.get("model_type"),
                                                  *(config.get("architectures") or [] if isinstance(config.get("architectures"), list) else [])) if v)
    if _INKLING_WORDS.search(words):
        return INKLING
    if _HY4_WORDS.search(words):
        return HY4
    return None


def family_from_hints(name: str, size: int) -> str | None:
    """The family when the manifest could not be unpacked: by the file's name, else by its size (a guess the caller says out loud)."""
    low = str(name or "").lower()
    if _INKLING_WORDS.search(low):
        return INKLING
    if _HY4_WORDS.search(low):
        return HY4
    if size >= HY4_MIN_BYTES:
        return HY4
    if 0 < size <= INKLING_MAX_BYTES:
        return INKLING
    return None


def inspect(path: Path) -> dict:
    """{"layout", "size", "family", "arch", "manifest_read", "guessed", "name", "build"} of one .strata file. ValueError for a file that is not a valid .strata."""
    head = read_header(path)
    result = {"layout": head["layout"], "size": head["size"], "profile": head["profile"], "family": None, "arch": None, "manifest_read": False,
              "guessed": False, "name": None}
    try:
        manifest = read_manifest(path, head)
    except NeedsZstd:
        result["family"] = family_from_hints(Path(path).name, head["size"])
        result["guessed"] = result["family"] is not None
        return result
    result["manifest_read"] = True
    result["family"] = family_of_manifest(manifest)
    result["arch"] = str(manifest.get("arch") or "") or None
    config = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}
    result["name"] = next((str(v) for v in (manifest.get("name"), manifest.get("model_name"), config.get("_name_or_path"), manifest.get("build")) if isinstance(v, str) and v), None)
    result["build"] = str(manifest.get("build") or "") or None
    return result
