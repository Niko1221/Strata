"""tools/inkling_tokenizer.py - build Strata's serve tokenizer directory (vocab.json, merges.txt, token_type.json,
tokenizer.json, chat_template.jinja) for Thinking Machines Inkling-2-Mini (o200k byte-level BPE, 200,058 tokens).

Sources supported:
  * Section 0 manifest of an Inkling-2-Mini `.strata` v1.1 file (provides `chat_template` + `config`, and `tokenizer_json` if embedded);
  * Explicit HF `tokenizer.json` + `chat_template.jinja` files (hosted beside the `.strata` files on HF or exported via `strata-inkling --export-tokenizer`).
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

PRE = "o200k"
DEFAULT_BOS = 200028                 # <|begin_of_text|>
DEFAULT_EOS = 200006                 # <|content_model_end_sampling|>
DEFAULT_PAD = 199999                 # <|endoftext|>


class NotInklingTokenizer(ValueError):
    """The file's tokenizer is not the o200k byte-level BPE tokenizer Inkling-2-Mini uses."""


def _section0(frame: bytes, file_size: int) -> tuple[int, int]:
    """(offset, length) of Section 0 in the 4 KiB header frame of a `.strata` file, for either header layout.

    The 4 KiB layout (every file up to 0.2.22) has its six 64-byte section entries at 0x10; the 16 KiB-aligned layout (the
    re-adjusted Inkling SQ-20 and Vision) at 0x20, with a u16 tile_layout_id at 0x10.  Nothing in the header names the layout, so a
    table is taken only when it is valid: every section_id is its index, every non-empty section is aligned to the layout's page
    (4 KiB / 16 KiB), starts after the first 4 KiB, lies inside the file and overlaps no other.  Same rule as the engine's reader
    (engine/include/strata/artifact/strata_layout.hpp).  A header that reads as both, or whose table points nowhere, raises ValueError; one that fits neither layout exactly is read as the 4 KiB layout, as before."""
    if len(frame) < 0x20 + 6 * 64 + 32:
        raise ValueError("a .strata header frame is 4096 bytes; this one is shorter")
    entry = struct.Struct("<QQ32sIIQ")

    def probe(base: int, page: int):
        ents = [entry.unpack_from(frame, base + 64 * i) for i in range(6)]
        ids = all(e[3] == i for i, e in enumerate(ents))
        live = sorted((e[0], e[0] + e[1], i) for i, e in enumerate(ents) if e[1])
        aligned = all(a % page == 0 for a, _, _ in live)
        ranges = all(a >= 4096 and (not file_size or b <= file_size) for a, b, _ in live) and all(y[0] >= x[1] for x, y in zip(live, live[1:]))
        return ents, aligned, ids, ranges, bool(live)

    a, b = probe(0x10, 4096), probe(0x20, 16384)
    a_strict, b_strict = all(a[1:]), all(b[1:])
    if a_strict and b_strict:
        raise ValueError("ambiguous .strata header: the section table reads as both the 4 KiB and the 16 KiB layout")
    # the 16 KiB table only when it is valid and the 4 KiB one is not; otherwise the 4 KiB table, read as it always was
    off, length = (b if b_strict else a)[0][0][:2]
    if not length or off < 4096 or (file_size and off + length > file_size):
        raise ValueError("invalid .strata header: no usable Section 0 in either the 4 KiB or the 16 KiB section table")
    return off, length


def read_manifest(path) -> dict:
    """Read Section 0 of a `.strata` v1.1 file as a dict. Raises RuntimeError if Python has no zstd module."""
    with open(path, "rb") as f:
        frame = f.read(4096)
        if frame[:8] != b"STRATA\x00\x01":
            raise ValueError(f"{path} is not a .strata file (no STRATA\\x00\\x01 magic)")
        f.seek(0, 2)
        size = f.tell()
        off, length = _section0(frame, size)   # either header layout (4 KiB or 16 KiB aligned)
        f.seek(off)
        raw = f.read(length)
    try:
        from compression import zstd as _z
        text = _z.decompress(raw)
    except ImportError:
        try:
            import zstandard
            text = zstandard.ZstdDecompressor().decompress(raw, max_output_size=1 << 30)
        except ImportError:
            raise RuntimeError("this Python cannot read zstd: use --hf-json and --template or strata-inkling --export-tokenizer") from None
    return json.loads(text.decode("utf-8"))


def _check_pre(tok: dict) -> None:
    pre = tok.get("pre_tokenizer") or {}
    parts = pre.get("pretokenizers") if pre.get("type") == "Sequence" else None
    if not parts or len(parts) != 2:
        raise NotInklingTokenizer("the pre-tokenizer is not the o200k Split + ByteLevel sequence")
    first, second = parts[0], parts[1]
    rx = (first.get("pattern") or {}).get("Regex", "")
    if first.get("type") != "Split" or r"\p{N}{1,3}" not in rx or first.get("behavior") != "Isolated":
        raise NotInklingTokenizer("split 1 of the pre-tokenizer is not the o200k rule")
    if second.get("type") != "ByteLevel" or second.get("add_prefix_space") or second.get("use_regex"):
        raise NotInklingTokenizer("the byte mapping is not the plain GPT-2 ByteLevel one")
    if (tok.get("model") or {}).get("type") != "BPE":
        raise NotInklingTokenizer("the model is not byte-level BPE")


def convert(tokenizer_json, chat_template: str, out_dir, *, bos: int | None = DEFAULT_BOS,
            eos: int | None = DEFAULT_EOS, pad: int | None = DEFAULT_PAD) -> dict:
    """Write Strata's tokenizer directory from HF tokenizer.json and chat_template.jinja."""
    tok = json.loads(tokenizer_json) if isinstance(tokenizer_json, (str, bytes)) else tokenizer_json
    _check_pre(tok)
    base = tok["model"]["vocab"]
    added = tok.get("added_tokens") or []
    size = max(max(base.values()) if base else -1, max((a["id"] for a in added), default=-1)) + 1
    tokens: list[str | None] = [None] * size
    types = [1] * size
    for t, i in base.items():
        tokens[i] = t
    for a in added:
        aid = int(a["id"])
        if tokens[aid] is not None and tokens[aid] != a["content"]:
            raise NotInklingTokenizer(f"token id {aid} conflicts: {tokens[aid]!r} vs {a['content']!r}")
        tokens[aid] = a["content"]
        types[aid] = 3 if a.get("special") else 4
    for i, t in enumerate(tokens):
        if t is None:
            tokens[i] = f"<|unused_{i}|>"
            types[i] = 1
    merges = [m if isinstance(m, str) else " ".join(m) for m in tok["model"]["merges"]]
    special = {k: int(v) for k, v in (("bos", bos), ("eos", eos), ("pad", pad)) if v is not None}
    cfg = {
        "model": "gpt2",
        "pre": PRE,
        "vocab_size": size,
        "n_merges": len(merges),
        "special_ids": special,
        "add_bos_token": False,
        "source": "Inkling-2-Mini o200k tokenizer.json + chat_template.jinja",
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(tokens)}, ensure_ascii=False), encoding="utf-8")
    (out / "merges.txt").write_text("\n".join(merges), encoding="utf-8")
    (out / "token_type.json").write_text(json.dumps(types), encoding="utf-8")
    (out / "tokenizer.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "chat_template.jinja").write_text(chat_template, encoding="utf-8", newline="\n")
    return cfg


def convert_manifest(manifest: dict, out_dir, *, tokenizer_json_path=None) -> dict:
    """Convert a Section 0 manifest (plus fallback `tokenizer_json_path` when manifest omits `tokenizer_json`)."""
    tc = (manifest.get("config") or {}).get("text_config") or {}
    cfg_root = manifest.get("config") or {}
    tj = manifest.get("tokenizer_json")
    if tj is None:
        if tokenizer_json_path is None or not Path(tokenizer_json_path).is_file():
            raise RuntimeError("manifest has no embedded tokenizer_json and no tokenizer.json file was provided")
        tj = Path(tokenizer_json_path).read_text(encoding="utf-8")
    ct = manifest.get("chat_template") or ""
    if not ct:
        raise RuntimeError("manifest has no chat_template")
    eos = cfg_root.get("eos_token_id", tc.get("eos_token_id", DEFAULT_EOS))
    bos = cfg_root.get("bos_token_id", tc.get("bos_token_id", DEFAULT_BOS))
    pad = cfg_root.get("pad_token_id", tc.get("pad_token_id", DEFAULT_PAD))
    return convert(tj, ct, out_dir, bos=bos, eos=eos, pad=pad)


def convert_files(hf_json_path, template_path, out_dir, *, bos: int = DEFAULT_BOS,
                  eos: int = DEFAULT_EOS, pad: int = DEFAULT_PAD) -> dict:
    return convert(Path(hf_json_path).read_text(encoding="utf-8"), Path(template_path).read_text(encoding="utf-8"),
                   out_dir, bos=bos, eos=eos, pad=pad)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--strata", help="an Inkling-2-Mini .strata file")
    ap.add_argument("--hf-json", help="Hugging Face tokenizer.json")
    ap.add_argument("--template", help="chat_template.jinja")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.strata:
        cfg = convert_manifest(read_manifest(a.strata), a.out, tokenizer_json_path=a.hf_json)
    elif a.hf_json and a.template:
        cfg = convert_files(a.hf_json, a.template, a.out)
    else:
        ap.error("give --strata (optionally with --hf-json), or --hf-json and --template")
    print(f"tokenizer: vocab {cfg['vocab_size']}, merges {cfg['n_merges']}, pre {cfg['pre']}, specials {cfg['special_ids']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
