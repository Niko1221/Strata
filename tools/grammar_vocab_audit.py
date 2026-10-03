"""Compare the native GBNF vocabulary with every ID in Strata's existing tokenizer.

The native executable performs matching. Python only checks the emitted byte table.
No model or GPU is needed. Writes a receipt and native test log in a new directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import time

from strata_tokenizer import Tokenizer


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exe", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    table = a.out / "vocab.bin"
    command = [str(a.exe.resolve()), str(a.cases.resolve()), str(a.tokenizer.resolve()), str(table.resolve())]
    start = time.monotonic()
    with (a.out / "grammar-native-tests.txt").open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=60, check=True)
    native_seconds = time.monotonic() - start
    read = lambda name: json.loads((a.tokenizer / name).read_text(encoding="utf-8"))
    meta, vocab, types = read("tokenizer.json"), read("vocab.json"), read("token_type.json")
    tokens = [None] * len(vocab)
    for token, i in vocab.items():
        tokens[i] = token
    tk = Tokenizer(tokens, (a.tokenizer / "merges.txt").read_text(encoding="utf-8").splitlines(),
                   types, meta["pre"], meta["special_ids"])
    stops = {248044, 248046}  # current engine end IDs; test profile is explicit
    data = table.read_bytes()
    assert data[:5] == b"SVOC1" and struct.unpack_from("<I", data, 5)[0] == len(tokens)
    offset, normal, excluded = 9, 0, 0
    for i in range(len(tokens)):
        size, = struct.unpack_from("<I", data, offset)
        offset += 4
        actual = data[offset:offset + size]
        offset += size
        if types[i] == 1 and i not in stops:
            expected = tk.token_bytes(i)
            normal += 1
        else:
            expected = b""
            excluded += 1
        assert actual == expected, f"token {i} native bytes differ"
    assert offset == len(data)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    receipt = {"result": "pass", "command": command, "native_seconds": native_seconds,
               "binary_sha256": sha(a.exe), "vocabulary_size": len(tokens),
               "normal_tokens_compared": normal, "control_or_unused_excluded": excluded,
               "stop_ids": sorted(stops), "tokenizer_metadata": meta,
               "byte_table_sha256": sha(table),
               "artifact_sha256": {name: sha(a.tokenizer / name) for name in
                                   ("tokenizer.json", "vocab.json", "token_type.json", "merges.txt")}}
    (a.out / "tokenizer-identity.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("result", "vocabulary_size", "normal_tokens_compared",
                                            "control_or_unused_excluded", "native_seconds")}))


if __name__ == "__main__":
    main()
