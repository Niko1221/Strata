"""Prepare verified small headers, or verify complete existing GGUF shards."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import urllib.request

from model_spec import DEFAULT_SOURCE, HEADERS, SHARDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--verify-full", action="store_true")
    args = parser.parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    for filename, (size, digest) in SHARDS.items():
        target = args.model_dir / filename
        header_size, header_digest = HEADERS[filename]
        if args.verify_full:
            if target.stat().st_size != size:
                raise ValueError(f"a complete {size}-byte shard is required: {target}")
            hasher = hashlib.sha256()
            with target.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                    hasher.update(chunk)
            if hasher.hexdigest() != digest:
                raise ValueError(f"full shard SHA-256 mismatch: {target}")
            print(f"verified complete shard: {filename}", flush=True)
            continue
        if target.exists():
            if target.stat().st_size not in (header_size, size):
                raise ValueError(f"unexpected existing file size: {target}")
            with target.open("rb") as stream:
                content = stream.read(header_size)
        else:
            request = urllib.request.Request(
                args.source.rstrip("/") + "/" + filename,
                headers={"Range": f"bytes=0-{header_size - 1}"},
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                if response.status != 206 or response.headers.get("Content-Range") != (
                    f"bytes 0-{header_size - 1}/{size}"
                ):
                    raise IOError(f"source did not return the exact header range: {filename}")
                content = response.read(header_size + 1)
        if len(content) != header_size or hashlib.sha256(content).hexdigest() != header_digest:
            raise ValueError(f"pinned header SHA-256 mismatch: {filename}")
        if not target.exists():
            temporary = target.with_suffix(".header.tmp")
            temporary.write_bytes(content)
            temporary.replace(target)
        print(f"verified header: {filename} ({header_size} bytes)", flush=True)
    if not args.verify_full:
        print("Headers are not full weights. Inference fetches missing exact tensor ranges over HTTP.")


if __name__ == "__main__":
    main()
