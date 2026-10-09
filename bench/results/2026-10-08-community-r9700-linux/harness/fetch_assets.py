#!/usr/bin/env python3
"""Fetch the goal's pinned original GGUFs; verify bytes and share identical PLE files."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request

REPO = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
REVISION = "ed59f92082b1e93c0e96d60a8b11aab089b52f09"
ROOT = Path(__file__).resolve().parents[3]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant", choices=["IQ3_S", "IQ2_XS"], required=True)
    parser.add_argument("--assets", type=Path, default=ROOT.parent / "r9700-assets")
    args = parser.parse_args()
    assets = args.assets.resolve()
    assets.mkdir(parents=True, exist_ok=True)
    with (assets / ".fetch.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        url = f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true"
        with urllib.request.urlopen(url, timeout=60) as response:
            metadata = json.load(response)
        if metadata["sha"] != REVISION:
            raise RuntimeError("model revision mismatch")
        save(assets / "hf-model-metadata.json", metadata)
        manifest_path = assets / "gguf-manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        selected = [f for f in metadata["siblings"]
                    if f["rfilename"].startswith(args.quant + "/") and f["rfilename"].endswith(".gguf")]
        if len(selected) != 2:
            raise RuntimeError("expected exactly two GGUF shards")
        for item in selected:
            relative = item["rfilename"]
            target = assets / "gguf" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            expected = item["lfs"]["sha256"]
            size = item["size"]
            if not target.exists():
                # Only reuse a sibling after hashing it; names or an old manifest alone are insufficient.
                for previous in manifest.values():
                    source = Path(previous["path"])
                    if (previous["sha256"] == expected and source.is_file()
                            and source.stat().st_size == size and sha256(source) == expected):
                        os.link(source, target)
                        print(f"linked identical shard: {target}", flush=True)
                        break
            if not target.exists():
                partial = target.with_suffix(".gguf.part")
                have = partial.stat().st_size if partial.exists() else 0
                if shutil.disk_usage(assets).free < size - have + (16 << 30):
                    raise RuntimeError("insufficient disk space with 16 GiB headroom")
                print(f"fetch {relative}: {have}/{size} bytes", flush=True)
                subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                                "--retry", "5", "--retry-delay", "2", "--connect-timeout", "30",
                                "--speed-limit", "1024", "--speed-time", "120",
                                "--continue-at", "-", "--output", str(partial),
                                f"https://huggingface.co/{REPO}/resolve/{REVISION}/{relative}"], check=True)
                if partial.stat().st_size != size or sha256(partial) != expected:
                    raise RuntimeError(f"size/hash mismatch; retained for inspection: {partial}")
                partial.replace(target)
            elif target.stat().st_size != size or sha256(target) != expected:
                raise RuntimeError(f"existing shard failed size/hash verification: {target}")
            manifest[relative] = {"repo": REPO, "revision": REVISION, "path": str(target),
                                  "size": size, "sha256": expected}
            save(manifest_path, manifest)
            print(f"verified {relative}: {expected}", flush=True)


if __name__ == "__main__":
    main()
