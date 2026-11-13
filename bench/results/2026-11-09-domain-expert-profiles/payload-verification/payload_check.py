"""Read-only GGUF payload validation against a frozen Hugging Face LFS manifest.

CPU only. Results, progress and the process handle stay beside this script.
Input payloads are opened as rb and never modified. Repeated start calls do
not launch another worker. No model loader or GPU library is imported.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
META = HERE / "source-model-metadata.json"
EXPECTED_META_SHA = "4c11beae2055edfaf739a04733a131cd8ccffb7bef3bbe3b625af08b07ab5634"


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(name, value):
    target = HERE / name
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


def load_metadata():
    raw = META.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_META_SHA
    data = json.loads(raw)
    assert data["revision"] == "ed59f92082b1e93c0e96d60a8b11aab089b52f09"
    assert len(data["files"]) == 2
    for entry in data["files"]:
        rel = Path(entry["rfilename"])
        assert not rel.is_absolute() and ".." not in rel.parts
        assert entry["size"] == entry["lfs"]["size"]
        assert len(entry["lfs"]["sha256"]) == 64
    return data


def signature(path):
    st = path.stat()
    return dict(device=st.st_dev, inode=st.st_ino, bytes=st.st_size,
                mtime_ns=st.st_mtime_ns)


def proc(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()
        return dict(state=fields[0], start_ticks=int(fields[19]))
    except FileNotFoundError:
        return None


def status():
    if (HERE / "receipt.json").exists():
        return json.loads((HERE / "receipt.json").read_text())
    if not (HERE / "handle.json").exists():
        return {"status": "not_started"}
    handle = json.loads((HERE / "handle.json").read_text())
    current = proc(handle["pid"])
    live = bool(current and current["start_ticks"] == handle["start_ticks"]
                and current["state"] != "Z")
    progress = json.loads((HERE / "progress.json").read_text()) if (HERE / "progress.json").exists() else None
    return dict(status="running" if live else "handle_missing_or_terminal",
                handle=handle, progress=progress)


def inspect(model_root):
    meta = load_metadata()
    files = []
    for entry in meta["files"]:
        sig = signature(model_root / entry["rfilename"])
        assert sig["bytes"] == entry["size"]
        files.append(dict(file=entry["rfilename"], observed_bytes=sig["bytes"],
                          expected_sha256=entry["lfs"]["sha256"]))
    return dict(status="inputs_present", metadata_sha256=EXPECTED_META_SHA, files=files)


def worker(model_root):
    started = utc()
    t0 = time.monotonic()
    os.nice(19)
    subprocess.run(["ionice", "-c", "3", "-p", str(os.getpid())], check=True)
    meta = load_metadata()
    result = dict(status="running", started_utc=started, repository=meta["repository"],
                  revision=meta["revision"], metadata_sha256=EXPECTED_META_SHA,
                  verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  cpu_nice=os.getpriority(os.PRIO_PROCESS, 0),
                  io_priority=subprocess.check_output(["ionice", "-p", str(os.getpid())], text=True).strip(),
                  method="sequential full SHA256; read-only 8MiB chunks; stable file stat before/after",
                  files=[])
    try:
        for entry in meta["files"]:
            path = model_root / entry["rfilename"]
            before = signature(path)
            assert before["bytes"] == entry["size"]
            digest = hashlib.sha256()
            total = 0
            last_notice = 0
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
                    total += len(chunk)
                    if total - last_notice >= 512 * 1024 * 1024:
                        save("progress.json", dict(updated_utc=utc(), file=entry["rfilename"],
                             bytes_read=total, file_bytes=entry["size"],
                             completed_files=len(result["files"])))
                        last_notice = total
                    if time.monotonic() - t0 > 1800:
                        raise TimeoutError("Own read-only hashing deadline exceeded")
            after = signature(path)
            got = digest.hexdigest()
            assert before == after, "Input file changed during verification"
            assert total == entry["size"]
            result["files"].append(dict(file=entry["rfilename"], bytes=total,
                   expected_sha256=entry["lfs"]["sha256"], actual_sha256=got,
                   match=got == entry["lfs"]["sha256"], stable_stat=True))
        result["status"] = "succeeded" if all(x["match"] for x in result["files"]) else "hash_mismatch"
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    result.update(finished_utc=utc(), seconds=time.monotonic() - t0)
    save("receipt.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("inspect", "start", "worker", "status"))
    parser.add_argument("--model-root", type=Path)
    args = parser.parse_args()
    if args.mode == "status":
        result = status()
    else:
        assert args.model_root is not None
        if args.mode == "inspect":
            result = inspect(args.model_root)
        elif args.mode == "worker":
            worker(args.model_root)
            return
        else:
            result = status()
            if result["status"] == "not_started":
                inspect(args.model_root)
                lock = os.open(HERE / "start.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.close(lock)
                with (HERE / "worker.log").open("ab") as log:
                    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                            "worker", "--model-root", str(args.model_root)],
                            cwd=HERE, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                            start_new_session=True)
                handle = dict(pid=child.pid, start_ticks=proc(child.pid)["start_ticks"], started_utc=utc())
                save("handle.json", handle)
                result = status()
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
