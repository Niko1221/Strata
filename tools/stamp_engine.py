# tools/stamp_engine.py - write engine/BUILD.json the way setup.py's build_engine does, so START-HERE.bat on every
# PC sees "engine already built for this PC" instead of compiling again (the same source hash).  For a pool of PCs
# with different cards: build one engine by hand for all their architectures (e.g. CMAKE_CUDA_ARCHITECTURES=
# 75;86;89;120), copy it to every PC, and stamp it there:
#     .venv\Scripts\python tools\stamp_engine.py "<path to nvcc.exe>" 75,86,89,120 [engine folder]
import hashlib, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SETUP = (ROOT / "setup.py").read_text(encoding="utf-8")   # setup.py's own values, so the two never drift apart
LLAMA_CPP_COMMIT = re.search(r'^LLAMA_CPP_COMMIT = "([0-9a-f]+)"', _SETUP, re.M).group(1)
ENGINE_SOURCES = tuple(re.findall(r'"([^"]+)"', re.search(r"^ENGINE_SOURCES = \((.*)\)", _SETUP, re.M).group(1)))


def source_hash(parts) -> str:
    h = hashlib.sha256(LLAMA_CPP_COMMIT.encode())
    for part in parts:
        base = ROOT / part
        for f in [base] if base.is_file() else sorted(x for x in base.rglob("*") if x.is_file()):
            h.update(f.relative_to(ROOT).as_posix().encode() + b"\0" + f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def main() -> int:
    eng = ROOT / (sys.argv[3] if len(sys.argv) > 3 else "engine")
    exe = eng / ("strata.exe" if (eng / "strata.exe").exists() else "strata")
    m = re.search(rb"engine=(\d+\.\d+\.\d+)\n", Path(exe).read_bytes())
    if not m:
        print("no engine= version string in the exe"); return 1
    nvcc = sys.argv[1]  # path to nvcc.exe, for cuda_dirs
    bindir = Path(nvcc).parent
    dirs = [str(d) for d in (bindir, bindir / "x64", bindir.parent / "lib64") if d.is_dir()]
    archs = sorted(int(a) for a in sys.argv[2].split(","))
    stamp = eng / "BUILD.json"
    old = json.loads(stamp.read_text()) if stamp.exists() else {}
    stamp.write_text(json.dumps({
        "source": "local",
        "version": m.group(1).decode(),
        "archs": archs,
        "vision": old.get("vision", "none"),
        "cuda_dirs": dirs,
        "src": source_hash(ENGINE_SOURCES),
        "vision_src": old.get("vision_src"),
    }, indent=1))
    print("stamped:", stamp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
