"""Copy build-time third-party notices into a fresh, hash-inventoried directory."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("out","ggml-source","cuda-dir","node-dir","host-node-modules","control-dir","microsoft-documents"):
        parser.add_argument("--"+name,required=True,type=Path)
    args=parser.parse_args()
    out=args.out.resolve()
    if out.exists():
        raise SystemExit("Notice stages are immutable; use a fresh output directory")
    if subprocess.check_output(["git","-C",str(args.ggml_source),"rev-parse","HEAD"],text=True).strip()!="3cf03257f219afbe7334045ff7c6a06ac68c627d":
        raise SystemExit("Unexpected ggml licence source")
    entries={}
    def copy(source,name,expected=None):
        source=Path(source)
        if not source.is_file() or source.is_symlink() or source.stat().st_size>8*1024*1024:
            raise ValueError("Missing, linked or oversized notice: "+str(source))
        if expected and sha(source)!=expected:
            raise ValueError("Pinned vendor notice changed: "+name)
        target=out/name
        if not target.resolve().is_relative_to(out) or name in entries:
            raise ValueError("Unsafe or duplicate notice member")
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        entries[name]=sha(target)
    copy(ROOT/"LICENSE","Strata-MIT.txt")
    copy(args.ggml_source/"LICENSE","ggml-MIT.txt")
    libstrix_notice=ROOT/"third_party/libstrix/NOTICE.md"
    if libstrix_notice.is_file():
        copy(libstrix_notice,"libstrix-NOTICE.md")
    copy(args.cuda_dir/"EULA.txt","NVIDIA-EULA.txt")
    copy(args.cuda_dir/"LICENSE","NVIDIA-LICENSE.txt")
    copy(args.node_dir/"LICENSE","Node-LICENSE.txt","ed34dd8e3f0a78dbaf00d0444ce8e285b015b765379c2e17880455f70370f8e9")
    copy(ROOT/"packaging/hakua-third-party-notices.md","README.md")
    qwen=json.loads((ROOT/"data/experimental-speed-projection/PROVENANCE.json").read_text(encoding="utf-8"))["license"]
    copy(ROOT/"data/experimental-speed-projection/LICENSE","Qwen-Community-License-1.0.txt",qwen["sha256"])
    vendor=json.loads((ROOT/"packaging/microsoft-2022-notice-sources.json").read_text(encoding="utf-8"))
    for item in vendor:
        if not re.fullmatch(r"[A-Za-z0-9.-]+",item["file"]):
            raise ValueError("Unsafe vendor notice name")
        copy(args.microsoft_documents/item["file"],item["file"],item["sha256"])
    for package in ("react","react-dom","scheduler","katex","prismjs","pdfjs-dist","html-to-image"):
        folder=args.host_node_modules/package
        found=False
        for source in sorted(folder.rglob("*")):
            if source.is_file() and re.match(r"^(license|licence|notice|copying)",source.name,re.I):
                copy(source,"gui/"+package+"/"+source.relative_to(folder).as_posix())
                found=True
        if not found:
            raise ValueError("Missing GUI dependency licence: "+package)
    python=args.control_dir/"python"
    python_licenses=[p for p in python.iterdir() if p.is_file() and p.name.upper().startswith("LICENSE")]
    if not python_licenses:
        raise ValueError("Missing Python licence")
    for source in python_licenses:
        copy(source,"python/"+source.name)
    site=python/"Lib/site-packages"
    for info in sorted(site.glob("*.dist-info")):
        for source in sorted(info.rglob("*")):
            if source.is_file() and (re.match(r"^(license|licence|notice|copying)",source.name,re.I) or "licenses" in source.relative_to(info).parts):
                copy(source,"python/distributions/"+info.name+"/"+source.relative_to(info).as_posix())
    (out/"NOTICES.json").write_text(json.dumps({"schema":1,"microsoftSources":vendor,"qwenSource":qwen,"files":entries},indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"notices":len(entries),"directory":str(out)}))

if __name__=="__main__":
    main()
