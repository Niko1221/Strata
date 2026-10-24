"""The decisive arms: engine 0.1.45 built from the v0.1.45 tree locally with CUDA 13.1
(the same compiler and the same local flags as the 0.1.46 arms), twice, so the old code
is bracketed as well.

  r7 = 0.1.45 local @ 13.1
  r8 = 0.1.45 local @ 13.1   (the bracket)

Compared with r3 and r6 (0.1.46 local @ 13.1), this isolates the 0.1.45 -> 0.1.45.3
code change, with the toolkit, the build flags and the arch list all held constant.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"D:\Strata")
OUT = Path(r"D:\Strata-data\ab2")

ARMS = [
    ("r7-0.1.45-cuda131", r"D:\Strata-data\engine-0.1.45-cuda13.1\strata.exe", 8127),
    ("r8-0.1.45-cuda131", r"D:\Strata-data\engine-0.1.45-cuda13.1\strata.exe", 8128),
]

base = json.load(open(ROOT / "strata-swift-iq3_s.json", encoding="utf-8-sig"))
for tag, exe, port in ARMS:
    cfg = dict(base)
    cfg["exe"] = exe
    cfg["port"] = port
    cfg["log"] = str(OUT / f"engine-{tag}.log")
    (OUT / f"strata-{tag}.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    print(f"{tag}: {exe}", flush=True)

runner = Path(r"C:\Users\user\AppData\Local\Temp\opencode\ab_run.py")
for tag, exe, port in ARMS:
    print(f"\n===== {tag} =====", flush=True)
    rc = subprocess.run([sys.executable, str(runner), str(OUT / f"strata-{tag}.json"),
                         str(port), str(OUT / tag)]).returncode
    print(f"{tag} finished with {rc}", flush=True)
print("ALL DONE", flush=True)
