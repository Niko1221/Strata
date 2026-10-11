"""等 Strata 載入完成：每 2 秒打 /health 與 /metrics，記 RAM／VRAM 曲線，loaded=true 才結束。

    python wait_ready.py --tag coder [--url http://127.0.0.1:8080] [--max-min 30]
輸出：_strata_test/03_startup_<tag>.json（耗時＋曲線）
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import requests

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="run")
ap.add_argument("--url", default="http://127.0.0.1:8080")
ap.add_argument("--max-min", type=float, default=30)
A = ap.parse_args()
t0 = time.time()
curve, health, first_http = [], None, None
while time.time() - t0 < A.max_min * 60:
    row = {"s": round(time.time() - t0, 1)}
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10).stdout.strip()
        row["vram_mib"] = float(q)
    except Exception:  # noqa: BLE001
        pass
    try:
        import psutil
        vm = psutil.virtual_memory()
        row["ram_used_gib"] = round((vm.total - vm.available) / 2**30, 2)
        row["swap_used_gib"] = round(psutil.swap_memory().used / 2**30, 2)
    except Exception:  # noqa: BLE001
        pass
    try:
        health = requests.get(A.url + "/health", timeout=5).json()
        if first_http is None:
            first_http = row["s"]
        row["loaded"] = health.get("loaded")
    except Exception:  # noqa: BLE001
        row["loaded"] = None
    curve.append(row)
    if row["s"] % 30 < 2.1:
        print(json.dumps(row), flush=True)
    if row.get("loaded") is True:
        break
    time.sleep(2)
out = {"tag": A.tag, "http_up_s": first_http, "loaded_s": curve[-1]["s"] if curve and curve[-1].get("loaded") else None,
       "health": health, "peak_ram_gib": max((r.get("ram_used_gib") or 0) for r in curve),
       "peak_vram_mib": max((r.get("vram_mib") or 0) for r in curve), "curve": curve}
Path(rf"E:\Strata\_strata_test\03_startup_{A.tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print("READY" if out["loaded_s"] else "NOT READY", json.dumps({k: v for k, v in out.items() if k != "curve"}), flush=True)
