#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Этап VIII (п. 11): manifest прогона A/B-стенда — фиксация идентичности сборки и условий.

Пишет manifest.json: sha256 бинарника движка, конфига (без секретов), всех скриптов и
проб стенда, версии инструментов, снимок nvidia-smi, build_info движка из /props (если
сервер поднят), привязка к дереву исходников.

Привязка к commit SHA дерева (п. 11 «отдельная папка для каждого commit SHA»): git на
машине не установлен (вопрос Q1), поэтому идентичность дерева фиксируется двумя
независимыми способами: (а) build_info/версия движка из /props, (б) sha256 бинарника.
Оба поля пишутся в manifest, а каталог прогона именуется <UTC-штамп>-<метка сборки>.

usage: ab-manifest.py --out DIR [--config PATH] [--engine PATH] [--label МЕТКА]
"""
import argparse
import datetime
import hashlib
import json
import pathlib
import platform
import re
import subprocess
import sys
import urllib.request

HOME = pathlib.Path.home()
PROJ = HOME / "strata185" / "Strata+4xV100+NVlink"
D = HOME / "strata185"
SECRET_KEYS = re.compile(r"api_key|token|secret|password", re.I)


def sha256(p):
    p = pathlib.Path(p)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sh(cmd, timeout=30):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception:
        return ""


def redact(obj):
    if isinstance(obj, dict):
        return {k: ("***" if SECRET_KEYS.search(k) else redact(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    return obj


def props_build_info():
    """build_info движка из /props (сервер должен быть поднят). Секрет берём из рабочего конфига."""
    cfgp = D / "Strata" / "strata-185-iq3_s.json"
    try:
        cfg = json.loads(cfgp.read_text(encoding="utf-8-sig"))
        key = cfg.get("api_key", "")
        req = urllib.request.Request("http://127.0.0.1:8080/props", headers={"Authorization": "Bearer " + key})
        with urllib.request.urlopen(req, timeout=10) as r:
            body = json.loads(r.read().decode())
        return {"ok": True, "build_info": body.get("build_info"), "model": body.get("model"),
                "version": body.get("version")}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(D / "Strata" / "strata-185-iq3_s.json"))
    ap.add_argument("--engine", default=str(D / "Strata" / "engine-cuda12" / "strata"))
    ap.add_argument("--label", default="base")
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cfgp = pathlib.Path(a.config)
    cfg = json.loads(cfgp.read_text(encoding="utf-8-sig")) if cfgp.exists() else {}

    m = {
        "kind": "ab-stand manifest",
        "project": str(PROJ),
        "captured_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "build_label": a.label,
        "note": a.note,
        "engine_binary": {"path": str(a.engine), "sha256": sha256(a.engine)},
        "config": {"path": str(cfgp), "sha256": sha256(cfgp),
                   "args": redact(cfg.get("args")), "env": redact(cfg.get("env")),
                   "layer_split": (cfg["args"][cfg["args"].index("--layer-split") + 1]
                                   if "--layer-split" in cfg.get("args", []) else None),
                   "port": cfg.get("port"), "gpu": cfg.get("gpu"), "max_context": None},
        "tree_identity": {
            "git": "git не установлен на машине (открытый вопрос Q1) — SHA дерева недоступен",
            "engine_build_info": props_build_info(),
            "binary_sha256_is_primary_identity": True,
        },
        "host": {
            "hostname": platform.node(),
            "kernel": platform.release(),
            "python": sys.version.split()[0],
            "driver": sh(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).splitlines()[:1],
            "gpus": sh(["nvidia-smi", "--query-gpu=index,name,compute_cap,memory.total", "--format=csv,noheader"]).splitlines(),
            "nvcc": sh([str(D / "cuda-12.8/bin/nvcc"), "--version"]).splitlines()[-1:] ,
            "host_cc": sh([str(D / "gcc185/usr/bin/g++-14"), "--version"]).splitlines()[:1],
        },
        "artifacts_sha256": {},
        "nvidia_smi_before": sh(["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu,"
                                                 "temperature.gpu,power.draw,clocks.current.sm", "--format=csv"]),
    }
    # все скрипты и пробы стенда — их sha256 входят в manifest
    for p in sorted(list((PROJ / "scripts").glob("*")) + list((PROJ / "source").glob("*")) +
                   list((PROJ / "build").glob("*")) + list((PROJ / "tests").rglob("*"))):
        if p.is_file() and not p.name.startswith("__"):
            m["artifacts_sha256"][str(p.relative_to(PROJ))] = sha256(p)

    (out / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("manifest:", out / "manifest.json")
    print("  engine sha256:", m["engine_binary"]["sha256"])
    print("  config sha256:", m["config"]["sha256"], "layer-split:", m["config"]["layer_split"])
    print("  build_info:", m["tree_identity"]["engine_build_info"].get("build_info"))


if __name__ == "__main__":
    main()
