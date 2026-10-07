"""tools/glm_setup.py - prepare a GLM-5.3 run: the tokenizer folder, the run config, the launcher.

GLM-5.3 runs on its own engine, `strata-glm` (src/glm/, docs/GLM53.md), behind the same server and web app as the
Qwen models.  The installer offers it (`setup.py --family glm`, tools/glm_install.py, which calls `prepare` below);
this script does the same three steps for a model folder that is already on the disk:

  1. the tokenizer folder the server reads (vocab.json, merges.txt, token_type.json, tokenizer.json with the `glm`
     pre-tokenizer, and serve/glm/chat_template.jinja - the container ships no chat template);
  2. strata-glm53.json: the engine, its arguments, the tokenizer, `"family": "glm"`;
  3. run-glm53.bat (Windows) or run-glm53.sh: the server on that config.

    python tools/glm_setup.py --model D:/models/GLM-5.3-colibri-int4-g64 [--exe build/strata-glm.exe]
                              [--max-context 8192] [--ram-gb 0] [--port 8080]

Nothing is downloaded and the model folder is only read.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
WIN = os.name == "nt"
sys.path.insert(0, str(ROOT / "tools"))


def check_model(model: pathlib.Path) -> dict:
    cfg_path = model / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"{model} has no config.json: not a model folder")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("model_type") != "glm_moe_dsa":
        raise SystemExit(f"{model}: model_type is {cfg.get('model_type')!r}; strata-glm runs 'glm_moe_dsa' (GLM-5.2/5.3)")
    if not (model / "tokenizer.json").exists():
        raise SystemExit(f"{model} has no tokenizer.json")
    shards = sorted(model.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"{model} has no .safetensors shards")
    numbered = [re.fullmatch(r".*-(\d+)-of-(\d+)\.safetensors", p.name) for p in shards]
    if any(numbered):
        totals = {int(m[2]) for m in numbered if m}
        if len(totals) != 1 or any(m is None for m in numbered):
            raise ValueError("inconsistent safetensors shard names")
        total = totals.pop()
        if {int(m[1]) for m in numbered} != set(range(1, total + 1)):
            raise ValueError(f"incomplete model: expected all {total} safetensors shards")
    return cfg


def find_exe(given: str | None) -> pathlib.Path:
    name = "strata-glm.exe" if WIN else "strata-glm"
    candidates = [pathlib.Path(given)] if given else [ROOT / "engine" / name, ROOT / "build" / name,
                                                      ROOT / "build" / "Release" / name]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise SystemExit("strata-glm was not found (looked in: " + ", ".join(str(c) for c in candidates) + "). Build it: "
                     "cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build --target strata-glm")


def prepare(model, exe, *, root=ROOT, max_context=8192, ram_gb=0, port=8080, name="glm53",
            kv="f32", gpu=None, prefetch=False, host="127.0.0.1", api_key=None, open_browser=True,
            python=None):
    """Write a GLM configuration and launcher. Shared by the installer and the standalone helper."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("name must contain only letters, numbers, underscores and hyphens")
    if not 16 <= max_context <= 131072 or not 1 <= port <= 65535 or ram_gb < 0:
        raise ValueError("invalid context, port or RAM budget")
    if kv not in ("f32", "bf16", "fp8") or (gpu is not None and gpu < 0):
        raise ValueError("invalid KV format or GPU number")
    if host not in ("127.0.0.1", "localhost", "::1") and not api_key:
        raise ValueError("a non-loopback host requires an API key")
    model, exe, root = pathlib.Path(model).resolve(), pathlib.Path(exe).resolve(), pathlib.Path(root).resolve()
    check_model(model)
    if not exe.is_file():
        raise ValueError(f"engine not found: {exe}")
    import strata_tokenizer as ST
    tok_dir = root / "data" / name / "tokenizer"
    ST.extract_hf(model / "tokenizer.json", tok_dir, "glm", ROOT / "serve/glm/chat_template.jinja")
    args = ["--model", str(model), "--max-context", str(max_context), "--kv", kv]
    if ram_gb > 0:
        args += ["--ram-gb", str(ram_gb)]
    if gpu is not None:
        args += ["--gpu", str(gpu)]
    if prefetch:
        args += ["--prefetch"]
    cfg_path = root / f"strata-{name}.json"
    # Keep server preferences on reconfiguration, while replacing engine-specific fields.
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    cfg.update(exe=str(exe), args=args, cwd=str(root), tokenizer=str(tok_dir), model_name="glm-5.3", family="glm",
               log=str(root / f"strata-{name}.log"), port=port, host=host, open_browser=open_browser,
               engine_silence_s=1800)
    if api_key is not None:
        cfg["api_key"] = api_key
    for unsupported in ("vision", "parallel", "vram_elastic", "layer_split", "gpu", "effort_position"):
        cfg.pop(unsupported, None)
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    # the server takes the port from --port only (its default is 8095), so the launcher says it
    command = [str(python or sys.executable), str(root / "serve/server.py"), "--engine", "strata", "--config", str(cfg_path),
               "--port", str(port)]
    if open_browser:
        command.append("--open")
    script = root / (f"run-{name}.bat" if WIN else f"run-{name}.sh")
    if WIN:
        # Disable delayed expansion and escape literal percent signs in paths for cmd.exe.
        cmd = subprocess.list2cmdline(command).replace("%", "%%")
        text = '@echo off\nsetlocal DisableDelayedExpansion\ntitle Strata GLM-5.3\ncd /d "' + str(root).replace("%", "%%") + '"\n' + cmd + '\nif errorlevel 1 pause\n'
        script.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
    else:
        script.write_text("#!/bin/sh\ncd " + shlex.quote(str(root)) + " || exit\nexec " + shlex.join(command) + "\n", encoding="utf-8")
        script.chmod(0o755)
    return cfg_path, script


def start_config(path, *, port=None, open_browser=None, keep=None):
    path = pathlib.Path(path)
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if keep:
        for k in ("host", "api_key", "open_browser"):
            if keep.get(k) is not None:
                cfg[k] = keep[k]
        cfg_file = json.dumps(cfg, indent=2) + "\n"
        path.write_text(cfg_file, encoding="utf-8")
    cmd = [sys.executable, str(ROOT / "serve/server.py"), "--engine", "strata", "--config", str(path),
           "--port", str(port or cfg.get("port") or 8080)]   # the server's own default (8095) is not the config's
    if cfg.get("open_browser", True) if open_browser is None else open_browser:
        cmd += ["--open"]
    return subprocess.call(cmd, cwd=ROOT)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the GLM-5.3 int4-g64 folder (config.json, tokenizer.json, shards)")
    ap.add_argument("--exe", help="the strata-glm executable (default: engine/ or build/ in this folder)")
    ap.add_argument("--max-context", type=int, default=8192,
                    help="tokens of context (the KV cache takes 180 KB per token in RAM; default 8192 = 1.5 GB)")
    ap.add_argument("--ram-gb", type=float, default=0.0,
                    help="the engine's RAM budget in GB (0: 85%% of the RAM free when it starts)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--kv", choices=["f32", "bf16", "fp8"], default="f32")
    ap.add_argument("--gpu", type=int, help="CUDA device number; omit for CPU execution")
    ap.add_argument("--prefetch", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--api-key")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--name", default="glm53", help="the name in the config and launcher file names")
    a = ap.parse_args()

    cfg_path, script = prepare(a.model, find_exe(a.exe), max_context=a.max_context, ram_gb=a.ram_gb,
                               port=a.port, name=a.name, kv=a.kv, gpu=a.gpu, prefetch=a.prefetch,
                               host=a.host, api_key=a.api_key, open_browser=not a.no_browser)
    print(f"config: {cfg_path}")
    print(f"launcher: {script}")
    print("GLM-5.3 needs about 20 GB of free RAM to start (11.6 GB of weights resident, the KV, the smallest expert "
          "cache); the rest of the RAM becomes its expert cache.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
