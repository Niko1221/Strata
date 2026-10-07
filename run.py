#!/usr/bin/env python3
"""Run an installed Strata model from YAML; installation stays in setup.py.

    .venv/bin/python run.py init
    .venv/bin/python run.py models
    .venv/bin/python run.py check
    .venv/bin/python run.py run --config strata.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from serve import launchconfig

ROOT = Path(__file__).resolve().parent


def check_files(cfg: dict):
    exe = Path(cfg["exe"])
    exe = exe if exe.is_absolute() else Path(cfg["cwd"]) / exe
    if not exe.is_file():
        raise ValueError(f"engine not found: {exe}; run ./setup.sh first")
    tokenizer = Path(cfg.get("tokenizer") or ROOT / "pack/full/tokenizer")
    for name in ("vocab.json", "merges.txt", "token_type.json"):
        if not (tokenizer / name).is_file():
            raise ValueError(f"tokenizer not found: {tokenizer / name}; run ./setup.sh first")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["init", "models", "check", "run"], nargs="?", default="run")
    ap.add_argument("--config", type=Path, default=ROOT / "strata.yaml", help="launch YAML file (default: strata.yaml)")
    a = ap.parse_args(argv)
    path = a.config.expanduser().resolve()
    try:
        if a.command == "models":
            models = launchconfig.installed()
            if not models:
                print("No installed models; run ./setup.sh first.")
            for name, config in models.items():
                print(f"{name}: {config}")
            return 0
        if a.command == "init":
            template = (ROOT / "strata.example.yaml").read_text(encoding="utf-8")
            models = launchconfig.installed()
            if models:
                template = template.replace("model: iq2_xs", f"model: {next(iter(models))}", 1)
            with path.open("x", encoding="utf-8") as f:
                f.write(template)
            print(f"Created {path}; edit it, then run make check or make run.")
            return 0
        if not path.is_file():
            raise ValueError(f"launch file not found: {path}; run make init first")
        cfg = launchconfig.load(path)
        check_files(cfg)
        context = launchconfig.arg_value(cfg["args"], "--max-context") or "engine default"
        print(f"{cfg.get('model_name', 'Strata')}: {cfg['host']}:{cfg['port']}, context {context}, "
              f"API key {'set' if cfg['api_key'] else 'unset'}", flush=True)
        if a.command == "check":
            print("Launch config, engine and tokenizer checked; the GPU and model were not loaded.")
            return 0
        mtp = launchconfig.arg_value(cfg["args"], "--mtp")
        if mtp is not None:
            from setup import refresh_draft_vocab
            rt = Path(mtp)
            rt = rt if rt.is_absolute() else Path(cfg["cwd"]) / rt
            refresh_draft_vocab(rt, cfg.get("draft_vocab", "cjk"))
        # Stay in the foreground: the server owns Ctrl+C/SIGTERM and closes its engine on exit.
        os.chdir(ROOT)
        from serve import server
        args = ["--engine", "strata", "--config", str(path)]
        if cfg.get("open_browser") is True:
            args += ["--open"]
        return server.main(args)
    except (OSError, ValueError) as e:
        ap.error(str(e))


if __name__ == "__main__":
    sys.exit(main())
