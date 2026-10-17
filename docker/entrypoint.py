#!/usr/bin/env python3
"""Strata's container start: prepare, then hand over to the server.

Everything the repository's setup.py would fetch on a first run was fetched at image build
time instead (see docker/Dockerfile), so this touches no network.  The two things that
depend on the MODEL rather than on the image are done here, once:

  1. the tokenizer, exported from the mounted GGUF into the /data volume.  It reads the
     GGUF's metadata only - never the weights - so it takes a moment on a 23 GB file, and
     tools/strata_tokenizer.py writes <pack>/tokenizer/{vocab.json,merges.txt,...}.
  2. the engine's run config (strata-<model>.json), which is the recorded set of choices
     setup.py would have written: the engine's path, its arguments, the context length, the
     tokenizer directory and the log.

Both are skipped when they are already on the volume, so a restart is instant and a
recreated container reuses the volume.  The config is rewritten only when the settings it
would contain actually changed (say a different CONTEXT), so an edit you leave in the file
is not silently replaced by a stale one.

CONFIG=<path> uses a config you wrote yourself, as it is, and writes nothing.

Lastly the server is exec'd - not spawned - so it becomes PID 1 and `docker stop` sends
SIGTERM to the process that can shut the engine down cleanly.
"""
from __future__ import annotations

import json
import os
import pathlib
import shlex
import subprocess
import sys

# The image's copy of the project.  Hardcoded rather than derived from __file__: this script
# is installed at /usr/local/bin/strata-entrypoint, which is not inside the tree.
ROOT = pathlib.Path("/opt/strata")

PY = sys.executable
ENGINE = ROOT / "engine" / "strata"
SERVER = ROOT / "serve" / "server.py"
TOKENIZER_TOOL = ROOT / "tools" / "strata_tokenizer.py"
CUDA_LIB_DIR = "/usr/local/cuda/lib64"

# The model: a file bind-mounted read-only.  Its name inside the container is ours to
# choose, which is what lets a symlinked model on the host be mounted at all (Docker cannot
# follow a symlink out of the directory it mounts).
MODEL = pathlib.Path(os.environ.get("STRATA_MODEL", "/models/model.gguf"))
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen3.6-35b-a3b").strip()
MODEL_ARCH = os.environ.get("MODEL_ARCH", "qwen35moe").strip()

DATA = pathlib.Path(os.environ.get("STRATA_DATA", "/data"))
HOST = os.environ.get("HOST", "0.0.0.0").strip()
PORT = os.environ.get("PORT", "8080").strip()
# The context is how much text the model keeps in mind, and its KV cache is the VRAM that
# costs: ~1.5 GiB at 32K for this family on an 8 GiB card, which is a fifth of the card.  So
# the default is chosen from the card's size when CONTEXT is not given - the same tiers
# setup.py uses, with a smaller one for cards below its 12 GB floor.
CONTEXT = os.environ.get("CONTEXT", "").strip()
# Any further engine arguments, verbatim (the escape hatch for options this script does not
# model).  Parsed like a shell word list, so quoting works.
ENGINE_ARGS_EXTRA = os.environ.get("ENGINE_ARGS_EXTRA", "").strip()
API_KEY = (os.environ.get("API_KEY") or os.environ.get("STRATA_API_KEY") or "").strip()
CONFIG_OVERRIDE = os.environ.get("CONFIG", "").strip()

# Container environment variables that are this script's own inputs, not the engine's.
RESERVED_ENV = {"STRATA_DATA", "STRATA_MODEL", "STRATA_API_KEY", "STRATA_TOKENIZER"}

TAG = MODEL_NAME
# The same layout setup.py uses on a PC: <data>/packs/<tag>/tokenizer, <data>/config/strata-<tag>.json.
TOKENIZER = DATA / "packs" / TAG / "tokenizer"
GENERATED_CONFIG = DATA / "config" / f"strata-{TAG}.json"
LOG = DATA / f"strata-{TAG}.log"

LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")


def say(msg: str) -> None:
    print(f"strata: {msg}", flush=True)


def die(msg: str, hint: str = "") -> None:
    print(f"strata: ERROR: {msg}", file=sys.stderr, flush=True)
    if hint:
        print(f"strata:        {hint}", file=sys.stderr, flush=True)
    sys.exit(1)


def gpu_vram_gib() -> float | None:
    """The first card's total VRAM in GiB, from nvidia-smi; None when it cannot say."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20, check=True).stdout
        return float(out.strip().splitlines()[0]) / 1024.0
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def default_context(vram_gib: float | None) -> str:
    """The context to serve when CONTEXT is not given: the KV cache is VRAM, so a small card
    gets a small one.  The 16.5/24.5 GiB steps are setup.py's (own_path_context); the 10 GiB
    step covers the cards below its 12 GB floor, where 32K would eat a fifth of the card."""
    if vram_gib is None:
        return "8192"
    if vram_gib <= 10:
        return "8192"
    if vram_gib <= 16.5:
        return "32768"
    if vram_gib <= 24.5:
        return "65536"
    return "131072"


def export_tokenizer() -> None:
    """The tokenizer goes next to the pack the engine would use, on the volume, once."""
    if (TOKENIZER / "vocab.json").exists():
        say(f"tokenizer: already exported ({TOKENIZER})")
        return
    say(f"tokenizer: exporting from {MODEL.name} (GGUF metadata only, no weights are read) ...")
    TOKENIZER.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([PY, str(TOKENIZER_TOOL), "--gguf", str(MODEL), "--out", str(TOKENIZER.parent)], check=True)
    if not (TOKENIZER / "vocab.json").exists():
        die(f"the tokenizer export wrote nothing to {TOKENIZER}",
            "check that the GGUF carries a tokenizer (tokenizer.ggml.tokens)")


def engine_env() -> dict:
    """Engine settings given as container environment variables, into the config's "env" block.

    The engine reads a number of switches from its environment (STRATA_Q36_NO_MMQ,
    STRATA_Q36_NO_MTP_CPU, STRATA_MMQ_KQ, ...), and serve/server.py passes the config's "env"
    to the engine it starts.  So any STRATA_* variable set on `docker run` is handed on, which
    is how the project's own escape hatches stay reachable through this container.  The few
    this script consumes itself are left out: they are not the engine's.
    """
    return {k: v for k, v in os.environ.items()
            if k.startswith("STRATA_") and k not in RESERVED_ENV and v}


def engine_args(context: str) -> list[str]:
    """The engine's arguments: what setup.py writes for this family.  Any further argument is
    added verbatim through ENGINE_ARGS_EXTRA - there are no MTP/POOL_WORKERS/EXPERT_VRAM_GIB
    variables here, those are engine arguments and pass as such."""
    args = ["--native", str(MODEL), "--max-context", context, "--prefill", "auto", "--mtp", "2"]
    if ENGINE_ARGS_EXTRA:
        args += shlex.split(ENGINE_ARGS_EXTRA)
    return args


def run_config(context: str) -> dict:
    cfg = {
        "exe": str(ENGINE),
        "args": engine_args(context),
        "cwd": str(ROOT),
        "tokenizer": str(TOKENIZER),
        "model_name": MODEL_NAME,
        "arch": MODEL_ARCH,          # how serve/ knows this family's --mtp is a draft count
        "log": str(LOG),
        "lib_dirs": [CUDA_LIB_DIR],
        "port": int(PORT) if PORT.isdigit() else PORT,
        "host": HOST,
    }
    env = engine_env()
    if env:
        cfg["env"] = env
    if API_KEY:
        cfg["api_key"] = API_KEY
    return cfg


def write_config(context: str) -> pathlib.Path:
    """Write the generated config, or keep the one already there.  Returns the path to use."""
    if CONFIG_OVERRIDE:
        path = pathlib.Path(CONFIG_OVERRIDE)
        if not path.is_file():
            die(f"CONFIG={CONFIG_OVERRIDE} does not exist")
        say(f"config: {path} (from CONFIG, used as it is)")
        return path

    cfg = run_config(context)
    path = GENERATED_CONFIG
    path.parent.mkdir(parents=True, exist_ok=True)

    current = None
    if path.is_file():
        try:
            current = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            say(f"config: {path} does not parse; rewriting it")
    if current == cfg:
        say(f"config: {path} (already written)")
        return path

    tmp = path.with_name(path.name + ".tmp")          # whole or not at all, as setup.py writes it
    tmp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    say(f"config: {path} ({'settings changed, rewritten' if current is not None else 'written'})")
    return path


def main() -> None:
    if not ENGINE.is_file():
        die(f"the engine is missing ({ENGINE})", "the image did not build correctly")
    if not MODEL.is_file():
        die(f"no model at {MODEL}",
            "mount a directory that holds your GGUF read-only and point STRATA_MODEL at it, "
            "e.g. -v /dir:/models:ro -e STRATA_MODEL=/models/your-model.gguf (docker/run.sh does this)")

    DATA.mkdir(parents=True, exist_ok=True)

    size_gib = MODEL.stat().st_size / 2**30
    say(f"model: {MODEL} ({size_gib:.1f} GiB)")
    say(f"data:  {DATA} (tokenizer, config and the engine's log live here)")

    # Serving beyond this container without a key would hand the model to anyone who can
    # reach the port.  The container always binds 0.0.0.0 (it has to, to be reachable
    # through a published port); what decides who can reach it is the host-side publish,
    # which docker/run.sh pins to 127.0.0.1 unless BIND_HOST says otherwise.
    if HOST not in LOCAL_HOSTS and not API_KEY:
        say(f"note: no API key.  This container listens on {HOST}:{PORT} inside its own namespace, so only what the "
            f"publish maps (docker/run.sh: 127.0.0.1) can reach it.  Publishing it further? Set API_KEY=...")

    # The context is what decides whether the model fits, so the number actually used is said
    # out loud rather than left to the config file.
    vram = gpu_vram_gib()
    context = CONTEXT or default_context(vram)
    say(f"context: {context} tokens" + (" (from CONTEXT)" if CONTEXT else " (chosen for your GPU)"))
    env = engine_env()
    if env:
        say("engine settings: " + ", ".join(f"{k}={v}" for k, v in sorted(env.items())))

    export_tokenizer()
    cfg_path = write_config(context)

    cmd = [PY, str(SERVER), "--engine", "strata", "--config", str(cfg_path), "--port", str(PORT), "--host", HOST]
    if API_KEY:
        cmd += ["--api-key", API_KEY]
    say(f"starting the server on {HOST}:{PORT} - the model loads first (1-3 minutes, longer on a cold cache)")
    os.execv(PY, cmd)


if __name__ == "__main__":
    main()
