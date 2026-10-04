#!/usr/bin/env python3
"""serve_dsv4.py - put Strata's OpenAI/Anthropic API and web app on top of dsv4_run.

Additive by rule: nothing under the Strata checkout is edited. This script IMPORTS serve.server,
swaps its engine class for one that spawns ./build-out/dsv4_run --serve instead of strata --serve,
and calls its main(). dsv4_run already speaks the same line protocol (READY / INFO / GEN / PP / T /
DONE / ERR / STOP / QUIT), so the swap is a subclass with no behaviour of its own.

    python3 tools/serve_dsv4.py --model /path/DeepSeek-V4-Flash-...-00001-of-00003.gguf --port 8095
    python3 tools/serve_dsv4.py --cuda --model ... --port 8095      # the real CUDA engine (dsv4_run_cuda)

Then:  http://127.0.0.1:8095/            the web app (Chat / Monitor / About)
       http://127.0.0.1:8095/v1/chat/completions      OpenAI
       http://127.0.0.1:8095/v1/messages              Anthropic

The tokenizer is taken out of the GGUF's metadata (no weights needed) with Strata's own
tools/strata_tokenizer.py, into <workdir>/tokenizer/, because serve/server.py needs vocab.json,
merges.txt, token_type.json and chat_template.jinja.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT_ROOT = HERE.parent
STRATA_ROOT = EXT_ROOT.parent          # dsv4_ext sits inside the Strata checkout
for p in (str(STRATA_ROOT), str(STRATA_ROOT / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)


def build_engine_args(a) -> list[str]:
    """The dsv4_run command line, WITHOUT --serve: StrataEngine adds that itself."""
    args = ["--model", str(Path(a.model).resolve()), "--ctx", str(a.ctx)]
    if a.threads:
        args += ["--threads", str(a.threads)]
    if a.expert_vram_pct is not None:
        args += ["--expert-vram-pct", str(a.expert_vram_pct)]
    if a.expert_cache is not None:
        args += ["--expert-cache", str(a.expert_cache)]
    if a.expert_vram_reserve_mib is not None:
        args += ["--expert-vram-reserve-mib", str(a.expert_vram_reserve_mib)]
    if a.vram_total_mib is not None:
        args += ["--vram-total-mib", str(a.vram_total_mib)]
    if a.vram_free_mib is not None:
        args += ["--vram-free-mib", str(a.vram_free_mib)]
    if a.expert_profile:
        args += ["--expert-profile", str(Path(a.expert_profile).resolve())]
    if a.expert_profile_save:
        args += ["--expert-profile-save", str(Path(a.expert_profile_save).resolve())]
    if a.expert_ram:
        args += ["--expert-ram", a.expert_ram]
    if a.ram_cache_mib is not None:
        args += ["--ram-cache-mib", str(a.ram_cache_mib)]
    if a.expert_adapt_swaps:
        args += ["--expert-adapt-swaps", str(a.expert_adapt_swaps)]
    if a.cpu:
        args += ["--cpu"]
    if a.no_qat_sim:
        args += ["--no-qat-sim"]
    if a.max_layers:
        args += ["--max-layers", str(a.max_layers)]
    if a.eos_ids:
        args += ["--eos-ids", a.eos_ids]
    # extra dsv4_run flags, verbatim: --engine-arg=--verify-slots (repeatable)
    for extra in a.engine_arg or []:
        args += extra.split()
    return args


def patch_tokenizer_byte_fallback() -> None:
    """Let token_bytes() survive a vocabulary that is not purely byte-level.

    Byte-level BPE stores every token as characters from the 256-entry byte alphabet, which is what
    token_bytes() reverses. DeepSeek-V4's vocabulary is not purely that: it carries literal unicode
    (fullwidth punctuation and similar) in ordinary entries. For those, UNICODE_TO_BYTE has no entry
    and token_bytes() raises KeyError in the middle of a stream, which kills the connection handler.

    The fallback encodes such a token as UTF-8, which is what the string already is. It is installed
    here rather than in tools/strata_tokenizer.py because dsv4_ext never edits the Strata checkout.
    """
    import strata_tokenizer as ST
    if getattr(ST.Tokenizer, "_dsv4_byte_fallback", False):
        return
    original = ST.Tokenizer.token_bytes

    def token_bytes(self, i: int) -> bytes:
        try:
            return original(self, i)
        except KeyError:
            cache = self.__dict__.setdefault("_bytes_cache", {})
            b = cache.get(i)
            if b is None:
                b = cache[i] = self.tokens[i].encode("utf-8")
            return b

    ST.Tokenizer.token_bytes = token_bytes
    ST.Tokenizer._dsv4_byte_fallback = True


def ensure_tokenizer(a, workdir: Path) -> Path:
    """<workdir>/tokenizer/ from the GGUF, or reuse what is already there."""
    out = Path(a.tokenizer_dir) if a.tokenizer_dir else workdir
    tdir = out / "tokenizer"
    if (tdir / "vocab.json").exists():
        return tdir
    sys.path.insert(0, str(STRATA_ROOT / "tools"))
    import strata_tokenizer as ST
    print(f"extracting the tokenizer from {Path(a.model).name} ...", flush=True)
    try:
        cfg = ST.extract(a.model, str(out))
    except ValueError as e:
        raise SystemExit(f"the GGUF has no usable tokenizer: {e}")
    print("tokenizer/: vocab %d, merges %d, pre %s" % (cfg["vocab_size"], cfg["n_merges"], cfg["pre"]), flush=True)
    return tdir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the FIRST shard of the GGUF (…-00001-of-00003.gguf)")
    ap.add_argument("--exe", default=None, help="the dsv4_run binary (default: build-out/dsv4_run, or dsv4_run_cuda with --cuda)")
    ap.add_argument("--cuda", action="store_true", help="use build-out/dsv4_run_cuda: real CUDA device layer (sh build.sh --cuda)")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to also answer to your network (set --api-key)")
    ap.add_argument("--ctx", type=int, default=2048, help="the engine's context (READY <ctx>)")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--cpu", action="store_true", help="no GPU layer at all")
    ap.add_argument("--workdir", default=str(EXT_ROOT / "serve-work"), help="tokenizer, config and engine log go here")
    ap.add_argument("--tokenizer-dir", default=None, help="reuse an existing tokenizer directory (its tokenizer/ is used)")
    ap.add_argument("--model-name", default="deepseek-v4-flash", help="the name /v1/models reports")
    ap.add_argument("--api-key", default=os.environ.get("DSV4_API_KEY", ""))
    ap.add_argument("--open", action="store_true", help="open the web app once the model is ready")
    ap.add_argument("--lazy", action="store_true", help="load the model on the first request instead of at start")
    # dsv4_run's own knobs, passed through
    ap.add_argument("--expert-vram-pct", default=None, help="share of VRAM for hot experts, or 'auto'")
    ap.add_argument("--expert-cache", type=int, default=None, help="absolute number of VRAM expert slots")
    ap.add_argument("--expert-vram-reserve-mib", type=int, default=None)
    ap.add_argument("--vram-total-mib", type=int, default=None, help="override the driver's VRAM total")
    ap.add_argument("--vram-free-mib", type=int, default=None, help="override the driver's free VRAM")
    ap.add_argument("--expert-profile", default=None)
    ap.add_argument("--expert-profile-save", default=None,
                    help="write the routing profile here when the engine stops (builds the profile the next start reads)")
    ap.add_argument("--expert-ram", choices=["all", "profile"], default=None)
    ap.add_argument("--ram-cache-mib", type=int, default=None,
                    help="host-RAM LRU arena for MISS experts (engine default 8192, 0 = read straight from disk)")
    ap.add_argument("--expert-adapt-swaps", type=int, default=None)
    ap.add_argument("--no-qat-sim", action="store_true")
    ap.add_argument("--max-layers", type=int, default=None, help="debug: run the first N layers only")
    ap.add_argument("--eos-ids", default=None, help="comma-separated ids that end a turn (default: the GGUF's eos)")
    ap.add_argument("--engine-arg", action="append", help="extra dsv4_run flags, verbatim (repeatable)")
    ap.add_argument("--engine-env", action="append", metavar="KEY=VAL",
                    help="environment for the engine process (repeatable). This build's VRAM size in the "
                         "CPU-emulated mode is DSV4_EMULATED_VRAM_MIB (default 8192)")
    a = ap.parse_args()

    if a.exe is None:
        a.exe = str(EXT_ROOT / "build-out" / ("dsv4_run_cuda" if a.cuda else "dsv4_run"))
    exe = Path(a.exe).resolve()
    if not exe.exists():
        raise SystemExit(f"{exe} not found: run sh build.sh"
                         f"{' --cuda' if a.cuda else ''} first")
    if exe.name == "dsv4_run" and a.cuda:
        raise SystemExit("--cuda wants the CUDA binary: pass --exe build-out/dsv4_run_cuda")
    if a.cuda and a.cpu:
        raise SystemExit("--cuda and --cpu contradict each other: --cpu disables the device layer entirely")

    workdir = Path(a.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    tdir = ensure_tokenizer(a, workdir)
    log = str(workdir / "dsv4_engine.log")

    # A --expert-profile that does not exist is not fatal, but it silently turns --expert-ram profile into
    # "no expert in RAM at all": every MISS then reads its weights from the shard. Say so before starting.
    if a.expert_profile and not Path(a.expert_profile).exists():
        print(f"WARNING: --expert-profile {a.expert_profile} does not exist: the engine starts with no profile,\n"
              f"         so --expert-ram {a.expert_ram or 'profile'} keeps ZERO experts in RAM.\n"
              f"         Build one with:  python3 tools/make_expert_profile.py --help", file=sys.stderr, flush=True)
    if a.expert_ram == "all":
        print("note: --expert-ram all copies every expert into host RAM (tens of GiB): expect the page cache "
              "and the model to fight for memory.", file=sys.stderr, flush=True)
    # The CUDA binary asks the driver for its real free VRAM; the emulated one reads this env var instead.
    if a.cuda and any(kv.startswith("DSV4_EMULATED_VRAM_MIB=") for kv in (a.engine_env or [])):
        print("note: DSV4_EMULATED_VRAM_MIB is ignored by the CUDA binary: --expert-vram-pct plans against the\n"
              "      driver's real free VRAM (see --expert-vram-reserve-mib / --vram-free-mib to steer it).",
              file=sys.stderr, flush=True)

    cfg = {
        "exe": str(exe),
        "args": build_engine_args(a),
        "cwd": str(EXT_ROOT),
        "log": log,
        "tokenizer": str(tdir),
        "model_name": a.model_name,
    }
    # serve/server.py's child_env() copies cfg["env"] onto the engine process.
    env = {}
    for kv in a.engine_env or []:
        k, _, v = kv.partition("=")
        if k:
            env[k] = v
    if env:
        cfg["env"] = env
    cfg_path = workdir / "dsv4_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    # Before server.py builds its Tokenizer from tokenizer/vocab.json.
    patch_tokenizer_byte_fallback()
    import serve.server as S

    # --- stop ids ------------------------------------------------------------------------------------------
    # serve/server.py builds Service.stop_ids from the Qwen strings "<|im_end|>" and "<|endoftext|>". DeepSeek's
    # vocabulary has no such tokens, so they become ordinary pieces ('<', '|', 'im', '_', 'end', 'of', 'text', '>'):
    # the first '>' of any HTML (or '|' of a markdown table) ended the reply, and the model's real end-of-sentence
    # token was not a stop id at all (it showed up as text and the reply was labelled "length").
    def _dsv4_stop_ids() -> set:
        if a.eos_ids:
            return {int(x) for x in a.eos_ids.replace(" ", "").split(",") if x}
        try:
            vocab = json.loads((tdir / "vocab.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return set()
        eos_text = "<\uff5cend\u2581of\u2581sentence\uff5c>"      # <｜end▁of▁sentence｜>, id 1 in this GGUF
        return {vocab[eos_text]} if eos_text in vocab else set()

    _stop_ids = _dsv4_stop_ids()
    if _stop_ids:
        _service_init = S.Service.__init__

        def _init_with_stop_ids(self, *args, **kw):
            _service_init(self, *args, **kw)
            self.stop_ids = set(_stop_ids)

        S.Service.__init__ = _init_with_stop_ids
        print(f"stop ids for the server: {sorted(_stop_ids)} (replace the Qwen <|im_end|>/<|endoftext|> pieces)",
              file=sys.stderr, flush=True)
    else:
        print("WARNING: end-of-sentence token not found: the server keeps its Qwen stop ids", file=sys.stderr, flush=True)

    class Dsv4Engine(S.StrataEngine):
        """Same protocol, different process. Everything else (the pump thread, the STOP handling, the
        silence watchdog, restart, sessions) is inherited unchanged."""

        def __init__(self, exe_, args, cwd=None, log=None, env=None, lazy=False):
            super().__init__(exe_, args, cwd=cwd, log=log, env=env, lazy=lazy)
            # StrataEngine looks for --native/--pack; this engine names its model --model.
            self.model_path = next((args[i + 1] for i, x in enumerate(args) if x == "--model"), str(exe_))
            self.info.setdefault("engine", "dsv4_ext")

    S.StrataEngine = Dsv4Engine

    argv = ["serve_dsv4", "--engine", "strata", "--config", str(cfg_path),
            "--tokenizer", str(tdir), "--port", str(a.port), "--host", a.host]
    if a.api_key:
        argv += ["--api-key", a.api_key]
    if a.open:
        argv.append("--open")
    if a.lazy:
        argv.append("--lazy")
    print(f"dsv4_run: {' '.join(cfg['args'])}\nengine log: {log}", file=sys.stderr, flush=True)
    sys.argv = argv
    return S.main()


if __name__ == "__main__":
    sys.exit(main())
