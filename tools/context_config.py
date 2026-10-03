#!/usr/bin/env python3
"""tools/context_config.py - a setup-written engine config at another context length, for the run-<model> scripts.

    python tools/context_config.py strata-iq2_xs.json 1M      -> prints the path of strata-iq2_xs.ctx1048576.json

CONTEXT is a token count (131072) or a count with a K / M suffix (512K, 1M; K = 1024).  The new config is the given
one with the context-dependent engine flags redone by setup.py's rules (its step 7):

  * --max-context CONTEXT;
  * rope scaling only past the trained 262,144 positions: yarn, factor CONTEXT / 262144 (an explicit --rope-scaling
    linear in the source config is kept as the method; its factor is derived again);
  * KV streaming (--kv-resident 32768) from 64K up, as setup turns it on (not with --kv k8v4, which never streams);
  * --kv int8 above 8K when the source config has no --kv (an 8K install runs fp16).

Everything else (pack, model files, GPUs, port, log) is copied unchanged.  The source config is never modified.
"""
import json
import re
import sys
from pathlib import Path

TRAINED = 262144            # the model's trained context (setup.py resolve_rope)
STREAM_FROM = 65536         # setup.py: KV streaming from 64K
STREAM_CELLS = "32768"


def parse_context(text: str) -> int:
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kKmM]?)\s*", text)
    if not m:
        raise ValueError(f"not a context length: {text!r} (use e.g. 131072, 512K or 1M)")
    n = float(m.group(1)) * {"": 1, "k": 1024, "m": 1024 * 1024}[m.group(2).lower()]
    if n != int(n) or n < 256:
        raise ValueError(f"context {text!r}: a whole number of tokens, 256 or more")
    return int(n)


def drop(args: list, flag: str, has_value: bool = True) -> list:
    out, i = [], 0
    while i < len(args):
        if args[i] == flag:
            i += 2 if has_value else 1
            continue
        out.append(args[i])
        i += 1
    return out


def value(args: list, flag: str):
    return args[args.index(flag) + 1] if flag in args[:-1] else None


def with_context(args: list, ctx: int) -> list:
    method = value(args, "--rope-scaling")
    kv = value(args, "--kv")
    for flag in ("--max-context", "--rope-scaling", "--rope-scale", "--kv-resident"):
        args = drop(args, flag)
    args += ["--max-context", str(ctx)]
    if ctx > TRAINED:
        args += ["--rope-scaling", method if method in ("yarn", "linear") else "yarn",
                 "--rope-scale", f"{ctx / TRAINED:g}"]
    if ctx > 8192 and kv is None:
        kv = "int8"
        args += ["--kv", kv]
    if ctx >= STREAM_FROM and kv != "k8v4":
        args += ["--kv-resident", STREAM_CELLS]
    return args


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    src = Path(sys.argv[1]).resolve()
    try:
        ctx = parse_context(sys.argv[2])
    except ValueError as e:
        print(f"context_config: {e}", file=sys.stderr)
        return 2
    cfg = json.loads(src.read_text(encoding="utf-8-sig"))
    cfg["args"] = with_context(list(cfg["args"]), ctx)
    dst = src.with_name(f"{src.stem}.ctx{ctx}.json")
    dst.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    kv_gb = ctx * 13 * (576 if value(cfg["args"], "--kv") == "q4_0" else 1056) / 1e9
    rope = value(cfg["args"], "--rope-scaling")
    print(f"context_config: {ctx} tokens" + (f", rope {rope} x{value(cfg['args'], '--rope-scale')} (experimental)"
                                             if rope else "") + f", KV cache ~{kv_gb:.1f} GB", file=sys.stderr)
    print(dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
