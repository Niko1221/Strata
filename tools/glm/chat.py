#!/usr/bin/env python3
"""tools/glm/chat.py - one-command chat with the GLM-5.3-Flash engine (strata-glm).

Starts the engine (or attaches to a running one later), tokenizes with the model's own
tokenizer (tools/strata_tokenizer.py, the glm4 pre-tokenizer), and talks the engine's wire
protocol: `GEN <max_new> [keys] <ids>` in, `T <rid> <id>` lines out, `DONE` at the end.

Every GEN starts a fresh sequence (the driver's contract), so each turn re-encodes the whole
conversation - exactly what the chat template describes.

Usage:
  python tools/glm/chat.py --pack /path/to/glm-pack [--exe /path/to/strata-glm]
                           [--max-context 32768] [--kv int8] [--temperature 0.7]
                           [--max-new 512] [--gpus 0,1]

`--exe` defaults to $STRATA_GLM_EXE, else build/strata-glm next to this repository.
Type /quit or Ctrl-D to leave; /reset starts a new conversation.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from strata_tokenizer import Tokenizer  # noqa: E402


def find_engine(a: argparse.Namespace) -> pathlib.Path:
    if a.exe:
        p = pathlib.Path(a.exe)
    else:
        p = pathlib.Path(os.environ.get("STRATA_GLM_EXE") or (ROOT / "build" / "strata-glm"))
    if not p.is_file():
        raise SystemExit(f"strata-glm not found at {p} (build it with -DSTRATA_BUILD_GLM=ON, or pass --exe)")
    return p


def find_pack(a: argparse.Namespace) -> pathlib.Path:
    p = pathlib.Path(a.pack)
    if not (p / "index.txt").is_file():
        raise SystemExit(f"{p} is not a GLM pack (no index.txt); build one with tools/iq_pack.py --compat-bf16")
    return p


class Engine:
    def __init__(self, exe: pathlib.Path, pack: pathlib.Path, a: argparse.Namespace):
        argv = [str(exe), "--glm-pack", str(pack), "--serve",
                "--max-context", str(a.max_context), "--kv", a.kv]
        if a.gpus:
            argv += ["--gpu", a.gpus]
        if a.parallel > 1:
            argv += ["--parallel", str(a.parallel)]
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=sys.stderr, text=True, bufsize=1)
        self.rid = 0
        line = self.proc.stdout.readline()
        while line and not line.startswith("READY"):
            if line.startswith("ERR"):
                raise SystemExit(f"engine error: {line.strip()}")
            line = self.proc.stdout.readline()
        if not line:
            raise SystemExit("the engine exited before READY (see its output above)")

    def generate(self, ids: list[int], max_new: int, keys: str) -> list[int]:
        self.rid += 1
        self.proc.stdin.write(f"GEN {max_new} rid={self.rid}{keys} {','.join(map(str, ids))}\n")
        self.proc.stdin.flush()
        out: list[int] = []
        for line in self.proc.stdout:
            parts = line.split()
            if parts and parts[0] == "T":
                # `T <id>` (one sequence) or `T <rid> <id>` (--parallel N, N > 1)
                if len(parts) == 2:
                    out.append(int(parts[1]))
                elif len(parts) == 3 and parts[1] == str(self.rid):
                    out.append(int(parts[2]))
            elif parts and parts[0] == "DONE":
                break
            elif parts and parts[0] == "ERR":
                raise SystemExit(f"engine error: {line.strip()}")
        return out


def render_turn(template_path: pathlib.Path, history: list[dict], ask: str) -> str:
    """The chat prompt: the model's own chat_template.jinja (it carries the reasoning-effort and
    tool blocks GLM-5.3 ships with), rendered with the conversation so far.  `loopcontrols` is
    required: the template uses `{% break %}` (a jinja2 extension, not core)."""
    import jinja2
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True,
                             extensions=["jinja2.ext.loopcontrols"])
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(Exception(m))
    env.filters["tojson"] = lambda v, ensure_ascii=False: __import__("json").dumps(v, ensure_ascii=ensure_ascii)
    messages = [*history, {"role": "user", "content": ask}]
    return env.from_string(template_path.read_text(encoding="utf-8")).render(messages=messages, add_generation_prompt=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="chat with the GLM-5.3-Flash engine")
    ap.add_argument("--pack", required=True)
    ap.add_argument("--exe", default=None)
    ap.add_argument("--max-context", type=int, default=32768)
    ap.add_argument("--kv", default="int8", choices=("fp16", "int8"))
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--gpus", default=None)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=None)
    a = ap.parse_args()

    engine = Engine(find_engine(a), find_pack(a), a)
    gguf = next((pathlib.Path(a.pack).parent / p.name for p in pathlib.Path(a.pack).parent.glob("*-00001-of-*.gguf")), None)
    tok = (Tokenizer.from_dir(pathlib.Path(a.pack) / "tokenizer") if (pathlib.Path(a.pack) / "tokenizer" / "vocab.json").exists()
           else Tokenizer.from_gguf(gguf))
    print(f"strata-glm ready: pack={a.pack} kv={a.kv} ctx={a.max_context} (type /quit to leave)")
    history: list[dict] = []
    keys = f" temperature={a.temperature}" if a.temperature else ""
    while True:
        try:
            ask = input("\n你 > ").strip()
        except EOFError:
            break
        if not ask:
            continue
        if ask in ("/quit", "/exit"):
            break
        if ask == "/reset":
            history.clear()
            print("(new conversation)")
            continue
        text = render_turn(pathlib.Path(a.pack) / "tokenizer" / "chat_template.jinja",
                           history, ask)
        ids = tok.encode(text, parse_special=True)
        print("GLM > ", end="", flush=True)
        got = engine.generate(ids, a.max_new, keys)
        reply = tok.decode(got)
        print(reply)
        history.append({"role": "user", "content": ask})
        history.append({"role": "assistant", "content": reply})
    engine.proc.stdin.write("QUIT\n")
    engine.proc.stdin.flush()
    engine.proc.wait(timeout=60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
