"""tools/memt_engine.py - drive MemT-1.0's memory control lines on strata-q36.

The engine (`strata-q36 --serve`, or `strata --serve` on a qwen35moe file with
`memt.version`) reads one request per line and answers on stdout:

    MEMQUERY <max_new> [key=value ...] <id,id,...>    a query: read the memory
    MEMCOMMIT <id,id,...>                            write the prompt into memory
    MEMSELECT <session>                              the session the following lines act on
    MEMSNAP <path> <identity> [overwrite]            export the memory state
    MEMLOAD <path> <identity>                        import it
    MEMRESET                                         zero the memory state
    MEMFORGET                                        drop the cached prompt state, keep the memory

This module sends those lines and parses the answers, and computes the
`memt-memory-v1` identity the engine compares snapshots against.  It is the
transport `terminal/serve` uses; the codec itself is memt_snapshot.py on the
Python side and memt_snapshot.hpp on the engine side.

    with MemtEngine.spawn(["strata-q36", "--serve", "--model", path]) as e:
        e.load(snapshot, identity)
        answer = e.query(prompt_ids, max_new=32)
        e.commit(history_ids)
        e.snapshot(out, identity)
"""
from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

__all__ = ["MemtEngineError", "MemtEngine", "identity_from_model_dir", "format_ids"]


class MemtEngineError(RuntimeError):
    """The engine answered ERR, or the stream ended before the answer did."""


def format_ids(ids: Iterable[int]) -> str:
    return ",".join(str(int(x)) for x in ids)


def _sampling_keys(sampling: dict) -> str:
    """The engine's key=value tokens (the GEN keys; only the ones given)."""
    order = ("temperature", "top_p", "top_k", "min_p", "penalty_last_n", "penalty_repeat", "penalty_freq",
             "penalty_present", "seed")
    parts = []
    for key in order:
        if key in sampling and sampling[key] is not None:
            parts.append("%s=%s" % (key, sampling[key]))
    return (" " + " ".join(parts)) if parts else ""


def checkpoint_chat_template(root: Path, settings: dict | None = None) -> str | None:
    template = root / "chat_template.jinja"
    if template.is_file():
        return template.read_text(encoding="utf-8")
    if settings is None:
        config = root / "tokenizer_config.json"
        settings = json.loads(config.read_text(encoding="utf-8")) if config.is_file() else {}
    return settings.get("chat_template")


def identity_from_model_dir(model_dir: str | Path) -> str:
    """The `memt-memory-v1` identity of a local HF checkpoint directory.

    Mirrors memt_sglang's snapshot_identity: the backbone config, the `memt`
    object, the codec, the chat template and the tokenizer vocabulary.  Needs
    `config.json`, the chat template (tokenizer_config.json or
    chat_template.jinja) and `tokenizer.json`; the engine never computes this
    itself, it only compares the string it is given.
    """
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import memt_snapshot  # the codec module lives beside this one

    root = Path(model_dir)
    config_file = root / "config.json"
    if not config_file.exists():
        raise MemtEngineError("no config.json in %s" % root)
    try:
        config = json.loads(config_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise MemtEngineError("cannot read %s: %s" % (config_file, exc)) from exc
    tokenizer_config = root / "tokenizer_config.json"
    settings = json.loads(tokenizer_config.read_text(encoding="utf-8")) if tokenizer_config.exists() else {}
    template = checkpoint_chat_template(root, settings)
    if not isinstance(template, str) or not template:
        raise MemtEngineError("no chat template in %s" % root)
    vocab = checkpoint_vocabulary(root, settings)
    return memt_snapshot.snapshot_identity(config, {"chat_template": template, "vocab": vocab})


def checkpoint_vocabulary(root: Path, settings: dict | None = None) -> dict[str, int]:
    if settings is None:
        tokenizer_config = root / "tokenizer_config.json"
        settings = json.loads(tokenizer_config.read_text(encoding="utf-8")) if tokenizer_config.is_file() else {}
    vocab_file = root / "tokenizer.json"
    if not vocab_file.exists():
        raise MemtEngineError("tokenizer.json is required to rebuild the vocabulary: %s" % vocab_file)
    tokenizer = json.loads(vocab_file.read_text(encoding="utf-8"))
    vocab = {}
    model = tokenizer.get("model") or {}
    base = model.get("vocab") or {}
    if isinstance(base, dict):
        vocab.update({str(tok): int(idx) for tok, idx in base.items()})
    else:
        raise MemtEngineError("tokenizer.json's model.vocab is not a table")
    for added in tokenizer.get("added_tokens") or ():
        if "content" in added and "id" in added:
            vocab[str(added["content"])] = int(added["id"])
    for extra in (tokenizer.get("added_tokens_decoder") or {}).values() if isinstance(
            tokenizer.get("added_tokens_decoder"), dict) else ():
        if "content" in extra and "id" in extra:
            vocab[str(extra["content"])] = int(extra["id"])
    for token_id, token in settings.get("added_tokens_decoder", {}).items():
        vocab[str(token["content"])] = int(token_id)
    return vocab


class MemtEngine:
    """A line protocol connection: `write(line)` and `readline() -> str`."""

    def __init__(self, write: Callable[[str], None], readline: Callable[[], str],
                 process: Optional[subprocess.Popen] = None, context: int = 0):
        self._write = write
        self._readline = readline
        self.process = process
        self.context = context

    # ---- construction
    @classmethod
    def spawn(cls, argv: list[str], *, cwd: Optional[str] = None,
              ready_timeout: Optional[float] = None) -> "MemtEngine":
        """Start the engine and wait for its READY line (`READY <ctx> ...`).

        A model load can take minutes; `ready_timeout` bounds that wait and kills
        the process when it expires (the default waits forever, the old
        behaviour). Lines are read by a thread so the timeout can be applied.
        """
        process = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL, text=True, bufsize=1)
        if process.stdin is None or process.stdout is None:
            process.kill()
            raise MemtEngineError("cannot pipe the engine")
        lines: "queue.Queue[str]" = queue.Queue()

        def pump() -> None:
            try:
                for line in process.stdout:
                    lines.put(line)
            finally:
                lines.put("")   # EOF, or the process was killed

        threading.Thread(target=pump, name="memt-engine-reader", daemon=True).start()

        def flush(line: str) -> None:
            process.stdin.write(line)
            process.stdin.flush()

        engine = cls(flush, lines.get, process=process)
        deadline = None if ready_timeout is None else time.monotonic() + float(ready_timeout)
        while True:
            try:
                line = lines.get() if deadline is None else \
                    lines.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                process.kill()
                raise MemtEngineError("the engine did not print READY within %.1f s" % ready_timeout)
            if line == "":
                process.kill()
                raise MemtEngineError("the engine exited before READY")
            if line.startswith("READY"):
                fields = line.split()
                engine.context = int(fields[1]) if len(fields) > 1 and fields[1].lstrip("-").isdigit() else 0
                return engine
            if line.startswith("ERR"):
                process.kill()
                raise MemtEngineError(line.strip())

    def close(self) -> None:
        if self.process is None:
            return
        try:
            if self.process.stdin is not None:
                self.process.stdin.close()
        finally:
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def __enter__(self) -> "MemtEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- transport
    def send(self, line: str) -> None:
        if "\n" in line or "\r" in line:
            raise MemtEngineError("a control line cannot contain a newline")
        self._write(line + "\n")

    def _answer(self, until: tuple[str, ...], on_line: Optional[Callable[[str, float], None]] = None) -> list[str]:
        lines = []
        while True:
            line = self._readline()
            now = time.perf_counter()
            if line == "":
                raise MemtEngineError("the engine stream ended")
            line = line.rstrip("\r\n")
            if line.startswith("ERR"):
                raise MemtEngineError(line[3:].strip() or "engine error")
            lines.append(line)
            if on_line is not None:
                on_line(line, now)
            if any(line == u or line.startswith(u + " ") for u in until):
                return lines

    # ---- requests
    def query_timed(self, ids: Iterable[int], max_new: int, **sampling) -> dict:
        """A read-only query with its timing: client-side TTFT/end, the engine's own prompt/decode ms.

        The engine flushes `T <id>` as it emits, so `first_ms` is a real time to
        the first token. Its `DONE` line also carries the prompt and decode times
        it measured; both are reported.
        """
        ids = list(ids)
        if not ids or max_new < 1:
            raise MemtEngineError("query needs a prompt and max_new >= 1")
        marks: list[float] = []

        def on_line(line: str, now: float) -> None:
            if line.startswith("T "):
                marks.append(now)

        started = time.perf_counter()
        self.send("MEMQUERY %d%s %s" % (max_new, _sampling_keys(sampling), format_ids(ids)))
        answer = self._answer(("DONE",), on_line)
        ended = time.perf_counter()
        tokens = [int(line.split()[1]) for line in answer if line.startswith("T ")]
        fields = answer[-1].split()
        # DONE <produced> <prompt tokens> <prompt ms> <decode ms> <finish> ...
        if len(fields) < 6 or fields[0] != "DONE":
            raise MemtEngineError("the engine's answer ends with %r, not a DONE line with timings"
                                  % (answer[-1],))
        return {
            "ids": tokens,
            "first_ms": (marks[0] - started) * 1000.0 if marks else None,
            "end_ms": (ended - started) * 1000.0,
            "prompt_ms": float(fields[3]),
            "decode_ms": float(fields[4]),
            "finish": fields[5],
            "prompt_tokens": len(ids),
        }

    def query(self, ids: Iterable[int], max_new: int, **sampling) -> list[int]:
        """A read-only query: returns the generated token ids."""
        return self.query_timed(ids, max_new, **sampling)["ids"]

    def select_session(self, slot: int) -> None:
        """`MEMSELECT <slot>`: the session the following control lines act on."""
        if type(slot) is not int or slot < 0:
            raise MemtEngineError("a session is a whole number, 0 or more")
        self.send("MEMSELECT %d" % slot)
        self._answer(("MEMSELECT",))

    def commit(self, ids: Iterable[int]) -> dict:
        """Write the prompt into memory; returns the engine's MEMDONE fields."""
        ids = list(ids)
        if not ids:
            raise MemtEngineError("commit needs a prompt")
        self.send("MEMCOMMIT %s" % format_ids(ids))
        line = self._answer(("MEMDONE",))[-1]
        fields = line.split()
        return {"read": int(fields[1]), "prompt": int(fields[2]), "ms": float(fields[3])}

    def snapshot(self, path: str | Path, identity: str, overwrite: bool = False) -> None:
        path = str(path)
        if " " in path or " " in identity:
            raise MemtEngineError("the engine protocol cannot carry spaces in paths or identities")
        self.send("MEMSNAP %s %s%s" % (path, identity, " overwrite" if overwrite else ""))
        self._answer(("MEMSNAP",))

    def load(self, path: str | Path, identity: str) -> None:
        path = str(path)
        if " " in path or " " in identity:
            raise MemtEngineError("the engine protocol cannot carry spaces in paths or identities")
        self.send("MEMLOAD %s %s" % (path, identity))
        self._answer(("MEMLOAD",))

    def reset(self) -> None:
        self.send("MEMRESET")
        self._answer(("MEMRESET",))

    def forget(self) -> None:
        """Drop the session's cached prompt state (KV, GDN, checkpoints), keep the memory.

        A later query then reads its whole prompt afresh: the reference server's
        Nano configuration has no retained KV either, so parity and speed
        measurements of prompt processing compare like with like.
        """
        self.send("MEMFORGET")
        self._answer(("MEMFORGET",))
