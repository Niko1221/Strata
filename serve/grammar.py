"""Constrained decoding (GBNF grammars) at the HTTP boundary: the wire format, the registration key,
and the end-of-stream canary.  The engine caches a parsed grammar by a server-computed content hash,
so the same text registers once per engine generation; a restart clears both sides."""
import hashlib
from pathlib import Path

from serve.structured import StructuredOutputError, validated_json

# Vendored verbatim from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d (grammars/json.gbnf),
# MIT: the grammar a response_format={"type":"json_object"} request runs under.
JSON_OBJECT_GBNF = (Path(__file__).parent / "grammars" / "json.gbnf").read_text(encoding="utf-8")

MAX_GRAMMAR_BYTES = 1 << 20   # a bare `grammar` parameter's ceiling


def grammar_sha16(text: str) -> str:
    """The registration key: the content hash.  The engine treats it as opaque; two servers that
    register the same text share nothing, and a restart re-registers what a request needs."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def wire(text: str) -> str:
    """The GRAMMAR line carries the grammar to the end of the line, so a multi-line GBNF text rides
    with newlines and backslashes escaped; the engine unescapes in one pass and validates byte_len
    against the original, so a truncated line is a parse failure, not a silent short grammar."""
    return text.replace("\\", "\\\\").replace("\n", "\\n")


def check_param(value) -> str:
    """Validate a bare `grammar` request parameter; raise ValueError (→ 400) on anything else."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("grammar: expected a non-empty GBNF string")
    if len(value.encode("utf-8")) > MAX_GRAMMAR_BYTES:
        raise ValueError(f"grammar: larger than {MAX_GRAMMAR_BYTES} bytes")
    return value


def grammar_canary(chunks, validator):
    """The grammar path streams unbuffered - the engine constrained every token, so the JSON is
    already well-formed as it comes.  The canary at the stream's end validates the assembled text:
    a failure is a grammar-engine bug (the validator saw the same shape the schema demanded), said
    as one, and the routes answer it with a 502, not a silent stream of broken JSON."""
    collected: list[str] = []
    for c in chunks:
        yield c
        if validator is not None and isinstance(c, dict) and not c.get("prompt_progress"):
            choice = (c.get("choices") or [{}])[0]
            delta = choice.get("delta")
            if isinstance(delta, dict) and delta.get("content"):
                collected.append(delta["content"])
            finish = choice.get("finish_reason")
            if finish is not None:
                try:
                    validated_json("".join(collected), validator, finish)
                except StructuredOutputError as e:
                    # an unfinished stream is the budget, not the constraint: say which
                    if finish != "stop":
                        raise StructuredOutputError(
                            f"the constrained answer did not fit the output budget: {e}") from e
                    raise StructuredOutputError(
                        f"grammar engine bug: a constrained stream failed its own schema: {e}") from e
