"""Deterministic synthetic retrieval input, using the shipped tokenizer and formatter."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from contracts import EXPECTED, ids_sha, require, roundtrip

ROOT = Path(__file__).resolve().parents[2]
TOKENIZER = Path("H:/Strata/models/packs/qwen36-ud-q4_k_xl/tokenizer")
MODEL = Path("H:/Strata/models/qwen36-UD-Q4_K_XL/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")
PROTECTED = (MODEL, Path(str(MODEL) + ".q36epk"), Path(str(MODEL) + ".q36epk.verified"))
SYSTEM = "Read the synthetic ledger. Return only the requested JSON object, without commentary."
QUESTION = (
    "\nReturn the lookup codes for BEGIN, MIDDLE and END, exactly as JSON with keys "
    '"begin", "middle", "end". No other keys, reasoning, markdown or explanation.\n'
)
FILLER = "The archive shelf holds ordinary notes about river stones and quiet gardens.\n"


def metadata() -> dict:
    """No weight/pack reads: presence, size and modification time only."""
    return {str(p): {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in PROTECTED}


def load() -> tuple[object, object]:
    sys.path[:0] = [str(ROOT / "tools"), str(ROOT)]
    from strata_tokenizer import Tokenizer
    from terminal.serve.frontend import ChatTemplate

    cfg = json.loads((TOKENIZER / "tokenizer.json").read_text(encoding="utf-8"))
    vocab = json.loads((TOKENIZER / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for text, index in vocab.items():
        tokens[index] = text
    tok = Tokenizer(
        tokens,
        (TOKENIZER / "merges.txt").read_text(encoding="utf-8").split("\n"),
        json.loads((TOKENIZER / "token_type.json").read_text()),
        cfg["pre"],
        cfg["special_ids"],
    )
    # This reader parses only the GGUF header; it never reads/maps tensor data.
    from gguf_reader import GGUFFile

    header = GGUFFile(MODEL)
    md = header.metadata
    require(tok.tokens == md["tokenizer.ggml.tokens"], "extracted vocabulary differs from actual GGUF metadata")
    require(list(tok.ranks) == [tuple(m.split(" ")) for m in md["tokenizer.ggml.merges"]], "merge order mismatch")
    require(
        tok.token_types == md["tokenizer.ggml.token_type"] and tok.pre == md["tokenizer.ggml.pre"],
        "tokenizer type/pre mismatch",
    )
    require(all(md[k] == v for k, v in tok.special_ids.items()), "special ID mismatch")
    require(
        (TOKENIZER / "chat_template.jinja").read_text(encoding="utf-8") == md["tokenizer.chat_template"],
        "actual GGUF template mismatch",
    )
    return tok, ChatTemplate(TOKENIZER / "chat_template.jinja")


def render(template: object, user: str) -> str:
    return template.render(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        enable_thinking=False,
        preserve_thinking=True,
    )


def prepare(tok: object, template: object, tick: object = lambda: None) -> dict:
    """Vary whole text, never slice IDs or truncate the rendered prompt."""
    best = None
    for lines in range(100, 350):
        tick()
        a = lines // 2
        user = (
            "Synthetic ledger v1.\nBEGIN lookup code: ALPHA7K\n"
            + FILLER * a
            + "MIDDLE lookup code: BRAVO9M\n"
            + FILLER * (lines - a)
            + "END lookup code: CHARLIE4P\n"
            + QUESTION
        )
        text = render(template, user)
        # Search by full encoding, not by capacity. Overshoot is rejected before roundtrip's capacity check.
        count = len(tok.encode(text, parse_special=True))
        if count > 3096:
            break
        if 3048 <= count <= 3096 and (best is None or abs(count - 3072) < abs(best["actualInputCount"] - 3072)):
            ids = roundtrip(tok, text)
            offsets = {
                name: len(tok.encode(text[: text.index(code)], parse_special=True)) for name, code in EXPECTED.items()
            }
            require(
                offsets["begin"] < count * 0.15
                and 0.35 * count < offsets["middle"] < 0.65 * count
                and offsets["end"] > 0.85 * count,
                "lookup positions are not beginning/middle/end",
            )
            best = dict(
                user=user,
                renderedText=text,
                ids=ids,
                actualInputCount=count,
                orderedIdsSHA256=ids_sha(ids),
                lookupTokenOffsets=offsets,
                expected=EXPECTED,
                fillerLines=lines,
                target=3072,
                tolerance=24,
                outputBudget=128,
                capacity=4096,
                mtpHeadroom=3,
                thinking=False,
                formatter="serve.frontend.ChatTemplate",
                tokenizerBinding=(
                    "exact vocabulary, ordered merges, token types, pre, special IDs and chat template "
                    "equal actual GGUF header; no tensor data read"
                ),
                encoding="strata_tokenizer.Tokenizer, parse_special=True; exact decode/re-encode",
                consumedInputEvidence="not executed; O1 native consumed-ID hash not implemented",
            )
    require(best is not None, "could not build 3072 +/-24 actual tokenizer input")
    return best


def extension(tok: object, template: object, fixture: dict, generated: list[int]) -> dict:
    """Real follow-up chat. Keep the original empty-think postprompt verbatim."""
    answer = tok.decode(generated, errors="strict")
    eos = tok.tokens[tok.special_ids["tokenizer.ggml.eos_token_id"]]
    if answer.endswith(eos):
        answer = answer[: -len(eos)]
    history = fixture.get("messages") or [
        {"role": "system", "content": SYSTEM}, {"role": "user", "content": fixture["user"]},
    ]
    messages = history + [
        {"role": "assistant", "content": answer, "reasoning_content": ""},
        {"role": "user", "content": "Repeat the same three lookup codes as the same JSON object."},
    ]
    text = template.render(
        messages,
        enable_thinking=False,
        preserve_thinking=True,
    )
    ids = roundtrip(tok, text)
    base = fixture["ids"]
    require(ids[: len(base)] == base, "formatter changed original prefix; no cache claim possible")
    require(
        ids[: len(base) + len(generated)] == base + generated,
        "formatter changed generated continuation; boundary inconclusive",
    )
    require(0 < len(ids) - len(base) <= 256, "follow-up delta outside predeclared bound")
    return dict(
        renderedText=text,
        ids=ids,
        actualInputCount=len(ids),
        delta=len(ids) - len(base),
        orderedIdsSHA256=ids_sha(ids),
        continuationUsesActualGeneratedText=True,
        messages=messages,
    )
