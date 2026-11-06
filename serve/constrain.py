"""serve/constrain.py - constrained decoding for the JSON response formats: a token mask per step, from llguidance.

The engine (strata --serve, `mask=1` on GEN, INFO token_mask=1, or token_mask=2 when it can mask a whole MTP
window) asks before every verify window with a line `MQ` (or `MQ d1 d2 ...`, the drafts it would verify; only an
engine that said token_mask=2 sends them); the server answers on stdin with one line:

    MF                  free: the window runs as always (drafts, MTP) - the thinking, before the answer
    MF cut=<id,...>     free, but the window's accepted tokens end at the first of these ids (</think>), so no draft
                        past the end of the thinking is accepted unconstrained
    MK <base64>         constrained: a one-token window (no drafts), and the head's logits outside the mask are set to
                        -1e30 before the request's own sampler picks (greedy or sampled, penalties as asked).  The mask
                        is little-endian uint32 words, bit v = token v allowed; words past the end are "not allowed".
    MKN <n> <b0> ...    constrained, n rows: row i is the mask for the token after the engine's i-th draft (the same
                        encoding as MK).  Only sent in answer to `MQ d...`; an engine that knows only MK never sees it.

The grammar is the request's JSON schema (json_object: any object), and with tools also "<tool_call> + anything": a
turn that calls tools stays a tool call (the server's own parser reads it), a turn that answers is the schema's JSON
token for token.  Leading whitespace is allowed before either.  The thinking is not constrained: the mask starts
after </think> (or at once when the prompt has already closed it, i.e. thinking off).

llguidance is optional (`pip install llguidance`): without it, or with an engine that does not say token_mask=1 or
token_mask=2, nothing changes - the prompt directive and the validation after the turn (serve/structured.py) remain
the contract.
The validation also stays ON with the mask: it is the check that the mask and the schema agree, not the guarantee.
"""
from __future__ import annotations
import base64
import json
import threading

try:
    import llguidance as _llg
    import llguidance.numpy  # noqa: F401  (part of the package; imported so a broken install fails here)
except ImportError:                     # optional dependency
    _llg = None

KEY = "_strata_constraint"              # the sampling dict's key the engines read (never sent to the engine as text)
THINK_END = b"</think>"
TOOL_CALL = "<tool_call>"
SPECIAL_MARK = b"\xff"                  # llguidance: a special token's bytes are 0xFF + its text
_noted: set[str] = set()
_note_lock = threading.Lock()


def available() -> bool:
    return _llg is not None


def note_once(msg: str) -> None:
    with _note_lock:
        if msg in _noted:
            return
        _noted.add(msg)
    print(f"[strata] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ the vocabulary
def _vocab_bytes(tok):
    """-> (bytes per token id, special ids).  Three tokenizers are known: the tests' ByteTokenizer (one id per byte,
    SPECIALS after 256), and the pack's tokenizer (strata_tokenizer.Tokenizer: `token_bytes(id)`, the token types
    from token_type.json - 3 = control, 4 = user-defined such as <think> / <tool_call> - are special)."""
    specials_list = getattr(tok, "SPECIALS", None)
    if specials_list is not None and not hasattr(tok, "token_bytes"):          # ByteTokenizer
        tokens = [bytes([i]) for i in range(256)] + [SPECIAL_MARK + s.encode() for s in specials_list]
        return tokens, list(range(256, 256 + len(specials_list)))
    if not hasattr(tok, "token_bytes"):
        raise ValueError("this tokenizer has no token_bytes(); constrained decoding needs the bytes of every token")
    n = None
    for attr in ("n_vocab", "vocab_size"):
        v = getattr(tok, attr, None)
        n = v() if callable(v) else v
        if isinstance(n, int) and n > 0:
            break
    if not isinstance(n, int) or n <= 0:
        toks = getattr(tok, "tokens", None)
        n = len(toks) if toks is not None else None
    if not isinstance(n, int) or n <= 0:
        raise ValueError("cannot tell this tokenizer's vocabulary size")
    types = getattr(tok, "types", None) or getattr(tok, "token_types", None)
    tokens, specials = [], []
    for i in range(n):
        b = tok.token_bytes(i) or b""
        special = types is not None and i < len(types) and types[i] in (3, 4)
        if special:
            specials.append(i)
            b = SPECIAL_MARK + b
        tokens.append(bytes(b))
    return tokens, specials


class _Wrapper:
    """What llguidance.TokenizerWrapper reads: tokens (bytes), eos/bos ids, special ids, and a callable tokenizer."""
    def __init__(self, tok, tokens, specials, eos):
        self.tok, self.tokens, self.special_token_ids = tok, tokens, specials
        self.eos_token_id, self.bos_token_id = eos, None

    def __call__(self, s):
        if isinstance(s, (bytes, bytearray)):
            s = bytes(s).decode("utf-8", errors="replace")
        return list(self.tok.encode(s))


class Vocab:
    """The tokenizer as llguidance sees it.  Built once per server (about a second for 248K tokens)."""
    def __init__(self, tok, stop_ids):
        if _llg is None:
            raise ValueError("llguidance is not installed")
        self.tok = tok
        self.tokens, self.specials = _vocab_bytes(tok)
        self.n = len(self.tokens)
        im_end = tok.encode("<|im_end|>", parse_special=True)
        eos = im_end[0] if len(im_end) == 1 else min(stop_ids)
        self.eos = sorted(set(stop_ids) | {eos})
        self.lltok = _llg.LLTokenizer(_llg.TokenizerWrapper(_Wrapper(tok, self.tokens, self.specials, eos)),
                                      eos_token=self.eos)
        # the window cut while the thinking runs: </think> when it is one token, else every token ending in '>'
        one = self.single(THINK_END.decode())
        self.cut = [one] if one is not None else [i for i, b in enumerate(self.tokens) if b.endswith(b">")]
        # the cut ids for a draft lookahead (1-1): the think_end cut above, plus <tool_call> when the tokenizer
        # writes it as one special token - both end the constrained turn, so a draft at or past them is not fed
        tool = self.single(TOOL_CALL)
        self.cut_ids = list(self.cut) + ([tool] if tool is not None and tool not in self.cut else [])

    def single(self, text: str):
        """The id of `text` when the tokenizer writes it as ONE special token (<tool_call>, </think>), else None."""
        ids = self.tok.encode(text, parse_special=True)
        if len(ids) == 1 and ids[0] in set(self.specials):
            return ids[0]
        return None

    def literal(self, text: str) -> str:
        """`text` in a Lark rule: the special token itself when the model has one for it, else the string."""
        one = self.single(text)
        return f"<[{one}]>" if one is not None else json.dumps(text)

    def bytes_of(self, t: int) -> bytes:
        return self.tokens[t] if 0 <= t < self.n else b""


# ------------------------------------------------------------------------------------------------ the grammar
def format_of(sampling: dict):
    """The request's JSON format from its sampling dict (the request itself): chat's response_format, or the
    Responses API's text.format.  None: plain text (nothing to constrain)."""
    rf = sampling.get("response_format")
    if rf is None and isinstance(sampling.get("text"), dict):
        fmt = sampling["text"].get("format")
        if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
            rf = {"type": "json_schema", "json_schema": {k: fmt[k] for k in ("name", "schema", "strict") if k in fmt}}
        elif isinstance(fmt, dict) and fmt.get("type") == "json_object":
            rf = {"type": "json_object"}
    if not isinstance(rf, dict) or rf.get("type") not in ("json_object", "json_schema"):
        return None
    return rf


def grammar(rf: dict, with_tools: bool, vocab: Vocab) -> str:
    """The Lark grammar: optional whitespace, then the schema's JSON - or, with tools, a tool call."""
    if rf["type"] == "json_object":
        schema = {"type": "object"}
    else:
        schema = (rf.get("json_schema") or {}).get("schema")
        if not isinstance(schema, dict):
            raise ValueError("json_schema without a schema object")
    lines = ["start: WS? answer",
             "WS: /[ \\t\\r\\n]+/",
             "answer: json" + (" | calls" if with_tools else ""),
             "json: %json " + json.dumps(schema, ensure_ascii=False)]
    if with_tools:
        # the call itself is not constrained here: the server's tool-call parser reads it as it always did.
        # Anything may follow the opening tag - a tokenizer that writes <tool_call> as plain text (no special
        # token, so feed() cannot free the turn) must still be able to finish the call.
        lines += [f"calls: {vocab.literal(TOOL_CALL)} REST",
                  "REST: /[\\s\\S]*/"]
    return _llg.LLMatcher.grammar_from_lark("\n".join(lines) + "\n")


# ------------------------------------------------------------------ a masked window with drafts (1-1)
def masks_for_window(m, drafts, cut_ids):
    """The masks for rows 0..k of one verify window: row 0 for the token the head picks now, row i for the token
    after the engine's i-th draft (so an MTP window stays constrained token by token).  The matcher's state is
    restored to what it was on entry (rollback in a finally); a rollback failure raises RuntimeError (1-5).

    How many drafts are valid is probed on a deep copy, one token at a time: llguidance's validate_tokens reports
    k=0 for some single-byte tokens that the grammar accepts (':' after a key), and a refused consume leaves the
    matcher in a permanent error state that rollback cannot undo (measured, llguidance 1.9.1).  The copy takes the
    error instead; the original only consumes drafts the copy already accepted, so its consume cannot fail.
    There is one row per token the copy consumed, plus the head's row: when the grammar ends inside the window the
    last row is the EOS-only mask, which is what lets the window stop there."""
    drafts = list(drafts)
    for i, d in enumerate(drafts):            # think_end / tool_call sit outside the grammar: cut before them
        if d in cut_ids:
            drafts = drafts[:i]
            break
    k = 0
    if drafts:
        probe = m.deep_copy()
        for d in drafts:
            if not probe.consume_tokens([d]):  # the first token that leaves the grammar: no rows past it
                break
            k += 1
    masks = [m.compute_bitmask()]
    done = 0
    try:
        for d in drafts[:k]:                  # consume one draft at a time, one row of the window per step
            if not m.consume_tokens([d]):     # accepted on the copy, so this should not happen; keep the rows so far (1-5)
                note_once("constrained decoding: a validated draft was refused during the mask lookahead; "
                          "the masked window keeps the rows up to it")
                break
            done += 1
            if m.is_stopped():
                # the grammar ends here.  Every draft was consumed (the engine will ask for the token after
                # the last one): its row is the EOS-only mask.  Drafts remain (they sit past the end): the
                # window is cut at the end - no row past it (test_6; a consume past it would error anyway).
                if done == k and k == len(drafts):
                    masks.append(m.compute_bitmask())
                break
            masks.append(m.compute_bitmask())
    finally:
        if done and not m.rollback(done):
            raise RuntimeError("llguidance rollback failed")
    return masks


# ------------------------------------------------------------------------------------------------ one turn
class Constraint:
    """One generation pass of one request: the matcher, and whether the answer (the constrained part) has started."""
    def __init__(self, vocab: Vocab, grammar_text: str, armed: bool):
        self.vocab, self.armed = vocab, armed
        self.m = _llg.LLMatcher(vocab.lltok, grammar_text, log_level=0)
        if self.m.is_error():
            raise ValueError(f"grammar: {self.m.get_error()}")
        self.tail = b""
        self.failed = None                  # why the mask was given up (the validation after the turn then decides)
        self.masked = 0                     # tokens decoded under a mask (diagnostics, tests)
        self.tool_call = vocab.single(TOOL_CALL) # its id, when the model has one token for it

    # --- the engine's question
    def active(self) -> bool:
        return self.armed and self.failed is None and not self.m.is_error()

    def mask(self) -> bytes | None:
        """The packed mask for the next token, or None when the next token is free."""
        if not self.active():
            return None
        return self.m.compute_bitmask()

    def reply(self, drafts=None) -> str:
        """The answer to the engine's MQ line (without the newline).  `drafts` (1-2): the engine's draft tokens
        for this window, sent only by an engine that said token_mask=2; empty or None keeps the one-row answer."""
        if drafts and self.armed and self.failed is None and not self.m.is_error():
            try:
                masks = masks_for_window(self.m, drafts, self.vocab.cut_ids)
            except Exception as e:                        # 1-5: rollback or matcher failure - drop the mask, log it
                self.failed = str(e).splitlines()[0][:160] or "mask lookahead failed"
                note_once(f"constrained decoding: gave the mask up in a turn ({self.failed}); "
                          "the turn is validated after")
            else:
                self.masked += len(masks)
                if len(masks) == 1:                       # 1-3: one row keeps the current MK form (old-engine safe)
                    return "MK " + base64.b64encode(masks[0]).decode("ascii")
                rows = " ".join(base64.b64encode(b).decode("ascii") for b in masks)
                return f"MKN {len(masks)} {rows}"
        bits = self.mask()
        if bits is None:
            if not self.armed and self.vocab.cut:
                return "MF cut=" + ",".join(str(i) for i in self.vocab.cut)
            return "MF"
        self.masked += 1
        return "MK " + base64.b64encode(bits).decode("ascii")

    # --- what came out
    def feed(self, t: int) -> None:
        """A token the engine emitted (in order)."""
        if self.failed is not None:
            return
        if not self.armed:
            self.tail = (self.tail + self.vocab.bytes_of(t))[-64:]
            if self.tail.endswith(THINK_END):
                self.armed = True
            return
        # --- a tool call: the turn is free from here ---
        if self.tool_call is not None and t == self.tool_call:
            self.armed = False                  # a tool call: the turn is free from here (not validated)
            self.failed = "tool call"           # mask() -> None, reply() -> "MF" for the rest of the turn
            return
        if not self.m.consume_token(t):
            self.failed = self.m.get_error() or f"token {t} is outside the grammar"
            why = (self.failed.strip().splitlines() or ["?"])[0][:160]
            note_once(f"constrained decoding: gave the mask up in a turn ({why}); the turn is validated after")

    def observe(self, ids) -> None:
        """Tokens the server put into the stream itself (the thinking budget's wrap-up, which ends in </think>)."""
        for t in ids:
            if self.armed:
                break
            self.feed(t)

    @staticmethod
    def allowed(bits: bytes, t: int) -> bool:
        w = t >> 5
        return 4 * w + 4 <= len(bits) and (int.from_bytes(bits[4 * w:4 * w + 4], "little") >> (t & 31)) & 1 == 1

    def mock_replay(self, script, max_new, cancel, delay=0.0):
        """MockEngine under a mask: a "model" that wants to write `script`.  Where the mask forbids its next token it
        drops it and goes on with the rest (prose around the JSON disappears, a wrongly typed value loses its quotes);
        when the script runs out it ends the turn if the grammar may end, else takes the grammar's forced tokens."""
        import time
        i, out = 0, 0
        while out < max_new and not cancel.is_set():
            if delay:
                time.sleep(delay)
            bits = self.mask()
            if bits is not None:
                self.masked += 1
            t = None
            while i < len(script):
                cand = script[i]
                i += 1
                if bits is None or self.allowed(bits, cand):
                    t = cand
                    break
            if t is None:                                        # the script is used up
                if bits is None:
                    return
                eos = next((e for e in self.vocab.eos if self.allowed(bits, e)), None)
                ff = self.m.compute_ff_tokens()
                t = eos if eos is not None else (ff[0] if ff else None)
                if t is None:
                    return                                       # nothing forced: the turn ends short ("length")
            self.feed(t)
            out += 1
            yield t
            if t in self.vocab.eos:
                return
