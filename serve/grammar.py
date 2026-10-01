"""llguidance grammar state; the native sampler applies its mask before sampling.

No retries, repair pass or post-hoc insertion of missing fields. A matcher belongs
to one request; the tokenizer trie is shared across requests for the same model.
"""
import json
from functools import lru_cache

import llguidance

from serve.structured import StructuredOutputError


@lru_cache(maxsize=4)
def grammar_tokenizer(tok, stops):
    class Wrapper:
        eos_token_id = stops[0]
        bos_token_id = None
        if hasattr(tok, "tokens"):
            tokens = [tok.token_bytes(i) for i in range(len(tok.tokens))]
            special_token_ids = list(tok.special_tokens.values())
        else:  # the byte tokenizer used by the HTTP tests
            tokens = [bytes([i]) for i in range(256)] + [s.encode() for s in tok.SPECIALS]
            special_token_ids = list(range(256, len(tokens)))

        def __call__(self, text):
            text = text.decode() if isinstance(text, bytes) else text
            # Forced JSON bytes are literal text, even when they spell a model
            # control tag. GGUF's USER_DEFINED tokenizer path always recognizes
            # those tags, so use the plain BPE path for the grammar's byte spans.
            return tok._encode_plain(text) if hasattr(tok, "_encode_plain") else tok.encode(text)

    return llguidance.LLTokenizer(llguidance.TokenizerWrapper(Wrapper()),
                                 n_vocab=len(Wrapper.tokens), eos_token=list(stops))


class GrammarDecoder:
    def __init__(self, tok, stops, schema, thinking=False):
        self.tokenizer = grammar_tokenizer(tok, tuple(sorted(stops)))
        self.specials = tuple(tok.special_tokens.values()) if hasattr(tok, "special_tokens") else \
            tuple(range(256, 256 + len(tok.SPECIALS)))
        self.allowed_specials = set(stops)
        self.think_end = tok.encode("</think>", parse_special=True)[0] if thinking else None
        if thinking:
            self.allowed_specials.add(self.think_end)
        payload = json.dumps(schema, ensure_ascii=False, allow_nan=False)
        if thinking:
            close = tok.encode("</think>", parse_special=True)
            end = "</think>" if len(close) == 1 else '"</think>"'
            # Thinking stays free-form, but EOS cannot finish before the JSON body.
            grammar = llguidance.LLMatcher.grammar_from_lark(
                'start: reasoning /[ \\t\\r\\n]*/ %json ' + payload +
                '\nreasoning: /(.|\\n)*/ ' + end + '\n')
        else:
            grammar = llguidance.LLMatcher.grammar_from_json_schema(payload)
        self.matcher = llguidance.LLMatcher(self.tokenizer, grammar)
        if self.matcher.is_error() or self.matcher.get_grammar_warnings():
            raise ValueError("response_format schema cannot be enforced: " +
                             (self.matcher.get_error() or str(self.matcher.get_grammar_warnings())))

    def mask(self, vocab):
        if vocab != self.tokenizer.vocab_size:
            raise StructuredOutputError("grammar tokenizer and native model vocabulary differ")
        mask = self.matcher.compute_bitmask()
        if self.matcher.is_error():
            raise StructuredOutputError("grammar matcher failed: " + self.matcher.get_error())
        # llguidance includes a sentinel word beyond the model's vocabulary.
        size = ((vocab + 31) // 32) * 4
        if len(mask) < size:
            raise StructuredOutputError("grammar matcher returned an incomplete token mask")
        mask = bytearray(mask[:size])
        for token in self.specials:
            if token not in self.allowed_specials:
                mask[token // 8] &= ~(1 << (token % 8))
        if not any(mask):
            raise StructuredOutputError("grammar matcher permits no model token")
        return mask

    def accept(self, token):
        if not self.matcher.consume_token(token):
            raise StructuredOutputError("native sampler selected a token forbidden by the grammar")
        if token == self.think_end:
            self.allowed_specials.discard(token)
