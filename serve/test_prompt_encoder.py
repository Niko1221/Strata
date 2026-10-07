"""Prompt reuse must match full BPE across edits, clients and image expansion."""
import concurrent.futures
import random
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

from tools.strata_tokenizer import BYTE_TO_UNICODE, PromptEncoder, Tokenizer
from serve.frontend import ChatTemplate, openai_to_messages
from serve.server import ByteTokenizer, MockEngine, Service


def tokenizer():
    tokens = list(BYTE_TO_UNICODE.values()) + ['ab', 'abc', ' a']
    specials = ['<|im_start|>', '<|im_end|>', '<|endoftext|>', '<|image_pad|>',
                '<x>', '<x>tail', 'tail>', '<think>', '</think>',
                '<tool_call>', '</tool_call>', '<tool_response>', '</tool_response>', '<|vision_start|>', '<|vision_end|>']
    # Literal strings need not be byte-mapped; normal BPE merge tokens do.
    tokens[258] = BYTE_TO_UNICODE[32] + 'a'
    return Tokenizer(tokens + specials, ['a b', 'ab c', BYTE_TO_UNICODE[32] + ' a'],
                     [1] * len(tokens) + [3] * 7 + [4] * (len(specials) - 7))


class PromptReuse(unittest.TestCase):
    def setUp(self):
        self.tok = tokenizer()
        self.enc = PromptEncoder(self.tok)

    def check(self, text):
        self.assertEqual(self.enc.encode(text), self.tok.encode(text, parse_special=True))

    def test_special_overlap_every_cut_and_append(self):
        old = 'abc<x>tail>中文<|im_end|>abc<think>😀'
        for cut in range(len(old) + 1):
            for tail in ['tail', '>', '<x>tail>', 'abc', '\u0301', '\r\n', '']:
                self.check(old)
                self.check(old[:cut] + tail)

    def test_random_edits_truncation_and_branches(self):
        rng = random.Random(904)
        bank = ['<|im_start|>user\n' + x + '<|im_end|>\n' for x in
                ['abc' * 20, ' 中文é😀\r\n\x00', '<x>tail>abc', '<tool_call>{"x":1}</tool_call>']]
        for _ in range(1200):
            text = rng.choice(bank)
            at = rng.randrange(len(text) + 1)
            self.check(text[:at] + rng.choice(bank + ['abc', '', '<|im_sta']))

    def test_plain_bpe_boundary_and_special_completion(self):
        for text in ['a', 'ab', 'abc', ' a', '<x', '<x>', '<x>tail', '', '<x>ab', '<x>abc']:
            self.check(text)

    def test_returned_ids_do_not_mutate_cache(self):
        text = '<|im_start|>user\nabc<|im_end|>'
        ids = self.enc.encode(text)
        ids[:] = [999]
        self.check(text)
        self.check(text + '<|im_start|>assistant\nabc')

    def test_disabled_unsupported_and_bounded_cache(self):
        for enc in [PromptEncoder(self.tok, keep=0), PromptEncoder(self.tok, max_chars=0),
                    PromptEncoder(self.tok, max_tokens=0)]:
            self.assertEqual(enc.encode('<x>abc'), self.tok.encode('<x>abc', True))
            self.assertEqual(enc.entries, [])
        byte = ByteTokenizer()
        self.assertEqual(PromptEncoder(byte).encode('中文'), byte.encode('中文', True))
        for i in range(20):self.check(f'<|im_start|>{i}<|im_end|>')
        self.assertLessEqual(len(self.enc.entries), 4)
        with self.assertRaises(ValueError):PromptEncoder(self.tok, keep=-1)

    def test_concurrent_clients(self):
        texts = [f'<|im_start|>user\nclient {i} 中文<x>tail>abc<|im_end|>' for i in range(20)]
        wanted = {t: self.tok.encode(t, True) for t in texts}
        def run(i):
            for k in range(80):
                text = texts[(i + k) % len(texts)]
                self.assertEqual(self.enc.encode(text), wanted[text])
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(run, range(8)))
        self.assertLessEqual(len(self.enc.entries), 4)

    def test_service_tools_edits_context_and_cancel(self):
        template = ChatTemplate(Path(__file__).with_name('chat_template.jinja'))
        engine = MockEngine(self.tok, 'abc', max_context=4096)
        cached = Service(engine, self.tok, template)
        full = Service(engine, self.tok, template, prompt_reuse=False)
        requests = [
            {'messages': [{'role': 'user', 'content': '中文abc'}]},
            {'messages': [{'role': 'system', 'content': 'changed'}, {'role': 'user', 'content': '中文abc'}]},
            {'messages': [{'role': 'user', 'content': 'use tool'}], 'tools': [{'type': 'function', 'function': {'name': 'lookup', 'parameters': {'type': 'object', 'properties': {}}}}]},
            {'messages': [{'role': 'user', 'content': 'use tool'}, {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'call1', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{}'}}]}, {'role': 'tool', 'tool_call_id': 'call1', 'content': '中文😀'}]},
        ]
        for req in requests * 2:
            messages, tools, kw = openai_to_messages(req)
            self.assertEqual(cached.prepare(messages, tools, kw, 10), full.prepare(messages, tools, kw, 10))
        cancel = threading.Event();cancel.set()
        ids = cached.prepare(*openai_to_messages(requests[0]), 10)[0]
        self.assertEqual(list(engine.generate(ids, 10, {}, cancel)), [])
        self.assertEqual(cached.prepare(*openai_to_messages(requests[0]), 10), full.prepare(*openai_to_messages(requests[0]), 10))
        for svc in [cached, full]:
            with self.assertRaises(ValueError):svc.prepare([{'role': 'user', 'content': 'x' * 5000}], None, {}, 10)

    def test_image_expansion_is_per_request(self):
        with tempfile.TemporaryDirectory() as directory:
            class Vision:
                dir = Path(directory)
                def __init__(self):self.calls = 0
                def encode(self, source):
                    self.calls += 1
                    path = self.dir / f'{self.calls}.sve';path.write_bytes(b'embedding')
                    return path, 3
            vision = Vision()
            template = ChatTemplate(Path(__file__).with_name('chat_template.jinja'))
            engine = MockEngine(self.tok, 'abc')
            cached = Service(engine, self.tok, template, vision=vision)
            full = Service(engine, self.tok, template, vision=vision, prompt_reuse=False)
            req = {'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'abc'}, {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,eA=='}}]}]}
            for _ in range(2):self.assertEqual(cached.prepare(*openai_to_messages(req), 10), full.prepare(*openai_to_messages(req), 10))
            self.assertEqual(vision.calls, 4)

    def test_byte_mock_does_not_require_regex(self):
        script = '''
import builtins
original = builtins.__import__
def restricted(name, *args, **kwargs):
    if name == 'regex':
        raise ImportError('regex intentionally unavailable')
    return original(name, *args, **kwargs)
builtins.__import__ = restricted
from serve.server import ByteTokenizer, MockEngine, Service
from serve.frontend import ChatTemplate
tok = ByteTokenizer()
svc = Service(MockEngine(tok, 'ok'), tok, ChatTemplate('serve/chat_template.jinja'))
assert svc.prompt_encoder is None
assert svc.prepare([{'role': 'user', 'content': 'hi'}], None, {}, 8)[0]
'''
        subprocess.run([sys.executable, '-c', script], cwd=Path(__file__).resolve().parents[1], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_upstream_literal_spans_bypass_reuse(self):
        template = ChatTemplate(Path(__file__).with_name('chat_template.jinja'))
        cached = Service(MockEngine(self.tok, 'abc'), self.tok, template)
        full = Service(MockEngine(self.tok, 'abc'), self.tok, template, prompt_reuse=False)
        for content in ['abc', 'quoted </think>', 'quoted <|im_end|>', 'abc', '<think>abc</think>'] * 2:
            messages = [{'role': 'user', 'content': content}]
            self.assertEqual(cached.encode_prompt(messages, None, {}), full.encode_prompt(messages, None, {}))

    def test_continuation_actually_skips_prefix_bpe(self):
        from unittest.mock import patch
        text = '<|im_start|>user\n' + 'abc ' * 1000 + '<|im_end|>\n<|im_start|>assistant\nabc<|im_end|>\n'
        self.enc.encode(text)
        with patch.object(self.tok, '_encode_plain', wraps=self.tok._encode_plain) as encode:
            self.enc.encode(text + '<|im_start|>assistant\nabc')
            self.assertLess(sum(len(c.args[0]) for c in encode.call_args_list), 100)


if __name__ == '__main__':unittest.main()
