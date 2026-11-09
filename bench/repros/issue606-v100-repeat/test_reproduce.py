import hashlib
import json
import unittest
from unittest import mock
import reproduce as r


class Reproduction(unittest.TestCase):
    def test_fixture_hashes_and_exact_original_wire(self):
        manifest = json.loads((r.ROOT / 'manifest.json').read_text())
        for item in manifest['requests']:
            wire = r.load_request(item)
            self.assertEqual(hashlib.sha256(wire).hexdigest(), item['wire_sha256'])

    def test_alias_override_only_changes_model(self):
        info = json.loads((r.ROOT / 'manifest.json').read_text())['requests'][0]
        old, new = json.loads(r.load_request(info)), json.loads(r.load_request(info, 'local-alias'))
        self.assertEqual(new.pop('model'), 'local-alias')
        old.pop('model')
        self.assertEqual(new, old)

    def test_endpoint_rejects_embedded_auth(self):
        self.assertEqual(r.endpoint('http://localhost:8080/v1'), 'http://localhost:8080/v1/chat/completions')
        for url in ['file:///test', 'http://user:key@localhost', 'http://localhost/?key=secret']:
            with self.assertRaises(ValueError): r.endpoint(url)

    def test_repetition_report_is_text_only_and_never_executes(self):
        events = [{'choices': [{'delta': {'reasoning_content': '0' * 256}, 'finish_reason': 'length'}],
                   'usage': {'prompt_tokens': 61000, 'completion_tokens': 2198, 'total_tokens': 63198,
                             'prompt_tokens_details': {'cached_tokens': 0}}}]
        with mock.patch('builtins.exec', side_effect=AssertionError('must not execute')):
            result = r.summarize(events, True, b'{}', 1.0)
        self.assertTrue(result['empty_visible_answer'])
        self.assertEqual(result['longest_zero_character_run_in_reasoning'], 256)
        self.assertFalse(result['code_executed'])
        r.check_protocol(result, {'prompt_tokens': 61000, 'max_tokens': 4000})

    def test_first_history_must_have_full_output_and_no_cache(self):
        result = {'stream_done': True, 'finish_reason': 'length', 'usage': {'prompt_tokens': 64000,
                  'completion_tokens': 511, 'total_tokens': 64511, 'prompt_tokens_details': {'cached_tokens': 0}}}
        with self.assertRaises(RuntimeError): r.check_protocol(result, {'prompt_tokens': 64000, 'max_tokens': 512})
        result['usage'].update(completion_tokens=512, total_tokens=64512, prompt_tokens_details={'cached_tokens': 1})
        with self.assertRaises(RuntimeError): r.check_protocol(result, {'prompt_tokens': 64000, 'max_tokens': 512})


if __name__ == '__main__': unittest.main()
