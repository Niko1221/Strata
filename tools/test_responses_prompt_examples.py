"""Synthetic fixtures exercise the actual parser; no model capabilities are claimed."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from responses_prompt_examples import (attempts, tool_examples, example_prefix, request_with_examples,
                                        coding_instructions, export)
from serve.frontend import OutputParser
from serve.responses import native_tools
from jsonschema import Draft202012Validator
from responses_native_coding_probe import command_observations


class PromptExamples(unittest.TestCase):
    def test_real_shell_status_and_observed_subcommand_failure_are_separate(self):
        command = 'python3 -B -m unittest -v test_settings.py; echo exit=$?'
        failed = {'command': command, 'exit_code': 0, 'aggregated_output': 'Ran 12 tests in 0.002s\n\nFAILED (failures=4, errors=7)\nexit=1\n'}
        passed = {'command': command, 'exit_code': 0, 'aggregated_output': 'Ran 12 tests in 0.002s\r\n\r\nOK\r\nexit=0\r\n'}
        missing = {'command': "cat -- 'input files/does-not-exist.fixture'; echo exit=$?", 'exit_code': 0,
                   'aggregated_output': "cat: 'input files/does-not-exist.fixture': No such file or directory\nexit=1\n"}
        report = command_observations([failed, missing, passed])
        self.assertTrue(report['client_observed_failing_tests'])
        self.assertTrue(report['client_verified_after_failure'])
        self.assertTrue(report['expected_missing_file_failure'])
        self.assertEqual(report['errors_with_zero_shell_status'], [0, 1])
        empty = {**passed, 'aggregated_output': 'exit=0\n'}
        self.assertFalse(command_observations([failed, empty])['client_verified_after_failure'])
        self.assertFalse(command_observations([passed, failed])['client_verified_after_failure'])
        self.assertFalse(command_observations([{**missing, 'aggregated_output': 'No such file or directory'}])['expected_missing_file_failure'])

    def test_all_tools_both_shells_parse_to_declared_arguments(self):
        tools = json.loads((ROOT / 'docs/codex/tool-declarations-0.160.0.json').read_text(encoding='utf-8'))
        originals = copy.deepcopy(tools)
        for tool, _, _ in native_tools(tools):
            for platform in ('windows', 'ubuntu'):
                for example in tool_examples(tool['name'], platform):
                    with self.subTest(tool=tool['name'], platform=platform, task=example['task']):
                        Draft202012Validator(tool['parameters']).validate(example['arguments'])
                        parser = OutputParser(thinking=False, tools=[tool], stream_tools=True)
                        events = parser.feed(example['native']) + parser.finish()
                        calls = [e.call for e in events if e.kind == 'tool_call']
                        self.assertEqual(len(calls), 1)
                        self.assertEqual(calls[0].arguments, example['arguments'])
                        self.assertEqual(json.loads(''.join(e.text for e in events if e.kind == 'tool_args')), example['arguments'])
        self.assertEqual(tools, originals)

    def test_bounds_and_baseline_identity(self):
        request = {'input': 'Read current goal.', 'instructions': 'Only read.', 'tools': []}
        examples = tool_examples('get_goal')
        self.assertEqual(request_with_examples(request, examples, 0), request)
        for count in (1, 2, 3):
            hinted = request_with_examples(request, examples, count)
            self.assertEqual(hinted['input'], request['input'])
            self.assertEqual(hinted['tools'], request['tools'])
            self.assertTrue(hinted['instructions'].endswith('Only read.'))
            self.assertEqual(hinted['instructions'].count('\nUser:'), count)
        for count in (-1, 4):
            with self.assertRaises(ValueError):
                example_prefix(examples, count)
        with self.assertRaises(ValueError):
            attempts(1, True)
        self.assertEqual(list(attempts(0, True)), [0, 1, 2, 3])
        self.assertEqual(list(attempts(0, True, 1)), [0, 1])

    def test_coding_profiles_have_exact_count_and_platform(self):
        for platform in ('windows', 'ubuntu'):
            baseline = coding_instructions(platform, 0)
            for count in range(4):
                result = coding_instructions(platform, count)
                self.assertTrue(result.endswith(baseline))
                self.assertEqual(result.count('\nUser:'), count)
            self.assertNotIn('EXAMPLE ', baseline)
        self.assertIn('PowerShell', coding_instructions('windows', 0))
        self.assertNotIn('PowerShell', coding_instructions('ubuntu', 3))

    def test_export_is_complete_and_hints_only_change_instructions(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'fixtures'
            rows = export(out)
            self.assertEqual(len(rows), 266)  # 2 OS * 12 tools * 4 * 2 strictness + 30 schemas * 2 + 7 modes * 2
            for row in rows:
                request = json.loads((out / row['file']).read_text(encoding='utf-8'))
                baseline = json.loads((out / row['fixture'] / 'baseline.json').read_text(encoding='utf-8'))
                request.pop('instructions', None)
                self.assertEqual(request, baseline)


if __name__ == '__main__':
    unittest.main()
