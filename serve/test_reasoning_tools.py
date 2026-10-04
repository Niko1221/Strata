import unittest

from serve.frontend import OutputParser

CALL = '<tool_call><function=read_file><parameter=limit>120</parameter><parameter=path>file.txt</parameter></function></tool_call>'

class ReasoningToolTests(unittest.TestCase):
    def test_calls_in_both_channels_and_split_tags(self):
        for streaming in (False, True):
            for thinking in (False, True):
                for width in (1, 2, 7, 100000):
                    with self.subTest(streaming=streaming, thinking=thinking, width=width):
                        text = ('before' + CALL + 'after</think>answer' + CALL if thinking else 'answer' + CALL)
                        parser = OutputParser(thinking=thinking, stream_tools=streaming)
                        events = []
                        for i in range(0, len(text), width):
                            events += parser.feed(text[i:i+width])
                        events += parser.finish()
                        calls = [e.call for e in events if e.kind == 'tool_call']
                        self.assertEqual(len(calls), 2 if thinking else 1)
                        for call in calls:
                            self.assertEqual(call.name, 'read_file')
                            self.assertEqual(call.arguments, {'limit': 120, 'path': 'file.txt'})
                        self.assertEqual(''.join(e.text for e in events if e.kind == 'reasoning'), 'beforeafter' if thinking else '')
                        self.assertEqual(''.join(e.text for e in events if e.kind == 'content'), 'answer')

    def test_unfinished_reasoning_call_is_not_executable(self):
        p = OutputParser()
        events = p.feed('before<tool_call><function=read_file>') + p.finish()
        self.assertFalse(any(e.kind == 'tool_call' for e in events))
        self.assertEqual(''.join(e.text for e in events if e.kind == 'reasoning'), 'before<tool_call><function=read_file>')

if __name__ == '__main__':
    unittest.main()
