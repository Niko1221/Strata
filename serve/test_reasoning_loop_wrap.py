"""CPU-only regression for reasoning_loop_recovery "wrap" and "nudge": python -m unittest serve.test_reasoning_loop_wrap."""
import threading
import unittest
from serve.server import Service, ByteTokenizer, MockEngine, HIGH_EFFORT, REASONING_WRAP_UP
from serve.test_reasoning_loop_recovery import Tokenizer, RecordingEngine


class LoopWrap(unittest.TestCase):
    def setUp(self):
        self.tok = Tokenizer()
        self.ids = self.tok.encode('<|im_start|>user\nSolve this unchanged task<|im_end|>\n'
                                   '<|im_start|>assistant\n<think>\n', parse_special=True)
        self.loop = 'Need maybe Agent support agent seven b. ' * 700

    def run_mode(self, mode, scripts, sampling=None):
        engine = RecordingEngine(self.tok, scripts)
        service = Service(engine, self.tok, None)
        service.answer_room = False
        service.reasoning_loop_recovery = mode
        events = list(service.run(self.ids, True, [], 40000, sampling if sampling is not None else {},
                                  threading.Event()))
        text = {k: ''.join(e[1].text or '' for e in events if e[0] == 'event' and e[1].kind == k)
                for k in ('reasoning', 'content')}
        return engine, events, text

    def test_wrap_closes_the_thinking_and_answers(self):
        engine, events, text = self.run_mode('wrap', [self.loop, 'Complete solution.'])
        self.assertEqual(len(engine.calls), 2)
        self.assertEqual(text['content'].strip(), 'Complete solution.')
        self.assertIn('time to give my answer', text['reasoning'])
        self.assertLess(len(text['reasoning']), len(self.loop) // 2)          # cut early, not at the end
        wrap = self.tok.encode(REASONING_WRAP_UP, parse_special=True)
        self.assertEqual(engine.calls[1][0][-len(wrap):], wrap)               # the next pass goes on after the wrap-up
        self.assertEqual(events[-1][1]['finish'], 'stop')
        self.assertEqual(events[-1][1]['reasoning_recoveries'], 1)
        self.assertEqual(engine.sampling_calls[1], {})                        # sampling untouched

    def test_wrap_never_fires_on_novel_reasoning(self):
        novel = ' '.join(str(i) for i in range(4000)) + '\n</think>\nAnswer.'
        engine, events, text = self.run_mode('wrap', [novel])
        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(text['content'].strip(), 'Answer.')
        self.assertEqual(events[-1][1]['reasoning_recoveries'], 0)

    def test_nudge_escapes_without_a_wrap(self):
        request = {'temperature': .3, 'top_p': .95, 'top_k': 20, 'presence_penalty': 0}
        engine, events, text = self.run_mode('nudge', [self.loop, 'A new idea.\n</think>\nAnswer.'], request)
        self.assertEqual(len(engine.calls), 2)
        self.assertEqual((engine.sampling_calls[1]['temperature'], engine.sampling_calls[1]['presence_penalty']),
                         (1.0, 1.5))
        self.assertEqual((engine.sampling_calls[1]['top_p'], engine.sampling_calls[1]['top_k']), (.95, 20))
        self.assertEqual(request['temperature'], .3)                          # the client's dict is not changed
        self.assertNotIn('time to give my answer', text['reasoning'])
        self.assertEqual(text['content'].strip(), 'Answer.')
        self.assertEqual(events[-1][1]['reasoning_recoveries'], 1)

    def test_nudge_then_wrap_when_it_loops_again(self):
        engine, events, text = self.run_mode('nudge', [self.loop, self.loop, 'Complete solution.'])
        self.assertEqual(len(engine.calls), 3)
        self.assertIn('time to give my answer', text['reasoning'])
        self.assertEqual(text['content'].strip(), 'Complete solution.')
        self.assertEqual(events[-1][1]['reasoning_recoveries'], 2)
        self.assertEqual(events[-1][1]['finish'], 'stop')

    def test_off_and_stop_are_unchanged(self):
        engine, events, _ = self.run_mode(False, [self.loop + '\n</think>\nx'])
        self.assertEqual(len(engine.calls), 1)
        engine, events, _ = self.run_mode('stop', [self.loop])
        self.assertEqual((len(engine.calls), events[-1][1]['finish']), (1, 'length'))

    def run_runs(self, mode, scripts):
        engine = RecordingEngine(self.tok, scripts)
        service = Service(engine, self.tok, None)
        service.answer_room = False
        service.reasoning_loop_recovery = mode
        service.repeat_stop_tokens = 256
        events = list(service.run(self.ids, True, [], 40000, {'temperature': .3}, threading.Event()))
        text = {k: ''.join(e[1].text or '' for e in events if e[0] == 'event' and e[1].kind == k)
                for k in ('reasoning', 'content')}
        return engine, events, text

    def test_a_one_token_run_in_the_thinking_is_wrapped_not_emptied(self):
        engine, events, text = self.run_runs('wrap', ['Counting: ' + '4' * 600, 'Answer: 333.'])
        self.assertEqual(len(engine.calls), 2)
        self.assertIn('time to give my answer', text['reasoning'])
        self.assertEqual(text['content'].strip(), 'Answer: 333.')
        self.assertEqual(events[-1][1]['finish'], 'stop')

    def test_a_one_token_run_is_nudged_first(self):
        engine, events, text = self.run_runs('nudge', ['Counting: ' + '4' * 600, 'ok.\n</think>\nAnswer: 333.'])
        self.assertEqual(len(engine.calls), 2)
        self.assertEqual(engine.sampling_calls[1]['presence_penalty'], 1.5)
        self.assertEqual(text['content'].strip(), 'Answer: 333.')

    def test_a_one_token_run_in_the_answer_still_ends_it(self):
        engine, events, text = self.run_runs('nudge', ['ok.\n</think>\nAnswer: ' + '4' * 600])
        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(events[-1][1]['finish'], 'length')

    def test_off_keeps_606_as_it_was(self):
        engine, events, text = self.run_runs(False, ['Counting: ' + '4' * 600])
        self.assertEqual((len(engine.calls), events[-1][1]['finish']), (1, 'length'))


if __name__ == '__main__':
    unittest.main()
