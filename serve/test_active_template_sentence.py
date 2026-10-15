"""CPU-only coverage for #728 against executed custom templates, without a model pack.

    python -m unittest serve.test_active_template_sentence -v
"""
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate
from serve.server import (ByteTokenizer, EFFORT_TEXT, MockEngine, Service,
                          focused_recovery_prompt, template_effort_sentence)


class ActiveTemplateSentence(unittest.TestCase):
    def template(self, branch, render=True):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "chat_template.jinja"
        source = ('{% set resolved_reasoning_effort = reasoning_effort|default("medium") %}\n' + branch + '\n'
                  '{{ "<|im_start|>system\\n" ~ messages[0].content }}'
                  + ('{% if reasoning_instructions is defined %}{{ "\\n" ~ reasoning_instructions }}{% endif %}' if render else '')
                  + '{{ "<|im_end|>\\n<|im_start|>user\\n" ~ messages[1].content ~ "<|im_end|>\\n<|im_start|>assistant\\n<think>\\n" }}')
        path.write_text(source)
        return ChatTemplate(path)

    def custom(self, high="CUSTOM HIGH", low="CUSTOM LOW"):
        return self.template('{% if resolved_reasoning_effort == "xhigh" %}'
                             '{% set reasoning_instructions = "' + high + '" %}'
                             '{% elif resolved_reasoning_effort == "low" %}'
                             '{% set reasoning_instructions = "' + low + '" %}{% endif %}')

    def service(self, tpl):
        tok = ByteTokenizer()
        return Service(MockEngine(tok, "OK"), tok, tpl)

    def test_double_quoted_condition_and_assignment(self):
        tpl = self.custom()
        self.assertEqual(template_effort_sentence(tpl, "xhigh"), "CUSTOM HIGH")
        self.assertEqual(template_effort_sentence(tpl, "low"), "CUSTOM LOW")

    def test_single_quoted_and_computed_instruction(self):
        tpl = self.template("{% if resolved_reasoning_effort == 'xhigh' %}"
                            "{% set reasoning_instructions = 'CUSTOM ' ~ resolved_reasoning_effort|upper %}{% endif %}")
        self.assertEqual(template_effort_sentence(tpl, "xhigh"), "CUSTOM XHIGH")

    def test_escaped_quote_is_executed(self):
        tpl = self.template("{% set reasoning_instructions = 'Don\\'t skip verification.' %}")
        self.assertEqual(template_effort_sentence(tpl, "xhigh"), "Don't skip verification.")

    def test_does_not_read_assignment_from_unexecuted_else(self):
        tpl = self.template('{% if resolved_reasoning_effort == "xhigh" %}HIGH WITHOUT AN ASSIGNMENT'
                            '{% else %}{% set reasoning_instructions = "LOW ONLY" %}{% endif %}')
        self.assertIsNone(template_effort_sentence(tpl, "xhigh"))
        self.assertEqual(template_effort_sentence(tpl, "low"), "LOW ONLY")

    def test_instruction_must_appear_in_first_system_message(self):
        tpl = self.template('{% set reasoning_instructions = "UNUSED" %}', render=False)
        self.assertIsNone(template_effort_sentence(tpl, "xhigh"))

    def test_missing_instruction_uses_bundled_fallback(self):
        svc = self.service(self.template(""))
        self.assertEqual(svc.recovery_effort_sentences, (EFFORT_TEXT["xhigh"], EFFORT_TEXT["low"]))

    def test_unsupported_template_probe_is_a_fallback(self):
        tpl = self.template('{{ raise_exception("unsupported probe") }}')
        self.assertIsNone(template_effort_sentence(tpl, "xhigh"))
        self.assertEqual(self.service(tpl).recovery_effort_sentences[0], EFFORT_TEXT["xhigh"])

    def test_services_keep_their_own_instructions(self):
        first = self.service(self.custom("FIRST HIGH", "FIRST LOW"))
        second = self.service(self.custom("SECOND HIGH", "SECOND LOW"))
        self.assertEqual(first.recovery_effort_sentences, ("FIRST HIGH", "FIRST LOW"))
        self.assertEqual(second.recovery_effort_sentences, ("SECOND HIGH", "SECOND LOW"))

    def test_splice_preserves_user_text_and_exact_generated_ids(self):
        svc = self.service(self.custom())
        messages = [{"role": "system", "content": "Do the task."}, {"role": "user", "content": "CUSTOM HIGH is user text."}]
        ids = svc.tok.encode(svc.template.render(messages, reasoning_effort="xhigh"), parse_special=True)
        generated = svc.tok.encode("中文 literal </think>", parse_special=False)
        result = focused_recovery_prompt(svc.tok, ids, generated, svc.recovery_effort_sentences)
        self.assertIsNotNone(result)
        self.assertEqual(result[-len(generated):], generated)
        self.assertIn("CUSTOM LOW", svc.tok.decode(result))
        self.assertIn("CUSTOM HIGH is user text.", svc.tok.decode(result))
        self.assertEqual(svc.tok.decode(result).count("CUSTOM HIGH"), 1)

    def test_default_splice_contract_stays_unchanged(self):
        tok = ByteTokenizer()
        ids = tok.encode('<|im_start|>system\n' + EFFORT_TEXT["xhigh"] + '<|im_end|>', parse_special=True)
        result = focused_recovery_prompt(tok, ids, [])
        self.assertIn(EFFORT_TEXT["low"], tok.decode(result))

    def test_real_recovery_call_uses_service_template(self):
        class RecordingEngine(MockEngine):
            def generate(self, ids, *args, **kwargs):
                self.prompts.append(list(ids))
                yield from super().generate(ids, *args, **kwargs)
        tpl = self.custom()
        tok = ByteTokenizer()
        engine = RecordingEngine(tok, ['Repeated analysis words for checking the result. ' * 700, 'Done.\n</think>\nOK'],
                                 max_context=100000)
        engine.prompts = []
        svc = Service(engine, tok, tpl)
        svc.reasoning_loop_recovery = True
        ids = tok.encode(tpl.render([{"role": "system", "content": "Solve."}, {"role": "user", "content": "Task."}],
                                    reasoning_effort="xhigh"), parse_special=True)
        events = list(svc.run(ids, True, [], 40000, {}, threading.Event()))
        self.assertEqual(len(engine.prompts), 2)
        self.assertIn("CUSTOM LOW", tok.decode(engine.prompts[1]))
        self.assertNotIn("CUSTOM HIGH", tok.decode(engine.prompts[1]))
        self.assertEqual(''.join(ev.text for kind, ev in events if kind == "event" and ev.kind == "content"), "OK")


if __name__ == "__main__":
    unittest.main()
