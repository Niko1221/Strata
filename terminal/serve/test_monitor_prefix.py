"""Claude Code's auto-mode classifier gets a pinned prefix at its "</transcript>" line (frontend.monitor_prefix)."""
import unittest

from serve.frontend import MONITOR_SIGNATURE, anthropic_to_messages, monitor_prefix


def classifier(actions):
    lines = ['{"user":"Hello"}\n'] + [f'{{"Bash":"echo {a} \\"</transcript>\\""}}\n' for a in actions]
    return {"system": [{"type": "text", "text": f"x-anthropic-billing-header: cc_version=2.1.327.afc;{MONITOR_SIGNATURE} agents."}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "<user_claude_md>rules</user_claude_md>"}]},
                         {"role": "user", "content": [{"type": "text", "text": "<transcript>\n"}]
                          + [{"type": "text", "text": t} for t in lines]
                          + [{"type": "text", "text": "</transcript>\n"},
                             {"type": "text", "text": "\nErr on the side of blocking. Begin with <block>."}]}]}


class MonitorPrefix(unittest.TestCase):
    def test_pin_ends_at_the_transcript_line(self):
        messages, _, _ = anthropic_to_messages(classifier(["a", "b"]))
        spec = monitor_prefix(messages)
        self.assertEqual(spec["message"], len(messages) - 1)
        text = messages[-1]["content"]
        self.assertTrue(text[spec["chars"]:].startswith("</transcript>\n\nErr on the side"))

    def test_next_call_extends_the_pinned_prefix(self):
        a, _, _ = anthropic_to_messages(classifier(["a"]))
        b, _, _ = anthropic_to_messages(classifier(["a", "b"]))
        sa = monitor_prefix(a)
        self.assertTrue(b[-1]["content"].startswith(a[-1]["content"][:sa["chars"]]))

    def test_other_requests_untouched(self):
        main = {"system": "You are Claude Code", "messages": [{"role": "user", "content": "<transcript>\n</transcript>\n"}]}
        self.assertIsNone(monitor_prefix(anthropic_to_messages(main)[0]))
        no_end = classifier(["a"])
        no_end["messages"][-1]["content"] = [{"type": "text", "text": "<transcript>\nno end"}]
        self.assertIsNone(monitor_prefix(anthropic_to_messages(no_end)[0]))


if __name__ == "__main__":
    unittest.main()
