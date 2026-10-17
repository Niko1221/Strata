"""Config "skills_compact": opencode's <available_skills> list keeps each skill's name and the first sentence of its
description, without the location line, so the system message is shorter.  Off (the default), nothing changes."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import serve.frontend as fe  # noqa: E402

LONG = ("Use when the user asks about PDF files. It covers reading, merging, splitting and rotating pages, forms, "
        "watermarks, encryption and OCR on scanned pages, with helper scripts for each step and a long list of pitfalls.")
SYS = ("You are opencode.\n"
       "<available_skills>\n"
       "  <skill>\n"
       "    <name>pdf</name>\n"
       f"    <description>{LONG}</description>\n"
       "    <location>file:///C:/Users/me/.config/opencode/skills/pdf/SKILL.md</location>\n"
       "  </skill>\n"
       "  <skill>\n"
       "    <name>notes</name>\n"
       "    <description>Takes notes</description>\n"
       "    <location>file:///C:/Users/me/.config/opencode/skills/notes/SKILL.md</location>\n"
       "  </skill>\n"
       "</available_skills>\n"
       "Instructions after the list.")


class SkillsCompact(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, fe, "SKILLS_COMPACT", fe.SKILLS_COMPACT)

    def convert(self, on, system=SYS):
        fe.SKILLS_COMPACT = on
        msgs, _, _ = fe.openai_to_messages({"messages": [{"role": "system", "content": system},
                                                         {"role": "user", "content": "hi"}]})
        return msgs

    def test_off_changes_nothing(self):
        self.assertEqual(self.convert(False)[0]["content"], SYS)

    def test_first_sentence_and_no_location(self):
        out = self.convert(True)[0]["content"]
        self.assertIn("<description>Use when the user asks about PDF files.</description>", out)
        self.assertNotIn("<location>", out)
        self.assertNotIn("watermarks", out)
        self.assertIn("<name>pdf</name>", out)
        self.assertIn("<name>notes</name>", out)
        self.assertIn("<description>Takes notes</description>", out)   # no sentence end: kept whole when short
        self.assertTrue(out.startswith("You are opencode.\n<available_skills>"))
        self.assertTrue(out.endswith("</available_skills>\nInstructions after the list."))
        self.assertLess(len(out), len(SYS))

    def test_a_long_first_sentence_is_cut_at_a_word(self):
        desc = "word " * 60
        out = self.convert(True, SYS.replace(LONG, desc))[0]["content"]
        cut = out.split("<description>", 1)[1].split("</description>", 1)[0]
        self.assertLessEqual(len(cut), 164)
        self.assertTrue(cut.endswith(" ..."))

    def test_the_same_list_gives_the_same_text(self):
        self.assertEqual(self.convert(True)[0]["content"], self.convert(True)[0]["content"])

    def test_without_a_skill_list_nothing_changes(self):
        self.assertEqual(self.convert(True, "You are opencode.")[0]["content"], "You are opencode.")

    def test_anthropic_requests_too(self):
        fe.SKILLS_COMPACT = True
        msgs, _, _ = fe.anthropic_to_messages({"system": SYS, "max_tokens": 8,
                                                "messages": [{"role": "user", "content": "hi"}]})
        self.assertNotIn("<location>", msgs[0]["content"])


if __name__ == "__main__":
    unittest.main()
