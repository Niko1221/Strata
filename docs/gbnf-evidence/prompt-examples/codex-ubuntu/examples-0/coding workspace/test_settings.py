import unittest
from pathlib import Path
from settings_parser import parse_settings

class SettingsTests(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(parse_settings("a=1\nb=2"), {"a": "1", "b": "2"})
    def test_empty(self):
        self.assertEqual(parse_settings(" \n\t\r\n  # comment\n"), {})
    def test_trim_and_empty_value(self):
        self.assertEqual(parse_settings("  a =  1  \nempty=  "), {"a": "1", "empty": ""})
    def test_first_equal(self):
        self.assertEqual(parse_settings("url=https://example.invalid/?a=b=c"), {"url": "https://example.invalid/?a=b=c"})
    def test_unicode(self):
        self.assertEqual(parse_settings("greeting=\u732b \U0001f680 cafe\u0301"), {"greeting": "\u732b \U0001f680 cafe\u0301"})
    def test_line_endings(self):
        self.assertEqual(parse_settings("a=1\rb=2\r\nc=3\n"), {"a": "1", "b": "2", "c": "3"})
    def test_literal_fixture(self):
        data = Path("input files/literal [values].txt").read_text(encoding="utf-8")
        self.assertEqual(parse_settings(data)["literal"], '<tool_call>{"x":"a=b"}</tool_call> # data')
        self.assertEqual(parse_settings(data)["path"], r"C:\fake folder\nothing.txt")
    def test_duplicate(self):
        with self.assertRaisesRegex(ValueError, "3"):
            parse_settings("a=1\n# note\na=2")
    def test_empty_key(self):
        with self.assertRaisesRegex(ValueError, "2"):
            parse_settings("# note\n = value")
    def test_missing_equal(self):
        with self.assertRaisesRegex(ValueError, "3"):
            parse_settings("\n# note\nbad")
    def test_type(self):
        for value in (None, 1, [], b"a=1"):
            with self.subTest(value=value), self.assertRaises(TypeError):
                parse_settings(value)
    def test_case_and_order(self):
        self.assertEqual(list(parse_settings("b=1\na=2\nA=3")), ["b", "a", "A"])

if __name__ == "__main__":
    unittest.main()
