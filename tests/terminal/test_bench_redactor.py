"""#4483: `bench run --credential-env` took the value "none" (serve's own "no key") as a key, and the redactor replaced
the key as a case-insensitive substring, so "nonempty" in the report became "<key>mpty". The key is now matched as
written and as a whole word, and "none" is no key, as it already was for a run config's api_key."""
from __future__ import annotations

import unittest

from strata_cli import bench_facts as bf


class CredentialValue(unittest.TestCase):
    def test_none_empty_and_blank_are_no_key(self):
        for value in (None, "", "   ", "none", "None", " NONE "):
            self.assertIsNone(bf.credential_value(value), repr(value))

    def test_a_key_is_kept_as_written(self):
        self.assertEqual(bf.credential_value(" abcd1234 "), "abcd1234")


class KeyRedaction(unittest.TestCase):
    def test_the_key_is_replaced_as_a_word(self):
        clean = bf.redactor("abcd1234")
        self.assertEqual(clean("x-api-key: abcd1234 sent"), "x-api-key: <key> sent")
        self.assertEqual(clean("?key=abcd1234&x=1"), "?key=<key>&x=1")

    def test_a_substring_or_another_case_is_not_the_key(self):
        clean = bf.redactor("abcd1234")
        self.assertEqual(clean("sha256 00abcd1234ff"), "sha256 00abcd1234ff")
        self.assertEqual(clean("ABCD1234"), "ABCD1234")

    def test_none_as_the_key_redacts_nothing(self):
        clean = bf.redactor("none")
        text = "time to the first nonempty reasoning; api_key none; None"
        self.assertEqual(clean(text), text)

    def test_a_bearer_token_is_still_hidden_whatever_the_key(self):
        self.assertEqual(bf.redactor("")("Authorization: Bearer zz9zz9zz9zz"), "Authorization: Bearer <key>")


if __name__ == "__main__":
    unittest.main()
