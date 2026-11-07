"""#4472: the engine arguments a benchmark report and `strata` print go through `_redact_argv`, and the benchmark drops
config keys and `STRATA_*` variables through `SECRET_KEY`. Both matched "token" and "key" anywhere in a name, so
`--conversation-cache-min-tokens 20000` was written as `<redacted>`, and a flag with no value whose name matched
(`--no-token-graph`) hid the option after it."""
from __future__ import annotations

import unittest

from strata_cli import bench_facts as bf
from strata_cli.cli import _redact_argv


class RedactArgv(unittest.TestCase):
    def test_counts_with_token_in_the_name_stay_readable(self):
        args = ["--conversation-cache-min-tokens", "20000", "--max-tokens", "64", "--prompt-tokens", "4096"]
        self.assertEqual(_redact_argv(args), args)

    def test_a_flag_without_a_value_does_not_hide_the_next_option(self):
        args = ["--no-token-graph", "--kv", "int8", "--native-ple-key", "--prefill", "16384"]
        self.assertEqual(_redact_argv(args), args)

    def test_credentials_are_still_hidden(self):
        self.assertEqual(_redact_argv(["--api-key", "s3cret", "--hf-token=abc", "--key", "k2", "--password", "pw"]),
                         ["--api-key", "<redacted>", "--hf-token=<redacted>", "--key", "<redacted>",
                          "--password", "<redacted>"])

    def test_a_value_that_starts_with_one_dash_is_still_a_value(self):
        self.assertEqual(_redact_argv(["--api-key", "-x9", "--kv", "int8"]), ["--api-key", "<redacted>", "--kv", "int8"])

    def test_a_secret_option_at_the_end_keeps_its_name(self):
        self.assertEqual(_redact_argv(["--kv", "int8", "--no-api-key"]), ["--kv", "int8", "--no-api-key"])


class SecretKeyNames(unittest.TestCase):
    def test_whole_words_only(self):
        secret = ["api_key", "apikey", "token_secret", "STRATA_HF_TOKEN", "secret_key_base", "STRATA_AUTH", "password"]
        plain = ["fit_max_tokens", "reasoning_budget_tokens", "STRATA_SPEC_MIN_TOKENS", "keyboard_layout", "monkey"]
        self.assertEqual([n for n in secret if not bf.SECRET_KEY.search(n)], [])
        self.assertEqual([n for n in plain if bf.SECRET_KEY.search(n)], [])


if __name__ == "__main__":
    unittest.main()
