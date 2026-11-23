"""The Unsloth Inkling GGUF downloads are pinned to a commit (0221-DSPARK, release blocker found 2026-11-23).

setup verifies every downloaded GGUF against the byte size and SHA-256 table in strata_control/inkling_family.py. On 2026-11-23 14:54 UTC
(commit 19bf97a66719d0165f6f8082a9d59581912a3863 "Support mainline llama.cpp") unsloth/Inkling-2-Mini-GGUF replaced every main GGUF (+1,280 bytes
each) and the drafter (402,278,560 -> 503,889,952 bytes, another tensor layout) under the same file names, so resolve/main no longer matches the
table and every install of a GGUF would fail its check. The shipped source is the commit whose bytes the table describes."""
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strata_control import inkling_family as F  # noqa: E402

PINNED = "800c946d7b9c830c86290df49afbbe91db8455f5"


class GgufPin(unittest.TestCase):
    def test_the_shipped_gguf_source_is_a_commit_not_main(self):
        src = F.gguf_sources(env={})                      # no override: the shipped file
        self.assertTrue(src, "no shipped GGUF source")
        base = src[0]["base"]
        self.assertTrue(re.search(r"^https://huggingface\.co/unsloth/Inkling-2-Mini-GGUF/resolve/[0-9a-f]{40}$", base), base)
        self.assertEqual(base.rsplit("/", 1)[1], PINNED)

    def test_every_gguf_url_goes_through_the_pin(self):
        src = F.sources_for(F.DSPARK_GGUF_FILE, env={})[0]
        for name in list(F.UNSLOTH_GGUF_FILES):
            urls = F.urls_for(name, src)
            self.assertEqual(urls, [f"https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/{PINNED}/{name}"])

    def test_the_table_still_describes_the_pinned_bytes(self):
        # the pin is only right while the table is the one measured against it (check_pin.py compared all 13 files to the repo at that commit)
        self.assertEqual(F.UNSLOTH_GGUF_FILES["Inkling-2-Mini-UD-Q4_K_XL.gguf"][0], 17_629_759_456)
        self.assertEqual(F.UNSLOTH_GGUF_FILES[F.DSPARK_GGUF_FILE], (402_278_560, "2ec78510da837e8fc3be54a14757a0c5fbd2d8d9e85a0d92853fafdf674f3773"))
        for name, (size, sha) in F.UNSLOTH_GGUF_FILES.items():
            self.assertGreater(size, 100_000_000, name)
            self.assertEqual(len(sha), 64, name)

    def test_an_override_still_wins(self):
        src = F.gguf_sources(env={F.GGUF_SOURCE_ENV: "https://example.org/gguf"})
        self.assertEqual(src[0]["base"], "https://example.org/gguf")
        self.assertEqual(src[-1]["base"].rsplit("/", 1)[1], PINNED)


if __name__ == "__main__":
    unittest.main()
