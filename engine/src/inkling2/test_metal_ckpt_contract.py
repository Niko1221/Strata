"""Source contract for the Metal Inkling backend's prompt-end checkpoint (#6627), runnable anywhere (no Mac needed).

The scheduler (ink_serve.h) only resumes a chat whose last turn ended in hidden thinking / a tool call from the checkpoint its backend
saved at the end of the prompt. A backend that keeps the default (returns false) reads the whole history again every turn: the 0.2.21
release grid measured a median prompt reuse of 0% on Metal. The behavioural test needs a Mac: metal/test_metal_prompt_ckpt_live.py.

  python -m unittest engine/src/inkling2/test_metal_ckpt_contract.py
"""
import os
import re
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, "metal", "ink_metal_backend.mm"), encoding="utf-8").read()


def body(name):
    m = re.search(r"bool %s\(int slot, int seq_len\) override \{" % name, SRC)
    assert m, "ink_metal_backend.mm has no override of %s: the Metal engine never reuses a chat's prompt" % name
    depth, i = 1, m.end()
    while depth:
        depth += {"{": 1, "}": -1}.get(SRC[i], 0)
        i += 1
    return SRC[m.end():i]


class MetalPromptCheckpoint(unittest.TestCase):
    def test_both_hooks_are_overridden(self):
        body("save_prompt_checkpoint")
        body("restore_prompt_checkpoint")

    def test_save_keeps_everything_generation_overwrites(self):
        s = body("save_prompt_checkpoint")
        self.assertIn("runner_.snapshot_slot(slot, ck.sconv)", s, "ShortConv histories")
        self.assertIn("visit_kv_stride", s, "SWA rings of the 40 sliding-window visits")
        self.assertIn("kv_k[v]", s)
        self.assertIn("kv_v[v]", s)
        self.assertIn("dsp_frnn_slot", s, "the drafter's latest f_rnn")
        self.assertIn("dsp_ctx_k[l]", s, "the drafter's ring layers")
        self.assertIn("dspark_is_swa", s)

    def test_restore_puts_it_all_back_and_forgets_the_verify_state(self):
        r = body("restore_prompt_checkpoint")
        for needle in ("runner_.restore_slot(slot, ck.sconv)", "kv_k[v]", "kv_v[v]", "dsp_frnn_slot", "dsp_ctx_k[l]",
                       "last_ver_slot_", "last_tree_slot_", "ck.seq_len != seq_len"):
            self.assertIn(needle, r)

    def test_a_reset_slot_has_no_checkpoint(self):
        # alloc_slot / free_slot / reset_slot all drop it (the CPU backend does the same in reset_slot)
        self.assertEqual(len(re.findall(r"drop_ckpt\(slot\);", SRC)), 3)


class MetalBatchLogits(unittest.TestCase):
    """The runner unembeds the first min(M, 64) rows of a batch; a batch of more than 64 rows with several wanted rows (several requests
    ending their prompt together, --parallel > 1) must be told which rows to read (0.2.21 GRID: wrong first tokens at --parallel 32)."""

    def test_backend_names_the_rows_when_the_batch_is_longer_than_64(self):
        self.assertRegex(SRC, r"(?s)st\.M > 64.*?logits_rows\.push_back\(m\)")

    def test_runner_gathers_those_rows(self):
        runner = open(os.path.join(HERE, "metal", "ink_metal_runner.h"), encoding="utf-8").read()
        self.assertIn("std::vector<int> logits_rows;", runner)
        self.assertIn("b_h_gather", runner)
        self.assertIn("step.logits_rows[j]", runner)


if __name__ == "__main__":
    unittest.main()
