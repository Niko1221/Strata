"""Real native-protocol scheduler regressions; mock backend checks KV integrity on restore."""
import os
import subprocess
import sys
import unittest

ENGINE = sys.argv.pop(1)
PREFIX = list(range(1, 21))
A = PREFIX + [21, 22, 23, 24]
B = PREFIX + [31, 32, 33, 34]


class Engine:
    def __init__(self, no_checkpoint=False):
        env = dict(os.environ)
        if no_checkpoint:
            env["INK_TEST_NO_CHECKPOINT"] = "1"
        self.p = subprocess.Popen([ENGINE], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, env=env)
        self.info = {}
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise AssertionError("engine exited before READY")
            if line.startswith("INFO "):
                self.info = dict(part.split("=", 1) for part in line.split()[1:] if "=" in part)
            if line.startswith("READY "):
                break

    def request(self, ids, pin=20, slot=None, ckpt=None):
        self.prefill_progress = []
        cmd = "GEN 3" if slot is None else f"BGEN {slot} 0"
        if pin is not None:
            cmd += f" pin={pin}"
        if ckpt is not None:
            cmd += f" ckpt={ckpt}"
        self.p.stdin.write(cmd + " " + ",".join(map(str, ids)) + "\n")
        self.p.stdin.flush()
        tokens, reuse = [], 0
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise AssertionError("engine exited during request")
            if line.startswith("PP "):
                self.prefill_progress.append(int(line.split()[1]))
            if line.startswith("ERR "):
                return line.strip()
            if line.startswith("T "):
                tokens.append(int(line.split()[1]))
            if line.startswith("DONE "):
                reuse = int(line.split()[-1])
                if slot is None:
                    return tokens, reuse
            if line.startswith("BADM ") and slot is not None:
                return tokens, reuse

    def close(self):
        self.p.stdin.write("QUIT\n")
        self.p.stdin.flush()
        self.p.stdin.close()
        self.p.wait(timeout=10)
        self.p.stdout.close()
        if self.p.returncode:
            raise AssertionError(f"engine exit {self.p.returncode}")


class SharedPrefix(unittest.TestCase):
    def setUp(self):
        self.engine = Engine()
        self.addCleanup(self.engine.close)

    def cold(self, ids):
        engine = Engine()
        try:
            return engine.request(ids, pin=None)[0]
        finally:
            engine.close()

    def test_new_conversations_and_repeated_branches_match_cold(self):
        self.assertEqual(self.engine.request(A)[1], 0)
        for prompt in (B, A, B):
            tokens, reuse = self.engine.request(prompt)
            self.assertEqual(reuse, 20)
            self.assertEqual(tokens, self.cold(prompt))

    def test_continuation_prefers_longer_resident_prefix(self):
        tokens, _ = self.engine.request(A)
        followup = A + tokens + [41, 42]
        result, reuse = self.engine.request(followup)
        self.assertGreater(reuse, 20)
        self.assertEqual(result, self.cold(followup))

    def test_prompt_end_checkpoint_survives_shared_checkpoint(self):
        tokens, _ = self.engine.request(A)
        again, reuse = self.engine.request(A)
        self.assertEqual(reuse, len(A) - 1)
        self.assertEqual(tokens, again)

    def test_a_continuation_pass_with_ckpt_0_keeps_the_first_passs_checkpoint(self):
        # the serve layer's thinking-budget wrap-up: pass 2 is the prompt, what pass 1 wrote and a wrap-up; the client's next
        # turn carries the prompt and the reply only (no thinking), so it can only resume from pass 1's prompt-end checkpoint
        self.assertEqual(self.engine.info.get("ckpt_key"), "1")
        first, _ = self.engine.request(A, pin=None)
        cont = A + first + [41, 42, 43, 44, 45, 46, 47, 48]
        _, reuse = self.engine.request(cont, pin=None, ckpt=0)
        self.assertEqual(reuse, len(A) + len(first) - 1)     # (the stream held: the prompt and what it fed, not the last token)
        nxt = A + [51, 52, 53, 54, 55]
        tokens, reuse = self.engine.request(nxt, pin=None)
        self.assertEqual(reuse, len(A) - 1)
        self.assertEqual(tokens, self.cold(nxt))

    def test_without_ckpt_0_the_continuation_pass_replaces_the_checkpoint(self):
        first, _ = self.engine.request(A, pin=None)
        cont = A + first + [41, 42, 43, 44, 45, 46, 47, 48]
        self.engine.request(cont, pin=None)
        nxt = A + [51, 52, 53, 54, 55]
        tokens, reuse = self.engine.request(nxt, pin=None)
        self.assertEqual(reuse, 0)
        self.assertEqual(tokens, self.cold(nxt))

    def test_ckpt_0_in_a_batch_slot_keeps_that_slots_checkpoint(self):
        self.engine.request(A, pin=None, slot=1)
        cont = A + [41, 42, 43]
        self.assertEqual(self.engine.request(cont, pin=None, slot=1, ckpt=0)[1], len(A))
        nxt = A + [51, 52, 53, 54, 55]
        self.assertEqual(self.engine.request(nxt, pin=None, slot=1)[1], len(A) - 1)

    def test_changed_instructions_cold_then_reusable(self):
        self.engine.request(A)
        changed = [40] + B[1:]
        tokens, reuse = self.engine.request(changed)
        self.assertEqual(reuse, 0)
        self.assertEqual(tokens, self.cold(changed))
        next_prompt = changed[:20] + [39, 38, 37, 36]
        tokens, reuse = self.engine.request(next_prompt)
        self.assertEqual(reuse, 20)
        self.assertEqual(tokens, self.cold(next_prompt))

    def test_shorter_rewind_invalidates_later_checkpoints(self):
        self.engine.request(A)
        shorter = PREFIX[:10] + [45, 46]
        self.assertEqual(self.engine.request(shorter, pin=10)[1], 0)
        self.assertEqual(self.engine.request(B)[1], 0)

    def test_slots_never_restore_each_others_state(self):
        self.assertEqual(self.engine.request(A, slot=0)[1], 0)
        self.assertEqual(self.engine.request(B, slot=1)[1], 0)
        self.assertEqual(self.engine.request(B, slot=0)[1], 20)
        self.assertEqual(self.engine.request(A, slot=1)[1], 20)

    def test_unsupported_checkpoint_falls_back_to_prefill(self):
        engine = Engine(no_checkpoint=True)
        self.addCleanup(engine.close)
        engine.request(A)
        tokens, reuse = engine.request(B)
        self.assertEqual(reuse, 0)
        self.assertEqual(tokens, self.cold(B))

    def test_unpinned_branch_does_not_reuse_stale_global_kv(self):
        self.engine.request(A)
        unpinned = PREFIX[:10] + [44] * 15
        self.assertEqual(self.engine.request(unpinned, pin=None)[1], 0)
        tokens, reuse = self.engine.request(B)
        self.assertEqual(reuse, 0)
        self.assertEqual(tokens, self.cold(B))

    def test_aligned_checkpoint_preserves_cold_prefill_batches(self):
        ids = list(range(1, 131)) + [31, 32, 33, 34]
        self.engine.request(ids, pin=128)  # two complete 64-token chunks
        pinned_progress = self.engine.prefill_progress
        cold = Engine()
        self.addCleanup(cold.close)
        cold.request(ids, pin=None)
        self.assertEqual(pinned_progress, cold.prefill_progress)

    def test_pin_validation(self):
        for pin in (0, -1, "garbage", "20x", 2147483648, len(A)):
            self.assertTrue(self.engine.request(A, pin=pin).startswith("ERR "), pin)
        self.assertEqual(self.engine.request(A)[1], 0)

    def test_info_reports_shared_prefix_capability(self):
        self.assertEqual(self.engine.info.get("shared_prefix"), "1")
        no_ckpt = Engine(no_checkpoint=True)
        self.addCleanup(no_ckpt.close)
        self.assertEqual(no_ckpt.info.get("shared_prefix"), "0")

    def test_bco_reset_invalidates_shared_checkpoint(self):
        self.assertEqual(self.engine.request(A, slot=0)[1], 0)
        self.engine.p.stdin.write("BCO 0 1,2,3\n")
        self.engine.p.stdin.flush()
        self.assertTrue(self.engine.p.stdout.readline().startswith("BCOK 0 "))
        tokens, reuse = self.engine.request(B)
        self.assertEqual(reuse, 0)
        self.assertEqual(tokens, self.cold(B))

    def test_concurrent_slots_warm_and_reuse_independently(self):
        # Admit slot 1 with slot_read=1 so it stays active while slot 0 prefills and saves its checkpoint.
        self.engine.p.stdin.write("BGEN 1 8 slot_read=1 " + ",".join(map(str, B)) + "\n")
        self.engine.p.stdin.flush()
        while not self.engine.p.stdout.readline().startswith("BADM 1 "):
            pass
        self.engine.p.stdin.write("BGEN 0 0 pin=20 " + ",".join(map(str, A)) + "\n")
        self.engine.p.stdin.flush()
        slot0_done, slot1_done = False, False
        while not (slot0_done and slot1_done):
            line = self.engine.p.stdout.readline()
            if not line:
                raise AssertionError("engine exited during concurrent batch")
            if line.startswith("BADM 0 "):
                slot0_done = True
            if line.startswith("BDONE 1 "):
                slot1_done = True
        tokens, reuse = self.engine.request(B)
        self.assertEqual(reuse, 20)
        self.assertEqual(tokens, self.cold(B))


if __name__ == "__main__":
    unittest.main()
