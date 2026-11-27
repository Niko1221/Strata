"""Real native-protocol scheduler regressions; mock backend checks KV integrity on restore."""
import os
import subprocess
import sys
import time
import unittest

ENGINE = sys.argv.pop(1)
PREFIX = list(range(1, 21))
A = PREFIX + [21, 22, 23, 24]
B = PREFIX + [31, 32, 33, 34]


class Engine:
    def __init__(self, no_checkpoint=False, tail_role=None):
        env = dict(os.environ)
        if no_checkpoint:
            env["INK_TEST_NO_CHECKPOINT"] = "1"
        if tail_role is not None:
            env["INK_TEST_TAIL_ROLE"] = str(tail_role)
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


class TailLevel(unittest.TestCase):
    """#7241 (--tail-role-token ROLE): the thinking level at the END of the prompt.  A prompt is the conversation HISTORY, then (for a level other
    than the default) a short system turn that starts with the ROLE token, then the model's turn.  With the switch on, the prompt checkpoint is one
    token in front of that turn, so a request that changes only the level reads one token plus the tail; with it off (the default) the checkpoint
    is at the end of the prompt and a changed tail reads everything again.  The tokens never change."""
    ROLE = 5
    HIST = list(range(10, 40))                        # 30 tokens of conversation
    HIST2 = HIST + list(range(40, 46)) + [47, 48]     # the next turn of the same chat
    TAIL_MED = [5, 6, 7, 8]
    TAIL_LOW = [5, 6, 9, 8]

    def chain(self):
        return (("med", self.HIST + self.TAIL_MED), ("low", self.HIST + self.TAIL_LOW), ("default", self.HIST),
                ("med_again", self.HIST + self.TAIL_MED), ("next_med", self.HIST2 + self.TAIL_MED), ("next_default", self.HIST2))

    def run_chain(self, role):
        e = Engine(tail_role=role)
        try:
            return {name: e.request(ids, pin=None) for name, ids in self.chain()}
        finally:
            e.close()

    def test_a_level_switch_reads_only_the_tail_when_the_switch_is_on(self):
        on = self.run_chain(self.ROLE)
        n, n2 = len(self.HIST), len(self.HIST2)
        self.assertEqual(on["med"][1], 0)
        self.assertEqual(on["low"][1], n - 1)             # another level, same conversation: the checkpoint in front of the tail
        self.assertEqual(on["default"][1], n - 1)         # no tail at all
        self.assertEqual(on["med_again"][1], n - 1)
        self.assertEqual(on["next_med"][1], n - 1)        # the chat moved on: its history still starts with the checkpoint's tokens
        self.assertEqual(on["next_default"][1], n2 - 1)   # and the new checkpoint (one token in front of the new tail) serves the next level switch

    def test_without_the_switch_a_changed_tail_reads_everything_again(self):
        off = self.run_chain(None)
        self.assertEqual(off["low"][1], 0)                # 0.2.22 / default: the checkpoint sat at the end of the first prompt

    def test_the_tokens_are_the_same_on_and_off_and_cold(self):
        on, off = self.run_chain(self.ROLE), self.run_chain(None)
        for name, ids in self.chain():
            self.assertEqual(on[name][0], off[name][0], name)
            cold = Engine()                                # a fresh engine reads the whole prompt: the reference
            try:
                self.assertEqual(on[name][0], cold.request(ids, pin=None)[0], name)
            finally:
                cold.close()

    def test_a_system_turn_at_the_start_is_not_a_tail(self):
        head = [5, 6, 7, 8] + list(range(10, 40))         # the head's own level turn, 34 tokens, no tail
        e = Engine(tail_role=self.ROLE)
        try:
            e.request(head, pin=None)
            self.assertEqual(e.request(head + [41], pin=None)[1], len(head) - 1)    # the ordinary end-of-prompt checkpoint
        finally:
            e.close()


class QuitShutdown(unittest.TestCase):
    """QUIT, then stdin closed while the backend is still being torn down: the exit code must be 0 every time.

    The stdin reader is a detached thread sitting in getline().  It used to keep its mutex and queue in the serve loop's frame, so the
    EOF that follows QUIT (the client closes stdin a moment later) reached a destroyed mutex (std::terminate, "Invalid argument") or a
    freed frame (segmentation fault).  And with stdin still open the process did not exit at all (exit() waited for the reader).
    """

    def run_once(self, teardown_ms, close_after_s):
        """close_after_s: None leaves stdin open, 0 closes it at once, else after that many seconds (the backend is torn down meanwhile)."""
        env = dict(os.environ)
        env["INK_TEST_SLOW_TEARDOWN_MS"] = str(teardown_ms)
        p = subprocess.Popen([ENGINE], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        while not p.stdout.readline().startswith("READY "):
            pass
        p.stdin.write("QUIT\n")
        p.stdin.flush()
        if close_after_s is not None:
            time.sleep(close_after_s)
            p.stdin.close()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
            self.fail("the engine did not exit after QUIT (stdin %s)" % ("open" if close_after_s is None else "closed"))
        err = p.stderr.read()
        p.stdout.close()
        p.stderr.close()
        self.assertEqual(p.returncode, 0, f"engine exit {p.returncode}: {err[-300:]}")

    def test_stdin_closed_at_once(self):
        for _ in range(10):
            self.run_once(100, 0)

    def test_stdin_closed_while_the_backend_is_torn_down(self):
        for _ in range(20):
            self.run_once(400, 0.05)

    def test_stdin_left_open(self):
        for _ in range(5):
            self.run_once(50, None)


if __name__ == "__main__":
    unittest.main()
