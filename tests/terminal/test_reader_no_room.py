"""A launch onto a card another program has filled: the picture reader's out-of-memory lines are read as what they are (strata_control/reader_room.py),
and the launch's error says so in plain words with the next step, not the last line of whatever a debugger printed.  Lane 10 (V100 box), 11-10:
"Python Exception <class 'gdb.error'>: Couldn't get registers: No such process".
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import runner as R
from strata_control import reader_room, split_engine

ALLOC = "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 865.48 MiB on device 0: cudaMalloc failed: out of memory"
BUFFER = "alloc_tensor_range: failed to allocate CUDA0 buffer of size 907524736"
ASSERT = "/data/split-pkg/llama.cpp/ggml/src/ggml-backend.cpp:188: GGML_ASSERT(buffer) failed"
GDB = ["[New LWP 477206]", "❌️ Couldn't get registers: No such process.", "Python Exception <class 'gdb.error'>: Couldn't get registers: No such process."]
MAC = ["WARNING: Using native backtrace. Set GGML_BACKTRACE_LLDB for more info.", "3   strata-vision   0x0000000102a4c9b8 ggml_abort + 132",
       "4   strata-vision   0x0000000102a2f1c0 ggml_backend_alloc_ctx_tensors_from_buft + 220"]


class Find(unittest.TestCase):
    def test_the_three_lines_of_the_v100_launch(self):
        room = reader_room.find([ALLOC, BUFFER, ASSERT])
        self.assertEqual(room.device, 0)
        self.assertAlmostEqual(room.need_mib, 865.48, places=2)

    def test_the_buffer_line_alone_gives_the_size_in_mib(self):
        room = reader_room.find([BUFFER])
        self.assertEqual(room.device, 0)
        self.assertAlmostEqual(room.need_mib, 865.48, places=1)

    def test_amd_buffer_names(self):
        self.assertEqual(reader_room.find(["failed to allocate ROCm2 buffer of size 1048576"]).device, 2)

    def test_the_text_may_come_as_one_string(self):
        self.assertEqual(reader_room.find("\n".join([ALLOC, ASSERT])).device, 0)

    def test_a_plain_out_of_memory_without_numbers_is_still_found(self):
        room = reader_room.find(["CUDA error: cudaMalloc failed: out of memory"])
        self.assertEqual((room.device, room.need_mib), (None, None))

    def test_the_last_failed_allocation_is_the_one_reported(self):
        later = "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 12.00 MiB on device 1: cudaMalloc failed: out of memory"
        room = reader_room.find([ALLOC, later])
        self.assertEqual((room.device, room.need_mib), (1, 12.0))

    def test_lines_that_are_not_about_memory_are_not_found(self):
        for line in ("ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device",       # the harmless one: the reader runs on the processor then
                     "CUDA error: device kernel image is invalid", "strata generate: CUDA error: out of memory",
                     "cudaMalloc failed: CUDA-capable device(s) is/are busy or unavailable", ""):
            with self.subTest(line=line):
                self.assertIsNone(reader_room.find([line]))
        self.assertIsNone(reader_room.find([]))
        self.assertIsNone(reader_room.find(None))

    def test_the_readers_card_number_is_looked_up_in_the_list_it_was_shown(self):
        self.assertEqual(reader_room.physical_card(0, "2,3"), 2)
        self.assertEqual(reader_room.physical_card(1, [2, 3]), 3)
        self.assertEqual(reader_room.physical_card(1, "2,3"), 3)
        self.assertEqual(reader_room.physical_card(0, None), 0)          # it saw every card: its number is the card's
        self.assertEqual(reader_room.physical_card(0, ""), 0)
        self.assertIsNone(reader_room.physical_card(5, "2,3"))
        self.assertIsNone(reader_room.physical_card(0, "GPU-8c3f"))      # a UUID: not a number we can print
        self.assertIsNone(reader_room.physical_card(None, "0"))


class Words(unittest.TestCase):
    """The owner's wording (11-10), exactly."""

    def test_the_summary_line_when_the_processor_reads(self):
        self.assertEqual(reader_room.on_processor(0, 865.48, 114),
                         "Pictures are read on the processor, which is slower: graphics card 0 has no room for the picture reader "
                         "(it needs about 0.8 GB; 0.1 GB are free). To read them on the card, close the other programs that use it and start again.")

    def test_the_error_when_the_launch_ends(self):
        said, nxt = reader_room.could_not_start(0, 865.48, 114)
        self.assertEqual(said, "The picture reader could not start: graphics card 0 has no room for it (it needs about 0.8 GB; 0.1 GB are free).")
        self.assertEqual(nxt, "Close the other programs that use the card and start again, or add --vision off to start without pictures.")

    def test_what_is_not_known_is_left_out(self):
        self.assertEqual(reader_room.could_not_start(2, 865.48, None)[0], "The picture reader could not start: graphics card 2 has no room for it (it needs about 0.8 GB).")
        self.assertEqual(reader_room.could_not_start(2, None, 3000)[0], "The picture reader could not start: graphics card 2 has no room for it (2.9 GB are free).")
        self.assertEqual(reader_room.could_not_start(None, None, None)[0], "The picture reader could not start: the graphics card has no room for it.")
        self.assertEqual(reader_room.could_not_start(None, 865.48, 114)[0],
                         "The picture reader could not start: the graphics card has no room for it (it needs about 0.8 GB; 0.1 GB are free).")
        self.assertIn("the graphics card has no room for the picture reader.", reader_room.on_processor(None))

    def test_a_card_with_next_to_nothing_free_is_not_called_a_tenth(self):
        self.assertIn("(less than 0.1 GB are free)", reader_room.on_processor(0, None, 10))
        self.assertIn("(less than 0.1 GB are free)", reader_room.on_processor(0, None, 0))


class LaunchSaysIt(unittest.TestCase):
    """explain_engine_exit on the engine log a full card leaves, on every platform's version of it."""

    def setUp(self):
        self.free = [114]
        patcher = mock.patch.object(split_engine, "probe_cards", lambda indexes=None: [split_engine.Card(0, "Tesla", 16384, 60, free_mib=self.free[0])] if self.free else [])
        patcher.start()
        self.addCleanup(patcher.stop)

    def log(self, lines, cfg=None):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        log = Path(folder.name) / "engine.log"
        log.write_text("\n".join(lines) + "\n", encoding="utf-8")
        path = Path(folder.name) / "run.json"
        path.write_text(json.dumps({"log": str(log), **(cfg or {})}), encoding="utf-8")
        return path, log

    def explain(self, lines, cfg=None, code=1):
        path, log = self.log(lines, cfg)
        text, nxt = R.explain_engine_exit({"state": "failed", "error": "server_exited", "engine_exit_code": code}, path)
        return text, nxt, log

    NEW2 = "The picture reader could not start: graphics card 0 has no room for it (it needs about 0.8 GB; 0.1 GB are free)."
    STEP = "Close the other programs that use the card and start again, or add --vision off to start without pictures."

    def test_linux_with_gdb_as_seen_on_the_v100_box(self):
        text, nxt, log = self.explain(["loading the model"] + [ALLOC, BUFFER, ASSERT] + GDB)
        self.assertEqual(text, self.NEW2)
        self.assertTrue(nxt.startswith(self.STEP), nxt)
        self.assertIn(str(log), nxt)
        self.assertNotIn("gdb", text + nxt)
        self.assertNotIn("Exception", text + nxt)

    def test_linux_without_a_debugger_the_backtrace_frames_follow(self):
        frames = ["/opt/strata/strata-vision(ggml_abort+0xd0) [0x55d7c4a1f0d0]", "/opt/strata/strata-vision(+0x3f1a2) [0x55d7c4a3f1a2]"]
        text, nxt, _ = self.explain([ALLOC, BUFFER, ASSERT] + frames)
        self.assertEqual(text, self.NEW2)
        self.assertTrue(nxt.startswith(self.STEP))

    def test_macos_style_frames_and_warnings(self):
        text, nxt, _ = self.explain([ALLOC, BUFFER, ASSERT] + MAC)
        self.assertEqual(text, self.NEW2)

    def test_windows_the_assert_is_the_last_line_and_the_exit_code_is_3(self):
        text, nxt, _ = self.explain([ALLOC, BUFFER, ASSERT], code=3)
        self.assertEqual(text, self.NEW2)
        self.assertNotIn("GGML_ASSERT", text + nxt)

    def test_the_free_memory_is_left_out_when_it_cannot_be_read(self):
        self.free.clear()
        text, _, _ = self.explain([ALLOC, BUFFER, ASSERT] + GDB)
        self.assertEqual(text, "The picture reader could not start: graphics card 0 has no room for it (it needs about 0.8 GB).")

    def test_the_card_is_the_one_in_the_config_in_the_engines_order(self):
        text, _, _ = self.explain([ALLOC.replace("device 0", "device 1"), ASSERT], cfg={"gpu": [2, 3], "gpu_order": "as_given"})
        self.assertIn("graphics card 3 has no room", text)

    def test_a_reader_on_its_own_card_names_that_card(self):
        text, _, _ = self.explain([ALLOC, ASSERT], cfg={"gpu": [0, 1], "vision": {"cuda_device": 1}})
        self.assertIn("graphics card 1 has no room", text)

    def test_without_a_number_it_says_the_card(self):
        for cfg in (None, {"gpu": [2, 3], "gpu_order": "as_given"}):
            with self.subTest(cfg=cfg):
                text, _, _ = self.explain(["CUDA error: cudaMalloc failed: out of memory", "GGML_ASSERT failed"], cfg=cfg)
                self.assertEqual(text, "The picture reader could not start: the graphics card has no room for it.")

    def test_the_engines_own_out_of_memory_later_in_the_log_wins(self):
        text, nxt, _ = self.explain([ALLOC, BUFFER, "strata serve: the GPU has too little free VRAM for the prompt path: turn images off (setup: --vision no)"])
        self.assertIn("too little free VRAM", text)
        self.assertEqual(nxt.split(" The engine's log")[0], R.VRAM_STOP_NEXT)

    def test_other_reader_trouble_is_not_called_a_full_card(self):
        text, nxt, _ = self.explain(["ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device"])
        self.assertNotIn("no room", text)
        self.assertIn("The engine stopped while it was loading the model", text)


if __name__ == "__main__":
    unittest.main()
