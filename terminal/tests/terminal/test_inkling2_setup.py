"""Inkling-2-Mini in setup (install_inkling2, strata_control/ink2_family.py): its pinned files, the engine arguments it writes for a card, the K/V
figure against the engine's own layout, and the places a family is registered."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path
from unittest import mock

import setup
from strata_control import ink2_family as F

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT.parent / "engine" / "src" / "inkling2" / "ink2_engine.cu"


def engine_flags() -> set:
    """The options strata-ink2's main() parses (`a == "--x"`)."""
    return set(re.findall(r'a == "(--[a-z0-9-]+)"', ENGINE.read_text(encoding="utf-8")))


class Files(unittest.TestCase):
    def test_every_file_is_pinned_at_the_revision(self):
        self.assertRegex(F.REVISION, r"^[0-9a-f]{40}$")
        self.assertEqual(setup.HF_REVISIONS[F.REPO], F.REVISION)
        self.assertIs(setup.REPO_PINS[F.REPO], F.FILES)
        for name, (size, sha) in F.FILES.items():
            self.assertGreater(size, 0, name)
            self.assertRegex(sha, r"^[0-9a-f]{64}$", name)
            self.assertNotEqual(set(sha), {"0"}, name)
            self.assertEqual(setup.file_pin(setup.hf(F.REPO) + name), (size, sha))

    def test_what_the_engine_and_the_pack_read_is_downloaded(self):
        for name in ("model-00001-of-00001.safetensors", "mtp.safetensors", "tokenizer.json", "chat_template.jinja", "generation_config.json",
                     "processor_config.json"):
            self.assertIn(name, F.FILES)
        src = ENGINE.read_text(encoding="utf-8")
        self.assertIn('"/model-00001-of-00001.safetensors"', src)
        self.assertIn('"/mtp.safetensors"', src)


class KV(unittest.TestCase):
    def test_the_figure_follows_the_engines_layout(self):
        src = ENGINE.read_text(encoding="utf-8")
        c = {k: int(v) for k, v in re.findall(r"\b(NL|NKV|HD|LOOP_BEGIN|LOOP_END|PASSES|BLOCK|DL) = (\d+)", src)}
        self.assertIn("bool full_layer(int L) { return L % BLOCK == BLOCK - 1; }", src)
        self.assertIn("S.cap = S.full ? E.ctx : WINDOW + PF_RING_EXTRA;", src)          # only the full-attention visits grow with the context
        visits = list(range(c["NL"])) + [l for _ in range(c["PASSES"] - 1) for l in range(c["LOOP_BEGIN"], c["LOOP_END"])]
        full = sum(1 for l in visits if l % c["BLOCK"] == c["BLOCK"] - 1)
        self.assertEqual(full, F.KV_FULL_VISITS)
        self.assertIn("Y.local = i == 0 || i == 2;", src)                               # the drafter: one global layer of DL
        self.assertEqual(c["DL"] - 2, F.DRAFT_FULL_LAYERS)
        self.assertEqual((c["NKV"], c["HD"]), (F.KV_HEADS, F.HEAD_DIM))
        self.assertEqual(F.kv_bytes_per_token(False), 32768)
        self.assertEqual(F.kv_bytes_per_token(True), 36864)                              # measured: 864 MiB from 8K to 32K context
        self.assertEqual(setup.INK2_FAMILIES["inkling2"]["kv_bytes_per_token"], 36864)


class Args(unittest.TestCase):
    def test_resident_on_a_24_gb_card(self):
        ctx = F.recommended_context(24576)
        self.assertEqual(ctx, 65536)
        args = F.engine_args("/m", ctx, 24576, 0, 22)
        self.assertEqual(args[args.index("--spec") + 1], "1")
        self.assertNotIn("--vram-reserve-mib", args)
        self.assertNotIn("--cpu-threads", args)

    def test_offload_on_an_8_gb_card(self):
        ctx = F.recommended_context(8192)                                                   # no desktop on the card: f16 reaches 32K
        self.assertEqual(ctx, 32768)
        args = F.engine_args("/m", ctx, 8192, 0, 22)
        self.assertEqual(args[args.index("--vram-reserve-mib") + 1], "0")
        self.assertEqual(args[args.index("--cpu-threads") + 1], "22")
        self.assertEqual(args[args.index("--kv-grow") + 1], "1")
        self.assertEqual((args[args.index("--kv") + 1], args[args.index("--chunk") + 1]), ("f16", "1024"))
        self.assertNotIn("--spec", args)
        # a desktop holding 500 MiB (measured on the 3090 as an 8 GB card: i8r reads a 30K prompt, refuses a 55K one cleanly): 32K in i8r
        self.assertEqual(F.recommended_context(8192, 1500, 500), 32768)
        self.assertEqual(F.offload_kv(8192, 1500, 500, 32768), "i8r")
        no_cpu = F.engine_args("/m", 32768, 8192, 39, None, 500)
        self.assertEqual((no_cpu[no_cpu.index("--kv") + 1], no_cpu[no_cpu.index("--chunk") + 1]), ("i8r", "1024"))
        self.assertNotIn("--cpu-threads", no_cpu)

    def test_offload_contexts_by_card(self):
        self.assertEqual(F.recommended_context(12288, 1500, 600), 65536)
        self.assertEqual(F.offload_kv(12288, 1500, 600, 65536), "f16")
        self.assertEqual(F.recommended_context(16384, 1500, 800), 131072)
        self.assertEqual(F.recommended_context(7168, 0, 300), 8192)                          # too small for anything better

    def test_the_offload_floor_follows_the_engines_pools(self):
        src = ENGINE.read_text(encoding="utf-8")
        guard = (ENGINE.parent / "ink2_grow_guard.cuh").read_text(encoding="utf-8")
        self.assertIn("constexpr int FETCH_MAX = 64", src)
        self.assertIn("(size_t) (budget * 0.2) / MX_BYTES", src)                          # the MXFP8 pool's share of the room
        self.assertIn("int g_off_margin_mib = 128;", src)
        self.assertIn("int grow_step = 8192;", src)                                       # the K/V grows in 8192-cell steps
        self.assertIn("for (;; T /= 2) {", src)                                           # pf_fit halves the chunk ...
        self.assertIn("if (fits || T <= 256) { g_pf_full = !fits; return T; }", src)      # ... down to 256, then the request ends
        self.assertIn("if (ns - drop < FETCH_MAX) return false;", guard)                   # each pool keeps its minimum
        self.assertIn("ERR the context is full", guard)
        self.assertEqual(F.OFFLOAD_MARGIN_MIB, 128)
        self.assertGreaterEqual(F.MIN_POOLS_MIB * 2**20, 64 * 8211456 / 0.2)               # 64 MXFP8 slots at the least (MX_BYTES)
        self.assertIn("// 4478976 / 8211456", src)
        self.assertEqual(F.offload_room_mib(8192, 8192, 500), F.offload_room_mib(8192, 8192) - 500)   # a desktop's VRAM comes out of the room
        self.assertLess(F.offload_room_mib(8192, 49152, 500, "i8r"), 0)                     # measured: the 55K prompt stopped at 40,960
        self.assertLess(F.offload_room_mib(6144, 8192, 0, "i8r"), 0)                        # a 6 GB card is too small

    def test_the_8_bit_cell_follows_the_engine(self):
        src = ENGINE.read_text(encoding="utf-8")
        self.assertIn('else if (k == "i8r") E.kvm = 3;', src)
        self.assertIn("case 2: return m >= 2 ? (KVD / KVG) * 2 : 0;", src)                  # K scales: fp16 per 64
        self.assertIn("default: return m >= 2 ? (KVD / (m == 4 ? V4G : KVG)) * 2 : 0;", src)  # V scales: per 64 but for k8v4
        self.assertEqual(F.KV_CELL_BYTES["i8r"], 1024 + 1024 + 2 * (1024 // 64) * 2)
        for text, kl in F.KL_VS_F32.items():                                                # the closest 8-bit format to f32 K/V
            self.assertEqual(min(kl, key=lambda k: kl[k][0]), "i8r", text)
            self.assertEqual(min(kl, key=lambda k: kl[k][1]), "i8r", text)

    def test_a_reserve_or_a_long_context_moves_a_card_to_offload(self):
        self.assertTrue(F.resident_fits(24576, 65536))
        self.assertFalse(F.resident_fits(24576, 131072))
        self.assertFalse(F.resident_fits(24576, 65536, 3072))

    def test_every_flag_is_one_the_engine_parses(self):
        known = engine_flags()
        self.assertIn("--vram-reserve-mib", known)
        for vram in (8192, 12288, 16384, 24576, 49152):
            for threads in (None, 7, 22):
                args = F.engine_args("/m", F.recommended_context(vram), vram, 0, threads)
                flags = [x for x in args if x.startswith("--")]
                self.assertEqual(set(flags) - known, set(), (vram, threads))
                self.assertFalse("--spec" in flags and "--cpu-threads" in flags)          # the engine refuses the pair
                self.assertFalse("--cpu-threads" in flags and "--vram-reserve-mib" not in flags)

    def test_cpu_threads_per_socket(self):
        with mock.patch.object(setup, "cpu_core_layout", return_value=(96, [24, 24])), mock.patch.object(setup, "cpu_cores", return_value=None):
            self.assertEqual(setup.ink2_cpu_threads(), 22)                                 # measured best on 2 x 24 cores
        with mock.patch.object(setup, "cpu_core_layout", return_value=(16, [8])), mock.patch.object(setup, "cpu_cores", return_value=None):
            self.assertEqual(setup.ink2_cpu_threads(), 7)


class Registered(unittest.TestCase):
    def test_offered_on_nvidia_linux_only(self):
        with mock.patch.object(setup.sys, "platform", "linux"):
            self.assertTrue(setup.ink2_offered(False))
            self.assertFalse(setup.ink2_offered(True))
        with mock.patch.object(setup.sys, "platform", "win32"):
            self.assertFalse(setup.ink2_offered(False))
        self.assertNotIn("inkling2", setup.OWN_PATH_FAMILIES)                               # that table's installer is the single-GGUF one

    def test_its_config_is_read_back_and_not_tuned(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "strata-inkling2-mini.json"
            p.write_text(json.dumps({"exe": "x/strata-ink2", "arch": "inkling2", "model_name": "inkling-2-mini",
                                     "args": F.engine_args("/m", 65536, 24576, 0, None)}), encoding="utf-8")
            ch = setup.choices_from_config(p)
            self.assertEqual((ch["family"], ch["model"], ch["context"], ch["vision"]), ("inkling2", "mini", 65536, "none"))
            self.assertEqual(setup.calibration_targets([p], None, None), [])

    def test_the_start_leaves_its_arguments_alone(self):
        cfg = {"exe": "x/strata-ink2", "arch": "inkling2", "args": F.engine_args("/m", 32768, 8192, 0, 22)}
        before = list(cfg["args"])
        with mock.patch.object(setup, "write_config") as w:
            out = setup.upgrade_config(Path("strata-inkling2-mini.json"), cfg)
        self.assertEqual(out["args"], before)
        w.assert_not_called()


if __name__ == "__main__":
    unittest.main()
