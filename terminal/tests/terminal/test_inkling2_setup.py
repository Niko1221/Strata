"""Inkling-2-Mini's official checkpoint in setup (install_ink2, strata_control/ink2_family.py): its pinned files, the engine arguments it writes for a card, the K/V
figure against the engine's own layout, and the places a family is registered."""

from __future__ import annotations

import json
import os
import re
import tempfile
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
        self.assertEqual(F.kv_bytes_per_token(True), 36864)


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
        self.assertIn("(size_t) (budget * 0.2) / pool_slot(false)", src)                  # the MXFP8 pool's share of the room ...
        self.assertIn("size_t pool_slot(bool nv) { return ex_bytes(nv ? 0 : NL - 1); }", src)
        self.assertIn("g_is_gguf ? g_gg_a[r] + g_gg_b[r] : r ? MX_BYTES : NV_BYTES;", src)   # ... MX_BYTES a slot for the checkpoint
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
                args = F.engine_args("/m", F.recommended_context(vram), vram, 0, threads, 0, 4096)
                flags = [x for x in args if x.startswith("--")]
                self.assertEqual(set(flags) - known, set(), (vram, threads))
                self.assertFalse("--spec" in flags and "--cpu-threads" in flags)          # the engine refuses the pair
                self.assertFalse("--cpu-threads" in flags and "--vram-reserve-mib" not in flags)

    def test_conversation_cache_from_the_ram(self):
        self.assertEqual(F.conversation_cache_mib(187, True)[0], 4096)
        self.assertEqual(F.conversation_cache_mib(29.8, True)[0], 4096)                    # a 32 GB PC, experts in RAM: 6.4 GiB spare
        self.assertEqual(F.conversation_cache_mib(27.0, True)[0], 2048)
        self.assertEqual(F.conversation_cache_mib(25.0, True), (0, 25.0 - 16.8 - 0.65 - 6.0))   # off below 1.5 GiB left
        self.assertEqual(F.conversation_cache_mib(16, False)[0], 4096)                     # resident: the experts are not in RAM
        args = F.engine_args("/m", 32768, 8192, 0, 22, 0, 2048)
        self.assertEqual(args[args.index("--conversation-cache-mib") + 1], "2048")
        self.assertNotIn("--conversation-cache-mib", F.engine_args("/m", 32768, 8192, 0, 22))
        # the RAM setup asks of an offload PC holds the engine, the pictures' weights, the OS and the largest cache
        self.assertGreaterEqual(F.RAM_GB * 1e9 / 2**30, F.OFFLOAD_RAM_GIB + F.VISION_RAM_GIB + F.OS_RAM_GIB + F.CACHE_MIB[0] / 1024)
        self.assertTrue(F.VISION)
        self.assertIn("processor_config.json", F.FILES)

    def test_cpu_threads_per_socket(self):
        with mock.patch.object(setup, "cpu_core_layout", return_value=(96, [24, 24])), mock.patch.object(setup, "cpu_cores", return_value=None):
            self.assertEqual(setup.ink2_cpu_threads(), 22)                                 # measured best on 2 x 24 cores
        with mock.patch.object(setup, "cpu_core_layout", return_value=(16, [8])), mock.patch.object(setup, "cpu_cores", return_value=None):
            self.assertEqual(setup.ink2_cpu_threads(), 7)


class Registered(unittest.TestCase):
    def test_offered_on_linux_for_cuda_and_hip(self):
        with mock.patch.object(setup, "CONTROL", False):
            with mock.patch.object(setup.sys, "platform", "linux"):
                self.assertTrue(setup.ink2_offered(False))
                self.assertTrue(setup.ink2_offered(True))
            with mock.patch.object(setup.sys, "platform", "win32"):
                self.assertFalse(setup.ink2_offered(False))
                self.assertFalse(setup.ink2_offered(True))
            with mock.patch.object(setup.sys, "platform", "darwin"):
                self.assertFalse(setup.ink2_offered(False))
                self.assertFalse(setup.ink2_offered(True))
        with mock.patch.object(setup, "CONTROL", True), mock.patch.object(setup.sys, "platform", "linux"):
            self.assertFalse(setup.ink2_offered(False))
            self.assertFalse(setup.ink2_offered(True))
        self.assertNotIn("inkling2", setup.OWN_PATH_FAMILIES)                               # that table's installer is the single-GGUF one

    def test_hip_arch_set_allows_wave32_and_refuses_gfx906(self):
        self.assertIn("gfx1151", setup.INK2_HIP_ARCHS)
        self.assertIn("gfx1201", setup.INK2_HIP_ARCHS)
        self.assertIn("gfx1100", setup.INK2_HIP_ARCHS)
        self.assertNotIn("gfx906", setup.INK2_HIP_ARCHS)
        self.assertIn("AMD", F.ABOUT)
        self.assertIn("NVIDIA", F.ABOUT)
        args = mock.MagicMock()
        with mock.patch.object(setup, "CONTROL", False), mock.patch.object(setup.sys, "platform", "linux"), \
                mock.patch.object(setup, "say"), mock.patch.object(setup, "ok"), \
                mock.patch.object(setup, "fail", side_effect=RuntimeError("refused")) as fail_mock:
            with self.assertRaises(RuntimeError):
                setup.install_ink2(args, "inkling-2-mini", {"arch": "gfx906", "vram_gb": 32.0}, True,
                                       [{"arch": "gfx906"}], 13, [], 8080, 64.0, None)
        self.assertIn("gfx906", fail_mock.call_args[0][0])

    def test_its_config_is_read_back_and_not_tuned(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "strata-inkling-2-mini-official.json"
            p.write_text(json.dumps({"exe": "x/strata-ink2", "arch": "inkling2", "model_name": "inkling-2-mini",
                                     "args": F.engine_args("/m", 65536, 24576, 0, None)}), encoding="utf-8")
            ch = setup.choices_from_config(p)
            self.assertEqual((ch["family"], ch["model"], ch["context"], ch["vision"]), ("inkling-2-mini", "official", 65536, "none"))
            self.assertEqual(setup.calibration_targets([p], None, None), [])

    def test_the_start_leaves_its_arguments_alone(self):
        cfg = {"exe": "x/strata-ink2", "arch": "inkling2", "args": F.engine_args("/m", 32768, 8192, 0, 22)}
        before = list(cfg["args"])
        with mock.patch.object(setup, "write_config") as w:
            out = setup.upgrade_config(Path("strata-inkling-2-mini-official.json"), cfg)
        self.assertEqual(out["args"], before)
        w.assert_not_called()


class OneFamily(unittest.TestCase):
    """The official checkpoint is a source ("model") of upstream's Inkling family: install_inkling2 hands it to install_ink2 (strata-ink2), the
    SQ builds stay on strata-inkling."""

    def test_the_family_lists_it_and_routes_it_to_strata_ink2(self):
        fam = setup.INKLING_FAMILIES["inkling-2-mini"]
        self.assertNotIn(F.MODEL, fam["models"])                                         # the catalog sizes every entry as an SQ build
        self.assertTrue({"SQ-20", "SQ-15", "SQ-10"} <= set(fam["models"]))
        self.assertNotIn("INK2_FAMILIES", vars(setup))
        src = setup.INKLING_COMMUNITY_SOURCES["safetensors"]
        self.assertEqual((src["exe"], src["arch"], src["model"]), (F.EXE, F.ARCH, F.MODEL))
        from serve import inkling_kind
        cfg = {"exe": "/x/engine/strata-ink2", "arch": F.ARCH, "args": F.engine_args("/m", 32768, 24576, 0, None)}
        self.assertFalse(inkling_kind.is_inkling(cfg))                                   # never strata-inkling's start rules

    def test_install_inkling2_hands_the_official_source_over(self):
        args = mock.MagicMock(model=F.MODEL, context=None, vision=None)
        with mock.patch.object(setup, "install_ink2", return_value=7) as ink2, mock.patch.object(setup, "say"), \
                mock.patch.object(setup, "ok"), mock.patch.object(setup, "ink2_offered", return_value=True):
            got = setup.install_inkling2(args, "inkling-2-mini", {"vram_gb": 24.0, "arch": 86}, False, None, 13, [], 8080, 64.0, None)
        self.assertEqual(got, 7)
        self.assertEqual(ink2.call_args[0][1], "inkling-2-mini")


class Build(unittest.TestCase):
    def test_hip_build_cmake_args_and_stamp_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rocm = root / "rocm"
            clang = rocm / "llvm" / "bin" / "clang++"
            clang.parent.mkdir(parents=True)
            clang.write_text("", encoding="utf-8")
            bitcode = rocm / "lib" / "llvm" / "amdgcn" / "bitcode"
            bitcode.mkdir(parents=True)
            extra_pkg = root / "rocm_libs"
            dirs = [str(rocm / "lib"), str(extra_pkg / "lib")]
            builds = []

            def fake_cmake(src, bdir, target, defs, vcvars, bat):
                builds.append((bdir, target, list(defs)))
                bdir.mkdir(parents=True, exist_ok=True)
                (bdir / F.EXE).write_text("hip-bin", encoding="utf-8")

            src_hash = ["h1"]
            with mock.patch.object(setup, "ROOT", root), \
                    mock.patch.object(setup, "engine_src", return_value=root / "engine-src"), \
                    mock.patch.object(setup, "rocm_root", return_value=(rocm, dirs)) as rocm_mock, \
                    mock.patch.object(setup, "cmake_build", side_effect=fake_cmake), \
                    mock.patch.object(setup, "source_hash", side_effect=lambda _parts: src_hash[0]), \
                    mock.patch.object(setup.shutil, "which", side_effect=lambda name: f"/usr/bin/{name}"), \
                    mock.patch.object(setup, "say"), mock.patch.object(setup, "ok"), \
                    mock.patch.dict(os.environ, {}, clear=False):
                gpu = {"arch": "gfx1151", "archs": ["gfx1151"]}
                exe, got_dirs = setup.ink2_build(gpu, True, hip=True, amd=[gpu])
                self.assertEqual(exe, root / "engine" / F.EXE)
                self.assertEqual(got_dirs, dirs)
                self.assertEqual(len(builds), 1)
                bdir, target, defs = builds[0]
                self.assertEqual(bdir, root / "build-ink2-hip")
                self.assertEqual(target, F.EXE)
                self.assertIn("-DSTRATA_ENABLE_HIP=ON", defs)
                self.assertIn("-DSTRATA_ENABLE_CUDA=OFF", defs)
                self.assertIn("-DSTRATA_BUILD_TESTS=OFF", defs)
                self.assertIn("-DCMAKE_HIP_ARCHITECTURES=gfx1151", defs)
                self.assertIn(f"-DCMAKE_HIP_COMPILER={clang}", defs)
                self.assertIn(f"-DCMAKE_HIP_COMPILER_ROCM_ROOT={rocm}", defs)
                self.assertIn(f"-DCMAKE_PREFIX_PATH={rocm};{extra_pkg}", defs)
                self.assertIn(f"-DCMAKE_HIP_FLAGS=-isystem {rocm}/include --rocm-path={rocm} --rocm-device-lib-path={bitcode}", defs)
                rocm_mock.assert_called_once_with(["gfx1151"])

                stamp = json.loads((root / "engine" / (F.EXE + ".json")).read_text(encoding="utf-8"))
                self.assertEqual(stamp, {"backend": "hip", "archs": ["gfx1151"], "src": "h1", "lib_dirs": dirs})

                exe2, got_dirs2 = setup.ink2_build({"arch": "gfx1151"}, True, hip=True, amd=[gpu])
                self.assertEqual((exe2, got_dirs2), (exe, dirs))
                self.assertEqual(len(builds), 1)

                src_hash[0] = "h2"
                setup.ink2_build({"arch": "gfx1151"}, True, hip=True, amd=[gpu])
                self.assertEqual(len(builds), 2)

                setup.ink2_build({"arch": "gfx1201"}, True, hip=True)
                self.assertEqual(len(builds), 3)
                self.assertIn("-DCMAKE_HIP_ARCHITECTURES=gfx1201", builds[2][2])

    def test_cuda_build_rebuilds_cleanly_over_a_hip_stamp(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            eng = root / "engine"
            eng.mkdir()
            (eng / F.EXE).write_text("hip-bin", encoding="utf-8")
            (eng / (F.EXE + ".json")).write_text(
                json.dumps({"backend": "hip", "archs": ["gfx1151"], "src": "h1", "lib_dirs": ["/rocm/lib"]}),
                encoding="utf-8",
            )
            nvcc = root / "cuda" / "bin" / "nvcc"
            nvcc.parent.mkdir(parents=True)
            nvcc.write_text("", encoding="utf-8")
            builds = []

            def fake_cmake(src, bdir, target, defs, vcvars, bat):
                builds.append((bdir, target, list(defs)))
                bdir.mkdir(parents=True, exist_ok=True)
                (bdir / F.EXE).write_text("cuda-bin", encoding="utf-8")

            with mock.patch.object(setup, "ROOT", root), \
                    mock.patch.object(setup, "engine_src", return_value=root / "engine-src"), \
                    mock.patch.object(setup, "install_build_tools", return_value=(str(nvcc), None)), \
                    mock.patch.object(setup, "toolkit_root_defs", return_value=[]), \
                    mock.patch.object(setup, "host_compiler_defs", return_value=[]), \
                    mock.patch.object(setup, "cmake_build", side_effect=fake_cmake), \
                    mock.patch.object(setup, "source_hash", return_value="h1"), \
                    mock.patch.object(setup, "say"), mock.patch.object(setup, "ok"):
                exe, _ = setup.ink2_build({"arch": 86}, True, hip=False)
                self.assertEqual(exe, eng / F.EXE)
                self.assertEqual(len(builds), 1)
                self.assertEqual(builds[0][0], root / "build-ink2")
                self.assertIn("-DSTRATA_ENABLE_CUDA=ON", builds[0][2])
                self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=86", builds[0][2])

    def test_source_hash_tracks_hip_wmma_and_hip_compat(self):
        with tempfile.TemporaryDirectory() as d:
            eng = Path(d)
            (eng / "CMakeLists.txt").write_text("cmake", encoding="utf-8")
            (eng / "src" / "inkling2").mkdir(parents=True)
            (eng / "include" / "strata" / "hip_compat").mkdir(parents=True)
            wmma = eng / "src" / "inkling2" / "ink2_wmma_hip.cuh"
            compat = eng / "include" / "strata" / "hip_compat" / "hip_compat.h"
            wmma.write_text("v1", encoding="utf-8")
            compat.write_text("c1", encoding="utf-8")
            with mock.patch.object(setup, "engine_src", return_value=eng):
                h0 = setup.source_hash(setup.INK2_SOURCES)
                wmma.write_text("v2", encoding="utf-8")
                h1 = setup.source_hash(setup.INK2_SOURCES)
                compat.write_text("c2", encoding="utf-8")
                h2 = setup.source_hash(setup.INK2_SOURCES)
            self.assertNotEqual(h0, h1)
            self.assertNotEqual(h1, h2)


if __name__ == "__main__":
    unittest.main()
