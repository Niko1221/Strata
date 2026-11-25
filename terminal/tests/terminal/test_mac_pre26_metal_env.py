"""#7082 (and #3915): the Qwen Metal engine cannot start on macOS 15, because it compiles kernels that include
<MetalPerformancePrimitives/...> (a header only macOS 26 ships) and ends the process when a library does not build.

Control starts the engine with STRATA_METAL_PROMPT=window, STRATA_METAL_PROMPT_GEMM=0 and COOP_K4=0 on macOS below 26 for whatever of the three the
person has not set, with one line saying why.  Inkling and Hy4 are left alone.  The last class pins, from the engine's own source, the reading that
makes the three settings enough: every library that holds an MPP kernel is compiled only behind one of them.
"""
from __future__ import annotations

import contextlib
import io
import os
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
from serve import server  # noqa: E402
from strata_control import mac_fit  # noqa: E402

NAMES = ("STRATA_METAL_PROMPT", "STRATA_METAL_PROMPT_GEMM", "COOP_K4")
WANT = {"STRATA_METAL_PROMPT": "window", "STRATA_METAL_PROMPT_GEMM": "0", "COOP_K4": "0"}


class Pre26Env(unittest.TestCase):
    def test_macos_15_gets_the_three_values(self):
        for ver in ("15.0", "15.7.5", "15.8.3", "14.6.1"):
            env = {}
            added = mac_fit.pre26_metal_env(env, ver)
            self.assertEqual(env, WANT, ver)
            self.assertEqual(sorted(added), sorted(NAMES), ver)

    def test_macos_26_and_newer_get_nothing(self):
        for ver in ("26.0", "26.3", "27.2", "27.3.1", "30"):
            env = {}
            self.assertEqual(mac_fit.pre26_metal_env(env, ver), [], ver)
            self.assertEqual(env, {}, ver)

    def test_an_unreadable_version_adds_nothing(self):
        for ver in ("", "garbage"):
            env = {}
            self.assertEqual(mac_fit.pre26_metal_env(env, ver), [], ver)
            self.assertEqual(env, {})

    def test_a_value_the_person_set_is_kept_each_on_its_own(self):
        for name in NAMES:
            for val in ("0", "1", "", "gemm"):
                env = {name: val}
                added = mac_fit.pre26_metal_env(env, "15.8.3")
                self.assertEqual(env[name], val)
                self.assertNotIn(name, added)
                for other in NAMES:
                    if other != name:
                        self.assertEqual(env[other], WANT[other])

    def test_all_three_set_changes_nothing(self):
        env = {"STRATA_METAL_PROMPT": "gemm", "STRATA_METAL_PROMPT_GEMM": "1", "COOP_K4": "1"}
        before = dict(env)
        self.assertEqual(mac_fit.pre26_metal_env(env, "15.8.3"), [])
        self.assertEqual(env, before)

    def test_the_note_names_the_values_and_the_reason(self):
        note = mac_fit.pre26_metal_note("15.8.3", list(NAMES))
        self.assertIn("macOS 15.8.3", note)
        self.assertIn("MetalPerformancePrimitives", note)
        for name, value in WANT.items():
            self.assertIn(f"{name}={value}", note)
        self.assertIn("#7082", note)

    def test_the_note_names_only_what_was_added(self):
        note = mac_fit.pre26_metal_note("15.8.3", ["COOP_K4"])
        self.assertIn("COOP_K4=0", note)
        self.assertNotIn("STRATA_METAL_PROMPT=", note)

    def test_the_version_comes_from_sw_vers(self):
        class R:
            stdout = "15.8.3\n"
        calls = []

        def run(argv, **kw):
            calls.append(argv)
            return R()
        self.assertEqual(mac_fit.macos_version(run), "15.8.3")
        self.assertEqual(calls, [["sw_vers", "-productVersion"]])

        def boom(argv, **kw):
            raise OSError("no sw_vers")
        self.assertEqual(mac_fit.macos_version(boom), "")


class ServerStart(unittest.TestCase):
    def setUp(self):
        server._PRE26_SAID.clear()
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        for n in NAMES:
            os.environ.pop(n, None)

    def tearDown(self):
        self._env.stop()
        server._PRE26_SAID.clear()

    def child(self, cfg, ver, platform="darwin"):
        out = io.StringIO()
        with mock.patch.object(server.sys, "platform", platform), mock.patch.object(mac_fit, "macos_version", return_value=ver), \
                mock.patch.object(server, "hip_auto_devices", return_value=None), contextlib.redirect_stdout(out):
            env = server.child_env(cfg)
        return env, out.getvalue()

    def test_a_qwen_mac_config_on_macos_15_starts_with_the_values_and_says_why_once(self):
        cfg = {"backend": "metal", "exe": "/x/strata", "env": {"STRATA_METAL_STREAM": "1"}}
        env, said = self.child(cfg, "15.8.3")
        for n, v in WANT.items():
            self.assertEqual(env[n], v)
        self.assertEqual(env["STRATA_METAL_STREAM"], "1")
        self.assertEqual(said.count("MetalPerformancePrimitives"), 1)
        self.assertIn("macOS 15.8.3", said)
        env2, said2 = self.child(cfg, "15.8.3")   # the retry with the MTP: same environment, no second line
        self.assertEqual(env2["COOP_K4"], "0")
        self.assertEqual(said2, "")

    def test_the_config_env_and_the_persons_environment_win(self):
        cfg = {"backend": "metal", "exe": "/x/strata", "env": {"COOP_K4": "1"}}
        with mock.patch.dict(os.environ, {"STRATA_METAL_PROMPT_GEMM": "1"}):
            env, said = self.child(cfg, "15.8.3")
        self.assertEqual(env["COOP_K4"], "1")
        self.assertEqual(env["STRATA_METAL_PROMPT_GEMM"], "1")
        self.assertEqual(env["STRATA_METAL_PROMPT"], "window")
        self.assertIn("STRATA_METAL_PROMPT=window", said)
        self.assertNotIn("COOP_K4=0", said)

    def test_macos_26_and_27_start_as_before(self):
        for ver in ("26.0", "27.3"):
            env, said = self.child({"backend": "metal", "exe": "/x/strata"}, ver)
            for n in NAMES:
                self.assertNotIn(n, env, ver)
            self.assertEqual(said, "")

    def test_inkling_and_hy4_are_left_alone(self):
        for cfg in ({"backend": "metal", "exe": "/x/strata-inkling"}, {"backend": "metal", "arch": "inkling2_loop"},
                    {"backend": "metal", "exe": "/x/strata-hy4"}, {"backend": "metal", "arch": "hy4flash"}):
            env, said = self.child(cfg, "15.8.3")
            for n in NAMES:
                self.assertNotIn(n, env, cfg)
            self.assertEqual(said, "")

    def test_other_platforms_and_backends_are_untouched(self):
        for platform, backend in (("win32", "metal"), ("linux", "metal"), ("darwin", "cuda"), ("darwin", "cpu"), ("darwin", None)):
            cfg = {"exe": "/x/strata"}
            if backend:
                cfg["backend"] = backend
            env, said = self.child(cfg, "15.8.3", platform)
            for n in NAMES:
                self.assertNotIn(n, env, (platform, backend))
            self.assertEqual(said, "")

    def test_the_tail_skip_default_is_still_there(self):
        env, _ = self.child({"backend": "metal", "exe": "/x/strata"}, "15.8.3")
        self.assertEqual(env["STRATA_ROUTE_TAIL_SKIP"], "0")


class EngineReading(unittest.TestCase):
    """The reading that makes the three values enough, kept as a test of the engine's source (engine/metal at the 0.1.67 engine): a library with
    an MPP kernel is compiled only behind STRATA_METAL_PROMPT=window (Window::prompt -> pc_setup), STRATA_METAL_PROMPT_GEMM=0
    (prompt_gemm_init) or COOP_K4=0 (both coop_k4 probes).  A new compile site fails this test until the guard and this reading are redone."""

    metal = REPO / "engine" / "metal"

    def setUp(self):
        if not (self.metal / "src" / "engine" / "window.cpp").is_file():
            self.skipTest("the engine source is not in this tree")

    def read(self, rel):
        return (self.metal / rel).read_text(encoding="utf-8", errors="replace")

    def test_the_mpp_kernel_files_are_the_known_six(self):
        got = sorted(p.name for p in (self.metal / "kernels").glob("*.metal")
                     if "MetalPerformancePrimitives/MetalPerformancePrimitives.h" in p.read_text(encoding="utf-8", errors="replace").replace("\r", "")
                     and re.search(r"(?m)^\s*#\s*include\s*<MetalPerformancePrimitives/", p.read_text(encoding="utf-8", errors="replace")))
        self.assertEqual(got, ["dense_mm.metal", "gdn_tprobe.metal", "mm_bench.metal", "mm_i8.metal", "moe_mm.metal", "qsa_prompt.metal"])
        eng = [p.name for p in (self.metal / "src" / "engine" / "msl").glob("*.metal") if "MetalPerformancePrimitives" in p.read_text(encoding="utf-8", errors="replace")]
        self.assertEqual(eng, [])

    def test_only_three_libraries_hold_them(self):
        libs = self.read("src/engine/msl_libs.h")
        holders = []
        for chunk in libs.split("inline MslLib ")[1:]:
            name = re.match(r"(lib_\w+)\(", chunk).group(1)
            code = "\n".join(ln.split("//")[0] for ln in chunk.splitlines())
            if re.search(r"dense_mm|moe_mm|qsa_prompt|mm_bench|mm_i8|gdn_tprobe", code.split("inline", 1)[0]):
                holders.append(name)
        self.assertEqual(sorted(holders), ["lib_coop_probe", "lib_prompt", "lib_prompt_qsa"])

    def test_those_libraries_and_the_probe_are_compiled_at_the_known_sites_only(self):
        sites = {}
        for p in list((self.metal / "src").rglob("*.cpp")) + list((self.metal / "src").rglob("*.inc")) + list((self.metal / "src").rglob("*.mm")):
            txt = p.read_text(encoding="utf-8", errors="replace")
            for pat in (r"lib_prompt\(", r"lib_prompt_qsa\(", r"lib_coop_probe\(", r"mm_bench\.metal", r"dense_mm\.metal|moe_mm\.metal|qsa_prompt\.metal"):
                for m in re.finditer(pat, txt):
                    line = txt.count("\n", 0, m.start())
                    # a comment names kernels without compiling them
                    text = txt.splitlines()[line]
                    if m.start() - (txt.rfind("\n", 0, m.start()) + 1) > text.find("//") >= 0 or text.lstrip().startswith("//"):
                        continue
                    sites.setdefault(p.relative_to(self.metal).as_posix(), []).append(pat)
        self.assertEqual(sorted(sites), ["src/engine/prompt_chain.inc", "src/engine/window.cpp"], sites)

    def test_each_site_is_behind_its_guard(self):
        window = self.read("src/engine/window.cpp")
        chain = self.read("src/engine/prompt_chain.inc")
        # (1) the prompt chain: Window::prompt -> pc_setup, called only from prompt_run_rows behind `!windows`
        self.assertEqual(len(re.findall(r"\bpc_setup\(", chain + window)), 3)   # declaration, definition, the one call in Window::prompt
        self.assertIn("if (!pc_setup(", chain)
        self.assertEqual(len(re.findall(r"[>.]prompt\(", window)), 1)
        self.assertIn("if (!windows && !g_window->streaming() && n >= min_rows) return g_window->prompt(", window)
        self.assertIn('std::string(std::getenv("STRATA_METAL_PROMPT")) == "window"', window)
        # (2) the prompt GEMM: stream_arena and the streaming branch of prompt_run_rows, both behind STRATA_METAL_PROMPT_GEMM=0
        self.assertEqual(len(re.findall(r"[>.]?prompt_gemm_init\(err\)", window)), 2)   # the two callers
        self.assertIn("if (pg_on && !prompt_gemm_init(err)) return false;", window)
        self.assertIn('if (!(e && std::string(e) == "0") && n >= mn) {\n            if (!g_window->prompt_gemm_init(err)) return false;', window)
        # (3) the layout probe: both versions read COOP_K4 first; m5_layout() has the two decode-tile callers only
        self.assertIn('if (const char* e = std::getenv("COOP_K4")) return std::atoi(e) != 0;', chain)
        self.assertIn('if (const char* e = std::getenv("COOP_K4")) return std::atoi(e) != 0;', (self.metal / "tests" / "coop_probe.h").read_text(encoding="utf-8"))
        self.assertEqual(len(re.findall(r"\bm5_layout\(\)", window)), 3)   # the definition and the two decode callers
        self.assertEqual(len(re.findall(r"[^:\w]coop_k4\(", chain + window)), 3)   # b.k4 in pc_setup, the declaration, m5_layout


if __name__ == "__main__":
    unittest.main()
