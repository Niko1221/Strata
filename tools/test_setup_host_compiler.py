"""setup.py takes a host compiler nvcc accepts when this PC's gcc is newer than the toolkit allows.

    python -m unittest tools.test_setup_host_compiler

The toolkit's own include/crt/host_config.h gives the limit (`#if __GNUC__ > 14` for CUDA 12.9, `> 16` for 13.4), so
no version table has to be kept here: a distro that moves past the toolkit (Arch / CachyOS: gcc 16) is handled by the
gcc the user installed beside it - the same one a llama.cpp CUDA build on that PC needs.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


def run(fn, *args, **kw):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        return fn(*args, **kw), out.getvalue()


class PosixCase(unittest.TestCase):
    """The host-gcc path is the POSIX one (Windows drives MSVC): these tests read it with `setup.WIN` off on any
    host, as test_setup_amd does; the Windows answer itself is pinned in test_windows_returns_no_hint."""

    def setUp(self):
        self.win = setup.WIN
        setup.WIN = False
        self.addCleanup(setattr, setup, "WIN", self.win)


class Limit(PosixCase):
    """The limit comes from the toolkit, not from this file."""

    def toolkit(self, guard):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        root = Path(d.name) / "cuda-12.9"
        (root / "bin").mkdir(parents=True)
        (root / "bin" / "nvcc").write_text("", encoding="utf-8")
        (root / "include" / "crt").mkdir(parents=True)
        (root / "include" / "crt" / "host_config.h").write_text(
            f"#if defined(__GNUC__)\n#if __GNUC__ > {guard}\n#error -- unsupported GNU version!\n#endif\n#endif\n",
            encoding="utf-8")
        return str(root / "bin" / "nvcc")

    def test_cuda_12_9_stops_at_gcc_14(self):
        self.assertEqual(setup.cuda_host_gcc_limit(self.toolkit(14)), 14)

    def test_cuda_13_4_admits_gcc_16(self):
        self.assertEqual(setup.cuda_host_gcc_limit(self.toolkit(16)), 16)

    def test_no_header_means_no_limit(self):
        self.assertIsNone(setup.cuda_host_gcc_limit("/nonexistent/bin/nvcc"))
        self.assertIsNone(setup.cuda_host_gcc_limit(None))


class Pick(PosixCase):
    """What a build does with that limit."""

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        (root / "bin").mkdir()
        (root / "bin" / "nvcc").write_text("", encoding="utf-8")
        (root / "include" / "crt").mkdir(parents=True)
        (root / "include" / "crt" / "host_config.h").write_text("#if __GNUC__ > 14\n#error no\n#endif\n",
                                                                 encoding="utf-8")
        self.nvcc = str(root / "bin" / "nvcc")
        self.gxx14, self.gcc14 = "/usr/bin/g++-14", "/usr/bin/gcc-14"

    def patch(self, default_path, have, default_version="16"):
        """`default_path` = the g++ first on PATH, `default_version` what it reports; `have` = {name: path} else."""
        def which(name):
            return default_path if name == "g++" else have.get(name)

        def fake_out(cmd):
            name = Path(cmd[0]).name
            if name == "g++":
                return f"{default_version}.2.1\n"
            return name.split("-")[-1] if "-" in name else ""
        return [mock.patch.object(setup.shutil, "which", which), mock.patch.object(setup, "out", fake_out)]

    def test_a_default_gcc_nvcc_accepts_changes_nothing(self):
        patches = self.patch("/usr/bin/g++", {"g++-14": self.gxx14, "gcc-14": self.gcc14}, default_version="14")
        with patches[0], patches[1], mock.patch.dict(os.environ, {"CC": "", "CXX": "", "CUDAHOSTCXX": ""}):
            self.assertEqual(setup.cuda_host_compilers(self.nvcc), (None, None))
            self.assertEqual(setup.host_compiler_defs(True, self.nvcc), [])       # the default path, byte-identical

    def test_a_newer_default_takes_the_newest_gcc_the_toolkit_accepts(self):
        patches = self.patch("/usr/bin/g++", {"g++-14": self.gxx14, "gcc-14": self.gcc14, "g++-13": "/usr/bin/g++-13",
                                              "gcc-13": "/usr/bin/gcc-13"})
        with patches[0], patches[1], mock.patch.dict(os.environ, {"CC": "", "CXX": "", "CUDAHOSTCXX": ""}):
            self.assertEqual(setup.cuda_host_compilers(self.nvcc), (self.gcc14, self.gxx14))
            self.assertEqual(setup.host_compiler_defs(True, self.nvcc),
                             ["-DCMAKE_C_COMPILER=/usr/bin/gcc-14",
                              "-DCMAKE_CXX_COMPILER=/usr/bin/g++-14",
                              "-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14"])

    def test_cc_and_cxx_still_win(self):
        patches = self.patch("/usr/bin/g++", {"g++-14": self.gxx14, "gcc-14": self.gcc14})
        with patches[0], patches[1], mock.patch.dict(os.environ, {"CC": "/opt/cc", "CXX": "/opt/c++",
                                                                   "CUDAHOSTCXX": ""}):
            self.assertEqual(setup.host_compiler_defs(True, self.nvcc),
                             ["-DCMAKE_C_COMPILER=/opt/cc", "-DCMAKE_CXX_COMPILER=/opt/c++",
                              "-DCMAKE_CUDA_HOST_COMPILER=/opt/c++"])

    def test_cudahostcxx_is_left_to_supply_the_host_compiler(self):
        """CUDAHOSTCXX may name a compiler outside PATH: nothing is picked for CC / CXX (that would move the C++
        build), and no install is asked for."""
        patches = self.patch("/usr/bin/g++", {})                     # no gcc-N to pick
        with patches[0], patches[1], mock.patch.object(setup, "warn") as warned, \
             mock.patch.dict(os.environ, {"CC": "", "CXX": "", "CUDAHOSTCXX": "/opt/cuda-tools/bin/g++-14"}):
            self.assertEqual(setup.host_compiler_defs(True, self.nvcc),
                             ["-DCMAKE_CUDA_HOST_COMPILER=/opt/cuda-tools/bin/g++-14"])
        self.assertFalse(warned.called)

    def test_a_numbered_gcc_does_not_move_cc_cxx_when_cudahostcxx_is_set(self):
        patches = self.patch("/usr/bin/g++", {"g++-14": self.gxx14, "gcc-14": self.gcc14})
        with patches[0], patches[1], mock.patch.dict(os.environ, {"CC": "", "CXX": "",
                                                                  "CUDAHOSTCXX": "/usr/bin/g++-14"}):
            self.assertEqual(setup.host_compiler_defs(True, self.nvcc),
                             ["-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14"])

    def test_a_cpp_only_build_is_left_alone(self):
        patches = self.patch("/usr/bin/g++", {"g++-14": self.gxx14, "gcc-14": self.gcc14})
        with patches[0], patches[1], mock.patch.dict(os.environ, {"CC": "", "CXX": "", "CUDAHOSTCXX": ""}):
            self.assertEqual(setup.host_compiler_defs(False, self.nvcc), [])

    def test_nothing_acceptable_installed_says_what_to_install(self):
        patches = self.patch("/usr/bin/g++", {})
        with patches[0], patches[1], mock.patch.dict(os.environ, {"CC": "", "CXX": "", "CUDAHOSTCXX": ""}):
            defs, text = run(setup.host_compiler_defs, True, self.nvcc)
        self.assertEqual(defs, [])
        self.assertIn("nvcc accepts up to GCC 14", text)
        # the patched which() resolves no package manager, so this box's hint is the generic fallback - the
        # distro-specific command itself is pinned in InstallHint, and the warn() embedding is in test_the_warning_...
        self.assertIn("install gcc-14 and g++-14 with your package manager", text)
        self.assertIn("CUDAHOSTCXX", text)                  # the escape hatch stays visible


class InstallHint(PosixCase):
    """The command names this distro's package manager; setup prints it and never runs it."""

    def hint_with(self, resolved, distro="cachyos"):
        """install_hint(14) when which() resolves exactly the names in `resolved` and this distro is `distro` ->
        (command, has_escape)."""
        with mock.patch.object(setup.shutil, "which", lambda name: f"/usr/bin/{name}" if name in resolved else None), \
             mock.patch.object(setup, "distro_id", lambda: distro):
            return setup.install_hint(14)

    def test_pacman(self):
        hint, has_escape = self.hint_with({"pacman"}, distro="cachyos")
        self.assertIn("sudo pacman -S gcc14 gcc14-libs", hint)
        self.assertFalse(has_escape)

    def test_plain_arch_uses_the_aur_helper(self):
        """gcc14 is in the AUR on plain Arch, not in its repositories: pacman -S would die on "target not found"."""
        hint, has_escape = self.hint_with({"pacman", "paru"}, distro="arch")
        self.assertEqual(hint, "paru -S gcc14 gcc14-libs")
        self.assertFalse(has_escape)

    def test_plain_arch_without_a_helper_names_the_aur(self):
        hint, has_escape = self.hint_with({"pacman"}, distro="arch")
        self.assertIn("AUR", hint)
        self.assertNotIn("pacman -S", hint)                  # a command that would not install it
        self.assertTrue(has_escape)                          # the way out is named in the hint itself

    def test_apt(self):
        hint, has_escape = self.hint_with({"apt"})
        self.assertEqual(hint, "sudo apt install gcc-14 g++-14")
        self.assertFalse(has_escape)

    def test_dnf(self):
        hint, has_escape = self.hint_with({"dnf"})
        self.assertEqual(hint, "sudo dnf install gcc14 gcc14-c++")
        self.assertFalse(has_escape)

    def test_zypper(self):
        hint, has_escape = self.hint_with({"zypper"})
        self.assertEqual(hint, "sudo zypper install gcc14 gcc14-c++")
        self.assertFalse(has_escape)

    def test_pacman_wins_the_order(self):
        hint, _ = self.hint_with({"pacman", "apt"})
        self.assertIn("pacman", hint)

    def test_nothing_recognised_falls_back_generically(self):
        hint, has_escape = self.hint_with({})
        self.assertIn("install gcc-14 and g++-14 with your package manager", hint)
        self.assertIn("CC, CXX and CUDAHOSTCXX", hint)      # the way out is in the fallback itself
        self.assertNotIn("sudo", hint)
        self.assertTrue(has_escape)                         # the caller must not repeat the escape hatch

    def test_the_commands_are_shell_safe(self):
        """The whole point is a command the user pastes into a shell: nothing in it may be a metacharacter
        (the "(" of an appended parenthetical note broke exactly that).  + . are safe word characters: apt's
        package name is g++-14."""
        for pm in ("pacman", "apt", "dnf", "zypper"):
            hint, _ = self.hint_with({pm})
            self.assertRegex(hint, r"^[A-Za-z0-9 +_.-]+$", f"{pm}: {hint!r}")

    def test_windows_returns_no_hint(self):
        with mock.patch.object(setup, "WIN", True):
            self.assertEqual(setup.install_hint(14), ("", True))

    def test_the_warning_carries_the_command(self):
        """g++ 16 on the box, nvcc stops at 14, no gcc-N installed: warn() must embed this distro's line."""
        def which(name):
            return {"pacman": "/usr/bin/pacman", "g++": "/usr/bin/g++"}.get(name)

        def fake_out(cmd):
            return "16.2.1\n" if Path(cmd[0]).name == "g++" else ""
        with mock.patch.object(setup, "cuda_host_gcc_limit", lambda nvcc: 14), \
             mock.patch.object(setup.shutil, "which", which), \
             mock.patch.object(setup, "distro_id", lambda: "cachyos"), \
             mock.patch.object(setup, "out", fake_out), \
             mock.patch.object(setup, "warn") as warned, \
             mock.patch.dict(os.environ, {"CC": "", "CXX": "", "CUDAHOSTCXX": ""}):
            self.assertEqual(setup.cuda_host_compilers("/usr/local/cuda-12.9/bin/nvcc"), (None, None))
        text = warned.call_args[0][0]
        self.assertIn("sudo pacman -S gcc14 gcc14-libs", text)
        self.assertIn("the CUDA files fail without it", text)


class ThisBox(PosixCase):
    """The toolkits installed here, read from their real headers."""

    def test_the_installed_toolkits(self):
        """`/opt/cuda` is the unversioned path (Arch's package, a local install): whichever release it holds, its
        own header gives the limit - 14 for 12.x, 16 for 13.4 - so the test does not pin one of them."""
        for nvcc, expect in (("/usr/local/cuda-12.9/bin/nvcc", (14,)), ("/usr/local/cuda-13.4/bin/nvcc", (16,)),
                             ("/opt/cuda/bin/nvcc", (14, 16))):
            if Path(nvcc).exists():
                self.assertIn(setup.cuda_host_gcc_limit(nvcc), expect, nvcc)


if __name__ == "__main__":
    unittest.main()
