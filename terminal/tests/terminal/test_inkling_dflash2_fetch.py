"""The DFlash 2 sidecar (dflash2.safetensors) of an Inkling .strata model: Control fetches it beside the model on a new install and on update, verifies it against its pins,
shows its size, and uses DSpark when it is missing or the download fails. The engine drafts with it by default (up to 16 slots) and falls back to DSpark without it.

    python -I terminal/tests/terminal/test_inkling_dflash2_fetch.py
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strata_control import inkling_family as F  # noqa: E402

import setup  # noqa: E402


class Dflash2PinTest(unittest.TestCase):
    def test_every_pin_is_a_size_and_a_sha(self):
        pins = F.pins_for(F.DFLASH2_FILE)
        self.assertGreaterEqual(len(pins), 1)
        for size, sha in pins:
            self.assertGreater(size, 100_000_000)
            self.assertRegex(sha, r"^[0-9a-f]{64}$")
        self.assertIn((989_975_616, "217980ab00d116693fa5946df7bbfd967caf5fac1f529afd541fbd470ddb54ee"), pins)     # the original checkpoint is always accepted
        self.assertEqual(F.size_of(F.DFLASH2_FILE), pins[0][0])                                                   # the size shown is the first source's

    def test_a_mirror_comes_first_and_the_original_repository_is_always_there(self):
        src = F.sources_for(F.DFLASH2_FILE, env={F.DFLASH2_SOURCE_ENV: "https://example.org/d"})
        self.assertEqual(src[0]["base"], "https://example.org/d")
        urls = [u for s in src for u in F.urls_for(F.DFLASH2_FILE, s)]
        self.assertTrue(any("huggingface.co/incoai/Inkling-2-Mini-DFlash2/resolve/4f9088ea66c21f2b99ed657d1941c306b03e84d2/model.safetensors" in u for u in urls))

    def test_the_notice_files_name_the_licences_and_the_credit(self):
        notices = ROOT.parent / "notices"
        apache = (notices / "INKLING-2-MINI-DFLASH2-APACHE-2.0.txt").read_text(encoding="utf-8")
        self.assertIn("Apache License 2.0", apache)
        self.assertIn(F.DFLASH2_COMMIT, apache)
        basalt = (notices / "BASALT-MIT.txt").read_text(encoding="utf-8")
        self.assertIn("MIT License", basalt)
        self.assertIn("tjm8874", basalt)


class Dflash2FetchTest(unittest.TestCase):
    def _fetch(self, have=False, verify_fails=False, fetch_fails=False, own=False, files=()):
        said = []
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            for n in files:
                (folder / n).write_bytes(b"x")
            calls = []

            def fetch(a, name, fld, own_):
                calls.append(name)
                if fetch_fails:
                    raise SystemExit(1)
                return fld / name

            def verify(dst, name, own_):
                if verify_fails:
                    raise SystemExit(1)

            real_fail = setup.fail
            with mock.patch.object(setup, "inkling_fetch", side_effect=fetch), \
                    mock.patch.object(setup, "inkling_have", return_value=have), \
                    mock.patch.object(setup, "inkling_verify", side_effect=verify), \
                    mock.patch.object(setup, "say", side_effect=lambda *x, **k: said.append(" ".join(map(str, x)))), \
                    mock.patch.object(setup, "ok", side_effect=lambda m, *x, **k: said.append(str(m))), \
                    mock.patch.object(setup, "warn", side_effect=lambda m, *x, **k: said.append("WARN " + str(m))):
                got = setup.inkling_fetch_dflash2(mock.Mock(), folder, own)
            self.assertIs(setup.fail, real_fail)                    # the soft failure handler is always taken off again
            left = sorted(p.name for p in folder.iterdir())
        return got.name if got else None, calls, said, left

    def test_a_missing_file_is_fetched_and_its_size_is_shown_first(self):
        got, calls, said, _ = self._fetch()
        self.assertEqual((got, calls), (F.DFLASH2_FILE, [F.DFLASH2_FILE]))
        mb = f"{F.size_of(F.DFLASH2_FILE) / 1e6:.0f} MB"
        self.assertTrue(any(F.DFLASH2_FILE in m and mb in m for m in said), said)

    def test_a_file_that_is_there_and_checks_out_is_kept_without_a_download(self):
        got, calls, said, _ = self._fetch(have=True)
        self.assertEqual((got, calls), (F.DFLASH2_FILE, []))

    def test_a_file_that_does_not_check_out_means_dspark(self):
        got, calls, said, _ = self._fetch(have=True, verify_fails=True)
        self.assertEqual((got, calls), (None, []))
        self.assertTrue(any("DSpark" in m for m in said))

    def test_a_failed_download_means_dspark_and_leaves_nothing_behind(self):
        got, calls, said, left = self._fetch(fetch_fails=True, files=(F.DFLASH2_FILE, F.DFLASH2_FILE + ".part"))
        self.assertEqual((got, calls), (None, [F.DFLASH2_FILE]))
        self.assertEqual(left, [])
        self.assertTrue(any("DSpark is used" in m for m in said))

    def test_the_persons_own_folder_is_never_written_to(self):
        got, calls, said, _ = self._fetch(own=True)
        self.assertEqual((got, calls), (None, []))
        got, calls, said, _ = self._fetch(own=True, have=True)                # but a file the person put there is used
        self.assertEqual((got, calls), (F.DFLASH2_FILE, []))


class Dflash2UpdateTest(unittest.TestCase):
    def _update(self, cfg_extra=None, model_name="Inkling-2-Mini-SQ-20.strata", in_models_folder=True):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            models = td / "models"
            folder = (models / F.FAMILY) if in_models_folder else (td / "elsewhere")
            folder.mkdir(parents=True)
            model = folder / model_name
            model.write_bytes(b"x")
            cfg = {"arch": F.ARCH, "args": ["--native", str(model), "--max-context", "32768"], **(cfg_extra or {})}
            a = mock.Mock(models_dir=str(models), gguf_dir=None)
            seen = []
            with mock.patch.object(setup, "inkling_fetch_dflash2", side_effect=lambda a_, f, own: seen.append((f.name, own)) or None):
                setup.inkling_dflash2_for_config(a, cfg)
        return seen

    def test_an_update_brings_the_sidecar_to_an_existing_sq_install(self):
        self.assertEqual(self._update(), [(F.FAMILY, False)])

    def test_a_model_outside_the_models_folder_counts_as_the_persons_own(self):
        self.assertEqual(self._update(in_models_folder=False), [("elsewhere", True)])

    def test_gguf_builds_and_other_families_are_left_alone(self):
        self.assertEqual(self._update(model_name="Inkling-2-Mini-UD-Q4_K_XL.gguf"), [])
        self.assertEqual(self._update({"arch": "qwen3"}), [])

    def test_update_install_calls_it_for_every_config(self):
        import inspect
        src = inspect.getsource(setup.update_install)
        self.assertIn("inkling_dflash2_for_config(a, cfg)", src)


if __name__ == "__main__":
    unittest.main()
