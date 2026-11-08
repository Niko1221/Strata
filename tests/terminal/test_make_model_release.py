"""packaging/make_model_release.py: the model-release plan is a dry run, case A leaves the engines alone, case B rebuilds them, and the
steps that need the release key only print a placeholder.

    python -m unittest tests.terminal.test_make_model_release
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "packaging"), str(ROOT / "tools"), str(ROOT)]
import make_model_release as M  # noqa: E402
import test_qwen4_intake as T  # noqa: E402
import qwen4_intake as Q  # noqa: E402

PREV = {"strata-engine-0.1.57-linux-x86_64-cuda-cuda13.tar.gz": 720_000_000, "strata-engine-0.1.57-linux-x86_64-sycl.tar.gz": 187_000_000,
        "strata-engine-0.1.57-windows-x86_64-cuda-cuda13.zip": 900_000_000, "strata-engine-0.1.57-macos-arm64-metal.tar.gz": 150_000_000,
        "strata-dsv41-0.1.57-linux-x86_64-cuda-cuda13.tar.gz": 445_000_000, "strata-control-0.1.57-linux-x86_64.tar.gz": 99_000_000,
        "Strata-0.2.11-windows-x64-setup.exe": 80_000_000, "Strata-0.2.11-windows-x64-setup.exe.sig": 400,
        "Strata-0.2.11-linux-x86_64.AppImage": 90_000_000, "Strata-0.2.11-macos-arm64.dmg": 160_000_000, "Strata.app.tar.gz": 70_000_000,
        "strata-feed.json": 9000, "strata-feed.json.sig": 200, "SHA256SUMS.txt": 5000, "latest.json": 1500, "install.sh": 9000}


def make_prev(d: Path) -> Path:
    d.mkdir()
    for n, size in PREV.items():
        with (d / n).open("wb") as f:
            f.truncate(size)
    return d


def report(name: str) -> dict:
    return Q.run([Q.read_local(p) for p in T.files_for(name)], build_tokenizer=False)


def run(argv) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = M.main(argv)
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
            if isinstance(e.code, str):
                err.write(e.code)
    return code, out.getvalue(), err.getvalue()


class Plan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.prev = make_prev(Path(cls.tmp.name) / "prev")
        cls.a = Path(cls.tmp.name) / "a.json"
        cls.a.write_text(json.dumps(report("q36_ud_q4_k_xl")))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def base(self, rep, *extra):
        return ["--intake", str(rep), "--id", "m1", "--app-version", "0.2.12", "--sequence", "122", "--prev-upload", str(self.prev), *extra]

    def test_case_a_is_feed_and_app_only(self):
        code, out, err = run(self.base(self.a, "--json", "--revision", "abc", "--hf-repo", "x/y"))
        self.assertEqual(code, 0, err)
        p = json.loads(out)
        self.assertEqual(p["case"], "A")
        names = {c["name"] for c in p["changed"]}
        self.assertFalse(any(n.startswith(("strata-engine-", "strata-dsv41-", "strata-control-")) for n in names))
        self.assertTrue({"strata-feed.json", "strata-feed.json.sig", "SHA256SUMS.txt", "latest.json", "Strata-0.2.12-windows-x64-setup.exe"} <= names)
        kept = {k["name"] for k in p["kept"]}
        self.assertEqual(kept, {n for n in PREV if n.startswith(("strata-engine-", "strata-dsv41-", "strata-control-"))})
        self.assertFalse(any("Build the Qwen engine" in s["what"] for s in p["steps"]))
        self.assertEqual(p["entry"]["family"], "qwen35moe")
        self.assertEqual(p["entry"]["hf"], {"repo": "x/y", "revision": "abc"})
        self.assertIs(p["entry"]["pictures"], False)               # no picture reader next to this set

    def test_case_b_rebuilds_engines_and_controls_at_the_new_version(self):
        code, out, err = run(self.base(self.a, "--json", "--case", "B", "--engine-version", "0.1.58"))
        self.assertEqual(code, 0, err)
        p = json.loads(out)
        names = {c["name"] for c in p["changed"]}
        self.assertIn("strata-engine-0.1.58-linux-x86_64-sycl.tar.gz", names)
        self.assertIn("strata-engine-0.1.58-windows-x86_64-cuda-cuda13.zip", names)
        self.assertIn("strata-control-0.1.58-linux-x86_64.tar.gz", names)
        self.assertFalse(any("0.1.57" in n for n in names))
        self.assertIn("strata-dsv41-0.1.57-linux-x86_64-cuda-cuda13.tar.gz", {k["name"] for k in p["kept"]})
        self.assertTrue(any("Mac" in s["what"] and s["owner_only"] for s in p["steps"]))
        self.assertTrue(p["upload_bytes"] > 1_800_000_000)
        code, _, err = run(self.base(self.a, "--case", "B"))
        self.assertNotEqual(code, 0)                                # case B needs an engine version

    def test_case_c_is_refused(self):
        p = Path(self.tmp.name) / "c.json"
        p.write_text(json.dumps(report("ds41_ud_q8_k_xl")))
        code, out, err = run(self.base(p))
        self.assertNotEqual(code, 0)
        self.assertIn("case C", err)
        self.assertEqual(out, "")

    def test_it_is_a_dry_run_and_never_touches_a_key(self):
        code, out, err = run(self.base(self.a, "--execute"))
        self.assertEqual(code, 2)
        self.assertIn("REFUSED", out)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            M.main(self.base(self.a, "--key", "k.json"))             # there is no such option
        code, out, err = run(self.base(self.a))
        self.assertEqual(code, 0, err)
        key_lines = [l for l in out.splitlines() if "--key" in l or "sign-package" in l]
        self.assertTrue(key_lines)
        for l in key_lines:
            self.assertIn("<FEED-KEY>", l)
        src = Path(M.__file__).read_text(encoding="utf-8")
        for bad in ("subprocess", "os.system", "urllib", "requests", "socket", "shutil.copy"):
            self.assertNotIn(bad, src)

    def test_write_makes_the_entry_and_fills_the_templates(self):
        t = Path(self.tmp.name) / "tpl"
        t.mkdir()
        (t / "RELEASE-NOTES.md").write_text("# {{model_title}} {{app_version}}\n{{file_rows}}\n[FILL: a number] {{unknown}}\n", encoding="utf-8")
        out = Path(self.tmp.name) / "out"
        code, _, err = run(self.base(self.a, "--write", str(out), "--templates", str(t), "--revision", "abc"))
        self.assertEqual(code, 0, err)
        entry = json.loads((out / "models" / "m1.model.json").read_text())
        self.assertIn("DRAFT", entry["_status"])
        self.assertEqual(entry["files"][0]["name"], "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")
        self.assertEqual(entry["drafter"], "mtp")
        self.assertEqual(entry["min_engine"], "[FILL: minimum engine version]")
        notes = (out / "RELEASE-NOTES.md").read_text()
        self.assertIn("# Qwen3.6-35B-A3B 0.2.12", notes)
        self.assertIn("[FILL: a number]", notes)
        self.assertIn("{{unknown}}", notes)                              # a key it does not know stays visible
        self.assertTrue((out / "PLAN.md").is_file() and (out / "plan.json").is_file())
        code, _, err = run(self.base(self.a, "--write", str(out)))      # not an empty folder
        self.assertNotEqual(code, 0)

    def test_the_templates_in_phaseA_tests_have_what_the_script_fills(self):
        d = Path(r"C:\Users\AI-Server\Desktop\Strata\phaseA-tests\qwen4\templates")
        if not d.is_dir():
            self.skipTest("no templates folder")
        out = Path(self.tmp.name) / "out2"
        code, _, err = run(self.base(self.a, "--write", str(out), "--templates", str(d), "--revision", "abc", "--hf-repo", "x/y"))
        self.assertEqual(code, 0, err)
        for name in ("RELEASE-NOTES.md", "MODEL-CARD.md", "CHECKLIST.md"):
            text = (out / name).read_text(encoding="utf-8")
            self.assertIn("DRAFT", text.splitlines()[0])
            self.assertNotIn("{{", text, name)                          # every key the templates use is one the script fills
        notes = (out / "RELEASE-NOTES.md").read_text(encoding="utf-8")
        self.assertTrue(notes.rstrip().endswith('height="50"></a>'))      # the coffee button is last
        self.assertIn("[FILL", notes)


if __name__ == "__main__":
    unittest.main()
