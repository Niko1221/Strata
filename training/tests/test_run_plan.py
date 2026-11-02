"""`run_plan.py` 的单测：两步最关键的行为——**挂死续跑**与**失败不连坐**。

用 `python -c` 当"步骤"，靠退出码模拟各种结局：
  0 = 成功、3 = 挂死（bench 的约定）、2 = 普通失败。
重试次数写在临时文件里，第二次读到就不再返回 3 —— 这是对 `--resume` 的最小可信模拟。
"""

from __future__ import annotations

# --- path bootstrap: this file sits under training/<sub>/; shared modules are one level up ---
import pathlib as _p, sys as _s
for _d in (_p.Path(__file__).resolve().parent,
           _p.Path(__file__).resolve().parents[1],
           _p.Path(__file__).resolve().parents[1] / "backward",
           _p.Path(__file__).resolve().parents[1] / "bench",
           _p.Path(__file__).resolve().parents[1] / "mezo"):
    if str(_d) not in _s.path:
        _s.path.insert(0, str(_d))


import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_plan as R  # noqa: E402

PY = json.dumps(sys.executable)


def _step(name, code=0, **kw):
    s = {"name": name, "cwd": HERE,
         "cmd": [sys.executable, "-c", f"import sys; sys.exit({code})"]}
    s.update(kw)
    return s


class RunOne(unittest.TestCase):
    def setUp(self):
        self.logdir = tempfile.mkdtemp()

    def test_success(self):
        r = R.run_one(_step("ok"), self.logdir)
        self.assertEqual(r["exit"], 0)
        self.assertEqual(r["attempts"], [0])
        self.assertTrue(os.path.isfile(r["log"]))

    def test_failure_without_retry_rule_stops(self):
        r = R.run_one(_step("bad", 2), self.logdir)
        self.assertEqual(r["exit"], 2)
        self.assertEqual(r["attempts"], [2])

    def test_hang_exit_3_is_retried_with_append(self):
        """退出码 3（bench 挂死）⇒ 追加 --resume 重跑一次，之后成功。"""
        marker = os.path.join(self.logdir, "tried")
        code = (
            "import os, sys\n"
            f"m = {marker!r}\n"
            "if os.path.exists(m): sys.exit(0)\n"
            "open(m, 'w').write('1')\n"
            "sys.exit(3)\n"
        )
        step = {"name": "bench", "cwd": HERE, "cmd": [sys.executable, "-c", code],
                "append": ["--resume"],
                "retry_on": {"3": {"times": 3}}}
        r = R.run_one(step, self.logdir)
        self.assertEqual(r["attempts"], [3, 0])
        self.assertEqual(r["exit"], 0)

    def test_retries_are_bounded(self):
        step = _step("always3", 3, retry_on={"3": {"times": 2}})
        r = R.run_one(step, self.logdir)
        self.assertEqual(r["attempts"], [3, 3, 3])       # 1 次 + 2 次重试
        self.assertEqual(r["exit"], 3)

    def test_append_is_not_used_on_first_attempt(self):
        marker = os.path.join(self.logdir, "args")
        code = ("import sys; open(%r, 'w').write(' '.join(sys.argv[1:]))" % marker)
        step = {"name": "args", "cwd": HERE, "cmd": [sys.executable, "-c", code, "base"],
                "append": ["--resume"], "retry_on": {"9": {"times": 1}}}
        R.run_one(step, self.logdir)
        with open(marker, encoding="utf-8") as f:
            self.assertEqual(f.read(), "base")

    def test_timeout_is_not_zero(self):
        step = {"name": "sleepy", "cwd": HERE, "timeout": 1,
                "cmd": [sys.executable, "-c", "import time; time.sleep(30)"]}
        r = R.run_one(step, self.logdir)
        self.assertEqual(r["exit"], "timeout")


class Expand(unittest.TestCase):
    def test_vars_in_list_and_str(self):
        os.environ["MY_PACK"] = "/x/pack"
        self.addCleanup(os.environ.pop, "MY_PACK", None)
        got = R.expand(["python", "x.py", "--pack", "$MY_PACK"], os.environ)
        self.assertEqual(got[-1], "/x/pack")
        self.assertEqual(R.expand("--gguf ${MY_PACK}/a.gguf", os.environ), "--gguf /x/pack/a.gguf")


class Cli(unittest.TestCase):
    def _plan(self, steps):
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        json.dump({"steps": steps}, f, ensure_ascii=False)
        f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def _run(self, steps, extra=()):
        d = tempfile.mkdtemp()
        rep = os.path.join(d, "report.json")
        argv = [self._plan(steps), "--report", rep, "--logdir", os.path.join(d, "logs")] + list(extra)
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        old = sys.argv
        sys.argv = ["run_plan.py"] + argv
        try:
            with redirect_stdout(buf):
                code = R.main()
        finally:
            sys.argv = old
        if not os.path.exists(rep):          # --dry-run 只打印，不写报告
            return code, None, buf.getvalue()
        with open(rep, encoding="utf-8") as f:
            return code, json.load(f), buf.getvalue()

    def test_stop_on_fail_skips_the_rest(self):
        code, rep, _ = self._run([_step("a"), _step("b", 2), _step("c")])
        self.assertEqual(code, 1)
        self.assertEqual(rep["failed"], ["b"])
        self.assertEqual(rep["skipped"], ["c"])          # c 没跑，但要如实记下"没跑"

    def test_continue_on_fail(self):
        code, rep, _ = self._run([_step("a", 2, on_fail="continue"), _step("b")])
        self.assertEqual(rep["failed"], ["a"])
        self.assertEqual(len(rep["steps"]), 2)
        self.assertEqual(rep["steps"][1]["exit"], 0)

    def test_only_and_dry_run(self):
        steps = [_step("a"), _step("b")]
        code, _, out = self._run(steps, ["--only", "b", "--dry-run"])
        self.assertEqual(code, 0)
        self.assertIn("[b]", out)
        self.assertNotIn("[a]", out)

    def test_unknown_only_is_fatal(self):
        with self.assertRaises(SystemExit):
            self._run([_step("a")], ["--only", "zzz"])


if __name__ == "__main__":
    unittest.main()
