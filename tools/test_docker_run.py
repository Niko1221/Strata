"""docker/run.sh names its mode and never guesses it from the shape of an argument.

Runs the real script with stub `docker` and `curl` programs first on PATH, so what is asserted is
the command line run.sh hands to Docker - no daemon, no image and no GPU are needed.  The stubs
also record every call, which is how "the model is forwarded as --native" and "serve forwards the
STRATA_* switches" are checked.

    python -m unittest tools.test_docker_run
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_SH = ROOT / "docker" / "run.sh"
BASH = shutil.which("bash")

# `docker`: record the call, then answer only what run.sh asks about (`info`, `image inspect`, the
# two `ps` forms).  `ps -a` (the leftover-container check) answers nothing, `ps` (the health-check
# loop) names the container, which is what makes the loop stop instead of "the container exited".
DOCKER_STUB = """#!/bin/sh
printf '%s\\n' "$*" >> "$DOCKER_LOG"
case "$1 $2" in
  "image inspect") exit 0 ;;
  "ps -a") exit 0 ;;
esac
case "$1" in
  info) exit 0 ;;
  ps) [ -n "$STUB_CONTAINER" ] && echo "$STUB_CONTAINER"; exit 0 ;;
  *) exit 0 ;;
esac
"""

# `curl`: the health probe, the model list and (for SMOKE=1) one chat answer.
CURL_STUB = """#!/bin/sh
for a in "$@"; do case "$a" in http*) url="$a" ;; esac; done
case "$url" in
  */health) exit 0 ;;
  */v1/models) echo '{"data":[{"id":"strata"}]}'; exit 0 ;;
  */v1/chat/completions)
    echo '{"choices":[{"finish_reason":"stop","message":{"content":"hello world"}}],"usage":{"total_tokens":3}}'
    exit 0 ;;
  *) exit 0 ;;
esac
"""


@unittest.skipIf(os.name == "nt" or BASH is None, "needs bash and POSIX stubs")
class RunSh(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        self.log = self.tmp / "docker.log"
        for name, body in (("docker", DOCKER_STUB), ("curl", CURL_STUB)):
            p = self.bin / name
            p.write_text(body, encoding="utf-8")
            p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        self.model = self.tmp / "model.gguf"
        self.model.write_bytes(b"GGUF")   # run.sh resolves and mounts it; nothing reads it

    def run_script(self, *args, **env):
        e = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}",
                 DOCKER_LOG=str(self.log), MODEL_FILE=str(self.model), STUB_CONTAINER="strata")
        for k, v in env.items():
            if v is None:
                e.pop(k, None)
            else:
                e[k] = v
        return subprocess.run([BASH, str(RUN_SH), *args], env=e, capture_output=True, text=True,
                              timeout=60)

    def calls(self) -> str:
        return self.log.read_text(encoding="utf-8") if self.log.exists() else ""

    def test_help_is_this_scripts_own_help(self):
        r = self.run_script("--help")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("docker/run.sh serve", r.stdout)
        self.assertNotIn("strata generate", r.stdout)
        self.assertEqual(self.calls(), "", "help must not touch Docker")

    def test_serve_help_is_help_too(self):
        r = self.run_script("serve", "--help")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("docker/run.sh serve", r.stdout)
        self.assertEqual(self.calls(), "")

    def test_unknown_mode_is_refused(self):
        r = self.run_script("frobnicate")
        self.assertEqual(r.returncode, 1)
        self.assertIn("unknown mode 'frobnicate'", r.stderr)
        self.assertEqual(self.calls(), "")

    def test_serve_rejects_arguments_it_cannot_pass_on(self):
        r = self.run_script("serve", "--port", "9000")
        self.assertEqual(r.returncode, 1)
        self.assertIn("serve takes no arguments", r.stderr)
        self.assertEqual(self.calls(), "", "nothing may be started when the call was refused")

    def test_no_model_is_an_error_that_names_the_variable(self):
        r = self.run_script("serve", MODEL_FILE="")
        self.assertEqual(r.returncode, 1)
        self.assertIn("MODEL_FILE", r.stderr)

    def test_a_model_that_does_not_exist_is_refused(self):
        r = self.run_script("serve", MODEL_FILE=str(self.tmp / "nope.gguf"))
        self.assertEqual(r.returncode, 1)
        self.assertIn("does not exist", r.stderr)

    def test_engine_help_needs_no_model_and_no_gpu(self):
        r = self.run_script("engine", "--help", MODEL_FILE="")
        self.assertEqual(r.returncode, 0, r.stderr)
        call = self.calls()
        self.assertIn("--entrypoint /opt/strata/engine/strata", call)
        self.assertTrue(call.rstrip().endswith("--help"), call)
        self.assertNotIn(" -v ", call, "no model is mounted for a usage question")

    def test_engine_forwards_the_arguments_and_names_the_mounted_model(self):
        r = self.run_script("engine", "--bench-pp", "512", "--bench-tg", "8")
        self.assertEqual(r.returncode, 0, r.stderr)
        call = self.calls().rstrip()
        self.assertIn("--entrypoint /opt/strata/engine/strata", call)
        self.assertTrue(call.endswith("--native /models/model.gguf --bench-pp 512 --bench-tg 8"), call)

    def test_engine_keeps_a_model_the_caller_named(self):
        r = self.run_script("engine", "--native", "/other/m.gguf", "--bench-tg", "8")
        self.assertEqual(r.returncode, 0, r.stderr)
        call = self.calls().rstrip()
        self.assertIn("--native /other/m.gguf --bench-tg 8", call)
        self.assertNotIn("--native /models/model.gguf", call)

    def test_serve_starts_the_container_and_reports_what_it_serves(self):
        r = self.run_script("serve", API_KEY="secret")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("models: strata", r.stdout)
        self.assertIn("done: strata is serving.", r.stdout)
        call = self.calls()
        self.assertIn("--gpus all", call)
        self.assertIn("-p 0.0.0.0:11434:11434", call)
        self.assertIn("-e PORT=11434", call)
        self.assertIn(f"{self.model}:/models/model.gguf:ro", call)

    def test_serve_forwards_the_engine_switches(self):
        r = self.run_script("serve", STRATA_MMQ_KQ="1", API_KEY="secret")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("-e STRATA_MMQ_KQ=1", self.calls())

    def test_exposing_the_port_needs_a_key(self):
        r = self.run_script("serve", BIND_HOST="0.0.0.0")
        self.assertEqual(r.returncode, 1)
        self.assertIn("set API_KEY first", r.stderr)
        started = [c for c in self.calls().splitlines() if c.startswith("run ")]
        self.assertEqual(started, [], "the call was refused, so no container may be started")

    def test_the_end_to_end_check_is_opt_in(self):
        r = self.run_script("serve", API_KEY="secret")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("/v1/chat/completions", r.stdout)

        self.log.unlink()
        r = self.run_script("serve", SMOKE="1", API_KEY="secret")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("request: POST /v1/chat/completions", r.stdout)
        self.assertIn("answer: 'hello world'", r.stdout)


if __name__ == "__main__":
    unittest.main()
