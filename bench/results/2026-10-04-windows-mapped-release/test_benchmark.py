"""Protocol and evidence checks against a local fake server; no model or GPU work."""
import importlib.util
import hashlib
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("local_benchmark", HERE / "benchmark.py")
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


class FakeServer(ThreadingHTTPServer):
    def __init__(self):
        super().__init__(("127.0.0.1", 0), FakeHandler)
        self.mode = "ok"
        self.metrics = {"engine": {"max_context": 32768}, "live": {"state": "idle", "queued": 0},
                        "totals": {"requests": 0}, "requests": [], "time": 1,
                        "hardware": {"ram_total": 32 * 2**30, "ram_used": 10 * 2**30,
                                     "gpus": [{"index": 0, "mem_used": 100}, {"index": 1, "mem_used": 200}],
                                     "disk_read_mb": 3.5}}


class FakeHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps(self.server.metrics).encode("utf-8"))

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        if self.server.mode == "http_error":
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "Synthetic request rejection"}')
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def send(value):
            self.wfile.write(b"data: " + json.dumps(value).encode("utf-8") + b"\n\n")
            self.wfile.flush()

        send({"choices": [{"delta": {"content": "Verified output."}, "finish_reason": None}]})
        if self.server.mode == "stream_error":
            send({"error": {"message": "Synthetic engine failure"}})
            return
        reused = 1 if self.server.mode == "reused" else 0
        engine = {"prompt_tokens": 100, "prompt_total": 100, "prompt_read": 100 - reused,
                  "reused": reused, "prompt_ms": 100, "decode_ms": 200,
                  "output_tokens": 8, "engine_generated": 8, "finish": "length",
                  "file_mb": 12.5, "ram_blobs": 3, "file_blobs": 9}
        self.server.metrics["totals"]["requests"] += 2 if self.server.mode == "other_client" else 1
        self.server.metrics["requests"].insert(0, engine)
        send({"choices": [{"delta": {}, "finish_reason": "length"}],
              "usage": {"prompt_tokens": 100, "completion_tokens": 8,
                        "prompt_tokens_details": {"cached_tokens": reused}},
              "timings": {"prompt_n": 100 - reused, "predicted_n": 8,
                          "prompt_ms": 100, "predicted_ms": 200}})
        if self.server.mode != "missing_done":
            self.wfile.write(b"data: [DONE]\n\n")


class BenchmarkProtocolTest(unittest.TestCase):
    def setUp(self):
        self.server = FakeServer()
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.temporary = tempfile.TemporaryDirectory(prefix="test-benchmark-", dir=HERE)
        self.out = Path(self.temporary.name).resolve()
        self.assertEqual(self.out.parent, HERE)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temporary.cleanup()

    def run_trial(self, mode):
        self.server.mode = mode
        url = f"http://127.0.0.1:{self.server.server_port}"
        return BENCH.perform(url, self.out, "fake-session", mode, 1,
                             {"max_tokens": 8}, 100, interval=0.01, timeout=3)

    def test_success_has_real_rates_and_both_gpus(self):
        row = self.run_trial("ok")
        self.assertEqual(row["status"], "ok", row.get("error"))
        self.assertEqual(row["prefill_tok_s"], 1000)
        self.assertEqual(row["decode_tok_s"], 40)
        self.assertEqual(row["engine"]["file_mb"], 12.5)
        samples = [json.loads(line) for line in (self.out / row["telemetry"]).read_text(encoding="utf-8").splitlines()]
        self.assertEqual(samples[0]["hardware"]["ram_available"], 22 * 2**30)
        self.assertEqual(len(samples[0]["hardware"]["gpus"]), 2)

    def test_rejects_cached_prompt_missing_done_and_other_client(self):
        for mode, message in (("reused", "cached tokens"), ("missing_done", "without [DONE]"),
                              ("other_client", "counter mismatch")):
            with self.subTest(mode=mode):
                row = self.run_trial(mode)
                self.assertEqual(row["status"], "failed")
                self.assertIn(message, row["error"]["message"])
                self.assertTrue((self.out / f"{mode}-attempt-1-record.json").is_file())

    def test_stream_error_retains_partial_text_and_error_payload(self):
        row = self.run_trial("stream_error")
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["text"], "Verified output.")
        self.assertIn("Synthetic engine failure", row["error"]["message"])
        self.assertIn("Synthetic engine failure", (self.out / row["raw_stream"]).read_text(encoding="utf-8"))

    def test_http_error_body_is_preserved(self):
        row = self.run_trial("http_error")
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["error"]["http_status"], 400)
        self.assertIn("Synthetic request rejection", row["error"]["body"])

    def test_summary_reports_failed_attempts_separately(self):
        good = {"label": "tokens-1024-run-1", "status": "ok", "client_ttft_s": 1,
                "client_elapsed_s": 2, "prefill_tok_s": 1024, "decode_tok_s": 256}
        failed = {"label": "tokens-1024-run-1", "status": "failed"}
        group = BENCH.summary([failed, good], [1024])["1024"]
        self.assertEqual(group["successful_runs"], 1)
        self.assertEqual(group["failed_attempts"], 1)
        self.assertEqual(group["decode_tok_s"]["median"], 256)

    def test_executable_hash_uses_config_working_directory(self):
        payload = b"Synthetic binary contents; never executed."
        executable = self.out / "strata.exe"
        executable.write_bytes(payload)
        for name in ("strata.exe", str(executable)):
            with self.subTest(name=name):
                record = BENCH.executable_record({"exe": name, "cwd": str(self.out)})
                self.assertEqual(record["path"], str(executable))
                self.assertEqual(record["sha256"], hashlib.sha256(payload).hexdigest())
                self.assertEqual(record["size_bytes"], len(payload))
        with self.assertRaises(FileNotFoundError):
            BENCH.executable_record({"exe": "missing.exe", "cwd": str(self.out)})

    def test_provenance_supports_artifact_schemas_and_rejects_wrong_binary(self):
        executable = {"path": str(self.out / "strata.exe"), "sha256": "a" * 64}
        artifact = {"sha256": executable["sha256"], "bytes": 123}
        path = self.out / "provenance.json"
        for artifacts in ({"strata.exe": artifact}, [{"file": "strata.exe", **artifact}]):
            source = {"strata_version": "0.1.38", "commit": "fork-commit",
                      "engine_source_fingerprint": "engine-fingerprint", "artifacts": artifacts,
                      "source_file_hashes": [{"path": "large-file-list", "sha256": "omitted"}],
                      "toolchain": {"nvcc": "13.1.80", "cuda_architectures": [86, 89]}}
            BENCH.write_json(path, source)
            record = BENCH.engine_provenance_record(path, executable)
            self.assertEqual(record["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(record["version"], "0.1.38")
            self.assertEqual(record["commit"], "fork-commit")
            self.assertEqual(record["fingerprint"], "engine-fingerprint")
            self.assertTrue(record["executable_hash_verified"])
            self.assertNotIn("source_file_hashes", record)
            with self.assertRaisesRegex(ValueError, "does not match"):
                BENCH.engine_provenance_record(path, {**executable, "sha256": "b" * 64})

    def test_release_build_metadata_does_not_invent_commit_or_hash_verification(self):
        path = self.out / "BUILD.json"
        BENCH.write_json(path, {"version": "0.1.39", "source": "release", "cuda": "13.0", "archs": [86, 89]})
        record = BENCH.engine_provenance_record(path, {"path": str(self.out / "strata.exe"), "sha256": "a" * 64})
        self.assertEqual(record["version"], "0.1.39")
        self.assertEqual(record["toolchain"]["cuda"], "13.0")
        self.assertIsNone(record["commit"])
        self.assertIsNone(record["fingerprint"])
        self.assertFalse(record["executable_hash_verified"])


if __name__ == "__main__":
    unittest.main()
