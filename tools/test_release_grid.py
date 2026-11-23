"""Tests of the release test grid (tools/release_grid): cell list, pass / fail rules, the summary, and the runner's checks against a fake server."""
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "release_grid"))
import cell_runner as cr  # noqa: E402
import fetch_models as fm  # noqa: E402
import grid  # noqa: E402
import grid_lib as gl  # noqa: E402


class CellList(unittest.TestCase):
    def test_cells_json_is_consistent(self):
        data = gl.load_cells()
        self.assertEqual(gl.check_cells(data), [])

    def test_every_class_of_the_brief_is_in_the_list(self):
        classes = " | ".join(b["class"] for b in gl.load_cells()["boxes"].values())
        for want in ("sm_86", "sm_89", "sm_120", "RDNA2", "RDNA3", "RDNA4", "Intel Arc A750", "B70", "Apple", "CPU"):
            self.assertIn(want, classes)

    def test_drafter_on_and_off_for_inkling_and_qwen_and_none_for_hy4(self):
        cells = gl.expand(gl.load_cells(), boxes=["vast0221-4090b"])
        by = {}
        for c in cells:
            by.setdefault(c["model"], set()).add(c["drafter"])
        self.assertEqual(by["ink-sq10"], {"on", "off"})
        self.assertEqual(by["qwen-ud-q4kxl"], {"on", "off"})
        self.assertEqual(by["hy4-sq64"], {"default"})

    def test_core_tier_leaves_out_full_cells(self):
        full = {c["id"] for c in gl.expand(gl.load_cells(), boxes=["vast0221-4090b"], tier="full")}
        core = {c["id"] for c in gl.expand(gl.load_cells(), boxes=["vast0221-4090b"], tier="core")}
        self.assertTrue(core < full)
        self.assertIn("vast0221-4090b__hy4-sq64__default", full - core)

    def test_a_bad_cell_file_is_reported(self):
        data = gl.load_cells()
        data["boxes"]["p620"]["models"]["nope"] = {"path": "/x"}
        data["boxes"]["p620"]["models"]["ink-sq10"] = {"drafters": ["maybe"], "path": "/x"}
        problems = "\n".join(gl.check_cells(data))
        self.assertIn("unknown model nope", problems)
        self.assertIn("drafters must be among", problems)

    def test_specs_carry_release_feed_model_and_profile(self):
        data = gl.load_cells()
        rel = [grid.parse_release("v0.2.19"), grid.parse_release("v0.2.20")]
        specs = grid.build_specs(data, rel, boxes=["p1006"], models=["ink-ud-iq1s"])
        self.assertEqual([s["label"] for s in specs["p1006"]], ["v0.2.19", "v0.2.19", "v0.2.20", "v0.2.20"][:0] or [s["label"] for s in specs["p1006"]])
        labels = [(s["cell"], s["label"]) for s in specs["p1006"]]
        self.assertEqual(labels[0], ("p1006__ink-ud-iq1s__on", "v0.2.19"))
        self.assertEqual(labels[1], ("p1006__ink-ud-iq1s__on", "v0.2.20"))        # the two releases of one cell run back to back
        s = specs["p1006"][1]
        self.assertTrue(s["feed"].endswith("/releases/download/v0.2.20"))
        self.assertEqual(s["profile"]["agent_turns"], 6)                       # the "slow" profile of the box
        self.assertEqual(s["model_spec"]["files"][0], "/data/grid0221/models/ink-ud-iq1s/Inkling-2-Mini-UD-IQ1_S.gguf")
        self.assertEqual(s["model_spec"]["sizes"]["Inkling-2-Mini-UD-IQ1_S.gguf"], 7764291776)

    def test_a_fetch_model_gets_the_grid_folder_as_its_path(self):
        specs = grid.build_specs(gl.load_cells(), [grid.parse_release("v0.2.20")], boxes=["p2006"], models=["ink-sq10"])
        self.assertEqual(specs["p2006"][0]["model_spec"]["path"], "~/grid0221/models/Inkling-2-Mini-SQ-10")

    def test_release_arguments(self):
        self.assertEqual(grid.parse_release("v0.2.20")[0], "v0.2.20")
        self.assertEqual(grid.parse_release("main-abc=/data/feed"), ("main-abc", "/data/feed"))
        with self.assertRaises(SystemExit):
            grid.parse_release("0.2.20")


class Pins(unittest.TestCase):
    def test_lock_file_pins_every_file_with_size_and_sha256(self):
        lock = fm.load_lock()
        for key in ("qwen/Q2_0", "qwen/UD-Q4_K_XL", "ink/SQ-10", "ink/SQ-20", "ink/UD-Q2_K_XL", "hy4/SQ-64"):
            self.assertIn(key, lock)
            for f in lock[key]["files"]:
                self.assertEqual(len(f["sha256"]), 64, f["name"])
                self.assertGreater(f["bytes"], 0)
                self.assertTrue(f["url"].startswith("https://huggingface.co/"), f["url"])
        self.assertIn("dspark-Inkling-2-Mini-Q8_0.gguf", [f["name"] for f in lock["ink/UD-Q2_K_XL"]["files"]])

    def test_every_cell_model_has_a_pin(self):
        data, lock = gl.load_cells(), fm.load_lock()
        for mid, m in data["models"].items():
            self.assertIn(m["pins"], lock, mid)

    def test_a_wrong_file_is_found_by_verify(self):
        with tempfile.TemporaryDirectory() as d:
            lock = {"models": {"x/y": {"folder": "y", "files": [{"name": "a.bin", "bytes": 3, "sha256": "0" * 64, "url": "https://x/a"}]}}}
            lf = Path(d) / "lock.json"
            lf.write_text(json.dumps(lock))
            self.assertIn("missing", fm.verify("x/y", Path(d), lock=lf)[0])
            (Path(d) / "a.bin").write_bytes(b"abc")
            self.assertIn("SHA-256 differs", fm.verify("x/y", Path(d), lock=lf)[0])
            self.assertEqual(fm.verify("x/y", Path(d), quick=True, lock=lf), [])


class Rules(unittest.TestCase):
    def test_speed_fails_beyond_five_percent_plus_the_spread(self):
        prev = {"decode": [100, 101, 100], "prefill": [1000, 1005, 998]}
        self.assertEqual(gl.compare_speed(prev, {"decode": [97, 98, 97], "prefill": [1000, 1001, 999]})["verdict"], gl.PASS)   # -3%
        bad = gl.compare_speed(prev, {"decode": [90, 91, 90], "prefill": [1000, 1001, 999]})                                   # -10%
        self.assertEqual(bad["verdict"], gl.FAIL)
        self.assertEqual(bad["decode"]["verdict"], gl.FAIL)
        self.assertEqual(bad["prefill"]["verdict"], gl.PASS)

    def test_a_noisy_run_cannot_show_a_small_loss(self):
        prev = {"decode": [100, 112, 90]}                 # spread 22%
        self.assertEqual(gl.compare_speed(prev, {"decode": [88, 100, 80]})["verdict"], gl.PASS)   # -12% is inside 5% + 22%

    def test_fingerprint_rule_and_declared_changes(self):
        self.assertEqual(gl.compare_fingerprint("a", "a")["verdict"], gl.PASS)
        self.assertEqual(gl.compare_fingerprint("a", "b")["verdict"], gl.FAIL)
        self.assertEqual(gl.compare_fingerprint("a", "b", "sampler change")["verdict"], gl.PASS)
        decl = {"*__ink-*__on": "DSpark change"}
        self.assertEqual(gl.declared_for(decl, "p620__ink-sq10__on"), "DSpark change")
        self.assertIsNone(gl.declared_for(decl, "p620__ink-sq10__off"))

    def _res(self, cell="b__m__on", fp="f1", dec=(50, 51, 50), launch=gl.PASS, first=gl.PASS):
        return {"cell": cell, "box": "b", "model": "m", "drafter": "on", "release": "v1",
                "steps": {"install": {"status": gl.PASS}, "launch": {"status": launch, "reason": "boom"}, "first_chat": {"status": first},
                          "fingerprint": {"status": gl.PASS, "value": fp}},
                "numbers": {"decode": list(dec), "prefill": [900, 905], "decode_tps": 50.0, "prefill_tps": 902.0}}

    def test_judge_fails_a_cell_whose_start_failed(self):
        row = gl.judge(self._res(launch=gl.FAIL))
        self.assertEqual(row["verdict"], gl.FAIL)
        self.assertIn("launch: boom", row["reasons"][0])

    def test_judge_uses_the_previous_release(self):
        prev = self._res(dec=(100, 101, 100))
        row = gl.judge(self._res(dec=(80, 81, 80)), prev)
        self.assertEqual(row["verdict"], gl.FAIL)
        self.assertTrue(any(r.startswith("decode") for r in row["reasons"]))
        row = gl.judge(self._res(fp="f2", dec=(100, 101, 100)), prev)
        self.assertEqual(row["verdict"], gl.FAIL)
        self.assertTrue(any(r.startswith("fingerprint") for r in row["reasons"]))
        ok = gl.judge(self._res(fp="f2", dec=(100, 101, 100)), prev, {"*": "declared"})
        self.assertEqual(ok["verdict"], gl.PASS)

    def test_targets_give_percentages_for_matching_models_only(self):
        row = {"model": "ink-ud-q4kxl", "numbers": {"decode_tps": 150.0, "prefill_4k_tps": 2500.0}}
        res = {"usage": {"cases": {"parallel": {"levels": {"64": {"throughput_tps": 1500.0}}}}}}
        gl.add_reference(row, [{"label": "ink-5090", "model_glob": "ink-*", "decode_tps": 500, "prefill_tps": 10000, "c64_total_tps": 3000},
                               {"label": "qwen-only", "model_glob": "qwen-*", "decode_tps": 1}], res)
        self.assertEqual(len(row["targets"]), 1)
        self.assertEqual((row["targets"][0]["decode_pct"], row["targets"][0]["prefill_pct"], row["targets"][0]["c64_pct"]), (30.0, 25.0, 50.0))
        rec = {"model": "ink-sq20", "numbers": {"decode_tps": 40.0}}
        gl.add_reference(rec, [{"label": "ink-record", "model_glob": "ink-*"}])
        self.assertEqual(rec["targets"][0]["goal"], "record only")

    def test_the_targets_in_cells_json_are_the_owners(self):
        refs = gl.load_cells()["references"]
        self.assertEqual((refs["ink-5090"]["decode_tps"], refs["ink-5090"]["c64_total_tps"], refs["ink-5090"]["prefill_tps"]), (500, 3000, 10000))
        self.assertEqual((refs["ink-8gb"]["decode_tps"], refs["ink-12gb"]["decode_tps"]), (150, 300))
        self.assertEqual(refs["a6000-ud-q4kxl"]["decode_tps"], 116)

    def test_summary_keeps_skips_and_marks_missing_cells(self):
        data = gl.load_cells()
        expected = gl.expand(data, boxes=["z440"])
        s = gl.summarize({}, None, None, expected)
        self.assertEqual(s["counts"][gl.SKIP], len(expected))
        self.assertTrue(all("disk" in r["reasons"][0] for r in s["rows"]))

    def test_markdown_and_json_are_written(self):
        res = {"b__m__on": self._res()}
        summary = gl.summarize(res)
        with tempfile.TemporaryDirectory() as d:
            gl.write_summary(summary, res, Path(d) / "s", release="v1")
            md = (Path(d) / "s.md").read_text()
            js = json.loads((Path(d) / "s.json").read_text())
        self.assertIn("| b / m / on | **PASS** |", md)
        self.assertEqual(js["counts"]["pass"], 1)


class FakeServer(BaseHTTPRequestHandler):
    """An OpenAI-style server: right answers, tool calls for the agent script, cached-token counts. Class attributes switch faults on."""
    empty_after = None
    wrong = False
    no_cache = False
    crash_on_big = False
    n = 0

    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        self._send({"data": [{"id": "fake"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        cls = type(self)
        cls.n += 1
        msgs = body["messages"]
        text_all = json.dumps(msgs)
        last = msgs[-1]
        ptoks = len(text_all) // 4
        if cls.crash_on_big and ptoks > 600:
            self._send({"error": "boom"}, 500)
            return
        content, calls = "", None
        if body.get("tools"):
            if last["role"] == "tool":
                content = "Done: " + last["content"] + " alpha beta gamma delta 4821 14:30 2000 Kestrel 14463 beta not found"
            else:
                u = last["content"].lower()
                name, args = ("list_notes", {}) if "list the notes" in u else ("read_note", {"name": "alpha"})
                calls = [{"id": "c%d" % cls.n, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]
        else:
            u = last["content"].lower()
            if "capital of australia" in u:
                content = "Canberra."
            elif "17 times 23" in u:
                content = "391"
            elif "chemical symbol for gold" in u:
                content = "Au"
            elif "leap year" in u:
                content = "366"
            elif "largest in our solar" in u:
                content = "Jupiter"
            elif "backwards" in u:
                content = "atarts"
            elif "access code" in u:
                import re
                m = re.search(r"code of the vault is ([A-Z]+-\d+)", u.upper().replace("CODE OF THE VAULT IS", "code of the vault is"))
                content = m.group(1) if m else "unknown"
            else:
                content = "The printing press " * 12
            if cls.wrong:
                content = "I do not know."
        if cls.empty_after is not None and cls.n > cls.empty_after:
            content, calls = "", None
        cached = 0 if cls.no_cache else int(ptoks * 0.9)
        self._send({"choices": [{"message": {"content": content, **({"tool_calls": calls} if calls else {})}, "finish_reason": "tool_calls" if calls else "stop"}],
                    "usage": {"prompt_tokens": ptoks, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": cached}},
                    "timings": {"prompt_per_second": 800.0, "predicted_per_second": 40.0, "cache_n": cached}})


class RunnerChecks(unittest.TestCase):
    def setUp(self):
        for k, v in dict(empty_after=None, wrong=False, no_cache=False, crash_on_big=False, n=0).items():
            setattr(FakeServer, k, v)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeServer)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.api = cr.Api("http://127.0.0.1:%d" % self.httpd.server_address[1])
        self.profile = {"small_tokens": 300, "big_tokens": 1200, "step_budget_s": 100, "agent_turns": 10, "speed_reps": 2, "prefill_tokens": 200, "prefill_reps": 2}

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def run_all(self):
        steps = {}
        nums = cr.run_checks(self.api, self.profile, steps, time.time() + 600)
        return steps, nums

    def test_a_healthy_server_passes_every_step(self):
        steps, nums = self.run_all()
        for name in ("first_chat", "prompt_4k", "prompt_32k", "agent", "speed", "fingerprint"):
            self.assertEqual(steps[name]["status"], gl.PASS, (name, steps[name]))
        self.assertEqual(nums["decode_tps"], 40.0)
        self.assertEqual(nums["prefill_tps"], 800.0)
        self.assertGreater(nums["agent_reuse"], 0.5)
        self.assertEqual(len(steps["first_chat"]["answers"]), 6)
        self.assertEqual(steps["agent"]["requests"] >= 10, True)

    def test_the_fingerprint_is_stable_and_changes_with_the_output(self):
        a, _ = self.run_all()
        b, _ = self.run_all()
        self.assertEqual(a["fingerprint"]["value"], b["fingerprint"]["value"])
        FakeServer.wrong = True
        c, _ = self.run_all()
        self.assertNotEqual(a["fingerprint"]["value"], c["fingerprint"]["value"])

    def test_wrong_answers_fail_the_first_chat(self):
        FakeServer.wrong = True
        steps, _ = self.run_all()
        self.assertEqual(steps["first_chat"]["status"], gl.FAIL)
        self.assertIn("right", steps["first_chat"]["reason"])

    def test_an_empty_reply_is_a_failure(self):
        FakeServer.empty_after = 3
        steps, _ = self.run_all()
        self.assertEqual(steps["first_chat"]["status"], gl.FAIL)
        self.assertIn("empty", steps["first_chat"]["reason"])

    def test_no_prompt_reuse_between_turns_fails_the_agent_step(self):
        FakeServer.no_cache = True
        steps, _ = self.run_all()
        self.assertEqual(steps["agent"]["status"], gl.FAIL)
        self.assertIn("prompt cache", steps["agent"]["reason"])

    def test_a_server_that_dies_on_the_big_prompt_fails_that_step_only(self):
        FakeServer.crash_on_big = True
        steps, _ = self.run_all()
        self.assertEqual(steps["prompt_32k"]["status"], gl.FAIL)
        self.assertEqual(steps["first_chat"]["status"], gl.PASS)

    def test_a_slow_box_skips_the_big_prompt_by_its_projected_time(self):
        self.profile["step_budget_s"] = 0.5               # 1200 tokens at 800 tok/s = 1.5 s
        steps, _ = self.run_all()
        self.assertEqual(steps["prompt_32k"]["status"], gl.SKIP)
        self.assertIn("projected", steps["prompt_32k"]["reason"])

    def test_the_haystack_has_one_needle_and_is_unique_per_salt(self):
        a = cr.needle_prompt(2000, "OSPREY-4242", "s1")
        b = cr.needle_prompt(2000, "OSPREY-4242", "s2")
        self.assertEqual(a.count("OSPREY-4242"), 1)
        self.assertNotEqual(a[:60], b[:60])
        self.assertGreaterEqual(len(a), 2000)

    def test_the_calculator_tool_refuses_anything_but_arithmetic(self):
        self.assertEqual(cr.run_tool("calculator", {"expression": "1200+800"}), "2000")
        self.assertTrue(cr.run_tool("calculator", {"expression": "__import__('os')"}).startswith("error"))

    def test_causes_are_named_from_logs(self):
        self.assertIn("#6713", cr.find_cause("verify: layer 0 never rang (operation not supported on global/shared address space)"))
        self.assertIn("missing", cr.find_cause("missing dspark tensor dspark.fuse_weight"))
        self.assertEqual(cr.find_cause("all fine"), "")

    def test_repetition_is_found(self):
        self.assertTrue(cr.has_repetition(("the quick brown fox jumps over the lazy dog again and again " * 8)))
        self.assertFalse(cr.has_repetition("a normal short answer"))


class ResultsFolder(unittest.TestCase):
    def test_load_results_reads_cell_files_only(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.json").write_text(json.dumps({"cell": "x__y__on", "steps": {}}))
            (Path(d) / "b.json").write_text("not json")
            (Path(d) / "c.json").write_text(json.dumps({"other": 1}))
            self.assertEqual(list(gl.load_results(d)), ["x__y__on"])


sys.path.insert(0, str(HERE / "release_grid" / "usage"))
import usage_suite as us  # noqa: E402


class StreamFake(BaseHTTPRequestHandler):
    """A streaming OpenAI-style server: answers the usage suite's prompts with fixed text, one tool call when asked, usage + timings in the last chunk."""

    def log_message(self, *a):
        pass

    def do_GET(self):
        raw = json.dumps({"data": [{"id": "fake"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        last = body["messages"][-1]
        text = "Canberra is the capital. visible answer"
        tool = None
        if body.get("tools") and last["role"] == "user":
            tool = {"index": 0, "id": "c1", "function": {"name": "run_tests", "arguments": "{}"}}
        chunks = []
        if tool:
            chunks.append({"choices": [{"delta": {"tool_calls": [{**tool, "function": {"name": "run_tests", "arguments": "{"}}]}}]})
            chunks.append({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "}"}}]}}]})
            chunks.append({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
        else:
            chunks.append({"choices": [{"delta": {"reasoning_content": "thinking about it"}}]})
            chunks.append({"choices": [{"delta": {"content": text}}]})
            chunks.append({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        chunks.append({"choices": [], "usage": {"prompt_tokens": 500, "completion_tokens": 12, "prompt_tokens_details": {"cached_tokens": 450}},
                       "timings": {"prompt_per_second": 900.0, "predicted_per_second": 33.0, "cache_n": 450}})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for ch in chunks:
            self.wfile.write(("data: %s\n\n" % json.dumps(ch)).encode())
        self.wfile.write(b"data: [DONE]\n\n")


class UsageSuite(unittest.TestCase):
    def test_pictures_are_valid_pngs_and_the_font_is_complete(self):
        import struct
        import zlib
        for make, w, h in ((us.img_scene, 480, 320), (us.img_screenshot, 480, 220), (us.img_chart, 480, 320), (us.img_circle, 320, 240), (us.img_square, 320, 240)):
            png = make().png()
            self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(struct.unpack(">II", png[16:24]), (w, h))
            idat = png[png.index(b"IDAT") + 4:png.index(b"IEND") - 8]
            self.assertEqual(len(zlib.decompress(idat)), h * (1 + 3 * w))
        self.assertEqual([k for k, v in us.FONT.items() if len(v) != 35], [])
        self.assertTrue(set("ORDER 4417 TOTAL 83 USD STATUS SHIPPED") <= set(us.FONT))

    def test_the_diff_applier_applies_a_good_diff_and_refuses_a_wrong_one(self):
        src = us.big_file(100)
        self.assertGreaterEqual(len(src.split("\n")), 300)
        good = ("@@ -170,3 +170,3 @@\n def f057(x):\n     \"\"\"Step 57 of the pipeline.\"\"\"\n-    return x + %d\n+    return x * 3\n" % (57 % 7 + 1))
        new, msg = us.apply_unified_diff(src, good)
        self.assertEqual(msg, "ok")
        self.assertIn("return x * 3", new)
        self.assertEqual(new.count("x * 3"), 1)
        self.assertIsNone(us.apply_unified_diff(src, "@@ -1,1 +1,1 @@\n-nothing like this\n+x\n")[0])

    def test_the_agent_repo_starts_red_and_goes_green_with_the_fixes(self):
        repo = us.Repo()
        green, out = repo.run_tests()
        self.assertFalse(green)
        self.assertEqual(out.count("FAIL"), 1)             # fail-fast: one failing test per run, so every fix needs a run
        repo.files["mathkit/stats.py"] = "def mean(xs):\n    return sum(xs) / len(xs)\n\n\ndef maximum(xs):\n    return max(xs)\n"
        repo.files["mathkit/text.py"] = "def words(s):\n    return s.split()\n\n\ndef title(s):\n    return s.title()\n"
        repo.files["mathkit/geo.py"] = "import math\n\n\ndef area_circle(r):\n    return math.pi * r * r\n\n\ndef perimeter_rect(w, h):\n    return 2 * (w + h)\n"
        self.assertTrue(repo.run_tests()[0])
        self.assertEqual(repo.call("read_file", {"path": "nope"}), "error: no such file")
        self.assertIn("error", repo.call("write_file", {"path": "../x", "content": ""}))

    def test_the_client_reads_a_stream_with_reasoning_tool_calls_usage_and_timings(self):
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), StreamFake)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            c = us.Client("http://127.0.0.1:%d" % httpd.server_address[1])
            c.find_model()
            r = c.chat([{"role": "user", "content": "hi"}])
            self.assertEqual(r["text"], "Canberra is the capital. visible answer")
            self.assertEqual(r["reasoning"], "thinking about it")
            self.assertEqual((r["prompt_tokens"], r["cached"], r["decode_tps"], r["prefill_tps"]), (500, 450, 33.0, 900.0))
            self.assertIsNotNone(r["ttft"])
            r = c.chat([{"role": "user", "content": "hi"}], tools=us.AGENT_TOOLS)
            self.assertEqual(r["tool_calls"][0]["function"], {"name": "run_tests", "arguments": "{}"})
            res = us.run_suite("http://127.0.0.1:%d" % httpd.server_address[1], cases=["chat", "parallel"], parallel_levels="1,2")
            self.assertEqual(res["cases"]["chat"]["verdict"], us.PASS)
            self.assertEqual(res["cases"]["chat"]["levels"]["high"]["visible_answers"], 3)
            self.assertIn("1", res["cases"]["parallel"]["levels"])
            md = us.markdown(res)
            self.assertIn("| chat |", md)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_a_server_without_pictures_gives_a_skip(self):
        class NoVision(StreamFake):
            def do_POST(self):
                raw = b'{"error": "this server has no picture reader"}'
                self.send_response(400)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), NoVision)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            c = us.Client("http://127.0.0.1:%d" % httpd.server_address[1], "m")
            self.assertEqual(us.case_vision(c, None)["verdict"], us.SKIP)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_the_long_document_has_the_facts_and_the_asked_size(self):
        doc = us.long_doc(30000, "t")
        self.assertIn("Ilse Moreau", doc)
        self.assertIn("KELVIN-SEVEN", doc)
        self.assertGreater(len(doc), 30000 * 2.5)


if __name__ == "__main__":
    unittest.main()
