"""pytest for tools/iq_eval: extractors, the HumanEval runner, compare.py and run.py (against a fake HTTP server; no engine, no GPU).

    D:\\works\\projects\\experiments\\exstrata\\.venv\\Scripts\\python.exe -m pytest terminal/tools/iq_eval/test_iq_eval.py
"""
import importlib.util
import json
import random
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(f"iq_{name}", HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


score, compare, run, subsets = _load("score"), _load("compare"), _load("run"), _load("subsets")


# ---- extractors ----

def test_last_boxed_nested_and_truncated():
    assert score.last_boxed(r"so \boxed{1} no, \boxed{\frac{3}{4}}.") == r"\frac{3}{4}"
    assert score.last_boxed(r"\boxed{\frac{3}{4") is None
    assert score.last_boxed("no answer") is None


def test_gsm_number():
    assert score.gsm_equiv(score.extract_number(r"... \boxed{\$1,234.50}"), "1234.5")
    assert score.gsm_equiv(score.extract_number(r"\boxed{18 \text{ dollars}}"), "18")
    assert score.gsm_equiv(score.extract_number("Total is 5 apples and then 70000."), "70000")   # no box: last number
    assert not score.gsm_equiv(score.extract_number(r"\boxed{17}"), "18")
    assert score.extract_number("none") is None


def test_gsm_ignores_think_block():
    assert score.extract_number("<think>maybe 99</think>The answer is \\boxed{7}") == "7"
    assert score.extract_number("<think>still thinking 99") is None


@pytest.mark.parametrize("pred,gold,ok", [
    (r"\dfrac{1}{2}", r"\frac12", True),
    (r"0.5", r"\frac{1}{2}", True),
    (r"x = 5", "5", True),
    (r"\left( 3, \frac{\pi}{2} \right)", r"\left( 3, \frac{\pi}{2} \right)", True),
    (r"(3,\frac{\pi}{2})", r"\left( 3, \frac{\pi}{2} \right)", True),
    (r"90^\circ", "90", True),
    (r"\text{(A)}", r"\text{(A)}", True),
    (r"1,000", "1000", True),
    (r"\frac{1}{3}", r"\frac{1}{2}", False),
    (None, "5", False),
])
def test_math_equiv(pred, gold, ok):
    assert score.math_equiv(pred, gold) is ok


@pytest.mark.parametrize("text,valid,want", [
    ("C", "ABCDEFGHIJ", "C"),
    ("**C**", "ABCDEFGHIJ", "C"),
    ("(D)", "ABCDEFGHIJ", "D"),
    ("The answer is B.", "ABCDEFGHIJ", "B"),
    ("Answer: **J**", "ABCDEFGHIJ", "J"),
    (r"so \boxed{E}", "ABCDEFGHIJ", "E"),
    ("A. Because it fits\nbut actually the answer is C", "ABCDEFGHIJ", "C"),
    ("<think>answer is A</think>B", "ABCDEFGHIJ", "B"),
    ("I cannot decide", "ABCDE", None),
    ("The answer is G", "ABCDE", None),                    # outside the valid letters
    ("", "ABCDE", None),
])
def test_extract_letter(text, valid, want):
    assert score.extract_letter(text, valid) == want


# ---- HumanEval runner ----

GOLD = {"prompt": "def add(a, b):\n    \"\"\"Add.\"\"\"\n",
        "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(-1, 1) == 0\n",
        "entry_point": "add"}


def test_humaneval_correct_wrong_body_and_timeout():
    ok, _ = score.run_humaneval(GOLD, "Here:\n```python\ndef add(a, b):\n    return a + b\n```")
    assert ok
    ok, _ = score.run_humaneval(GOLD, "```python\n    return a + b\n```")             # a bare body continues the prompt
    assert ok
    ok, note = score.run_humaneval(GOLD, "```python\ndef add(a, b):\n    return a - b\n```")
    assert not ok and "AssertionError" in note
    ok, note = score.run_humaneval(GOLD, "```python\ndef add(a, b):\n    while True:\n        pass\n```", timeout=1)
    assert not ok and note == "timeout"
    ok, _ = score.run_humaneval(GOLD, "no code at all")
    assert not ok


def test_humaneval_runs_in_fresh_dir():
    out = "```python\nimport os\ndef add(a, b):\n    open('leak.txt', 'w').write('x')\n    return a + b\n```"
    ok, _ = score.run_humaneval(GOLD, out)
    assert ok and not Path("leak.txt").exists()


def test_score_item_each_set():
    row = lambda o: {"output": o, "completion_tokens": 5, "finish_reason": "stop"}
    assert score.score_item({"id": "g/1", "set": "gsm8k", "gold": "18"}, row(r"\boxed{18}"))["correct"]
    assert score.score_item({"id": "m/1", "set": "math500", "gold": r"\frac{1}{2}"}, row(r"\boxed{0.5}"))["correct"]
    assert score.score_item({"id": "q/1", "set": "mmlu_pro", "gold": "D"}, row("D"))["correct"]
    assert not score.score_item({"id": "j/1", "set": "jcommonsenseqa", "gold": "C"}, row("F"))["correct"]
    assert score.score_item({"id": "h/1", "set": "humaneval", "gold": GOLD}, row("```python\ndef add(a,b):\n    return a+b\n```"))["correct"]
    assert not score.score_item({"id": "g/2", "set": "gsm8k", "gold": "18"}, {"error": "boom"})["correct"]


# ---- subsets ----

def test_stratified_is_proportional_and_fixed():
    rows = [{"c": "a", "i": i} for i in range(60)] + [{"c": "b", "i": i} for i in range(30)] + [{"c": "c", "i": i} for i in range(10)]
    s1 = subsets.stratified(rows, "c", 20, random.Random(0))
    s2 = subsets.stratified(rows, "c", 20, random.Random(0))
    assert s1 == s2 and len(s1) == 20
    assert [sum(r["c"] == c for r in s1) for c in "abc"] == [12, 6, 2]


# ---- compare ----

def _scored(flags, set_="s", tokens=100, cut=()):
    return {f"{set_}/{i}": {"id": f"{set_}/{i}", "set": set_, "correct": bool(f), "completion_tokens": tokens,
                            "finish_reason": "length" if i in cut else "stop"} for i, f in enumerate(flags)}


def test_mcnemar_exact_known_values():
    assert compare.mcnemar_exact(0, 0) == 1.0
    assert compare.mcnemar_exact(5, 5) == 1.0
    assert compare.mcnemar_exact(0, 10) == pytest.approx(2 / 1024)
    assert compare.mcnemar_exact(3, 9) == pytest.approx(2 * 299 / 4096)   # n=12: C(12,0..3) = 1+12+66+220


def test_compare_identical_is_non_inferior():
    flags = [1, 0] * 150
    r = compare.compare(_scored(flags), _scored(flags), margin=2, boot=500)
    assert r["pooled"]["diff"] == 0 and r["pooled"]["p"] == 1.0 and r["verdict"] == "NON-INFERIOR"


def test_compare_clear_drop_is_inferior():
    ref = [1] * 300
    cand = [1] * 240 + [0] * 60                                     # -20 pp
    r = compare.compare(_scored(ref), _scored(cand), margin=2, boot=500)
    assert r["pooled"]["diff"] == pytest.approx(-0.2) and r["verdict"] == "INFERIOR" and r["pooled"]["p"] < 1e-10


def test_compare_small_sample_is_inconclusive_and_pairs_by_id():
    ref = _scored([1, 1, 1, 1, 0, 0, 1, 1, 1, 1])
    cand = _scored([1, 1, 1, 0, 0, 0, 1, 1, 1, 1])
    del cand["s/9"]                                                 # an id only in REF is ignored
    r = compare.compare(ref, cand, margin=2, boot=500)
    assert r["ids"] == 9 and r["only_a"] == 1 and r["verdict"] == "INCONCLUSIVE"


def test_compare_per_set_tokens_and_truncation():
    ref = {**_scored([1] * 50, "a", tokens=100), **_scored([1] * 50, "b", tokens=200)}
    cand = {**_scored([1] * 50, "a", tokens=300, cut=range(10)), **_scored([1] * 50, "b", tokens=200)}
    r = compare.compare(ref, cand, boot=200)
    assert set(r["per_set"]) == {"a", "b"}
    assert r["per_set"]["a"]["tok_a"] == 100 and r["per_set"]["a"]["tok_b"] == 300
    assert r["per_set"]["a"]["cut_b"] == 0.2 and r["pooled"]["tok_b"] == 250
    assert "NON-INFERIOR" in compare.render(r)


def test_compare_cli_exit_code(tmp_path, capsys):
    f1, f2 = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    f1.write_text("".join(json.dumps(x) + "\n" for x in _scored([1] * 200).values()))
    f2.write_text("".join(json.dumps(x) + "\n" for x in _scored([1] * 150 + [0] * 50).values()))
    assert compare.main([str(f1), str(f2), "--boot", "200"]) == 2
    assert "INFERIOR" in capsys.readouterr().out


# ---- run.py against a fake server ----

class _Fake(BaseHTTPRequestHandler):
    seen: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Fake.seen.append((self.path, body))
        if "FAIL" in body["messages"][0]["content"] and _Fake.fail_left > 0:
            _Fake.fail_left -= 1
            self.send_response(500)
            self.end_headers()
            return
        out = json.dumps({"choices": [{"message": {"content": r"\boxed{4}", "reasoning_content": "think"}, "finish_reason": "stop"}],
                          "usage": {"prompt_tokens": 7, "completion_tokens": 11}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


@pytest.fixture
def fake_server():
    _Fake.seen, _Fake.fail_left = [], 0
    srv = HTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def _subsets_file(tmp_path, n=3):
    p = tmp_path / "subsets.jsonl"
    p.write_text("".join(json.dumps({"id": f"gsm8k/{i}", "set": "gsm8k", "prompt": f"What is 2+2? #{i}", "gold": "4"}) + "\n"
                         for i in range(n)))
    return p


def test_run_request_shape_records_and_resume(fake_server, tmp_path):
    sub, out = _subsets_file(tmp_path), tmp_path / "out.jsonl"
    assert run.main(["--base-url", fake_server, "--subsets", str(sub), "--out", str(out), "--max-tokens", "123"]) == 0
    path, body = _Fake.seen[0]
    assert path == "/v1/chat/completions" and body["temperature"] == 0 and body["max_tokens"] == 123
    assert "chat_template_kwargs" not in body and body["messages"] == [{"role": "user", "content": "What is 2+2? #0"}]
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert [r["id"] for r in rows] == ["gsm8k/0", "gsm8k/1", "gsm8k/2"]
    assert rows[0]["output"] == r"\boxed{4}" and rows[0]["reasoning"] == "think" and rows[0]["completion_tokens"] == 11
    assert rows[0]["prompt_tokens"] == 7 and rows[0]["finish_reason"] == "stop" and rows[0]["seconds"] >= 0

    # resume: nothing to do for finished ids, a new item runs alone
    n = len(_Fake.seen)
    assert run.main(["--base-url", fake_server, "--subsets", str(sub), "--out", str(out)]) == 0
    assert len(_Fake.seen) == n
    _subsets_file(tmp_path, 4)
    run.main(["--base-url", fake_server, "--subsets", str(sub), "--out", str(out)])
    assert len(_Fake.seen) == n + 1 and len(out.read_text().splitlines()) == 4


def test_run_error_is_recorded_then_retried_on_resume(fake_server, tmp_path, monkeypatch):
    monkeypatch.setattr(run.time, "sleep", lambda s: None)
    sub, out = tmp_path / "s.jsonl", tmp_path / "o.jsonl"
    sub.write_text(json.dumps({"id": "gsm8k/0", "set": "gsm8k", "prompt": "FAIL", "gold": "4"}) + "\n")
    _Fake.fail_left = 99
    run.main(["--base-url", fake_server, "--subsets", str(sub), "--out", str(out)])
    assert "error" in json.loads(out.read_text().splitlines()[0])
    _Fake.fail_left = 0
    run.main(["--base-url", fake_server, "--subsets", str(sub), "--out", str(out)])
    assert "error" not in json.loads(out.read_text().splitlines()[-1])


def test_run_then_score_end_to_end(fake_server, tmp_path):
    sub, out, scored = _subsets_file(tmp_path), tmp_path / "o.jsonl", tmp_path / "sc.jsonl"
    run.main(["--base-url", fake_server, "--subsets", str(sub), "--out", str(out), "--limit", "2"])
    assert score.main([str(out), "--out", str(scored), "--subsets", str(sub)]) == 0
    res = [json.loads(x) for x in scored.read_text().splitlines()]
    assert len(res) == 2 and all(r["correct"] for r in res)
