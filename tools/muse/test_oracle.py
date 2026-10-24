from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from tools.gguf_reader import GGUFFile
from tools.muse import fixture, reference
from tools.muse.compare import compare


def mutant(function, replacements, overrides=None):
    """Compile deliberately wrong graph variants, without production fault hooks.

    Every source anchor must match exactly once, so refactors cannot silently
    disable a sensitivity check. Only test-local function globals are changed.
    """
    source = inspect.getsource(function)
    for before, after in replacements:
        if source.count(before) != 1:
            raise AssertionError(f"mutation anchor not unique: {before}")
        source = source.replace(before, after)
    namespace = dict(vars(reference))
    namespace.update(overrides or {})
    exec(compile(source, "<deliberately-wrong-oracle>", "exec"), namespace)
    return namespace[function.__name__]


class FixtureTests(unittest.TestCase):
    def test_deterministic_serialization_and_readback(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a", Path(directory) / "b"
            expected = fixture.generate(a)
            fixture.generate(b)
            for name in ("model.gguf", "reference.json"):
                self.assertEqual((a / name).read_bytes(), (b / name).read_bytes())
            model = GGUFFile(a / "model.gguf")
            self.assertEqual(model.version, 3)
            self.assertEqual(model.metadata["general.architecture"], "muse-spark")
            self.assertEqual(model.metadata["muse-spark.attention.sliding_window_pattern"], [True, False])
            self.assertEqual(len(model.tensors), 49)
            original = fixture.weights()
            meta, loaded = reference.load_f32(a / "model.gguf")
            for info in model.tensors:
                self.assertEqual(info.type_name, "F32")
                self.assertEqual(info.shape, list(reversed(original[info.name].shape)))
                self.assertEqual(info.offset % 32, 0)
                np.testing.assert_array_equal(loaded[info.name], original[info.name])
                self.assertTrue(np.all(loaded[info.name] != 0), info.name)
                self.assertGreater(float(np.ptp(loaded[info.name])), 0, info.name)
            self.assertEqual(expected["tokens"], [1, 5, 8, 3, 7, 2, 9])
            self.assertEqual(expected["generated_tokens"], [])
            logits = reference.forward(meta, loaded, fixture.TOKENS)
            self.assertEqual(logits.shape, (7, 64))
            self.assertEqual(logits.dtype, np.float32)
            self.assertTrue(np.isfinite(logits).all())
            self.assertGreater(float(np.std(logits)), 0.1)
            np.testing.assert_array_equal(logits, np.asarray(expected["logits"], dtype=np.float32))

    def test_reject_truncated_and_non_f32_tensors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.gguf"
            fixture.write_model(path, fixture.metadata(), fixture.weights())
            path.write_bytes(path.read_bytes()[:-4])
            with self.assertRaisesRegex(ValueError, "truncated"):
                reference.load_f32(path)
            from tools.gguf_writer import GGUFWriter
            writer = GGUFWriter()
            writer.add_f32("unsupported.weight", np.ones(8, dtype=np.float32))
            writer.tensors[-1].type_name = "F16"
            writer.write(path)
            with self.assertRaisesRegex(ValueError, "only accepts F32"):
                reference.load_f32(path)

    def test_fixture_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-m", "tools.muse.fixture", "--output-dir", directory],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((Path(directory) / "model.gguf").exists())
            data = json.loads((Path(directory) / "reference.json").read_text())
            self.assertEqual(np.asarray(data["logits"]).shape, (7, 64))


class GraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Exercise serialized FP32 metadata, not Python's unrounded literals.
        cls.directory = tempfile.TemporaryDirectory()
        path = Path(cls.directory.name) / "fixture.gguf"
        fixture.write_model(path, fixture.metadata(), fixture.weights())
        cls.meta, cls.weights = reference.load_f32(path)
        cls.logits = reference.forward(cls.meta, cls.weights, fixture.TOKENS)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def assert_sensitive(self, candidate):
        actual = candidate(self.meta, self.weights, fixture.TOKENS)
        self.assertTrue(np.isfinite(actual).all())
        difference = float(np.max(np.abs(actual - self.logits)))
        self.assertGreater(difference, 1e-4, f"insensitive mutation: max_abs={difference}")
        # Each bug must fail the actual advertised numerical acceptance gate.
        with self.assertRaises(ValueError):
            compare({"tokens": fixture.TOKENS, "logits": self.logits.tolist()},
                    {"tokens": fixture.TOKENS, "logits": actual.tolist()})

    def test_every_graph_stage_is_observable(self):
        cases = {
            "embedding norm omitted": [("x = rms(tensors[\"token_embd.weight\"][tokens], eps)",
                                        "x = tensors[\"token_embd.weight\"][tokens]")],
            "pre attention norm omitted": [("a = rms(x, eps, w[\"attn_norm.weight\"])", "a = x")],
            "centered norm folded twice": [("a = rms(x, eps, w[\"attn_norm.weight\"])",
                                            "a = rms(x, eps, 1 + w[\"attn_norm.weight\"])")],
            "Q norm omitted": [("q = rms(q, eps, w[\"attn_q_norm.weight\"])", "q = q")],
            "K norm omitted": [("k = rms(k, eps, w[\"attn_k_norm.weight\"])", "k = k")],
            "Q folded scale added twice": [("q = rms(q, eps, w[\"attn_q_norm.weight\"])",
                                            "q = rms(q, eps, 1 + w[\"attn_q_norm.weight\"])")],
            "local RoPE omitted": [("if local:\n", "if False:\n")],
            "global incorrectly uses RoPE": [("if local:\n", "if True:\n")],
            "attention sqrt scale omitted": [("* F(dim ** -0.5)", "* F(1)")],
            "GQA cyclic not contiguous": [("np.repeat(k, heads // kvheads, axis=1)",
                                           "np.tile(k, (1, heads // kvheads, 1))"),
                                          ("np.repeat(v, heads // kvheads, axis=1)",
                                           "np.tile(v, (1, heads // kvheads, 1))")],
            "local mask omitted": [("start = max(0, t - m[\"attention.sliding_window\"] + 1) if local else 0",
                                    "start = 0")],
            "global mask made local": [("if local else 0", "if local else max(0, t - 2)")],
            "future attention leakage": [("k[start:t+1]", "k[start:n]"), ("v[start:t+1]", "v[start:n]")],
            "per head output norm omitted": [("attended = rms(attended, eps).reshape(n, heads * dim)",
                                              "attended = attended.reshape(n, heads * dim)")],
            "output norm across heads": [("attended = rms(attended, eps).reshape(n, heads * dim)",
                                          "attended = rms(attended.reshape(n, heads * dim), eps)")],
            "sigmoid attention gate omitted": [("(attended * gate) @", "attended @")],
            "gate before head norm": [("attended = rms(attended, eps).reshape(n, heads * dim)",
                                       "attended = attended.reshape(n, heads * dim)"),
                                      ("(attended * gate) @", "rms((attended * gate).reshape(n, heads, dim), eps).reshape(n, heads * dim) @")],
            "post attention norm omitted": [("attention = rms(attention, post_eps)", "attention = attention")],
            "attention alpha omitted": [("x * w[\"attn_resid_alpha.weight\"]", "x")],
            "attention beta omitted": [("attention * w[\"attn_resid_beta.weight\"]", "attention")],
            "FFN stream norm omitted": [("ffn = rms(stream, eps, w[\"ffn_norm.weight\"])", "ffn = stream")],
            "router reads unnormalized stream": [("route(ffn,", "route(stream,")],
            "latent prenorm omitted": [("latent = rms(ffn @ w[\"ffn_latent_down.weight\"].T, eps, w[\"ffn_pre_exp_norm.weight\"])",
                                        "latent = ffn @ w[\"ffn_latent_down.weight\"].T")],
            "latent preexpert scale before RMS": [("rms(ffn @ w[\"ffn_latent_down.weight\"].T, eps, w[\"ffn_pre_exp_norm.weight\"])",
                                                   "input_scaled_norm(ffn @ w[\"ffn_latent_down.weight\"].T, w[\"ffn_pre_exp_norm.weight\"], eps)")],
            "only one expert selected": [("m[\"expert_used_count\"], m[\"expert_weights_scale\"]", "1, m[\"expert_weights_scale\"]")],
            "SwiGLU sigmoid omitted": [("((g * sigmoid(g)) * u)", "(g * u)")],
            "expert postnorm omitted": [("expert_out = input_scaled_norm(expert_out, w[\"ffn_post_exp_norm.weight\"], post_eps)",
                                         "expert_out = expert_out")],
            "expert postnorm scale after RMS": [("input_scaled_norm(expert_out, w[\"ffn_post_exp_norm.weight\"], post_eps)",
                                                 "rms(expert_out, post_eps, w[\"ffn_post_exp_norm.weight\"])")],
            "expert normalization after accumulation": [("expert_out = input_scaled_norm(expert_out, w[\"ffn_post_exp_norm.weight\"], post_eps)",
                                                         "expert_out = expert_out"),
                                                        ("branch = mixed @", "mixed = input_scaled_norm(mixed, w[\"ffn_post_exp_norm.weight\"], post_eps)\n        branch = mixed @")],
            "FFN postnorm omitted": [("branch = input_scaled_norm(branch, w[\"post_ffw_norm.weight\"], post_eps)",
                                      "branch = branch")],
            "FFN scale after RMS": [("input_scaled_norm(branch, w[\"post_ffw_norm.weight\"], post_eps)",
                                     "rms(branch, post_eps, w[\"post_ffw_norm.weight\"])")],
            "FFN alpha omitted": [("stream * w[\"ffn_resid_alpha.weight\"]", "stream")],
            "FFN beta omitted": [("branch * w[\"ffn_resid_beta.weight\"]", "branch")],
            "final weighted norm omitted": [("final = rms(x, eps, tensors[\"output_norm.weight\"])", "final = x")],
            "logit scale omitted": [("* F(m[\"logit_scale\"])", "* F(1)")],
            "softcap omitted": [("logits = cap * np.tanh(logits / cap)", "logits = logits")],
            "softcap before logit scale": [("* F(m[\"logit_scale\"])", "* F(1)"),
                                           ("logits = cap * np.tanh(logits / cap)",
                                            "logits = cap * np.tanh(logits / cap) * F(m[\"logit_scale\"])")],
        }
        for name, edits in cases.items():
            with self.subTest(stage=name):
                self.assert_sensitive(mutant(reference.forward, edits))

    def test_router_misuse_changes_full_logits(self):
        cases = {
            "correction bias omitted": [("choice = probabilities + bias", "choice = probabilities")],
            "bias applied before sigmoid": [("probabilities = sigmoid(x @ router.T)",
                                             "probabilities = sigmoid(x @ router.T + bias)"),
                                            ("choice = probabilities + bias", "choice = probabilities")],
            "biased mixing weights": [("np.take_along_axis(probabilities, selected, axis=-1)",
                                       "np.take_along_axis(choice, selected, axis=-1)")],
            "topk normalization omitted": [("mixing = mixing / (mixing.sum(axis=-1, keepdims=True, dtype=F) + F(1e-15))",
                                            "mixing = mixing")],
            "softmax instead of sigmoid": [("probabilities = sigmoid(x @ router.T)",
                                            "probabilities = np.exp(x @ router.T)\n    probabilities /= probabilities.sum(axis=-1, keepdims=True)")],
            "router multiplier folded twice": [("sigmoid(x @ router.T)", "sigmoid(F(0.7) * (x @ router.T))")],
            "expert weight scale omitted": [("mixing * F(scale)", "mixing")],
        }
        for name, edits in cases.items():
            with self.subTest(stage=name):
                route = mutant(reference.route, edits)
                self.assert_sensitive(mutant(reference.forward, [], {"route": route}))

    def test_causal_prefix_and_local_boundary(self):
        for length in range(1, len(fixture.TOKENS) + 1):
            actual = reference.forward(self.meta, self.weights, fixture.TOKENS[:length])
            np.testing.assert_allclose(actual, self.logits[:length], atol=3e-6, rtol=3e-6)
        no_local = mutant(reference.forward, [("if local else 0", "if False else 0")])
        actual = no_local(self.meta, self.weights, fixture.TOKENS)
        np.testing.assert_array_equal(actual[:3], self.logits[:3])
        self.assertGreater(float(np.max(np.abs(actual[3:] - self.logits[3:]))), 1e-3)
        off_by_one = mutant(reference.forward, [("t - m[\"attention.sliding_window\"] + 1",
                                                "t - m[\"attention.sliding_window\"]")])
        actual = off_by_one(self.meta, self.weights, fixture.TOKENS)
        np.testing.assert_array_equal(actual[:3], self.logits[:3])
        self.assertGreater(float(np.max(np.abs(actual[3] - self.logits[3]))), 1e-3)

    def test_distinct_postnorm_epsilon_is_observable(self):
        meta = dict(self.meta)
        meta["muse-spark.post_norm_epsilon"] = meta["muse-spark.attention.layer_norm_rms_epsilon"]
        changed = reference.forward(meta, self.weights, fixture.TOKENS)
        self.assertGreater(float(np.max(np.abs(changed - self.logits))), 1e-3)
        np.testing.assert_array_equal(reference.sigmoid(np.array([-1000, 1000], dtype=np.float32)), [0, 1])

    def test_normalization_order_and_rope_hand_values(self):
        x = np.array([1, 3], dtype=np.float32)
        scale = np.array([2, 0.5], dtype=np.float32)
        expected = (x * scale) / np.sqrt(np.mean((x * scale) ** 2) + np.float32(1e-8))
        np.testing.assert_array_equal(reference.input_scaled_norm(x, scale, 1e-8), expected)
        self.assertFalse(np.allclose(expected, reference.rms(x, 1e-8, scale)))
        x = np.array([[[1, 2, 3, 4]], [[1, 2, 3, 4]]], dtype=np.float32)
        result = reference.rope(x, 10000)
        np.testing.assert_array_equal(result[0], x[0])
        angles = np.array([1, 0.01], dtype=np.float32)
        expected = np.r_[x[1, 0, :2] * np.cos(angles) - x[1, 0, 2:] * np.sin(angles),
                         x[1, 0, 2:] * np.cos(angles) + x[1, 0, :2] * np.sin(angles)]
        np.testing.assert_allclose(result[1, 0], expected, atol=1e-7)

    def test_fixture_exercises_all_experts_and_bias_changes_selection(self):
        selections = []
        corrected_choices = []

        def observed(x, router, bias, top_k, scale):
            selected, mixing = reference.route(x, router, bias, top_k, scale)
            unbiased, _ = reference.route(x, router, np.zeros_like(bias), top_k, scale)
            selections.append(selected)
            corrected_choices.append(any(set(a) != set(b) for a, b in zip(selected, unbiased)))
            scores = reference.sigmoid(x @ router.T) + bias
            self.assertTrue(np.all(np.diff(np.sort(scores, axis=-1), axis=-1) > 1e-5))
            return selected, mixing

        mutant(reference.forward, [], {"route": observed})(self.meta, self.weights, fixture.TOKENS)
        self.assertEqual(len(selections), 2)
        for selected, corrected in zip(selections, corrected_choices):
            self.assertEqual(set(selected.ravel()), {0, 1, 2, 3})
            self.assertTrue(corrected)

    def test_router_bias_selection_only_and_epsilon(self):
        x = np.array([[1, 0]], dtype=np.float32)
        router = np.array([[-1, 0], [0.1, 0], [1, 0], [-0.2, 0]], dtype=np.float32)
        bias = np.array([0.9, 0, 0, 0], dtype=np.float32)
        selected, mixing = reference.route(x, router, bias, 2, 1.3)
        np.testing.assert_array_equal(selected, [[0, 2]])
        p = reference.sigmoid(np.array([-1, 1], dtype=np.float32))
        np.testing.assert_allclose(mixing[0], p / p.sum() * np.float32(1.3), atol=1e-7)
        # 1e-15 is significant for very small probabilities (normal scores do
        # not expose omitting this denominator stabilizer).
        router[:, 0] = -40
        _, mixing = reference.route(x, router, bias, 2, 1.3)
        p = reference.sigmoid(np.array([-40, -40], dtype=np.float32))
        np.testing.assert_allclose(mixing[0], p / (p.sum() + np.float32(1e-15)) * np.float32(1.3), rtol=1e-6)
        self.assertLess(float(mixing.sum()), 0.02)

    def test_malformed_metadata_and_tensors(self):
        cases = [
            ("general.architecture", "qwen"),
            ("muse-spark.attention.sliding_window_pattern", [True]),
            ("muse-spark.attention.sliding_window_pattern", [1, 0]),
            ("muse-spark.attention.head_count_kv", 3),
            ("muse-spark.expert_used_count", 5),
            ("muse-spark.post_norm_epsilon", float("nan")),
            ("muse-spark.expert_gating_func", 1),
            ("muse-spark.embedding_length", 63),
        ]
        for key, value in cases:
            with self.subTest(metadata=key):
                meta = dict(self.meta)
                meta[key] = value
                with self.assertRaises(ValueError):
                    reference.forward(meta, self.weights, fixture.TOKENS)
        for tokens in ([], [-1], [64], [True], [1] * 33):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                reference.forward(self.meta, self.weights, tokens)
        for problem in ("missing", "shape", "nonfinite"):
            tensors = dict(self.weights)
            key = "blk.0.post_ffw_norm.weight"
            if problem == "missing":
                tensors.pop(key)
            elif problem == "shape":
                tensors[key] = tensors[key][:3]
            else:
                tensors[key] = tensors[key].copy()
                tensors[key][0] = np.nan
            with self.subTest(tensor=problem), self.assertRaises(ValueError):
                reference.forward(self.meta, tensors, fixture.TOKENS)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.good = {"tokens": [1, 2], "logits": [[0.1, -2, 3], [0, 0.5, -1]]}

    def test_explicit_tolerance_and_strict_shapes(self):
        self.assertIn("PASS 2 x 3", compare(self.good, self.good))
        near = copy.deepcopy(self.good)
        near["logits"][0][0] += 1e-6
        compare(self.good, near)
        bad_values = [None, [], {"tokens": [1], "logits": [[1]]}]
        for logits in ([[1], [1]], [[1, 2, 3]], [[1], [1, 2]],
                       [[1, 2, float("nan")], [0, 0.5, -1]],
                       [[1, 2, float("inf")], [0, 0.5, -1]],
                       [["0.1", -2, 3], [0, 0.5, -1]],
                       [[0.1, -2, 3.1], [0, 0.5, -1]]):
            bad_values.append({"tokens": [1, 2], "logits": logits})
        bad_values.append({"tokens": [True, 2], "logits": self.good["logits"]})
        bad_values.append({"tokens": [2, 1], "logits": self.good["logits"]})
        for bad in bad_values:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                compare(self.good, bad)
        for tolerances in ({"atol": -1}, {"rtol": float("nan")}, {"atol": float("inf")}):
            with self.assertRaises(ValueError):
                compare(self.good, self.good, **tolerances)

    def test_comparison_cli_returns_nonzero_for_all_failure_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected, actual = root / "reference.json", root / "actual.json"
            expected.write_text(json.dumps(self.good))
            examples = {
                "pass": self.good,
                "numeric": {"tokens": [1, 2], "logits": [[1, 2, 3], [0, 0.5, -1]]},
                "shape": {"tokens": [1, 2], "logits": [[1], [1]]},
                "nonfinite": {"tokens": [1, 2], "logits": [[float("nan"), -2, 3], [0, 0.5, -1]]},
            }
            for name, example in examples.items():
                actual.write_text(json.dumps(example))
                run = subprocess.run([sys.executable, "-m", "tools.muse.compare", "--reference", str(expected),
                                      "--actual", str(actual)], capture_output=True, text=True)
                with self.subTest(case=name):
                    self.assertEqual(run.returncode, 0 if name == "pass" else 1, run.stdout + run.stderr)
                    self.assertIn("PASS" if name == "pass" else "FAIL", run.stdout + run.stderr)
            actual.write_text("not json")
            run = subprocess.run([sys.executable, "-m", "tools.muse.compare", "--reference", str(expected),
                                  "--actual", str(actual)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 1)


if __name__ == "__main__":
    unittest.main()
