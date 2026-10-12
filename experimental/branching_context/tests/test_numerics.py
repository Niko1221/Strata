import math
import unittest
from context_time_travel.numerics import compare_logits, assert_logits_close


class NumericalOracleTests(unittest.TestCase):
    def test_exact_vectors_pass(self):
        result = assert_logits_close([4.0, 1.0, -3.0], [4.0, 1.0, -3.0], atol=0, rtol=0)
        self.assertEqual(result.max_absolute_error, 0)
        self.assertEqual(result.argmax_expected, 0)

    def test_small_numerical_error_passes(self):
        r = assert_logits_close([4.00001, 1.0, -3.0], [4.0, 1.0, -3.0], atol=2e-5, rtol=0)
        self.assertTrue(r.pass_elementwise)

    def test_corruption_is_detected(self):
        with self.assertRaises(AssertionError):
            assert_logits_close([4.0, 1.0, 3.0], [4.0, 1.0, -3.0], atol=1e-3, rtol=0)

    def test_top1_flip_is_detected(self):
        r = compare_logits([1.0, 4.0], [4.0, 1.0], atol=0.01, rtol=0)
        self.assertFalse(r.pass_elementwise)
        self.assertFalse(r.top1_stable_within_tolerance)

    def test_close_tie_is_not_mislabelled_large_margin(self):
        r = compare_logits([1.00001, 1.00002], [1.00002, 1.00001], atol=2e-5, rtol=0)
        self.assertTrue(r.pass_elementwise)
        self.assertTrue(r.top1_stable_within_tolerance)

    def test_invalid_tolerance_rejected(self):
        with self.assertRaises(ValueError):
            compare_logits([1.], [1.], atol=-0.1, rtol=0)

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            compare_logits([math.nan, 1.], [2., 1.], atol=0.01, rtol=0)

    def test_nonmatching_shape_rejected(self):
        with self.assertRaises(ValueError):
            compare_logits([1, 2], [1], atol=0.01, rtol=0)

    def test_kl_zero_for_identical_values(self):
        r = compare_logits([1., 2., -4.], [1., 2., -4.], atol=0, rtol=0)
        self.assertLess(r.kl_expected_to_actual, 1e-12)

    def test_shift_in_logits_detected_by_elementwise_even_if_softmax_same(self):
        r = compare_logits([101., 102.], [1., 2.], atol=1e-5, rtol=0)
        self.assertFalse(r.pass_elementwise)
        self.assertLess(r.kl_expected_to_actual, 1e-12)


if __name__ == "__main__":
    unittest.main()
