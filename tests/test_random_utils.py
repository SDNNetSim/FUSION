import unittest
from fusion.utils.random import set_seed, get_uniform_rv, get_exponential_rv


class TestRandomGenerators(unittest.TestCase):
    """
    Test random_helpers.py
    """

    def test_set_seed(self):
        """
        Tests the set seed method.
        """
        set_seed(42)
        result1 = get_uniform_rv()
        set_seed(42)
        result2 = get_uniform_rv()
        self.assertEqual(result1, result2, "The results should be the same when the same seed is set.")

    def test_uniform_rv_without_scale(self):
        """
        Tests the uniform random variable method without a scale parameter.
        """
        set_seed(42)
        result = get_uniform_rv()
        self.assertTrue(0 <= result <= 1, "The result should be within [0, 1].")

    def test_uniform_rv_with_scale(self):
        """
        Tests the uniform random variable with a scale parameter.
        """
        set_seed(42)
        scale_param = 10
        result = get_uniform_rv(scale_param)
        self.assertTrue(0 <= result <= scale_param, "The result should be within [0, scale_param].")

    def test_exponential_rv(self):
        """
        Tests the exponential random variable method.
        """
        set_seed(42)
        scale_param = 5
        result = get_exponential_rv(scale_param)
        self.assertTrue(result >= 0, "The result should be non-negative.")
