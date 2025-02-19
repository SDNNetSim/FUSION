"""
test_helpers.py

This module contains unit tests for the utility functions in the helpers module,
with a focus on verifying the behavior of the retry_request decorator.
"""

import unittest
import time
from gpt.utils.helpers import retry_request


# Define a custom exception for testing purposes.
class DummyException(Exception):
    pass


class TestRetryRequest(unittest.TestCase):
    def test_retry_success(self):
        """
        Test that a function decorated with retry_request eventually succeeds
        after a few failures.
        """
        call_count = [0]  # Use a mutable type to track call count in the nested function.

        @retry_request(max_retries=5, delay=0.01, exceptions=(DummyException,))
        def sometimes_fail():
            call_count[0] += 1
            if call_count[0] < 3:
                raise DummyException("Intentional failure")
            return "Success"

        result = sometimes_fail()
        self.assertEqual(result, "Success")
        self.assertEqual(call_count[0], 3)

    def test_retry_failure(self):
        """
        Test that the decorated function raises the exception after exhausting retries.
        """

        @retry_request(max_retries=2, delay=0.01, exceptions=(DummyException,))
        def always_fail():
            raise DummyException("Persistent failure")

        with self.assertRaises(DummyException):
            always_fail()

    def test_no_retry_on_unlisted_exception(self):
        """
        Test that the decorator does not catch exceptions that are not specified for retry.
        """

        @retry_request(max_retries=2, delay=0.01, exceptions=(DummyException,))
        def raise_value_error():
            raise ValueError("This error should not be retried")

        with self.assertRaises(ValueError):
            raise_value_error()

    def test_immediate_success(self):
        """
        Test that if the function succeeds on the first try, no retries are performed.
        """

        @retry_request(max_retries=3, delay=0.01, exceptions=(DummyException,))
        def immediate_success():
            return "Immediate Success"

        result = immediate_success()
        self.assertEqual(result, "Immediate Success")


if __name__ == '__main__':
    unittest.main()
