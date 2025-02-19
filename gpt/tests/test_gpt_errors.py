"""
test_gpt_errors.py

This module contains unit tests for the custom exception classes in the GPT module.
It verifies that each exception correctly stores and returns the provided error messages and attributes.
"""

import unittest
from gpt.errors.gpt_errors import GptError, GptApiError, GptConnectionError, GptInvalidResponseError


class TestGptErrors(unittest.TestCase):

    def test_gpt_api_error(self):
        """
        Test that GptApiError correctly stores the error message and status code.
        """
        error_message = "API error occurred"
        status_code = 404
        error = GptApiError(error_message, status_code=status_code)
        self.assertEqual(str(error), error_message)
        self.assertEqual(error.status_code, status_code)
        self.assertTrue(isinstance(error, GptError))

    def test_gpt_connection_error(self):
        """
        Test that GptConnectionError correctly stores and returns the error message.
        """
        error_message = "Connection failed"
        error = GptConnectionError(error_message)
        self.assertEqual(str(error), error_message)
        self.assertTrue(isinstance(error, GptError))

    def test_gpt_invalid_response_error(self):
        """
        Test that GptInvalidResponseError correctly stores and returns the error message.
        """
        error_message = "Invalid response"
        error = GptInvalidResponseError(error_message)
        self.assertEqual(str(error), error_message)
        self.assertTrue(isinstance(error, GptError))


if __name__ == '__main__':
    unittest.main()
