"""
test_gpt_config.py

This module contains unit tests for the GptConfig class, ensuring that configuration is correctly
loaded from environment variables and defaults are applied as expected.
"""

import os
import unittest
from unittest.mock import patch

from gpt.config.gpt_config import GptConfig


class TestGptConfig(unittest.TestCase):

    def test_from_env_success(self):
        """
        Test that GptConfig.from_env correctly loads all configuration values from environment variables.
        """
        test_api_key = "test_api_key_value"
        test_endpoint = "https://api.test.com/v1/endpoint"
        test_timeout = "10"  # Environment variables are strings

        env_vars = {
            "GPT_API_KEY": test_api_key,
            "GPT_ENDPOINT": test_endpoint,
            "GPT_TIMEOUT": test_timeout,
        }
        with patch.dict(os.environ, env_vars):
            config = GptConfig.from_env()
            self.assertEqual(config.api_key, test_api_key)
            self.assertEqual(config.endpoint, test_endpoint)
            self.assertEqual(config.timeout, int(test_timeout))

    def test_from_env_defaults(self):
        """
        Test that GptConfig.from_env uses default values for endpoint and timeout when not provided in environment variables.
        """
        test_api_key = "default_test_api_key"
        env_vars = {
            "GPT_API_KEY": test_api_key
            # GPT_ENDPOINT and GPT_TIMEOUT are not set
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = GptConfig.from_env()
            self.assertEqual(config.api_key, test_api_key)
            # Check default values defined in the class
            self.assertEqual(config.endpoint, "https://api.openai.com/v1/engines/davinci-codex/completions")
            self.assertEqual(config.timeout, 30)

    def test_from_env_missing_api_key(self):
        """
        Test that GptConfig.from_env raises a ValueError when GPT_API_KEY is missing.
        """
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                GptConfig.from_env()


if __name__ == '__main__':
    unittest.main()
