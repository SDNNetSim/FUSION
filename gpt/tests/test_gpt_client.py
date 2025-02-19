"""
test_gpt_client.py

This module contains unit tests for the GptClient class, ensuring that API calls are handled correctly,
including success scenarios, error responses, network issues, and invalid JSON responses.
"""

import unittest
from unittest.mock import patch, MagicMock
import requests

from gpt.client.gpt_client import GptClient
from gpt.config.gpt_config import GptConfig
from gpt.models.gpt_response import GptResponse
from gpt.errors.gpt_errors import GptApiError, GptConnectionError

# Sample configuration for testing
TEST_API_KEY = "test-api-key"
TEST_ENDPOINT = "https://api.test.com/v1/test"
TEST_TIMEOUT = 5


class DummyGptConfig(GptConfig):
    def __init__(self):
        super().__init__(api_key=TEST_API_KEY, endpoint=TEST_ENDPOINT, timeout=TEST_TIMEOUT)


class TestGptClient(unittest.TestCase):
    def setUp(self):
        self.config = DummyGptConfig()
        self.client = GptClient(config=self.config)
        self.sample_prompt = "Test prompt"
        self.sample_request_payload = {
            "prompt": self.sample_prompt,
            "max_tokens": 150,
            "temperature": 0.7,
            "top_p": 1.0
        }
        self.sample_response_data = {
            "choices": [
                {
                    "text": "Test response text"
                }
            ]
        }

    @patch("gpt.client.gpt_client.requests.post")
    def test_send_prompt_success(self, mock_post):
        # Set up the mock response for a successful API call
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.sample_response_data
        mock_post.return_value = mock_response

        # Call send_prompt and verify that the returned GptResponse is as expected.
        response: GptResponse = self.client.send_prompt(self.sample_prompt)
        self.assertEqual(response.text, "Test response text")
        self.assertEqual(response.raw, self.sample_response_data)

        # Verify that the request was called with correct parameters.
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], self.config.endpoint)
        self.assertEqual(kwargs["json"], self.sample_request_payload)
        self.assertEqual(kwargs["headers"]["Authorization"], f"Bearer {self.config.api_key}")
        self.assertEqual(kwargs["timeout"], self.config.timeout)

    @patch("gpt.client.gpt_client.requests.post")
    def test_send_prompt_connection_error(self, mock_post):
        # Configure the mock to raise a requests exception simulating a network error.
        mock_post.side_effect = requests.exceptions.RequestException("Network failure")

        with self.assertRaises(GptConnectionError):
            self.client.send_prompt(self.sample_prompt)

    @patch("gpt.client.gpt_client.requests.post")
    def test_send_prompt_api_error(self, mock_post):
        # Simulate a non-200 response from the API.
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_post.return_value = mock_response

        with self.assertRaises(GptApiError) as context:
            self.client.send_prompt(self.sample_prompt)
        self.assertIn("400", str(context.exception))
        self.assertIn("Bad Request", str(context.exception))

    @patch("gpt.client.gpt_client.requests.post")
    def test_send_prompt_invalid_json(self, mock_post):
        # Simulate a successful HTTP response but with invalid JSON content.
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("No JSON object could be decoded")
        mock_post.return_value = mock_response

        with self.assertRaises(GptApiError) as context:
            self.client.send_prompt(self.sample_prompt)
        self.assertIn("invalid JSON", str(context.exception).lower())


if __name__ == '__main__':
    unittest.main()
