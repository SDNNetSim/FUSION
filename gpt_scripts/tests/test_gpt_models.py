"""
test_gpt_models.py

This module contains unit tests for the data models in the GPT module,
including tests for GptRequest and GptResponse classes.
"""

import unittest
from gpt.models.gpt_request import GptRequest
from gpt.models.gpt_response import GptResponse


class TestGptRequest(unittest.TestCase):
    def test_to_dict_defaults(self):
        """
        Test that GptRequest.to_dict returns the correct dictionary with default values.
        """
        prompt_text = "Test prompt"
        request = GptRequest(prompt=prompt_text)
        expected_dict = {
            "prompt": prompt_text,
            "max_tokens": 150,
            "temperature": 0.7,
            "top_p": 1.0
        }
        self.assertEqual(request.to_dict(), expected_dict)

    def test_to_dict_with_additional_params(self):
        """
        Test that additional_params are correctly merged into the dictionary output.
        """
        prompt_text = "Another test prompt"
        additional = {"frequency_penalty": 0.5, "presence_penalty": 0.3}
        request = GptRequest(prompt=prompt_text, additional_params=additional)
        expected_dict = {
            "prompt": prompt_text,
            "max_tokens": 150,
            "temperature": 0.7,
            "top_p": 1.0,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.3
        }
        self.assertEqual(request.to_dict(), expected_dict)


class TestGptResponse(unittest.TestCase):
    def test_from_dict_with_choices(self):
        """
        Test that GptResponse.from_dict extracts text from the first choice.
        """
        data = {
            "choices": [
                {"text": "Generated response text"}
            ],
            "strategy": "Optimized network strategy"
        }
        response = GptResponse.from_dict(data)
        self.assertEqual(response.text, "Generated response text")
        self.assertEqual(response.strategy, "Optimized network strategy")
        self.assertEqual(response.raw, data)

    def test_from_dict_without_choices(self):
        """
        Test that GptResponse.from_dict returns empty text when no choices are provided.
        """
        data = {
            "choices": [],
        }
        response = GptResponse.from_dict(data)
        self.assertEqual(response.text, "")
        self.assertIsNone(response.strategy)
        self.assertEqual(response.raw, data)

    def test_from_dict_without_strategy(self):
        """
        Test that GptResponse.from_dict correctly handles responses without a strategy field.
        """
        data = {
            "choices": [
                {"text": "Only response text"}
            ]
        }
        response = GptResponse.from_dict(data)
        self.assertEqual(response.text, "Only response text")
        self.assertIsNone(response.strategy)


if __name__ == '__main__':
    unittest.main()
