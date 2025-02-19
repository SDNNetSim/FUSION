# pylint: disable=too-few-public-methods

"""
gpt_client.py

This module provides the GptClient class, which handles communication with the GPT API.
It abstracts HTTP requests, error handling, and response parsing for the GPT module.
"""

import logging

import requests

from gpt.config.gpt_config import GptConfig
from gpt.models.gpt_request import GptRequest
from gpt.models.gpt_response import GptResponse
from gpt.errors.gpt_errors import GptApiError, GptConnectionError
from gpt.utils.helpers import retry_request


class GptClient:
    """
    A client for interacting with the GPT API.
    """

    def __init__(self, config: GptConfig):
        """
        Initialize the GptClient with a given configuration.

        :param config: An instance of GptConfig containing API settings.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    @retry_request
    def send_prompt(self, prompt: str, **kwargs) -> GptResponse:
        """
        Sends a prompt to the GPT API and returns a parsed response.

        :param prompt: The prompt text to send to the GPT API.
        :param kwargs: Additional parameters to include in the API request.
        :return: A GptResponse object containing the API's response.
        :raises GptApiError: If the API returns an error status code or invalid data.
        :raises GptConnectionError: If a network-related error occurs.
        """
        # Build the request payload using GptRequest
        request_payload = GptRequest(prompt=prompt, **kwargs)
        payload_dict = request_payload.to_dict()

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(
                self.config.endpoint,
                json=payload_dict,
                headers=headers,
                timeout=self.config.timeout
            )
        except requests.exceptions.RequestException as e:
            self.logger.error("Network error during API call: %s", e)
            raise GptConnectionError("Failed to connect to the GPT API.") from e

        if response.status_code != 200:
            self.logger.error("GPT API returned error %s: %s", response.status_code, response.text)
            raise GptApiError(f"API error {response.status_code}: {response.text}")

        try:
            response_data = response.json()
        except ValueError as e:
            self.logger.error("Failed to parse JSON response: %s", e)
            raise GptApiError("Received invalid JSON from GPT API.") from e

        # Create and return a GptResponse from the received data
        return GptResponse.from_dict(response_data)
