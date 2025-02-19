"""
gpt_config.py

This module provides the GptConfig class for managing configuration settings
for the GPT API integration, such as API keys, endpoints, and timeout settings.
"""

import os
from dataclasses import dataclass


@dataclass
class GptConfig:
    """
    Configuration settings for the GPT API integration.

    Attributes:
        api_key (str): API key for authentication with the GPT API.
        endpoint (str): URL endpoint for the GPT API.
        timeout (int): Request timeout in seconds.
    """
    api_key: str
    endpoint: str = "https://api.openai.com/v1/engines/davinci-codex/completions"
    timeout: int = 30

    @classmethod
    def from_env(cls) -> "GptConfig":
        """
        Creates a GptConfig instance from environment variables.

        Expected Environment Variables:
            GPT_API_KEY: API key for the GPT API (required).
            GPT_ENDPOINT: URL endpoint for the GPT API (optional).
            GPT_TIMEOUT: Request timeout in seconds (optional).

        Returns:
            GptConfig: Configured instance of GptConfig.

        Raises:
            ValueError: If the GPT_API_KEY environment variable is not set.
        """
        api_key = os.getenv("GPT_API_KEY")
        if not api_key:
            raise ValueError("Environment variable GPT_API_KEY is not set.")

        endpoint = os.getenv("GPT_ENDPOINT", cls.endpoint)
        timeout = int(os.getenv("GPT_TIMEOUT", cls.timeout))

        return cls(api_key=api_key, endpoint=endpoint, timeout=timeout)
