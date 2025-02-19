"""
gpt_request.py

This module defines the GptRequest data model used to construct requests for the GPT API.
It encapsulates all necessary parameters and provides a method to convert the request into a dictionary
suitable for JSON serialization.
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class GptRequest:
    """
    Represents a request payload for the GPT API.

    Attributes:
        prompt (str): The text prompt to send to the GPT API.
        max_tokens (int): The maximum number of tokens to generate in the response.
        temperature (float): Sampling temperature to control randomness (0.0 to 1.0).
        top_p (float): Nucleus sampling probability threshold (0.0 to 1.0).
        additional_params (Dict[str, Any]): A dictionary of any additional parameters.
    """
    prompt: str
    max_tokens: int = 150
    temperature: float = 0.7
    top_p: float = 1.0
    additional_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the GptRequest instance into a dictionary for JSON serialization.

        Returns:
            dict: The request payload as a dictionary.
        """
        payload = {
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p
        }
        payload.update(self.additional_params)
        return payload
