"""
gpt_response.py

This module defines the GptResponse data model to encapsulate responses from the GPT API.
It provides methods for parsing a JSON dictionary into a GptResponse object.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class GptResponse:
    """
    Represents a response received from the GPT API.

    Attributes:
        text (str): The generated text from the GPT API.
        strategy (Optional[str]): An optional field for strategy recommendations, if available.
        raw (Dict[str, Any]): The full raw response data from the API.
    """
    text: str
    strategy: Optional[str] = None
    raw: Dict[str, Any] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GptResponse":
        """
        Creates an instance of GptResponse from a dictionary.

        This method parses the JSON response received from the GPT API, expecting the response
        to contain a list of choices. The text from the first choice is used as the main output.

        Args:
            data: A dictionary representing the JSON response from the GPT API.

        Returns:
            GptResponse: The parsed response object.
        """
        # Extract text from the first choice if available.
        text = ""
        choices = data.get("choices", [])
        if choices and isinstance(choices, list):
            first_choice = choices[0]
            text = first_choice.get("text", "")

        # Optionally extract a strategy if provided in the response.
        strategy = data.get("strategy")

        return cls(text=text, strategy=strategy, raw=data)
