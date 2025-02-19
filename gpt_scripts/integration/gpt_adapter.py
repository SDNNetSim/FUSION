"""
gpt_adapter.py

This module defines an adapter class for integrating GPT API calls with the optical networking simulator.
It translates simulation data into GPT prompts and processes the API responses to extract actionable insights.
"""

from gpt.client.gpt_client import GptClient
from gpt.errors.gpt_errors import GptApiError
from gpt.models.gpt_response import GptResponse


class GptAdapter:
    """
    Adapter class that bridges the GPT API with the optical networking simulation.
    It transforms simulation data into prompts and interprets GPT responses into strategies or insights.
    """

    def __init__(self, gpt_client: GptClient):
        """
        Initialize the adapter with a GptClient instance.

        :param gpt_client: An instance of GptClient configured to interact with the GPT API.
        """
        self.gpt_client = gpt_client

    def generate_network_strategy(self, network_data: dict) -> dict:
        """
        Generates a network strategy by sending simulation data as a prompt to the GPT API.

        :param network_data: Dictionary containing network simulation parameters and metrics.
        :return: A dictionary containing the strategy recommendations.
        :raises GptApiError: If the GPT API call results in an error.
        """
        prompt = self._build_prompt(network_data)
        gpt_response: GptResponse = self.gpt_client.send_prompt(prompt)
        return self._process_response(gpt_response)

    def _build_prompt(self, network_data: dict) -> str:
        """
        Construct a GPT prompt from the provided network simulation data.

        :param network_data: Dictionary containing simulation parameters.
        :return: A formatted string prompt for the GPT API.
        """
        prompt_lines = [
            "Analyze the following optical network data and suggest optimization strategies:",
        ]
        for key, value in network_data.items():
            prompt_lines.append(f"{key}: {value}")
        return "\n".join(prompt_lines)

    def _process_response(self, gpt_response: GptResponse) -> dict:
        """
        Process the GPT API response to extract network strategy recommendations.

        :param gpt_response: An instance of GptResponse containing the API response.
        :return: A dictionary with the strategy details.
        """
        # Assuming the GPT response object contains a 'strategy' attribute; fallback to full text if not.
        strategy_text = getattr(gpt_response, "strategy", gpt_response.text)
        return {"strategy": strategy_text}
