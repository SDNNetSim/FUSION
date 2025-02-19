# pylint: disable=unnecessary-pass

"""
gpt_errors.py

This module defines custom exception classes for handling errors within the GPT module.
"""


class GptError(Exception):
    """Base exception class for GPT module errors."""
    pass


class GptApiError(GptError):
    """Exception raised when the GPT API returns an error response."""

    def __init__(self, message: str, status_code: int = None):
        super().__init__(message)
        self.status_code = status_code


class GptConnectionError(GptError):
    """Exception raised when there is a network-related error during a GPT API call."""
    pass


class GptInvalidResponseError(GptError):
    """Exception raised when the GPT API returns an invalid or unexpected response."""
    pass
