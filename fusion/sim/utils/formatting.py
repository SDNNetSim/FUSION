"""
String formatting and conversion utilities.

This module provides functions for converting between different string formats
and formatting data for display.
"""

from typing import Any


def snake_to_title(snake_str: str) -> str:
    """
    Convert a snake string to a title string.

    :param snake_str: String to convert in snake case
    :type snake_str: str
    :return: String in title case
    :rtype: str
    """
    words_list = snake_str.split("_")
    title_str = " ".join(word.capitalize() for word in words_list)
    return title_str


def int_to_string(number: int) -> str:
    """
    Convert an integer to a string with thousands separator.

    :param number: Number to convert
    :type number: int
    :return: Original number as a string with commas
    :rtype: str
    """
    return f"{number:,}"


def list_to_title(input_list: list[tuple[str, Any]]) -> str:
    """
    Convert a list to a title case string.

    :param input_list: Input list to convert, each element is a word
    :type input_list: list[tuple[str, Any]]
    :return: Title string
    :rtype: str
    """
    if not input_list:
        return ""

    unique_list = []
    for item in input_list:
        if item[0] not in unique_list:
            unique_list.append(item[0])

    if len(unique_list) > 1:
        return ", ".join(unique_list[:-1]) + " & " + unique_list[-1]

    return unique_list[0]
