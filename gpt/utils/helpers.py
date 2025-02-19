"""
helpers.py

This module provides utility functions for the GPT module, including retry logic and logging helpers.
"""

import time
import logging
from functools import wraps
from typing import Callable, Any, Tuple

logger = logging.getLogger(__name__)


def retry_request(max_retries: int = 3, delay: float = 1.0,
                  exceptions: Tuple[Exception, ...] = (Exception,)) -> Callable:
    """
    Decorator to retry a function call if one of the specified exceptions is raised.

    :param max_retries: Maximum number of retry attempts.
    :param delay: Delay in seconds between retries.
    :param exceptions: Tuple of exception classes that should trigger a retry.
    :return: The decorated function with retry logic applied.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            attempt = 0
            while attempt < max_retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    logger.warning("Attempt %s/%s failed for %s with error: %s", attempt, max_retries, func.__name__, e)
                    if attempt >= max_retries:
                        logger.error("All %s attempts failed for function %s", max_retries, func.__name__)
                        raise
                    time.sleep(delay)

        return wrapper

    return decorator
