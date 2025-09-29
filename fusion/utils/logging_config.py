"""
Centralized logging configuration for FUSION.

This module provides standardized logging setup for all FUSION components.
"""

import logging
import logging.handlers
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from fusion.utils.os import find_project_root

# Default log format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DETAILED_FORMAT = (
    "%(asctime)s - %(name)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s"
)

# Log levels mapping
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# Global logger cache
_loggers: dict[str, logging.Logger] = {}


def _create_console_handler(
    log_level: int, formatter: logging.Formatter
) -> logging.StreamHandler:
    """
    Create and configure console handler.


    :param log_level: Logging level for the handler
    :type log_level: int
    :param formatter: Formatter for log messages
    :type formatter: logging.Formatter
    :return: Configured console handler
    :rtype: logging.StreamHandler
    """
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    return console_handler


def _create_file_handler(
    log_file: str,
    log_dir: str | None,
    log_level: int,
    formatter: logging.Formatter,
    file_mode: str,
    max_bytes: int,
    backup_count: int,
) -> logging.handlers.RotatingFileHandler:
    """
    Create and configure rotating file handler.


    :param log_file: Name of the log file
    :type log_file: str
    :param log_dir: Directory for log files, defaults to logs/ in project root
    :type log_dir: str | None
    :param log_level: Logging level for the handler
    :type log_level: int
    :param formatter: Formatter for log messages
    :type formatter: logging.Formatter
    :param file_mode: File open mode ('a' for append, 'w' for overwrite)
    :type file_mode: str
    :param max_bytes: Maximum size of log file before rotation
    :type max_bytes: int
    :param backup_count: Number of backup files to keep
    :type backup_count: int
    :return: Configured rotating file handler
    :rtype: logging.handlers.RotatingFileHandler
    """
    if log_dir is None:
        # Default to logs directory in project root
        project_root = Path(find_project_root())
        log_dir_path = project_root / "logs"
    else:
        log_dir_path = Path(log_dir)

    # Create log directory if it doesn't exist
    log_dir_path.mkdir(parents=True, exist_ok=True)

    log_path = log_dir_path / log_file

    # Use rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_path, mode=file_mode, maxBytes=max_bytes, backupCount=backup_count
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    return file_handler


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: str | None = None,
    log_dir: str | None = None,
    console: bool = True,
    file_mode: str = "a",
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5,
    format_string: str | None = None,
) -> logging.Logger:
    """
    Set up a standardized logger for FUSION modules.

    Creates a logger with optional file and console handlers. File handlers
    use rotation to prevent unbounded growth.

    :param name: Logger name (typically __name__ of the calling module)
    :param level: Logging level as string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    :param log_file: Optional log file name (created in log_dir)
    :param log_dir: Directory for log files (defaults to logs/ in project root)
    :param console: Whether to output to console
    :param file_mode: File open mode ('a' for append, 'w' for overwrite)
    :param max_bytes: Maximum size of log file before rotation
    :param backup_count: Number of backup files to keep
    :param format_string: Custom format string (uses DEFAULT_FORMAT if None)
    :return: Configured logger instance
    """
    # Check if logger already exists
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Set log level
    log_level = LOG_LEVELS.get(level.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Use provided format or default
    formatter = logging.Formatter(format_string or DEFAULT_FORMAT)

    # Console handler
    if console:
        console_handler = _create_console_handler(log_level, formatter)
        logger.addHandler(console_handler)

    # File handler with rotation
    if log_file:
        file_handler = _create_file_handler(
            log_file, log_dir, log_level, formatter, file_mode, max_bytes, backup_count
        )
        logger.addHandler(file_handler)

    # Cache the logger
    _loggers[name] = logger

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger or create a basic one.

    This is a convenience function for modules that need a logger but don't
    require special configuration.

    :param name: Logger name (typically __name__)
    :return: Logger instance
    """
    if name in _loggers:
        return _loggers[name]

    return setup_logger(name)


def configure_simulation_logging(
    sim_name: str, erlang: float, thread_num: int | None = None, log_level: str = "INFO"
) -> logging.Logger:
    """
    Configure logging specifically for simulation runs.

    Creates a logger with both console and file output, with the file name
    based on simulation parameters.

    :param sim_name: Name of the simulation
    :param erlang: Erlang value being simulated
    :param thread_num: Optional thread number for parallel runs
    :param log_level: Logging level
    :return: Configured logger for the simulation
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build log file name
    if thread_num is not None:
        log_file = f"{sim_name}_erlang{erlang}_thread{thread_num}_{timestamp}.log"
        logger_name = f"simulation.{sim_name}.thread{thread_num}"
    else:
        log_file = f"{sim_name}_erlang{erlang}_{timestamp}.log"
        logger_name = f"simulation.{sim_name}"

    return setup_logger(
        name=logger_name,
        level=log_level,
        log_file=log_file,
        format_string=DETAILED_FORMAT,
    )


# Type variable for decorator
T = TypeVar("T", bound=Callable[..., Any])


def log_function_call(logger: logging.Logger) -> Callable[[T], T]:
    """
    Decorator to log function calls with arguments and return values.

    Useful for debugging complex function flows.

    :param logger: Logger instance to use
    """

    def decorator(func: T) -> T:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Log function entry
            args_repr = [repr(a) for a in args]
            kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
            signature = ", ".join(args_repr + kwargs_repr)
            logger.debug(f"Calling {func.__name__}({signature})")

            try:
                # Call function
                result = func(*args, **kwargs)
                # Log successful return
                logger.debug(f"{func.__name__} returned {result!r}")
                return result
            except Exception as e:
                # Log exception
                logger.error(f"{func.__name__} raised {e.__class__.__name__}: {e}")
                raise

        return wrapper  # type: ignore[return-value]

    return decorator


class LoggerAdapter(logging.LoggerAdapter):
    """
    Custom logger adapter for adding contextual information.

    Useful for adding consistent context (like request IDs, user IDs, etc.)
    to all log messages from a specific component.
    """

    def __init__(self, logger: logging.Logger, extra: dict[str, Any]):
        """
        Initialize adapter with extra context.

        :param logger: Base logger
        :param extra: Dictionary of extra context to add to all messages
        """
        super().__init__(logger, extra)

    def process(self, msg: Any, kwargs: Any) -> tuple[str, Any]:
        """
        Add extra context to log messages.


        :param msg: The log message
        :type msg: Any
        :param kwargs: Additional keyword arguments
        :type kwargs: Any
        :return: Processed message and kwargs
        :rtype: tuple[str, Any]
        """
        # Add extra context to the message
        if self.extra:
            extra_str = " - ".join([f"{k}={v}" for k, v in self.extra.items()])
            return f"[{extra_str}] {msg}", kwargs
        else:
            return str(msg), kwargs
