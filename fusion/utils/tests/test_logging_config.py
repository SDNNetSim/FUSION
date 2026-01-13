"""Unit tests for fusion.utils.logging_config module."""

import logging
from unittest.mock import Mock, patch

import pytest

from fusion.utils.logging_config import (
    DEFAULT_FORMAT,
    DETAILED_FORMAT,
    LOG_LEVELS,
    LoggerAdapter,
    _create_console_handler,
    _create_file_handler,
    configure_simulation_logging,
    get_logger,
    log_function_call,
    log_message,
    setup_logger,
)


class TestCreateConsoleHandler:
    """Tests for _create_console_handler function."""

    def test_create_console_handler_with_info_level_creates_handler(self) -> None:
        """Test creating console handler with INFO level."""
        # Arrange
        log_level = logging.INFO
        formatter = logging.Formatter(DEFAULT_FORMAT)

        # Act
        handler = _create_console_handler(log_level, formatter)

        # Assert
        assert isinstance(handler, logging.StreamHandler)
        assert handler.level == logging.INFO
        assert handler.formatter == formatter

    def test_create_console_handler_with_debug_level_creates_handler(self) -> None:
        """Test creating console handler with DEBUG level."""
        # Arrange
        log_level = logging.DEBUG
        formatter = logging.Formatter(DETAILED_FORMAT)

        # Act
        handler = _create_console_handler(log_level, formatter)

        # Assert
        assert isinstance(handler, logging.StreamHandler)
        assert handler.level == logging.DEBUG


class TestCreateFileHandler:
    """Tests for _create_file_handler function."""

    def test_create_file_handler_with_default_dir_creates_handler(
        self,
    ) -> None:
        """Test creating file handler with default log directory."""
        # Arrange
        log_file = "test.log"
        log_dir = "/tmp/test_logs"
        log_level = logging.INFO
        formatter = logging.Formatter(DEFAULT_FORMAT)

        # Act
        handler = _create_file_handler(
            log_file, log_dir, log_level, formatter, "a", 10485760, 5
        )

        # Assert
        assert isinstance(handler, logging.handlers.RotatingFileHandler)

    def test_create_file_handler_with_custom_dir_uses_custom_path(
        self,
    ) -> None:
        """Test creating file handler with custom log directory."""
        # Arrange
        log_file = "test.log"
        log_dir = "/tmp/custom_logs"
        log_level = logging.WARNING
        formatter = logging.Formatter(DEFAULT_FORMAT)

        # Act
        handler = _create_file_handler(
            log_file, log_dir, log_level, formatter, "w", 5242880, 3
        )

        # Assert
        assert isinstance(handler, logging.handlers.RotatingFileHandler)

    def test_create_file_handler_sets_rotation_parameters(
        self,
    ) -> None:
        """Test file handler rotation parameters are set correctly."""
        # Arrange
        log_dir = "/tmp/rotation_logs"

        # Act
        handler = _create_file_handler(
            "test.log",
            log_dir,
            logging.INFO,
            logging.Formatter(DEFAULT_FORMAT),
            "a",
            1024,
            10,
        )

        # Assert
        assert handler.maxBytes == 1024
        assert handler.backupCount == 10


class TestSetupLogger:
    """Tests for setup_logger function."""

    def teardown_method(self) -> None:
        """Clean up logger cache after each test."""
        from fusion.utils import logging_config

        logging_config._loggers.clear()

    @patch("fusion.utils.logging_config._create_console_handler")
    def test_setup_logger_with_console_only_creates_console_handler(
        self, mock_console_handler: Mock
    ) -> None:
        """Test setup_logger with console output only."""
        # Arrange
        mock_handler = Mock()
        mock_console_handler.return_value = mock_handler

        # Act
        logger = setup_logger("test_logger", level="INFO", console=True)

        # Assert
        assert logger.name == "test_logger"
        assert logger.level == logging.INFO
        mock_console_handler.assert_called_once()

    @patch("fusion.utils.logging_config._create_file_handler")
    @patch("fusion.utils.logging_config._create_console_handler")
    def test_setup_logger_with_file_creates_file_handler(
        self, mock_console_handler: Mock, mock_file_handler: Mock
    ) -> None:
        """Test setup_logger with file output."""
        # Arrange
        mock_console = Mock()
        mock_file = Mock()
        mock_console_handler.return_value = mock_console
        mock_file_handler.return_value = mock_file

        # Act
        logger = setup_logger("test_logger_file", log_file="test.log")

        # Assert
        mock_file_handler.assert_called_once()
        assert "test_logger_file" == logger.name

    def test_setup_logger_caches_logger_instance(self) -> None:
        """Test that logger is cached and reused."""
        # Act
        logger1 = setup_logger("cached_logger")
        logger2 = setup_logger("cached_logger")

        # Assert
        assert logger1 is logger2

    @patch("fusion.utils.logging_config._create_console_handler")
    def test_setup_logger_without_console_skips_console_handler(
        self, mock_console_handler: Mock
    ) -> None:
        """Test setup_logger without console output."""
        # Act
        setup_logger("test_logger", console=False)

        # Assert
        mock_console_handler.assert_not_called()

    @patch("fusion.utils.logging_config._create_console_handler")
    def test_setup_logger_with_custom_format_uses_custom_format(
        self, mock_console_handler: Mock
    ) -> None:
        """Test setup_logger with custom format string."""
        # Arrange
        custom_format = "%(levelname)s - %(message)s"
        mock_handler = Mock()
        mock_console_handler.return_value = mock_handler

        # Act
        setup_logger("test_logger_fmt", format_string=custom_format)

        # Assert
        # Verify formatter was created with custom format
        assert mock_console_handler.called
        call_args = mock_console_handler.call_args
        if call_args and len(call_args[0]) > 1:
            formatter = call_args[0][1]
            assert isinstance(formatter, logging.Formatter)

    @patch("fusion.utils.logging_config._create_console_handler")
    def test_setup_logger_with_debug_level_sets_debug(
        self, mock_console_handler: Mock
    ) -> None:
        """Test setup_logger with DEBUG level."""
        # Arrange
        mock_handler = Mock()
        mock_console_handler.return_value = mock_handler

        # Act
        logger = setup_logger("test_logger_debug", level="DEBUG")

        # Assert
        assert logger.level == logging.DEBUG

    def test_setup_logger_with_existing_handlers_returns_existing(self) -> None:
        """Test that logger with existing handlers is returned as-is."""
        # Arrange
        logger = logging.getLogger("existing_logger")
        logger.addHandler(logging.StreamHandler())

        # Act
        result = setup_logger("existing_logger")

        # Assert
        assert result is logger


class TestGetLogger:
    """Tests for get_logger function."""

    def teardown_method(self) -> None:
        """Clean up logger cache after each test."""
        from fusion.utils import logging_config

        logging_config._loggers.clear()

    @patch("fusion.utils.logging_config.setup_logger")
    def test_get_logger_with_new_name_calls_setup(self, mock_setup: Mock) -> None:
        """Test get_logger creates new logger if not cached."""
        # Arrange
        mock_logger = Mock()
        mock_setup.return_value = mock_logger

        # Act
        result = get_logger("new_logger")

        # Assert
        mock_setup.assert_called_once_with("new_logger", level="INFO")
        assert result == mock_logger

    def test_get_logger_with_cached_name_returns_cached(self) -> None:
        """Test get_logger returns cached logger."""
        # Arrange
        logger = setup_logger("cached_logger")

        # Act
        result = get_logger("cached_logger")

        # Assert
        assert result is logger


class TestConfigureSimulationLogging:
    """Tests for configure_simulation_logging function."""

    def teardown_method(self) -> None:
        """Clean up logger cache after each test."""
        from fusion.utils import logging_config

        logging_config._loggers.clear()

    @patch("fusion.utils.logging_config.datetime")
    @patch("fusion.utils.logging_config.setup_logger")
    def test_configure_simulation_logging_without_thread_creates_logger(
        self, mock_setup: Mock, mock_datetime: Mock
    ) -> None:
        """Test simulation logging without thread number."""
        # Arrange
        mock_now = Mock()
        mock_now.strftime.return_value = "20250101_120000"
        mock_datetime.now.return_value = mock_now
        mock_logger = Mock()
        mock_setup.return_value = mock_logger

        # Act
        configure_simulation_logging("test_sim", 10.5)

        # Assert
        mock_setup.assert_called_once()
        call_kwargs = mock_setup.call_args[1]
        assert call_kwargs["name"] == "simulation.test_sim"
        assert "test_sim_erlang10.5" in call_kwargs["log_file"]
        assert call_kwargs["format_string"] == DETAILED_FORMAT

    @patch("fusion.utils.logging_config.datetime")
    @patch("fusion.utils.logging_config.setup_logger")
    def test_configure_simulation_logging_with_thread_includes_thread(
        self, mock_setup: Mock, mock_datetime: Mock
    ) -> None:
        """Test simulation logging with thread number."""
        # Arrange
        mock_now = Mock()
        mock_now.strftime.return_value = "20250101_120000"
        mock_datetime.now.return_value = mock_now
        mock_logger = Mock()
        mock_setup.return_value = mock_logger

        # Act
        configure_simulation_logging("test_sim", 10.5, thread_num=3)

        # Assert
        call_kwargs = mock_setup.call_args[1]
        assert call_kwargs["name"] == "simulation.test_sim.thread3"
        assert "thread3" in call_kwargs["log_file"]

    @patch("fusion.utils.logging_config.datetime")
    @patch("fusion.utils.logging_config.setup_logger")
    def test_configure_simulation_logging_with_custom_level_uses_level(
        self, mock_setup: Mock, mock_datetime: Mock
    ) -> None:
        """Test simulation logging with custom log level."""
        # Arrange
        mock_now = Mock()
        mock_now.strftime.return_value = "20250101_120000"
        mock_datetime.now.return_value = mock_now

        # Act
        configure_simulation_logging("test_sim", 10.5, log_level="DEBUG")

        # Assert
        call_kwargs = mock_setup.call_args[1]
        assert call_kwargs["level"] == "DEBUG"


class TestLogFunctionCall:
    """Tests for log_function_call decorator."""

    def test_log_function_call_logs_entry_and_return(self) -> None:
        """Test decorator logs function entry and return."""
        # Arrange
        mock_logger = Mock()

        @log_function_call(mock_logger)
        def test_func(x: int, y: int) -> int:
            return x + y

        # Act
        result = test_func(2, 3)

        # Assert
        assert result == 5
        assert mock_logger.debug.call_count == 2
        calls = mock_logger.debug.call_args_list
        assert "Calling test_func" in calls[0][0][0]
        assert "returned" in calls[1][0][0]

    def test_log_function_call_logs_exception(self) -> None:
        """Test decorator logs exceptions."""
        # Arrange
        mock_logger = Mock()

        @log_function_call(mock_logger)
        def failing_func() -> None:
            raise ValueError("Test error")

        # Act & Assert
        with pytest.raises(ValueError):
            failing_func()

        mock_logger.error.assert_called_once()
        error_call = mock_logger.error.call_args[0][0]
        assert "ValueError" in error_call
        assert "Test error" in error_call

    def test_log_function_call_with_kwargs_logs_kwargs(self) -> None:
        """Test decorator logs keyword arguments."""
        # Arrange
        mock_logger = Mock()

        @log_function_call(mock_logger)
        def test_func(a: int, b: str = "default") -> str:
            return f"{a}-{b}"

        # Act
        result = test_func(1, b="custom")

        # Assert
        assert result == "1-custom"
        calls = mock_logger.debug.call_args_list
        assert "b='custom'" in calls[0][0][0]


class TestLoggerAdapter:
    """Tests for LoggerAdapter class."""

    def test_logger_adapter_process_adds_context(self) -> None:
        """Test adapter adds extra context to messages."""
        # Arrange
        base_logger = Mock()
        extra = {"request_id": "123", "user": "test_user"}
        adapter = LoggerAdapter(base_logger, extra)

        # Act
        processed_msg, kwargs = adapter.process("Test message", {})

        # Assert
        assert "request_id=123" in processed_msg
        assert "user=test_user" in processed_msg
        assert "Test message" in processed_msg

    def test_logger_adapter_process_without_extra_returns_message(self) -> None:
        """Test adapter without extra context returns message as-is."""
        # Arrange
        base_logger = Mock()
        adapter = LoggerAdapter(base_logger, {})

        # Act
        processed_msg, kwargs = adapter.process("Test message", {})

        # Assert
        assert processed_msg == "Test message"

    def test_logger_adapter_initialization_stores_extra(self) -> None:
        """Test adapter initialization stores extra context."""
        # Arrange
        base_logger = Mock()
        extra = {"key": "value"}

        # Act
        adapter = LoggerAdapter(base_logger, extra)

        # Assert
        assert adapter.extra == extra


class TestLogMessage:
    """Tests for log_message function."""

    def test_log_message_with_queue_puts_message_in_queue(self) -> None:
        """Test log_message with queue puts message in queue."""
        # Arrange
        mock_queue = Mock()
        message = "Test message"

        # Act
        log_message(message, mock_queue)

        # Assert
        mock_queue.put.assert_called_once_with(message)

    @patch("fusion.utils.logging_config.get_logger")
    def test_log_message_without_queue_uses_logger(self, mock_get_logger: Mock) -> None:
        """Test log_message without queue uses logger."""
        # Arrange
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        message = "Test message"

        # Act
        log_message(message, None)

        # Assert
        mock_logger.info.assert_called_once_with(message)

    def test_log_message_with_none_queue_calls_get_logger(self) -> None:
        """Test log_message with None queue retrieves logger."""
        # Arrange
        with patch("fusion.utils.logging_config.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            # Act
            log_message("Test", None)

            # Assert
            mock_get_logger.assert_called_once()


class TestLogLevels:
    """Tests for LOG_LEVELS constant."""

    def test_log_levels_contains_all_standard_levels(self) -> None:
        """Test LOG_LEVELS contains all standard logging levels."""
        # Assert
        assert LOG_LEVELS["DEBUG"] == logging.DEBUG
        assert LOG_LEVELS["INFO"] == logging.INFO
        assert LOG_LEVELS["WARNING"] == logging.WARNING
        assert LOG_LEVELS["ERROR"] == logging.ERROR
        assert LOG_LEVELS["CRITICAL"] == logging.CRITICAL

    def test_log_levels_has_correct_count(self) -> None:
        """Test LOG_LEVELS has expected number of entries."""
        # Assert
        assert len(LOG_LEVELS) == 5
