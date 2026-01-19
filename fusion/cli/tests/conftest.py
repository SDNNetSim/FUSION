"""Shared fixtures and test utilities for CLI tests."""

from argparse import Namespace
from configparser import ConfigParser
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from fusion.configs.constants import DEFAULT_THREAD_NAME


@pytest.fixture
def mock_config_parser() -> MagicMock:
    """Provide a mock ConfigParser for testing configuration loading."""
    mock_parser = MagicMock(spec=ConfigParser)
    mock_parser.sections.return_value = ["sim"]
    mock_parser.has_section.return_value = True
    mock_parser.has_option.return_value = True
    mock_parser.__getitem__.return_value = {"test_option": "test_value"}
    mock_parser.items.return_value = [("test_key", "test_value")]
    return mock_parser


@pytest.fixture
def valid_config_dict() -> dict[str, Any]:
    """Provide a valid configuration dictionary for tests."""
    return {
        DEFAULT_THREAD_NAME: {
            "config_path": "/path/to/config.ini",
            "run_id": "test_run_001",
            "output_path": "/path/to/output",
            "simulation_time": 1000,
            "num_requests": 10,
        }
    }


@pytest.fixture
def sample_args() -> Namespace:
    """Provide sample command line arguments for tests."""
    return Namespace(
        config_path="test_config.ini",
        run_id="test_run",
        debug=False,
        verbose=False,
        agent_type="rl",
    )


@pytest.fixture
def sample_args_dict() -> dict[str, Any]:
    """Provide sample command line arguments as dictionary."""
    return {
        "config_path": "test_config.ini",
        "run_id": "test_run",
        "debug": False,
        "verbose": False,
        "output_path": "/test/output",
    }


@pytest.fixture
def mock_logger(monkeypatch: Any) -> MagicMock:
    """Provide a mock logger for testing."""
    mock_log = MagicMock()
    monkeypatch.setattr("fusion.utils.logging_config.get_logger", lambda _: mock_log)
    return mock_log


@pytest.fixture
def temp_config_file(tmp_path: Path) -> Path:
    """Create a temporary config file for testing."""
    config_content = """[general_settings]
run_id = test_run
output_path = /tmp/output
simulation_time = 1000
num_requests = 10
erlang_start = 0.1
erlang_stop = 1.0
erlang_step = 0.1
mod_assumption = static
mod_assumption_path = /test/path
holding_time = 1.0
thread_erlangs = true
guard_slots = 1
max_iters = 100
dynamic_lps = false
fixed_grid = true
pre_calc_mod_selection = false
max_segments = 5
route_method = dijkstra
allocation_method = first_fit
save_snapshots = false
snapshot_step = 10
print_step = 10
spectrum_priority = first
save_step = 10
save_start_end_slots = false

[topology_settings]
network = nsfnet
bw_per_slot = 25.0
cores_per_link = 1
const_link_weight = true
is_only_core_node = false
"""
    config_file = tmp_path / "test_config.ini"
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def invalid_config_file(tmp_path: Path) -> Path:
    """Create an invalid config file for testing error handling."""
    config_content = """[invalid_section]
some_option = value
"""
    config_file = tmp_path / "invalid_config.ini"
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def multi_thread_config_file(tmp_path: Path) -> Path:
    """Create a config file with multiple thread sections."""
    config_content = """[general_settings]
run_id = test_run
output_path = /tmp/output
simulation_time = 1000
num_requests = 10
erlang_start = 0.1
erlang_stop = 1.0
erlang_step = 0.1
mod_assumption = static
mod_assumption_path = /test/path
holding_time = 1.0
thread_erlangs = true
guard_slots = 1
max_iters = 100
dynamic_lps = false
fixed_grid = true
pre_calc_mod_selection = false
max_segments = 5
route_method = dijkstra
allocation_method = first_fit
save_snapshots = false
snapshot_step = 10
print_step = 10
spectrum_priority = first
save_step = 10
save_start_end_slots = false

[topology_settings]
network = nsfnet
bw_per_slot = 25.0
cores_per_link = 1
const_link_weight = true
is_only_core_node = false

[s2]
simulation_time = 2000
run_id = thread1_run

[s3]
simulation_time = 3000
run_id = thread2_run
"""
    config_file = tmp_path / "multi_thread_config.ini"
    config_file.write_text(config_content)
    return config_file
