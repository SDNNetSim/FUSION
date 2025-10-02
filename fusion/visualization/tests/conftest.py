"""Pytest configuration and shared fixtures for visualization tests."""

import json
from pathlib import Path
from typing import Dict, Any
import pytest
import numpy as np

from fusion.visualization.domain.entities import (
    Run,
    MetricDefinition,
    PlotConfiguration,
    Plot,
)
from fusion.visualization.domain.value_objects import (
    PlotId,
    DataVersion,
    MetricValue,
    DataType,
    PlotType,
)
from fusion.visualization.infrastructure.adapters import (
    V1DataAdapter,
    V2DataAdapter,
    DataAdapterRegistry,
)


@pytest.fixture
def sample_v1_data() -> Dict[str, Any]:
    """Sample V1 format data."""
    return {
        "blocking_mean": 0.045,
        "iter_stats": {
            "0": {
                "sim_block_list": [0.05, 0.04, 0.045, 0.043],
                "hops_mean": 3.2,
                "hops_list": [3, 3, 4, 3, 3],
                "lengths_mean": 450.5,
                "lengths_list": [450, 460, 440, 455],
                "snapshots_dict": {},
                "mods_used_dict": {},
            },
            "1": {
                "sim_block_list": [0.046, 0.044, 0.045, 0.044],
                "hops_mean": 3.3,
                "hops_list": [3, 4, 3, 3, 3],
                "lengths_mean": 455.0,
                "lengths_list": [455, 465, 445, 450],
                "snapshots_dict": {},
                "mods_used_dict": {},
            },
        },
        "sim_start_time": "0429_21_14_39_491949",
        "sim_end_time": "0429_21_16_42_123456",
    }


@pytest.fixture
def sample_v2_data() -> Dict[str, Any]:
    """Sample V2 format data."""
    return {
        "version": "v2",
        "metrics": {
            "blocking_probability": 0.045,
        },
        "iterations": [
            {
                "iteration": 0,
                "sim_block_list": [0.05, 0.04, 0.045, 0.043],
                "hops_mean": 3.2,
                "hops_list": [3, 3, 4, 3, 3],
                "lengths_mean": 450.5,
                "lengths_list": [450, 460, 440, 455],
                "metadata": {},
            },
            {
                "iteration": 1,
                "sim_block_list": [0.046, 0.044, 0.045, 0.044],
                "hops_mean": 3.3,
                "hops_list": [3, 4, 3, 3, 3],
                "lengths_mean": 455.0,
                "lengths_list": [455, 465, 445, 450],
                "metadata": {},
            },
        ],
        "timing": {
            "start_time": "2024-04-29T21:14:39.491949",
            "end_time": "2024-04-29T21:16:42.123456",
            "duration_seconds": 122.631507,
        },
        "metadata": {
            "network": "NSFNet",
            "algorithm": "ppo_obs_7",
        },
    }


@pytest.fixture
def sample_run(tmp_path: Path) -> Run:
    """Sample Run entity."""
    run_path = tmp_path / "NSFNet" / "0606" / "timestamp123" / "s1"
    run_path.mkdir(parents=True, exist_ok=True)

    return Run(
        id="timestamp123",
        network="NSFNet",
        date="0606",
        algorithm="ppo_obs_7",
        path=run_path.parent,
        metadata={"run_id": "abc123", "seed": 1},
    )


@pytest.fixture
def sample_metric_definition() -> MetricDefinition:
    """Sample MetricDefinition."""
    return MetricDefinition(
        name="blocking_probability",
        data_type=DataType.FLOAT,
        source_path="$.blocking_mean",
        unit="probability",
        description="Network blocking probability",
    )


@pytest.fixture
def sample_plot_configuration() -> PlotConfiguration:
    """Sample PlotConfiguration."""
    return PlotConfiguration(
        plot_type=PlotType.LINE,
        metrics=["blocking_probability"],
        algorithms=["ppo_obs_7", "dqn_obs_7"],
        traffic_volumes=[600, 700, 800, 900, 1000],
        title="Blocking Probability vs Traffic Volume",
        x_label="Traffic Volume (Erlang)",
        y_label="Blocking Probability",
        include_ci=True,
    )


@pytest.fixture
def sample_plot(sample_plot_configuration: PlotConfiguration) -> Plot:
    """Sample Plot entity."""
    return Plot(
        id=PlotId.generate(),
        title="Blocking Probability vs Traffic Volume",
        configuration=sample_plot_configuration,
    )


@pytest.fixture
def v1_adapter() -> V1DataAdapter:
    """V1 data adapter instance."""
    return V1DataAdapter()


@pytest.fixture
def v2_adapter() -> V2DataAdapter:
    """V2 data adapter instance."""
    return V2DataAdapter()


@pytest.fixture
def adapter_registry() -> DataAdapterRegistry:
    """Data adapter registry instance."""
    return DataAdapterRegistry()


@pytest.fixture
def sample_metric_values() -> list[MetricValue]:
    """Sample MetricValue list for aggregation tests."""
    return [
        MetricValue(value=0.045, data_type=DataType.FLOAT, unit="probability"),
        MetricValue(value=0.042, data_type=DataType.FLOAT, unit="probability"),
        MetricValue(value=0.048, data_type=DataType.FLOAT, unit="probability"),
        MetricValue(value=0.044, data_type=DataType.FLOAT, unit="probability"),
        MetricValue(value=0.046, data_type=DataType.FLOAT, unit="probability"),
    ]


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_data_file(tmp_path: Path, sample_v1_data: Dict[str, Any]) -> Path:
    """Create a sample data file."""
    data_file = tmp_path / "150_erlang.json"
    with open(data_file, 'w') as f:
        json.dump(sample_v1_data, f)
    return data_file
