"""Integration test fixtures."""

import json
from pathlib import Path

import pytest

from fusion.visualization.application.services import (
    CacheService,
    PlotService,
    ValidationService,
)
from fusion.visualization.infrastructure.adapters import DataAdapterRegistry
from fusion.visualization.infrastructure.processors import (
    BlockingProcessor,
)
from fusion.visualization.infrastructure.renderers import MatplotlibRenderer
from fusion.visualization.infrastructure.repositories import (
    FileMetadataRepository,
    JsonSimulationRepository,
)


@pytest.fixture
def integration_data_dir(tmp_path: Path) -> Path:
    """
    Create a realistic data directory structure for integration tests.

    Structure:
    tmp_path/
    ├── NSFNet/
    │   └── 0606/
    │       ├── 1715_12_30_45_123456/
    │       │   ├── s1/
    │       │   │   ├── 600_erlang.json
    │       │   │   ├── 700_erlang.json
    │       │   │   └── 800_erlang.json
    │       │   └── metadata.json
    │       └── 1715_12_35_20_654321/
    │           ├── s2/
    │           │   ├── 600_erlang.json
    │           │   ├── 700_erlang.json
    │           │   └── 800_erlang.json
    │           └── metadata.json
    """
    # Create directory structure
    network = "NSFNet"
    date = "0606"

    # Run 1: PPO algorithm
    run1_id = "1715_12_30_45_123456"
    run1_path = tmp_path / network / date / run1_id / "s1"
    run1_path.mkdir(parents=True)

    # Create data files for run 1
    for erlang in [600, 700, 800]:
        data = {
            "blocking_mean": 0.02 + (erlang - 600) * 0.01,  # Increases with traffic
            "iter_stats": {
                "0": {
                    "sim_block_list": [0.02, 0.021, 0.019, 0.02],
                    "hops_mean": 3.2,
                    "hops_list": [3, 3, 4, 3, 3],
                    "lengths_mean": 450.5,
                    "lengths_list": [450, 460, 440, 455],
                    "computation_time_mean": 0.015,
                },
                "1": {
                    "sim_block_list": [0.021, 0.022, 0.020, 0.021],
                    "hops_mean": 3.3,
                    "hops_list": [3, 4, 3, 3, 3],
                    "lengths_mean": 455.0,
                    "lengths_list": [455, 465, 445, 450],
                    "computation_time_mean": 0.016,
                },
            },
            "sim_start_time": "0606_12_30_45_123456",
            "sim_end_time": "0606_12_35_20_654321",
        }
        data["blocking_mean"] = 0.02 + (erlang - 600) * 0.01
        file_path = run1_path / f"{erlang}_erlang.json"
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    # Create metadata for run 1
    metadata1 = {
        "run_id": "ppo_run1",
        "path_algorithm": "ppo",
        "obs_space": "obs_7",
        "network": network,
        "date": date,
        "seed": 1,
    }
    with open(run1_path.parent / "metadata.json", "w") as f:
        json.dump(metadata1, f, indent=2)

    # Run 2: DQN algorithm
    run2_id = "1715_12_35_20_654321"
    run2_path = tmp_path / network / date / run2_id / "s2"
    run2_path.mkdir(parents=True)

    # Create data files for run 2 (slightly different performance)
    for erlang in [600, 700, 800]:
        data = {
            "blocking_mean": 0.025 + (erlang - 600) * 0.012,  # Slightly worse than PPO
            "iter_stats": {
                "0": {
                    "sim_block_list": [0.025, 0.026, 0.024, 0.025],
                    "hops_mean": 3.4,
                    "hops_list": [3, 4, 3, 4, 3],
                    "lengths_mean": 460.5,
                    "lengths_list": [460, 470, 450, 465],
                    "computation_time_mean": 0.018,
                },
                "1": {
                    "sim_block_list": [0.026, 0.027, 0.025, 0.026],
                    "hops_mean": 3.5,
                    "hops_list": [4, 4, 3, 3, 4],
                    "lengths_mean": 465.0,
                    "lengths_list": [465, 475, 455, 460],
                    "computation_time_mean": 0.019,
                },
            },
            "sim_start_time": "0606_12_35_20_654321",
            "sim_end_time": "0606_12_40_15_987654",
        }
        file_path = run2_path / f"{erlang}_erlang.json"
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    # Create metadata for run 2
    metadata2 = {
        "run_id": "dqn_run1",
        "path_algorithm": "dqn",
        "obs_space": "obs_7",
        "network": network,
        "date": date,
        "seed": 2,
    }
    with open(run2_path.parent / "metadata.json", "w") as f:
        json.dump(metadata2, f, indent=2)

    return tmp_path


@pytest.fixture
def integration_data_dir_v2(tmp_path: Path) -> Path:
    """
    Create a realistic data directory with V2 format data.
    """
    network = "USNet"
    date = "0611"

    run1_id = "1715_13_30_45_123456"
    run1_path = tmp_path / network / date / run1_id / "s1"
    run1_path.mkdir(parents=True)

    # Create V2 format data files
    for erlang in [600, 700, 800]:
        data = {
            "version": "v2",
            "metrics": {
                "blocking_probability": 0.018 + (erlang - 600) * 0.009,
            },
            "iterations": [
                {
                    "iteration": 0,
                    "sim_block_list": [0.018, 0.019, 0.017, 0.018],
                    "hops_mean": 3.1,
                    "hops_list": [3, 3, 3, 3, 3],
                    "lengths_mean": 440.5,
                    "lengths_list": [440, 450, 430, 445],
                    "computation_time_mean": 0.014,
                    "metadata": {},
                },
                {
                    "iteration": 1,
                    "sim_block_list": [0.019, 0.020, 0.018, 0.019],
                    "hops_mean": 3.2,
                    "hops_list": [3, 3, 4, 3, 3],
                    "lengths_mean": 445.0,
                    "lengths_list": [445, 455, 435, 440],
                    "computation_time_mean": 0.015,
                    "metadata": {},
                },
            ],
            "timing": {
                "start_time": "2024-06-11T13:30:45.123456",
                "end_time": "2024-06-11T13:35:20.654321",
                "duration_seconds": 275.530865,
            },
            "metadata": {
                "network": network,
                "algorithm": "ppo_obs_7",
                "seed": 3,
            },
        }
        file_path = run1_path / f"{erlang}_erlang.json"
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    # Create metadata
    metadata = {
        "run_id": "ppo_run2_v2",
        "path_algorithm": "ppo",
        "obs_space": "obs_7",
        "network": network,
        "date": date,
        "seed": 3,
    }
    with open(run1_path.parent / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return tmp_path


@pytest.fixture
def simulation_repository(integration_data_dir: Path) -> JsonSimulationRepository:
    """Simulation repository with test data."""
    adapter_registry = DataAdapterRegistry()
    return JsonSimulationRepository(
        base_path=integration_data_dir,
        adapter_registry=adapter_registry,
    )


@pytest.fixture
def metadata_repository(integration_data_dir: Path) -> FileMetadataRepository:
    """Metadata repository with test data."""
    return FileMetadataRepository(base_path=integration_data_dir)


@pytest.fixture
def cache_service(tmp_path: Path) -> CacheService:
    """Cache service for integration tests."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return CacheService(cache_dir=cache_dir)


@pytest.fixture
def plot_service(
    simulation_repository: JsonSimulationRepository,
    metadata_repository: FileMetadataRepository,
    cache_service: CacheService,
) -> PlotService:
    """Fully configured plot service for integration tests."""
    return PlotService(
        simulation_repository=simulation_repository,
        metadata_repository=metadata_repository,
        cache_service=cache_service,
    )


@pytest.fixture
def blocking_processor() -> BlockingProcessor:
    """Blocking probability processor."""
    return BlockingProcessor()


@pytest.fixture
def matplotlib_renderer(tmp_path: Path) -> MatplotlibRenderer:
    """Matplotlib renderer for integration tests."""
    output_dir = tmp_path / "plots"
    output_dir.mkdir()
    return MatplotlibRenderer(output_dir=output_dir)


@pytest.fixture
def validation_service() -> ValidationService:
    """Validation service for integration tests."""
    return ValidationService()
