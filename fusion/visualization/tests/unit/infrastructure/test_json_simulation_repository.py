"""Unit tests for JsonSimulationRepository."""

import json
from pathlib import Path
from typing import Any

import pytest

from fusion.visualization.domain.entities.run import Run
from fusion.visualization.domain.exceptions.domain_exceptions import (
    DataFormatError,
    RunDataNotFoundError,
)
from fusion.visualization.infrastructure.repositories import (
    FileMetadataRepository,
    JsonSimulationRepository,
)


class TestJsonSimulationRepository:
    """Test suite for JsonSimulationRepository."""

    @pytest.fixture
    def metadata_repository(self) -> FileMetadataRepository:
        """Create metadata repository for tests."""
        return FileMetadataRepository(cache_ttl_seconds=3600, enable_cache=True)

    @pytest.fixture
    def repository(
        self, tmp_path: Path, metadata_repository: FileMetadataRepository
    ) -> JsonSimulationRepository:
        """Create repository instance for tests."""
        return JsonSimulationRepository(
            base_path=tmp_path,
            metadata_repository=metadata_repository,
        )

    @pytest.fixture
    def setup_test_data(
        self, tmp_path: Path, sample_v1_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Set up test data directory structure."""
        # Create directory structure: {network}/{date}/{run_id}/s1/{traffic}_erlang.json
        network = "NSFNet"
        date = "0606"
        run_id = "timestamp123"

        run_path = tmp_path / network / date / run_id
        data_path = run_path / "s1"
        data_path.mkdir(parents=True)

        # Create metadata file
        metadata = {
            "path_algorithm": "ppo_obs_7",
            "run_id": run_id,
            "network": network,
            "obs_space": "obs_7",
        }
        with open(run_path / "metadata.json", "w") as f:
            json.dump(metadata, f)

        # Create data files for different traffic volumes
        for traffic_volume in [150, 200, 250]:
            data_file = data_path / f"{traffic_volume}_erlang.json"
            with open(data_file, "w") as f:
                json.dump(sample_v1_data, f)

        return {
            "network": network,
            "date": date,
            "run_id": run_id,
            "run_path": run_path,
            "traffic_volumes": [150, 200, 250],
        }

    def test_find_runs_success(
        self, repository: JsonSimulationRepository, setup_test_data: dict[str, Any]
    ) -> None:
        """Should find runs successfully."""
        data = setup_test_data
        runs = repository.find_runs(
            network=data["network"],
            dates=[data["date"]],
        )

        assert len(runs) == 1
        assert runs[0].id == data["run_id"]
        assert runs[0].network == data["network"]
        assert runs[0].date == data["date"]
        assert runs[0].algorithm == "ppo_obs_7"

    def test_find_runs_with_algorithm_filter(
        self, repository: JsonSimulationRepository, setup_test_data: dict[str, Any]
    ) -> None:
        """Should filter runs by algorithm."""
        data = setup_test_data

        # Should find the run
        runs = repository.find_runs(
            network=data["network"],
            dates=[data["date"]],
            algorithm="ppo_obs_7",
        )
        assert len(runs) == 1

        # Should not find the run (wrong algorithm)
        runs = repository.find_runs(
            network=data["network"],
            dates=[data["date"]],
            algorithm="dqn_obs_7",
        )
        assert len(runs) == 0

    def test_find_runs_with_run_ids_filter(
        self, repository: JsonSimulationRepository, setup_test_data: dict[str, Any]
    ) -> None:
        """Should filter runs by run IDs."""
        data = setup_test_data

        # Should find the run
        runs = repository.find_runs(
            network=data["network"],
            dates=[data["date"]],
            run_ids=[data["run_id"]],
        )
        assert len(runs) == 1

        # Should not find the run (wrong ID)
        runs = repository.find_runs(
            network=data["network"],
            dates=[data["date"]],
            run_ids=["nonexistent"],
        )
        assert len(runs) == 0

    def test_find_runs_nonexistent_date(
        self, repository: JsonSimulationRepository, tmp_path: Path
    ) -> None:
        """Should handle nonexistent date gracefully."""
        runs = repository.find_runs(
            network="NSFNet",
            dates=["9999"],
        )
        assert len(runs) == 0

    def test_get_run_data_success(
        self,
        repository: JsonSimulationRepository,
        setup_test_data: dict[str, Any],
        sample_v1_data: dict[str, Any],
    ) -> None:
        """Should load run data successfully."""
        data = setup_test_data
        run = Run(
            id=data["run_id"],
            network=data["network"],
            date=data["date"],
            algorithm="ppo_obs_7",
            path=data["run_path"],
        )

        canonical_data = repository.get_run_data(run, traffic_volume=150)

        assert canonical_data is not None
        assert canonical_data.blocking_probability == sample_v1_data["blocking_mean"]
        assert canonical_data.version == "v1"

    def test_get_run_data_uses_cache(
        self, repository: JsonSimulationRepository, setup_test_data: dict[str, Any]
    ) -> None:
        """Should use cache for repeated requests."""
        data = setup_test_data
        run = Run(
            id=data["run_id"],
            network=data["network"],
            date=data["date"],
            algorithm="ppo_obs_7",
            path=data["run_path"],
        )

        # First call - loads from file
        data1 = repository.get_run_data(run, traffic_volume=150)

        # Second call - should use cache
        data2 = repository.get_run_data(run, traffic_volume=150)

        assert data1 is data2  # Same object reference (cached)

    def test_get_run_data_not_found(
        self, repository: JsonSimulationRepository, setup_test_data: dict[str, Any]
    ) -> None:
        """Should raise error when data file not found."""
        data = setup_test_data
        run = Run(
            id=data["run_id"],
            network=data["network"],
            date=data["date"],
            algorithm="ppo_obs_7",
            path=data["run_path"],
        )

        with pytest.raises(RunDataNotFoundError):
            repository.get_run_data(run, traffic_volume=999)

    def test_get_run_data_invalid_json(
        self, repository: JsonSimulationRepository, setup_test_data: dict[str, Any]
    ) -> None:
        """Should raise error for invalid JSON."""
        data = setup_test_data
        run = Run(
            id=data["run_id"],
            network=data["network"],
            date=data["date"],
            algorithm="ppo_obs_7",
            path=data["run_path"],
        )

        # Create invalid JSON file
        invalid_file = data["run_path"] / "s1" / "999_erlang.json"
        with open(invalid_file, "w") as f:
            f.write("{ invalid json }")

        with pytest.raises(DataFormatError):
            repository.get_run_data(run, traffic_volume=999)

    def test_get_run_data_batch_success(
        self, repository: JsonSimulationRepository, setup_test_data: dict[str, Any]
    ) -> None:
        """Should load multiple traffic volumes efficiently."""
        data = setup_test_data
        run = Run(
            id=data["run_id"],
            network=data["network"],
            date=data["date"],
            algorithm="ppo_obs_7",
            path=data["run_path"],
        )

        result = repository.get_run_data_batch(
            run,
            traffic_volumes=[150, 200, 250],
        )

        assert len(result) == 3
        assert 150 in result
        assert 200 in result
        assert 250 in result

    def test_get_run_data_batch_partial(
        self, repository: JsonSimulationRepository, setup_test_data: dict[str, Any]
    ) -> None:
        """Should handle partial data availability gracefully."""
        data = setup_test_data
        run = Run(
            id=data["run_id"],
            network=data["network"],
            date=data["date"],
            algorithm="ppo_obs_7",
            path=data["run_path"],
        )

        # Request mix of existing and non-existing volumes
        result = repository.get_run_data_batch(
            run,
            traffic_volumes=[150, 200, 999],  # 999 doesn't exist
        )

        assert len(result) == 2
        assert 150 in result
        assert 200 in result
        assert 999 not in result

    def test_exists_run_directory(
        self, repository: JsonSimulationRepository, setup_test_data: dict[str, Any]
    ) -> None:
        """Should check if run directory exists."""
        data = setup_test_data
        run = Run(
            id=data["run_id"],
            network=data["network"],
            date=data["date"],
            algorithm="ppo_obs_7",
            path=data["run_path"],
        )

        assert repository.exists(run) is True

        # Non-existent run
        run_nonexistent = Run(
            id="nonexistent",
            network=data["network"],
            date=data["date"],
            algorithm="ppo_obs_7",
            path=data["run_path"].parent / "nonexistent",
        )
        assert repository.exists(run_nonexistent) is False

    def test_exists_specific_traffic_volume(
        self, repository: JsonSimulationRepository, setup_test_data: dict[str, Any]
    ) -> None:
        """Should check if specific traffic volume exists."""
        data = setup_test_data
        run = Run(
            id=data["run_id"],
            network=data["network"],
            date=data["date"],
            algorithm="ppo_obs_7",
            path=data["run_path"],
        )

        assert repository.exists(run, traffic_volume=150) is True
        assert repository.exists(run, traffic_volume=999) is False

    def test_get_available_traffic_volumes(
        self, repository: JsonSimulationRepository, setup_test_data: dict[str, Any]
    ) -> None:
        """Should return sorted list of available traffic volumes."""
        data = setup_test_data
        run = Run(
            id=data["run_id"],
            network=data["network"],
            date=data["date"],
            algorithm="ppo_obs_7",
            path=data["run_path"],
        )

        volumes = repository.get_available_traffic_volumes(run)

        assert volumes == [150, 200, 250]

    def test_get_metadata(
        self, repository: JsonSimulationRepository, setup_test_data: dict[str, Any]
    ) -> None:
        """Should load run metadata."""
        data = setup_test_data
        run = Run(
            id=data["run_id"],
            network=data["network"],
            date=data["date"],
            algorithm="ppo_obs_7",
            path=data["run_path"],
        )

        metadata = repository.get_metadata(run)

        assert metadata["path_algorithm"] == "ppo_obs_7"
        assert metadata["run_id"] == data["run_id"]

    def test_clear_cache(
        self, repository: JsonSimulationRepository, setup_test_data: dict[str, Any]
    ) -> None:
        """Should clear the data cache."""
        data = setup_test_data
        run = Run(
            id=data["run_id"],
            network=data["network"],
            date=data["date"],
            algorithm="ppo_obs_7",
            path=data["run_path"],
        )

        # Load some data to populate cache
        repository.get_run_data(run, traffic_volume=150)
        assert len(repository._data_cache) > 0

        # Clear cache
        repository.clear_cache()
        assert len(repository._data_cache) == 0

    def test_get_cache_stats(
        self, repository: JsonSimulationRepository, setup_test_data: dict[str, Any]
    ) -> None:
        """Should return cache statistics."""
        data = setup_test_data
        run = Run(
            id=data["run_id"],
            network=data["network"],
            date=data["date"],
            algorithm="ppo_obs_7",
            path=data["run_path"],
        )

        # Initially empty
        stats = repository.get_cache_stats()
        assert stats["cached_entries"] == 0

        # Load some data
        repository.get_run_data(run, traffic_volume=150)
        repository.get_run_data(run, traffic_volume=200)

        stats = repository.get_cache_stats()
        assert stats["cached_entries"] == 2
