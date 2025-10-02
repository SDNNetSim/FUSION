"""Unit tests for FileMetadataRepository."""

import json
import pytest
import time
from pathlib import Path
from typing import Dict, Any

from fusion.visualization.infrastructure.repositories import (
    FileMetadataRepository,
)
from fusion.visualization.domain.exceptions.domain_exceptions import (
    RepositoryError,
    MetadataNotFoundError,
)


class TestFileMetadataRepository:
    """Test suite for FileMetadataRepository."""

    @pytest.fixture
    def repository(self) -> FileMetadataRepository:
        """Create repository instance for tests."""
        return FileMetadataRepository(
            cache_ttl_seconds=1,  # Short TTL for testing
            enable_cache=True,
        )

    @pytest.fixture
    def setup_test_structure(self, tmp_path: Path) -> Dict[str, Any]:
        """Set up test directory structure with metadata."""
        network = "NSFNet"
        date = "0606"

        # Create multiple runs
        runs_data = []
        for i, algorithm in enumerate(["ppo_obs_7", "dqn_obs_7", "k_shortest_path_4"]):
            run_id = f"run_{i}"
            run_path = tmp_path / network / date / run_id
            run_path.mkdir(parents=True)

            metadata = {
                "path_algorithm": algorithm,
                "run_id": run_id,
                "network": network,
                "seed": i,
            }

            with open(run_path / "metadata.json", 'w') as f:
                json.dump(metadata, f)

            runs_data.append({
                "id": run_id,
                "path": run_path,
                "metadata": metadata,
            })

        return {
            "base_path": tmp_path,
            "network": network,
            "date": date,
            "runs": runs_data,
        }

    def test_discover_runs_success(self, repository: FileMetadataRepository, setup_test_structure: Dict[str, Any]) -> None:
        """Should discover all runs successfully."""
        data = setup_test_structure

        runs = repository.discover_runs(
            base_path=data["base_path"],
            network=data["network"],
            dates=[data["date"]],
        )

        assert len(runs) == 3
        algorithms = [r["algorithm"] for r in runs]
        assert "ppo_obs_7" in algorithms
        assert "dqn_obs_7" in algorithms
        assert "k_shortest_path_4" in algorithms

    def test_discover_runs_multiple_dates(self, repository: FileMetadataRepository, tmp_path: Path) -> None:
        """Should discover runs across multiple dates."""
        network = "NSFNet"

        # Create runs on different dates
        for date in ["0606", "0611"]:
            run_path = tmp_path / network / date / "run1"
            run_path.mkdir(parents=True)

            metadata = {"path_algorithm": "ppo_obs_7", "run_id": "run1"}
            with open(run_path / "metadata.json", 'w') as f:
                json.dump(metadata, f)

        runs = repository.discover_runs(
            base_path=tmp_path,
            network=network,
            dates=["0606", "0611"],
        )

        assert len(runs) == 2

    def test_discover_runs_nonexistent_date(self, repository: FileMetadataRepository, tmp_path: Path) -> None:
        """Should handle nonexistent date gracefully."""
        runs = repository.discover_runs(
            base_path=tmp_path,
            network="NSFNet",
            dates=["9999"],
        )

        assert len(runs) == 0

    def test_get_run_metadata_from_file(self, repository: FileMetadataRepository, setup_test_structure: Dict[str, Any]) -> None:
        """Should load metadata from file."""
        data = setup_test_structure
        run_data = data["runs"][0]

        metadata = repository.get_run_metadata(run_data["path"])

        assert metadata["path_algorithm"] == run_data["metadata"]["path_algorithm"]
        assert metadata["run_id"] == run_data["metadata"]["run_id"]

    def test_get_run_metadata_uses_cache(self, repository: FileMetadataRepository, setup_test_structure: Dict[str, Any]) -> None:
        """Should use cache for repeated requests."""
        data = setup_test_structure
        run_path = data["runs"][0]["path"]

        # First call - loads from file
        metadata1 = repository.get_run_metadata(run_path)

        # Second call - should use cache
        metadata2 = repository.get_run_metadata(run_path)

        assert metadata1 is metadata2  # Same object reference (cached)

    def test_get_run_metadata_fallback_to_input_json(self, repository: FileMetadataRepository, tmp_path: Path) -> None:
        """Should fallback to input.json if metadata.json doesn't exist."""
        run_path = tmp_path / "run1"
        run_path.mkdir(parents=True)

        metadata = {"path_algorithm": "ppo_obs_7", "run_id": "run1"}
        with open(run_path / "input.json", 'w') as f:
            json.dump(metadata, f)

        result = repository.get_run_metadata(run_path)

        assert result["path_algorithm"] == "ppo_obs_7"

    def test_get_run_metadata_no_file_uses_defaults(self, repository: FileMetadataRepository, tmp_path: Path) -> None:
        """Should return default metadata when no file exists."""
        run_path = tmp_path / "run1"
        run_path.mkdir(parents=True)

        metadata = repository.get_run_metadata(run_path)

        assert metadata["path_algorithm"] == "unknown"
        assert metadata["run_id"] == "run1"

    def test_cache_metadata(self, repository: FileMetadataRepository, tmp_path: Path) -> None:
        """Should cache metadata successfully."""
        run_path = tmp_path / "run1"
        metadata = {"path_algorithm": "ppo_obs_7", "run_id": "run1"}

        repository.cache_metadata(run_path, metadata)

        # Should retrieve from cache
        cached = repository.get_cached_metadata(run_path)
        assert cached is not None
        assert cached["path_algorithm"] == "ppo_obs_7"

    def test_cache_expiration(self, repository: FileMetadataRepository, tmp_path: Path) -> None:
        """Should expire cache entries after TTL."""
        run_path = tmp_path / "run1"
        metadata = {"path_algorithm": "ppo_obs_7", "run_id": "run1"}

        repository.cache_metadata(run_path, metadata)

        # Should be cached immediately
        cached = repository.get_cached_metadata(run_path)
        assert cached is not None

        # Wait for cache to expire (TTL is 1 second)
        time.sleep(1.1)

        # Should be expired now
        cached = repository.get_cached_metadata(run_path)
        assert cached is None

    def test_clear_cache(self, repository: FileMetadataRepository, tmp_path: Path) -> None:
        """Should clear all cached metadata."""
        run_path = tmp_path / "run1"
        metadata = {"path_algorithm": "ppo_obs_7", "run_id": "run1"}

        repository.cache_metadata(run_path, metadata)
        assert repository.get_cached_metadata(run_path) is not None

        repository.clear_cache()
        assert repository.get_cached_metadata(run_path) is None

    def test_cache_disabled(self, tmp_path: Path) -> None:
        """Should not cache when caching is disabled."""
        repository = FileMetadataRepository(enable_cache=False)

        run_path = tmp_path / "run1"
        metadata = {"path_algorithm": "ppo_obs_7", "run_id": "run1"}

        repository.cache_metadata(run_path, metadata)

        # Should not be cached
        cached = repository.get_cached_metadata(run_path)
        assert cached is None

    def test_get_cache_stats(self, repository: FileMetadataRepository, tmp_path: Path) -> None:
        """Should return cache statistics."""
        # Initially empty
        stats = repository.get_cache_stats()
        assert stats["total_entries"] == 0
        assert stats["valid_entries"] == 0

        # Add some cached entries
        for i in range(3):
            run_path = tmp_path / f"run{i}"
            metadata = {"path_algorithm": f"algo{i}", "run_id": f"run{i}"}
            repository.cache_metadata(run_path, metadata)

        stats = repository.get_cache_stats()
        assert stats["total_entries"] == 3
        assert stats["valid_entries"] == 3
        assert stats["expired_entries"] == 0

    def test_prune_expired_entries(self, repository: FileMetadataRepository, tmp_path: Path) -> None:
        """Should remove expired cache entries."""
        # Add some entries
        for i in range(3):
            run_path = tmp_path / f"run{i}"
            metadata = {"path_algorithm": f"algo{i}", "run_id": f"run{i}"}
            repository.cache_metadata(run_path, metadata)

        assert len(repository._cache) == 3

        # Wait for expiration
        time.sleep(1.1)

        # Prune expired entries
        removed = repository.prune_expired_entries()

        assert removed == 3
        assert len(repository._cache) == 0

    def test_invalid_json_handling(self, repository: FileMetadataRepository, tmp_path: Path) -> None:
        """Should handle invalid JSON gracefully."""
        run_path = tmp_path / "run1"
        run_path.mkdir(parents=True)

        # Create invalid JSON file
        with open(run_path / "metadata.json", 'w') as f:
            f.write("{ invalid json }")

        # Should fall back to defaults
        metadata = repository.get_run_metadata(run_path)
        assert metadata["path_algorithm"] == "unknown"
