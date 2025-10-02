"""Unit tests for RepositoryFactory."""

import pytest
from pathlib import Path

from fusion.visualization.infrastructure.repositories import (
    RepositoryFactory,
    get_default_factory,
    configure_default_factory,
    get_simulation_repository,
    get_metadata_repository,
)
from fusion.visualization.domain.repositories import (
    SimulationRepository,
    MetadataRepository,
)


class TestRepositoryFactory:
    """Test suite for RepositoryFactory."""

    @pytest.fixture
    def factory(self, tmp_path: Path) -> RepositoryFactory:
        """Create factory instance for tests."""
        return RepositoryFactory(
            base_path=tmp_path,
            cache_ttl_seconds=3600,
            enable_cache=True,
        )

    def test_initialization(self, tmp_path: Path) -> None:
        """Should initialize with correct configuration."""
        factory = RepositoryFactory(
            base_path=tmp_path,
            cache_ttl_seconds=1800,
            enable_cache=False,
        )

        assert factory.base_path == tmp_path
        assert factory.cache_ttl_seconds == 1800
        assert factory.enable_cache is False

    def test_default_base_path(self) -> None:
        """Should use default base path when none provided."""
        factory = RepositoryFactory()

        assert factory.base_path is not None
        assert isinstance(factory.base_path, Path)

    def test_create_metadata_repository(self, factory: RepositoryFactory) -> None:
        """Should create metadata repository."""
        repo = factory.create_metadata_repository()

        assert repo is not None
        assert isinstance(repo, MetadataRepository)

    def test_create_metadata_repository_singleton(self, factory: RepositoryFactory) -> None:
        """Should return same instance on repeated calls."""
        repo1 = factory.create_metadata_repository()
        repo2 = factory.create_metadata_repository()

        assert repo1 is repo2

    def test_create_simulation_repository(self, factory: RepositoryFactory) -> None:
        """Should create simulation repository."""
        repo = factory.create_simulation_repository()

        assert repo is not None
        assert isinstance(repo, SimulationRepository)

    def test_create_simulation_repository_singleton(self, factory: RepositoryFactory) -> None:
        """Should return same instance on repeated calls."""
        repo1 = factory.create_simulation_repository()
        repo2 = factory.create_simulation_repository()

        assert repo1 is repo2

    def test_configure_resets_singletons(self, factory: RepositoryFactory, tmp_path: Path) -> None:
        """Should reset cached instances when reconfigured."""
        # Get initial instances
        repo1 = factory.create_simulation_repository()

        # Reconfigure
        new_path = tmp_path / "new_data"
        new_path.mkdir()
        factory.configure(base_path=new_path)

        # Get new instances
        repo2 = factory.create_simulation_repository()

        # Should be different instances
        assert repo1 is not repo2
        assert factory.base_path == new_path

    def test_configure_partial_update(self, factory: RepositoryFactory) -> None:
        """Should allow partial configuration updates."""
        original_path = factory.base_path
        original_ttl = factory.cache_ttl_seconds

        # Update only cache_ttl
        factory.configure(cache_ttl_seconds=7200)

        assert factory.base_path == original_path
        assert factory.cache_ttl_seconds == 7200

    def test_clear_all_caches(self, factory: RepositoryFactory, tmp_path: Path) -> None:
        """Should clear caches in all repositories."""
        # Create repositories and use them
        sim_repo = factory.create_simulation_repository()
        meta_repo = factory.create_metadata_repository()

        # Add some cached data
        metadata = {"path_algorithm": "ppo", "run_id": "run1"}
        run_path = tmp_path / "run1"
        run_path.mkdir()
        meta_repo.cache_metadata(run_path, metadata)

        assert meta_repo.get_cached_metadata(run_path) is not None

        # Clear all caches
        factory.clear_all_caches()

        assert meta_repo.get_cached_metadata(run_path) is None

    def test_get_cache_stats(self, factory: RepositoryFactory, tmp_path: Path) -> None:
        """Should return cache statistics from all repositories."""
        # Create repositories
        sim_repo = factory.create_simulation_repository()
        meta_repo = factory.create_metadata_repository()

        # Add some cached metadata
        run_path = tmp_path / "run1"
        run_path.mkdir()
        metadata = {"path_algorithm": "ppo", "run_id": "run1"}
        meta_repo.cache_metadata(run_path, metadata)

        stats = factory.get_cache_stats()

        assert "metadata" in stats
        assert stats["metadata"]["total_entries"] == 1


class TestGlobalFactoryFunctions:
    """Test suite for global factory functions."""

    def test_get_default_factory(self) -> None:
        """Should return default factory instance."""
        factory = get_default_factory()

        assert factory is not None
        assert isinstance(factory, RepositoryFactory)

    def test_get_default_factory_singleton(self) -> None:
        """Should return same instance on repeated calls."""
        factory1 = get_default_factory()
        factory2 = get_default_factory()

        assert factory1 is factory2

    def test_configure_default_factory(self, tmp_path: Path) -> None:
        """Should configure default factory."""
        configure_default_factory(
            base_path=tmp_path,
            cache_ttl_seconds=7200,
        )

        factory = get_default_factory()
        assert factory.base_path == tmp_path
        assert factory.cache_ttl_seconds == 7200

    def test_get_simulation_repository(self) -> None:
        """Should return simulation repository from default factory."""
        repo = get_simulation_repository()

        assert repo is not None
        assert isinstance(repo, SimulationRepository)

    def test_get_metadata_repository(self) -> None:
        """Should return metadata repository from default factory."""
        repo = get_metadata_repository()

        assert repo is not None
        assert isinstance(repo, MetadataRepository)

    def test_repositories_use_same_factory(self) -> None:
        """Should use same factory instance for all repositories."""
        sim_repo1 = get_simulation_repository()
        sim_repo2 = get_simulation_repository()

        assert sim_repo1 is sim_repo2  # Singleton from same factory
