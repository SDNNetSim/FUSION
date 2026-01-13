"""Abstract repository interface for simulation data access."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from fusion.visualization.domain.entities.run import Run
from fusion.visualization.infrastructure.adapters.canonical_data import CanonicalData


class SimulationRepository(ABC):
    """
    Abstract repository for accessing simulation data.

    This interface defines the contract for data access, allowing
    different implementations (JSON files, databases, etc.) while
    keeping the domain logic independent of storage details.
    """

    @abstractmethod
    def find_runs(
        self,
        network: str,
        dates: list[str],
        algorithm: str | None = None,
        run_ids: list[str] | None = None,
    ) -> list[Run]:
        """
        Find runs matching the given criteria.

        Args:
            network: Network name (e.g., "NSFNet", "USNet")
            dates: List of date strings (e.g., ["0606", "0611"])
            algorithm: Optional algorithm filter (e.g., "ppo_obs_7")
            run_ids: Optional list of specific run IDs to include

        Returns:
            List of matching Run entities

        Raises:
            RepositoryError: If data access fails
        """
        pass

    @abstractmethod
    def get_run_data(
        self,
        run: Run,
        traffic_volume: float,
    ) -> CanonicalData:
        """
        Load data for a specific run and traffic volume.

        Args:
            run: The Run entity to load data for
            traffic_volume: Traffic volume in Erlang

        Returns:
            CanonicalData with the run's simulation results

        Raises:
            RunDataNotFoundError: If data file doesn't exist
            DataFormatError: If data cannot be parsed
            RepositoryError: If other data access issues occur
        """
        pass

    @abstractmethod
    def get_run_data_batch(
        self,
        run: Run,
        traffic_volumes: list[float],
    ) -> dict[float, CanonicalData]:
        """
        Load data for multiple traffic volumes efficiently.

        Args:
            run: The Run entity to load data for
            traffic_volumes: List of traffic volumes in Erlang

        Returns:
            Dictionary mapping traffic volumes to CanonicalData

        Raises:
            RepositoryError: If data access fails
        """
        pass

    @abstractmethod
    def exists(self, run: Run, traffic_volume: float | None = None) -> bool:
        """
        Check if data exists for a run.

        Args:
            run: The Run entity to check
            traffic_volume: Optional specific traffic volume to check

        Returns:
            True if data exists, False otherwise
        """
        pass

    @abstractmethod
    def get_available_traffic_volumes(self, run: Run) -> list[float]:
        """
        Get list of available traffic volumes for a run.

        Args:
            run: The Run entity to query

        Returns:
            Sorted list of available traffic volumes

        Raises:
            RepositoryError: If data access fails
        """
        pass

    @abstractmethod
    def get_metadata(self, run: Run) -> dict[str, Any]:
        """
        Get metadata for a run.

        Args:
            run: The Run entity to get metadata for

        Returns:
            Dictionary of metadata

        Raises:
            RepositoryError: If metadata cannot be loaded
        """
        pass
