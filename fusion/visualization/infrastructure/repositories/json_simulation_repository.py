"""JSON-based implementation of SimulationRepository."""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

from fusion.visualization.domain.repositories.simulation_repository import (
    SimulationRepository,
)
from fusion.visualization.domain.entities.run import Run
from fusion.visualization.infrastructure.adapters import (
    CanonicalData,
    get_adapter,
)
from fusion.visualization.domain.exceptions.domain_exceptions import (
    RepositoryError,
    RunDataNotFoundError,
    DataFormatError,
)

logger = logging.getLogger(__name__)


class JsonSimulationRepository(SimulationRepository):
    """
    Repository for JSON-based simulation data.

    This implementation loads data from the standard file structure:
    {base_path}/{network}/{date}/{timestamp}/s{N}/{erlang}_erlang.json
    """

    def __init__(
        self,
        base_path: Path,
        metadata_repository: Optional[Any] = None,
        adapter_registry: Optional[Any] = None,
    ):
        """
        Initialize JSON repository.

        Args:
            base_path: Root directory containing simulation data
            metadata_repository: Optional metadata repository for faster discovery
            adapter_registry: Optional adapter registry (currently unused, kept for compatibility)
        """
        self.base_path = Path(base_path)
        self.metadata_repository = metadata_repository
        self.adapter_registry = adapter_registry
        self._data_cache: Dict[tuple, CanonicalData] = {}

    def find_runs(
        self,
        network: str,
        dates: List[str],
        algorithm: Optional[str] = None,
        run_ids: Optional[List[str]] = None,
    ) -> List[Run]:
        """Find runs matching the given criteria."""
        runs: List[Run] = []

        for date in dates:
            date_path = self.base_path / network / date

            if not date_path.exists():
                logger.warning(
                    f"Date directory not found: {date_path}. Skipping."
                )
                continue

            # Discover run directories
            try:
                for run_dir in date_path.iterdir():
                    if not run_dir.is_dir():
                        continue

                    # If specific run IDs requested, filter
                    if run_ids and run_dir.name not in run_ids:
                        continue

                    # Load metadata to determine algorithm
                    try:
                        metadata = self._load_run_metadata(run_dir)
                        run_algorithm = metadata.get("path_algorithm", "unknown")

                        # Filter by algorithm if specified
                        if algorithm and run_algorithm != algorithm:
                            continue

                        # Create Run entity
                        run = Run(
                            id=run_dir.name,
                            network=network,
                            date=date,
                            algorithm=run_algorithm,
                            path=run_dir,
                            metadata=metadata,
                        )
                        runs.append(run)

                    except Exception as e:
                        logger.warning(
                            f"Failed to load metadata for {run_dir}: {e}. Skipping."
                        )
                        continue

            except PermissionError as e:
                raise RepositoryError(
                    f"Permission denied accessing {date_path}: {e}"
                ) from e

        return runs

    def get_run_data(
        self,
        run: Run,
        traffic_volume: float,
    ) -> CanonicalData:
        """Load data for a specific run and traffic volume."""
        # Check cache first
        cache_key = (run.full_id, traffic_volume)
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]

        # Determine file path
        # Try multiple possible locations for the data file
        possible_paths = [
            run.get_data_file(traffic_volume),
            run.path / f"s1/{int(traffic_volume)}_erlang.json",
            run.path / f"{int(traffic_volume)}_erlang.json",
        ]

        data_file = None
        for path in possible_paths:
            if path.exists():
                data_file = path
                break

        if data_file is None:
            raise RunDataNotFoundError(
                f"Data file not found for run {run.full_id} "
                f"at traffic volume {traffic_volume}. "
                f"Tried paths: {[str(p) for p in possible_paths]}"
            )

        # Load and parse JSON
        try:
            with open(data_file, 'r') as f:
                raw_data = json.load(f)
        except json.JSONDecodeError as e:
            raise DataFormatError(
                f"Invalid JSON in {data_file}: {e}"
            ) from e
        except Exception as e:
            raise RepositoryError(
                f"Failed to load data from {data_file}: {e}"
            ) from e

        # Convert to canonical format using adapter
        try:
            adapter = get_adapter(raw_data)
            canonical_data = adapter.to_canonical(raw_data)

            # Cache the result
            self._data_cache[cache_key] = canonical_data

            return canonical_data

        except Exception as e:
            raise DataFormatError(
                f"Failed to convert data to canonical format: {e}"
            ) from e

    def get_run_data_batch(
        self,
        run: Run,
        traffic_volumes: List[float],
    ) -> Dict[float, CanonicalData]:
        """Load data for multiple traffic volumes efficiently."""
        result = {}

        for tv in traffic_volumes:
            try:
                result[tv] = self.get_run_data(run, tv)
            except RunDataNotFoundError as e:
                logger.warning(f"Skipping traffic volume {tv}: {e}")
                continue

        return result

    def exists(
        self,
        run: Run,
        traffic_volume: Optional[float] = None,
    ) -> bool:
        """Check if data exists for a run."""
        if traffic_volume is None:
            # Just check if run directory exists
            return run.exists()

        # Check if specific traffic volume file exists in multiple possible locations
        possible_paths = [
            run.get_data_file(traffic_volume),
            run.path / f"s1/{int(traffic_volume)}_erlang.json",
            run.path / f"{int(traffic_volume)}_erlang.json",
        ]

        return any(path.exists() for path in possible_paths)

    def get_available_traffic_volumes(self, run: Run) -> List[float]:
        """Get list of available traffic volumes for a run."""
        traffic_volumes: List[float] = []

        if not run.exists():
            return traffic_volumes

        # Look for files matching pattern *_erlang.json
        # Check both in run directory and s1 subdirectory
        search_paths = [run.path, run.path / "s1"]

        for search_path in search_paths:
            if not search_path.exists():
                continue

            for file_path in search_path.glob("*_erlang.json"):
                # Extract traffic volume from filename
                try:
                    filename = file_path.stem  # Remove .json
                    tv_str = filename.replace("_erlang", "")
                    tv = float(tv_str)
                    if tv not in traffic_volumes:
                        traffic_volumes.append(tv)
                except ValueError:
                    logger.warning(
                        f"Could not parse traffic volume from filename: {file_path}"
                    )
                    continue

        return sorted(traffic_volumes)

    def get_metadata(self, run: Run) -> Dict[str, Any]:
        """Get metadata for a run."""
        return self._load_run_metadata(run.path)

    def _load_run_metadata(self, run_path: Path) -> Dict[str, Any]:
        """
        Load metadata from run directory.

        Looks for metadata in multiple possible locations:
        1. metadata.json (DRL runs)
        2. input.json (standard runs)
        3. config.json (alternative naming)
        """
        # Use metadata repository if available
        if self.metadata_repository:
            try:
                result = self.metadata_repository.get_run_metadata(run_path)
                return dict(result)
            except Exception as e:
                logger.debug(
                    f"Metadata repository failed, falling back to direct load: {e}"
                )

        # Try multiple metadata file locations
        metadata_files = [
            run_path / "metadata.json",
            run_path / "input.json",
            run_path / "config.json",
        ]

        for metadata_file in metadata_files:
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        return dict(metadata)
                except Exception as e:
                    logger.warning(
                        f"Failed to load metadata from {metadata_file}: {e}"
                    )
                    continue

        # No metadata found, return minimal info
        logger.debug(f"No metadata file found in {run_path}")
        return {
            "path_algorithm": "unknown",
            "run_id": run_path.name,
        }

    def clear_cache(self) -> None:
        """Clear the data cache."""
        self._data_cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cached_entries": len(self._data_cache),
            "cache_keys": list(self._data_cache.keys()),
        }
