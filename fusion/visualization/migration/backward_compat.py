"""
Backward compatibility layer for legacy visualization API.

This module provides wrappers and adapters that allow legacy code to work
with the new visualization system without requiring immediate changes.
"""

import warnings
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

from fusion.visualization.application.dto import PlotRequestDTO
from fusion.visualization.application.services import CacheService, PlotService
from fusion.visualization.application.use_cases import GeneratePlotUseCase
from fusion.visualization.domain.value_objects import PlotType
from fusion.visualization.infrastructure.adapters import DataAdapterRegistry
from fusion.visualization.infrastructure.processors import BlockingProcessor
from fusion.visualization.infrastructure.renderers import MatplotlibRenderer
from fusion.visualization.infrastructure.repositories import (
    FileMetadataRepository,
    JsonSimulationRepository,
)


def legacy_deprecation_warning(old_api: str, new_api: str) -> None:
    """
    Issue a deprecation warning for legacy API usage.

    Args:
        old_api: Name of the deprecated API
        new_api: Name of the new API to use instead
    """
    warnings.warn(
        f"'{old_api}' is deprecated and will be removed in version 7.0. "
        f"Please use '{new_api}' instead. "
        f"See migration guide at: docs/visualization/migration_guide.md",
        DeprecationWarning,
        stacklevel=3,
    )


def legacy_plot_wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator for legacy plot functions.

    Wraps legacy plot functions to issue deprecation warnings and
    translate parameters to new format.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Issue deprecation warning
        legacy_deprecation_warning(old_api=func.__name__, new_api="fusion.visualization.generate_plot")

        # Call original function
        return func(*args, **kwargs)

    return wrapper


class LegacyPlotAdapter:
    """
    Adapter that provides the legacy PlotStats API using the new system.

    This allows legacy code like:
        plot_obj = PlotStats(sims_info_dict=config)
        plot_obj.plot_blocking()

    To work with the new visualization system without changes.

    Example:
        >>> # Legacy code (still works with adapter)
        >>> from fusion.visualization.legacy import PlotStats
        >>> plot_obj = PlotStats(sims_info_dict=my_config)
        >>> plot_obj.plot_blocking()
    """

    def __init__(
        self,
        sims_info_dict: dict[str, Any] | None = None,
        base_path: Path | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize legacy adapter.

        Args:
            sims_info_dict: Legacy configuration dictionary
            base_path: Base path for data files
            **kwargs: Additional legacy parameters
        """
        legacy_deprecation_warning(old_api="PlotStats", new_api="fusion.visualization.generate_plot")

        self.sims_info_dict = sims_info_dict or {}
        self.base_path = base_path or Path("../../data/output")
        self.kwargs = kwargs

        # Setup new system components
        self._setup_new_system()

    def _setup_new_system(self) -> None:
        """Initialize new visualization system components."""
        # Create repositories
        adapter_registry = DataAdapterRegistry()
        self.sim_repo = JsonSimulationRepository(
            base_path=self.base_path,
            adapter_registry=adapter_registry,
        )
        self.meta_repo = FileMetadataRepository(base_path=self.base_path)

        # Create services
        cache_dir = Path.home() / ".fusion" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_service = CacheService(cache_dir=cache_dir)

        self.plot_service = PlotService(
            simulation_repository=self.sim_repo,
            metadata_repository=self.meta_repo,
            cache_service=self.cache_service,
        )

        # Create processor and renderer
        self.processor = BlockingProcessor()

        output_dir = Path("./figures")
        output_dir.mkdir(exist_ok=True)
        self.renderer = MatplotlibRenderer(output_dir=output_dir)

        # Create use case
        self.use_case = GeneratePlotUseCase(
            plot_service=self.plot_service,
            processor=self.processor,
            renderer=self.renderer,
        )

    def _extract_legacy_params(self) -> dict[str, Any]:
        """Extract parameters from legacy sims_info_dict."""
        params = {}

        # Extract network
        if "networks_matrix" in self.sims_info_dict:
            networks = self.sims_info_dict["networks_matrix"]
            if networks and len(networks) > 0 and len(networks[0]) > 0:
                params["network"] = networks[0][-1]

        # Extract dates
        if "dates_matrix" in self.sims_info_dict:
            dates = self.sims_info_dict["dates_matrix"]
            if dates and len(dates) > 0 and len(dates[0]) > 0:
                params["dates"] = [dates[0][-1]]

        # Extract algorithms (from path_algorithm_matrix or similar)
        if "path_algorithm_matrix" in self.sims_info_dict:
            algos = self.sims_info_dict["path_algorithm_matrix"]
            if algos and len(algos) > 0:
                params["algorithms"] = algos[0] if isinstance(algos[0], list) else [algos[0]]

        # Extract traffic volumes (from erlangs or similar)
        if "erlangs" in self.sims_info_dict:
            params["traffic_volumes"] = self.sims_info_dict["erlangs"]

        return params

    def plot_blocking(
        self,
        title: str = "Blocking Probability vs Traffic Volume",
        save_path: Path | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Generate blocking probability plot (legacy API).

        Args:
            title: Plot title
            save_path: Where to save the plot
            **kwargs: Additional parameters

        Returns:
            PlotResultDTO from new system
        """
        # Extract parameters from legacy config
        params = self._extract_legacy_params()

        # Set defaults
        if "network" not in params:
            raise ValueError("Network not found in sims_info_dict")
        if "dates" not in params:
            raise ValueError("Dates not found in sims_info_dict")
        if "traffic_volumes" not in params:
            params["traffic_volumes"] = [600, 700, 800, 900, 1000]

        # Create request
        request = PlotRequestDTO(
            network=params["network"],
            dates=params["dates"],
            plot_type=PlotType.LINE,
            metrics=["blocking_probability"],
            algorithms=params.get("algorithms", []),
            traffic_volumes=params["traffic_volumes"],
            title=title,
            x_label="Traffic Volume (Erlang)",
            y_label="Blocking Probability",
            save_path=save_path or Path(f"./figures/blocking_{params['network']}.png"),
            include_ci=kwargs.get("include_ci", True),
        )

        # Execute using new system
        result = self.use_case.execute(request)

        if not result.success:
            warnings.warn(f"Plot generation failed: {result.error_message}", stacklevel=2)

        return result

    def plot_rewards(
        self,
        title: str = "Rewards Over Time",
        save_path: Path | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Generate rewards plot (legacy API).

        Args:
            title: Plot title
            save_path: Where to save the plot
            **kwargs: Additional parameters

        Returns:
            PlotResultDTO from new system
        """
        params = self._extract_legacy_params()

        if "network" not in params:
            raise ValueError("Network not found in sims_info_dict")
        if "dates" not in params:
            raise ValueError("Dates not found in sims_info_dict")

        request = PlotRequestDTO(
            network=params["network"],
            dates=params["dates"],
            plot_type=PlotType.LINE,
            metrics=["rewards"],
            algorithms=params.get("algorithms", []),
            traffic_volumes=params.get("traffic_volumes", []),
            title=title,
            x_label="Episode",
            y_label="Reward",
            save_path=save_path or Path(f"./figures/rewards_{params['network']}.png"),
        )

        result = self.use_case.execute(request)

        if not result.success:
            warnings.warn(f"Plot generation failed: {result.error_message}", stacklevel=2)

        return result

    def _save_plot(self, file_name: str) -> None:
        """
        Legacy save plot method.

        This method is kept for compatibility but doesn't do anything
        as the new system handles saving automatically.
        """
        warnings.warn(
            "The _save_plot method is deprecated. Plots are now saved automatically by the new system.",
            DeprecationWarning,
            stacklevel=2,
        )


class LegacyConfigAdapter:
    """
    Adapter for legacy configuration format.

    Converts legacy configuration dictionaries to new PlotRequestDTO format.
    """

    @staticmethod
    def adapt_config(legacy_config: dict[str, Any]) -> PlotRequestDTO:
        """
        Convert legacy configuration to new PlotRequestDTO.

        Args:
            legacy_config: Legacy configuration dictionary

        Returns:
            PlotRequestDTO for new system

        Raises:
            ValueError: If required fields are missing
        """
        # Extract required fields
        network = legacy_config.get("network")
        if not network:
            raise ValueError("Missing required field: network")

        dates = legacy_config.get("dates", [])
        if not dates:
            raise ValueError("Missing required field: dates")

        # Handle plot type
        plot_type_str = legacy_config.get("plot_type", "blocking")
        plot_type = PlotType.LINE  # Default

        # Extract algorithms
        algorithms = legacy_config.get("algorithms", [])
        obs_spaces = legacy_config.get("observation_spaces", [])

        # Combine algorithms and observation spaces if both present
        if obs_spaces and algorithms:
            combined_algorithms = []
            for algo in algorithms:
                for obs in obs_spaces:
                    combined_algorithms.append(f"{algo}_{obs}")
            algorithms = combined_algorithms

        # Extract traffic volumes
        traffic_volumes = legacy_config.get("traffic_volumes", [600, 700, 800, 900, 1000])

        # Create request
        request = PlotRequestDTO(
            network=network,
            dates=dates,
            plot_type=plot_type,
            metrics=[legacy_config.get("metric", "blocking_probability")],
            algorithms=algorithms,
            traffic_volumes=traffic_volumes,
            title=legacy_config.get("title", f"{plot_type_str.title()} Plot"),
            x_label=legacy_config.get("x_label", "Traffic Volume (Erlang)"),
            y_label=legacy_config.get("y_label", "Value"),
            save_path=Path(legacy_config.get("save_path", f"./figures/{plot_type_str}.png")),
            include_ci=legacy_config.get("include_ci", True),
        )

        return request


def legacy_find_times(
    dates_dict: dict[str, str],
    filter_dict: dict[str, Any] | None = None,
    base_path: Path | None = None,
) -> dict[str, list[list[str]]]:
    """
    Legacy find_times function compatibility wrapper.

    This function mimics the old find_times API but uses the new
    repository system under the hood.

    Args:
        dates_dict: Dictionary mapping dates to networks
        filter_dict: Optional filters
        base_path: Base path for data files

    Returns:
        Legacy-format sims_info_dict
    """
    legacy_deprecation_warning(old_api="find_times", new_api="SimulationRepository.find_runs")

    base_path = base_path or Path("../../data/output")

    # Setup repositories
    adapter_registry = DataAdapterRegistry()
    sim_repo = JsonSimulationRepository(
        base_path=base_path,
        adapter_registry=adapter_registry,
    )

    # Find runs using new system
    all_runs = []
    for date, network in dates_dict.items():
        runs = sim_repo.find_runs(network=network, dates=[date])
        all_runs.extend(runs)

    # Convert to legacy format
    sims_info_dict = {
        "networks_matrix": [[run.network for run in all_runs]],
        "dates_matrix": [[run.date for run in all_runs]],
        "times_matrix": [[run.id for run in all_runs]],
        "path_algorithm_matrix": [[run.algorithm for run in all_runs]],
    }

    return sims_info_dict


# Convenience functions for common legacy patterns
def legacy_plot_blocking(config: dict[str, Any]) -> Any:
    """
    Generate blocking plot using legacy config format.

    Args:
        config: Legacy configuration dictionary

    Returns:
        Plot result
    """
    legacy_deprecation_warning(old_api="legacy_plot_blocking", new_api="generate_plot")

    adapter = LegacyPlotAdapter(sims_info_dict=config)
    return adapter.plot_blocking()


def legacy_plot_rewards(config: dict[str, Any]) -> Any:
    """
    Generate rewards plot using legacy config format.

    Args:
        config: Legacy configuration dictionary

    Returns:
        Plot result
    """
    legacy_deprecation_warning(old_api="legacy_plot_rewards", new_api="generate_plot")

    adapter = LegacyPlotAdapter(sims_info_dict=config)
    return adapter.plot_rewards()
