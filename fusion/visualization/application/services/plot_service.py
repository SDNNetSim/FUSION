"""High-level plot service for orchestrating plot operations."""

from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from fusion.visualization.application.dto import (
    PlotRequestDTO,
    PlotResultDTO,
    BatchPlotRequestDTO,
    BatchPlotResultDTO,
    ComparisonRequestDTO,
    ComparisonResultDTO,
)
from fusion.visualization.application.use_cases import (
    GeneratePlotUseCase,
    BatchGeneratePlotsUseCase,
    CompareAlgorithmsUseCase,
)
from fusion.visualization.application.ports import (
    DataProcessorPort,
    PlotRendererPort,
    CachePort,
)
from fusion.visualization.domain.repositories import SimulationRepository
from fusion.visualization.domain.value_objects.plot_specification import PlotType

logger = logging.getLogger(__name__)


class PlotService:
    """
    High-level service for plot operations.

    This provides a unified interface for all plotting functionality,
    orchestrating use cases and handling common configuration.
    """

    def __init__(
        self,
        simulation_repository: SimulationRepository,
        data_processor: Optional[DataProcessorPort] = None,
        plot_renderer: Optional[PlotRendererPort] = None,
        cache: Optional[CachePort] = None,
        metadata_repository: Optional[Any] = None,
        cache_service: Optional[CachePort] = None,
    ):
        """
        Initialize plot service.

        Args:
            simulation_repository: Repository for simulation data access
            data_processor: Processor for data transformation
            plot_renderer: Renderer for plot generation
            cache: Optional cache for performance
            metadata_repository: Optional metadata repository (legacy parameter)
            cache_service: Optional cache service (alias for cache)
        """
        self.simulation_repository = simulation_repository
        self.metadata_repository = metadata_repository
        self.data_processor = data_processor
        self.plot_renderer = plot_renderer

        # Handle cache aliases
        if cache_service is not None and cache is None:
            cache = cache_service
        self.cache = cache
        self.cache_service = cache

        # Initialize use cases (only if we have required dependencies)
        if data_processor and plot_renderer:
            self.generate_plot_use_case: Optional[GeneratePlotUseCase] = GeneratePlotUseCase(
                simulation_repository=simulation_repository,
                data_processor=data_processor,
                plot_renderer=plot_renderer,
                cache=cache,
            )

            self.batch_generate_plots_use_case: Optional[BatchGeneratePlotsUseCase] = BatchGeneratePlotsUseCase(
                generate_plot_use_case=self.generate_plot_use_case
            )

            self.compare_algorithms_use_case: Optional[CompareAlgorithmsUseCase] = CompareAlgorithmsUseCase(
                simulation_repository=simulation_repository,
                cache=cache,
            )
        else:
            self.generate_plot_use_case = None
            self.batch_generate_plots_use_case = None
            self.compare_algorithms_use_case = None

    def generate_plot(
        self,
        network: str,
        dates: List[str],
        plot_type: str | PlotType,
        algorithms: Optional[List[str]] = None,
        traffic_volumes: Optional[List[float]] = None,
        save_path: Optional[Path] = None,
        **kwargs: Any,
    ) -> PlotResultDTO:
        """
        Generate a single plot.

        Args:
            network: Network name (e.g., "NSFNet")
            dates: List of date strings (e.g., ["0606"])
            plot_type: Type of plot (e.g., "blocking", PlotType.BLOCKING)
            algorithms: Optional list of algorithms to plot
            traffic_volumes: Optional list of traffic volumes
            save_path: Optional output path
            **kwargs: Additional configuration options

        Returns:
            PlotResultDTO with generation results
        """
        # Convert plot_type to PlotType if string
        if isinstance(plot_type, str):
            plot_type = PlotType(plot_type)

        request = PlotRequestDTO(
            network=network,
            dates=dates,
            plot_type=plot_type,
            algorithms=algorithms,
            traffic_volumes=traffic_volumes,
            save_path=save_path,
            **kwargs,
        )

        logger.info(
            f"Generating {plot_type.value} plot for {network} on dates {dates}"
        )

        if self.generate_plot_use_case is None:
            raise RuntimeError("PlotService not fully initialized with data_processor and plot_renderer")
        return self.generate_plot_use_case.execute(request)

    def batch_generate(
        self,
        network: str,
        dates: List[str],
        plot_configs: List[Dict[str, Any]],
        parallel: bool = True,
        max_workers: int = 4,
        output_dir: Optional[Path] = None,
    ) -> BatchPlotResultDTO | List[PlotResultDTO]:
        """
        Generate multiple plots in batch.

        Args:
            network: Network name
            dates: List of date strings
            plot_configs: List of plot configuration dictionaries
            parallel: Whether to generate plots in parallel
            max_workers: Maximum number of parallel workers
            output_dir: Optional output directory for all plots

        Returns:
            BatchPlotResultDTO with results for all plots
        """
        # Convert plot configs to PlotRequestDTO objects
        plot_requests = []
        for config in plot_configs:
            # Ensure plot_type is PlotType
            plot_type = config.get("plot_type")
            if isinstance(plot_type, str):
                config["plot_type"] = PlotType(plot_type)

            # Set output path if output_dir specified
            if output_dir and "save_path" not in config:
                plot_name = f"{config['plot_type'].value}.png"
                config["save_path"] = output_dir / plot_name

            request = PlotRequestDTO(
                network=network,
                dates=dates,
                **config,
            )
            plot_requests.append(request)

        batch_request = BatchPlotRequestDTO(
            network=network,
            dates=dates,
            plots=plot_requests,
            parallel=parallel,
            max_workers=max_workers,
            output_dir=output_dir,
        )

        logger.info(
            f"Batch generating {len(plot_requests)} plots "
            f"(parallel={parallel}, workers={max_workers})"
        )

        if self.batch_generate_plots_use_case is None:
            raise RuntimeError("PlotService not fully initialized with data_processor and plot_renderer")
        return self.batch_generate_plots_use_case.execute(batch_request)

    def compare_algorithms(
        self,
        network: str,
        dates: List[str],
        algorithms: List[str],
        metric: str = "blocking_probability",
        traffic_volumes: Optional[List[float]] = None,
        save_path: Optional[Path] = None,
        **kwargs: Any,
    ) -> ComparisonResultDTO:
        """
        Compare algorithms statistically.

        Args:
            network: Network name
            dates: List of date strings
            algorithms: List of algorithms to compare (must be at least 2)
            metric: Metric to compare (default: "blocking_probability")
            traffic_volumes: Optional list of traffic volumes
            save_path: Optional output path for comparison plot
            **kwargs: Additional configuration options

        Returns:
            ComparisonResultDTO with statistical comparison results
        """
        request = ComparisonRequestDTO(
            network=network,
            dates=dates,
            algorithms=algorithms,
            metric=metric,
            traffic_volumes=traffic_volumes,
            save_path=save_path,
            **kwargs,
        )

        logger.info(
            f"Comparing {len(algorithms)} algorithms on metric '{metric}' "
            f"for {network} on dates {dates}"
        )

        if self.compare_algorithms_use_case is None:
            raise RuntimeError("PlotService not fully initialized with data_processor and plot_renderer")
        return self.compare_algorithms_use_case.execute(request)

    def get_available_networks(self) -> List[str]:
        """
        Get list of available networks.

        Returns:
            List of network names
        """
        # This would query the repository for available networks
        # For now, return common networks
        return ["NSFNet", "USNet", "Pan-European", "dt_network"]

    def get_available_dates(self, network: str) -> List[str]:
        """
        Get list of available dates for a network.

        Args:
            network: Network name

        Returns:
            List of date strings
        """
        # This would query the repository for available dates
        # Implementation depends on repository capabilities
        logger.warning("get_available_dates not yet implemented")
        return []

    def get_available_algorithms(
        self, network: str, dates: List[str]
    ) -> List[str]:
        """
        Get list of available algorithms for given network and dates.

        Args:
            network: Network name
            dates: List of date strings

        Returns:
            List of algorithm names
        """
        try:
            runs = self.simulation_repository.find_runs(
                network=network,
                dates=dates,
            )

            # Extract unique algorithms
            algorithms = sorted(set(run.algorithm for run in runs))
            return algorithms

        except Exception as e:
            logger.error(f"Error getting available algorithms: {e}")
            return []

    def clear_cache(self) -> None:
        """Clear all cached data."""
        if self.cache:
            self.cache.clear()
            logger.info("Cache cleared")
        else:
            logger.warning("No cache configured")
