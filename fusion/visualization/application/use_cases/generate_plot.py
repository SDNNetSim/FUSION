"""Use case for generating a single plot."""

from __future__ import annotations

import logging
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fusion.visualization.application.dto import PlotRequestDTO, PlotResultDTO
from fusion.visualization.application.ports import (
    CachePort,
    DataProcessorPort,
    PlotRendererPort,
)
from fusion.visualization.domain.repositories import SimulationRepository

if TYPE_CHECKING:
    from fusion.visualization.application.services.plot_service import PlotService
from fusion.visualization.domain.entities.plot import (
    Plot,
    PlotConfiguration,
)
from fusion.visualization.domain.entities.run import Run
from fusion.visualization.domain.exceptions import (
    ProcessingError,
    RenderError,
    RepositoryError,
    ValidationError,
)
from fusion.visualization.domain.value_objects.plot_id import PlotId
from fusion.visualization.infrastructure.adapters.canonical_data import CanonicalData

logger = logging.getLogger(__name__)


class GeneratePlotUseCase:
    """
    Use case for generating a single plot.

    This orchestrates the entire plot generation workflow:
    1. Validate request
    2. Load data from repository
    3. Process data using appropriate processor
    4. Render plot using renderer
    5. Return result
    """

    def __init__(
        self,
        simulation_repository: SimulationRepository | None = None,
        data_processor: DataProcessorPort | None = None,
        plot_renderer: PlotRendererPort | None = None,
        cache: CachePort | None = None,
        plot_service: PlotService | None = None,  # Legacy parameter
        processor: DataProcessorPort | None = None,  # Legacy alias for data_processor
        renderer: PlotRendererPort | None = None,  # Legacy alias for plot_renderer
    ):
        """
        Initialize use case.

        Args:
            simulation_repository: Repository for accessing simulation data
            data_processor: Processor for transforming data
            plot_renderer: Renderer for creating plots
            cache: Optional cache for performance
            plot_service: Legacy parameter (ignored)
            processor: Legacy alias for data_processor
            renderer: Legacy alias for plot_renderer
        """
        self.plot_service = plot_service

        # Handle legacy parameter aliases
        if processor is not None and data_processor is None:
            data_processor = processor
        if renderer is not None and plot_renderer is None:
            plot_renderer = renderer

        # If plot_service is provided, extract repository and cache from it
        if plot_service is not None:
            if simulation_repository is None:
                simulation_repository = plot_service.simulation_repository
            if cache is None:
                cache = plot_service.cache

        self.simulation_repository = simulation_repository
        self.data_processor = data_processor
        self.plot_renderer = plot_renderer
        self.cache = cache

    def execute(self, request: PlotRequestDTO) -> PlotResultDTO:
        """
        Execute the plot generation use case.

        Args:
            request: Plot request DTO

        Returns:
            PlotResultDTO with result information
        """
        started_at = datetime.now()
        plot_id = PlotId.generate()

        logger.info(f"Starting plot generation {plot_id}")

        try:
            # 1. Validate request
            errors = request.validate()
            if errors:
                error_msg = f"Invalid request: {'; '.join(errors)}"
                logger.error(error_msg)
                return self._failure_result(
                    plot_id=str(plot_id),
                    error=error_msg,
                    started_at=started_at,
                )

            # 2. Create plot entity
            plot = self._create_plot_entity(plot_id, request)
            plot.validate()

            # 3. Load data
            plot.start_loading()
            runs, data = self._load_data(request)
            plot.mark_loaded()

            logger.info(f"Loaded data for {len(runs)} runs")

            # 4. Process data
            plot.start_processing()
            # Use the first metric from request, or fall back to blocking_probability
            metric_name = (
                request.metrics[0] if request.metrics else "blocking_probability"
            )
            processed_data = self._process_data(
                runs=runs,
                data=data,
                metric_name=metric_name,
                traffic_volumes=request.traffic_volumes or [],
                include_ci=request.include_ci,
            )

            # Create specification
            specification = self._create_specification(request, processed_data)
            plot.mark_processed(specification)

            logger.info(f"Processed data for {len(processed_data.y_data)} algorithms")

            # 5. Render plot
            plot.start_rendering()
            output_path = request.save_path or Path(f"plot_{plot_id}.{request.format}")
            if self.plot_renderer is None:
                raise RenderError("Plot renderer not configured")
            render_result = self.plot_renderer.render(
                specification=specification,
                output_path=output_path,
                dpi=request.dpi,
                format=request.format,
            )

            if not render_result.success:
                raise RenderError(render_result.error or "Rendering failed")

            plot.mark_completed()

            logger.info(f"Successfully generated plot {plot_id} at {output_path}")

            # 6. Return success result
            completed_at = datetime.now()
            return PlotResultDTO(
                success=True,
                plot_id=str(plot_id),
                output_path=output_path,
                plot_type=request.plot_type.value,
                algorithms=list(processed_data.y_data.keys()),
                traffic_volumes=processed_data.x_data,
                num_runs=len(runs),
                started_at=started_at,
                completed_at=completed_at,
                duration=completed_at - started_at,
                metadata=request.metadata,
            )

        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            return self._failure_result(
                plot_id=str(plot_id),
                error=f"Validation error: {e}",
                started_at=started_at,
            )

        except RepositoryError as e:
            logger.error(f"Repository error: {e}")
            return self._failure_result(
                plot_id=str(plot_id),
                error=f"Failed to load data: {e}",
                started_at=started_at,
            )

        except ProcessingError as e:
            logger.error(f"Processing error: {e}")
            return self._failure_result(
                plot_id=str(plot_id),
                error=f"Failed to process data: {e}",
                started_at=started_at,
            )

        except RenderError as e:
            logger.error(f"Render error: {e}")
            return self._failure_result(
                plot_id=str(plot_id),
                error=f"Failed to render plot: {e}",
                started_at=started_at,
            )

        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            return self._failure_result(
                plot_id=str(plot_id),
                error=f"Unexpected error: {e}",
                started_at=started_at,
            )

    def _create_plot_entity(self, plot_id: PlotId, request: PlotRequestDTO) -> Plot:
        """Create Plot entity from request."""
        config = PlotConfiguration(
            plot_type=request.plot_type,
            metrics=request.metrics or ["blocking_probability"],
            algorithms=request.algorithms or [],
            traffic_volumes=request.traffic_volumes or [],
            title=request.title,
            x_label=request.x_label,
            y_label=request.y_label,
            include_ci=request.include_ci,
            include_baselines=request.include_baselines,
            dpi=request.dpi,
            figsize=request.figsize,
            save_path=str(request.save_path) if request.save_path else None,
            metadata=request.metadata,
        )

        return Plot(
            id=plot_id,
            title=request.title or f"{request.plot_type.value} - {request.network}",
            configuration=config,
        )

    def _load_data(
        self, request: PlotRequestDTO
    ) -> tuple[list[Run], dict[str, dict[float, CanonicalData]]]:
        """
        Load simulation data.

        Returns:
            Tuple of (runs, data) where data is
            run_id -> traffic_volume -> CanonicalData
        """
        if self.simulation_repository is None:
            raise RepositoryError("Simulation repository not configured")

        # Find matching runs
        runs = self.simulation_repository.find_runs(
            network=request.network,
            dates=request.dates,
            algorithm=None,  # Load all algorithms, filter later
            run_ids=request.run_ids,
        )

        if not runs:
            raise RepositoryError(
                f"No runs found for network={request.network}, dates={request.dates}"
            )

        # Filter by algorithms if specified
        if request.algorithms:
            runs = [r for r in runs if r.algorithm in request.algorithms]

        if not runs:
            raise RepositoryError(f"No runs found for algorithms: {request.algorithms}")

        # Load data for each run and traffic volume
        data: dict[str, dict[float, CanonicalData]] = {}
        for run in runs:
            data[run.id] = {}

            # Get available traffic volumes if not specified
            traffic_volumes = request.traffic_volumes
            if not traffic_volumes:
                traffic_volumes = (
                    self.simulation_repository.get_available_traffic_volumes(run)
                )

            # Load data for each traffic volume
            for tv in traffic_volumes:
                try:
                    if self.cache and request.cache_enabled:
                        cache_key = f"run_data:{run.id}:{tv}"
                        if self.simulation_repository is None:
                            raise RepositoryError(
                                "Simulation repository not configured"
                            )
                        canonical_data = self.cache.get_or_compute(
                            key=cache_key,
                            compute_fn=partial(
                                self.simulation_repository.get_run_data, run, tv
                            ),
                            ttl_seconds=3600,  # 1 hour cache
                        )
                    else:
                        if self.simulation_repository is None:
                            raise RepositoryError(
                                "Simulation repository not configured"
                            )
                        canonical_data = self.simulation_repository.get_run_data(
                            run, tv
                        )

                    data[run.id][tv] = canonical_data

                except Exception as e:
                    logger.warning(
                        f"Failed to load data for run {run.id} at {tv} Erlang: {e}"
                    )
                    # Continue loading other data

        return runs, data

    def _process_data(
        self,
        runs: list[Run],
        data: dict[str, dict[float, CanonicalData]],
        metric_name: str,
        traffic_volumes: list[float],
        include_ci: bool,
    ) -> Any:
        """Process raw data into plottable format."""
        if self.data_processor is None:
            raise ProcessingError("Data processor not configured")

        if not self.data_processor.can_process(metric_name):
            raise ProcessingError(
                f"Processor cannot handle metric: {metric_name}. "
                f"Supported: {self.data_processor.get_supported_metrics()}"
            )

        return self.data_processor.process(
            runs=runs,
            data=data,
            metric_name=metric_name,
            traffic_volumes=traffic_volumes,
            include_ci=include_ci,
        )

    def _create_specification(
        self, request: PlotRequestDTO, processed_data: Any
    ) -> Any:
        """Create PlotSpecification from processed data."""
        from fusion.visualization.domain.value_objects.plot_specification import (
            PlotSpecification,
        )

        return PlotSpecification(
            plot_type=request.plot_type,
            title=request.title or f"{request.plot_type.value} - {request.network}",
            x_label=request.x_label or "Traffic Volume (Erlang)",
            y_label=request.y_label
            or request.plot_type.value.replace("_", " ").title(),
            x_data=processed_data.x_data,
            y_data=processed_data.y_data,
            errors=processed_data.errors,
            metadata=processed_data.metadata or {},
        )

    def _failure_result(
        self, plot_id: str, error: str, started_at: datetime
    ) -> PlotResultDTO:
        """Create failure result."""
        completed_at = datetime.now()
        return PlotResultDTO(
            success=False,
            plot_id=plot_id,
            error=error,
            started_at=started_at,
            completed_at=completed_at,
            duration=completed_at - started_at,
        )
