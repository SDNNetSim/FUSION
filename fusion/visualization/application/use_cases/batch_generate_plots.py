"""Use case for batch generation of multiple plots."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import TYPE_CHECKING

from fusion.visualization.application.dto import (
    BatchPlotRequestDTO,
    BatchPlotResultDTO,
    PlotRequestDTO,
    PlotResultDTO,
)
from fusion.visualization.application.use_cases.generate_plot import GeneratePlotUseCase

if TYPE_CHECKING:
    from fusion.visualization.application.ports import (
        DataProcessorPort,
        PlotRendererPort,
    )
    from fusion.visualization.application.services.plot_service import PlotService

logger = logging.getLogger(__name__)


class BatchGeneratePlotsUseCase:
    """
    Use case for batch plot generation.

    This generates multiple plots either sequentially or in parallel,
    collecting results and providing overall batch statistics.
    """

    def __init__(
        self,
        generate_plot_use_case: GeneratePlotUseCase | None = None,
        plot_service: PlotService | None = None,  # Legacy parameter
        processor: DataProcessorPort | None = None,  # Legacy parameter
        renderer: PlotRendererPort | None = None,  # Legacy parameter
    ):
        """
        Initialize batch use case.

        Args:
            generate_plot_use_case: Single plot generation use case
            plot_service: Legacy parameter - will create GeneratePlotUseCase if provided
            processor: Legacy parameter - will create GeneratePlotUseCase if provided
            renderer: Legacy parameter - will create GeneratePlotUseCase if provided
        """
        # Handle legacy parameters by creating a GeneratePlotUseCase
        if generate_plot_use_case is None and (plot_service or processor or renderer):
            # Type checking: processor and renderer must be non-None
            if processor is not None and renderer is not None:
                generate_plot_use_case = GeneratePlotUseCase(
                    plot_service=plot_service,
                    processor=processor,
                    renderer=renderer,
                )

        self.generate_plot_use_case = generate_plot_use_case
        self.plot_service = plot_service
        self.processor = processor
        self.renderer = renderer

    def execute(
        self, request: BatchPlotRequestDTO | list[PlotRequestDTO]
    ) -> BatchPlotResultDTO | list[PlotResultDTO]:
        """
        Execute batch plot generation.

        Args:
            request: BatchPlotRequestDTO or List[PlotRequestDTO] (legacy)

        Returns:
            BatchPlotResultDTO or List[PlotResultDTO] (legacy)
        """
        started_at = datetime.now()

        # Handle legacy list input
        if isinstance(request, list):
            logger.info(
                f"Starting batch generation of {len(request)} plots (legacy list mode)"
            )
            # Create a BatchPlotRequestDTO from the list
            batch_request = BatchPlotRequestDTO(
                network=request[0].network if request else "",
                dates=request[0].dates if request else [],
                plots=request,
                parallel=False,
            )
            result = self._execute_batch(batch_request, started_at)
            # Return list of results for legacy compatibility
            return result.results

        # Normal BatchPlotRequestDTO handling
        logger.info(
            f"Starting batch generation of {len(request.plots)} plots "
            f"(parallel={request.parallel})"
        )

        return self._execute_batch(request, started_at)

    def _execute_batch(
        self, request: BatchPlotRequestDTO, started_at: datetime
    ) -> BatchPlotResultDTO:
        """Execute batch plot generation with a BatchPlotRequestDTO."""
        # Validate request
        errors = request.validate()
        if errors:
            logger.error(f"Invalid batch request: {'; '.join(errors)}")
            # Return empty results with errors
            return BatchPlotResultDTO(
                results=[],
                started_at=started_at,
                completed_at=datetime.now(),
                duration=datetime.now() - started_at,
            )

        # Generate plots
        if request.parallel:
            results = self._generate_parallel(request)
        else:
            results = self._generate_sequential(request)

        completed_at = datetime.now()
        duration = completed_at - started_at

        logger.info(
            f"Batch generation completed: "
            f"{sum(1 for r in results if r.success)}/{len(results)} successful "
            f"in {duration.total_seconds():.2f}s"
        )

        return BatchPlotResultDTO(
            results=results,
            started_at=started_at,
            completed_at=completed_at,
            duration=duration,
        )

    def _generate_sequential(self, request: BatchPlotRequestDTO) -> list[PlotResultDTO]:
        """Generate plots sequentially."""
        results = []

        for i, plot_request in enumerate(request.plots):
            logger.info(f"Generating plot {i + 1}/{len(request.plots)}")

            try:
                if self.generate_plot_use_case is None:
                    raise RuntimeError("generate_plot_use_case is not initialized")
                result = self.generate_plot_use_case.execute(plot_request)
                results.append(result)

                # Stop on error if requested
                if not result.success and request.stop_on_error:
                    logger.warning(
                        f"Stopping batch generation due to error: {result.error}"
                    )
                    break

            except Exception as e:
                logger.exception(f"Unexpected error generating plot {i}: {e}")
                results.append(
                    PlotResultDTO(
                        success=False,
                        plot_id=f"batch_{i}",
                        error=str(e),
                    )
                )

                if request.stop_on_error:
                    break

        return results

    def _generate_parallel(self, request: BatchPlotRequestDTO) -> list[PlotResultDTO]:
        """Generate plots in parallel using thread pool."""
        results: list[PlotResultDTO | None] = [None] * len(request.plots)

        if self.generate_plot_use_case is None:
            raise RuntimeError("generate_plot_use_case is not initialized")

        with ThreadPoolExecutor(max_workers=request.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self.generate_plot_use_case.execute, plot_request): i
                for i, plot_request in enumerate(request.plots)
            }

            # Collect results as they complete
            for future in as_completed(future_to_index):
                index = future_to_index[future]

                try:
                    result = future.result()
                    results[index] = result

                    if not result.success:
                        logger.warning(f"Plot {index} failed: {result.error}")

                except Exception as e:
                    logger.exception(f"Unexpected error in parallel plot {index}: {e}")
                    results[index] = PlotResultDTO(
                        success=False,
                        plot_id=f"batch_{index}",
                        error=str(e),
                    )

        # Filter out None values (shouldn't happen but type checker needs assurance)
        return [r for r in results if r is not None]
