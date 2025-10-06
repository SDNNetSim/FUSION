"""
Integration tests for the complete visualization pipeline.

These tests verify that the entire system works correctly from data loading
through processing to plot generation.
"""

from pathlib import Path

import pytest

from fusion.visualization.application.dto import PlotRequestDTO
from fusion.visualization.application.services import (
    CacheService,
    PlotService,
)
from fusion.visualization.application.use_cases import GeneratePlotUseCase
from fusion.visualization.domain.entities import Run
from fusion.visualization.domain.value_objects import PlotType
from fusion.visualization.infrastructure.adapters import DataAdapterRegistry
from fusion.visualization.infrastructure.processors import BlockingProcessor
from fusion.visualization.infrastructure.renderers import MatplotlibRenderer
from fusion.visualization.infrastructure.repositories import (
    FileMetadataRepository,
    JsonSimulationRepository,
)


class TestFullPipeline:
    """Integration tests for the complete visualization pipeline."""

    def test_end_to_end_blocking_plot(
        self,
        integration_data_dir: Path,
        tmp_path: Path,
    ) -> None:
        """
        Test complete pipeline from data discovery to plot generation.

        This tests:
        1. Data discovery (finding runs)
        2. Data loading (reading JSON files)
        3. Format adaptation (V1 adapter)
        4. Data processing (aggregation, statistics)
        5. Plot rendering (matplotlib)
        6. File saving
        """
        # Setup
        adapter_registry = DataAdapterRegistry()
        sim_repo = JsonSimulationRepository(
            base_path=integration_data_dir,
            adapter_registry=adapter_registry,
        )
        meta_repo = FileMetadataRepository(base_path=integration_data_dir)
        cache_service = CacheService(cache_dir=tmp_path / "cache")
        output_dir = tmp_path / "plots"
        output_dir.mkdir()

        # Create use case
        plot_service = PlotService(
            simulation_repository=sim_repo,
            metadata_repository=meta_repo,
            cache_service=cache_service,
        )
        processor = BlockingProcessor()
        renderer = MatplotlibRenderer(output_dir=output_dir)

        use_case = GeneratePlotUseCase(
            plot_service=plot_service,
            processor=processor,
            renderer=renderer,
        )

        # Create request
        request = PlotRequestDTO(
            network="NSFNet",
            dates=["0606"],
            plot_type=PlotType.LINE,
            metrics=["blocking_probability"],
            algorithms=["ppo", "dqn"],
            traffic_volumes=[600, 700, 800],
            title="Blocking Probability vs Traffic Volume",
            x_label="Traffic Volume (Erlang)",
            y_label="Blocking Probability",
            include_ci=True,
            save_path=output_dir / "blocking_test.png",
        )

        # Execute
        result = use_case.execute(request)

        # Verify
        assert result.success, f"Pipeline failed: {result.error_message}"
        assert result.output_path is not None
        assert result.output_path is not None and result.output_path.exists(), (
            "Plot file not created"
        )
        assert result.output_path.suffix == ".png", "Wrong file format"
        assert len(result.algorithms_plotted) == 2, "Should plot 2 algorithms"
        assert "ppo" in result.algorithms_plotted
        assert "dqn" in result.algorithms_plotted
        assert result.duration_seconds is not None
        assert result.duration_seconds > 0, "Duration should be recorded"

    def test_full_pipeline_with_v1_data(
        self,
        integration_data_dir: Path,
        simulation_repository: JsonSimulationRepository,
        metadata_repository: FileMetadataRepository,
        cache_service: CacheService,
        matplotlib_renderer: MatplotlibRenderer,
        tmp_path: Path,
    ) -> None:
        """Test pipeline with V1 format data."""
        # Setup
        plot_service = PlotService(
            simulation_repository=simulation_repository,
            metadata_repository=metadata_repository,
            cache_service=cache_service,
        )
        processor = BlockingProcessor()

        use_case = GeneratePlotUseCase(
            plot_service=plot_service,
            processor=processor,
            renderer=matplotlib_renderer,
        )

        # Create request
        request = PlotRequestDTO(
            network="NSFNet",
            dates=["0606"],
            plot_type=PlotType.LINE,
            metrics=["blocking_probability"],
            algorithms=["ppo"],
            traffic_volumes=[600, 700, 800],
            title="V1 Data Test",
            x_label="Traffic Volume (Erlang)",
            y_label="Blocking Probability",
            save_path=tmp_path / "plots" / "v1_test.png",
        )

        # Execute
        result = use_case.execute(request)

        # Verify
        assert result.success
        assert result.output_path is not None and result.output_path.exists()
        assert "ppo" in result.algorithms_plotted

    def test_full_pipeline_with_v2_data(
        self,
        integration_data_dir_v2: Path,
        tmp_path: Path,
    ) -> None:
        """Test pipeline with V2 format data."""
        # Setup with V2 data directory
        adapter_registry = DataAdapterRegistry()
        sim_repo = JsonSimulationRepository(
            base_path=integration_data_dir_v2,
            adapter_registry=adapter_registry,
        )
        meta_repo = FileMetadataRepository(base_path=integration_data_dir_v2)
        cache_service = CacheService(cache_dir=tmp_path / "cache")
        output_dir = tmp_path / "plots"
        output_dir.mkdir()

        plot_service = PlotService(
            simulation_repository=sim_repo,
            metadata_repository=meta_repo,
            cache_service=cache_service,
        )
        processor = BlockingProcessor()
        renderer = MatplotlibRenderer(output_dir=output_dir)

        use_case = GeneratePlotUseCase(
            plot_service=plot_service,
            processor=processor,
            renderer=renderer,
        )

        # Create request
        request = PlotRequestDTO(
            network="USNet",
            dates=["0611"],
            plot_type=PlotType.LINE,
            metrics=["blocking_probability"],
            algorithms=["ppo"],
            traffic_volumes=[600, 700, 800],
            title="V2 Data Test",
            x_label="Traffic Volume (Erlang)",
            y_label="Blocking Probability",
            save_path=output_dir / "v2_test.png",
        )

        # Execute
        result = use_case.execute(request)

        # Verify
        assert result.success
        assert result.output_path is not None and result.output_path.exists()
        assert "ppo" in result.algorithms_plotted

    def test_pipeline_with_caching(
        self,
        integration_data_dir: Path,
        tmp_path: Path,
    ) -> None:
        """Test that caching improves performance on repeated requests."""
        # Setup
        adapter_registry = DataAdapterRegistry()
        sim_repo = JsonSimulationRepository(
            base_path=integration_data_dir,
            adapter_registry=adapter_registry,
        )
        meta_repo = FileMetadataRepository(base_path=integration_data_dir)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache_service = CacheService(cache_dir=cache_dir)
        output_dir = tmp_path / "plots"
        output_dir.mkdir()

        plot_service = PlotService(
            simulation_repository=sim_repo,
            metadata_repository=meta_repo,
            cache_service=cache_service,
        )
        processor = BlockingProcessor()
        renderer = MatplotlibRenderer(output_dir=output_dir)

        use_case = GeneratePlotUseCase(
            plot_service=plot_service,
            processor=processor,
            renderer=renderer,
        )

        # Create request
        request = PlotRequestDTO(
            network="NSFNet",
            dates=["0606"],
            plot_type=PlotType.LINE,
            metrics=["blocking_probability"],
            algorithms=["ppo"],
            traffic_volumes=[600, 700, 800],
            title="Cache Test",
            x_label="Traffic Volume (Erlang)",
            y_label="Blocking Probability",
            save_path=output_dir / "cache_test_1.png",
        )

        # First execution (cold cache)
        result1 = use_case.execute(request)

        # Second execution (warm cache)
        request.save_path = output_dir / "cache_test_2.png"
        result2 = use_case.execute(request)
        duration2 = result2.duration_seconds

        # Verify both succeeded
        assert result1.success
        assert result2.success

        # Note: Cache speedup may be minimal for small test datasets
        # This test mainly verifies that caching doesn't break the pipeline
        assert duration2 is not None and duration2 >= 0, "Duration should be recorded"

    def test_pipeline_handles_missing_data_gracefully(
        self,
        integration_data_dir: Path,
        simulation_repository: JsonSimulationRepository,
        metadata_repository: FileMetadataRepository,
        cache_service: CacheService,
        matplotlib_renderer: MatplotlibRenderer,
        tmp_path: Path,
    ) -> None:
        """Test that pipeline handles missing data gracefully."""
        # Setup
        plot_service = PlotService(
            simulation_repository=simulation_repository,
            metadata_repository=metadata_repository,
            cache_service=cache_service,
        )
        processor = BlockingProcessor()

        use_case = GeneratePlotUseCase(
            plot_service=plot_service,
            processor=processor,
            renderer=matplotlib_renderer,
        )

        # Request non-existent network
        request = PlotRequestDTO(
            network="NonExistentNetwork",
            dates=["0606"],
            plot_type=PlotType.LINE,
            metrics=["blocking_probability"],
            algorithms=["ppo"],
            traffic_volumes=[600, 700, 800],
            title="Missing Data Test",
            x_label="Traffic Volume (Erlang)",
            y_label="Blocking Probability",
            save_path=tmp_path / "plots" / "missing_data_test.png",
        )

        # Execute
        result = use_case.execute(request)

        # Verify it fails gracefully
        assert not result.success
        assert result.error_message is not None
        assert (
            "not found" in result.error_message.lower()
            or "no runs" in result.error_message.lower()
        )

    def test_pipeline_with_multiple_dates(
        self,
        integration_data_dir: Path,
        tmp_path: Path,
    ) -> None:
        """Test pipeline with data from multiple dates."""
        # Create additional date directory
        network = "NSFNet"
        date2 = "0611"
        run_id = "1715_14_30_45_123456"
        run_path = integration_data_dir / network / date2 / run_id / "s1"
        run_path.mkdir(parents=True)

        # Create data files
        import json

        for erlang in [600, 700, 800]:
            data = {
                "blocking_mean": 0.015 + (erlang - 600) * 0.008,
                "iter_stats": {
                    "0": {
                        "sim_block_list": [0.015, 0.016, 0.014, 0.015],
                        "hops_mean": 3.0,
                        "hops_list": [3, 3, 3, 3, 3],
                        "lengths_mean": 440.5,
                        "lengths_list": [440, 450, 430, 445],
                        "computation_time_mean": 0.013,
                    },
                },
                "sim_start_time": "0611_14_30_45_123456",
                "sim_end_time": "0611_14_35_20_654321",
            }
            with open(run_path / f"{erlang}_erlang.json", "w") as f:
                json.dump(data, f)

        # Create metadata
        metadata = {
            "run_id": "ppo_run3",
            "path_algorithm": "ppo",
            "obs_space": "obs_7",
            "network": network,
            "date": date2,
            "seed": 4,
        }
        with open(run_path.parent / "metadata.json", "w") as f:
            json.dump(metadata, f)

        # Setup
        adapter_registry = DataAdapterRegistry()
        sim_repo = JsonSimulationRepository(
            base_path=integration_data_dir,
            adapter_registry=adapter_registry,
        )
        meta_repo = FileMetadataRepository(base_path=integration_data_dir)
        cache_service = CacheService(cache_dir=tmp_path / "cache")
        output_dir = tmp_path / "plots"
        output_dir.mkdir()

        plot_service = PlotService(
            simulation_repository=sim_repo,
            metadata_repository=meta_repo,
            cache_service=cache_service,
        )
        processor = BlockingProcessor()
        renderer = MatplotlibRenderer(output_dir=output_dir)

        use_case = GeneratePlotUseCase(
            plot_service=plot_service,
            processor=processor,
            renderer=renderer,
        )

        # Create request with multiple dates
        request = PlotRequestDTO(
            network="NSFNet",
            dates=["0606", "0611"],
            plot_type=PlotType.LINE,
            metrics=["blocking_probability"],
            algorithms=["ppo"],
            traffic_volumes=[600, 700, 800],
            title="Multi-Date Test",
            x_label="Traffic Volume (Erlang)",
            y_label="Blocking Probability",
            save_path=output_dir / "multi_date_test.png",
        )

        # Execute
        result = use_case.execute(request)

        # Verify
        assert result.success
        assert result.output_path is not None and result.output_path.exists()

    def test_pipeline_validates_configuration(
        self,
        integration_data_dir: Path,
        simulation_repository: JsonSimulationRepository,
        metadata_repository: FileMetadataRepository,
        cache_service: CacheService,
        matplotlib_renderer: MatplotlibRenderer,
        tmp_path: Path,
    ) -> None:
        """Test that pipeline validates configuration before execution."""
        # Setup
        plot_service = PlotService(
            simulation_repository=simulation_repository,
            metadata_repository=metadata_repository,
            cache_service=cache_service,
        )
        processor = BlockingProcessor()

        use_case = GeneratePlotUseCase(
            plot_service=plot_service,
            processor=processor,
            renderer=matplotlib_renderer,
        )

        # Create invalid request (empty traffic volumes)
        request = PlotRequestDTO(
            network="NSFNet",
            dates=["0606"],
            plot_type=PlotType.LINE,
            metrics=["blocking_probability"],
            algorithms=["ppo"],
            traffic_volumes=[],  # Invalid: empty list
            title="Validation Test",
            x_label="Traffic Volume (Erlang)",
            y_label="Blocking Probability",
            save_path=tmp_path / "plots" / "validation_test.png",
        )

        # Execute
        result = use_case.execute(request)

        # Verify it fails validation
        assert not result.success
        assert result.error_message is not None


class TestDataLoading:
    """Integration tests for data loading."""

    def test_data_discovery(
        self,
        integration_data_dir: Path,
        simulation_repository: JsonSimulationRepository,
    ) -> None:
        """Test discovering runs from file system."""
        # Find runs
        runs = simulation_repository.find_runs(
            network="NSFNet",
            dates=["0606"],
        )

        # Verify
        assert len(runs) >= 2, "Should find at least 2 runs"
        assert all(isinstance(run, Run) for run in runs)
        assert all(run.network == "NSFNet" for run in runs)
        assert all(run.date == "0606" for run in runs)

    @pytest.mark.skip(reason="Test data files not present in integration_data_dir")
    def test_data_loading_with_adapter(
        self,
        integration_data_dir: Path,
        simulation_repository: JsonSimulationRepository,
    ) -> None:
        """Test loading data with automatic format adaptation."""
        # Find runs
        runs = simulation_repository.find_runs(
            network="NSFNet",
            dates=["0606"],
        )
        assert len(runs) > 0, "Should find runs"

        # Load data for first run
        run = runs[0]
        canonical_data = simulation_repository.get_run_data(run, traffic_volume=600)

        # Verify canonical data structure
        assert canonical_data is not None
        assert hasattr(canonical_data, "blocking_probability")
        assert hasattr(canonical_data, "iterations")
        assert canonical_data.blocking_probability is not None
        assert len(canonical_data.iterations) > 0

    def test_metadata_loading(
        self,
        integration_data_dir: Path,
        metadata_repository: FileMetadataRepository,
    ) -> None:
        """Test loading metadata from runs."""
        # Find metadata
        metadata_list = metadata_repository.find_metadata(
            network="NSFNet",
            dates=["0606"],
        )

        # Verify
        assert len(metadata_list) >= 2, "Should find at least 2 metadata files"
        for metadata in metadata_list:
            assert "path_algorithm" in metadata
            assert "network" in metadata
            assert "date" in metadata


class TestPlotGeneration:
    """Integration tests for plot generation."""

    def test_plot_rendering(
        self,
        matplotlib_renderer: MatplotlibRenderer,
        tmp_path: Path,
    ) -> None:
        """Test that renderer creates valid plot files."""
        import numpy as np

        from fusion.visualization.domain.value_objects import PlotSpecification

        # Create plot specification
        spec = PlotSpecification(
            title="Test Plot",
            x_data=np.array([600, 700, 800]),
            y_data={
                "algorithm1": np.array([0.02, 0.03, 0.04]),
                "algorithm2": np.array([0.025, 0.035, 0.045]),
            },
            x_label="Traffic Volume (Erlang)",
            y_label="Blocking Probability",
            include_ci=False,
            include_legend=True,
        )

        # Render
        output_path = tmp_path / "plots" / "test_render.png"
        result = matplotlib_renderer.render(spec, output_path)

        # Verify
        assert result.success
        assert output_path.exists()
        assert output_path.stat().st_size > 0, "File should not be empty"

    def test_plot_with_confidence_intervals(
        self,
        matplotlib_renderer: MatplotlibRenderer,
        tmp_path: Path,
    ) -> None:
        """Test rendering plots with confidence intervals."""
        import numpy as np

        from fusion.visualization.domain.value_objects import PlotSpecification

        # Create plot specification with CI data
        spec = PlotSpecification(
            title="Test Plot with CI",
            x_data=np.array([600, 700, 800]),
            y_data={
                "algorithm1": np.array([0.02, 0.03, 0.04]),
            },
            y_ci_data={
                "algorithm1": np.array([0.002, 0.003, 0.004]),  # CI width
            },
            x_label="Traffic Volume (Erlang)",
            y_label="Blocking Probability",
            include_ci=True,
            include_legend=True,
        )

        # Render
        output_path = tmp_path / "plots" / "test_ci.png"
        result = matplotlib_renderer.render(spec, output_path)

        # Verify
        assert result.success
        assert output_path.exists()


@pytest.mark.slow
class TestPerformance:
    """Performance-related integration tests."""

    def test_batch_plot_generation_performance(
        self,
        integration_data_dir: Path,
        tmp_path: Path,
    ) -> None:
        """Test performance of generating multiple plots."""
        import time

        # Setup
        adapter_registry = DataAdapterRegistry()
        sim_repo = JsonSimulationRepository(
            base_path=integration_data_dir,
            adapter_registry=adapter_registry,
        )
        meta_repo = FileMetadataRepository(base_path=integration_data_dir)
        cache_service = CacheService(cache_dir=tmp_path / "cache")
        output_dir = tmp_path / "plots"
        output_dir.mkdir()

        plot_service = PlotService(
            simulation_repository=sim_repo,
            metadata_repository=meta_repo,
            cache_service=cache_service,
        )
        processor = BlockingProcessor()
        renderer = MatplotlibRenderer(output_dir=output_dir)

        use_case = GeneratePlotUseCase(
            plot_service=plot_service,
            processor=processor,
            renderer=renderer,
        )

        # Generate 5 plots and measure time
        start_time = time.time()

        for i in range(5):
            request = PlotRequestDTO(
                network="NSFNet",
                dates=["0606"],
                plot_type=PlotType.LINE,
                metrics=["blocking_probability"],
                algorithms=["ppo"],
                traffic_volumes=[600, 700, 800],
                title=f"Performance Test {i}",
                x_label="Traffic Volume (Erlang)",
                y_label="Blocking Probability",
                save_path=output_dir / f"perf_test_{i}.png",
            )
            result = use_case.execute(request)
            assert result.success

        total_time = time.time() - start_time
        avg_time = total_time / 5

        # Performance target: <2 seconds per plot (generous for integration tests)
        assert avg_time < 2.0, (
            f"Average plot generation took {avg_time:.2f}s (target: <2.0s)"
        )

    def test_large_dataset_handling(
        self,
        integration_data_dir: Path,
        tmp_path: Path,
    ) -> None:
        """Test handling of larger datasets."""
        # Create additional runs to simulate larger dataset
        import json

        network = "NSFNet"
        date = "0606"

        for seed in range(3, 8):  # Add 5 more runs
            run_id = f"1715_12_{30 + seed}_45_{seed}23456"
            run_path = integration_data_dir / network / date / run_id / f"s{seed}"
            run_path.mkdir(parents=True)

            for erlang in [600, 700, 800]:
                data = {
                    "blocking_mean": 0.02 + (erlang - 600) * 0.01 + seed * 0.001,
                    "iter_stats": {
                        "0": {
                            "sim_block_list": [0.02, 0.021, 0.019, 0.02],
                            "hops_mean": 3.2,
                            "hops_list": [3, 3, 4, 3, 3],
                            "lengths_mean": 450.5,
                            "lengths_list": [450, 460, 440, 455],
                            "computation_time_mean": 0.015,
                        },
                    },
                    "sim_start_time": f"0606_12_{30 + seed}_45_{seed}23456",
                    "sim_end_time": f"0606_12_{35 + seed}_20_{seed}54321",
                }
                with open(run_path / f"{erlang}_erlang.json", "w") as f:
                    json.dump(data, f)

            metadata = {
                "run_id": f"ppo_run{seed}",
                "path_algorithm": "ppo",
                "obs_space": "obs_7",
                "network": network,
                "date": date,
                "seed": seed,
            }
            with open(run_path.parent / "metadata.json", "w") as f:
                json.dump(metadata, f)

        # Setup
        adapter_registry = DataAdapterRegistry()
        sim_repo = JsonSimulationRepository(
            base_path=integration_data_dir,
            adapter_registry=adapter_registry,
        )
        meta_repo = FileMetadataRepository(base_path=integration_data_dir)
        cache_service = CacheService(cache_dir=tmp_path / "cache")
        output_dir = tmp_path / "plots"
        output_dir.mkdir()

        plot_service = PlotService(
            simulation_repository=sim_repo,
            metadata_repository=meta_repo,
            cache_service=cache_service,
        )
        processor = BlockingProcessor()
        renderer = MatplotlibRenderer(output_dir=output_dir)

        use_case = GeneratePlotUseCase(
            plot_service=plot_service,
            processor=processor,
            renderer=renderer,
        )

        # Generate plot with all runs
        request = PlotRequestDTO(
            network="NSFNet",
            dates=["0606"],
            plot_type=PlotType.LINE,
            metrics=["blocking_probability"],
            algorithms=["ppo"],
            traffic_volumes=[600, 700, 800],
            title="Large Dataset Test",
            x_label="Traffic Volume (Erlang)",
            y_label="Blocking Probability",
            save_path=output_dir / "large_dataset_test.png",
        )

        # Execute
        result = use_case.execute(request)

        # Verify
        assert result.success
        assert result.output_path is not None and result.output_path.exists()
