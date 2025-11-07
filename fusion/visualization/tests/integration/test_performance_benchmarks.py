"""
Performance benchmarks for visualization system.

These tests measure and compare performance metrics between old and new systems.
They help ensure that the new system meets or exceeds performance targets.
"""

import json
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest

from fusion.visualization.application.dto import PlotRequestDTO
from fusion.visualization.application.services import CacheService, PlotService
from fusion.visualization.application.use_cases import (
    BatchGeneratePlotsUseCase,
    GeneratePlotUseCase,
)
from fusion.visualization.domain.value_objects import PlotType
from fusion.visualization.infrastructure.adapters import DataAdapterRegistry
from fusion.visualization.infrastructure.processors import BlockingProcessor
from fusion.visualization.infrastructure.renderers import MatplotlibRenderer
from fusion.visualization.infrastructure.repositories import (
    FileMetadataRepository,
    JsonSimulationRepository,
)


class PerformanceTimer:
    """Context manager for timing code execution."""

    def __init__(self, name: str):
        self.name = name
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.duration: float | None = None

    def __enter__(self) -> "PerformanceTimer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> Literal[False]:
        self.end_time = time.perf_counter()
        assert self.start_time is not None  # For type checker
        self.duration = self.end_time - self.start_time
        return False


class PerformanceBenchmark:
    """Base class for performance benchmarks."""

    @staticmethod
    def create_large_dataset(base_path: Path, num_runs: int = 10) -> Path:
        """
        Create a large dataset for performance testing.

        Args:
            base_path: Base directory for data
            num_runs: Number of runs to create

        Returns:
            Path to created dataset
        """
        network = "NSFNet"
        date = "0606"

        for run_idx in range(num_runs):
            run_id = f"1715_12_{30 + run_idx}_45_{run_idx}23456"
            run_path = base_path / network / date / run_id / f"s{run_idx}"
            run_path.mkdir(parents=True, exist_ok=True)

            # Create data files for multiple traffic volumes
            for erlang in range(600, 1100, 50):  # 10 traffic volumes
                data = {
                    "blocking_mean": 0.01 + (erlang - 600) * 0.00005 + run_idx * 0.001,
                    "iter_stats": {},
                }

                # Add 100 iterations
                iter_stats: dict[str, Any] = {}
                for iter_idx in range(100):
                    iter_stats[str(iter_idx)] = {
                        "sim_block_list": [
                            0.01 + np.random.normal(0, 0.001) for _ in range(100)
                        ],
                        "hops_mean": 3.2 + np.random.normal(0, 0.1),
                        "hops_list": [3, 3, 4, 3, 3, 4, 3, 3, 3, 4],
                        "lengths_mean": 450.5 + np.random.normal(0, 10),
                        "lengths_list": [
                            450 + int(np.random.normal(0, 20)) for _ in range(10)
                        ],
                        "computation_time_mean": 0.015 + np.random.normal(0, 0.002),
                    }
                data["iter_stats"] = iter_stats

                data["sim_start_time"] = f"0606_12_{30 + run_idx}_45_{run_idx}23456"
                data["sim_end_time"] = f"0606_12_{35 + run_idx}_20_{run_idx}54321"

                file_path = run_path / f"{erlang}_erlang.json"
                with open(file_path, "w") as f:
                    json.dump(data, f)

            # Create metadata
            metadata = {
                "run_id": f"ppo_run{run_idx}",
                "path_algorithm": "ppo",
                "obs_space": "obs_7",
                "network": network,
                "date": date,
                "seed": run_idx,
            }
            with open(run_path.parent / "metadata.json", "w") as f:
                json.dump(metadata, f)

        return base_path


@pytest.mark.benchmark
class TestDataLoadingPerformance:
    """Benchmarks for data loading operations."""

    def test_run_discovery_performance(
        self,
        integration_data_dir: Path,
    ) -> None:
        """
        Benchmark: Time to discover runs in a directory.

        Target: <100ms for typical dataset (2-5 runs)
        """
        adapter_registry = DataAdapterRegistry()
        sim_repo = JsonSimulationRepository(
            base_path=integration_data_dir,
            adapter_registry=adapter_registry,
        )

        # Warm up
        _ = sim_repo.find_runs(network="NSFNet", dates=["0606"])

        # Benchmark
        with PerformanceTimer("run_discovery") as timer:
            runs = sim_repo.find_runs(network="NSFNet", dates=["0606"])

        # Verify
        assert len(runs) > 0, "No runs found"
        assert timer.duration is not None and timer.duration < 0.1, (
            f"Run discovery took {timer.duration:.3f}s (target: <0.1s)"
        )

        print(
            f"\n  Run discovery: {(timer.duration or 0) * 1000:.2f}ms "
            f"for {len(runs)} runs"
        )

    @pytest.mark.skip(reason="Test data files not present in integration_data_dir")
    def test_single_file_loading_performance(
        self,
        integration_data_dir: Path,
    ) -> None:
        """
        Benchmark: Time to load a single data file.

        Target: <50ms per file
        """
        adapter_registry = DataAdapterRegistry()
        sim_repo = JsonSimulationRepository(
            base_path=integration_data_dir,
            adapter_registry=adapter_registry,
        )

        # Find a run
        runs = sim_repo.find_runs(network="NSFNet", dates=["0606"])
        assert len(runs) > 0
        run = runs[0]

        # Warm up
        _ = sim_repo.get_run_data(run, traffic_volume=600)

        # Benchmark
        with PerformanceTimer("file_loading") as timer:
            data = sim_repo.get_run_data(run, traffic_volume=600)

        # Verify
        assert data is not None
        assert timer.duration is not None and timer.duration < 0.05, (
            f"File loading took {timer.duration:.3f}s (target: <0.05s)"
        )

        print(f"\n  File loading: {(timer.duration or 0) * 1000:.2f}ms")

    @pytest.mark.skip(reason="Test data files not present in integration_data_dir")
    def test_multiple_files_loading_performance(
        self,
        integration_data_dir: Path,
    ) -> None:
        """
        Benchmark: Time to load multiple data files.

        Target: <500ms for 10 files
        """
        adapter_registry = DataAdapterRegistry()
        sim_repo = JsonSimulationRepository(
            base_path=integration_data_dir,
            adapter_registry=adapter_registry,
        )

        # Find runs
        runs = sim_repo.find_runs(network="NSFNet", dates=["0606"])
        assert len(runs) > 0
        run = runs[0]

        # Benchmark loading 3 traffic volumes
        traffic_volumes = [600, 700, 800]

        with PerformanceTimer("multiple_files") as timer:
            for tv in traffic_volumes:
                _ = sim_repo.get_run_data(run, traffic_volume=tv)

        # Verify
        avg_time = (timer.duration or 0) / len(traffic_volumes)
        assert timer.duration is not None and timer.duration < 0.5, (
            f"Loading {len(traffic_volumes)} files took {timer.duration:.3f}s "
            f"(target: <0.5s)"
        )

        print(
            f"\n  Multiple files: {(timer.duration or 0) * 1000:.2f}ms total, "
            f"{avg_time * 1000:.2f}ms per file"
        )

    def test_data_adaptation_performance(
        self,
        integration_data_dir: Path,
    ) -> None:
        """
        Benchmark: Time for format adaptation.

        Target: <10ms per file
        """
        adapter_registry = DataAdapterRegistry()

        # Load raw data
        data_file = (
            integration_data_dir
            / "NSFNet"
            / "0606"
            / "1715_12_30_45_123456"
            / "s1"
            / "600_erlang.json"
        )
        with open(data_file) as f:
            raw_data = json.load(f)

        # Get adapter
        adapter = adapter_registry.get_adapter(raw_data)

        # Warm up
        _ = adapter.to_canonical(raw_data)

        # Benchmark
        with PerformanceTimer("adaptation") as timer:
            canonical = adapter.to_canonical(raw_data)

        # Verify
        assert canonical is not None
        assert timer.duration is not None and timer.duration < 0.01, (
            f"Adaptation took {timer.duration:.3f}s (target: <0.01s)"
        )

        print(f"\n  Data adaptation: {(timer.duration or 0) * 1000:.2f}ms")


@pytest.mark.benchmark
class TestProcessingPerformance:
    """Benchmarks for data processing operations."""

    def test_blocking_processor_performance(
        self,
        integration_data_dir: Path,
    ) -> None:
        """
        Benchmark: Time to process blocking probability data.

        Target: <200ms for standard dataset
        """
        adapter_registry = DataAdapterRegistry()
        sim_repo = JsonSimulationRepository(
            base_path=integration_data_dir,
            adapter_registry=adapter_registry,
        )

        # Find runs
        runs = sim_repo.find_runs(network="NSFNet", dates=["0606"], algorithm="ppo")
        assert len(runs) > 0

        # Load data
        traffic_volumes = [600, 700, 800]
        run_data: dict[str, dict[float, Any]] = {}
        for run in runs:
            run_data[run.id] = {}
            for tv in traffic_volumes:
                run_data[run.id][tv] = sim_repo.get_run_data(run, traffic_volume=tv)

        # Create processor
        processor = BlockingProcessor()

        # Warm up
        _ = processor.process(
            runs=runs,
            run_data=run_data,
            traffic_volumes=[float(v) for v in traffic_volumes],
        )

        # Benchmark
        with PerformanceTimer("processing") as timer:
            processed = processor.process(
                runs=runs,
                run_data=run_data,
                traffic_volumes=[float(v) for v in traffic_volumes],
            )

        # Verify
        assert processed is not None
        assert timer.duration is not None and timer.duration < 0.2, (
            f"Processing took {timer.duration:.3f}s (target: <0.2s)"
        )

        print(
            f"\n  Data processing: {(timer.duration or 0) * 1000:.2f}ms for "
            f"{len(runs)} runs, {len(traffic_volumes)} traffic volumes"
        )

    def test_aggregation_performance(
        self,
        integration_data_dir: Path,
    ) -> None:
        """
        Benchmark: Time to aggregate statistics across runs.

        Target: <100ms for typical dataset
        """
        from fusion.visualization.domain.services import MetricAggregationService

        adapter_registry = DataAdapterRegistry()
        sim_repo = JsonSimulationRepository(
            base_path=integration_data_dir,
            adapter_registry=adapter_registry,
        )

        # Find runs
        runs = sim_repo.find_runs(network="NSFNet", dates=["0606"], algorithm="ppo")
        if len(runs) < 2:
            pytest.skip("Need at least 2 runs for aggregation test")

        # Load data
        from fusion.visualization.domain.entities.metric import DataType
        from fusion.visualization.domain.value_objects.metric_value import MetricValue

        values: list[MetricValue] = []
        for run in runs:
            data = sim_repo.get_run_data(run, traffic_volume=600)
            if data.blocking_probability is not None:
                values.append(
                    MetricValue(
                        value=data.blocking_probability, data_type=DataType.FLOAT
                    )
                )

        # Create service
        aggregation_service = MetricAggregationService()

        # Warm up
        _ = aggregation_service.compute_statistics(values)

        # Benchmark
        with PerformanceTimer("aggregation") as timer:
            stats = aggregation_service.compute_statistics(values)

        # Verify
        assert stats is not None
        assert timer.duration is not None and timer.duration < 0.1, (
            f"Aggregation took {timer.duration:.3f}s (target: <0.1s)"
        )

        print(
            f"\n  Aggregation: {(timer.duration or 0) * 1000:.2f}ms "
            f"for {len(values)} values"
        )


@pytest.mark.benchmark
class TestRenderingPerformance:
    """Benchmarks for plot rendering operations."""

    def test_simple_plot_rendering_performance(
        self,
        tmp_path: Path,
    ) -> None:
        """
        Benchmark: Time to render a simple plot.

        Target: <600ms
        """
        from fusion.visualization.domain.value_objects import PlotSpecification

        output_dir = tmp_path / "plots"
        output_dir.mkdir()
        renderer = MatplotlibRenderer(output_dir=output_dir)

        # Create simple plot spec
        spec = PlotSpecification(
            title="Performance Test",
            x_data=np.array([600, 700, 800]),
            y_data={
                "algo1": np.array([0.02, 0.03, 0.04]),
                "algo2": np.array([0.025, 0.035, 0.045]),
            },
            x_label="Traffic Volume (Erlang)",
            y_label="Blocking Probability",
            include_ci=False,
            include_legend=True,
        )

        output_path = output_dir / "perf_test.png"

        # Warm up
        _ = renderer.render(spec, output_path)
        output_path.unlink()  # Delete warm-up file

        # Benchmark
        with PerformanceTimer("rendering") as timer:
            result = renderer.render(spec, output_path)

        # Verify
        assert result.success
        assert output_path.exists()
        assert timer.duration is not None and timer.duration < 0.6, (
            f"Rendering took {timer.duration:.3f}s (target: <0.6s)"
        )

        print(f"\n  Plot rendering: {(timer.duration or 0) * 1000:.2f}ms")

    def test_complex_plot_rendering_performance(
        self,
        tmp_path: Path,
    ) -> None:
        """
        Benchmark: Time to render a complex plot with CI.

        Target: <1.0s
        """
        from fusion.visualization.domain.value_objects import PlotSpecification

        output_dir = tmp_path / "plots"
        output_dir.mkdir()
        renderer = MatplotlibRenderer(output_dir=output_dir)

        # Create complex plot spec with 5 algorithms
        algorithms = [f"algo{i}" for i in range(5)]
        x_data = np.array([600, 650, 700, 750, 800, 850, 900, 950, 1000])
        y_data = {}
        y_ci_data = {}

        for algo in algorithms:
            y_data[algo] = np.array([0.01 + i * 0.005 for i in range(len(x_data))])
            y_ci_data[algo] = np.array([0.001 for _ in range(len(x_data))])

        spec = PlotSpecification(
            title="Complex Performance Test",
            x_data=x_data,
            y_data=y_data,
            y_ci_data=y_ci_data,
            x_label="Traffic Volume (Erlang)",
            y_label="Blocking Probability",
            include_ci=True,
            include_legend=True,
        )

        output_path = output_dir / "complex_perf_test.png"

        # Benchmark
        with PerformanceTimer("complex_rendering") as timer:
            result = renderer.render(spec, output_path)

        # Verify
        assert result.success
        assert output_path.exists()
        assert timer.duration is not None and timer.duration < 1.0, (
            f"Complex rendering took {timer.duration:.3f}s (target: <1.0s)"
        )

        print(
            f"\n  Complex plot rendering: {(timer.duration or 0) * 1000:.2f}ms "
            f"({len(algorithms)} algorithms, {len(x_data)} points, with CI)"
        )


@pytest.mark.benchmark
@pytest.mark.slow
class TestEndToEndPerformance:
    """End-to-end performance benchmarks."""

    def test_single_plot_generation_performance(
        self,
        integration_data_dir: Path,
        tmp_path: Path,
    ) -> None:
        """
        Benchmark: End-to-end time for single plot generation.

        Target: <1 second
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
            title="E2E Performance Test",
            x_label="Traffic Volume (Erlang)",
            y_label="Blocking Probability",
            save_path=output_dir / "e2e_perf.png",
        )

        # Warm up (first execution may be slower due to cold start)
        _ = use_case.execute(request)
        request.save_path = output_dir / "e2e_perf_warmup.png"

        # Benchmark
        request.save_path = output_dir / "e2e_perf_actual.png"
        with PerformanceTimer("e2e_single") as timer:
            result = use_case.execute(request)

        # Verify
        assert result.success
        assert timer.duration is not None and timer.duration < 1.0, (
            f"E2E single plot took {timer.duration:.3f}s (target: <1.0s)"
        )

        print(f"\n  End-to-end single plot: {(timer.duration or 0) * 1000:.2f}ms")

    def test_batch_plot_generation_performance(
        self,
        integration_data_dir: Path,
        tmp_path: Path,
    ) -> None:
        """
        Benchmark: Time to generate multiple plots.

        Target: <8 seconds for 10 plots
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

        plot_service = PlotService(
            simulation_repository=sim_repo,
            metadata_repository=meta_repo,
            cache_service=cache_service,
        )
        processor = BlockingProcessor()
        renderer = MatplotlibRenderer(output_dir=output_dir)

        use_case = BatchGeneratePlotsUseCase(
            plot_service=plot_service,
            processor=processor,
            renderer=renderer,
        )

        # Create 10 plot requests
        requests = []
        for i in range(10):
            requests.append(
                PlotRequestDTO(
                    network="NSFNet",
                    dates=["0606"],
                    plot_type=PlotType.LINE,
                    metrics=["blocking_probability"],
                    algorithms=["ppo"],
                    traffic_volumes=[600, 700, 800],
                    title=f"Batch Test {i}",
                    x_label="Traffic Volume (Erlang)",
                    y_label="Blocking Probability",
                    save_path=output_dir / f"batch_perf_{i}.png",
                )
            )

        # Benchmark
        with PerformanceTimer("batch") as timer:
            result = use_case.execute(requests)

        # Verify - handle both return types
        if isinstance(result, list):
            results = result
        else:
            results = result.results
        assert all(r.success for r in results), "Some plots failed"
        avg_time = (timer.duration or 0) / len(requests)
        assert timer.duration is not None and timer.duration < 8.0, (
            f"Batch generation took {timer.duration:.3f}s (target: <8.0s)"
        )

        print(
            f"\n  Batch plot generation: {(timer.duration or 0):.3f}s total, "
            f"{avg_time * 1000:.2f}ms per plot"
        )

    @pytest.mark.skip(
        reason="Cache performance is too variable in CI environments. "
        "Test passes locally but fails in CI due to disk I/O and system load variance."
    )
    def test_cached_performance_improvement(
        self,
        integration_data_dir: Path,
        tmp_path: Path,
    ) -> None:
        """
        Benchmark: Verify caching improves performance.

        Target: Second execution should be faster
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
        with PerformanceTimer("cold") as timer1:
            result1 = use_case.execute(request)

        # Second execution (warm cache)
        request.save_path = output_dir / "cache_test_2.png"
        with PerformanceTimer("warm") as timer2:
            result2 = use_case.execute(request)

        # Verify
        assert result1.success
        assert result2.success

        print("\n  Cache performance:")
        print(f"    Cold cache: {(timer1.duration or 0) * 1000:.2f}ms")
        print(f"    Warm cache: {(timer2.duration or 0) * 1000:.2f}ms")
        if timer1.duration and timer2.duration and timer2.duration > 0:
            print(f"    Speedup: {timer1.duration / timer2.duration:.2f}x")

        # Note: For small test datasets, cache speedup may be minimal
        # The important thing is that caching doesn't dramatically slow things down
        # Allow 2x variation to account for CI environment variability
        assert (
            timer1.duration is not None
            and timer2.duration is not None
            and timer2.duration <= timer1.duration * 2.0
        ), "Cached execution should not be significantly slower than cold execution"


@pytest.mark.benchmark
@pytest.mark.slow
class TestScalabilityBenchmarks:
    """Benchmarks for system scalability."""

    def test_performance_scales_with_data_size(
        self,
        tmp_path: Path,
    ) -> None:
        """
        Benchmark: Verify performance scales reasonably with data size.

        Tests with increasing numbers of runs to ensure no exponential slowdown.
        """
        # Create datasets of different sizes
        test_cases = [
            ("small", 2),
            ("medium", 5),
            ("large", 10),
        ]

        results = {}

        for name, num_runs in test_cases:
            # Create dataset
            data_dir = tmp_path / f"data_{name}"
            data_dir.mkdir()
            benchmark = PerformanceBenchmark()
            benchmark.create_large_dataset(data_dir, num_runs=num_runs)

            # Setup
            adapter_registry = DataAdapterRegistry()
            sim_repo = JsonSimulationRepository(
                base_path=data_dir,
                adapter_registry=adapter_registry,
            )
            cache_service = CacheService(cache_dir=tmp_path / f"cache_{name}")
            output_dir = tmp_path / f"plots_{name}"
            output_dir.mkdir()

            plot_service = PlotService(
                simulation_repository=sim_repo,
                metadata_repository=None,
                cache_service=cache_service,
            )
            processor = BlockingProcessor()
            renderer = MatplotlibRenderer(output_dir=output_dir)

            use_case = GeneratePlotUseCase(
                plot_service=plot_service,
                processor=processor,
                renderer=renderer,
            )

            request = PlotRequestDTO(
                network="NSFNet",
                dates=["0606"],
                plot_type=PlotType.LINE,
                metrics=["blocking_probability"],
                algorithms=["ppo"],
                traffic_volumes=[600, 700, 800, 900, 1000],
                title=f"Scalability Test - {name}",
                x_label="Traffic Volume (Erlang)",
                y_label="Blocking Probability",
                save_path=output_dir / f"scalability_{name}.png",
            )

            # Benchmark
            with PerformanceTimer(name) as timer:
                result = use_case.execute(request)

            assert result.success
            assert timer.duration is not None  # For type checker
            results[name] = {
                "num_runs": num_runs,
                "duration": timer.duration,
                "per_run": timer.duration / num_runs,
            }

        # Verify scaling is reasonable (should be roughly linear)
        print("\n  Scalability results:")
        for name, data in results.items():
            print(
                f"    {name:8s}: {(data['duration'] or 0) * 1000:6.2f}ms total, "
                f"{(data['per_run'] or 0) * 1000:6.2f}ms per run "
                f"({data['num_runs']} runs)"
            )

        # Check that performance doesn't degrade exponentially
        small_per_run = results["small"]["per_run"]
        large_per_run = results["large"]["per_run"]
        if small_per_run and large_per_run and small_per_run > 0:
            degradation = large_per_run / small_per_run
        else:
            degradation = 1.0

        assert degradation < 3.0, (
            f"Performance degraded {degradation:.1f}x (should be <3x)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "benchmark"])
