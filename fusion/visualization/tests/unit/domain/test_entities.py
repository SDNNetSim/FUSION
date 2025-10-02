"""Unit tests for domain entities."""

from pathlib import Path
from typing import Any

import pytest

from fusion.visualization.domain.entities import (
    DataFormat,
    DataSource,
    MetricDefinition,
    Plot,
    PlotConfiguration,
    PlotState,
    Run,
    SourceType,
)
from fusion.visualization.domain.exceptions import (
    InvalidMetricPathError,
    InvalidStateError,
    MetricExtractionError,
    ValidationError,
)
from fusion.visualization.domain.value_objects import (
    DataType,
    DataVersion,
    PlotId,
    PlotSpecification,
    PlotType,
)


class TestRun:
    """Tests for Run entity."""

    def test_create_run(self) -> None:
        """Should create Run entity."""
        run = Run(
            id="timestamp123",
            network="NSFNet",
            date="0606",
            algorithm="ppo_obs_7",
            path=Path("/data/NSFNet/0606/timestamp123"),
        )
        assert run.id == "timestamp123"
        assert run.network == "NSFNet"
        assert run.algorithm == "ppo_obs_7"

    def test_full_id_property(self) -> None:
        """Should return fully qualified ID."""
        run = Run(
            id="timestamp123",
            network="NSFNet",
            date="0606",
            algorithm="ppo_obs_7",
            path=Path("/data/NSFNet/0606/timestamp123"),
        )
        assert run.full_id == "NSFNet/0606/timestamp123"

    def test_get_data_file(self) -> None:
        """Should construct data file path correctly."""
        run = Run(
            id="timestamp123",
            network="NSFNet",
            date="0606",
            algorithm="ppo_obs_7",
            path=Path("/data/NSFNet/0606/timestamp123"),
        )

        data_file = run.get_data_file(150)
        assert data_file == Path("/data/NSFNet/0606/timestamp123/150_erlang.json")

    def test_get_data_file_float_volume(self) -> None:
        """Should handle float traffic volumes."""
        run = Run(
            id="timestamp123",
            network="NSFNet",
            date="0606",
            algorithm="ppo_obs_7",
            path=Path("/data"),
        )

        data_file = run.get_data_file(150.5)
        assert data_file == Path("/data/150.5_erlang.json")

    def test_run_equality(self) -> None:
        """Should compare runs for equality."""
        run1 = Run(
            id="timestamp123",
            network="NSFNet",
            date="0606",
            algorithm="ppo_obs_7",
            path=Path("/data"),
        )
        run2 = Run(
            id="timestamp123",
            network="NSFNet",
            date="0606",
            algorithm="ppo_obs_7",
            path=Path("/different/path"),
        )
        run3 = Run(
            id="timestamp456",
            network="NSFNet",
            date="0606",
            algorithm="ppo_obs_7",
            path=Path("/data"),
        )

        assert run1 == run2  # Same ID, different path
        assert run1 != run3  # Different ID

    def test_run_hashable(self) -> None:
        """Should be usable in sets and dicts."""
        run1 = Run(
            id="timestamp123",
            network="NSFNet",
            date="0606",
            algorithm="ppo_obs_7",
            path=Path("/data"),
        )
        run2 = Run(
            id="timestamp456",
            network="NSFNet",
            date="0606",
            algorithm="dqn_obs_7",
            path=Path("/data"),
        )

        run_set = {run1, run2}
        assert len(run_set) == 2


class TestMetricDefinition:
    """Tests for MetricDefinition entity."""

    def test_create_metric_definition(self) -> None:
        """Should create MetricDefinition."""
        metric = MetricDefinition(
            name="blocking_probability",
            data_type=DataType.FLOAT,
            source_path="$.blocking_mean",
            unit="probability",
            description="Network blocking probability",
        )
        assert metric.name == "blocking_probability"
        assert metric.data_type == DataType.FLOAT
        assert metric.unit == "probability"

    def test_empty_name_raises_error(self) -> None:
        """Should raise ValueError for empty name."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            MetricDefinition(
                name="",
                data_type=DataType.FLOAT,
                source_path="$.value",
            )

    def test_invalid_jsonpath_raises_error(self) -> None:
        """Should raise InvalidMetricPathError for invalid JSONPath."""
        with pytest.raises(InvalidMetricPathError):
            MetricDefinition(
                name="test",
                data_type=DataType.FLOAT,
                source_path="[[[invalid",
            )

    def test_extract_from_simple_path(self) -> None:
        """Should extract value from simple JSONPath."""
        metric = MetricDefinition(
            name="blocking",
            data_type=DataType.FLOAT,
            source_path="$.blocking_mean",
        )

        data = {"blocking_mean": 0.045}
        value = metric.extract_from(data)

        assert value.value == 0.045
        assert value.data_type == DataType.FLOAT

    def test_extract_from_nested_path(self) -> None:
        """Should extract value from nested JSONPath."""
        metric = MetricDefinition(
            name="hops",
            data_type=DataType.FLOAT,
            source_path="$.iter_stats.0.hops_mean",
        )

        data = {"iter_stats": {"0": {"hops_mean": 3.2}}}
        value = metric.extract_from(data)
        assert value.value == 3.2

    def test_extract_from_missing_path_raises_error(self) -> None:
        """Should raise MetricExtractionError when path not found."""
        metric = MetricDefinition(
            name="test",
            data_type=DataType.FLOAT,
            source_path="$.nonexistent",
        )

        data = {"other_field": 123}
        with pytest.raises(MetricExtractionError):
            metric.extract_from(data)


class TestPlot:
    """Tests for Plot entity."""

    def test_create_plot(self, sample_plot_configuration: Any) -> None:
        """Should create Plot entity."""
        plot = Plot(
            id=PlotId.generate(),
            title="Test Plot",
            configuration=sample_plot_configuration,
        )
        assert plot.title == "Test Plot"
        assert plot.state == PlotState.PENDING

    def test_validate_valid_plot(self, sample_plot: Any) -> None:
        """Should validate valid plot."""
        sample_plot.validate()  # Should not raise

    def test_validate_empty_title_raises_error(
        self, sample_plot_configuration: Any
    ) -> None:
        """Should raise ValidationError for empty title."""
        plot = Plot(
            id=PlotId.generate(),
            title="",
            configuration=sample_plot_configuration,
        )
        with pytest.raises(ValidationError, match="title cannot be empty"):
            plot.validate()

    def test_validate_no_algorithms_raises_error(self) -> None:
        """Should raise ValidationError when no algorithms specified."""
        config = PlotConfiguration(
            plot_type=PlotType.LINE,
            metrics=["blocking"],
            algorithms=[],  # Empty
            traffic_volumes=[600, 700],
        )
        plot = Plot(
            id=PlotId.generate(),
            title="Test",
            configuration=config,
        )
        with pytest.raises(ValidationError, match="at least one algorithm"):
            plot.validate()

    def test_state_transitions(self, sample_plot: Any) -> None:
        """Should transition through states correctly."""
        assert sample_plot.state == PlotState.PENDING

        sample_plot.start_loading()
        assert sample_plot.state == PlotState.LOADING

        sample_plot.mark_loaded()
        assert sample_plot.state == PlotState.LOADED

        sample_plot.start_processing()
        assert sample_plot.state == PlotState.PROCESSING

        spec = PlotSpecification(
            title="Test",
            x_data=None,  # type: ignore[arg-type]
            y_data={},
            x_label="X",
            y_label="Y",
        )
        sample_plot.mark_processed(spec)
        assert sample_plot.state == PlotState.PROCESSED

        sample_plot.start_rendering()
        assert sample_plot.state == PlotState.RENDERING

        sample_plot.mark_completed()
        assert sample_plot.state == PlotState.COMPLETED
        assert sample_plot.is_completed()

    def test_invalid_state_transition_raises_error(self, sample_plot: Any) -> None:
        """Should raise InvalidStateError for invalid transitions."""
        # Try to process before loading
        with pytest.raises(InvalidStateError):
            sample_plot.start_processing()

    def test_mark_failed(self, sample_plot: Any) -> None:
        """Should mark plot as failed with error message."""
        sample_plot.mark_failed("Test error")
        assert sample_plot.is_failed()
        assert sample_plot.error == "Test error"
        assert sample_plot.state == PlotState.FAILED


class TestDataSource:
    """Tests for DataSource entity."""

    def test_create_data_source(self, tmp_path: Any) -> None:
        """Should create DataSource entity."""
        data_file = tmp_path / "data.json"
        data_file.touch()

        source = DataSource(
            source_type=SourceType.FILE,
            location=data_file,
            format=DataFormat.JSON,
            version=DataVersion(major=1),
        )

        assert source.source_type == SourceType.FILE
        assert source.format == DataFormat.JSON
        assert source.version.major == 1

    def test_exists_for_file_source(self, tmp_path: Any) -> None:
        """Should check if file exists."""
        data_file = tmp_path / "data.json"

        source = DataSource(
            source_type=SourceType.FILE,
            location=data_file,
            format=DataFormat.JSON,
            version=DataVersion(major=1),
        )

        # File doesn't exist yet
        assert not source.exists()

        # Create file
        data_file.touch()
        assert source.exists()

    def test_path_converted_to_pathlib(self) -> None:
        """Should convert string path to Path."""
        source = DataSource(
            source_type=SourceType.FILE,
            location="/data/file.json",  # type: ignore[arg-type]
            format=DataFormat.JSON,
            version=DataVersion(major=1),
        )

        assert isinstance(source.location, Path)
