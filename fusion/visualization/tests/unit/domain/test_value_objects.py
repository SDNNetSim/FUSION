"""Unit tests for domain value objects."""

from uuid import UUID

import numpy as np
import pytest

from fusion.visualization.domain.value_objects import (
    Annotation,
    DataType,
    DataVersion,
    MetricValue,
    PlotId,
    PlotSpecification,
)


class TestPlotId:
    """Tests for PlotId value object."""

    def test_generate_creates_unique_ids(self) -> None:
        """Should generate unique IDs."""
        id1 = PlotId.generate()
        id2 = PlotId.generate()
        assert id1 != id2
        assert isinstance(id1.value, UUID)

    def test_from_string_parses_uuid(self) -> None:
        """Should parse UUID from string."""
        uuid_str = "12345678-1234-5678-1234-567812345678"
        plot_id = PlotId.from_string(uuid_str)
        assert str(plot_id.value) == uuid_str

    def test_from_string_invalid_raises_error(self) -> None:
        """Should raise ValueError for invalid UUID string."""
        with pytest.raises(ValueError):
            PlotId.from_string("not-a-uuid")

    def test_str_representation(self) -> None:
        """Should return string representation."""
        plot_id = PlotId.generate()
        str_repr = str(plot_id)
        assert len(str_repr) == 36  # UUID string length

    def test_immutability(self) -> None:
        """PlotId should be immutable (frozen dataclass)."""
        plot_id = PlotId.generate()
        from dataclasses import FrozenInstanceError

        with pytest.raises(FrozenInstanceError):
            plot_id.value = UUID("12345678-1234-5678-1234-567812345678")  # type: ignore[misc]


class TestDataVersion:
    """Tests for DataVersion value object."""

    def test_create_with_major_only(self) -> None:
        """Should create version with major number only."""
        version = DataVersion(major=1)
        assert version.major == 1
        assert version.minor == 0

    def test_create_with_major_and_minor(self) -> None:
        """Should create version with major and minor."""
        version = DataVersion(major=2, minor=3)
        assert version.major == 2
        assert version.minor == 3

    def test_negative_version_raises_error(self) -> None:
        """Should raise ValueError for negative version numbers."""
        with pytest.raises(ValueError):
            DataVersion(major=-1)
        with pytest.raises(ValueError):
            DataVersion(major=1, minor=-1)

    def test_from_string_major_only(self) -> None:
        """Should parse version string with major only."""
        version = DataVersion.from_string("v1")
        assert version.major == 1
        assert version.minor == 0

    def test_from_string_major_and_minor(self) -> None:
        """Should parse version string with major and minor."""
        version = DataVersion.from_string("v2.3")
        assert version.major == 2
        assert version.minor == 3

    def test_from_string_without_v_prefix(self) -> None:
        """Should parse version string without 'v' prefix."""
        version = DataVersion.from_string("2.3")
        assert version.major == 2
        assert version.minor == 3

    def test_str_representation_major_only(self) -> None:
        """Should format as 'vX' when minor is 0."""
        version = DataVersion(major=1)
        assert str(version) == "v1"

    def test_str_representation_with_minor(self) -> None:
        """Should format as 'vX.Y' when minor > 0."""
        version = DataVersion(major=2, minor=3)
        assert str(version) == "v2.3"

    def test_comparison_operators(self) -> None:
        """Should support version comparison."""
        v1 = DataVersion(major=1)
        v1_1 = DataVersion(major=1, minor=1)
        v2 = DataVersion(major=2)

        assert v1 < v1_1 < v2
        assert v2 > v1_1 > v1
        assert v1 <= v1
        assert v2 >= v2


class TestMetricValue:
    """Tests for MetricValue value object."""

    def test_create_float_value(self) -> None:
        """Should create float metric value."""
        value = MetricValue(value=0.045, data_type=DataType.FLOAT, unit="probability")
        assert value.value == 0.045
        assert value.data_type == DataType.FLOAT
        assert value.unit == "probability"

    def test_create_int_value(self) -> None:
        """Should create int metric value."""
        value = MetricValue(value=42, data_type=DataType.INT)
        assert value.value == 42
        assert value.data_type == DataType.INT

    def test_create_array_value(self) -> None:
        """Should create array metric value."""
        arr = np.array([1, 2, 3, 4, 5])
        value = MetricValue(value=arr, data_type=DataType.ARRAY)
        assert isinstance(value.value, np.ndarray)
        assert len(value.value) == 5

    def test_as_float_from_float(self) -> None:
        """Should return float from FLOAT type."""
        value = MetricValue(value=0.045, data_type=DataType.FLOAT)
        assert value.as_float == 0.045

    def test_as_float_from_int(self) -> None:
        """Should convert int to float."""
        value = MetricValue(value=42, data_type=DataType.INT)
        assert value.as_float == 42.0

    def test_as_float_from_array_raises_error(self) -> None:
        """Should raise TypeError when converting array to float."""
        value = MetricValue(value=[1, 2, 3], data_type=DataType.ARRAY)
        with pytest.raises(TypeError):
            _ = value.as_float

    def test_as_array_returns_numpy_array(self) -> None:
        """Should return numpy array."""
        value = MetricValue(value=[1, 2, 3], data_type=DataType.ARRAY)
        arr = value.as_array
        assert isinstance(arr, np.ndarray)
        assert len(arr) == 3

    def test_metadata_defaults_to_empty_dict(self) -> None:
        """Should default metadata to empty dict."""
        value = MetricValue(value=0.5, data_type=DataType.FLOAT)
        assert value.metadata == {}

    def test_immutability_raises_error(self) -> None:
        """MetricValue should be immutable."""
        value = MetricValue(value=0.5, data_type=DataType.FLOAT)
        with pytest.raises((AttributeError, TypeError)):
            value.value = 1.0  # type: ignore[misc]


class TestPlotSpecification:
    """Tests for PlotSpecification value object."""

    def test_create_valid_specification(self) -> None:
        """Should create valid plot specification."""
        x_data = np.array([1, 2, 3, 4, 5])
        y_data = {
            "algo1": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            "algo2": np.array([0.15, 0.25, 0.35, 0.45, 0.55]),
        }

        spec = PlotSpecification(
            title="Test Plot",
            x_data=x_data,
            y_data=y_data,
            x_label="X Axis",
            y_label="Y Axis",
        )

        assert spec.title == "Test Plot"
        assert len(spec.x_data) == 5
        assert len(spec.y_data) == 2
        assert "algo1" in spec.algorithms
        assert "algo2" in spec.algorithms

    def test_length_mismatch_raises_error(self) -> None:
        """Should raise ValueError when x_data and y_data lengths don't match."""
        x_data = np.array([1, 2, 3])
        y_data = {"algo1": np.array([0.1, 0.2])}  # Wrong length

        with pytest.raises(ValueError, match="Length mismatch"):
            PlotSpecification(
                title="Test",
                x_data=x_data,
                y_data=y_data,
                x_label="X",
                y_label="Y",
            )

    def test_error_bars_validation(self) -> None:
        """Should validate error bars match data."""
        x_data = np.array([1, 2, 3])
        y_data = {"algo1": np.array([0.1, 0.2, 0.3])}
        error_bars = {"algo1": np.array([0.01, 0.02])}  # Wrong length

        with pytest.raises(ValueError, match="Error bar length mismatch"):
            PlotSpecification(
                title="Test",
                x_data=x_data,
                y_data=y_data,
                x_label="X",
                y_label="Y",
                error_bars=error_bars,
            )

    def test_num_series_property(self) -> None:
        """Should return correct number of series."""
        x_data = np.array([1, 2, 3])
        y_data = {
            "algo1": np.array([0.1, 0.2, 0.3]),
            "algo2": np.array([0.2, 0.3, 0.4]),
            "algo3": np.array([0.3, 0.4, 0.5]),
        }

        spec = PlotSpecification(
            title="Test",
            x_data=x_data,
            y_data=y_data,
            x_label="X",
            y_label="Y",
        )

        assert spec.num_series == 3

    def test_has_error_bars(self) -> None:
        """Should detect presence of error bars."""
        x_data = np.array([1, 2, 3])
        y_data = {"algo1": np.array([0.1, 0.2, 0.3])}

        spec_without_errors = PlotSpecification(
            title="Test",
            x_data=x_data,
            y_data=y_data,
            x_label="X",
            y_label="Y",
        )
        assert not spec_without_errors.has_error_bars()

        error_bars = {"algo1": np.array([0.01, 0.02, 0.03])}
        spec_with_errors = PlotSpecification(
            title="Test",
            x_data=x_data,
            y_data=y_data,
            x_label="X",
            y_label="Y",
            error_bars=error_bars,
        )
        assert spec_with_errors.has_error_bars()

    def test_has_annotations(self) -> None:
        """Should detect presence of annotations."""
        x_data = np.array([1, 2, 3])
        y_data = {"algo1": np.array([0.1, 0.2, 0.3])}

        spec_without_annotations = PlotSpecification(
            title="Test",
            x_data=x_data,
            y_data=y_data,
            x_label="X",
            y_label="Y",
        )
        assert not spec_without_annotations.has_annotations()

        annotations = [Annotation(text="Point", x=1.5, y=0.15)]
        spec_with_annotations = PlotSpecification(
            title="Test",
            x_data=x_data,
            y_data=y_data,
            x_label="X",
            y_label="Y",
            annotations=annotations,
        )
        assert spec_with_annotations.has_annotations()
