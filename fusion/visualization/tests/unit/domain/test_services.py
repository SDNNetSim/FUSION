"""Unit tests for domain services."""

import pytest
import numpy as np
from typing import List, Any

from fusion.visualization.domain.services import (
    MetricAggregationService,
    DataValidationService,
    ValidationResult,
)
from fusion.visualization.domain.entities import AggregationStrategy
from fusion.visualization.domain.value_objects import MetricValue, DataType
from fusion.visualization.domain.exceptions import InsufficientDataError


class TestMetricAggregationService:
    """Tests for MetricAggregationService."""

    @pytest.fixture
    def service(self) -> MetricAggregationService:
        """Create service instance."""
        return MetricAggregationService()

    def test_aggregate_mean(self, service: MetricAggregationService, sample_metric_values: List[MetricValue]) -> None:
        """Should aggregate using mean."""
        result = service.aggregate(
            sample_metric_values,
            AggregationStrategy.MEAN
        )

        assert result.data_type == DataType.FLOAT
        expected_mean = np.mean([0.045, 0.042, 0.048, 0.044, 0.046])
        assert result.value == pytest.approx(expected_mean)
        assert result.metadata is not None
        assert "std" in result.metadata
        assert result.metadata["n_samples"] == 5

    def test_aggregate_median(self, service: MetricAggregationService, sample_metric_values: List[MetricValue]) -> None:
        """Should aggregate using median."""
        result = service.aggregate(
            sample_metric_values,
            AggregationStrategy.MEDIAN
        )

        assert result.value == 0.045  # Middle value
        assert result.metadata is not None
        assert result.metadata["n_samples"] == 5

    def test_aggregate_last(self, service: MetricAggregationService, sample_metric_values: List[MetricValue]) -> None:
        """Should return last value."""
        result = service.aggregate(
            sample_metric_values,
            AggregationStrategy.LAST
        )

        assert result.value == 0.046  # Last in list

    def test_aggregate_last_k(self, service: MetricAggregationService, sample_metric_values: List[MetricValue]) -> None:
        """Should aggregate last K values."""
        result = service.aggregate(
            sample_metric_values,
            AggregationStrategy.LAST_K,
            k=3
        )

        # Should average last 3: [0.048, 0.044, 0.046]
        expected = np.mean([0.048, 0.044, 0.046])
        assert result.value == pytest.approx(expected)

    def test_aggregate_last_k_insufficient_data(self, service: MetricAggregationService) -> None:
        """Should raise error when not enough data for LAST_K."""
        values = [
            MetricValue(value=0.1, data_type=DataType.FLOAT),
            MetricValue(value=0.2, data_type=DataType.FLOAT),
        ]

        with pytest.raises(InsufficientDataError):
            service.aggregate(values, AggregationStrategy.LAST_K, k=5)

    def test_aggregate_max(self, service: MetricAggregationService, sample_metric_values: List[MetricValue]) -> None:
        """Should return maximum value."""
        result = service.aggregate(
            sample_metric_values,
            AggregationStrategy.MAX
        )

        assert result.value == 0.048

    def test_aggregate_min(self, service: MetricAggregationService, sample_metric_values: List[MetricValue]) -> None:
        """Should return minimum value."""
        result = service.aggregate(
            sample_metric_values,
            AggregationStrategy.MIN
        )

        assert result.value == 0.042

    def test_aggregate_sum(self, service: MetricAggregationService, sample_metric_values: List[MetricValue]) -> None:
        """Should sum all values."""
        result = service.aggregate(
            sample_metric_values,
            AggregationStrategy.SUM
        )

        expected = sum([0.045, 0.042, 0.048, 0.044, 0.046])
        assert result.value == pytest.approx(expected)

    def test_aggregate_with_confidence_interval(self, service: MetricAggregationService, sample_metric_values: List[MetricValue]) -> None:
        """Should compute confidence interval."""
        result = service.aggregate(
            sample_metric_values,
            AggregationStrategy.CONFIDENCE_INTERVAL,
            confidence_level=0.95
        )

        assert result.metadata is not None
        assert "mean" in result.metadata
        assert "std" in result.metadata
        assert "ci_lower" in result.metadata
        assert "ci_upper" in result.metadata
        assert "margin_of_error" in result.metadata
        assert result.metadata["confidence_level"] == 0.95

        # CI bounds should be around the mean
        mean = result.metadata["mean"]
        ci_lower = result.metadata["ci_lower"]
        ci_upper = result.metadata["ci_upper"]
        assert ci_lower < mean < ci_upper

    def test_aggregate_confidence_interval_insufficient_data(self, service: MetricAggregationService) -> None:
        """Should raise error for CI with < 2 values."""
        values = [MetricValue(value=0.1, data_type=DataType.FLOAT)]

        with pytest.raises(InsufficientDataError):
            service.aggregate(
                values,
                AggregationStrategy.CONFIDENCE_INTERVAL
            )

    def test_aggregate_empty_list_raises_error(self, service: MetricAggregationService) -> None:
        """Should raise error for empty value list."""
        with pytest.raises(InsufficientDataError):
            service.aggregate([], AggregationStrategy.MEAN)

    def test_compute_statistics(self, service: MetricAggregationService, sample_metric_values: list) -> None:
        """Should compute comprehensive statistics."""
        stats = service.compute_statistics(sample_metric_values)

        assert "mean" in stats
        assert "median" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "n_samples" in stats
        assert stats["n_samples"] == 5

        # Should include quartiles
        assert "q25" in stats
        assert "q75" in stats
        assert "iqr" in stats

    def test_compute_statistics_empty_list(self, service: MetricAggregationService) -> None:
        """Should return empty dict for empty list."""
        stats = service.compute_statistics([])
        assert stats == {}


class TestDataValidationService:
    """Tests for DataValidationService."""

    @pytest.fixture
    def service(self) -> DataValidationService:
        """Create service instance."""
        return DataValidationService()

    def test_validate_valid_data(self, service: DataValidationService) -> None:
        """Should validate valid data."""
        data = {"field1": "value1", "field2": 123}
        result = service.validate(data)

        assert result.is_valid
        assert len(result.errors) == 0

    def test_validate_none_data(self, service: DataValidationService) -> None:
        """Should reject None data."""
        result = service.validate(None)  # type: ignore[arg-type]

        assert not result.is_valid
        assert len(result.errors) > 0
        assert "Data is None" in result.errors[0]

    def test_validate_non_dict_data(self, service: DataValidationService) -> None:
        """Should reject non-dict data."""
        result = service.validate("not a dict")  # type: ignore[arg-type]

        assert not result.is_valid
        assert "must be a dict" in result.errors[0]

    def test_validate_required_fields(self, service: DataValidationService) -> None:
        """Should check for required fields."""
        data = {"field1": "value1"}
        result = service.validate(
            data,
            required_fields=["field1", "field2", "field3"]
        )

        assert not result.is_valid
        assert len(result.errors) == 2
        assert any("field2" in e for e in result.errors)
        assert any("field3" in e for e in result.errors)

    def test_validate_simulation_data_valid(self, service: DataValidationService, sample_v1_data: dict) -> None:
        """Should validate valid simulation data."""
        result = service.validate_simulation_data(sample_v1_data)
        assert result.is_valid

    def test_validate_simulation_data_missing_common_fields(self, service: DataValidationService) -> None:
        """Should warn about missing common fields."""
        data = {"custom_field": 123}
        result = service.validate_simulation_data(data)

        # Should have warnings but still be valid
        assert len(result.warnings) > 0
        assert "common simulation fields" in result.warnings[0]

    def test_validate_simulation_data_deprecated_fields(self, service: DataValidationService) -> None:
        """Should warn about deprecated fields."""
        data = {
            "blocking_mean": 0.045,
            "observation_spaces": ["obs_7"],
        }
        result = service.validate_simulation_data(data)

        assert len(result.warnings) >= 1
        # Should have warnings about deprecated fields
        assert any("deprecated" in w.lower() for w in result.warnings)

    def test_validate_simulation_data_invalid_iter_stats(self, service: DataValidationService) -> None:
        """Should error on invalid iter_stats structure."""
        data = {
            "blocking_mean": 0.045,
            "iter_stats": "not a dict or list"
        }
        result = service.validate_simulation_data(data)

        assert not result.is_valid
        assert any("iter_stats" in e for e in result.errors)

    def test_validate_metric_path_valid(self, service: DataValidationService) -> None:
        """Should validate valid metric paths."""
        result = service.validate_metric_path("$.blocking_mean")
        assert result.is_valid

        result = service.validate_metric_path("blocking")
        assert result.is_valid

    def test_validate_metric_path_empty(self, service: DataValidationService) -> None:
        """Should reject empty metric path."""
        result = service.validate_metric_path("")
        assert not result.is_valid
        assert "cannot be empty" in result.errors[0]

    def test_suggest_missing_fields(self, service: DataValidationService) -> None:
        """Should suggest missing fields."""
        data = {"field1": "value1"}
        expected = ["field1", "field2", "field3"]

        missing = service.suggest_missing_fields(data, expected)

        assert "field2" in missing
        assert "field3" in missing
        assert "field1" not in missing


class TestValidationResult:
    """Tests for ValidationResult."""

    def test_create_valid_result(self) -> None:
        """Should create valid result."""
        result = ValidationResult(is_valid=True)
        assert result.is_valid
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_add_error(self) -> None:
        """Should add error and mark as invalid."""
        result = ValidationResult(is_valid=True)
        result.add_error("Test error")

        assert not result.is_valid
        assert "Test error" in result.errors

    def test_add_warning(self) -> None:
        """Should add warning without affecting validity."""
        result = ValidationResult(is_valid=True)
        result.add_warning("Test warning")

        assert result.is_valid
        assert "Test warning" in result.warnings

    def test_bool_conversion(self) -> None:
        """Should convert to bool based on validity."""
        valid = ValidationResult(is_valid=True)
        invalid = ValidationResult(is_valid=False)

        assert bool(valid) is True
        assert bool(invalid) is False

        # Can use in if statements
        if valid:
            pass  # Should execute
        else:
            pytest.fail("Should not reach here")

    def test_repr(self) -> None:
        """Should have meaningful repr."""
        result = ValidationResult(is_valid=True)
        result.add_error("Error 1")
        result.add_warning("Warning 1")

        repr_str = repr(result)
        assert "invalid" in repr_str
        assert "errors=1" in repr_str
        assert "warnings=1" in repr_str
