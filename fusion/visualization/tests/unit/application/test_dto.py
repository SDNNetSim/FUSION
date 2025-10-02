"""Unit tests for DTOs (Data Transfer Objects)."""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from fusion.visualization.application.dto import (
    BatchPlotResultDTO,
    ComparisonRequestDTO,
    PlotRequestDTO,
    PlotResultDTO,
    StatisticalComparison,
)
from fusion.visualization.domain.value_objects.plot_specification import PlotType


class TestPlotRequestDTO:
    """Tests for PlotRequestDTO."""

    def test_valid_request(self) -> None:
        """Should create valid request."""
        request = PlotRequestDTO(
            network="NSFNet",
            dates=["0606"],
            plot_type=PlotType.LINE,
            algorithms=["ppo_obs_7"],
            traffic_volumes=[600, 700],
        )

        errors = request.validate()
        assert len(errors) == 0

    def test_missing_network(self) -> None:
        """Should detect missing network."""
        request = PlotRequestDTO(
            network="",
            dates=["0606"],
            plot_type=PlotType.LINE,
        )

        errors = request.validate()
        assert any("network" in err for err in errors)

    def test_missing_dates(self) -> None:
        """Should detect missing dates."""
        request = PlotRequestDTO(
            network="NSFNet",
            dates=[],
            plot_type=PlotType.LINE,
        )

        errors = request.validate()
        assert any("date" in err for err in errors)

    def test_invalid_traffic_volumes(self) -> None:
        """Should detect invalid traffic volumes."""
        request = PlotRequestDTO(
            network="NSFNet",
            dates=["0606"],
            plot_type=PlotType.LINE,
            traffic_volumes=[-100, 0, 500],
        )

        errors = request.validate()
        assert any("traffic_volumes" in err for err in errors)

    def test_invalid_format(self) -> None:
        """Should detect invalid format."""
        request = PlotRequestDTO(
            network="NSFNet",
            dates=["0606"],
            plot_type=PlotType.LINE,
            format="invalid",
        )

        errors = request.validate()
        assert any("format" in err for err in errors)


class TestPlotResultDTO:
    """Tests for PlotResultDTO."""

    def test_success_result(self) -> None:
        """Should create success result."""
        started = datetime.now()
        completed = started + timedelta(seconds=5)

        result = PlotResultDTO(
            success=True,
            plot_id="test_123",
            output_path=Path("/tmp/plot.png"),
            plot_type="blocking",
            algorithms=["ppo", "dqn"],
            traffic_volumes=[600, 700],
            num_runs=10,
            started_at=started,
            completed_at=completed,
            duration=completed - started,
        )

        assert result.success
        assert result.duration_seconds == pytest.approx(5.0, rel=0.1)

    def test_failure_result(self) -> None:
        """Should create failure result."""
        result = PlotResultDTO(
            success=False,
            plot_id="test_123",
            error="Test error",
        )

        assert not result.success
        assert result.error == "Test error"

    def test_to_dict(self) -> None:
        """Should convert to dictionary."""
        result = PlotResultDTO(
            success=True,
            plot_id="test_123",
            algorithms=["ppo"],
        )

        result_dict = result.to_dict()

        assert result_dict["success"] is True
        assert result_dict["plot_id"] == "test_123"
        assert result_dict["algorithms"] == ["ppo"]


class TestBatchPlotResultDTO:
    """Tests for BatchPlotResultDTO."""

    def test_success_count(self) -> None:
        """Should count successful results."""
        results = [
            PlotResultDTO(success=True, plot_id="1"),
            PlotResultDTO(success=True, plot_id="2"),
            PlotResultDTO(success=False, plot_id="3", error="fail"),
        ]

        batch_result = BatchPlotResultDTO(results=results)

        assert batch_result.success_count == 2
        assert batch_result.failure_count == 1
        assert batch_result.total_count == 3
        assert batch_result.success_rate == pytest.approx(66.67, rel=0.1)

    def test_all_successful(self) -> None:
        """Should detect all successful."""
        results = [
            PlotResultDTO(success=True, plot_id="1"),
            PlotResultDTO(success=True, plot_id="2"),
        ]

        batch_result = BatchPlotResultDTO(results=results)

        assert batch_result.all_successful

    def test_not_all_successful(self) -> None:
        """Should detect failures."""
        results = [
            PlotResultDTO(success=True, plot_id="1"),
            PlotResultDTO(success=False, plot_id="2", error="fail"),
        ]

        batch_result = BatchPlotResultDTO(results=results)

        assert not batch_result.all_successful


class TestComparisonRequestDTO:
    """Tests for ComparisonRequestDTO."""

    def test_valid_request(self) -> None:
        """Should create valid comparison request."""
        request = ComparisonRequestDTO(
            network="NSFNet",
            dates=["0606"],
            algorithms=["ppo", "dqn"],
            metric="blocking_probability",
        )

        errors = request.validate()
        assert len(errors) == 0

    def test_insufficient_algorithms(self) -> None:
        """Should require at least 2 algorithms."""
        request = ComparisonRequestDTO(
            network="NSFNet",
            dates=["0606"],
            algorithms=["ppo"],
            metric="blocking",
        )

        errors = request.validate()
        assert any("algorithms" in err for err in errors)

    def test_invalid_confidence_level(self) -> None:
        """Should detect invalid confidence level."""
        request = ComparisonRequestDTO(
            network="NSFNet",
            dates=["0606"],
            algorithms=["ppo", "dqn"],
            metric="blocking",
            confidence_level=1.5,
        )

        errors = request.validate()
        assert any("confidence_level" in err for err in errors)


class TestStatisticalComparison:
    """Tests for StatisticalComparison."""

    def test_is_significant(self) -> None:
        """Should detect statistical significance."""
        comparison = StatisticalComparison(
            algorithm_a="ppo",
            algorithm_b="dqn",
            metric="blocking",
            mean_a=0.05,
            mean_b=0.03,
            std_a=0.01,
            std_b=0.01,
            p_value=0.01,
        )

        assert comparison.is_significant

    def test_not_significant(self) -> None:
        """Should detect non-significant difference."""
        comparison = StatisticalComparison(
            algorithm_a="ppo",
            algorithm_b="dqn",
            metric="blocking",
            mean_a=0.05,
            mean_b=0.048,
            std_a=0.01,
            std_b=0.01,
            p_value=0.3,
        )

        assert not comparison.is_significant

    def test_mean_difference(self) -> None:
        """Should calculate mean difference."""
        comparison = StatisticalComparison(
            algorithm_a="ppo",
            algorithm_b="dqn",
            metric="blocking",
            mean_a=0.05,
            mean_b=0.03,
            std_a=0.01,
            std_b=0.01,
        )

        assert comparison.mean_difference == pytest.approx(-0.02)

    def test_percent_improvement(self) -> None:
        """Should calculate percent improvement."""
        comparison = StatisticalComparison(
            algorithm_a="ppo",
            algorithm_b="dqn",
            metric="blocking",
            mean_a=0.05,
            mean_b=0.03,
            std_a=0.01,
            std_b=0.01,
        )

        # (0.03 - 0.05) / 0.05 * 100 = -40%
        assert comparison.percent_improvement == pytest.approx(-40.0)
