"""Unit tests for ValidationService."""

import pytest

from fusion.visualization.application.services import ValidationService


class TestValidationService:
    """Tests for ValidationService."""

    @pytest.fixture
    def service(self) -> ValidationService:
        """Create validation service."""
        return ValidationService()

    def test_valid_plot_config(self, service: ValidationService) -> None:
        """Should validate valid plot config."""
        config = {
            "network": "NSFNet",
            "dates": ["0606", "0611"],
            "plot_type": "blocking",
            "algorithms": ["ppo_obs_7", "dqn_obs_7"],
            "traffic_volumes": [600, 700, 800],
        }

        result = service.validate_plot_config(config)

        assert result.is_valid
        assert not result.has_errors
        # May have warnings, that's ok

    def test_missing_network(self, service: ValidationService) -> None:
        """Should detect missing network."""
        config = {
            "dates": ["0606"],
            "plot_type": "blocking",
        }

        result = service.validate_plot_config(config)

        assert not result.is_valid
        assert result.has_errors
        assert any("network" in err.lower() for err in result.errors)

    def test_invalid_network(self, service: ValidationService) -> None:
        """Should warn about unknown network."""
        config = {
            "network": "UnknownNet",
            "dates": ["0606"],
            "plot_type": "blocking",
        }

        result = service.validate_plot_config(config)

        # Should be valid but with warning
        assert result.is_valid  # Not an error, just warning
        assert result.has_warnings
        assert any("network" in warn.lower() for warn in result.warnings)

    def test_missing_dates(self, service: ValidationService) -> None:
        """Should detect missing dates."""
        config = {
            "network": "NSFNet",
            "plot_type": "blocking",
        }

        result = service.validate_plot_config(config)

        assert not result.is_valid
        assert any("dates" in err.lower() for err in result.errors)

    def test_empty_dates(self, service: ValidationService) -> None:
        """Should detect empty dates list."""
        config = {
            "network": "NSFNet",
            "dates": [],
            "plot_type": "blocking",
        }

        result = service.validate_plot_config(config)

        assert not result.is_valid
        assert any("dates" in err.lower() for err in result.errors)

    def test_invalid_date_format(self, service: ValidationService) -> None:
        """Should detect invalid date format."""
        config = {
            "network": "NSFNet",
            "dates": ["2024-06-06"],  # Wrong format, should be MMDD
            "plot_type": "blocking",
        }

        result = service.validate_plot_config(config)

        assert not result.is_valid
        assert any("date" in err.lower() for err in result.errors)

    def test_missing_plot_type(self, service: ValidationService) -> None:
        """Should detect missing plot_type."""
        config = {
            "network": "NSFNet",
            "dates": ["0606"],
        }

        result = service.validate_plot_config(config)

        assert not result.is_valid
        assert any("plot_type" in err.lower() for err in result.errors)

    def test_invalid_plot_type(self, service: ValidationService) -> None:
        """Should detect invalid plot_type."""
        config = {
            "network": "NSFNet",
            "dates": ["0606"],
            "plot_type": "invalid_type",
        }

        result = service.validate_plot_config(config)

        assert not result.is_valid
        assert any("plot_type" in err.lower() for err in result.errors)

    def test_invalid_traffic_volumes(self, service: ValidationService) -> None:
        """Should detect invalid traffic volumes."""
        config = {
            "network": "NSFNet",
            "dates": ["0606"],
            "plot_type": "blocking",
            "traffic_volumes": [-100, 0, 600],
        }

        result = service.validate_plot_config(config)

        assert not result.is_valid
        assert any("traffic" in err.lower() for err in result.errors)

    def test_invalid_format(self, service: ValidationService) -> None:
        """Should detect invalid format."""
        config = {
            "network": "NSFNet",
            "dates": ["0606"],
            "plot_type": "blocking",
            "format": "invalid",
        }

        result = service.validate_plot_config(config)

        assert not result.is_valid
        assert any("format" in err.lower() for err in result.errors)

    def test_low_dpi_warning(self, service: ValidationService) -> None:
        """Should warn about low DPI."""
        config = {
            "network": "NSFNet",
            "dates": ["0606"],
            "plot_type": "blocking",
            "dpi": 50,
        }

        result = service.validate_plot_config(config)

        assert result.is_valid  # Not an error
        assert result.has_warnings
        assert any("dpi" in warn.lower() for warn in result.warnings)

    def test_invalid_figsize(self, service: ValidationService) -> None:
        """Should detect invalid figsize."""
        config = {
            "network": "NSFNet",
            "dates": ["0606"],
            "plot_type": "blocking",
            "figsize": (10,),  # Should be (width, height)
        }

        result = service.validate_plot_config(config)

        assert not result.is_valid
        assert any("figsize" in err.lower() for err in result.errors)

    def test_valid_batch_config(self, service: ValidationService) -> None:
        """Should validate valid batch config."""
        config = {
            "network": "NSFNet",
            "dates": ["0606"],
            "plots": [
                {"plot_type": "blocking"},
                {"plot_type": "rewards", "algorithms": ["ppo_obs_7"]},
            ],
        }

        result = service.validate_batch_config(config)

        assert result.is_valid

    def test_empty_plots_in_batch(self, service: ValidationService) -> None:
        """Should detect empty plots list."""
        config = {
            "network": "NSFNet",
            "dates": ["0606"],
            "plots": [],
        }

        result = service.validate_batch_config(config)

        assert not result.is_valid
        assert any("plots" in err.lower() for err in result.errors)

    def test_invalid_plot_in_batch(self, service: ValidationService) -> None:
        """Should detect invalid plot in batch."""
        config = {
            "network": "NSFNet",
            "dates": ["0606"],
            "plots": [
                {"plot_type": "invalid_type"},
            ],
        }

        result = service.validate_batch_config(config)

        assert not result.is_valid
        assert any("plot_type" in err.lower() for err in result.errors)

    def test_valid_comparison_config(self, service: ValidationService) -> None:
        """Should validate valid comparison config."""
        config = {
            "network": "NSFNet",
            "dates": ["0606"],
            "algorithms": ["ppo_obs_7", "dqn_obs_7"],
            "metric": "blocking_probability",
        }

        result = service.validate_comparison_config(config)

        assert result.is_valid

    def test_insufficient_algorithms_for_comparison(self, service: ValidationService) -> None:
        """Should require at least 2 algorithms for comparison."""
        config = {
            "network": "NSFNet",
            "dates": ["0606"],
            "algorithms": ["ppo_obs_7"],
            "metric": "blocking",
        }

        result = service.validate_comparison_config(config)

        assert not result.is_valid
        assert any("algorithms" in err.lower() for err in result.errors)

    def test_suggest_fixes(self, service: ValidationService) -> None:
        """Should suggest fixes for common errors."""
        errors = [
            "Invalid network",
            "Invalid plot_type",
            "Invalid dates format",
        ]

        suggestions = service.suggest_fixes(errors)

        assert "network" in suggestions
        assert "plot_type" in suggestions
        assert "dates" in suggestions
