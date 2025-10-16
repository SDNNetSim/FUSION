"""
Tests for multi-seed result aggregation.
"""

from fusion.reporting.aggregation import (
    aggregate_seed_results,
    create_comparison_table,
    format_comparison_for_display,
)


class TestAggregateSeedResults:
    """Test aggregate_seed_results function."""

    def test_aggregate_simple_metrics(self):
        """Test aggregation of simple metrics across seeds."""
        results = [
            {"bp_overall": 0.05, "recovery_time_mean_ms": 52.3, "seed": 42},
            {"bp_overall": 0.06, "recovery_time_mean_ms": 51.8, "seed": 43},
            {"bp_overall": 0.055, "recovery_time_mean_ms": 52.0, "seed": 44},
        ]

        agg = aggregate_seed_results(results, ["bp_overall", "recovery_time_mean_ms"])

        # Check bp_overall
        assert "bp_overall" in agg
        assert abs(agg["bp_overall"]["mean"] - 0.055) < 1e-6
        assert agg["bp_overall"]["n"] == 3
        assert "std" in agg["bp_overall"]
        assert "ci95_lower" in agg["bp_overall"]
        assert "ci95_upper" in agg["bp_overall"]

        # Check recovery_time_mean_ms
        assert "recovery_time_mean_ms" in agg
        assert abs(agg["recovery_time_mean_ms"]["mean"] - 52.033) < 0.01
        assert agg["recovery_time_mean_ms"]["n"] == 3

    def test_aggregate_auto_detect_metrics(self):
        """Test auto-detection of numeric metrics."""
        results = [
            {"bp_overall": 0.05, "topology": "NSFNET", "seed": 42},
            {"bp_overall": 0.06, "topology": "NSFNET", "seed": 43},
        ]

        # Don't specify metrics - should auto-detect
        agg = aggregate_seed_results(results, metric_keys=None)

        # Should include bp_overall but not topology or seed
        assert "bp_overall" in agg
        assert "topology" not in agg
        assert "seed" not in agg

    def test_aggregate_single_seed(self):
        """Test aggregation with single seed (std should be 0)."""
        results = [{"bp_overall": 0.05, "seed": 42}]

        agg = aggregate_seed_results(results, ["bp_overall"])

        assert agg["bp_overall"]["mean"] == 0.05
        assert agg["bp_overall"]["std"] == 0.0
        assert agg["bp_overall"]["n"] == 1

    def test_aggregate_empty_results(self):
        """Test aggregation with empty results."""
        results = []

        agg = aggregate_seed_results(results, ["bp_overall"])

        assert len(agg) == 0

    def test_aggregate_missing_metrics(self):
        """Test aggregation when some results missing metrics."""
        results = [
            {"bp_overall": 0.05, "seed": 42},
            {"bp_overall": 0.06, "seed": 43},
            {"seed": 44},  # Missing bp_overall
        ]

        agg = aggregate_seed_results(results, ["bp_overall"])

        # Should only use first two results
        assert agg["bp_overall"]["n"] == 2
        assert abs(agg["bp_overall"]["mean"] - 0.055) < 1e-6


class TestCreateComparisonTable:
    """Test create_comparison_table function."""

    def test_comparison_shows_improvement(self):
        """Test that comparison correctly shows improvement."""
        baseline = [
            {"bp_overall": 0.10, "seed": 42},
            {"bp_overall": 0.11, "seed": 43},
        ]
        rl = [
            {"bp_overall": 0.08, "seed": 42},
            {"bp_overall": 0.09, "seed": 43},
        ]

        comp = create_comparison_table(baseline, rl, ["bp_overall"])

        assert "bp_overall" in comp
        assert abs(comp["bp_overall"]["baseline_mean"] - 0.105) < 1e-6
        assert abs(comp["bp_overall"]["rl_mean"] - 0.085) < 1e-6

        # Improvement: baseline is higher, RL is lower = positive improvement
        assert comp["bp_overall"]["improvement_pct"] > 0
        assert abs(comp["bp_overall"]["improvement_pct"] - 19.048) < 0.01

    def test_comparison_shows_regression(self):
        """Test that comparison correctly shows regression."""
        baseline = [{"recovery_time_mean_ms": 50.0, "seed": 42}]
        rl = [{"recovery_time_mean_ms": 60.0, "seed": 42}]

        comp = create_comparison_table(baseline, rl, ["recovery_time_mean_ms"])

        # For recovery time, higher is worse, so this should show negative improvement
        assert comp["recovery_time_mean_ms"]["improvement_pct"] < 0

    def test_comparison_includes_confidence_intervals(self):
        """Test that comparison includes CI95 for both baseline and RL."""
        baseline = [
            {"bp_overall": 0.10, "seed": 42},
            {"bp_overall": 0.12, "seed": 43},
        ]
        rl = [
            {"bp_overall": 0.08, "seed": 42},
            {"bp_overall": 0.09, "seed": 43},
        ]

        comp = create_comparison_table(baseline, rl, ["bp_overall"])

        assert "baseline_ci95" in comp["bp_overall"]
        assert "rl_ci95" in comp["bp_overall"]
        assert len(comp["bp_overall"]["baseline_ci95"]) == 2
        assert len(comp["bp_overall"]["rl_ci95"]) == 2


class TestFormatComparisonForDisplay:
    """Test format_comparison_for_display function."""

    def test_format_produces_table(self):
        """Test that formatting produces readable table."""
        comparison = {
            "bp_overall": {
                "baseline_mean": 0.105,
                "baseline_std": 0.007,
                "rl_mean": 0.085,
                "rl_std": 0.007,
                "improvement_pct": 19.05,
            }
        }

        formatted = format_comparison_for_display(comparison)

        assert "bp_overall" in formatted
        assert "0.1050" in formatted
        assert "0.0850" in formatted
        assert "+19.05%" in formatted
        assert "Baseline" in formatted
        assert "Improvement" in formatted

    def test_format_empty_comparison(self):
        """Test formatting of empty comparison."""
        formatted = format_comparison_for_display({})

        assert "No comparison data" in formatted
