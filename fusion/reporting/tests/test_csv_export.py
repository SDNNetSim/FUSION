"""
Tests for CSV export functionality.
"""

import csv

from fusion.reporting.csv_export import (
    append_result_to_csv,
    export_aggregated_results,
    export_comparison_table,
    export_results_to_csv,
)


class TestExportResultsToCSV:
    """Test export_results_to_csv function."""

    def test_export_creates_file(self, tmp_path):
        """Test that CSV file is created at specified path."""
        output_path = tmp_path / "results.csv"
        results = [
            {"bp_overall": 0.05, "seed": 42, "topology": "NSFNET"},
            {"bp_overall": 0.06, "seed": 43, "topology": "NSFNET"},
        ]

        export_results_to_csv(results, str(output_path))

        assert output_path.exists()

    def test_export_includes_all_columns(self, tmp_path):
        """Test that all columns from results are included."""
        output_path = tmp_path / "results.csv"
        results = [
            {"bp_overall": 0.05, "seed": 42, "recovery_time_mean_ms": 52.3},
            {"bp_overall": 0.06, "seed": 43, "recovery_time_mean_ms": 51.8},
        ]

        export_results_to_csv(results, str(output_path))

        # Read back and verify columns
        with open(output_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames

            assert "bp_overall" in fieldnames
            assert "seed" in fieldnames
            assert "recovery_time_mean_ms" in fieldnames

    def test_export_correct_row_count(self, tmp_path):
        """Test that correct number of rows are written."""
        output_path = tmp_path / "results.csv"
        results = [
            {"bp_overall": 0.05, "seed": 42},
            {"bp_overall": 0.06, "seed": 43},
            {"bp_overall": 0.055, "seed": 44},
        ]

        export_results_to_csv(results, str(output_path))

        with open(output_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            assert len(rows) == 3

    def test_export_priority_columns_first(self, tmp_path):
        """Test that priority columns appear first."""
        output_path = tmp_path / "results.csv"
        results = [
            {
                "recovery_time_mean_ms": 52.3,
                "seed": 42,
                "topology": "NSFNET",
                "bp_overall": 0.05,
            }
        ]

        export_results_to_csv(results, str(output_path))

        with open(output_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames

            # Check that priority columns are early in the list
            assert fieldnames.index("seed") < fieldnames.index("recovery_time_mean_ms")
            assert fieldnames.index("bp_overall") < fieldnames.index(
                "recovery_time_mean_ms"
            )

    def test_export_empty_results(self, tmp_path):
        """Test export with empty results list."""
        output_path = tmp_path / "results.csv"
        results = []

        export_results_to_csv(results, str(output_path))

        # File should not be created for empty results
        assert not output_path.exists()


class TestExportAggregatedResults:
    """Test export_aggregated_results function."""

    def test_export_aggregated_creates_file(self, tmp_path):
        """Test that aggregated results file is created."""
        output_path = tmp_path / "aggregated.csv"
        aggregated = {
            "bp_overall": {
                "mean": 0.055,
                "std": 0.005,
                "ci95_lower": 0.045,
                "ci95_upper": 0.065,
                "n": 3,
            }
        }

        export_aggregated_results(aggregated, str(output_path))

        assert output_path.exists()

    def test_export_aggregated_includes_metadata(self, tmp_path):
        """Test that metadata is included in aggregated export."""
        output_path = tmp_path / "aggregated.csv"
        aggregated = {
            "bp_overall": {
                "mean": 0.055,
                "std": 0.005,
                "ci95_lower": 0.045,
                "ci95_upper": 0.065,
                "n": 3,
            }
        }
        metadata = {"policy": "bc", "topology": "NSFNET"}

        export_aggregated_results(aggregated, str(output_path), metadata)

        with open(output_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            assert len(rows) == 1
            assert rows[0]["policy"] == "bc"
            assert rows[0]["topology"] == "NSFNET"
            assert rows[0]["metric"] == "bp_overall"


class TestExportComparisonTable:
    """Test export_comparison_table function."""

    def test_export_comparison_creates_file(self, tmp_path):
        """Test that comparison table file is created."""
        output_path = tmp_path / "comparison.csv"
        comparison = {
            "bp_overall": {
                "baseline_mean": 0.105,
                "baseline_std": 0.007,
                "baseline_ci95": (0.098, 0.112),
                "rl_mean": 0.085,
                "rl_std": 0.007,
                "rl_ci95": (0.078, 0.092),
                "improvement_pct": 19.05,
                "n_baseline": 2,
                "n_rl": 2,
            }
        }

        export_comparison_table(comparison, str(output_path))

        assert output_path.exists()

    def test_export_comparison_includes_all_fields(self, tmp_path):
        """Test that all comparison fields are exported."""
        output_path = tmp_path / "comparison.csv"
        comparison = {
            "bp_overall": {
                "baseline_mean": 0.105,
                "baseline_std": 0.007,
                "baseline_ci95": (0.098, 0.112),
                "rl_mean": 0.085,
                "rl_std": 0.007,
                "rl_ci95": (0.078, 0.092),
                "improvement_pct": 19.05,
                "n_baseline": 2,
                "n_rl": 2,
            }
        }

        export_comparison_table(comparison, str(output_path))

        with open(output_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            assert len(rows) == 1
            row = rows[0]
            assert "metric" in row
            assert "baseline_mean" in row
            assert "rl_mean" in row
            assert "improvement_pct" in row
            assert float(row["improvement_pct"]) == 19.05


class TestAppendResultToCSV:
    """Test append_result_to_csv function."""

    def test_append_creates_file_with_header(self, tmp_path):
        """Test that append creates file with header on first call."""
        output_path = tmp_path / "results.csv"
        result = {"bp_overall": 0.05, "seed": 42}

        append_result_to_csv(result, str(output_path))

        assert output_path.exists()

        # Verify header was written
        with open(output_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)

            assert "bp_overall" in fieldnames
            assert "seed" in fieldnames
            assert len(rows) == 1

    def test_append_adds_rows_without_duplicate_header(self, tmp_path):
        """Test that subsequent appends don't duplicate header."""
        output_path = tmp_path / "results.csv"

        # First append
        append_result_to_csv({"bp_overall": 0.05, "seed": 42}, str(output_path))

        # Second append
        append_result_to_csv({"bp_overall": 0.06, "seed": 43}, str(output_path))

        # Read and verify
        with open(output_path, encoding="utf-8") as f:
            lines = f.readlines()

            # Should have 1 header + 2 data rows
            assert len(lines) == 3

            # First line should be header
            assert "bp_overall" in lines[0]
