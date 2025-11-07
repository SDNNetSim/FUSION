"""
Tests for survivability metrics and reporting functionality.
"""

import numpy as np
import pytest

from fusion.core.metrics import SimStats


class TestFragmentationMetrics:
    """Test fragmentation computation and tracking."""

    @pytest.fixture
    def sim_stats(self) -> SimStats:
        """Create SimStats instance for testing."""
        engine_props = {
            "num_requests": 1000,
            "erlang": 100,
            "network": "NSFNET",
            "seed": 42,
        }
        return SimStats(engine_props, sim_info="test_simulation")

    @pytest.fixture
    def sample_network_spectrum(
        self,
    ) -> dict[tuple[int, int], dict[str, list[np.ndarray]]]:
        """Create sample network spectrum for testing."""
        # Simple network with 2 links
        return {
            (0, 1): {
                "cores_matrix": [
                    np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0]),  # Fragmented
                ]
            },
            (1, 0): {
                "cores_matrix": [
                    np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0]),
                ]
            },
            (1, 2): {
                "cores_matrix": [
                    np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),  # Less fragmented
                ]
            },
            (2, 1): {
                "cores_matrix": [
                    np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
                ]
            },
        }

    def test_find_free_blocks_contiguous(self, sim_stats: SimStats) -> None:
        """Test finding free blocks in contiguous spectrum."""
        slots = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0])

        blocks = sim_stats._find_free_blocks(slots)

        assert len(blocks) == 2
        assert (0, 4) in blocks  # First 4 slots
        assert (7, 9) in blocks  # Last 2 slots

    def test_find_free_blocks_all_free(self, sim_stats: SimStats) -> None:
        """Test finding free blocks when all slots are free."""
        slots = np.array([0, 0, 0, 0])

        blocks = sim_stats._find_free_blocks(slots)

        assert len(blocks) == 1
        assert blocks[0] == (0, 4)

    def test_find_free_blocks_all_occupied(self, sim_stats: SimStats) -> None:
        """Test finding free blocks when all slots are occupied."""
        slots = np.array([1, 1, 1, 1])

        blocks = sim_stats._find_free_blocks(slots)

        assert len(blocks) == 0

    def test_compute_fragmentation_proxy(
        self,
        sim_stats: SimStats,
        sample_network_spectrum: dict[tuple[int, int], dict[str, list[np.ndarray]]],
    ) -> None:
        """Test fragmentation proxy computation."""
        path = [0, 1, 2]

        frag = sim_stats.compute_fragmentation_proxy(path, sample_network_spectrum)

        # Fragmentation should be between 0 and 1
        assert 0.0 <= frag <= 1.0

        # Since we have fragmented spectrum, frag should be > 0
        assert frag > 0.0

    def test_compute_fragmentation_fully_fragmented(self, sim_stats: SimStats) -> None:
        """Test fragmentation when spectrum is fully fragmented."""
        # Every other slot occupied
        network_spectrum = {
            (0, 1): {"cores_matrix": [np.array([1, 0, 1, 0, 1, 0, 1, 0])]},
            (1, 0): {"cores_matrix": [np.array([1, 0, 1, 0, 1, 0, 1, 0])]},
        }
        path = [0, 1]

        frag = sim_stats.compute_fragmentation_proxy(path, network_spectrum)

        # Should have high fragmentation (many small blocks)
        assert frag > 0.5

    def test_compute_fragmentation_no_free_slots(self, sim_stats: SimStats) -> None:
        """Test fragmentation when no free slots available."""
        # All slots occupied
        network_spectrum = {
            (0, 1): {"cores_matrix": [np.array([1, 1, 1, 1, 1, 1])]},
            (1, 0): {"cores_matrix": [np.array([1, 1, 1, 1, 1, 1])]},
        }
        path = [0, 1]

        frag = sim_stats.compute_fragmentation_proxy(path, network_spectrum)

        # Should return 1.0 (fully fragmented)
        assert frag == 1.0

    def test_record_fragmentation(
        self,
        sim_stats: SimStats,
        sample_network_spectrum: dict[tuple[int, int], dict[str, list[np.ndarray]]],
    ) -> None:
        """Test recording fragmentation scores."""
        path = [0, 1, 2]

        # Record multiple times
        sim_stats.record_fragmentation(path, sample_network_spectrum)
        sim_stats.record_fragmentation(path, sample_network_spectrum)

        assert len(sim_stats.fragmentation_scores) == 2
        assert all(0.0 <= score <= 1.0 for score in sim_stats.fragmentation_scores)

    def test_get_fragmentation_stats(self, sim_stats: SimStats) -> None:
        """Test fragmentation statistics computation."""
        # Add some scores
        sim_stats.fragmentation_scores = [0.3, 0.4, 0.5, 0.6, 0.7]

        stats = sim_stats.get_fragmentation_stats()

        assert stats["mean"] == 0.5
        assert stats["count"] == 5
        assert "p95" in stats

    def test_get_fragmentation_stats_empty(self, sim_stats: SimStats) -> None:
        """Test fragmentation stats when no scores recorded."""
        stats = sim_stats.get_fragmentation_stats()

        assert stats["mean"] == 0.0
        assert stats["p95"] == 0.0
        assert stats["count"] == 0


class TestDecisionTimeMetrics:
    """Test decision time tracking."""

    @pytest.fixture
    def sim_stats(self) -> SimStats:
        """Create SimStats instance for testing."""
        engine_props = {"num_requests": 1000, "erlang": 100}
        return SimStats(engine_props, sim_info="test_simulation")

    def test_record_decision_time(self, sim_stats: SimStats) -> None:
        """Test recording policy decision times."""
        sim_stats.record_decision_time(0.5)
        sim_stats.record_decision_time(0.7)
        sim_stats.record_decision_time(0.3)

        assert len(sim_stats.decision_times_ms) == 3
        assert sim_stats.decision_times_ms == [0.5, 0.7, 0.3]

    def test_get_decision_time_stats(self, sim_stats: SimStats) -> None:
        """Test decision time statistics computation."""
        sim_stats.decision_times_ms = [0.3, 0.4, 0.5, 0.6, 0.7]

        stats = sim_stats.get_decision_time_stats()

        assert stats["mean"] == 0.5
        assert stats["count"] == 5
        assert "p95" in stats

    def test_get_decision_time_stats_empty(self, sim_stats: SimStats) -> None:
        """Test decision time stats when no times recorded."""
        stats = sim_stats.get_decision_time_stats()

        assert stats["mean"] == 0.0
        assert stats["p95"] == 0.0
        assert stats["count"] == 0


class TestCSVRowExport:
    """Test to_csv_row method."""

    @pytest.fixture
    def sim_stats(self) -> SimStats:
        """Create SimStats instance with data."""
        engine_props = {
            "num_requests": 1000,
            "erlang": 100,
            "network": "NSFNET",
            "seed": 42,
            "failure_settings": {"failure_type": "link"},
            "routing_settings": {"k_paths": 4},
            "offline_rl_settings": {"policy_type": "bc"},
        }
        stats = SimStats(engine_props, sim_info="test_simulation")

        # Add some data
        stats.block_mean = 0.05
        stats.block_variance = 0.001
        stats.block_ci_percent = 3.5
        stats.bit_rate_block_mean = 0.045

        stats.recovery_times_ms = [50.0, 52.0, 51.0]
        stats.failure_window_bp = [0.08, 0.09]
        stats.fragmentation_scores = [0.3, 0.4, 0.5]
        stats.decision_times_ms = [0.5, 0.6, 0.7]

        return stats

    def test_to_csv_row_includes_all_metrics(self, sim_stats: SimStats) -> None:
        """Test that CSV row includes all metrics."""
        row = sim_stats.to_csv_row()

        # Check experiment parameters
        assert row["topology"] == "NSFNET"
        assert row["load"] == 100
        assert row["failure_type"] == "link"
        assert row["k_paths"] == 4
        assert row["policy"] == "bc"
        assert row["seed"] == 42

        # Check standard metrics
        assert row["bp_overall"] == 0.05
        assert row["bp_variance"] == 0.001
        assert row["bp_ci_percent"] == 3.5
        assert row["bit_rate_bp"] == 0.045

        # Check survivability metrics
        assert "recovery_time_mean_ms" in row
        assert "recovery_time_p95_ms" in row
        assert "bp_window_fail_mean" in row
        assert "frag_proxy_mean" in row
        assert "decision_time_mean_ms" in row

    def test_to_csv_row_computes_stats(self, sim_stats: SimStats) -> None:
        """Test that CSV row correctly computes derived stats."""
        row = sim_stats.to_csv_row()

        # Check computed values
        assert row["recovery_time_mean_ms"] == 51.0
        assert abs(row["frag_proxy_mean"] - 0.4) < 1e-9
        assert abs(row["decision_time_mean_ms"] - 0.6) < 1e-9

    def test_to_csv_row_handles_missing_data(self) -> None:
        """Test CSV row export with minimal data."""
        engine_props = {"num_requests": 100, "erlang": 50}
        stats = SimStats(engine_props, sim_info="test_simulation")

        row = stats.to_csv_row()

        # Should not crash, should have defaults
        assert row["bp_overall"] == 0.0  # No data
        assert row["recovery_time_mean_ms"] == 0.0  # No recovery events
        assert row["frag_proxy_mean"] == 0.0  # No fragmentation scores
