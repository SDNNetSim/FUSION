"""
Integration tests for the complete survivability pipeline.

This module tests the end-to-end survivability workflow including:
- Failure injection and recovery
- Protection mechanisms (1+1)
- RL policy inference with action masking
- Dataset logging
- Metrics collection and reporting

These tests verify that all phase 2-5 components work together correctly.
"""

import json
from pathlib import Path
from typing import Any

import networkx as nx
import pytest

from fusion.core.metrics import SimStats
from fusion.modules.failures import FailureManager
from fusion.modules.rl.policies import KSPFFPolicy
from fusion.modules.routing import KPathCache
from fusion.reporting import DatasetLogger


class TestSurvivabilityPipelineIntegration:
    """Integration tests for survivability pipeline."""

    @pytest.fixture
    def engine_props(self) -> dict[str, Any]:
        """Create engine properties for testing."""
        return {
            "num_requests": 100,
            "erlang": 100,
            "network": "NSFNET",
            "seed": 42,
            "failure_settings": {
                "failure_type": "link",
                "t_fail_arrival_index": 50,
                "t_repair_after_arrivals": 20,
                "failed_link_src": 0,
                "failed_link_dst": 1,
            },
            "routing_settings": {"k_paths": 4, "route_method": "k_shortest_path"},
            "offline_rl_settings": {
                "policy_type": "ksp_ff",
                "fallback_policy": "ksp_ff",
            },
            "dataset_logging": {
                "log_offline_dataset": True,
                "dataset_output_path": "/tmp/test_dataset.jsonl",
                "epsilon_mix": 0.1,
            },
            "recovery_timing": {
                "protection_switchover_ms": 50.0,
                "restoration_latency_ms": 100.0,
                "failure_window_size": 1000,
            },
        }

    @pytest.fixture
    def sample_topology(self) -> nx.Graph:
        """Create sample network topology."""
        G = nx.Graph()
        # Simple 4-node network with redundant paths
        G.add_edges_from(
            [
                (0, 1, {"weight": 1}),
                (1, 2, {"weight": 1}),
                (2, 3, {"weight": 1}),
                (0, 4, {"weight": 1}),
                (4, 3, {"weight": 1}),
            ]
        )
        return G

    def test_failure_manager_integration(
        self, engine_props: dict[str, Any], sample_topology: nx.Graph
    ) -> None:
        """Test FailureManager integration with topology."""
        manager = FailureManager(engine_props, sample_topology)

        # Inject link failure
        event = manager.inject_failure(
            "link", t_fail=10.0, t_repair=20.0, link_id=(0, 1)
        )

        assert event["failure_type"] == "link"
        assert len(event["failed_links"]) == 1
        assert (0, 1) in event["failed_links"]

        # Verify path feasibility checking
        path_with_failure = [0, 1, 2]
        path_without_failure = [0, 4, 3]

        assert not manager.is_path_feasible(path_with_failure)
        assert manager.is_path_feasible(path_without_failure)

        # Verify repair
        repaired = manager.repair_failures(20.0)
        assert len(repaired) == 1
        assert manager.is_path_feasible(path_with_failure)

    def test_k_path_cache_with_failures(
        self, engine_props: dict[str, Any], sample_topology: nx.Graph
    ) -> None:
        """Test K-path cache integration with failure manager."""
        # Initialize components
        failure_manager = FailureManager(engine_props, sample_topology)
        k_path_cache = KPathCache(sample_topology, k=4, failure_manager=failure_manager)

        # Get K paths before failure
        paths_before = k_path_cache.get_k_paths(0, 3)
        assert len(paths_before) > 0

        # Inject failure
        failure_manager.inject_failure(
            "link", t_fail=10.0, t_repair=20.0, link_id=(0, 1)
        )

        # Get path features with failure
        paths = k_path_cache.get_k_paths(0, 3)
        features = k_path_cache.get_path_features(0, 3, slots_needed=5)

        assert len(features) == len(paths)
        # At least one path should be marked as having failure
        failure_masks = [f["failure_mask"] for f in features]
        assert any(mask > 0 for mask in failure_masks)

    def test_policy_with_action_masking(
        self, engine_props: dict[str, Any], sample_topology: nx.Graph
    ) -> None:
        """Test policy integration with action masking."""
        from fusion.modules.rl.policies.action_masking import compute_action_mask

        # Create policy
        policy = KSPFFPolicy(engine_props)

        # Create K-path cache with failures
        failure_manager = FailureManager(engine_props, sample_topology)
        failure_manager.inject_failure(
            "link", t_fail=10.0, t_repair=20.0, link_id=(1, 2)
        )

        k_path_cache = KPathCache(sample_topology, k=4, failure_manager=failure_manager)

        # Get paths and features
        k_paths = k_path_cache.get_k_paths(0, 3)
        k_path_features = k_path_cache.get_path_features(0, 3, slots_needed=5)

        # Compute action mask
        action_mask = compute_action_mask(k_paths, k_path_features, slots_needed=5)

        assert len(action_mask) == len(k_paths)
        # Some paths should be masked due to failure
        assert not all(action_mask), "Expected some paths to be masked"

    def test_dataset_logger_integration(
        self, engine_props: dict[str, Any], tmp_path: Path
    ) -> None:
        """Test dataset logger integration."""
        # Create dataset logger with temp path
        output_path = tmp_path / "test_dataset.jsonl"
        engine_props["dataset_logging"]["dataset_output_path"] = str(output_path)

        logger = DatasetLogger(str(output_path), engine_props)

        # Log some transitions
        for i in range(5):
            logger.log_transition(
                state={"src": 0, "dst": 3, "k_paths": [[0, 1, 3], [0, 4, 3]]},
                action=0 if i % 2 == 0 else 1,
                reward=1.0 if i % 2 == 0 else 0.0,
                next_state=None,
                action_mask=[True, True],
                meta={"request_id": i, "t": i * 10.0},
            )

        logger.close()

        # Verify file exists and is valid JSONL
        assert output_path.exists()

        with open(output_path, encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 5

            # Verify each line is valid JSON
            for line in lines:
                data = json.loads(line)
                assert "state" in data
                assert "action" in data
                assert "reward" in data
                assert "action_mask" in data
                assert "meta" in data

    def test_metrics_collection(self, engine_props: dict[str, Any]) -> None:
        """Test survivability metrics collection."""
        stats = SimStats(engine_props, sim_info="test_simulation")

        # Record some recovery events
        stats.recovery_times_ms.extend([50.5, 52.3, 48.9, 51.2])

        # Record fragmentation scores
        stats.fragmentation_scores.extend([0.25, 0.30, 0.28, 0.32])

        # Record decision times
        stats.decision_times_ms.extend([1.2, 1.5, 1.3, 1.4])

        # Verify metrics are tracked
        assert len(stats.recovery_times_ms) == 4
        assert len(stats.fragmentation_scores) == 4
        assert len(stats.decision_times_ms) == 4

        # Verify to_csv_row includes survivability metrics
        csv_row = stats.to_csv_row()
        assert "recovery_time_mean_ms" in csv_row
        assert "frag_proxy_mean" in csv_row
        assert "decision_time_mean_ms" in csv_row

    def test_end_to_end_workflow(
        self, engine_props: dict[str, Any], sample_topology: nx.Graph, tmp_path: Path
    ) -> None:
        """
        Test complete end-to-end survivability workflow.

        This test simulates a minimal version of the full pipeline:
        1. Initialize all components
        2. Inject failure
        3. Generate routing decisions with masking
        4. Log transitions to dataset
        5. Collect metrics
        6. Verify all outputs
        """
        # Setup output path
        dataset_path = tmp_path / "e2e_dataset.jsonl"
        engine_props["dataset_logging"]["dataset_output_path"] = str(dataset_path)

        # Initialize components
        failure_manager = FailureManager(engine_props, sample_topology)
        k_path_cache = KPathCache(sample_topology, k=4, failure_manager=failure_manager)
        policy = KSPFFPolicy(engine_props)
        dataset_logger = DatasetLogger(str(dataset_path), engine_props)
        stats = SimStats(engine_props, sim_info="e2e_test")

        # Simulate workflow
        # Step 1: Inject failure at t=100
        failure_event = failure_manager.inject_failure(
            "link", t_fail=100.0, t_repair=200.0, link_id=(0, 1)
        )
        assert len(failure_event["failed_links"]) > 0

        # Step 2: Process some requests
        for request_id in range(10):
            t = 50.0 + request_id * 10.0

            # Get K paths and features
            k_paths = k_path_cache.get_k_paths(0, 3)
            features = k_path_cache.get_path_features(0, 3, slots_needed=5)

            if not k_paths:
                continue

            # Compute action mask
            from fusion.modules.rl.policies.action_masking import compute_action_mask

            action_mask = compute_action_mask(k_paths, features, slots_needed=5)

            # Select path (use first feasible)
            action = next((i for i, mask in enumerate(action_mask) if mask), 0)

            # Log transition
            dataset_logger.log_transition(
                state={"src": 0, "dst": 3, "k_paths": k_paths},
                action=action,
                reward=1.0 if action_mask[action] else 0.0,
                next_state=None,
                action_mask=action_mask,
                meta={"request_id": request_id, "t": t},
            )

            # Record decision time
            stats.decision_times_ms.append(1.5)

        # Step 3: Repair failures
        repaired = failure_manager.repair_failures(200.0)
        assert len(repaired) > 0

        # Record recovery time
        recovery_time = 200.0 - 100.0  # Repair time - failure time
        stats.recovery_times_ms.append(recovery_time)

        # Step 4: Close logger and verify outputs
        dataset_logger.close()

        assert dataset_path.exists()
        assert len(stats.recovery_times_ms) > 0
        assert len(stats.decision_times_ms) > 0

        # Verify dataset content
        with open(dataset_path, encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) > 0
            for line in lines:
                data = json.loads(line)
                assert "action_mask" in data


class TestBackwardCompatibility:
    """Regression tests for backward compatibility."""

    def test_legacy_config_still_works(self) -> None:
        """Test that legacy configurations without survivability features still work."""
        legacy_props = {
            "num_requests": 100,
            "erlang": 100,
            "network": "NSFNET",
            "seed": 42,
            "routing_settings": {"route_method": "k_shortest_path"},
        }

        # Should be able to create SimStats without survivability settings
        stats = SimStats(legacy_props, sim_info="legacy_test")

        assert hasattr(stats, "recovery_times_ms")
        assert hasattr(stats, "fragmentation_scores")
        assert len(stats.recovery_times_ms) == 0

    def test_no_failures_mode_works(self) -> None:
        """Test that simulations with failure_type='none' work correctly."""
        props = {
            "num_requests": 100,
            "erlang": 100,
            "network": "NSFNET",
            "seed": 42,
            "failure_settings": {"failure_type": "none"},
        }

        stats = SimStats(props, sim_info="no_failures_test")

        # Should not crash when computing metrics with no failures
        csv_row = stats.to_csv_row()
        assert "recovery_time_mean_ms" in csv_row
        assert csv_row["recovery_time_mean_ms"] == 0.0
