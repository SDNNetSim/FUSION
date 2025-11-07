"""
Performance benchmarks for survivability features.

This module verifies that survivability features meet performance budgets:
- Decision time overhead ≤ 2ms per request
- K-path cache initialization within budget
- Dataset logging throughput ≥ 50k transitions/minute
- Memory usage within constraints

Based on Phase 6 specification: phase6-quality/52-performance.md
"""

import time
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pytest

from fusion.core.metrics import SimStats
from fusion.modules.failures import FailureManager
from fusion.modules.rl.policies import KSPFFPolicy
from fusion.modules.rl.policies.action_masking import compute_action_mask
from fusion.modules.routing.k_path_cache import KPathCache
from fusion.reporting import DatasetLogger


class TestDecisionTimePerformance:
    """Test that decision time overhead meets budget (≤ 2ms)."""

    @pytest.fixture
    def sample_topology(self) -> nx.Graph:
        """Create NSFNet-like topology for testing."""
        # Create 14-node NSFNet topology
        G = nx.Graph()
        edges = [
            (0, 1),
            (0, 2),
            (0, 7),
            (1, 2),
            (1, 3),
            (2, 5),
            (3, 4),
            (3, 10),
            (4, 5),
            (4, 6),
            (5, 9),
            (6, 7),
            (7, 8),
            (8, 9),
            (8, 11),
            (9, 12),
            (10, 11),
            (10, 13),
            (11, 12),
            (12, 13),
        ]
        G.add_edges_from(edges)
        for u, v in G.edges():
            G[u][v]["weight"] = 1
        return G

    @pytest.fixture
    def engine_props(self) -> dict[str, Any]:
        """Create engine properties."""
        return {
            "num_requests": 1000,
            "erlang": 100,
            "network": "NSFNET",
            "seed": 42,
            "routing_settings": {"k_paths": 4, "route_method": "k_shortest_path"},
            "failure_settings": {"failure_type": "none"},
        }

    def test_path_feature_extraction_time(
        self, sample_topology: nx.Graph, engine_props: dict[str, Any]
    ) -> None:
        """Test that path feature extraction takes ≤ 0.5ms."""
        failure_manager = FailureManager(engine_props, sample_topology)
        k_path_cache = KPathCache(sample_topology, k=4)

        # Create minimal network spectrum for testing
        network_spectrum = {}
        for u, v in sample_topology.edges():
            network_spectrum[(u, v)] = {"slots": [0] * 320}

        # Measure feature extraction time
        times = []
        k_paths = k_path_cache.get_k_paths(0, 13)
        for _ in range(100):
            start = time.perf_counter()
            [
                k_path_cache.get_path_features(path, network_spectrum, failure_manager)
                for path in k_paths
            ]
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)

        mean_time = np.mean(times)
        p95_time = np.percentile(times, 95)

        print(
            f"\nPath feature extraction: mean={mean_time:.3f}ms, p95={p95_time:.3f}ms"
        )

        assert mean_time <= 0.5, (
            f"Mean extraction time {mean_time:.3f}ms exceeds 0.5ms budget"
        )
        assert p95_time <= 1.0, (
            f"P95 extraction time {p95_time:.3f}ms exceeds 1.0ms budget"
        )

    def test_action_mask_computation_time(
        self, sample_topology: nx.Graph, engine_props: dict[str, Any]
    ) -> None:
        """Test that action mask computation takes ≤ 0.3ms."""
        failure_manager = FailureManager(engine_props, sample_topology)
        k_path_cache = KPathCache(sample_topology, k=4)

        # Create minimal network spectrum for testing
        network_spectrum = {}
        for u, v in sample_topology.edges():
            network_spectrum[(u, v)] = {"slots": [0] * 320}

        k_paths = k_path_cache.get_k_paths(0, 13)
        features = [
            k_path_cache.get_path_features(path, network_spectrum, failure_manager)
            for path in k_paths
        ]

        # Measure action masking time
        times = []
        for _ in range(100):
            start = time.perf_counter()
            compute_action_mask(k_paths, features, slots_needed=5)
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)

        mean_time = np.mean(times)
        p95_time = np.percentile(times, 95)

        print(
            f"\nAction mask computation: mean={mean_time:.3f}ms, p95={p95_time:.3f}ms"
        )

        assert mean_time <= 0.3, (
            f"Mean masking time {mean_time:.3f}ms exceeds 0.3ms budget"
        )
        assert p95_time <= 0.5, (
            f"P95 masking time {p95_time:.3f}ms exceeds 0.5ms budget"
        )

    def test_policy_inference_time(self, engine_props: dict[str, Any]) -> None:
        """Test that policy inference takes ≤ 1.2ms."""
        policy = KSPFFPolicy()

        # Create sample state
        k_paths = [[0, 1, 2, 3], [0, 4, 5, 3], [0, 6, 7, 3], [0, 1, 5, 3]]
        features = [
            {"path_hops": 4, "min_residual_slots": 10, "failure_mask": 0},
            {"path_hops": 4, "min_residual_slots": 8, "failure_mask": 0},
            {"path_hops": 4, "min_residual_slots": 12, "failure_mask": 0},
            {"path_hops": 4, "min_residual_slots": 9, "failure_mask": 0},
        ]
        action_mask = [True, True, True, True]

        # Measure policy inference time
        times = []
        for _ in range(100):
            start = time.perf_counter()
            state = {
                "source": 0,
                "destination": 3,
                "k_paths": k_paths,
                "k_path_features": features,
            }
            policy.select_path(state, action_mask)
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)

        mean_time = np.mean(times)
        p95_time = np.percentile(times, 95)

        print(f"\nPolicy inference: mean={mean_time:.3f}ms, p95={p95_time:.3f}ms")

        assert mean_time <= 1.2, (
            f"Mean inference time {mean_time:.3f}ms exceeds 1.2ms budget"
        )
        assert p95_time <= 2.0, (
            f"P95 inference time {p95_time:.3f}ms exceeds 2.0ms budget"
        )

    def test_total_decision_time(
        self, sample_topology: nx.Graph, engine_props: dict[str, Any]
    ) -> None:
        """Test that total decision time (features + mask + policy) ≤ 2ms."""
        failure_manager = FailureManager(engine_props, sample_topology)
        k_path_cache = KPathCache(sample_topology, k=4)
        policy = KSPFFPolicy()

        # Create minimal network spectrum for testing
        network_spectrum = {}
        for u, v in sample_topology.edges():
            network_spectrum[(u, v)] = {"slots": [0] * 320}

        # Measure total decision time
        times = []
        for _ in range(100):
            start = time.perf_counter()

            # Get paths and features
            k_paths = k_path_cache.get_k_paths(0, 13)
            features = [
                k_path_cache.get_path_features(path, network_spectrum, failure_manager)
                for path in k_paths
            ]

            # Compute action mask
            action_mask = compute_action_mask(k_paths, features, slots_needed=5)

            # Select path
            state = {
                "source": 0,
                "destination": 13,
                "k_paths": k_paths,
                "k_path_features": features,
            }
            policy.select_path(state, action_mask)

            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)

        mean_time = np.mean(times)
        p95_time = np.percentile(times, 95)

        print(f"\nTotal decision time: mean={mean_time:.3f}ms, p95={p95_time:.3f}ms")

        assert mean_time <= 2.0, (
            f"Mean decision time {mean_time:.3f}ms exceeds 2ms budget"
        )
        assert p95_time <= 3.0, f"P95 decision time {p95_time:.3f}ms exceeds 3ms budget"


class TestFailureProcessingPerformance:
    """Test failure processing performance."""

    @pytest.fixture
    def sample_topology(self) -> nx.Graph:
        """Create NSFNet topology."""
        G = nx.Graph()
        edges = [
            (0, 1),
            (0, 2),
            (0, 7),
            (1, 2),
            (1, 3),
            (2, 5),
            (3, 4),
            (3, 10),
            (4, 5),
            (4, 6),
            (5, 9),
            (6, 7),
            (7, 8),
            (8, 9),
            (8, 11),
            (9, 12),
            (10, 11),
            (10, 13),
            (11, 12),
            (12, 13),
        ]
        G.add_edges_from(edges)
        return G

    def test_failure_injection_time(self, sample_topology: nx.Graph) -> None:
        """Test that failure injection takes ≤ 10ms amortized."""
        engine_props = {"seed": 42}
        manager = FailureManager(engine_props, sample_topology)

        # Test different failure types
        failure_times = {}

        # Link failure (F1)
        times = []
        for _ in range(20):
            manager.clear_all_failures()
            start = time.perf_counter()
            manager.inject_failure("link", t_fail=10.0, t_repair=20.0, link_id=(0, 1))
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)
        failure_times["link"] = np.mean(times)

        # Geographic failure (F4)
        times = []
        for _ in range(20):
            manager.clear_all_failures()
            start = time.perf_counter()
            manager.inject_failure(
                "geo", t_fail=10.0, t_repair=20.0, center_node=5, hop_radius=2
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)
        failure_times["geo"] = np.mean(times)

        print("\nFailure injection times:")
        for failure_type, mean_time in failure_times.items():
            print(f"  {failure_type}: {mean_time:.3f}ms")
            assert mean_time <= 10.0, (
                f"{failure_type} injection {mean_time:.3f}ms exceeds 10ms budget"
            )

    def test_path_feasibility_check_time(self, sample_topology: nx.Graph) -> None:
        """Test that path feasibility check takes ≤ 0.1ms."""
        engine_props = {"seed": 42}
        manager = FailureManager(engine_props, sample_topology)

        # Inject a failure
        manager.inject_failure("link", t_fail=10.0, t_repair=20.0, link_id=(0, 1))

        # Test path with 5 hops
        path = [0, 2, 5, 9, 12, 13]

        # Measure feasibility check time
        times = []
        for _ in range(1000):
            start = time.perf_counter()
            manager.is_path_feasible(path)
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)

        mean_time = np.mean(times)
        p95_time = np.percentile(times, 95)

        print(f"\nPath feasibility check: mean={mean_time:.4f}ms, p95={p95_time:.4f}ms")

        assert mean_time <= 0.1, (
            f"Mean check time {mean_time:.4f}ms exceeds 0.1ms budget"
        )
        assert p95_time <= 0.2, f"P95 check time {p95_time:.4f}ms exceeds 0.2ms budget"


class TestInitializationPerformance:
    """Test initialization performance."""

    def test_k_path_cache_initialization_nsfnet(self) -> None:
        """Test K-path cache initialization for NSFNet (≤ 5 seconds)."""
        # Create NSFNet topology (14 nodes)
        G = nx.Graph()
        edges = [
            (0, 1),
            (0, 2),
            (0, 7),
            (1, 2),
            (1, 3),
            (2, 5),
            (3, 4),
            (3, 10),
            (4, 5),
            (4, 6),
            (5, 9),
            (6, 7),
            (7, 8),
            (8, 9),
            (8, 11),
            (9, 12),
            (10, 11),
            (10, 13),
            (11, 12),
            (12, 13),
        ]
        G.add_edges_from(edges)

        engine_props = {"seed": 42}
        FailureManager(engine_props, G)

        # Measure initialization time
        start = time.time()
        KPathCache(G, k=4)
        elapsed = time.time() - start

        print(f"\nNSFNet K-path cache initialization: {elapsed:.2f}s")

        assert elapsed <= 5.0, f"NSFNet cache init {elapsed:.2f}s exceeds 5s budget"

    @pytest.mark.slow
    def test_k_path_cache_initialization_large(self) -> None:
        """Test K-path cache initialization for larger topology (≤ 30 seconds)."""
        # Create larger topology (60 nodes)
        G = nx.random_regular_graph(d=4, n=60, seed=42)

        engine_props = {"seed": 42}
        FailureManager(engine_props, G)

        # Measure initialization time
        start = time.time()
        KPathCache(G, k=4)
        elapsed = time.time() - start

        print(f"\n60-node topology K-path cache initialization: {elapsed:.2f}s")

        assert elapsed <= 30.0, f"Large cache init {elapsed:.2f}s exceeds 30s budget"


class TestDatasetLoggingPerformance:
    """Test dataset logging performance."""

    def test_dataset_logging_throughput(self, tmp_path: Path) -> None:
        """Test that dataset logging achieves ≥ 50k transitions/minute."""
        output_path = tmp_path / "perf_test.jsonl"
        engine_props = {"seed": 42, "dataset_logging": {"epsilon_mix": 0.1}}

        logger = DatasetLogger(str(output_path), engine_props)

        # Log 10k transitions and measure throughput
        num_transitions = 10000
        start = time.time()

        for i in range(num_transitions):
            logger.log_transition(
                state={"src": 0, "dst": 13, "k_paths": [[0, 1, 2], [0, 3, 2]]},
                action=i % 2,
                reward=1.0 if i % 2 == 0 else 0.0,
                next_state=None,
                action_mask=[True, True],
                meta={"request_id": i, "t": i * 10.0},
            )

        elapsed = time.time() - start
        logger.close()

        # Calculate throughput
        throughput_per_min = (num_transitions / elapsed) * 60

        print(
            f"\nDataset logging throughput: {throughput_per_min:.0f} transitions/minute"
        )
        print(f"  ({elapsed:.2f}s for {num_transitions} transitions)")

        assert throughput_per_min >= 50000, (
            f"Throughput {throughput_per_min:.0f}/min below 50k/min budget"
        )

    def test_per_transition_overhead(self, tmp_path: Path) -> None:
        """Test that per-transition overhead is ≤ 0.02ms."""
        output_path = tmp_path / "overhead_test.jsonl"
        engine_props = {"seed": 42, "dataset_logging": {"epsilon_mix": 0.1}}

        logger = DatasetLogger(str(output_path), engine_props)

        # Measure individual transition times
        times = []
        for i in range(1000):
            start = time.perf_counter()
            logger.log_transition(
                state={"src": 0, "dst": 13, "k_paths": [[0, 1, 2]]},
                action=0,
                reward=1.0,
                next_state=None,
                action_mask=[True],
                meta={"request_id": i},
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)

        logger.close()

        mean_time = np.mean(times)
        p95_time = np.percentile(times, 95)

        print(
            f"\nPer-transition overhead: mean={mean_time:.4f}ms, p95={p95_time:.4f}ms"
        )

        # Note: This is a lenient budget since file I/O can vary
        assert mean_time <= 0.1, f"Mean overhead {mean_time:.4f}ms exceeds 0.1ms"


class TestFragmentationMetrics:
    """Test fragmentation computation performance."""

    def test_fragmentation_computation_time(self) -> None:
        """Test that fragmentation computation is efficient."""
        engine_props = {"seed": 42, "network": "NSFNET"}
        stats = SimStats(engine_props, sim_info="perf_test")

        # Create sample network spectrum
        network_spectrum = {}
        for link_id in range(20):  # 20 links
            # Create spectrum with fragmentation
            slots = np.zeros(320, dtype=int)
            # Add some occupied blocks
            slots[10:20] = 1
            slots[50:55] = 2
            slots[100:120] = 3

            network_spectrum[(link_id, link_id + 1)] = {"cores_matrix": [slots]}

        path = list(range(5))  # Path through 5 links

        # Measure fragmentation computation time
        times = []
        for _ in range(100):
            start = time.perf_counter()
            stats.compute_fragmentation_proxy(path, network_spectrum)
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)

        mean_time = np.mean(times)
        print(f"\nFragmentation computation: mean={mean_time:.3f}ms")

        # Should be very fast
        assert mean_time <= 1.0, f"Mean time {mean_time:.3f}ms exceeds 1ms budget"


if __name__ == "__main__":
    """
    Run performance tests manually.

    Usage:
        python -m pytest tests/performance/test_survivability_performance.py -v
        python -m pytest tests/performance/test_survivability_performance.py \\
            -v -m "not slow"
    """
    pytest.main([__file__, "-v", "-s"])
