"""Tests for heuristic policies.

Phase: P5.2 - Heuristic Policies
"""

from dataclasses import dataclass

import pytest

from fusion.interfaces.control_policy import ControlPolicy
from fusion.modules.rl.adapter import PathOption
from fusion.policies.heuristic_policy import (
    FirstFeasiblePolicy,
    HeuristicPolicy,
    LeastCongestedPolicy,
    LoadBalancedPolicy,
    RandomFeasiblePolicy,
    ShortestFeasiblePolicy,
)


@dataclass
class MockRequest:
    """Mock Request for testing."""

    request_id: int = 1
    source: str = "0"
    destination: str = "5"
    bandwidth_gbps: int = 100
    arrive_time: float = 0.0
    depart_time: float = 1.0


@dataclass
class MockNetworkState:
    """Mock NetworkState for testing."""

    pass


def create_path_option(
    path_index: int = 0,
    is_feasible: bool = True,
    weight_km: float = 100.0,
    congestion: float = 0.3,
    slots_needed: int = 4,
) -> PathOption:
    """Helper to create a PathOption for testing."""
    return PathOption(
        path_index=path_index,
        path=("0", "1", "2"),
        weight_km=weight_km,
        num_hops=2,
        modulation="QPSK",
        slots_needed=slots_needed,
        is_feasible=is_feasible,
        congestion=congestion,
        available_slots=0.7,
    )


class TestHeuristicPolicyProtocolCompliance:
    """Tests for heuristic policies implementing ControlPolicy protocol."""

    @pytest.mark.parametrize(
        "policy_class",
        [
            FirstFeasiblePolicy,
            ShortestFeasiblePolicy,
            LeastCongestedPolicy,
            RandomFeasiblePolicy,
            LoadBalancedPolicy,
        ],
    )
    def test_implements_control_policy(self, policy_class: type) -> None:
        """All heuristic policies should satisfy ControlPolicy protocol."""
        policy = policy_class()
        assert isinstance(policy, ControlPolicy)

    @pytest.mark.parametrize(
        "policy_class",
        [
            FirstFeasiblePolicy,
            ShortestFeasiblePolicy,
            LeastCongestedPolicy,
            RandomFeasiblePolicy,
            LoadBalancedPolicy,
        ],
    )
    def test_has_required_methods(self, policy_class: type) -> None:
        """All heuristic policies should have required methods."""
        policy = policy_class()
        assert hasattr(policy, "select_action")
        assert hasattr(policy, "update")
        assert hasattr(policy, "get_name")
        assert callable(policy.select_action)
        assert callable(policy.update)
        assert callable(policy.get_name)

    @pytest.mark.parametrize(
        "policy_class",
        [
            FirstFeasiblePolicy,
            ShortestFeasiblePolicy,
            LeastCongestedPolicy,
            RandomFeasiblePolicy,
            LoadBalancedPolicy,
        ],
    )
    def test_is_heuristic_policy(self, policy_class: type) -> None:
        """All heuristic policies should inherit from HeuristicPolicy."""
        policy = policy_class()
        assert isinstance(policy, HeuristicPolicy)


class TestFirstFeasiblePolicy:
    """Tests for FirstFeasiblePolicy."""

    def test_selects_first_feasible(self) -> None:
        """Should select the first feasible path in index order."""
        policy = FirstFeasiblePolicy()
        options = [
            create_path_option(path_index=0, is_feasible=False),
            create_path_option(path_index=1, is_feasible=True),
            create_path_option(path_index=2, is_feasible=True),
        ]

        action = policy.select_action(MockRequest(), options, MockNetworkState())

        assert action == 1

    def test_selects_first_when_all_feasible(self) -> None:
        """Should select index 0 when all are feasible."""
        policy = FirstFeasiblePolicy()
        options = [
            create_path_option(path_index=0, is_feasible=True),
            create_path_option(path_index=1, is_feasible=True),
            create_path_option(path_index=2, is_feasible=True),
        ]

        action = policy.select_action(MockRequest(), options, MockNetworkState())

        assert action == 0

    def test_returns_negative_one_when_none_feasible(self) -> None:
        """Should return -1 when no feasible option."""
        policy = FirstFeasiblePolicy()
        options = [
            create_path_option(path_index=0, is_feasible=False),
            create_path_option(path_index=1, is_feasible=False),
        ]

        action = policy.select_action(MockRequest(), options, MockNetworkState())

        assert action == -1

    def test_returns_negative_one_when_empty(self) -> None:
        """Should return -1 when options list is empty."""
        policy = FirstFeasiblePolicy()

        action = policy.select_action(MockRequest(), [], MockNetworkState())

        assert action == -1

    def test_get_name(self) -> None:
        """get_name should return class name."""
        policy = FirstFeasiblePolicy()
        assert policy.get_name() == "FirstFeasiblePolicy"


class TestShortestFeasiblePolicy:
    """Tests for ShortestFeasiblePolicy."""

    def test_selects_shortest_by_weight(self) -> None:
        """Should select the feasible path with minimum weight_km."""
        policy = ShortestFeasiblePolicy()
        options = [
            create_path_option(path_index=0, is_feasible=True, weight_km=200.0),
            create_path_option(path_index=1, is_feasible=True, weight_km=100.0),
            create_path_option(path_index=2, is_feasible=True, weight_km=150.0),
        ]

        action = policy.select_action(MockRequest(), options, MockNetworkState())

        assert action == 1

    def test_ignores_infeasible_paths(self) -> None:
        """Should ignore infeasible paths even if shorter."""
        policy = ShortestFeasiblePolicy()
        options = [
            create_path_option(path_index=0, is_feasible=False, weight_km=50.0),
            create_path_option(path_index=1, is_feasible=True, weight_km=200.0),
            create_path_option(path_index=2, is_feasible=True, weight_km=150.0),
        ]

        action = policy.select_action(MockRequest(), options, MockNetworkState())

        assert action == 2

    def test_tiebreak_by_first_occurrence(self) -> None:
        """Should select first path when tied on weight."""
        policy = ShortestFeasiblePolicy()
        options = [
            create_path_option(path_index=0, is_feasible=True, weight_km=100.0),
            create_path_option(path_index=1, is_feasible=True, weight_km=100.0),
            create_path_option(path_index=2, is_feasible=True, weight_km=100.0),
        ]

        action = policy.select_action(MockRequest(), options, MockNetworkState())

        assert action == 0

    def test_returns_negative_one_when_none_feasible(self) -> None:
        """Should return -1 when no feasible option."""
        policy = ShortestFeasiblePolicy()
        options = [
            create_path_option(path_index=0, is_feasible=False, weight_km=100.0),
        ]

        action = policy.select_action(MockRequest(), options, MockNetworkState())

        assert action == -1

    def test_get_name(self) -> None:
        """get_name should return class name."""
        policy = ShortestFeasiblePolicy()
        assert policy.get_name() == "ShortestFeasiblePolicy"


class TestLeastCongestedPolicy:
    """Tests for LeastCongestedPolicy."""

    def test_selects_least_congested(self) -> None:
        """Should select the feasible path with minimum congestion."""
        policy = LeastCongestedPolicy()
        options = [
            create_path_option(path_index=0, is_feasible=True, congestion=0.5),
            create_path_option(path_index=1, is_feasible=True, congestion=0.2),
            create_path_option(path_index=2, is_feasible=True, congestion=0.8),
        ]

        action = policy.select_action(MockRequest(), options, MockNetworkState())

        assert action == 1

    def test_ignores_infeasible_paths(self) -> None:
        """Should ignore infeasible paths even if less congested."""
        policy = LeastCongestedPolicy()
        options = [
            create_path_option(path_index=0, is_feasible=False, congestion=0.1),
            create_path_option(path_index=1, is_feasible=True, congestion=0.5),
            create_path_option(path_index=2, is_feasible=True, congestion=0.3),
        ]

        action = policy.select_action(MockRequest(), options, MockNetworkState())

        assert action == 2

    def test_tiebreak_by_first_occurrence(self) -> None:
        """Should select first path when tied on congestion."""
        policy = LeastCongestedPolicy()
        options = [
            create_path_option(path_index=0, is_feasible=True, congestion=0.3),
            create_path_option(path_index=1, is_feasible=True, congestion=0.3),
        ]

        action = policy.select_action(MockRequest(), options, MockNetworkState())

        assert action == 0

    def test_returns_negative_one_when_none_feasible(self) -> None:
        """Should return -1 when no feasible option."""
        policy = LeastCongestedPolicy()
        options = [
            create_path_option(path_index=0, is_feasible=False, congestion=0.1),
        ]

        action = policy.select_action(MockRequest(), options, MockNetworkState())

        assert action == -1

    def test_get_name(self) -> None:
        """get_name should return class name."""
        policy = LeastCongestedPolicy()
        assert policy.get_name() == "LeastCongestedPolicy"


class TestRandomFeasiblePolicy:
    """Tests for RandomFeasiblePolicy."""

    def test_selects_from_feasible_only(self) -> None:
        """Should only select from feasible paths."""
        policy = RandomFeasiblePolicy(seed=42)
        options = [
            create_path_option(path_index=0, is_feasible=False),
            create_path_option(path_index=1, is_feasible=True),
            create_path_option(path_index=2, is_feasible=False),
        ]

        # Run multiple times - should always return 1
        for _ in range(10):
            policy.reset_rng(seed=42)
            action = policy.select_action(MockRequest(), options, MockNetworkState())
            assert action == 1

    def test_reproducibility_with_seed(self) -> None:
        """Same seed should produce same sequence of selections."""
        options = [
            create_path_option(path_index=0, is_feasible=True),
            create_path_option(path_index=1, is_feasible=True),
            create_path_option(path_index=2, is_feasible=True),
        ]

        policy1 = RandomFeasiblePolicy(seed=12345)
        policy2 = RandomFeasiblePolicy(seed=12345)

        actions1 = [
            policy1.select_action(MockRequest(), options, MockNetworkState())
            for _ in range(20)
        ]
        actions2 = [
            policy2.select_action(MockRequest(), options, MockNetworkState())
            for _ in range(20)
        ]

        assert actions1 == actions2

    def test_different_seeds_produce_different_results(self) -> None:
        """Different seeds should typically produce different sequences."""
        options = [
            create_path_option(path_index=0, is_feasible=True),
            create_path_option(path_index=1, is_feasible=True),
            create_path_option(path_index=2, is_feasible=True),
            create_path_option(path_index=3, is_feasible=True),
            create_path_option(path_index=4, is_feasible=True),
        ]

        policy1 = RandomFeasiblePolicy(seed=111)
        policy2 = RandomFeasiblePolicy(seed=222)

        actions1 = [
            policy1.select_action(MockRequest(), options, MockNetworkState())
            for _ in range(50)
        ]
        actions2 = [
            policy2.select_action(MockRequest(), options, MockNetworkState())
            for _ in range(50)
        ]

        # Very unlikely to be identical with different seeds
        assert actions1 != actions2

    def test_reset_rng_restores_sequence(self) -> None:
        """reset_rng should restore original sequence."""
        policy = RandomFeasiblePolicy(seed=42)
        options = [
            create_path_option(path_index=0, is_feasible=True),
            create_path_option(path_index=1, is_feasible=True),
            create_path_option(path_index=2, is_feasible=True),
        ]

        # Get first sequence
        first_actions = [
            policy.select_action(MockRequest(), options, MockNetworkState())
            for _ in range(10)
        ]

        # Reset and get second sequence
        policy.reset_rng()
        second_actions = [
            policy.select_action(MockRequest(), options, MockNetworkState())
            for _ in range(10)
        ]

        assert first_actions == second_actions

    def test_reset_rng_with_new_seed(self) -> None:
        """reset_rng with new seed should change sequence."""
        policy = RandomFeasiblePolicy(seed=42)
        options = [
            create_path_option(path_index=0, is_feasible=True),
            create_path_option(path_index=1, is_feasible=True),
            create_path_option(path_index=2, is_feasible=True),
        ]

        # Get sequence with seed 42
        policy.reset_rng(seed=42)
        actions_42 = [
            policy.select_action(MockRequest(), options, MockNetworkState())
            for _ in range(20)
        ]

        # Get sequence with seed 99
        policy.reset_rng(seed=99)
        actions_99 = [
            policy.select_action(MockRequest(), options, MockNetworkState())
            for _ in range(20)
        ]

        assert actions_42 != actions_99

    def test_returns_negative_one_when_none_feasible(self) -> None:
        """Should return -1 when no feasible option."""
        policy = RandomFeasiblePolicy(seed=42)
        options = [
            create_path_option(path_index=0, is_feasible=False),
            create_path_option(path_index=1, is_feasible=False),
        ]

        action = policy.select_action(MockRequest(), options, MockNetworkState())

        assert action == -1

    def test_returns_negative_one_when_empty(self) -> None:
        """Should return -1 when options list is empty."""
        policy = RandomFeasiblePolicy(seed=42)

        action = policy.select_action(MockRequest(), [], MockNetworkState())

        assert action == -1

    def test_get_name_with_seed(self) -> None:
        """get_name should include seed when provided."""
        policy = RandomFeasiblePolicy(seed=42)
        assert policy.get_name() == "RandomFeasiblePolicy(seed=42)"

    def test_get_name_without_seed(self) -> None:
        """get_name should not include seed when not provided."""
        policy = RandomFeasiblePolicy()
        assert policy.get_name() == "RandomFeasiblePolicy"


class TestLoadBalancedPolicy:
    """Tests for LoadBalancedPolicy."""

    def test_default_alpha_is_half(self) -> None:
        """Default alpha should be 0.5."""
        policy = LoadBalancedPolicy()
        assert policy.alpha == 0.5

    def test_custom_alpha(self) -> None:
        """Should accept custom alpha values."""
        policy = LoadBalancedPolicy(alpha=0.7)
        assert policy.alpha == 0.7

    def test_alpha_zero_prioritizes_congestion(self) -> None:
        """alpha=0 should behave like LeastCongestedPolicy."""
        policy = LoadBalancedPolicy(alpha=0.0)
        options = [
            create_path_option(
                path_index=0, is_feasible=True, weight_km=50.0, congestion=0.8
            ),
            create_path_option(
                path_index=1, is_feasible=True, weight_km=200.0, congestion=0.1
            ),
        ]

        action = policy.select_action(MockRequest(), options, MockNetworkState())

        # Should select least congested (index 1) despite longer path
        assert action == 1

    def test_alpha_one_prioritizes_length(self) -> None:
        """alpha=1 should behave like ShortestFeasiblePolicy."""
        policy = LoadBalancedPolicy(alpha=1.0)
        options = [
            create_path_option(
                path_index=0, is_feasible=True, weight_km=50.0, congestion=0.9
            ),
            create_path_option(
                path_index=1, is_feasible=True, weight_km=200.0, congestion=0.1
            ),
        ]

        action = policy.select_action(MockRequest(), options, MockNetworkState())

        # Should select shortest (index 0) despite higher congestion
        assert action == 0

    def test_balanced_selection(self) -> None:
        """alpha=0.5 should balance length and congestion."""
        policy = LoadBalancedPolicy(alpha=0.5)
        # Options designed so the middle one has best combined score
        options = [
            create_path_option(
                path_index=0,
                is_feasible=True,
                weight_km=100.0,  # normalized: 0.5
                congestion=0.8,  # score = 0.5*0.5 + 0.5*0.8 = 0.65
            ),
            create_path_option(
                path_index=1,
                is_feasible=True,
                weight_km=200.0,  # normalized: 1.0
                congestion=0.1,  # score = 0.5*1.0 + 0.5*0.1 = 0.55
            ),
            create_path_option(
                path_index=2,
                is_feasible=True,
                weight_km=50.0,  # normalized: 0.25
                congestion=0.2,  # score = 0.5*0.25 + 0.5*0.2 = 0.225
            ),
        ]

        action = policy.select_action(MockRequest(), options, MockNetworkState())

        # Index 2 has best score (0.225)
        assert action == 2

    def test_handles_zero_weight(self) -> None:
        """Should handle paths with zero weight without division error."""
        policy = LoadBalancedPolicy(alpha=0.5)
        options = [
            create_path_option(
                path_index=0, is_feasible=True, weight_km=0.0, congestion=0.5
            ),
            create_path_option(
                path_index=1, is_feasible=True, weight_km=0.0, congestion=0.3
            ),
        ]

        action = policy.select_action(MockRequest(), options, MockNetworkState())

        # Should not raise - with all zero weights, congestion determines
        assert action == 1

    def test_ignores_infeasible_paths(self) -> None:
        """Should ignore infeasible paths."""
        policy = LoadBalancedPolicy(alpha=0.5)
        options = [
            create_path_option(
                path_index=0, is_feasible=False, weight_km=10.0, congestion=0.1
            ),
            create_path_option(
                path_index=1, is_feasible=True, weight_km=100.0, congestion=0.5
            ),
        ]

        action = policy.select_action(MockRequest(), options, MockNetworkState())

        assert action == 1

    def test_returns_negative_one_when_none_feasible(self) -> None:
        """Should return -1 when no feasible option."""
        policy = LoadBalancedPolicy()
        options = [
            create_path_option(path_index=0, is_feasible=False),
        ]

        action = policy.select_action(MockRequest(), options, MockNetworkState())

        assert action == -1

    def test_alpha_validation_too_low(self) -> None:
        """Should reject alpha < 0."""
        with pytest.raises(ValueError, match="alpha must be in"):
            LoadBalancedPolicy(alpha=-0.1)

    def test_alpha_validation_too_high(self) -> None:
        """Should reject alpha > 1."""
        with pytest.raises(ValueError, match="alpha must be in"):
            LoadBalancedPolicy(alpha=1.5)

    def test_get_name_includes_alpha(self) -> None:
        """get_name should include alpha value."""
        policy = LoadBalancedPolicy(alpha=0.7)
        assert policy.get_name() == "LoadBalancedPolicy(alpha=0.7)"


class TestHeuristicPolicyUpdate:
    """Tests for update() being a no-op."""

    @pytest.mark.parametrize(
        "policy_class",
        [
            FirstFeasiblePolicy,
            ShortestFeasiblePolicy,
            LeastCongestedPolicy,
            RandomFeasiblePolicy,
            LoadBalancedPolicy,
        ],
    )
    def test_update_is_noop(self, policy_class: type) -> None:
        """update() should be a no-op for all heuristic policies."""
        policy = policy_class()

        # Should not raise
        policy.update(MockRequest(), 1, 1.0)
        policy.update(MockRequest(), 0, -1.0)
        policy.update(MockRequest(), 2, 0.0)


class TestHeuristicPolicyEdgeCases:
    """Tests for edge cases across all policies."""

    @pytest.mark.parametrize(
        "policy_class",
        [
            FirstFeasiblePolicy,
            ShortestFeasiblePolicy,
            LeastCongestedPolicy,
            RandomFeasiblePolicy,
            LoadBalancedPolicy,
        ],
    )
    def test_single_feasible_option(self, policy_class: type) -> None:
        """Should handle single feasible option."""
        policy = policy_class()
        options = [create_path_option(path_index=0, is_feasible=True)]

        action = policy.select_action(MockRequest(), options, MockNetworkState())

        assert action == 0

    @pytest.mark.parametrize(
        "policy_class",
        [
            FirstFeasiblePolicy,
            ShortestFeasiblePolicy,
            LeastCongestedPolicy,
            RandomFeasiblePolicy,
            LoadBalancedPolicy,
        ],
    )
    def test_last_option_only_feasible(self, policy_class: type) -> None:
        """Should find last option when it's the only feasible one."""
        policy = policy_class()
        options = [
            create_path_option(path_index=0, is_feasible=False),
            create_path_option(path_index=1, is_feasible=False),
            create_path_option(path_index=2, is_feasible=True),
        ]

        action = policy.select_action(MockRequest(), options, MockNetworkState())

        assert action == 2


class TestPolicyImports:
    """Tests for policy imports from package."""

    def test_import_from_policies_package(self) -> None:
        """Should be able to import all policies from fusion.policies."""
        from fusion.policies import (
            ControlPolicy,
            FirstFeasiblePolicy,
            HeuristicPolicy,
            LeastCongestedPolicy,
            LoadBalancedPolicy,
            RandomFeasiblePolicy,
            ShortestFeasiblePolicy,
        )

        # Verify types
        assert FirstFeasiblePolicy is not None
        assert ShortestFeasiblePolicy is not None
        assert LeastCongestedPolicy is not None
        assert RandomFeasiblePolicy is not None
        assert LoadBalancedPolicy is not None
        assert HeuristicPolicy is not None
        assert ControlPolicy is not None
