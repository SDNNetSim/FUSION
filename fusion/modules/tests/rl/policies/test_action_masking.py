"""
Tests for action masking utilities.
"""

from fusion.modules.rl.policies import (
    KSPFFPolicy,
    apply_fallback_policy,
    compute_action_mask,
)


class TestComputeActionMask:
    """Test cases for compute_action_mask function."""

    def test_mask_failed_paths(self) -> None:
        """Test that paths with failure_mask=1 are masked."""
        k_paths = [[0, 1, 2], [0, 3, 2], [0, 4, 2]]
        k_path_features = [
            {"failure_mask": 1, "min_residual_slots": 10},  # Failed
            {"failure_mask": 0, "min_residual_slots": 5},  # OK
            {"failure_mask": 0, "min_residual_slots": 8},  # OK
        ]

        mask = compute_action_mask(k_paths, k_path_features, slots_needed=4)

        assert mask == [False, True, True]

    def test_mask_insufficient_spectrum(self) -> None:
        """Test that paths with insufficient spectrum are masked."""
        k_paths = [[0, 1, 2], [0, 3, 2], [0, 4, 2]]
        k_path_features = [
            {"failure_mask": 0, "min_residual_slots": 10},  # OK
            {"failure_mask": 0, "min_residual_slots": 2},  # Insufficient
            {"failure_mask": 0, "min_residual_slots": 5},  # OK
        ]

        mask = compute_action_mask(k_paths, k_path_features, slots_needed=4)

        assert mask == [True, False, True]

    def test_mask_both_conditions(self) -> None:
        """Test masking with both failure and spectrum constraints."""
        k_paths = [[0, 1, 2], [0, 3, 2], [0, 4, 2]]
        k_path_features = [
            {"failure_mask": 1, "min_residual_slots": 10},  # Failed
            {"failure_mask": 0, "min_residual_slots": 5},  # OK
            {"failure_mask": 0, "min_residual_slots": 2},  # Insufficient
        ]

        mask = compute_action_mask(k_paths, k_path_features, slots_needed=4)

        assert mask == [False, True, False]

    def test_all_paths_feasible(self) -> None:
        """Test when all paths are feasible."""
        k_paths = [[0, 1, 2], [0, 3, 2]]
        k_path_features = [
            {"failure_mask": 0, "min_residual_slots": 10},
            {"failure_mask": 0, "min_residual_slots": 8},
        ]

        mask = compute_action_mask(k_paths, k_path_features, slots_needed=4)

        assert mask == [True, True]

    def test_all_paths_infeasible(self) -> None:
        """Test when all paths are infeasible."""
        k_paths = [[0, 1, 2], [0, 3, 2]]
        k_path_features = [
            {"failure_mask": 1, "min_residual_slots": 10},
            {"failure_mask": 1, "min_residual_slots": 8},
        ]

        mask = compute_action_mask(k_paths, k_path_features, slots_needed=4)

        assert mask == [False, False]

    def test_boundary_case_exact_slots(self) -> None:
        """Test boundary case where slots exactly match."""
        k_paths = [[0, 1, 2]]
        k_path_features = [
            {"failure_mask": 0, "min_residual_slots": 4},
        ]

        mask = compute_action_mask(k_paths, k_path_features, slots_needed=4)

        assert mask == [True]

    def test_boundary_case_one_less_slot(self) -> None:
        """Test boundary case with one less slot than needed."""
        k_paths = [[0, 1, 2]]
        k_path_features = [
            {"failure_mask": 0, "min_residual_slots": 3},
        ]

        mask = compute_action_mask(k_paths, k_path_features, slots_needed=4)

        assert mask == [False]


class TestApplyFallbackPolicy:
    """Test cases for apply_fallback_policy function."""

    def test_fallback_succeeds(self) -> None:
        """Test that fallback policy returns valid index."""
        fallback = KSPFFPolicy()

        state = {"src": 0, "dst": 5, "paths": [{"path_hops": 3}, {"path_hops": 4}]}
        action_mask = [False, False]  # All blocked initially

        idx = apply_fallback_policy(state, fallback, action_mask)

        assert idx == 0  # Fallback gets all-true mask, picks first

    def test_fallback_returns_negative_on_failure(self) -> None:
        """Test that -1 returned when fallback also fails."""
        # Create a fallback that will fail
        fallback = KSPFFPolicy()

        state = {
            "src": 0,
            "dst": 5,
            "paths": [],  # No paths available
        }
        action_mask: list[bool] = []

        idx = apply_fallback_policy(state, fallback, action_mask)

        assert idx == -1

    def test_fallback_uses_relaxed_mask(self) -> None:
        """Test that fallback gets all-true mask."""
        fallback = KSPFFPolicy()

        state = {
            "src": 0,
            "dst": 5,
            "paths": [{"path_hops": 3}, {"path_hops": 4}, {"path_hops": 5}],
        }
        action_mask = [False, False, True]

        idx = apply_fallback_policy(state, fallback, action_mask)

        # Fallback should get [True, True, True], so picks index 0
        assert idx == 0
