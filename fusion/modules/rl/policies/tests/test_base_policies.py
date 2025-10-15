"""
Tests for baseline policies (KSP-FF and 1+1).
"""

import pytest

from fusion.modules.rl.policies import (
    AllPathsMaskedError,
    KSPFFPolicy,
    OnePlusOnePolicy,
)


class TestKSPFFPolicy:
    """Test cases for KSP-FF policy."""

    def test_ksp_ff_selects_first_feasible(self):
        """Test that KSP-FF always picks first unmasked path."""
        policy = KSPFFPolicy()

        state = {}  # Not used by KSP-FF
        action_mask = [False, False, True, True]

        selected = policy.select_path(state, action_mask)
        assert selected == 2  # First feasible

    def test_ksp_ff_raises_when_all_masked(self):
        """Test that exception raised when all paths masked."""
        policy = KSPFFPolicy()

        state = {}
        action_mask = [False, False, False, False]

        with pytest.raises(AllPathsMaskedError):
            policy.select_path(state, action_mask)

    def test_ksp_ff_selects_first_when_all_feasible(self):
        """Test that first path selected when all are feasible."""
        policy = KSPFFPolicy()

        state = {}
        action_mask = [True, True, True, True]

        selected = policy.select_path(state, action_mask)
        assert selected == 0

    def test_ksp_ff_get_name(self):
        """Test that policy name is correct."""
        policy = KSPFFPolicy()
        assert policy.get_name() == "KSPFFPolicy"


class TestOnePlusOnePolicy:
    """Test cases for 1+1 policy."""

    def test_one_plus_one_selects_primary_when_feasible(self):
        """Test that primary path selected when feasible."""
        policy = OnePlusOnePolicy()

        state = {}
        action_mask = [True, True]

        selected = policy.select_path(state, action_mask)
        assert selected == 0  # Primary

    def test_one_plus_one_selects_backup_when_primary_failed(self):
        """Test that backup selected when primary fails."""
        policy = OnePlusOnePolicy()

        state = {}
        action_mask = [False, True]

        selected = policy.select_path(state, action_mask)
        assert selected == 1  # Backup

    def test_one_plus_one_raises_when_both_masked(self):
        """Test that exception raised when both paths masked."""
        policy = OnePlusOnePolicy()

        state = {}
        action_mask = [False, False]

        with pytest.raises(AllPathsMaskedError):
            policy.select_path(state, action_mask)

    def test_one_plus_one_prefers_primary(self):
        """Test that primary always preferred over backup."""
        policy = OnePlusOnePolicy()

        state = {}
        action_mask = [True, True]

        # Run multiple times to ensure consistency
        for _ in range(10):
            selected = policy.select_path(state, action_mask)
            assert selected == 0

    def test_one_plus_one_get_name(self):
        """Test that policy name is correct."""
        policy = OnePlusOnePolicy()
        assert policy.get_name() == "OnePlusOnePolicy"
