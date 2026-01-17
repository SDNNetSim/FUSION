"""Tests for PathOption dataclass.

Phase: P4.1 - RLSimulationAdapter Scaffolding
Chunk: 1 - PathOption dataclass
"""

from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pytest

from fusion.modules.rl.adapter import (
    PathOption,
    compute_action_mask,
)


def create_valid_path_option(**kwargs: Any) -> PathOption:
    """Helper to create a valid PathOption with defaults."""
    defaults: dict[str, Any] = {
        "path_index": 0,
        "path": ("0", "1", "2"),
        "weight_km": 100.0,
        "num_hops": 2,
        "modulation": "QPSK",
        "slots_needed": 4,
        "is_feasible": True,
        "congestion": 0.3,
        "available_slots": 0.7,
    }
    defaults.update(kwargs)
    return PathOption(**defaults)


class TestPathOptionCreation:
    """Tests for PathOption creation and basic properties."""

    def test_create_minimal_path_option(self) -> None:
        """PathOption can be created with required fields."""
        opt = create_valid_path_option()

        assert opt.path_index == 0
        assert opt.path == ("0", "1", "2")
        assert opt.weight_km == 100.0
        assert opt.num_hops == 2
        assert opt.modulation == "QPSK"
        assert opt.slots_needed == 4
        assert opt.is_feasible is True
        assert opt.congestion == 0.3
        assert opt.available_slots == 0.7

    def test_create_with_optional_fields(self) -> None:
        """PathOption can be created with optional spectrum fields."""
        opt = create_valid_path_option(
            spectrum_start=100,
            spectrum_end=108,
            core_index=0,
            band="C",
        )

        assert opt.spectrum_start == 100
        assert opt.spectrum_end == 108
        assert opt.core_index == 0
        assert opt.band == "C"

    def test_optional_fields_default_to_none(self) -> None:
        """Optional fields default to None."""
        opt = create_valid_path_option()

        assert opt.spectrum_start is None
        assert opt.spectrum_end is None
        assert opt.core_index is None
        assert opt.band is None

    def test_modulation_can_be_none(self) -> None:
        """Modulation field can be None for infeasible paths."""
        opt = create_valid_path_option(modulation=None, is_feasible=False)

        assert opt.modulation is None

    def test_path_is_tuple(self) -> None:
        """Path must be a tuple for immutability."""
        opt = create_valid_path_option(path=("A", "B", "C", "D"))

        assert isinstance(opt.path, tuple)
        assert opt.path == ("A", "B", "C", "D")
        assert opt.num_hops == 2  # num_hops is separate from path length


class TestPathOptionImmutability:
    """Tests for PathOption frozen/immutable behavior."""

    def test_path_option_is_frozen(self) -> None:
        """PathOption should be frozen and immutable."""
        opt = create_valid_path_option()

        with pytest.raises(FrozenInstanceError):
            opt.is_feasible = False  # type: ignore[misc]

    def test_path_option_is_hashable(self) -> None:
        """PathOption should be hashable (usable as dict key)."""
        opt = create_valid_path_option()

        # Should not raise
        hash_value = hash(opt)
        assert isinstance(hash_value, int)

        # Can use as dict key
        d = {opt: "value"}
        assert d[opt] == "value"

    def test_equal_path_options_have_same_hash(self) -> None:
        """Equal PathOptions should have the same hash."""
        opt1 = create_valid_path_option()
        opt2 = create_valid_path_option()

        assert opt1 == opt2
        assert hash(opt1) == hash(opt2)


class TestPathOptionValidation:
    """Tests for PathOption validation in __post_init__."""

    def test_negative_path_index_raises(self) -> None:
        """Negative path_index should raise ValueError."""
        with pytest.raises(ValueError, match="path_index must be non-negative"):
            create_valid_path_option(path_index=-1)

    def test_negative_weight_raises(self) -> None:
        """Negative weight_km should raise ValueError."""
        with pytest.raises(ValueError, match="weight_km must be non-negative"):
            create_valid_path_option(weight_km=-10.0)

    def test_congestion_below_zero_raises(self) -> None:
        """Congestion below 0 should raise ValueError."""
        with pytest.raises(ValueError, match="congestion must be in"):
            create_valid_path_option(congestion=-0.1)

    def test_congestion_above_one_raises(self) -> None:
        """Congestion above 1 should raise ValueError."""
        with pytest.raises(ValueError, match="congestion must be in"):
            create_valid_path_option(congestion=1.1)

    def test_available_slots_below_zero_raises(self) -> None:
        """available_slots below 0 should raise ValueError."""
        with pytest.raises(ValueError, match="available_slots must be in"):
            create_valid_path_option(available_slots=-0.1)

    def test_available_slots_above_one_raises(self) -> None:
        """available_slots above 1 should raise ValueError."""
        with pytest.raises(ValueError, match="available_slots must be in"):
            create_valid_path_option(available_slots=1.5)

    def test_boundary_values_are_valid(self) -> None:
        """Boundary values (0 and 1) should be valid."""
        # Should not raise
        opt1 = create_valid_path_option(congestion=0.0, available_slots=0.0)
        opt2 = create_valid_path_option(congestion=1.0, available_slots=1.0)
        opt3 = create_valid_path_option(path_index=0, weight_km=0.0)

        assert opt1.congestion == 0.0
        assert opt2.congestion == 1.0
        assert opt3.weight_km == 0.0


class TestComputeActionMask:
    """Tests for compute_action_mask function."""

    def test_basic_action_mask(self) -> None:
        """Action mask should reflect feasibility."""
        options = [
            create_valid_path_option(path_index=0, is_feasible=True),
            create_valid_path_option(path_index=1, is_feasible=False),
            create_valid_path_option(path_index=2, is_feasible=True),
        ]

        mask = compute_action_mask(options, k_paths=3)

        assert mask.tolist() == [True, False, True]
        assert mask.dtype == np.bool_

    def test_empty_options(self) -> None:
        """Empty options should produce all-False mask."""
        mask = compute_action_mask([], k_paths=3)

        assert mask.tolist() == [False, False, False]

    def test_partial_options(self) -> None:
        """Fewer options than k_paths should leave remaining as False."""
        options = [
            create_valid_path_option(path_index=0, is_feasible=True),
        ]

        mask = compute_action_mask(options, k_paths=5)

        assert mask.tolist() == [True, False, False, False, False]

    def test_out_of_range_path_index_ignored(self) -> None:
        """Options with path_index >= k_paths should be ignored."""
        options = [
            create_valid_path_option(path_index=0, is_feasible=True),
            create_valid_path_option(path_index=5, is_feasible=True),  # Out of range
        ]

        mask = compute_action_mask(options, k_paths=3)

        assert mask.tolist() == [True, False, False]

    def test_all_infeasible(self) -> None:
        """All infeasible options should produce all-False mask."""
        options = [
            create_valid_path_option(path_index=0, is_feasible=False),
            create_valid_path_option(path_index=1, is_feasible=False),
            create_valid_path_option(path_index=2, is_feasible=False),
        ]

        mask = compute_action_mask(options, k_paths=3)

        assert mask.tolist() == [False, False, False]

    def test_all_feasible(self) -> None:
        """All feasible options should produce all-True mask."""
        options = [
            create_valid_path_option(path_index=0, is_feasible=True),
            create_valid_path_option(path_index=1, is_feasible=True),
            create_valid_path_option(path_index=2, is_feasible=True),
        ]

        mask = compute_action_mask(options, k_paths=3)

        assert mask.tolist() == [True, True, True]

    def test_mask_shape(self) -> None:
        """Mask shape should match k_paths."""
        options = [create_valid_path_option(path_index=0, is_feasible=True)]

        mask = compute_action_mask(options, k_paths=10)

        assert mask.shape == (10,)


class TestPathOptionProtection:
    """Tests for PathOption protection fields (P5.1)."""

    def test_unprotected_default_values(self) -> None:
        """Unprotected PathOption has None backup fields."""
        opt = create_valid_path_option()

        assert opt.backup_path is None
        assert opt.backup_feasible is None
        assert opt.backup_weight_km is None
        assert opt.backup_modulation is None
        assert opt.is_protected is False

    def test_unprotected_both_paths_feasible(self) -> None:
        """Unprotected path returns is_feasible for both_paths_feasible."""
        opt_feasible = create_valid_path_option(is_feasible=True)
        opt_infeasible = create_valid_path_option(is_feasible=False)

        assert opt_feasible.both_paths_feasible is True
        assert opt_infeasible.both_paths_feasible is False

    def test_protected_path_creation(self) -> None:
        """Protected PathOption can be created with backup fields."""
        opt = PathOption.from_protected_route(
            path_index=0,
            primary_path=["A", "B", "C"],
            backup_path=["A", "D", "E", "C"],
            primary_weight=100.0,
            backup_weight=150.0,
            primary_feasible=True,
            backup_feasible=True,
            primary_modulation="QPSK",
            backup_modulation="8-QAM",
            slots_needed=4,
            congestion=0.3,
        )

        assert opt.is_protected is True
        assert opt.path == ("A", "B", "C")
        assert opt.backup_path == ("A", "D", "E", "C")
        assert opt.weight_km == 100.0
        assert opt.backup_weight_km == 150.0
        assert opt.is_feasible is True
        assert opt.backup_feasible is True
        assert opt.modulation == "QPSK"
        assert opt.backup_modulation == "8-QAM"

    def test_protected_both_paths_feasible_true(self) -> None:
        """both_paths_feasible True when both paths have spectrum."""
        opt = PathOption.from_protected_route(
            path_index=0,
            primary_path=["A", "B"],
            backup_path=["A", "C", "B"],
            primary_weight=50.0,
            backup_weight=80.0,
            primary_feasible=True,
            backup_feasible=True,
            primary_modulation="QPSK",
            backup_modulation="QPSK",
            slots_needed=4,
            congestion=0.3,
        )

        assert opt.both_paths_feasible is True

    def test_protected_both_paths_feasible_false_backup(self) -> None:
        """both_paths_feasible False when backup infeasible."""
        opt = PathOption.from_protected_route(
            path_index=0,
            primary_path=["A", "B"],
            backup_path=["A", "C", "B"],
            primary_weight=50.0,
            backup_weight=80.0,
            primary_feasible=True,
            backup_feasible=False,
            primary_modulation="QPSK",
            backup_modulation=None,
            slots_needed=4,
            congestion=0.3,
        )

        assert opt.both_paths_feasible is False

    def test_protected_both_paths_feasible_false_primary(self) -> None:
        """both_paths_feasible False when primary infeasible."""
        opt = PathOption.from_protected_route(
            path_index=0,
            primary_path=["A", "B"],
            backup_path=["A", "C", "B"],
            primary_weight=50.0,
            backup_weight=80.0,
            primary_feasible=False,
            backup_feasible=True,
            primary_modulation=None,
            backup_modulation="QPSK",
            slots_needed=4,
            congestion=0.3,
        )

        assert opt.both_paths_feasible is False

    def test_total_weight_unprotected(self) -> None:
        """total_weight_km returns primary weight for unprotected."""
        opt = create_valid_path_option(weight_km=100.0)

        assert opt.total_weight_km == 100.0

    def test_total_weight_protected(self) -> None:
        """total_weight_km sums both paths for protected."""
        opt = PathOption.from_protected_route(
            path_index=0,
            primary_path=["A", "B"],
            backup_path=["A", "C", "B"],
            primary_weight=50.0,
            backup_weight=80.0,
            primary_feasible=True,
            backup_feasible=True,
            primary_modulation="QPSK",
            backup_modulation="QPSK",
            slots_needed=4,
            congestion=0.3,
        )

        assert opt.total_weight_km == 130.0

    def test_backup_hop_count_unprotected(self) -> None:
        """backup_hop_count returns None for unprotected."""
        opt = create_valid_path_option()

        assert opt.backup_hop_count is None

    def test_backup_hop_count_protected(self) -> None:
        """backup_hop_count returns correct value for protected."""
        opt = PathOption.from_protected_route(
            path_index=0,
            primary_path=["A", "B"],
            backup_path=["A", "C", "D", "B"],  # 3 hops
            primary_weight=50.0,
            backup_weight=80.0,
            primary_feasible=True,
            backup_feasible=True,
            primary_modulation="QPSK",
            backup_modulation="QPSK",
            slots_needed=4,
            congestion=0.3,
        )

        assert opt.backup_hop_count == 3

    def test_validation_is_protected_requires_backup_path(self) -> None:
        """is_protected=True requires backup_path."""
        with pytest.raises(ValueError, match="is_protected=True requires backup_path"):
            PathOption(
                path_index=0,
                path=("A", "B"),
                weight_km=50.0,
                num_hops=1,
                modulation="QPSK",
                slots_needed=4,
                is_feasible=True,
                congestion=0.5,
                available_slots=0.5,
                is_protected=True,
                backup_path=None,  # Missing!
                backup_feasible=True,
            )

    def test_validation_is_protected_requires_backup_feasible(self) -> None:
        """is_protected=True requires backup_feasible."""
        with pytest.raises(
            ValueError, match="is_protected=True requires backup_feasible"
        ):
            PathOption(
                path_index=0,
                path=("A", "B"),
                weight_km=50.0,
                num_hops=1,
                modulation="QPSK",
                slots_needed=4,
                is_feasible=True,
                congestion=0.5,
                available_slots=0.5,
                is_protected=True,
                backup_path=("A", "C", "B"),
                backup_feasible=None,  # Missing!
            )

    def test_from_unprotected_route_factory(self) -> None:
        """from_unprotected_route factory works correctly."""
        opt = PathOption.from_unprotected_route(
            path_index=1,
            path=["A", "B", "C"],
            weight_km=120.0,
            is_feasible=True,
            modulation="16-QAM",
            slots_needed=8,
            congestion=0.4,
        )

        assert opt.path_index == 1
        assert opt.path == ("A", "B", "C")
        assert opt.weight_km == 120.0
        assert opt.is_feasible is True
        assert opt.modulation == "16-QAM"
        assert opt.slots_needed == 8
        assert opt.congestion == 0.4
        assert opt.is_protected is False
        assert opt.backup_path is None

    def test_protected_path_frozen(self) -> None:
        """Protected PathOption should also be frozen."""
        opt = PathOption.from_protected_route(
            path_index=0,
            primary_path=["A", "B"],
            backup_path=["A", "C", "B"],
            primary_weight=50.0,
            backup_weight=80.0,
            primary_feasible=True,
            backup_feasible=True,
            primary_modulation="QPSK",
            backup_modulation="QPSK",
            slots_needed=4,
            congestion=0.3,
        )

        with pytest.raises(FrozenInstanceError):
            opt.is_protected = False  # type: ignore[misc]
