"""Tests for PathOption dataclass.

Phase: P4.1 - RLSimulationAdapter Scaffolding
Chunk: 1 - PathOption dataclass
"""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from fusion.modules.rl.adapter import (
    PathOption,
    compute_action_mask,
)


def create_valid_path_option(**kwargs) -> PathOption:
    """Helper to create a valid PathOption with defaults."""
    defaults = {
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
