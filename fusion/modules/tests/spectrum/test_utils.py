"""Unit tests for the SpectrumHelpers class."""

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from fusion.core.properties import SpectrumProps
from fusion.modules.spectrum.utils import SpectrumHelpers


@pytest.fixture
def engine_props() -> dict[str, Any]:
    """Provide engine properties for tests."""
    return {"allocation_method": "first_fit", "guard_slots": 1}


@pytest.fixture
def sdn_props() -> MagicMock:
    """Provide SDN properties for tests."""
    sdn = MagicMock()
    # Initialize cores_matrix with 2 cores and 10 slots each
    sdn.network_spectrum_dict = {
        (1, 2): {
            "cores_matrix": {
                "c": [np.zeros(10), np.zeros(10)],
                "l": [np.zeros(10), np.zeros(10)],
                "s": [np.zeros(10), np.zeros(10)],
            }
        },
        (2, 1): {
            "cores_matrix": {
                "c": [np.zeros(10), np.zeros(10)],
                "l": [np.zeros(10), np.zeros(10)],
                "s": [np.zeros(10), np.zeros(10)],
            }
        },
        (2, 3): {
            "cores_matrix": {
                "c": [np.zeros(10), np.zeros(10)],
                "l": [np.zeros(10), np.zeros(10)],
                "s": [np.zeros(10), np.zeros(10)],
            }
        },
        (3, 2): {
            "cores_matrix": {
                "c": [np.zeros(10), np.zeros(10)],
                "l": [np.zeros(10), np.zeros(10)],
                "s": [np.zeros(10), np.zeros(10)],
            }
        },
    }
    return sdn


@pytest.fixture
def spectrum_props() -> SpectrumProps:
    """Provide spectrum properties for tests."""
    props = SpectrumProps()
    props.path_list = [1, 2, 3]
    props.slots_needed = 2
    props.is_free = False
    props.forced_core = None
    props.forced_band = None
    props.forced_index = None
    return props


@pytest.fixture
def spectrum_helpers(engine_props: dict[str, Any], sdn_props: MagicMock, spectrum_props: SpectrumProps) -> SpectrumHelpers:
    """Provide SpectrumHelpers instance for tests."""
    return SpectrumHelpers(engine_props, sdn_props, spectrum_props)


class TestCheckFreeSpectrum:
    """Tests for _check_free_spectrum method."""

    def test_check_free_spectrum_with_free_slots_returns_true(self, spectrum_helpers: SpectrumHelpers) -> None:
        """Test that free spectrum slots are correctly identified."""
        # Arrange
        spectrum_helpers.current_band = "c"
        spectrum_helpers.core_number = 0
        spectrum_helpers.start_index = 0
        spectrum_helpers.end_index = 5

        # Act
        result = spectrum_helpers._check_free_spectrum((1, 2), (2, 1))

        # Assert
        assert result is True

    def test_check_free_spectrum_with_single_slot_no_guard_returns_true(
        self, spectrum_helpers: SpectrumHelpers, spectrum_props: SpectrumProps
    ) -> None:
        """Test single slot allocation with no guard band."""
        # Arrange
        spectrum_helpers.current_band = "c"
        spectrum_helpers.core_number = 0
        spectrum_helpers.start_index = 0
        spectrum_helpers.end_index = 0
        spectrum_props.slots_needed = 1
        spectrum_helpers.engine_props["guard_slots"] = 0

        # Act
        result = spectrum_helpers._check_free_spectrum((1, 2), (2, 1))

        # Assert
        assert result is True

    def test_check_free_spectrum_with_occupied_slots_returns_false(self, spectrum_helpers: SpectrumHelpers, sdn_props: MagicMock) -> None:
        """Test that occupied spectrum slots return false."""
        # Arrange
        spectrum_helpers.current_band = "c"
        spectrum_helpers.core_number = 0
        spectrum_helpers.start_index = 0
        spectrum_helpers.end_index = 5
        # Occupy some slots
        sdn_props.network_spectrum_dict[(1, 2)]["cores_matrix"]["c"][0][2] = 1

        # Act
        result = spectrum_helpers._check_free_spectrum((1, 2), (2, 1))

        # Assert
        assert result is False

    def test_check_free_spectrum_with_empty_spectrum_raises_error(
        self, spectrum_helpers: SpectrumHelpers, spectrum_props: SpectrumProps
    ) -> None:
        """Test that empty spectrum set raises ValueError."""
        # Arrange
        spectrum_helpers.current_band = "c"
        spectrum_helpers.core_number = 0
        spectrum_helpers.start_index = 15  # Beyond array bounds
        spectrum_helpers.end_index = 20
        spectrum_props.slots_needed = 5

        # Act & Assert
        with pytest.raises(ValueError, match="Spectrum set cannot be empty"):
            spectrum_helpers._check_free_spectrum((1, 2), (2, 1))


class TestCheckOtherLinks:
    """Tests for check_other_links method."""

    def test_check_other_links_with_all_free_sets_is_free_true(
        self, spectrum_helpers: SpectrumHelpers, spectrum_props: SpectrumProps
    ) -> None:
        """Test that all free links set is_free to True."""
        # Arrange
        spectrum_helpers.current_band = "c"
        spectrum_helpers.core_number = 0
        spectrum_helpers.start_index = 0
        spectrum_helpers.end_index = 5

        # Act
        spectrum_helpers.check_other_links()

        # Assert
        assert spectrum_props.is_free is True

    def test_check_other_links_with_occupied_link_sets_is_free_false(
        self,
        spectrum_helpers: SpectrumHelpers,
        spectrum_props: SpectrumProps,
        sdn_props: MagicMock,
    ) -> None:
        """Test that occupied link sets is_free to False."""
        # Arrange
        spectrum_helpers.current_band = "c"
        spectrum_helpers.core_number = 0
        spectrum_helpers.start_index = 0
        spectrum_helpers.end_index = 5
        # Occupy slot on second link
        sdn_props.network_spectrum_dict[(2, 3)]["cores_matrix"]["c"][0][3] = 1

        # Act
        spectrum_helpers.check_other_links()

        # Assert
        assert spectrum_props.is_free is False

    def test_check_other_links_with_no_guard_band_works_correctly(
        self, spectrum_helpers: SpectrumHelpers, spectrum_props: SpectrumProps
    ) -> None:
        """Test check_other_links with zero guard band."""
        # Arrange
        spectrum_helpers.current_band = "c"
        spectrum_helpers.core_number = 0
        spectrum_helpers.start_index = 0
        spectrum_helpers.end_index = 5
        spectrum_helpers.engine_props["guard_slots"] = 0

        # Act
        spectrum_helpers.check_other_links()

        # Assert
        assert spectrum_props.is_free is True

    def test_check_other_links_with_l_band_only_sets_is_free_true(
        self,
        spectrum_helpers: SpectrumHelpers,
        spectrum_props: SpectrumProps,
        sdn_props: MagicMock,
    ) -> None:
        """Test that L-band allocation works when C-band is occupied."""
        # Arrange
        # Occupy C-band
        sdn_props.network_spectrum_dict[(1, 2)]["cores_matrix"]["c"][0][:] = 1
        sdn_props.network_spectrum_dict[(2, 3)]["cores_matrix"]["c"][0][:] = 1

        spectrum_helpers.current_band = "l"
        spectrum_helpers.core_number = 0
        spectrum_helpers.start_index = 0
        spectrum_helpers.end_index = 5
        spectrum_props.slots_needed = 6
        spectrum_helpers.engine_props["guard_slots"] = 0

        # Act
        spectrum_helpers.check_other_links()

        # Assert
        assert spectrum_props.is_free is True


class TestUpdateSpecProps:
    """Tests for _update_spec_props method."""

    def test_update_spec_props_with_first_fit_sets_correct_slots(
        self, spectrum_helpers: SpectrumHelpers, spectrum_props: SpectrumProps
    ) -> None:
        """Test spectrum properties update for first fit."""
        # Arrange
        spectrum_helpers.current_band = "c"
        spectrum_helpers.core_number = 0
        spectrum_helpers.start_index = 0
        spectrum_helpers.end_index = 5

        # Act
        result = spectrum_helpers._update_spec_props()

        # Assert
        assert result.start_slot == 0
        assert result.end_slot == 6
        assert result.core_number == 0
        assert result.current_band == "c"

    def test_update_spec_props_with_last_fit_sets_correct_slots(
        self, spectrum_helpers: SpectrumHelpers, spectrum_props: SpectrumProps
    ) -> None:
        """Test spectrum properties update for last fit."""
        # Arrange
        spectrum_helpers.engine_props["allocation_method"] = "last_fit"
        spectrum_helpers.current_band = "c"
        spectrum_helpers.core_number = 1
        spectrum_helpers.start_index = 3
        spectrum_helpers.end_index = 8

        # Act
        result = spectrum_helpers._update_spec_props()

        # Assert
        assert result.start_slot == 8
        assert result.end_slot == 4
        assert result.core_number == 1

    def test_update_spec_props_with_forced_core_uses_forced_value(
        self, spectrum_helpers: SpectrumHelpers, spectrum_props: SpectrumProps
    ) -> None:
        """Test that forced core is used when set."""
        # Arrange
        spectrum_props.forced_core = 5
        spectrum_helpers.current_band = "c"
        spectrum_helpers.core_number = 0
        spectrum_helpers.start_index = 0
        spectrum_helpers.end_index = 5

        # Act
        result = spectrum_helpers._update_spec_props()

        # Assert
        assert result.core_number == 5

    def test_update_spec_props_with_forced_band_uses_forced_value(
        self, spectrum_helpers: SpectrumHelpers, spectrum_props: SpectrumProps
    ) -> None:
        """Test that forced band is used when set."""
        # Arrange
        spectrum_props.forced_band = "l"
        spectrum_helpers.current_band = "c"
        spectrum_helpers.core_number = 0
        spectrum_helpers.start_index = 0
        spectrum_helpers.end_index = 5

        # Act
        result = spectrum_helpers._update_spec_props()

        # Assert
        assert result.current_band == "l"


class TestCheckSuperChannels:
    """Tests for check_super_channels method."""

    def test_check_super_channels_with_valid_allocation_returns_true(self, spectrum_helpers: SpectrumHelpers) -> None:
        """Test that valid super-channel allocation succeeds."""
        # Arrange
        open_slots_matrix = [[0, 1, 2, 3, 4], [5, 6, 7, 8]]
        spectrum_helpers.current_band = "c"
        spectrum_helpers.core_number = 0
        spectrum_helpers.spectrum_props.slots_needed = 2

        # Act
        result = spectrum_helpers.check_super_channels(open_slots_matrix, flag="")

        # Assert
        assert result is True

    def test_check_super_channels_with_insufficient_slots_returns_false(self, spectrum_helpers: SpectrumHelpers) -> None:
        """Test that insufficient slots return False."""
        # Arrange
        open_slots_matrix = [[0, 1, 2, 3, 4], [5, 6, 7, 8]]
        spectrum_helpers.current_band = "c"
        spectrum_helpers.core_number = 0
        spectrum_helpers.spectrum_props.slots_needed = 10  # Too many

        # Act
        result = spectrum_helpers.check_super_channels(open_slots_matrix, flag="")

        # Assert
        assert result is False

    def test_check_super_channels_with_forced_index_uses_forced_value(
        self, spectrum_helpers: SpectrumHelpers, spectrum_props: SpectrumProps
    ) -> None:
        """Test that forced index is respected."""
        # Arrange
        open_slots_matrix = [[0, 1, 2, 3, 4], [5, 6, 7, 8]]
        spectrum_helpers.current_band = "c"
        spectrum_helpers.core_number = 0
        spectrum_props.forced_index = 5
        spectrum_props.slots_needed = 2

        # Act
        result = spectrum_helpers.check_super_channels(open_slots_matrix, flag="forced_index")

        # Assert
        assert result is True
        assert spectrum_helpers.start_index == 5

    def test_check_super_channels_with_last_fit_allocation(self, spectrum_helpers: SpectrumHelpers) -> None:
        """Test super-channel allocation with last fit strategy."""
        # Arrange
        open_slots_matrix = [[0, 1, 2, 3, 4], [5, 6, 7, 8]]
        spectrum_helpers.engine_props["allocation_method"] = "last_fit"
        spectrum_helpers.current_band = "c"
        spectrum_helpers.core_number = 0
        spectrum_helpers.spectrum_props.slots_needed = 2

        # Act
        result = spectrum_helpers.check_super_channels(open_slots_matrix, flag="")

        # Assert
        assert result is False  # Last fit has different logic

    def test_check_super_channels_with_no_guard_band(self, spectrum_helpers: SpectrumHelpers) -> None:
        """Test super-channel allocation with zero guard band."""
        # Arrange
        open_slots_matrix = [[0, 1, 2, 3, 4], [5, 6, 7, 8]]
        spectrum_helpers.current_band = "c"
        spectrum_helpers.core_number = 0
        spectrum_helpers.spectrum_props.slots_needed = 2
        spectrum_helpers.engine_props["guard_slots"] = 0

        # Act
        result = spectrum_helpers.check_super_channels(open_slots_matrix, flag="")

        # Assert
        assert result is True
