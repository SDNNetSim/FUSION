"""
Unit tests for fusion.modules.snr.utils module.

This module tests the utility functions for SNR calculation including:
- File loading based on core configuration
- Slot index computation
- SNR response validation
"""

from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest

from fusion.modules.snr.utils import (
    compute_response,
    get_loaded_files,
    get_slot_index,
)


class TestGetLoadedFiles:
    """Tests for get_loaded_files function."""

    def test_get_loaded_files_with_multi_fiber_returns_correct_files(self) -> None:
        """Test that multi-fiber configuration (core_num=0) loads correct files."""
        # Arrange
        core_num = 0
        cores_per_link = 7
        network = "test_network"
        file_mapping_dict = {"test_network": {"multi_fiber": {"mf": "mf_multi.npy", "gsnr": "gsnr_multi.npy"}}}

        mf_data = np.array([1, 2, 3])
        gsnr_data = np.array([4, 5, 6])

        # Act & Assert
        with patch("fusion.modules.snr.utils.np.load") as mock_load:
            mock_load.side_effect = [mf_data, gsnr_data]
            result_mf, result_gsnr = get_loaded_files(core_num, cores_per_link, file_mapping_dict, network)

            assert np.array_equal(result_mf, mf_data)
            assert np.array_equal(result_gsnr, gsnr_data)
            assert mock_load.call_count == 2

    def test_get_loaded_files_with_specific_core_returns_correct_files(self) -> None:
        """Test that specific core configuration loads correct files."""
        # Arrange
        core_num = 3
        cores_per_link = 7
        network = "test_network"
        file_mapping_dict = {"test_network": {(3, 7): {"mf": "mf_core3.npy", "gsnr": "gsnr_core3.npy"}}}

        mf_data = np.array([10, 20, 30])
        gsnr_data = np.array([40, 50, 60])

        # Act & Assert
        with patch("fusion.modules.snr.utils.np.load") as mock_load:
            mock_load.side_effect = [mf_data, gsnr_data]
            result_mf, result_gsnr = get_loaded_files(core_num, cores_per_link, file_mapping_dict, network)

            assert np.array_equal(result_mf, mf_data)
            assert np.array_equal(result_gsnr, gsnr_data)

    def test_get_loaded_files_with_invalid_key_raises_value_error(self) -> None:
        """Test that invalid core/cores_per_link combination raises ValueError."""
        # Arrange
        core_num = 99
        cores_per_link = 7
        network = "test_network"
        file_mapping_dict = {"test_network": {(3, 7): {"mf": "mf_core3.npy", "gsnr": "gsnr_core3.npy"}}}

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            get_loaded_files(core_num, cores_per_link, file_mapping_dict, network)

        assert "No matching file found" in str(exc_info.value)
        assert "core_num=99" in str(exc_info.value)
        assert "cores_per_link=7" in str(exc_info.value)

    def test_get_loaded_files_constructs_correct_file_paths(self) -> None:
        """Test that file paths are constructed correctly."""
        # Arrange
        core_num = 0
        cores_per_link = 7
        network = "nsfnet"
        file_mapping_dict = {"nsfnet": {"multi_fiber": {"mf": "mf_test.npy", "gsnr": "gsnr_test.npy"}}}

        # Act & Assert
        with patch("fusion.modules.snr.utils.np.load") as mock_load:
            mock_load.return_value = np.array([1, 2, 3])
            get_loaded_files(core_num, cores_per_link, file_mapping_dict, network)

            # Verify correct paths were used
            call_args = [call[0][0] for call in mock_load.call_args_list]
            assert "data/pre_calc/nsfnet/modulations/mf_test.npy" in call_args[0]
            assert "data/pre_calc/nsfnet/snr/gsnr_test.npy" in call_args[1]


class TestGetSlotIndex:
    """Tests for get_slot_index function."""

    def test_get_slot_index_with_l_band_returns_correct_index(self) -> None:
        """Test slot index calculation for L-band."""
        # Arrange
        current_band = "l"
        start_slot = 50
        engine_props = {"l_band": 320, "c_band": 320}

        # Act
        result = get_slot_index(current_band, start_slot, engine_props)

        # Assert
        assert result == 50  # L-band offset is 0

    def test_get_slot_index_with_c_band_returns_correct_index(self) -> None:
        """Test slot index calculation for C-band."""
        # Arrange
        current_band = "c"
        start_slot = 100
        engine_props = {"l_band": 320, "c_band": 320}

        # Act
        result = get_slot_index(current_band, start_slot, engine_props)

        # Assert
        assert result == 420  # 320 (L-band offset) + 100

    def test_get_slot_index_with_s_band_returns_correct_index(self) -> None:
        """Test slot index calculation for S-band."""
        # Arrange
        current_band = "s"
        start_slot = 75
        engine_props = {"l_band": 320, "c_band": 320}

        # Act
        result = get_slot_index(current_band, start_slot, engine_props)

        # Assert
        assert result == 715  # 320 (L) + 320 (C) + 75

    def test_get_slot_index_with_invalid_band_raises_value_error(self) -> None:
        """Test that invalid band raises ValueError."""
        # Arrange
        current_band = "x"
        start_slot = 50
        engine_props = {"l_band": 320, "c_band": 320}

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            get_slot_index(current_band, start_slot, engine_props)

        assert "Unexpected band: x" in str(exc_info.value)

    @pytest.mark.parametrize(
        "band,start,l_band,c_band,expected",
        [
            ("l", 0, 320, 320, 0),
            ("l", 100, 320, 320, 100),
            ("c", 0, 320, 320, 320),
            ("c", 200, 320, 320, 520),
            ("s", 0, 320, 320, 640),
            ("s", 150, 320, 320, 790),
        ],
    )
    def test_get_slot_index_with_various_inputs_returns_correct_values(
        self, band: str, start: int, l_band: int, c_band: int, expected: int
    ) -> None:
        """Test slot index calculation with parametrized inputs."""
        # Arrange
        engine_props = {"l_band": l_band, "c_band": c_band}

        # Act
        result = get_slot_index(band, start, engine_props)

        # Assert
        assert result == expected

    def test_get_slot_index_returns_integer_type(self) -> None:
        """Test that get_slot_index returns integer type."""
        # Arrange
        current_band = "c"
        start_slot = 100
        engine_props = {"l_band": 320, "c_band": 320}

        # Act
        result = get_slot_index(current_band, start_slot, engine_props)

        # Assert
        assert isinstance(result, int)


class TestComputeResponse:
    """Tests for compute_response function."""

    def test_compute_response_with_valid_conditions_returns_true(self) -> None:
        """Test compute_response returns True when all conditions are met."""
        # Arrange
        mod_format = 5
        snr_props = Mock()
        snr_props.modulation_format_mapping_dict = {5: "QPSK"}
        snr_props.bandwidth_mapping_dict = {"QPSK": 50}

        spectrum_props = Mock()
        spectrum_props.modulation = "QPSK"

        sdn_props = Mock()
        sdn_props.bandwidth = 25

        # Act
        result = compute_response(mod_format, snr_props, spectrum_props, sdn_props)

        # Assert
        assert result is True

    def test_compute_response_with_zero_mod_format_returns_false(self) -> None:
        """Test compute_response returns False when mod_format is 0."""
        # Arrange
        mod_format = 0
        snr_props = Mock()
        snr_props.modulation_format_mapping_dict = {0: "NONE"}
        snr_props.bandwidth_mapping_dict = {"NONE": 0}

        spectrum_props = Mock()
        spectrum_props.modulation = "NONE"

        sdn_props = Mock()
        sdn_props.bandwidth = 0

        # Act
        result = compute_response(mod_format, snr_props, spectrum_props, sdn_props)

        # Assert
        assert result is False

    def test_compute_response_with_mismatched_modulation_returns_false(self) -> None:
        """Test compute_response returns False when modulation formats don't match."""
        # Arrange
        mod_format = 5
        snr_props = Mock()
        snr_props.modulation_format_mapping_dict = {5: "QPSK"}
        snr_props.bandwidth_mapping_dict = {"QPSK": 50, "16QAM": 100}

        spectrum_props = Mock()
        spectrum_props.modulation = "16QAM"  # Mismatch

        sdn_props = Mock()
        sdn_props.bandwidth = 25

        # Act
        result = compute_response(mod_format, snr_props, spectrum_props, sdn_props)

        # Assert
        assert result is False

    def test_compute_response_with_insufficient_bandwidth_returns_false(self) -> None:
        """Test compute_response returns False when bandwidth requirements not met."""
        # Arrange
        mod_format = 5
        snr_props = Mock()
        snr_props.modulation_format_mapping_dict = {5: "QPSK"}
        snr_props.bandwidth_mapping_dict = {"QPSK": 50}

        spectrum_props = Mock()
        spectrum_props.modulation = "QPSK"

        sdn_props = Mock()
        sdn_props.bandwidth = 100  # Exceeds available bandwidth

        # Act
        result = compute_response(mod_format, snr_props, spectrum_props, sdn_props)

        # Assert
        assert result is False

    @pytest.mark.parametrize(
        "mod_format,mod_mapping,bw_mapping,spectrum_mod,sdn_bw,expected",
        [
            (1, {1: "BPSK"}, {"BPSK": 25}, "BPSK", 12, True),
            (2, {2: "QPSK"}, {"QPSK": 50}, "QPSK", 50, True),
            (0, {0: "NONE"}, {"NONE": 0}, "NONE", 0, False),  # Zero mod_format
            (
                3,
                {3: "16QAM"},
                {"16QAM": 100, "QPSK": 50},
                "QPSK",
                50,
                False,
            ),  # Mismatched modulation
            (
                4,
                {4: "64QAM"},
                {"64QAM": 75},
                "64QAM",
                100,
                False,
            ),  # Insufficient BW
        ],
    )
    def test_compute_response_with_various_conditions(
        self,
        mod_format: Any,
        mod_mapping: dict[Any, str],
        bw_mapping: dict[str, int],
        spectrum_mod: str,
        sdn_bw: int,
        expected: bool,
    ) -> None:
        """Test compute_response with parametrized conditions."""
        # Arrange
        snr_props = Mock()
        snr_props.modulation_format_mapping_dict = mod_mapping
        snr_props.bandwidth_mapping_dict = bw_mapping

        spectrum_props = Mock()
        spectrum_props.modulation = spectrum_mod

        sdn_props = Mock()
        sdn_props.bandwidth = sdn_bw

        # Act
        result = compute_response(mod_format, snr_props, spectrum_props, sdn_props)

        # Assert
        assert result is expected

    def test_compute_response_returns_boolean_type(self) -> None:
        """Test that compute_response returns boolean type."""
        # Arrange
        mod_format = 1
        snr_props = Mock()
        snr_props.modulation_format_mapping_dict = {1: "QPSK"}
        snr_props.bandwidth_mapping_dict = {"QPSK": 50}

        spectrum_props = Mock()
        spectrum_props.modulation = "QPSK"

        sdn_props = Mock()
        sdn_props.bandwidth = 25

        # Act
        result = compute_response(mod_format, snr_props, spectrum_props, sdn_props)

        # Assert
        assert isinstance(result, bool)

    def test_compute_response_with_string_bandwidth_converts_correctly(self) -> None:
        """Test compute_response handles string bandwidth conversion."""
        # Arrange
        mod_format = 5
        snr_props = Mock()
        snr_props.modulation_format_mapping_dict = {5: "QPSK"}
        snr_props.bandwidth_mapping_dict = {"QPSK": 50}

        spectrum_props = Mock()
        spectrum_props.modulation = "QPSK"

        sdn_props = Mock()
        sdn_props.bandwidth = "25"  # String bandwidth

        # Act
        result = compute_response(mod_format, snr_props, spectrum_props, sdn_props)

        # Assert
        assert result is True
