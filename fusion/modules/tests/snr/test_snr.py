"""
Unit tests for fusion.modules.snr.snr module.

This module tests the StandardSNRMeasurer class including:
- SNR calculation for paths and links
- Linear noise (ASE) calculations
- Nonlinear noise (SCI, XCI) calculations
- Cross-talk calculations for multi-core fibers
- SNR threshold and acceptability checks
"""

import math
from typing import Any
from unittest.mock import Mock, patch

import pytest

from fusion.modules.snr.snr import StandardSNRMeasurer


@pytest.fixture
def mock_engine_props() -> dict[str, Any]:
    """Provide mock engine properties."""
    return {
        "bw_per_slot": 12.5e9,
        "input_power": 1e-3,
        "fiber_attenuation": 0.2,
        "fiber_dispersion": 16.7,
        "nonlinear_coefficient": 1.3e-3,
        "bending_radius": 7.5e-3,
        "edfa_noise_figure": 4.5,
        "c_band": 320,
        "l_band": 320,
        "cores_per_link": 7,
        "xt_coefficient": -40,
    }


@pytest.fixture
def mock_sdn_props() -> Any:
    """Provide mock SDN properties."""
    sdn_props = Mock()
    sdn_props.network_spectrum_dict = {}
    sdn_props.topology = Mock()
    sdn_props.topology.has_edge = Mock(return_value=False)
    return sdn_props


@pytest.fixture
def mock_spectrum_props() -> Any:
    """Provide mock spectrum properties."""
    spectrum_props = Mock()
    spectrum_props.path_list = []
    spectrum_props.start_slot = 0
    spectrum_props.end_slot = 10
    spectrum_props.core_number = 0
    spectrum_props.current_band = "c"
    spectrum_props.modulation = "QPSK"
    spectrum_props.core_num = 0
    return spectrum_props


@pytest.fixture
def mock_route_props() -> Any:
    """Provide mock route properties."""
    return Mock()


@pytest.fixture
def snr_measurer(
    mock_engine_props: dict[str, Any],
    mock_sdn_props: Any,
    mock_spectrum_props: Any,
    mock_route_props: Any,
) -> StandardSNRMeasurer:
    """Provide StandardSNRMeasurer instance."""
    return StandardSNRMeasurer(mock_engine_props, mock_sdn_props, mock_spectrum_props, mock_route_props)


class TestStandardSNRMeasurerInitialization:
    """Tests for StandardSNRMeasurer initialization."""

    def test_initialization_sets_properties_correctly(
        self,
        mock_engine_props: dict[str, Any],
        mock_sdn_props: Any,
        mock_spectrum_props: Any,
        mock_route_props: Any,
    ) -> None:
        """Test that initialization sets all properties correctly."""
        # Act
        measurer = StandardSNRMeasurer(mock_engine_props, mock_sdn_props, mock_spectrum_props, mock_route_props)

        # Assert
        assert measurer.engine_props == mock_engine_props
        assert measurer.sdn_props == mock_sdn_props
        assert measurer.spectrum_props == mock_spectrum_props
        assert measurer.route_props == mock_route_props
        assert measurer._calculations_performed == 0
        assert measurer._total_snr_computed == 0.0

    def test_algorithm_name_returns_standard_snr(self, snr_measurer: StandardSNRMeasurer) -> None:
        """Test that algorithm_name property returns 'standard_snr'."""
        # Act
        name = snr_measurer.algorithm_name

        # Assert
        assert name == "standard_snr"

    def test_supports_multicore_returns_true(self, snr_measurer: StandardSNRMeasurer) -> None:
        """Test that supports_multicore property returns True."""
        # Act
        supports = snr_measurer.supports_multicore

        # Assert
        assert supports is True


class TestCalculateSNR:
    """Tests for calculate_snr method."""

    def test_calculate_snr_with_valid_path_returns_positive_value(self, snr_measurer: StandardSNRMeasurer) -> None:
        """Test calculate_snr with valid path returns positive SNR."""
        # Arrange
        path = [0, 1, 2]
        spectrum_info = {
            "start_slot": 0,
            "end_slot": 10,
            "core_number": 0,
            "band": "c",
        }

        # Act
        snr = snr_measurer.calculate_snr(path, spectrum_info)

        # Assert
        assert isinstance(snr, float)
        assert snr > 0 or math.isinf(snr)

    def test_calculate_snr_with_empty_path_raises_value_error(self, snr_measurer: StandardSNRMeasurer) -> None:
        """Test calculate_snr with empty path raises ValueError."""
        # Arrange
        path: list[Any] = []
        spectrum_info = {"start_slot": 0, "end_slot": 10}

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            snr_measurer.calculate_snr(path, spectrum_info)

        assert "Path cannot be empty" in str(exc_info.value)

    def test_calculate_snr_with_empty_spectrum_info_raises_value_error(self, snr_measurer: StandardSNRMeasurer) -> None:
        """Test calculate_snr with empty spectrum_info raises ValueError."""
        # Arrange
        path = [0, 1]
        spectrum_info: dict[str, Any] = {}

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            snr_measurer.calculate_snr(path, spectrum_info)

        assert "Spectrum info cannot be empty" in str(exc_info.value)

    def test_calculate_snr_without_start_slot_raises_value_error(self, snr_measurer: StandardSNRMeasurer) -> None:
        """Test calculate_snr without start_slot raises ValueError."""
        # Arrange
        path = [0, 1]
        spectrum_info = {"end_slot": 10}

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            snr_measurer.calculate_snr(path, spectrum_info)

        assert "start_slot" in str(exc_info.value)

    def test_calculate_snr_increments_calculation_counter(self, snr_measurer: StandardSNRMeasurer) -> None:
        """Test that calculate_snr increments the calculation counter."""
        # Arrange
        path = [0, 1]
        spectrum_info = {"start_slot": 0, "end_slot": 10}

        # Act
        snr_measurer.calculate_snr(path, spectrum_info)

        # Assert
        assert snr_measurer._calculations_performed == 1

    def test_calculate_snr_updates_total_snr_computed(self, snr_measurer: StandardSNRMeasurer) -> None:
        """Test that calculate_snr updates total SNR computed."""
        # Arrange
        path = [0, 1]
        spectrum_info = {"start_slot": 0, "end_slot": 10}

        # Act
        snr_measurer.calculate_snr(path, spectrum_info)

        # Assert
        assert snr_measurer._total_snr_computed > 0 or math.isinf(snr_measurer._total_snr_computed)

    def test_calculate_snr_sets_spectrum_properties(self, snr_measurer: StandardSNRMeasurer) -> None:
        """Test that calculate_snr sets spectrum properties."""
        # Arrange
        path = [0, 1, 2]
        spectrum_info = {
            "start_slot": 5,
            "end_slot": 15,
            "core_number": 2,
            "band": "l",
        }

        # Act
        snr_measurer.calculate_snr(path, spectrum_info)

        # Assert
        assert snr_measurer.spectrum_props.start_slot == 5  # type: ignore[attr-defined]
        assert snr_measurer.spectrum_props.end_slot == 15  # type: ignore[attr-defined]
        assert snr_measurer.spectrum_props.core_number == 2  # type: ignore[attr-defined]
        assert snr_measurer.spectrum_props.current_band == "l"  # type: ignore[attr-defined]


class TestCalculateLinkSNR:
    """Tests for calculate_link_snr method."""

    def test_calculate_link_snr_with_none_source_raises_value_error(self, snr_measurer: StandardSNRMeasurer) -> None:
        """Test calculate_link_snr with None source raises ValueError."""
        # Arrange
        spectrum_info = {"start_slot": 0, "end_slot": 10}

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            snr_measurer.calculate_link_snr(None, 1, spectrum_info)

        assert "Source and destination cannot be None" in str(exc_info.value)

    def test_calculate_link_snr_with_none_destination_raises_value_error(self, snr_measurer: StandardSNRMeasurer) -> None:
        """Test calculate_link_snr with None destination raises ValueError."""
        # Arrange
        spectrum_info = {"start_slot": 0, "end_slot": 10}

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            snr_measurer.calculate_link_snr(0, None, spectrum_info)

        assert "Source and destination cannot be None" in str(exc_info.value)

    def test_calculate_link_snr_without_network_spectrum_returns_zero(self, snr_measurer: StandardSNRMeasurer) -> None:
        """Test calculate_link_snr returns 0 when link not in network spectrum."""
        # Arrange
        spectrum_info = {"start_slot": 0, "end_slot": 10}

        # Act
        snr = snr_measurer.calculate_link_snr(0, 1, spectrum_info)

        # Assert
        assert snr == 0.0

    def test_calculate_link_snr_with_topology_uses_link_length(self, snr_measurer: StandardSNRMeasurer) -> None:
        """Test calculate_link_snr uses topology link length when available."""
        # Arrange
        snr_measurer.sdn_props.network_spectrum_dict = {(0, 1): {}}  # type: ignore[attr-defined]
        snr_measurer.sdn_props.topology.has_edge = Mock(return_value=True)  # type: ignore[attr-defined]
        snr_measurer.sdn_props.topology.__getitem__ = Mock(  # type: ignore[attr-defined]
            return_value={1: {"length": 200}}
        )
        spectrum_info = {"start_slot": 0, "end_slot": 10, "core_num": 0}
        snr_measurer.snr_props.bandwidth = 12.5e9
        snr_measurer.snr_props.center_psd = 1e-6
        snr_measurer.snr_props.center_frequency = 193.1e12

        # Act
        snr = snr_measurer.calculate_link_snr(0, 1, spectrum_info)

        # Assert
        assert isinstance(snr, float)
        assert snr > 0 or math.isinf(snr)

    def test_calculate_link_snr_returns_positive_snr(self, snr_measurer: StandardSNRMeasurer) -> None:
        """Test calculate_link_snr returns positive SNR value."""
        # Arrange
        snr_measurer.sdn_props.network_spectrum_dict = {(0, 1): {}}  # type: ignore[attr-defined]
        spectrum_info = {"start_slot": 0, "end_slot": 10, "core_num": 0}
        snr_measurer.snr_props.bandwidth = 12.5e9
        snr_measurer.snr_props.center_psd = 1e-6
        snr_measurer.snr_props.center_frequency = 193.1e12

        # Act
        snr = snr_measurer.calculate_link_snr(0, 1, spectrum_info)

        # Assert
        assert isinstance(snr, float)
        assert snr > 0 or math.isinf(snr)


class TestASENoiseCalculation:
    """Tests for _calculate_ase_noise method."""

    def test_calculate_ase_noise_with_short_link_returns_positive_value(self, snr_measurer: StandardSNRMeasurer) -> None:
        """Test ASE noise calculation for short link."""
        # Arrange
        link_length = 50.0  # km
        snr_measurer.snr_props.bandwidth = 12.5e9

        # Act
        ase_noise = snr_measurer._calculate_ase_noise(link_length)

        # Assert
        assert isinstance(ase_noise, float)
        assert ase_noise > 0

    def test_calculate_ase_noise_with_long_link_returns_higher_noise(self, snr_measurer: StandardSNRMeasurer) -> None:
        """Test that longer links produce more ASE noise."""
        # Arrange
        short_link = 50.0
        long_link = 500.0
        snr_measurer.snr_props.bandwidth = 12.5e9

        # Act
        ase_short = snr_measurer._calculate_ase_noise(short_link)
        ase_long = snr_measurer._calculate_ase_noise(long_link)

        # Assert
        assert ase_long > ase_short

    def test_calculate_ase_noise_scales_with_amplifiers(self, snr_measurer: StandardSNRMeasurer) -> None:
        """Test that ASE noise scales with number of amplifiers."""
        # Arrange
        link_80km = 80.0  # 1 amplifier
        link_160km = 160.0  # 2 amplifiers
        snr_measurer.snr_props.bandwidth = 12.5e9

        # Act
        ase_1_amp = snr_measurer._calculate_ase_noise(link_80km)
        ase_2_amp = snr_measurer._calculate_ase_noise(link_160km)

        # Assert
        assert ase_2_amp >= ase_1_amp

    @pytest.mark.parametrize("link_length", [10.0, 50.0, 100.0, 200.0, 500.0])
    def test_calculate_ase_noise_with_various_lengths(self, snr_measurer: StandardSNRMeasurer, link_length: float) -> None:
        """Test ASE noise calculation with various link lengths."""
        # Arrange
        snr_measurer.snr_props.bandwidth = 12.5e9

        # Act
        ase_noise = snr_measurer._calculate_ase_noise(link_length)

        # Assert
        assert isinstance(ase_noise, float)
        assert ase_noise > 0


class TestNonlinearNoiseCalculation:
    """Tests for nonlinear noise calculation methods."""

    def test_calculate_nonlinear_noise_returns_dict_with_components(self, snr_measurer: StandardSNRMeasurer) -> None:
        """Test calculate_nonlinear_noise returns dictionary with all components."""
        # Arrange
        path = [0, 1, 2]
        spectrum_info = {"start_slot": 0, "end_slot": 10, "core_number": 0, "band": "c"}

        # Act
        noise = snr_measurer.calculate_nonlinear_noise(path, spectrum_info)

        # Assert
        assert isinstance(noise, dict)
        assert "sci" in noise
        assert "xci" in noise
        assert "xpm" in noise
        assert "fwm" in noise

    def test_calculate_nonlinear_noise_all_values_non_negative(self, snr_measurer: StandardSNRMeasurer) -> None:
        """Test that all nonlinear noise components are non-negative."""
        # Arrange
        path = [0, 1]
        spectrum_info = {"start_slot": 0, "end_slot": 10, "core_number": 0, "band": "c"}

        # Act
        noise = snr_measurer.calculate_nonlinear_noise(path, spectrum_info)

        # Assert
        assert noise["sci"] >= 0
        assert noise["xci"] >= 0
        assert noise["xpm"] >= 0
        assert noise["fwm"] >= 0

    def test_calculate_sci_psd_returns_positive_value(self, snr_measurer: StandardSNRMeasurer) -> None:
        """Test _calculate_sci_psd returns positive value."""
        # Arrange
        snr_measurer.snr_props.link_dictionary = {
            "dispersion": 16.7,
            "attenuation": 0.2,
        }
        snr_measurer.snr_props.center_psd = 1e-6
        snr_measurer.snr_props.bandwidth = 12.5e9

        # Act
        sci_psd = snr_measurer._calculate_sci_psd()

        # Assert
        assert isinstance(sci_psd, float)
        assert sci_psd >= 0

    def test_calculate_sci_psd_with_no_link_dictionary_returns_zero(self, snr_measurer: StandardSNRMeasurer) -> None:
        """Test _calculate_sci_psd returns 0 when link_dictionary is None."""
        # Arrange
        snr_measurer.snr_props.link_dictionary = None

        # Act
        sci_psd = snr_measurer._calculate_sci_psd()

        # Assert
        assert sci_psd == 0.0

    def test_calculate_xci_without_network_spectrum_returns_zero(self, snr_measurer: StandardSNRMeasurer) -> None:
        """Test _calculate_xci returns 0 when no network spectrum available."""
        # Act
        xci = snr_measurer._calculate_xci(0)

        # Assert
        assert xci == 0.0

    def test_update_link_xci_returns_float(self, snr_measurer: StandardSNRMeasurer) -> None:
        """Test _update_link_xci returns float value."""
        # Arrange
        req_id = 1.0
        curr_link = Mock()
        curr_link.__getitem__ = Mock(return_value=[1, 1, 1, 0, 0])
        slot_index = 0
        curr_xci = 0.0
        snr_measurer.spectrum_props.core_num = 0  # type: ignore[attr-defined]
        snr_measurer.snr_props.center_frequency = 193.1e12

        # Act
        with patch("fusion.modules.snr.snr.np") as mock_np:
            mock_np.where.return_value = ([0, 1, 2],)
            result = snr_measurer._update_link_xci(req_id, curr_link, slot_index, curr_xci)

        # Assert
        assert isinstance(result, float)


class TestCrosstalkCalculation:
    """Tests for crosstalk calculation methods."""

    def test_calculate_crosstalk_returns_non_negative_value(self, snr_measurer: StandardSNRMeasurer) -> None:
        """Test calculate_crosstalk returns non-negative value."""
        # Arrange
        path = [0, 1, 2]
        core_num = 1
        spectrum_info = {"start_slot": 0, "end_slot": 10, "core_num": 1}

        # Act
        xt = snr_measurer.calculate_crosstalk(path, core_num, spectrum_info)

        # Assert
        assert isinstance(xt, float)
        assert xt >= 0

    def test_calculate_crosstalk_noise_for_center_core(self, snr_measurer: StandardSNRMeasurer) -> None:
        """Test crosstalk calculation for center core (core_num=0)."""
        # Arrange
        spectrum_info = {"core_num": 0}

        # Act
        xt = snr_measurer._calculate_crosstalk_noise(0, 1, spectrum_info)

        # Assert
        assert isinstance(xt, float)
        assert xt >= 0

    def test_calculate_crosstalk_noise_for_outer_core(self, snr_measurer: StandardSNRMeasurer) -> None:
        """Test crosstalk calculation for outer core."""
        # Arrange
        spectrum_info = {"core_num": 3}

        # Act
        xt = snr_measurer._calculate_crosstalk_noise(0, 1, spectrum_info)

        # Assert
        assert isinstance(xt, float)
        assert xt >= 0

    def test_calculate_pxt_scales_with_adjacent_cores(self, snr_measurer: StandardSNRMeasurer) -> None:
        """Test that crosstalk power scales with number of adjacent cores."""
        # Arrange
        num_adjacent_1 = 1
        num_adjacent_6 = 6
        snr_measurer.snr_props.link_dictionary = {"bending_radius": 7.5e-3}

        # Act
        pxt_1 = snr_measurer._calculate_pxt(num_adjacent_1)
        pxt_6 = snr_measurer._calculate_pxt(num_adjacent_6)

        # Assert
        assert pxt_6 > pxt_1

    @pytest.mark.parametrize("num_adjacent", [0, 1, 3, 6])
    def test_calculate_pxt_with_various_adjacent_counts(self, snr_measurer: StandardSNRMeasurer, num_adjacent: int) -> None:
        """Test _calculate_pxt with various adjacent core counts."""
        # Arrange
        snr_measurer.snr_props.link_dictionary = {"bending_radius": 7.5e-3}

        # Act
        pxt = snr_measurer._calculate_pxt(num_adjacent)

        # Assert
        assert isinstance(pxt, float)
        assert pxt >= 0


class TestSNRThresholdMethods:
    """Tests for SNR threshold and acceptability methods."""

    def test_get_required_snr_threshold_returns_positive_value(self, snr_measurer: StandardSNRMeasurer) -> None:
        """Test get_required_snr_threshold returns positive value."""
        # Act
        threshold = snr_measurer.get_required_snr_threshold("QPSK", 100.0)

        # Assert
        assert isinstance(threshold, float)
        assert threshold > 0

    @pytest.mark.parametrize(
        "modulation,expected_base",
        [
            ("BPSK", 6.0),
            ("QPSK", 9.0),
            ("8QAM", 12.0),
            ("16QAM", 15.0),
            ("32QAM", 18.0),
            ("64QAM", 21.0),
        ],
    )
    def test_get_required_snr_threshold_for_modulations(
        self,
        snr_measurer: StandardSNRMeasurer,
        modulation: str,
        expected_base: float,
    ) -> None:
        """Test SNR thresholds for different modulation formats."""
        # Arrange
        reach = 0.0  # No reach penalty

        # Act
        threshold = snr_measurer.get_required_snr_threshold(modulation, reach)

        # Assert
        assert threshold == expected_base

    def test_get_required_snr_threshold_increases_with_reach(self, snr_measurer: StandardSNRMeasurer) -> None:
        """Test that SNR threshold increases with transmission reach."""
        # Arrange
        modulation = "QPSK"
        short_reach = 100.0
        long_reach = 1000.0

        # Act
        threshold_short = snr_measurer.get_required_snr_threshold(modulation, short_reach)
        threshold_long = snr_measurer.get_required_snr_threshold(modulation, long_reach)

        # Assert
        assert threshold_long > threshold_short

    def test_get_required_snr_threshold_with_unknown_modulation(self, snr_measurer: StandardSNRMeasurer) -> None:
        """Test SNR threshold with unknown modulation uses default."""
        # Act
        threshold = snr_measurer.get_required_snr_threshold("UNKNOWN", 100.0)

        # Assert
        assert isinstance(threshold, float)
        assert threshold > 0  # Should use default value

    def test_is_snr_acceptable_returns_true_when_snr_exceeds_requirement(self, snr_measurer: StandardSNRMeasurer) -> None:
        """Test is_snr_acceptable returns True when SNR exceeds requirement."""
        # Arrange
        calculated_snr = 20.0
        required_snr = 15.0

        # Act
        result = snr_measurer.is_snr_acceptable(calculated_snr, required_snr)

        # Assert
        assert result is True

    def test_is_snr_acceptable_returns_false_when_snr_below_requirement(self, snr_measurer: StandardSNRMeasurer) -> None:
        """Test is_snr_acceptable returns False when SNR below requirement."""
        # Arrange
        calculated_snr = 10.0
        required_snr = 15.0

        # Act
        result = snr_measurer.is_snr_acceptable(calculated_snr, required_snr)

        # Assert
        assert result is False

    def test_is_snr_acceptable_with_margin(self, snr_measurer: StandardSNRMeasurer) -> None:
        """Test is_snr_acceptable considers margin."""
        # Arrange
        calculated_snr = 15.0
        required_snr = 14.0
        margin = 2.0

        # Act
        result = snr_measurer.is_snr_acceptable(calculated_snr, required_snr, margin)

        # Assert
        assert result is False  # 15.0 < (14.0 + 2.0)

    def test_is_snr_acceptable_with_equal_values(self, snr_measurer: StandardSNRMeasurer) -> None:
        """Test is_snr_acceptable returns True when SNR equals requirement."""
        # Arrange
        calculated_snr = 15.0
        required_snr = 15.0

        # Act
        result = snr_measurer.is_snr_acceptable(calculated_snr, required_snr)

        # Assert
        assert result is True


class TestMetricsAndReset:
    """Tests for get_metrics and reset methods."""

    def test_get_metrics_returns_correct_structure(self, snr_measurer: StandardSNRMeasurer) -> None:
        """Test get_metrics returns dictionary with expected keys."""
        # Act
        metrics = snr_measurer.get_metrics()

        # Assert
        assert isinstance(metrics, dict)
        assert "algorithm" in metrics
        assert "calculations_performed" in metrics
        assert "average_snr_computed" in metrics
        assert "supports_multicore" in metrics
        assert "noise_models" in metrics

    def test_get_metrics_shows_zero_calculations_initially(self, snr_measurer: StandardSNRMeasurer) -> None:
        """Test get_metrics shows 0 calculations when none performed."""
        # Act
        metrics = snr_measurer.get_metrics()

        # Assert
        assert metrics["calculations_performed"] == 0
        assert metrics["average_snr_computed"] == 0

    def test_get_metrics_calculates_average_correctly(self, snr_measurer: StandardSNRMeasurer) -> None:
        """Test get_metrics calculates average SNR correctly."""
        # Arrange
        snr_measurer._calculations_performed = 3
        snr_measurer._total_snr_computed = 60.0

        # Act
        metrics = snr_measurer.get_metrics()

        # Assert
        assert metrics["average_snr_computed"] == 20.0

    def test_reset_clears_calculation_counters(self, snr_measurer: StandardSNRMeasurer) -> None:
        """Test reset clears calculation counters."""
        # Arrange
        snr_measurer._calculations_performed = 5
        snr_measurer._total_snr_computed = 100.0

        # Act
        snr_measurer.reset()

        # Assert
        assert snr_measurer._calculations_performed == 0
        assert snr_measurer._total_snr_computed == 0.0

    def test_reset_reinitializes_snr_props(self, snr_measurer: StandardSNRMeasurer) -> None:
        """Test reset reinitializes SNR properties."""
        # Arrange
        original_props = snr_measurer.snr_props

        # Act
        snr_measurer.reset()

        # Assert
        assert snr_measurer.snr_props is not original_props

    def test_update_link_state_does_not_raise_error(self, snr_measurer: StandardSNRMeasurer) -> None:
        """Test update_link_state can be called without error."""
        # Arrange
        spectrum_info = {"start_slot": 0, "end_slot": 10}

        # Act & Assert (should not raise)
        snr_measurer.update_link_state(0, 1, spectrum_info)


class TestSetupSNRCalculation:
    """Tests for _setup_snr_calculation method."""

    def test_setup_snr_calculation_sets_bandwidth(self, snr_measurer: StandardSNRMeasurer) -> None:
        """Test _setup_snr_calculation sets bandwidth correctly."""
        # Arrange
        spectrum_info = {"start_slot": 0, "end_slot": 10}

        # Act
        snr_measurer._setup_snr_calculation(spectrum_info)

        # Assert
        expected_bandwidth = 11 * 12.5e9
        assert snr_measurer.snr_props.bandwidth == expected_bandwidth

    def test_setup_snr_calculation_sets_center_frequency(self, snr_measurer: StandardSNRMeasurer) -> None:
        """Test _setup_snr_calculation sets center frequency correctly."""
        # Arrange
        spectrum_info = {"start_slot": 10, "end_slot": 20}

        # Act
        snr_measurer._setup_snr_calculation(spectrum_info)

        # Assert
        # Center should be at slot 15.5 (10 + 11/2)
        expected_frequency = 15.5 * 12.5e9
        assert snr_measurer.snr_props.center_frequency == expected_frequency

    def test_setup_snr_calculation_sets_center_psd(self, snr_measurer: StandardSNRMeasurer) -> None:
        """Test _setup_snr_calculation sets center PSD correctly."""
        # Arrange
        spectrum_info = {"start_slot": 0, "end_slot": 10}
        input_power = 1e-3

        # Act
        snr_measurer._setup_snr_calculation(spectrum_info)

        # Assert
        assert snr_measurer.snr_props.bandwidth is not None
        expected_psd = input_power / snr_measurer.snr_props.bandwidth
        assert snr_measurer.snr_props.center_psd == expected_psd

    def test_setup_snr_calculation_with_custom_slot_width(
        self,
        mock_sdn_props: Any,
        mock_spectrum_props: Any,
        mock_route_props: Any,
    ) -> None:
        """Test _setup_snr_calculation with custom slot width."""
        # Arrange
        engine_props = {"bw_per_slot": 25e9, "input_power": 1e-3}
        measurer = StandardSNRMeasurer(engine_props, mock_sdn_props, mock_spectrum_props, mock_route_props)
        spectrum_info = {"start_slot": 0, "end_slot": 4}

        # Act
        measurer._setup_snr_calculation(spectrum_info)

        # Assert
        expected_bandwidth = 5 * 25e9
        assert measurer.snr_props.bandwidth == expected_bandwidth
