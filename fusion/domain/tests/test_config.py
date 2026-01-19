"""Tests for SimulationConfig."""

from __future__ import annotations

import pytest

from fusion.domain.config import (
    DEFAULT_BANDWIDTH_MAP,
    DEFAULT_FREQUENCY_SPACING,
    DEFAULT_INPUT_POWER,
    DEFAULT_LIGHT_FREQUENCY,
    DEFAULT_MCI_WORST,
    DEFAULT_MOD_FORMAT_MAP,
    DEFAULT_NSP_PER_BAND,
    DEFAULT_PLANCK_CONSTANT,
    DEFAULT_SPAN_LENGTH,
    SimulationConfig,
)


class TestSimulationConfigCreation:
    """Test SimulationConfig instantiation."""

    def test_create_minimal_config(self) -> None:
        """Test creating config with required fields only."""
        config = SimulationConfig(
            network_name="NSFNET",
            cores_per_link=1,
            band_list=("c",),
            band_slots={"c": 320},
            guard_slots=1,
            num_requests=1000,
            erlang=50.0,
            holding_time=5.0,
            route_method="k_shortest_path",
            k_paths=3,
            allocation_method="first_fit",
        )

        assert config.network_name == "NSFNET"
        assert config.cores_per_link == 1
        assert config.band_list == ("c",)
        assert config.k_paths == 3

    def test_create_full_config(self) -> None:
        """Test creating config with all fields."""
        config = SimulationConfig(
            network_name="USbackbone60",
            cores_per_link=7,
            band_list=("c", "l", "s"),
            band_slots={"c": 320, "l": 320, "s": 320},
            guard_slots=2,
            num_requests=5000,
            erlang=100.0,
            holding_time=10.0,
            route_method="congestion_aware",
            k_paths=5,
            allocation_method="best_fit",
            # Topology constraints
            span_length=80.0,
            max_link_length=500.0,
            max_span=10,
            max_transponders=100,
            single_core=True,
            # Features
            grooming_enabled=True,
            grooming_type="fixed",
            slicing_enabled=True,
            max_slices=4,
            snr_enabled=True,
            snr_type="snr_e2e",
            snr_recheck=True,
            can_partially_serve=True,
            # Protection
            protection_switchover_ms=30.0,
            restoration_latency_ms=75.0,
            # Physical layer
            input_power=2e-3,
            frequency_spacing=25e9,
            light_frequency=1.95e14,
            planck_constant=6.626e-34,
            noise_spectral_density=2.0,
            mci_worst=5e-27,
            nsp_per_band={"c": 1.8, "l": 2.0, "s": 2.1},
            # SNR
            request_bit_rate=25.0,
            request_snr=10.0,
            snr_thresholds={"QPSK": 6.0, "16-QAM": 10.0},
            # Modulation
            modulation_formats={"QPSK": {}, "16-QAM": {}},
            mod_per_bw={"50": {"QPSK": {}}},
            mod_format_map={2: "QPSK", 4: "16-QAM"},
            bandwidth_map={"QPSK": 200, "16-QAM": 400},
        )

        assert config.grooming_enabled is True
        assert config.grooming_type == "fixed"
        assert config.snr_type == "snr_e2e"
        assert config.max_slices == 4
        assert config.span_length == 80.0
        assert config.max_transponders == 100
        assert config.protection_switchover_ms == 30.0
        assert config.input_power == 2e-3
        assert config.request_bit_rate == 25.0

    def test_default_values(self) -> None:
        """Test that default values are correctly set."""
        config = SimulationConfig(
            network_name="test",
            cores_per_link=1,
            band_list=("c",),
            band_slots={"c": 320},
            guard_slots=1,
            num_requests=1000,
            erlang=50.0,
            holding_time=5.0,
            route_method="k_shortest_path",
            k_paths=3,
            allocation_method="first_fit",
        )

        # Topology defaults
        assert config.span_length == DEFAULT_SPAN_LENGTH
        assert config.max_link_length is None
        assert config.max_span is None
        assert config.max_transponders is None
        assert config.single_core is False

        # Feature defaults
        assert config.grooming_enabled is False
        assert config.grooming_type is None
        assert config.slicing_enabled is False
        assert config.max_slices == 1

        # Protection defaults
        assert config.protection_switchover_ms == 50.0
        assert config.restoration_latency_ms == 100.0

        # Physical layer defaults
        assert config.input_power == DEFAULT_INPUT_POWER
        assert config.frequency_spacing == DEFAULT_FREQUENCY_SPACING
        assert config.light_frequency == DEFAULT_LIGHT_FREQUENCY
        assert config.planck_constant == DEFAULT_PLANCK_CONSTANT
        assert config.mci_worst == DEFAULT_MCI_WORST
        assert config.nsp_per_band == DEFAULT_NSP_PER_BAND

        # SNR defaults
        assert config.request_bit_rate == 12.5
        assert config.request_snr == 8.5

        # Modulation defaults
        assert config.mod_format_map == DEFAULT_MOD_FORMAT_MAP
        assert config.bandwidth_map == DEFAULT_BANDWIDTH_MAP

    def test_frozen_immutability(self) -> None:
        """Test that SimulationConfig is immutable."""
        config = SimulationConfig(
            network_name="NSFNET",
            cores_per_link=1,
            band_list=("c",),
            band_slots={"c": 320},
            guard_slots=1,
            num_requests=1000,
            erlang=50.0,
            holding_time=5.0,
            route_method="k_shortest_path",
            k_paths=3,
            allocation_method="first_fit",
        )

        with pytest.raises(AttributeError):
            config.network_name = "other"  # type: ignore[misc]

        with pytest.raises(AttributeError):
            config.k_paths = 5  # type: ignore[misc]


class TestSimulationConfigValidation:
    """Test SimulationConfig validation in __post_init__."""

    def test_invalid_cores_per_link(self) -> None:
        """Test that cores_per_link < 1 raises ValueError."""
        with pytest.raises(ValueError, match="cores_per_link must be >= 1"):
            SimulationConfig(
                network_name="test",
                cores_per_link=0,
                band_list=("c",),
                band_slots={"c": 320},
                guard_slots=1,
                num_requests=1000,
                erlang=50.0,
                holding_time=5.0,
                route_method="k_shortest_path",
                k_paths=3,
                allocation_method="first_fit",
            )

    def test_empty_band_list(self) -> None:
        """Test that empty band_list raises ValueError."""
        with pytest.raises(ValueError, match="band_list cannot be empty"):
            SimulationConfig(
                network_name="test",
                cores_per_link=1,
                band_list=(),
                band_slots={},
                guard_slots=1,
                num_requests=1000,
                erlang=50.0,
                holding_time=5.0,
                route_method="k_shortest_path",
                k_paths=3,
                allocation_method="first_fit",
            )

    def test_negative_guard_slots(self) -> None:
        """Test that negative guard_slots raises ValueError."""
        with pytest.raises(ValueError, match="guard_slots must be >= 0"):
            SimulationConfig(
                network_name="test",
                cores_per_link=1,
                band_list=("c",),
                band_slots={"c": 320},
                guard_slots=-1,
                num_requests=1000,
                erlang=50.0,
                holding_time=5.0,
                route_method="k_shortest_path",
                k_paths=3,
                allocation_method="first_fit",
            )

    def test_invalid_num_requests(self) -> None:
        """Test that num_requests < 1 raises ValueError."""
        with pytest.raises(ValueError, match="num_requests must be >= 1"):
            SimulationConfig(
                network_name="test",
                cores_per_link=1,
                band_list=("c",),
                band_slots={"c": 320},
                guard_slots=1,
                num_requests=0,
                erlang=50.0,
                holding_time=5.0,
                route_method="k_shortest_path",
                k_paths=3,
                allocation_method="first_fit",
            )

    def test_invalid_erlang(self) -> None:
        """Test that erlang <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="erlang must be > 0"):
            SimulationConfig(
                network_name="test",
                cores_per_link=1,
                band_list=("c",),
                band_slots={"c": 320},
                guard_slots=1,
                num_requests=1000,
                erlang=0.0,
                holding_time=5.0,
                route_method="k_shortest_path",
                k_paths=3,
                allocation_method="first_fit",
            )

    def test_invalid_holding_time(self) -> None:
        """Test that holding_time <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="holding_time must be > 0"):
            SimulationConfig(
                network_name="test",
                cores_per_link=1,
                band_list=("c",),
                band_slots={"c": 320},
                guard_slots=1,
                num_requests=1000,
                erlang=50.0,
                holding_time=0.0,
                route_method="k_shortest_path",
                k_paths=3,
                allocation_method="first_fit",
            )

    def test_invalid_k_paths(self) -> None:
        """Test that k_paths < 1 raises ValueError."""
        with pytest.raises(ValueError, match="k_paths must be >= 1"):
            SimulationConfig(
                network_name="test",
                cores_per_link=1,
                band_list=("c",),
                band_slots={"c": 320},
                guard_slots=1,
                num_requests=1000,
                erlang=50.0,
                holding_time=5.0,
                route_method="k_shortest_path",
                k_paths=0,
                allocation_method="first_fit",
            )

    def test_invalid_max_slices(self) -> None:
        """Test that max_slices < 1 raises ValueError."""
        with pytest.raises(ValueError, match="max_slices must be >= 1"):
            SimulationConfig(
                network_name="test",
                cores_per_link=1,
                band_list=("c",),
                band_slots={"c": 320},
                guard_slots=1,
                num_requests=1000,
                erlang=50.0,
                holding_time=5.0,
                route_method="k_shortest_path",
                k_paths=3,
                allocation_method="first_fit",
                max_slices=0,
            )

    def test_missing_band_slots(self) -> None:
        """Test that missing band_slots entry raises ValueError."""
        with pytest.raises(ValueError, match="band_slots missing entry for band 'l'"):
            SimulationConfig(
                network_name="test",
                cores_per_link=1,
                band_list=("c", "l"),
                band_slots={"c": 320},  # Missing "l"
                guard_slots=1,
                num_requests=1000,
                erlang=50.0,
                holding_time=5.0,
                route_method="k_shortest_path",
                k_paths=3,
                allocation_method="first_fit",
            )

    def test_invalid_input_power(self) -> None:
        """Test that input_power <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="input_power must be > 0"):
            SimulationConfig(
                network_name="test",
                cores_per_link=1,
                band_list=("c",),
                band_slots={"c": 320},
                guard_slots=1,
                num_requests=1000,
                erlang=50.0,
                holding_time=5.0,
                route_method="k_shortest_path",
                k_paths=3,
                allocation_method="first_fit",
                input_power=0.0,
            )

    def test_invalid_span_length(self) -> None:
        """Test that span_length <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="span_length must be > 0"):
            SimulationConfig(
                network_name="test",
                cores_per_link=1,
                band_list=("c",),
                band_slots={"c": 320},
                guard_slots=1,
                num_requests=1000,
                erlang=50.0,
                holding_time=5.0,
                route_method="k_shortest_path",
                k_paths=3,
                allocation_method="first_fit",
                span_length=0.0,
            )


class TestSimulationConfigComputedProperties:
    """Test SimulationConfig computed properties."""

    @pytest.fixture
    def config(self) -> SimulationConfig:
        """Create a standard config for testing."""
        return SimulationConfig(
            network_name="NSFNET",
            cores_per_link=7,
            band_list=("c", "l"),
            band_slots={"c": 320, "l": 400},
            guard_slots=1,
            num_requests=1000,
            erlang=50.0,
            holding_time=5.0,
            route_method="k_shortest_path",
            k_paths=3,
            allocation_method="first_fit",
        )

    def test_total_slots(self, config: SimulationConfig) -> None:
        """Test total_slots computed property."""
        assert config.total_slots == 720  # 320 + 400

    def test_arrival_rate(self, config: SimulationConfig) -> None:
        """Test arrival_rate computed property."""
        assert config.arrival_rate == 10.0  # 50.0 / 5.0

    def test_is_multiband_true(self, config: SimulationConfig) -> None:
        """Test is_multiband for multi-band config."""
        assert config.is_multiband is True

    def test_is_multiband_false(self) -> None:
        """Test is_multiband for single-band config."""
        config = SimulationConfig(
            network_name="test",
            cores_per_link=1,
            band_list=("c",),
            band_slots={"c": 320},
            guard_slots=1,
            num_requests=1000,
            erlang=50.0,
            holding_time=5.0,
            route_method="k_shortest_path",
            k_paths=3,
            allocation_method="first_fit",
        )
        assert config.is_multiband is False

    def test_is_multicore_true(self, config: SimulationConfig) -> None:
        """Test is_multicore for MCF config."""
        assert config.is_multicore is True

    def test_is_multicore_false(self) -> None:
        """Test is_multicore for single-core config."""
        config = SimulationConfig(
            network_name="test",
            cores_per_link=1,
            band_list=("c",),
            band_slots={"c": 320},
            guard_slots=1,
            num_requests=1000,
            erlang=50.0,
            holding_time=5.0,
            route_method="k_shortest_path",
            k_paths=3,
            allocation_method="first_fit",
        )
        assert config.is_multicore is False

    def test_protection_enabled_true(self) -> None:
        """Test protection_enabled for 1+1 protection config."""
        config = SimulationConfig(
            network_name="test",
            cores_per_link=1,
            band_list=("c",),
            band_slots={"c": 320},
            guard_slots=1,
            num_requests=1000,
            erlang=50.0,
            holding_time=5.0,
            route_method="1plus1_protection",
            k_paths=3,
            allocation_method="first_fit",
        )
        assert config.protection_enabled is True

    def test_protection_enabled_false(self, config: SimulationConfig) -> None:
        """Test protection_enabled for non-protection config."""
        assert config.protection_enabled is False


class TestSimulationConfigFromEngineProps:
    """Test SimulationConfig.from_engine_props() adapter."""

    def test_from_minimal_props(self) -> None:
        """Test conversion from minimal engine_props."""
        props = {
            "network": "NSFNET",
            "num_requests": 1000,
        }
        config = SimulationConfig.from_engine_props(props)

        assert config.network_name == "NSFNET"
        assert config.num_requests == 1000
        # Check defaults
        assert config.cores_per_link == 1
        assert config.band_list == ("c",)
        assert config.k_paths == 3

    def test_from_full_props(self) -> None:
        """Test conversion from full engine_props."""
        props = {
            "network": "USbackbone60",
            "cores_per_link": 7,
            "band_list": ["c", "l", "s"],
            "c_band": 320,
            "l_band": 320,
            "s_band": 320,
            "guard_slots": 2,
            "num_requests": 5000,
            "arrival_rate": 20.0,
            "holding_time": 5.0,
            "route_method": "congestion_aware",
            "k_paths": 5,
            "allocation_method": "best_fit",
            "is_grooming_enabled": True,
            "grooming_type": "dynamic",
            "max_segments": 4,
            "snr_type": "snr_e2e",
            "snr_recheck": True,
            "can_partially_serve": True,
            "mod_per_bw": {"50": {"QPSK": {}}},
        }
        config = SimulationConfig.from_engine_props(props)

        assert config.network_name == "USbackbone60"
        assert config.cores_per_link == 7
        assert config.band_list == ("c", "l", "s")
        assert config.band_slots == {"c": 320, "l": 320, "s": 320}
        assert config.erlang == 100.0  # 20.0 * 5.0
        assert config.grooming_enabled is True
        assert config.grooming_type == "dynamic"
        assert config.snr_type == "snr_e2e"

    def test_from_props_with_physical_layer(self) -> None:
        """Test conversion with physical layer parameters."""
        props = {
            "network": "test",
            "num_requests": 100,
            "input_power": 2e-3,
            "frequency_spacing": 25e9,
            "span_length": 80.0,
            "mci_worst": 5e-27,
            "light_frequency": 1.95e14,
            "planck_constant": 6.626e-34,
            "noise_spectral_density": 2.0,
            "nsp": {"c": 1.8, "l": 2.0},
        }
        config = SimulationConfig.from_engine_props(props)

        assert config.input_power == 2e-3
        assert config.frequency_spacing == 25e9
        assert config.span_length == 80.0
        assert config.mci_worst == 5e-27
        assert config.nsp_per_band == {"c": 1.8, "l": 2.0}

    def test_from_props_with_topology_constraints(self) -> None:
        """Test conversion with topology constraint parameters."""
        props = {
            "network": "test",
            "num_requests": 100,
            "max_link_length": 500.0,
            "max_span": 10,
            "number_of_transponders": 50,
            "single_core": True,
        }
        config = SimulationConfig.from_engine_props(props)

        assert config.max_link_length == 500.0
        assert config.max_span == 10
        assert config.max_transponders == 50
        assert config.single_core is True

    def test_from_props_with_protection(self) -> None:
        """Test conversion with protection parameters."""
        props = {
            "network": "test",
            "num_requests": 100,
            "protection_switchover_ms": 30.0,
            "restoration_latency_ms": 75.0,
        }
        config = SimulationConfig.from_engine_props(props)

        assert config.protection_switchover_ms == 30.0
        assert config.restoration_latency_ms == 75.0

    def test_from_props_with_snr_config(self) -> None:
        """Test conversion with SNR configuration."""
        props = {
            "network": "test",
            "num_requests": 100,
            "request_bit_rate": 25.0,
            "request_snr": 10.0,
            "snr_thresholds": {"QPSK": 6.5, "16-QAM": 10.5},
        }
        config = SimulationConfig.from_engine_props(props)

        assert config.request_bit_rate == 25.0
        assert config.request_snr == 10.0
        assert config.snr_thresholds == {"QPSK": 6.5, "16-QAM": 10.5}

    def test_from_props_with_modulation_maps(self) -> None:
        """Test conversion with modulation mapping parameters."""
        props = {
            "network": "test",
            "num_requests": 100,
            "modulation_format_mapping_dict": {2: "QPSK", 4: "16-QAM"},
            "bandwidth_mapping_dict": {"QPSK": 200, "16-QAM": 400},
        }
        config = SimulationConfig.from_engine_props(props)

        assert config.mod_format_map == {2: "QPSK", 4: "16-QAM"}
        assert config.bandwidth_map == {"QPSK": 200, "16-QAM": 400}

    def test_from_props_empty_snr_type(self) -> None:
        """Test that empty snr_type string is treated as None."""
        props = {
            "network": "test",
            "num_requests": 100,
            "snr_type": "",
        }
        config = SimulationConfig.from_engine_props(props)

        assert config.snr_type is None
        assert config.snr_enabled is False

    def test_from_props_band_list_as_string(self) -> None:
        """Test handling of band_list as single string."""
        props = {
            "network": "test",
            "num_requests": 100,
            "band_list": "c",
        }
        config = SimulationConfig.from_engine_props(props)

        assert config.band_list == ("c",)

    def test_from_props_slicing_enabled(self) -> None:
        """Test that slicing_enabled is set when max_segments > 1."""
        props = {
            "network": "test",
            "num_requests": 100,
            "max_segments": 4,
        }
        config = SimulationConfig.from_engine_props(props)

        assert config.slicing_enabled is True
        assert config.max_slices == 4


class TestSimulationConfigToEngineProps:
    """Test SimulationConfig.to_engine_props() adapter."""

    def test_to_engine_props_basic(self) -> None:
        """Test conversion to engine_props."""
        config = SimulationConfig(
            network_name="NSFNET",
            cores_per_link=1,
            band_list=("c",),
            band_slots={"c": 320},
            guard_slots=1,
            num_requests=1000,
            erlang=50.0,
            holding_time=5.0,
            route_method="k_shortest_path",
            k_paths=3,
            allocation_method="first_fit",
        )

        props = config.to_engine_props()

        assert props["network"] == "NSFNET"
        assert props["cores_per_link"] == 1
        assert props["band_list"] == ["c"]  # List, not tuple
        assert props["c_band"] == 320
        assert props["guard_slots"] == 1
        assert props["num_requests"] == 1000
        assert props["arrival_rate"] == 10.0  # 50.0 / 5.0
        assert props["holding_time"] == 5.0
        assert props["route_method"] == "k_shortest_path"
        assert props["k_paths"] == 3
        assert props["allocation_method"] == "first_fit"

    def test_to_engine_props_multiband(self) -> None:
        """Test conversion with multiple bands."""
        config = SimulationConfig(
            network_name="test",
            cores_per_link=1,
            band_list=("c", "l", "s"),
            band_slots={"c": 320, "l": 400, "s": 300},
            guard_slots=1,
            num_requests=1000,
            erlang=50.0,
            holding_time=5.0,
            route_method="k_shortest_path",
            k_paths=3,
            allocation_method="first_fit",
        )

        props = config.to_engine_props()

        assert props["band_list"] == ["c", "l", "s"]
        assert props["c_band"] == 320
        assert props["l_band"] == 400
        assert props["s_band"] == 300

    def test_to_engine_props_features(self) -> None:
        """Test that feature flags are correctly converted."""
        config = SimulationConfig(
            network_name="test",
            cores_per_link=1,
            band_list=("c",),
            band_slots={"c": 320},
            guard_slots=1,
            num_requests=1000,
            erlang=50.0,
            holding_time=5.0,
            route_method="k_shortest_path",
            k_paths=3,
            allocation_method="first_fit",
            grooming_enabled=True,
            grooming_type="fixed",
            max_slices=4,
            snr_type="snr_e2e",
            snr_recheck=True,
        )

        props = config.to_engine_props()

        assert props["is_grooming_enabled"] is True
        assert props["grooming_type"] == "fixed"
        assert props["max_segments"] == 4
        assert props["snr_type"] == "snr_e2e"
        assert props["snr_recheck"] is True

    def test_to_engine_props_physical_layer(self) -> None:
        """Test that physical layer parameters are correctly converted."""
        config = SimulationConfig(
            network_name="test",
            cores_per_link=1,
            band_list=("c",),
            band_slots={"c": 320},
            guard_slots=1,
            num_requests=1000,
            erlang=50.0,
            holding_time=5.0,
            route_method="k_shortest_path",
            k_paths=3,
            allocation_method="first_fit",
            input_power=2e-3,
            frequency_spacing=25e9,
            span_length=80.0,
            mci_worst=5e-27,
            nsp_per_band={"c": 1.8},
        )

        props = config.to_engine_props()

        assert props["input_power"] == 2e-3
        assert props["frequency_spacing"] == 25e9
        assert props["span_length"] == 80.0
        assert props["mci_worst"] == 5e-27
        assert props["nsp"] == {"c": 1.8}

    def test_to_engine_props_topology_constraints(self) -> None:
        """Test that topology constraints are correctly converted."""
        config = SimulationConfig(
            network_name="test",
            cores_per_link=1,
            band_list=("c",),
            band_slots={"c": 320},
            guard_slots=1,
            num_requests=1000,
            erlang=50.0,
            holding_time=5.0,
            route_method="k_shortest_path",
            k_paths=3,
            allocation_method="first_fit",
            max_link_length=500.0,
            max_span=10,
            max_transponders=50,
            single_core=True,
        )

        props = config.to_engine_props()

        assert props["max_link_length"] == 500.0
        assert props["max_span"] == 10
        assert props["number_of_transponders"] == 50
        assert props["single_core"] is True

    def test_to_engine_props_protection(self) -> None:
        """Test that protection parameters are correctly converted."""
        config = SimulationConfig(
            network_name="test",
            cores_per_link=1,
            band_list=("c",),
            band_slots={"c": 320},
            guard_slots=1,
            num_requests=1000,
            erlang=50.0,
            holding_time=5.0,
            route_method="k_shortest_path",
            k_paths=3,
            allocation_method="first_fit",
            protection_switchover_ms=30.0,
            restoration_latency_ms=75.0,
        )

        props = config.to_engine_props()

        assert props["protection_switchover_ms"] == 30.0
        assert props["restoration_latency_ms"] == 75.0

    def test_to_engine_props_modulation_maps(self) -> None:
        """Test that modulation mappings are correctly converted."""
        config = SimulationConfig(
            network_name="test",
            cores_per_link=1,
            band_list=("c",),
            band_slots={"c": 320},
            guard_slots=1,
            num_requests=1000,
            erlang=50.0,
            holding_time=5.0,
            route_method="k_shortest_path",
            k_paths=3,
            allocation_method="first_fit",
            mod_format_map={2: "QPSK", 4: "16-QAM"},
            bandwidth_map={"QPSK": 200, "16-QAM": 400},
        )

        props = config.to_engine_props()

        assert props["modulation_format_mapping_dict"] == {2: "QPSK", 4: "16-QAM"}
        assert props["bandwidth_mapping_dict"] == {"QPSK": 200, "16-QAM": 400}


class TestSimulationConfigRoundtrip:
    """Test roundtrip conversion preserves data."""

    def test_roundtrip_basic(self) -> None:
        """Test from_engine_props -> to_engine_props roundtrip."""
        original = {
            "network": "NSFNET",
            "cores_per_link": 1,
            "band_list": ["c"],
            "c_band": 320,
            "guard_slots": 1,
            "num_requests": 1000,
            "arrival_rate": 10.0,
            "holding_time": 5.0,
            "route_method": "k_shortest_path",
            "k_paths": 3,
            "allocation_method": "first_fit",
            "is_grooming_enabled": False,
            "max_segments": 1,
            "snr_type": None,
        }

        config = SimulationConfig.from_engine_props(original)
        roundtrip = config.to_engine_props()

        assert roundtrip["network"] == original["network"]
        assert roundtrip["cores_per_link"] == original["cores_per_link"]
        assert roundtrip["band_list"] == original["band_list"]
        assert roundtrip["c_band"] == original["c_band"]
        assert roundtrip["guard_slots"] == original["guard_slots"]
        assert roundtrip["num_requests"] == original["num_requests"]
        assert roundtrip["arrival_rate"] == original["arrival_rate"]
        assert roundtrip["holding_time"] == original["holding_time"]
        assert roundtrip["route_method"] == original["route_method"]
        assert roundtrip["k_paths"] == original["k_paths"]
        assert roundtrip["allocation_method"] == original["allocation_method"]

    def test_roundtrip_complex(self) -> None:
        """Test roundtrip with complex configuration."""
        original = {
            "network": "USbackbone60",
            "cores_per_link": 7,
            "band_list": ["c", "l"],
            "c_band": 320,
            "l_band": 320,
            "guard_slots": 2,
            "num_requests": 5000,
            "arrival_rate": 20.0,
            "holding_time": 5.0,
            "route_method": "congestion_aware",
            "k_paths": 5,
            "allocation_method": "best_fit",
            "is_grooming_enabled": True,
            "grooming_type": "dynamic",
            "max_segments": 4,
            "snr_type": "snr_e2e",
            "snr_recheck": True,
        }

        config = SimulationConfig.from_engine_props(original)
        roundtrip = config.to_engine_props()

        assert roundtrip["network"] == original["network"]
        assert roundtrip["cores_per_link"] == original["cores_per_link"]
        assert roundtrip["is_grooming_enabled"] == original["is_grooming_enabled"]
        assert roundtrip["grooming_type"] == original["grooming_type"]
        assert roundtrip["max_segments"] == original["max_segments"]
        assert roundtrip["snr_type"] == original["snr_type"]

    def test_roundtrip_physical_layer(self) -> None:
        """Test roundtrip preserves physical layer parameters."""
        original = {
            "network": "test",
            "num_requests": 100,
            "arrival_rate": 10.0,
            "holding_time": 5.0,
            "input_power": 2e-3,
            "frequency_spacing": 25e9,
            "span_length": 80.0,
            "mci_worst": 5e-27,
            "nsp": {"c": 1.8, "l": 2.0},
        }

        config = SimulationConfig.from_engine_props(original)
        roundtrip = config.to_engine_props()

        assert roundtrip["input_power"] == original["input_power"]
        assert roundtrip["frequency_spacing"] == original["frequency_spacing"]
        assert roundtrip["span_length"] == original["span_length"]
        assert roundtrip["mci_worst"] == original["mci_worst"]
        assert roundtrip["nsp"] == original["nsp"]

    def test_roundtrip_topology_constraints(self) -> None:
        """Test roundtrip preserves topology constraints."""
        original = {
            "network": "test",
            "num_requests": 100,
            "arrival_rate": 10.0,
            "holding_time": 5.0,
            "max_link_length": 500.0,
            "max_span": 10,
            "number_of_transponders": 50,
            "single_core": True,
        }

        config = SimulationConfig.from_engine_props(original)
        roundtrip = config.to_engine_props()

        assert roundtrip["max_link_length"] == original["max_link_length"]
        assert roundtrip["max_span"] == original["max_span"]
        assert roundtrip["number_of_transponders"] == original["number_of_transponders"]
        assert roundtrip["single_core"] == original["single_core"]

    def test_roundtrip_preserves_erlang(self) -> None:
        """Test that erlang is correctly computed from arrival_rate * holding_time."""
        original = {
            "network": "test",
            "arrival_rate": 15.0,
            "holding_time": 4.0,
        }

        config = SimulationConfig.from_engine_props(original)
        assert config.erlang == 60.0  # 15.0 * 4.0

        roundtrip = config.to_engine_props()
        assert roundtrip["arrival_rate"] == 15.0
        assert roundtrip["holding_time"] == 4.0
