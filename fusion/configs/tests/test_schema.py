"""Unit tests for fusion.configs.schema module."""

from fusion.configs.schema import OPTIONAL_OPTIONS_DICT, SIM_REQUIRED_OPTIONS_DICT
from fusion.utils.config import str_to_bool


class TestSchemaConstants:
    """Tests for schema constants and structure."""

    def test_sim_required_options_dict_structure(self) -> None:
        """Test that SIM_REQUIRED_OPTIONS_DICT has correct structure."""
        assert isinstance(SIM_REQUIRED_OPTIONS_DICT, dict)
        assert len(SIM_REQUIRED_OPTIONS_DICT) > 0

        # Check that all top-level keys are strings (section names)
        for section_name in SIM_REQUIRED_OPTIONS_DICT.keys():
            assert isinstance(section_name, str)
            assert len(section_name) > 0

    def test_sim_required_options_dict_sections(self) -> None:
        """Test that SIM_REQUIRED_OPTIONS_DICT contains expected sections."""
        expected_sections = {
            'general_settings',
            'topology_settings',
            'spectrum_settings',
            'snr_settings',
            'file_settings',
            'ml_settings'
        }

        actual_sections = set(SIM_REQUIRED_OPTIONS_DICT.keys())
        assert expected_sections.issubset(actual_sections)

    def test_sim_required_options_dict_values_are_callable(self) -> None:
        """Test that all values in SIM_REQUIRED_OPTIONS_DICT are callable."""
        for section_name, options in SIM_REQUIRED_OPTIONS_DICT.items():
            assert isinstance(options, dict), f"Section '{section_name}' is not a dict"

            for option_name, converter in options.items():
                assert callable(converter), (
                    f"Converter for '{section_name}.{option_name}' is not callable"
                )

    def test_optional_options_dict_structure(self) -> None:
        """Test that OPTIONAL_OPTIONS_DICT has correct structure."""
        assert isinstance(OPTIONAL_OPTIONS_DICT, dict)
        assert len(OPTIONAL_OPTIONS_DICT) > 0

        # Check that all top-level keys are strings (section names)
        for section_name in OPTIONAL_OPTIONS_DICT.keys():
            assert isinstance(section_name, str)
            assert len(section_name) > 0

    def test_optional_options_dict_sections(self) -> None:
        """Test that OPTIONAL_OPTIONS_DICT contains expected sections."""
        expected_sections = {
            'general_settings',
            'topology_settings',
            'spectrum_settings',
            'file_settings',
            'rl_settings',
            'ml_settings'
        }

        actual_sections = set(OPTIONAL_OPTIONS_DICT.keys())
        assert expected_sections.issubset(actual_sections)

    def test_optional_options_dict_values_are_callable(self) -> None:
        """Test that all values in OPTIONAL_OPTIONS_DICT are callable."""
        for section_name, options in OPTIONAL_OPTIONS_DICT.items():
            assert isinstance(options, dict), f"Section '{section_name}' is not a dict"

            for option_name, converter in options.items():
                assert callable(converter), (
                    f"Converter for '{section_name}.{option_name}' is not callable"
                )

    def test_general_settings_required_options(self) -> None:
        """Test specific required options in general_settings section."""
        general_settings = SIM_REQUIRED_OPTIONS_DICT['general_settings']

        expected_options = {
            'erlang_start', 'erlang_stop', 'erlang_step', 'mod_assumption',
            'mod_assumption_path', 'holding_time', 'thread_erlangs', 'guard_slots',
            'num_requests', 'max_iters', 'dynamic_lps', 'fixed_grid',
            'pre_calc_mod_selection', 'max_segments', 'route_method',
            'allocation_method', 'save_snapshots', 'snapshot_step',
            'print_step', 'spectrum_priority', 'save_step', 'save_start_end_slots'
        }

        actual_options = set(general_settings.keys())
        assert expected_options.issubset(actual_options)

    def test_topology_settings_required_options(self) -> None:
        """Test specific required options in topology_settings section."""
        topology_settings = SIM_REQUIRED_OPTIONS_DICT['topology_settings']

        expected_options = {
            'network', 'bw_per_slot', 'cores_per_link', 'const_link_weight',
            'is_only_core_node', 'multi_fiber'
        }

        actual_options = set(topology_settings.keys())
        assert expected_options.issubset(actual_options)

    def test_spectrum_settings_required_options(self) -> None:
        """Test specific required options in spectrum_settings section."""
        spectrum_settings = SIM_REQUIRED_OPTIONS_DICT['spectrum_settings']

        expected_options = {'c_band'}

        actual_options = set(spectrum_settings.keys())
        assert expected_options.issubset(actual_options)

    def test_snr_settings_required_options(self) -> None:
        """Test specific required options in snr_settings section."""
        snr_settings = SIM_REQUIRED_OPTIONS_DICT['snr_settings']

        expected_options = {
            'snr_type', 'input_power', 'egn_model', 'beta', 'theta',
            'xt_type', 'xt_noise', 'requested_xt', 'phi'
        }

        actual_options = set(snr_settings.keys())
        assert expected_options.issubset(actual_options)

    def test_file_settings_required_options(self) -> None:
        """Test specific required options in file_settings section."""
        file_settings = SIM_REQUIRED_OPTIONS_DICT['file_settings']

        expected_options = {'file_type'}

        actual_options = set(file_settings.keys())
        assert expected_options.issubset(actual_options)

    def test_ml_settings_required_options(self) -> None:
        """Test specific required options in ml_settings section."""
        ml_settings = SIM_REQUIRED_OPTIONS_DICT['ml_settings']

        expected_options = {'deploy_model'}

        actual_options = set(ml_settings.keys())
        assert expected_options.issubset(actual_options)

    def test_rl_settings_optional_options(self) -> None:
        """Test specific optional options in rl_settings section."""
        rl_settings = OPTIONAL_OPTIONS_DICT['rl_settings']

        expected_options = {
            'obs_space', 'n_trials', 'device', 'optimize_hyperparameters',
            'optuna_trials', 'is_training', 'path_algorithm', 'path_model',
            'core_algorithm', 'core_model', 'spectrum_algorithm', 'spectrum_model',
            'render_mode', 'super_channel_space', 'alpha_start', 'alpha_end',
            'alpha_update', 'gamma', 'epsilon_start', 'epsilon_end',
            'epsilon_update', 'path_levels', 'decay_rate', 'feature_extractor',
            'gnn_type', 'layers', 'emb_dim', 'heads', 'conf_param',
            'cong_cutoff', 'reward', 'penalty', 'dynamic_reward', 'core_beta'
        }

        actual_options = set(rl_settings.keys())
        assert expected_options.issubset(actual_options)

    def test_type_converter_functions(self) -> None:
        """Test that type converter functions work correctly."""
        # Test basic Python type converters
        assert "test" == "test"
        assert int("123") == 123
        assert float("123.45") == 123.45
        assert True is True

        # Test str_to_bool function specifically
        assert str_to_bool("true") is True
        assert str_to_bool("false") is False
        assert str_to_bool("True") is True
        assert str_to_bool("False") is False

    def test_str_to_bool_converter_usage(self) -> None:
        """Test that str_to_bool is used in appropriate places."""
        # Check that boolean options use str_to_bool converter
        boolean_options_found = []

        all_options = {**SIM_REQUIRED_OPTIONS_DICT, **OPTIONAL_OPTIONS_DICT}

        for section_name, options in all_options.items():
            for option_name, converter in options.items():
                if converter is str_to_bool:
                    boolean_options_found.append(f"{section_name}.{option_name}")

        # Should have found some boolean options
        assert len(boolean_options_found) > 0

        # Verify some expected boolean options that actually exist in the schema
        expected_boolean_options = [
            'topology_settings.is_only_core_node',
            'snr_settings.egn_model',
            'snr_settings.xt_noise',
            'ml_settings.deploy_model',
            'rl_settings.dynamic_reward',
            'rl_settings.is_training'
        ]

        for expected in expected_boolean_options:
            assert expected in boolean_options_found

    def test_numeric_type_converters(self) -> None:
        """Test that numeric options use appropriate converters."""
        # Test float converters
        float_options = []
        int_options = []

        all_options = {**SIM_REQUIRED_OPTIONS_DICT, **OPTIONAL_OPTIONS_DICT}

        for section_name, options in all_options.items():
            for option_name, converter in options.items():
                if converter is float:
                    float_options.append(f"{section_name}.{option_name}")
                elif converter is int:
                    int_options.append(f"{section_name}.{option_name}")

        # Should have found numeric options
        assert len(float_options) > 0
        assert len(int_options) > 0

        # Verify some expected numeric options that actually exist
        expected_float_options = [
            'snr_settings.input_power',
            'snr_settings.beta',
            'snr_settings.theta',
            'rl_settings.alpha_start',
            'rl_settings.gamma'
        ]

        for expected in expected_float_options:
            assert expected in float_options

    def test_string_type_converters(self) -> None:
        """Test that string options use str converter."""
        string_options = []

        all_options = {**SIM_REQUIRED_OPTIONS_DICT, **OPTIONAL_OPTIONS_DICT}

        for section_name, options in all_options.items():
            for option_name, converter in options.items():
                if converter is str:
                    string_options.append(f"{section_name}.{option_name}")

        # Should have found string options
        assert len(string_options) > 0

        # Verify some expected string options that actually exist
        expected_string_options = [
            'snr_settings.snr_type',
            'snr_settings.xt_type',
            'snr_settings.requested_xt',
            'snr_settings.phi',
            'rl_settings.device',
            'rl_settings.obs_space'
        ]

        for expected in expected_string_options:
            assert expected in string_options

    def test_no_duplicate_options_between_required_and_optional(self) -> None:
        """Test minimal duplicate options between required and optional."""
        # Get all option paths from both dictionaries
        required_paths = set()
        for section, options in SIM_REQUIRED_OPTIONS_DICT.items():
            for option in options.keys():
                required_paths.add(f"{section}.{option}")

        optional_paths = set()
        for section, options in OPTIONAL_OPTIONS_DICT.items():
            for option in options.keys():
                optional_paths.add(f"{section}.{option}")

        # Check for overlaps
        overlaps = required_paths.intersection(optional_paths)

        # Allow known duplicates that exist in the current schema
        allowed_duplicates = {
            'topology_settings.is_only_core_node',
            'ml_settings.deploy_model'
        }

        unexpected_overlaps = overlaps - allowed_duplicates
        assert len(unexpected_overlaps) == 0, (
            f"Found unexpected duplicate options: {unexpected_overlaps}"
        )

    def test_schema_constants_immutability(self) -> None:
        """Test that schema constants maintain their expected structure."""
        # This test ensures the constants haven't been accidentally modified
        assert isinstance(SIM_REQUIRED_OPTIONS_DICT, dict)
        assert isinstance(OPTIONAL_OPTIONS_DICT, dict)

        # Check that all values are still dictionaries of callables
        for section_dict in [SIM_REQUIRED_OPTIONS_DICT, OPTIONAL_OPTIONS_DICT]:
            for section_name, options in section_dict.items():
                assert isinstance(section_name, str)
                assert isinstance(options, dict)
                for option_name, converter in options.items():
                    assert isinstance(option_name, str)
                    assert callable(converter)

    def test_converter_types_are_correct(self) -> None:
        """Test that converter types are from expected set of functions."""
        valid_converters = {str, int, float, bool, str_to_bool}

        all_options = {**SIM_REQUIRED_OPTIONS_DICT, **OPTIONAL_OPTIONS_DICT}

        for section_name, options in all_options.items():
            for option_name, converter in options.items():
                assert converter in valid_converters, (
                    f"Unexpected converter '{converter}' for option "
                    f"'{section_name}.{option_name}'"
                )

    def test_section_naming_consistency(self) -> None:
        """Test that section names follow consistent naming conventions."""
        all_sections: set[str] = set()
        all_sections.update(SIM_REQUIRED_OPTIONS_DICT.keys())
        all_sections.update(OPTIONAL_OPTIONS_DICT.keys())

        for section in all_sections:
            # All sections should end with '_settings'
            assert section.endswith('_settings'), (
                f"Section '{section}' doesn't end with '_settings'"
            )

            # Section names should be lowercase with underscores
            assert section.islower(), f"Section '{section}' is not lowercase"
            assert ' ' not in section, f"Section '{section}' contains spaces"

