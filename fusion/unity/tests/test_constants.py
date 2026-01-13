"""Unit tests for fusion.unity.constants module."""

from fusion.unity.constants import (
    BASH_SCRIPTS_DIR,
    BOOL_TRUE_VALUES,
    CLUSTER_RESULTS_DIR,
    CONFIG_FILE_PATH,
    EXPERIMENTS_DIR,
    MANIFEST_FILE,
    MANIFEST_META_FILE,
    OUTPUT_TO_INPUT_SEGMENTS,
    RESOURCE_KEYS,
    RL_ALGORITHMS,
    RSYNC_DEFAULT_OPTIONS,
    RUNS_INDEX_FILE,
    SIM_INPUT_PATTERN,
    SLURM_OUTPUT_PATTERN,
    SYNC_DELAY_SECONDS,
    TEMP_CONFIG_DIR,
    UNITY_BASE_DIR,
)


class TestConstants:
    """Tests for unity module constants."""

    def test_directory_constants_are_strings(self) -> None:
        """Test that directory constants are defined as strings."""
        # Arrange & Act
        directory_constants = [
            UNITY_BASE_DIR,
            EXPERIMENTS_DIR,
            BASH_SCRIPTS_DIR,
            CLUSTER_RESULTS_DIR,
            TEMP_CONFIG_DIR,
            CONFIG_FILE_PATH,
        ]

        # Assert
        for constant in directory_constants:
            assert isinstance(constant, str)
            assert len(constant) > 0

    def test_file_naming_constants_are_strings(self) -> None:
        """Test that file naming pattern constants are strings."""
        # Arrange & Act
        file_constants = [
            MANIFEST_FILE,
            MANIFEST_META_FILE,
            RUNS_INDEX_FILE,
            SLURM_OUTPUT_PATTERN,
            SIM_INPUT_PATTERN,
        ]

        # Assert
        for constant in file_constants:
            assert isinstance(constant, str)
            assert len(constant) > 0

    def test_resource_keys_contains_expected_values(self) -> None:
        """Test that RESOURCE_KEYS contains all expected SLURM resource keys."""
        # Arrange
        expected_keys = {"partition", "time", "mem", "cpus", "gpus", "nodes"}

        # Act & Assert
        assert RESOURCE_KEYS == expected_keys
        assert isinstance(RESOURCE_KEYS, set)

    def test_rl_algorithms_contains_expected_values(self) -> None:
        """Test that RL_ALGORITHMS contains all expected RL algorithm names."""
        # Arrange
        expected_algorithms = {
            "ppo",
            "qr_dqn",
            "a2c",
            "dqn",
            "epsilon_greedy_bandit",
            "ucb_bandit",
            "q_learning",
        }

        # Act & Assert
        assert RL_ALGORITHMS == expected_algorithms
        assert isinstance(RL_ALGORITHMS, set)

    def test_bool_true_values_contains_expected_values(self) -> None:
        """Test that BOOL_TRUE_VALUES contains expected truthy strings."""
        # Arrange
        expected_values = {"true", "yes", "1"}

        # Act & Assert
        assert BOOL_TRUE_VALUES == expected_values
        assert isinstance(BOOL_TRUE_VALUES, set)

    def test_rsync_options_are_valid(self) -> None:
        """Test that RSYNC_DEFAULT_OPTIONS contains valid rsync flags."""
        # Arrange
        expected_options = ["-avP", "--compress"]

        # Act & Assert
        assert RSYNC_DEFAULT_OPTIONS == expected_options
        assert isinstance(RSYNC_DEFAULT_OPTIONS, list)
        assert all(isinstance(opt, str) for opt in RSYNC_DEFAULT_OPTIONS)

    def test_sync_delay_is_positive_float(self) -> None:
        """Test that SYNC_DELAY_SECONDS is a positive float value."""
        # Arrange & Act & Assert
        assert isinstance(SYNC_DELAY_SECONDS, float)
        assert SYNC_DELAY_SECONDS > 0

    def test_output_to_input_segments_is_positive_int(self) -> None:
        """Test that OUTPUT_TO_INPUT_SEGMENTS is a positive integer."""
        # Arrange & Act & Assert
        assert isinstance(OUTPUT_TO_INPUT_SEGMENTS, int)
        assert OUTPUT_TO_INPUT_SEGMENTS > 0

    def test_unity_base_dir_value(self) -> None:
        """Test that UNITY_BASE_DIR has expected value."""
        # Arrange & Act & Assert
        assert UNITY_BASE_DIR == "unity"

    def test_experiments_dir_value(self) -> None:
        """Test that EXPERIMENTS_DIR has expected value."""
        # Arrange & Act & Assert
        assert EXPERIMENTS_DIR == "experiments"

    def test_manifest_file_value(self) -> None:
        """Test that MANIFEST_FILE has expected value."""
        # Arrange & Act & Assert
        assert MANIFEST_FILE == "manifest.csv"

    def test_runs_index_file_value(self) -> None:
        """Test that RUNS_INDEX_FILE has expected value."""
        # Arrange & Act & Assert
        assert RUNS_INDEX_FILE == "runs_index.json"
