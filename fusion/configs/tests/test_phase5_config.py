"""
Unit tests for Phase 5 configuration integration (P5.6).

Tests cover:
1. Loading config without phase5 sections works and uses defaults
2. Loading full phase5 config populates fields
3. CLI override test (CLI value beats INI)
4. Integration: config -> PolicyFactory.create() returns correct policy
5. Integration: config -> protection config has correct fields

Phase: P5.6 - Configuration + CLI Integration
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from fusion.configs.cli_to_config import CLIToConfigMapper
from fusion.configs.config import ConfigManager

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def minimal_config_content() -> str:
    """Minimal config without P5.6 sections."""
    return """
[general_settings]
holding_time = 0.2
mod_assumption = DEFAULT
mod_assumption_path = data/json/mods.json
erlang_start = 300
erlang_stop = 600
erlang_step = 300
max_iters = 2
guard_slots = 1
max_segments = 4
thread_erlangs = False
dynamic_lps = False
fixed_grid = False
pre_calc_mod_selection = False
spectrum_priority = None
num_requests = 100
save_snapshots = False
snapshot_step = 10
print_step = 10
save_step = 10
save_start_end_slots = False
is_grooming_enabled = False
can_partially_serve = False
transponder_usage_per_node = False
blocking_type_ci = False

[topology_settings]
network = NSFNet
bw_per_slot = 12.5
cores_per_link = 1
const_link_weight = False
is_only_core_node = True
multi_fiber = False

[spectrum_settings]
c_band = 320

[snr_settings]
snr_type = None
xt_type = without_length
beta = 0.5
theta = 0.0
input_power = 0.001
egn_model = False
phi = {"QPSK": 1}
bi_directional = True
xt_noise = False
requested_xt = {"QPSK": -26.19}
snr_recheck = False
recheck_adjacent_cores = False
recheck_crossband = False

[rl_settings]
obs_space = obs_3
n_trials = 1
device = cpu
optimize_hyperparameters = False
optuna_trials = 1
is_training = False
path_algorithm = first_fit
core_algorithm = first_fit
spectrum_algorithm = first_fit
render_mode = None
super_channel_space = 3
alpha_start = 0.000215
alpha_end = 0.000215
alpha_update = linear_decay
gamma = 0.1
epsilon_start = 0.01
epsilon_end = 0.01
epsilon_update = exp_decay
path_levels = 2
decay_rate = 0.4
feature_extractor = path_gnn
gnn_type = graph_conv
layers = 2
emb_dim = 64
heads = 4
conf_param = 2
cong_cutoff = 0.9
reward = 1
penalty = -10
dynamic_reward = False
core_beta = 0.1

[ml_settings]
deploy_model = False

[file_settings]
file_type = json
"""


@pytest.fixture
def full_phase5_config_content() -> str:
    """Config with full P5.6 sections."""
    return """
[general_settings]
holding_time = 0.2
mod_assumption = DEFAULT
mod_assumption_path = data/json/mods.json
erlang_start = 300
erlang_stop = 600
erlang_step = 300
max_iters = 2
guard_slots = 1
max_segments = 4
thread_erlangs = False
dynamic_lps = False
fixed_grid = False
pre_calc_mod_selection = False
spectrum_priority = None
num_requests = 100
save_snapshots = False
snapshot_step = 10
print_step = 10
save_step = 10
save_start_end_slots = False
is_grooming_enabled = False
can_partially_serve = False
transponder_usage_per_node = False
blocking_type_ci = False

[topology_settings]
network = NSFNet
bw_per_slot = 12.5
cores_per_link = 1
const_link_weight = False
is_only_core_node = True
multi_fiber = False

[spectrum_settings]
c_band = 320

[snr_settings]
snr_type = None
xt_type = without_length
beta = 0.5
theta = 0.0
input_power = 0.001
egn_model = False
phi = {"QPSK": 1}
bi_directional = True
xt_noise = False
requested_xt = {"QPSK": -26.19}
snr_recheck = False
recheck_adjacent_cores = False
recheck_crossband = False

[rl_settings]
obs_space = obs_3
n_trials = 1
device = cpu
optimize_hyperparameters = False
optuna_trials = 1
is_training = False
path_algorithm = first_fit
core_algorithm = first_fit
spectrum_algorithm = first_fit
render_mode = None
super_channel_space = 3
alpha_start = 0.000215
alpha_end = 0.000215
alpha_update = linear_decay
gamma = 0.1
epsilon_start = 0.01
epsilon_end = 0.01
epsilon_update = exp_decay
path_levels = 2
decay_rate = 0.4
feature_extractor = path_gnn
gnn_type = graph_conv
layers = 2
emb_dim = 64
heads = 4
conf_param = 2
cong_cutoff = 0.9
reward = 1
penalty = -10
dynamic_reward = False
core_beta = 0.1

[ml_settings]
deploy_model = False

[file_settings]
file_type = json

[policy_settings]
policy_type = heuristic
policy_name = load_balanced
k_paths = 5
device = cpu

[heuristic_settings]
alpha = 0.7
seed = 42

[protection_settings]
protection_enabled = True
disjointness_type = node
protection_switchover_ms = 25.0
restoration_latency_ms = 75.0
revert_to_primary = True
"""


@pytest.fixture
def minimal_config_file(minimal_config_content: str) -> Path:
    """Create temporary config file without P5.6 sections."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
        f.write(minimal_config_content)
        return Path(f.name)


@pytest.fixture
def full_phase5_config_file(full_phase5_config_content: str) -> Path:
    """Create temporary config file with P5.6 sections."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
        f.write(full_phase5_config_content)
        return Path(f.name)


# =============================================================================
# Backward Compatibility Tests
# =============================================================================


class TestBackwardCompatibility:
    """Tests for backward compatibility when P5.6 sections are absent."""

    def test_load_config_without_phase5_sections(self, minimal_config_file: Path) -> None:
        """Config loads successfully without policy/protection sections."""
        manager = ConfigManager()
        config = manager.load_config(str(minimal_config_file))

        assert config is not None
        assert config.general["holding_time"] == 0.2
        assert config.topology["network"] == "NSFNet"

    def test_phase5_sections_absent_returns_empty_dicts(self, minimal_config_file: Path) -> None:
        """Missing P5.6 sections result in empty dicts in raw config."""
        manager = ConfigManager()
        manager.load_config(str(minimal_config_file))

        # policy_settings and heuristic_settings not in raw config
        assert "policy_settings" not in manager._raw_config
        assert "heuristic_settings" not in manager._raw_config

    def test_existing_configs_still_load(self) -> None:
        """Existing default.ini template loads successfully."""
        template_path = Path(__file__).parent.parent / "templates" / "default.ini"

        if template_path.exists():
            manager = ConfigManager()
            config = manager.load_config(str(template_path))

            assert config is not None
            assert config.general is not None


# =============================================================================
# Full P5.6 Config Loading Tests
# =============================================================================


class TestPhase5ConfigLoading:
    """Tests for loading full P5.6 configuration."""

    def test_load_policy_settings(self, full_phase5_config_file: Path) -> None:
        """Policy settings section loads correctly."""
        manager = ConfigManager()
        manager.load_config(str(full_phase5_config_file))

        policy = manager._raw_config.get("policy_settings", {})

        assert policy["policy_type"] == "heuristic"
        assert policy["policy_name"] == "load_balanced"
        assert policy["k_paths"] == 5
        assert policy["device"] == "cpu"

    def test_load_heuristic_settings(self, full_phase5_config_file: Path) -> None:
        """Heuristic settings section loads correctly."""
        manager = ConfigManager()
        manager.load_config(str(full_phase5_config_file))

        heuristic = manager._raw_config.get("heuristic_settings", {})

        assert heuristic["alpha"] == 0.7
        assert heuristic["seed"] == 42

    def test_load_protection_settings(self, full_phase5_config_file: Path) -> None:
        """Protection settings section loads correctly."""
        manager = ConfigManager()
        manager.load_config(str(full_phase5_config_file))

        protection = manager._raw_config.get("protection_settings", {})

        assert protection["protection_enabled"] is True
        assert protection["disjointness_type"] == "node"
        assert protection["protection_switchover_ms"] == 25.0
        assert protection["restoration_latency_ms"] == 75.0


# =============================================================================
# CLI Override Tests
# =============================================================================


class TestCLIOverride:
    """Tests for CLI arguments overriding config file values."""

    def test_cli_overrides_policy_type(self, full_phase5_config_file: Path) -> None:
        """CLI policy_type overrides INI value."""
        manager = ConfigManager()
        manager.load_config(str(full_phase5_config_file))

        # Verify initial value from INI
        assert manager._raw_config["policy_settings"]["policy_type"] == "heuristic"

        # Override via CLI
        cli_args = {"policy_type": "ml"}
        manager.merge_cli_args(cli_args)

        # CLI should override
        assert manager._raw_config["policy_settings"]["policy_type"] == "ml"

    def test_cli_overrides_heuristic_alpha(self, full_phase5_config_file: Path) -> None:
        """CLI heuristic_alpha overrides INI value."""
        manager = ConfigManager()
        manager.load_config(str(full_phase5_config_file))

        # Verify initial value
        assert manager._raw_config["heuristic_settings"]["alpha"] == 0.7

        # Override via CLI
        cli_args = {"heuristic_alpha": 0.3}
        manager.merge_cli_args(cli_args)

        # CLI should override
        assert manager._raw_config["heuristic_settings"]["alpha"] == 0.3

    def test_cli_overrides_protection_enabled(self, full_phase5_config_file: Path) -> None:
        """CLI protection_enabled overrides INI value."""
        manager = ConfigManager()
        manager.load_config(str(full_phase5_config_file))

        # Verify initial value
        assert manager._raw_config["protection_settings"]["protection_enabled"] is True

        # Override via CLI
        cli_args = {"protection_enabled": False}
        manager.merge_cli_args(cli_args)

        # CLI should override
        assert manager._raw_config["protection_settings"]["protection_enabled"] is False

    def test_cli_adds_new_section(self, minimal_config_file: Path) -> None:
        """CLI adds policy_settings section when not in INI."""
        manager = ConfigManager()
        manager.load_config(str(minimal_config_file))

        # Verify section not present
        assert "policy_settings" not in manager._raw_config

        # Add via CLI
        cli_args = {"policy_type": "heuristic", "policy_name": "shortest"}
        manager.merge_cli_args(cli_args)

        # Section should now exist
        assert "policy_settings" in manager._raw_config
        assert manager._raw_config["policy_settings"]["policy_type"] == "heuristic"
        assert manager._raw_config["policy_settings"]["policy_name"] == "shortest"


# =============================================================================
# CLI Mapper Tests
# =============================================================================


class TestCLIToConfigMapper:
    """Tests for CLI argument mapping."""

    def test_map_policy_args(self) -> None:
        """Policy CLI args map to correct sections."""
        mapper = CLIToConfigMapper()

        args = {
            "policy_type": "ml",
            "policy_name": "custom",
            "policy_model_path": "/path/to/model.zip",
            "policy_fallback": "first_feasible",
        }

        result = mapper.map_args_to_config(args)

        assert "policy_settings" in result
        assert result["policy_settings"]["policy_type"] == "ml"
        assert result["policy_settings"]["policy_name"] == "custom"
        assert result["policy_settings"]["model_path"] == "/path/to/model.zip"
        assert result["policy_settings"]["fallback_policy"] == "first_feasible"

    def test_map_heuristic_args(self) -> None:
        """Heuristic CLI args map to correct sections."""
        mapper = CLIToConfigMapper()

        args = {
            "heuristic_alpha": 0.8,
            "heuristic_seed": 123,
        }

        result = mapper.map_args_to_config(args)

        assert "heuristic_settings" in result
        assert result["heuristic_settings"]["alpha"] == 0.8
        assert result["heuristic_settings"]["seed"] == 123

    def test_map_protection_args(self) -> None:
        """Protection CLI args map to correct sections."""
        mapper = CLIToConfigMapper()

        args = {
            "protection_enabled": True,
            "disjointness_type": "node",
            "protection_switchover_ms": 30.0,
            "restoration_latency_ms": 80.0,
        }

        result = mapper.map_args_to_config(args)

        assert "protection_settings" in result
        assert result["protection_settings"]["protection_enabled"] is True
        assert result["protection_settings"]["disjointness_type"] == "node"
        assert result["protection_settings"]["protection_switchover_ms"] == 30.0


# =============================================================================
# PolicyFactory Integration Tests
# =============================================================================


class TestPolicyFactoryIntegration:
    """Tests for config -> PolicyFactory integration."""

    def test_create_heuristic_from_config(self, full_phase5_config_file: Path) -> None:
        """PolicyFactory creates correct heuristic from config."""
        from fusion.policies import LoadBalancedPolicy, PolicyConfig, PolicyFactory

        manager = ConfigManager()
        manager.load_config(str(full_phase5_config_file))

        policy_settings = manager._raw_config.get("policy_settings", {})
        heuristic_settings = manager._raw_config.get("heuristic_settings", {})

        config = PolicyConfig(
            policy_type=policy_settings.get("policy_type", "heuristic"),
            policy_name=policy_settings.get("policy_name", "first_feasible"),
            alpha=heuristic_settings.get("alpha", 0.5),
        )

        policy = PolicyFactory.create(config)

        assert isinstance(policy, LoadBalancedPolicy)
        assert policy.alpha == 0.7

    def test_create_default_policy_from_empty_config(self, minimal_config_file: Path) -> None:
        """PolicyFactory creates FirstFeasible when config has no policy section."""
        from fusion.policies import FirstFeasiblePolicy, PolicyConfig, PolicyFactory

        manager = ConfigManager()
        manager.load_config(str(minimal_config_file))

        # No policy_settings section
        policy_settings = manager._raw_config.get("policy_settings", {})

        config = PolicyConfig(
            policy_type=policy_settings.get("policy_type", "heuristic"),
            policy_name=policy_settings.get("policy_name", "first_feasible"),
        )

        policy = PolicyFactory.create(config)

        assert isinstance(policy, FirstFeasiblePolicy)

    def test_create_random_policy_with_seed(self) -> None:
        """PolicyFactory creates seeded random policy from config."""
        from fusion.policies import PolicyConfig, PolicyFactory, RandomFeasiblePolicy

        config = PolicyConfig(
            policy_type="heuristic",
            policy_name="random",
            seed=12345,
        )

        policy = PolicyFactory.create(config)

        assert isinstance(policy, RandomFeasiblePolicy)
        assert policy._seed == 12345


# =============================================================================
# Protection Config Integration Tests
# =============================================================================


class TestProtectionConfigIntegration:
    """Tests for config -> protection pipeline integration."""

    def test_protection_config_fields(self, full_phase5_config_file: Path) -> None:
        """Protection config has all expected fields."""
        manager = ConfigManager()
        manager.load_config(str(full_phase5_config_file))

        protection = manager._raw_config.get("protection_settings", {})

        # All P5.6 protection fields present
        assert "protection_enabled" in protection
        assert "disjointness_type" in protection
        assert "protection_switchover_ms" in protection
        assert "restoration_latency_ms" in protection

    def test_create_protection_pipeline_from_config(self, full_phase5_config_file: Path) -> None:
        """ProtectionPipeline created with correct disjointness from config."""
        from fusion.pipelines import DisjointnessType, ProtectionPipeline

        manager = ConfigManager()
        manager.load_config(str(full_phase5_config_file))

        protection = manager._raw_config.get("protection_settings", {})

        disjointness_str = protection.get("disjointness_type", "link")
        disjointness = DisjointnessType(disjointness_str)

        switchover_ms = protection.get("protection_switchover_ms", 50.0)

        pipeline = ProtectionPipeline(
            disjointness=disjointness,
            switchover_latency_ms=switchover_ms,
        )

        assert pipeline.disjointness == DisjointnessType.NODE
        assert pipeline.get_switchover_latency() == 25.0

    def test_default_protection_disabled(self, minimal_config_file: Path) -> None:
        """Protection defaults to disabled when section absent."""
        manager = ConfigManager()
        manager.load_config(str(minimal_config_file))

        protection = manager._raw_config.get("protection_settings", {})

        # Default should be disabled
        protection_enabled = protection.get("protection_enabled", False)

        assert protection_enabled is False
