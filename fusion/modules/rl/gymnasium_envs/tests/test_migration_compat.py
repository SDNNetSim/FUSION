"""Backward compatibility tests for gymnasium_envs migration.

These tests ensure that existing code patterns continue to work
during the migration period.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch


class TestLegacyImports:
    """Test that legacy imports still work."""

    def test_simenv_import_from_gymnasium_envs(self) -> None:
        """SimEnv should be importable from gymnasium_envs."""
        from fusion.modules.rl.gymnasium_envs import SimEnv

        assert SimEnv is not None

    def test_simenv_import_from_general_sim_env(self) -> None:
        """SimEnv should be importable from general_sim_env."""
        from fusion.modules.rl.gymnasium_envs.general_sim_env import SimEnv

        assert SimEnv is not None

    def test_create_sim_env_import(self) -> None:
        """Factory should be importable."""
        from fusion.modules.rl.gymnasium_envs import create_sim_env

        assert callable(create_sim_env)

    def test_env_type_import(self) -> None:
        """EnvType should be importable."""
        from fusion.modules.rl.gymnasium_envs import EnvType

        assert hasattr(EnvType, "LEGACY")
        assert hasattr(EnvType, "UNIFIED")

    def test_constants_import(self) -> None:
        """Constants should still be importable."""
        from fusion.modules.rl.gymnasium_envs import (
            ARRIVAL_DICT_KEYS,
            DEFAULT_ARRIVAL_COUNT,
            DEFAULT_ITERATION,
            DEFAULT_SAVE_SIMULATION,
            DEFAULT_SIMULATION_KEY,
            SUPPORTED_SPECTRAL_BANDS,
        )

        assert DEFAULT_SIMULATION_KEY is not None
        assert SUPPORTED_SPECTRAL_BANDS is not None
        assert ARRIVAL_DICT_KEYS is not None
        assert DEFAULT_ARRIVAL_COUNT is not None
        assert DEFAULT_ITERATION is not None
        assert DEFAULT_SAVE_SIMULATION is not None


class TestLegacyInstantiation:
    """Test that legacy instantiation patterns work (mocked)."""

    def test_direct_simenv_creation(self) -> None:
        """Direct SimEnv creation pattern should route correctly (mocked)."""
        from fusion.modules.rl.gymnasium_envs import SimEnv

        # We mock the SimEnv to avoid needing full simulation config
        # This tests that the import and class exist
        with patch.object(SimEnv, "__init__", return_value=None):
            with patch.dict(os.environ, {"SUPPRESS_SIMENV_DEPRECATION": "1"}):
                # Verify class is callable
                assert callable(SimEnv)


class TestUnifiedEnvImports:
    """Test UnifiedSimEnv imports."""

    def test_unified_env_importable(self) -> None:
        """UnifiedSimEnv should be importable from environments."""
        from fusion.modules.rl.environments import UnifiedSimEnv

        assert UnifiedSimEnv is not None

    def test_action_mask_wrapper_importable(self) -> None:
        """ActionMaskWrapper should be importable from environments."""
        from fusion.modules.rl.environments import ActionMaskWrapper

        assert ActionMaskWrapper is not None

    def test_path_encoder_importable(self) -> None:
        """PathEncoder should be importable from environments."""
        from fusion.modules.rl.environments import PathEncoder

        assert PathEncoder is not None


class TestBothEnvsUsable:
    """Test that both environments can be used."""

    def test_can_create_both_env_types(self) -> None:
        """Can create both legacy (mocked) and unified envs."""
        from fusion.modules.rl.environments import UnifiedSimEnv
        from fusion.modules.rl.gymnasium_envs import create_sim_env

        config = {"k_paths": 3, "spectral_slots": 320}

        # Create legacy (mocked since it requires full config)
        mock_env = MagicMock()
        with patch("fusion.modules.rl.gymnasium_envs.SimEnv", return_value=mock_env) as mock_simenv:
            with patch.dict(os.environ, {"SUPPRESS_SIMENV_DEPRECATION": "1"}):
                legacy_env = create_sim_env(config, env_type="legacy")

        mock_simenv.assert_called_once()
        assert legacy_env is mock_env

        # Create unified (real)
        unified_env = create_sim_env(config, env_type="unified")
        assert isinstance(unified_env.unwrapped, UnifiedSimEnv)
        unified_env.close()

    def test_both_envs_have_standard_interface(self) -> None:
        """Both envs should have Gymnasium interface."""
        from fusion.modules.rl.gymnasium_envs import create_sim_env

        config = {"k_paths": 3, "spectral_slots": 320}

        # Check legacy (mocked) - factory routes correctly
        mock_env = MagicMock()
        mock_env.observation_space = MagicMock()
        mock_env.action_space = MagicMock()
        mock_env.reset = MagicMock()
        mock_env.step = MagicMock()

        with patch("fusion.modules.rl.gymnasium_envs.SimEnv", return_value=mock_env):
            with patch.dict(os.environ, {"SUPPRESS_SIMENV_DEPRECATION": "1"}):
                legacy_env = create_sim_env(config, env_type="legacy")

        assert hasattr(legacy_env, "observation_space")
        assert hasattr(legacy_env, "action_space")
        assert hasattr(legacy_env, "reset")
        assert hasattr(legacy_env, "step")

        # Check unified (real)
        unified_env = create_sim_env(config, env_type="unified")
        assert hasattr(unified_env, "observation_space")
        assert hasattr(unified_env, "action_space")
        assert hasattr(unified_env, "reset")
        assert hasattr(unified_env, "step")
        unified_env.close()


class TestUnifiedEnvSmoke:
    """Smoke tests for UnifiedSimEnv in standalone mode."""

    def test_unified_env_reset_returns_tuple(self) -> None:
        """Unified env reset should return (obs, info) tuple."""
        from fusion.modules.rl.gymnasium_envs import create_sim_env

        config = {"k_paths": 3, "spectral_slots": 320}
        env = create_sim_env(config, env_type="unified", wrap_action_mask=False)

        result = env.reset(seed=42)

        assert isinstance(result, tuple)
        assert len(result) == 2
        obs, info = result
        assert isinstance(obs, dict)
        assert isinstance(info, dict)
        env.close()

    def test_unified_env_step_returns_5_tuple(self) -> None:
        """Unified env step should return 5-tuple."""
        from fusion.modules.rl.gymnasium_envs import create_sim_env

        config = {"k_paths": 3, "spectral_slots": 320}
        env = create_sim_env(config, env_type="unified", wrap_action_mask=False)

        env.reset(seed=42)
        result = env.step(0)

        assert isinstance(result, tuple)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, dict)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        env.close()

    def test_unified_env_provides_action_mask(self) -> None:
        """Unified env should provide action_mask in info."""
        import numpy as np

        from fusion.modules.rl.gymnasium_envs import create_sim_env

        config = {"k_paths": 3, "spectral_slots": 320}
        env = create_sim_env(config, env_type="unified", wrap_action_mask=False)

        _, info = env.reset(seed=42)

        assert "action_mask" in info
        assert isinstance(info["action_mask"], np.ndarray)
        assert info["action_mask"].shape == (3,)  # k_paths
        env.close()

    def test_action_mask_wrapper_provides_action_masks_method(self) -> None:
        """ActionMaskWrapper should provide action_masks() method."""
        import numpy as np

        from fusion.modules.rl.gymnasium_envs import create_sim_env

        config = {"k_paths": 3, "spectral_slots": 320}
        env = create_sim_env(config, env_type="unified", wrap_action_mask=True)

        env.reset(seed=42)

        # ActionMaskWrapper should have action_masks() method
        assert hasattr(env, "action_masks")
        mask = env.action_masks()
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (3,)
        env.close()
