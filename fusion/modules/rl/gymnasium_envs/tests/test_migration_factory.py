"""Tests for create_sim_env factory function.

Phase: P4.3 - Migrate Existing RL Experiments
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from fusion.modules.rl.gymnasium_envs import EnvType, create_sim_env


class TestFactoryBasic:
    """Basic factory function tests."""

    def test_factory_function_exists(self) -> None:
        """Factory function should be importable."""
        from fusion.modules.rl.gymnasium_envs import create_sim_env

        assert callable(create_sim_env)

    def test_env_type_constants_exist(self) -> None:
        """EnvType constants should be defined."""
        from fusion.modules.rl.gymnasium_envs import EnvType

        assert EnvType.LEGACY == "legacy"
        assert EnvType.UNIFIED == "unified"

    def test_factory_creates_legacy_by_default(self) -> None:
        """Default should create legacy environment (mocked)."""
        config = {"k_paths": 3, "spectral_slots": 320}

        # Mock SimEnv since it requires full simulation config
        mock_env = MagicMock()
        with patch(
            "fusion.modules.rl.gymnasium_envs.SimEnv", return_value=mock_env
        ) as mock_simenv:
            with patch.dict(os.environ, {"SUPPRESS_SIMENV_DEPRECATION": "1"}):
                env = create_sim_env(config)

        # Verify SimEnv was called (legacy path)
        mock_simenv.assert_called_once()
        assert env is mock_env


class TestFactoryEnvTypeParameter:
    """Tests for explicit env_type parameter."""

    def test_explicit_legacy(self) -> None:
        """Explicit 'legacy' should call SimEnv (mocked)."""
        config = {"k_paths": 3, "spectral_slots": 320}

        mock_env = MagicMock()
        with patch(
            "fusion.modules.rl.gymnasium_envs.SimEnv", return_value=mock_env
        ) as mock_simenv:
            with patch.dict(os.environ, {"SUPPRESS_SIMENV_DEPRECATION": "1"}):
                env = create_sim_env(config, env_type=EnvType.LEGACY)

        mock_simenv.assert_called_once()
        assert env is mock_env

    def test_explicit_unified(self) -> None:
        """Explicit 'unified' should create UnifiedSimEnv."""
        from fusion.modules.rl.environments import ActionMaskWrapper, UnifiedSimEnv

        config = {"k_paths": 3, "spectral_slots": 320}
        env = create_sim_env(config, env_type=EnvType.UNIFIED)

        # Should be wrapped with ActionMaskWrapper by default
        assert isinstance(env, ActionMaskWrapper)
        # Unwrapped should be UnifiedSimEnv
        assert isinstance(env.unwrapped, UnifiedSimEnv)
        env.close()

    def test_unified_string_type(self) -> None:
        """String 'unified' should work."""
        from fusion.modules.rl.environments import UnifiedSimEnv

        config = {"k_paths": 3, "spectral_slots": 320}
        env = create_sim_env(config, env_type="unified")

        assert isinstance(env.unwrapped, UnifiedSimEnv)
        env.close()

    def test_unified_without_action_mask_wrapper(self) -> None:
        """Should be able to disable ActionMaskWrapper."""
        from fusion.modules.rl.environments import ActionMaskWrapper, UnifiedSimEnv

        config = {"k_paths": 3, "spectral_slots": 320}
        env = create_sim_env(config, env_type="unified", wrap_action_mask=False)

        # Should NOT be wrapped
        assert not isinstance(env, ActionMaskWrapper)
        assert isinstance(env, UnifiedSimEnv)
        env.close()


class TestFactoryEnvVars:
    """Environment variable tests."""

    def test_use_unified_env_true(self) -> None:
        """USE_UNIFIED_ENV=1 should create unified env."""
        from fusion.modules.rl.environments import UnifiedSimEnv

        config = {"k_paths": 3, "spectral_slots": 320}

        with patch.dict(os.environ, {"USE_UNIFIED_ENV": "1"}):
            env = create_sim_env(config)

        assert isinstance(env.unwrapped, UnifiedSimEnv)
        env.close()

    def test_use_unified_env_true_word(self) -> None:
        """USE_UNIFIED_ENV=true should create unified env."""
        from fusion.modules.rl.environments import UnifiedSimEnv

        config = {"k_paths": 3, "spectral_slots": 320}

        with patch.dict(os.environ, {"USE_UNIFIED_ENV": "true"}):
            env = create_sim_env(config)

        assert isinstance(env.unwrapped, UnifiedSimEnv)
        env.close()

    def test_use_unified_env_false(self) -> None:
        """USE_UNIFIED_ENV=0 should create legacy env (mocked)."""
        config = {"k_paths": 3, "spectral_slots": 320}

        mock_env = MagicMock()
        with patch(
            "fusion.modules.rl.gymnasium_envs.SimEnv", return_value=mock_env
        ) as mock_simenv:
            with patch.dict(
                os.environ,
                {"USE_UNIFIED_ENV": "0", "SUPPRESS_SIMENV_DEPRECATION": "1"},
            ):
                env = create_sim_env(config)

        mock_simenv.assert_called_once()
        assert env is mock_env

    def test_rl_env_type_unified(self) -> None:
        """RL_ENV_TYPE=unified should create unified env."""
        from fusion.modules.rl.environments import UnifiedSimEnv

        config = {"k_paths": 3, "spectral_slots": 320}

        with patch.dict(os.environ, {"RL_ENV_TYPE": "unified"}):
            env = create_sim_env(config)

        assert isinstance(env.unwrapped, UnifiedSimEnv)
        env.close()

    def test_rl_env_type_legacy(self) -> None:
        """RL_ENV_TYPE=legacy should create legacy env (mocked)."""
        config = {"k_paths": 3, "spectral_slots": 320}

        mock_env = MagicMock()
        with patch(
            "fusion.modules.rl.gymnasium_envs.SimEnv", return_value=mock_env
        ) as mock_simenv:
            with patch.dict(
                os.environ,
                {"RL_ENV_TYPE": "legacy", "SUPPRESS_SIMENV_DEPRECATION": "1"},
            ):
                env = create_sim_env(config)

        mock_simenv.assert_called_once()
        assert env is mock_env

    def test_explicit_overrides_env_var(self) -> None:
        """Explicit parameter should override env var."""
        config = {"k_paths": 3, "spectral_slots": 320}

        mock_env = MagicMock()
        with patch(
            "fusion.modules.rl.gymnasium_envs.SimEnv", return_value=mock_env
        ) as mock_simenv:
            # Set env var to unified, but explicitly request legacy
            with patch.dict(
                os.environ,
                {"USE_UNIFIED_ENV": "1", "SUPPRESS_SIMENV_DEPRECATION": "1"},
            ):
                env = create_sim_env(config, env_type="legacy")

        # Explicit parameter wins - SimEnv was called
        mock_simenv.assert_called_once()
        assert env is mock_env

    def test_rl_env_type_overrides_use_unified_env(self) -> None:
        """RL_ENV_TYPE should take priority over USE_UNIFIED_ENV."""
        config = {"k_paths": 3, "spectral_slots": 320}

        mock_env = MagicMock()
        with patch(
            "fusion.modules.rl.gymnasium_envs.SimEnv", return_value=mock_env
        ) as mock_simenv:
            # USE_UNIFIED_ENV=1 but RL_ENV_TYPE=legacy
            with patch.dict(
                os.environ,
                {
                    "USE_UNIFIED_ENV": "1",
                    "RL_ENV_TYPE": "legacy",
                    "SUPPRESS_SIMENV_DEPRECATION": "1",
                },
            ):
                env = create_sim_env(config)

        # RL_ENV_TYPE wins - SimEnv was called
        mock_simenv.assert_called_once()
        assert env is mock_env


class TestFactoryConfigFormats:
    """Test factory handles different config formats."""

    def test_dict_config_for_legacy(self) -> None:
        """Dict config should work for legacy env (mocked)."""
        config = {
            "k_paths": 3,
            "spectral_slots": 320,
            "num_requests": 50,
        }

        mock_env = MagicMock()
        with patch(
            "fusion.modules.rl.gymnasium_envs.SimEnv", return_value=mock_env
        ) as mock_simenv:
            with patch.dict(os.environ, {"SUPPRESS_SIMENV_DEPRECATION": "1"}):
                env = create_sim_env(config, env_type="legacy")

        mock_simenv.assert_called_once()
        assert env is mock_env

    def test_dict_config_for_unified(self) -> None:
        """Dict config should work for unified env."""
        from fusion.modules.rl.environments import UnifiedSimEnv

        config = {
            "k_paths": 3,
            "spectral_slots": 320,
            "reward": 1.0,
            "penalty": -1.0,
        }

        env = create_sim_env(config, env_type="unified")

        assert isinstance(env.unwrapped, UnifiedSimEnv)
        # Verify config was extracted
        assert env.unwrapped.config.k_paths == 3
        assert env.unwrapped.config.total_slots == 320
        env.close()

    def test_nested_s1_dict_for_legacy(self) -> None:
        """Nested s1 dict should work for legacy env (mocked)."""
        config = {
            "s1": {
                "k_paths": 3,
                "spectral_slots": 320,
            }
        }

        mock_env = MagicMock()
        with patch(
            "fusion.modules.rl.gymnasium_envs.SimEnv", return_value=mock_env
        ) as mock_simenv:
            with patch.dict(os.environ, {"SUPPRESS_SIMENV_DEPRECATION": "1"}):
                env = create_sim_env(config, env_type="legacy")

        mock_simenv.assert_called_once()
        assert env is mock_env
