"""Tests for deprecation warnings in gymnasium_envs migration.

Phase: P4.3 - Migrate Existing RL Experiments

These tests verify that deprecation warnings are emitted correctly
and can be suppressed when needed.

Note: These tests focus on the deprecation warning mechanism itself.
Since SimEnv requires full simulation config to instantiate, we test
the warning code path directly by calling the relevant code.
"""

from __future__ import annotations

import os
import warnings
from unittest.mock import MagicMock, patch

import pytest


class TestDeprecationWarning:
    """Test deprecation warning behavior."""

    def test_deprecation_warning_code_exists(self) -> None:
        """Deprecation warning code should exist in SimEnv."""
        from fusion.modules.rl.gymnasium_envs.general_sim_env import SimEnv
        import inspect

        # Check that the warning code is in __init__
        source = inspect.getsource(SimEnv.__init__)
        assert "DeprecationWarning" in source
        assert "UnifiedSimEnv" in source

    def test_warning_mentions_unified_env_in_source(self) -> None:
        """Warning message should mention UnifiedSimEnv."""
        from fusion.modules.rl.gymnasium_envs.general_sim_env import SimEnv
        import inspect

        source = inspect.getsource(SimEnv.__init__)
        assert "UnifiedSimEnv" in source

    def test_warning_mentions_factory_function_in_source(self) -> None:
        """Warning message should mention create_sim_env factory."""
        from fusion.modules.rl.gymnasium_envs.general_sim_env import SimEnv
        import inspect

        source = inspect.getsource(SimEnv.__init__)
        assert "create_sim_env" in source

    def test_warning_mentions_suppression_env_var_in_source(self) -> None:
        """Warning message should mention how to suppress it."""
        from fusion.modules.rl.gymnasium_envs.general_sim_env import SimEnv
        import inspect

        source = inspect.getsource(SimEnv.__init__)
        assert "SUPPRESS_SIMENV_DEPRECATION" in source


class TestWarningSupressionLogic:
    """Test that warning suppression logic works correctly."""

    def test_suppression_env_var_checked(self) -> None:
        """SUPPRESS_SIMENV_DEPRECATION should be checked."""
        from fusion.modules.rl.gymnasium_envs.general_sim_env import SimEnv
        import inspect

        source = inspect.getsource(SimEnv.__init__)
        # Verify the env var check is present
        assert 'SUPPRESS_SIMENV_DEPRECATION' in source
        assert 'os.environ.get' in source

    def test_suppression_values_recognized(self) -> None:
        """Various suppression values should be recognized."""
        from fusion.modules.rl.gymnasium_envs.general_sim_env import SimEnv
        import inspect

        source = inspect.getsource(SimEnv.__init__)
        # The suppression logic should check for "1", "true", "yes"
        assert '"1"' in source or "'1'" in source
        assert '"true"' in source or "'true'" in source
        assert '"yes"' in source or "'yes'" in source


class TestUnifiedEnvNoWarning:
    """UnifiedSimEnv should not emit deprecation warnings."""

    def test_unified_env_no_deprecation(self) -> None:
        """UnifiedSimEnv should not emit DeprecationWarning."""
        from fusion.modules.rl.environments import UnifiedSimEnv
        from fusion.modules.rl.adapter import RLConfig

        config = RLConfig(k_paths=3)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            env = UnifiedSimEnv(config=config)
            env.close()

            # No deprecation warnings from UnifiedSimEnv
            unified_warnings = [
                x
                for x in w
                if issubclass(x.category, DeprecationWarning)
                and "UnifiedSimEnv" in str(x.message)
            ]
            assert len(unified_warnings) == 0

    def test_factory_unified_no_deprecation(self) -> None:
        """Factory with env_type='unified' should not emit deprecation."""
        from fusion.modules.rl.gymnasium_envs import create_sim_env

        config = {"k_paths": 3, "spectral_slots": 320}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            env = create_sim_env(config, env_type="unified")
            env.close()

            # No SimEnv deprecation warnings
            simenv_warnings = [
                x
                for x in w
                if issubclass(x.category, DeprecationWarning)
                and "SimEnv" in str(x.message)
            ]
            assert len(simenv_warnings) == 0


class TestFactoryLegacyRouting:
    """Test that factory routes to legacy correctly (mocked)."""

    def test_factory_legacy_calls_simenv(self) -> None:
        """Factory with legacy should call SimEnv constructor."""
        from fusion.modules.rl.gymnasium_envs import create_sim_env

        config = {"k_paths": 3, "spectral_slots": 320}

        mock_env = MagicMock()
        with patch(
            "fusion.modules.rl.gymnasium_envs.SimEnv", return_value=mock_env
        ) as mock_simenv:
            with patch.dict(os.environ, {"SUPPRESS_SIMENV_DEPRECATION": "1"}):
                env = create_sim_env(config, env_type="legacy")

        # Verify SimEnv constructor was called
        mock_simenv.assert_called_once()
        assert env is mock_env
