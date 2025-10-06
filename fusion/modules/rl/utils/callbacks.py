"""
Custom callbacks for StableBaselines3 reinforcement learning training.

This module provides specialized callback implementations for monitoring
and managing RL training processes, including episodic reward tracking
and dynamic hyperparameter adjustment.
"""

# Standard library imports
import os
from typing import Any

# Third-party imports
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from fusion.utils.logging_config import get_logger

# Local imports
from fusion.utils.os import create_directory

__all__ = [
    "EpisodicRewardCallback",
    "GetModelParams",
    "LearnRateEntCallback",
]


class GetModelParams(BaseCallback):
    """
    Handles all methods related to custom callbacks in the StableBaselines3 library.

    This callback extracts model parameters and value function estimates
    during training for monitoring and analysis purposes.
    """

    def __init__(self, verbose: int = 0) -> None:
        """
        Initializes the GetModelParams callback.

        :param verbose: Verbosity level for logging
        :type verbose: int
        """
        super().__init__(verbose)

        self.model_params: dict[str, Any] | None = None
        self.value_estimate = 0.0

    def _on_step(self) -> bool:
        """
        Every step of the model this method is called.

        Retrieves the estimated value function for the PPO algorithm.

        :return: True to continue training
        :rtype: bool
        """
        self.model_params = self.model.get_parameters()

        obs = self.locals.get('obs_tensor')
        if obs is not None:
            try:
                values = self.model.policy.predict_values(obs=obs)  # type: ignore[misc]
                self.value_estimate = values[0][0].item()
            except (AttributeError, IndexError, TypeError):
                self.value_estimate = 0.0
        else:
            self.value_estimate = 0.0
        return True


class EpisodicRewardCallback(BaseCallback):
    """
    Finds rewards episodically for any DRL algorithm from SB3.

    This callback tracks episode rewards and saves them periodically
    for analysis and visualization purposes.
    """

    def __init__(self, verbose: int = 0) -> None:
        """
        Initializes the EpisodicRewardCallback.

        :param verbose: Verbosity level for logging
        :type verbose: int
        """
        super().__init__(verbose)
        self.episode_rewards_array = np.array([])
        self._callback_logger = get_logger(__name__)
        self.current_episode_reward = 0
        self.max_iters: int | None = None
        self.sim_dict: dict[str, Any] | None = None

        self.iteration = 0
        self.trial = 1
        self.current_step = 0

        self.rewards_matrix_array: np.ndarray | None = None

    @property
    def episode_rewards(self) -> np.ndarray:
        """Access episode rewards for backward compatibility with tests."""
        return self.episode_rewards_array

    @property
    def rewards_matrix(self) -> np.ndarray | None:
        """Access rewards matrix for backward compatibility with tests."""
        return self.rewards_matrix_array

    @rewards_matrix.setter
    def rewards_matrix(self, value: np.ndarray | None) -> None:
        """Set rewards matrix for backward compatibility with tests."""
        self.rewards_matrix_array = value

    def _save_drl_trial_rewards(self) -> None:
        """
        Save the average rewards for the current DRL trial to a NumPy file.

        Creates the necessary directory structure and saves rewards with
        descriptive filename including Erlang load, core count, trial number,
        and iteration information.
        """
        if self.sim_dict is None:
            return

        erlang = float(self.sim_dict['erlang_start'])
        cores = int(self.sim_dict['cores_per_link'])
        file_path = os.path.join(
            "logs",
            self.sim_dict["path_algorithm"],
            self.sim_dict["network"],
            self.sim_dict["date"],
            self.sim_dict["sim_start"],
        )
        create_directory(directory_path=file_path)

        file_name = os.path.join(
            file_path,
            f"rewards_e{erlang}_routes_c{cores}_t{self.trial}_iter_{self.iteration}.npy",
        )
        if self.rewards_matrix_array is not None:
            rewards_array = self.rewards_matrix_array[
                :self.iteration + 1, :
            ].mean(axis=0)
        else:
            rewards_array = np.array([])
        np.save(file_name, rewards_array)

    def _on_step(self) -> bool:
        if self.rewards_matrix_array is None and self.sim_dict is not None:
            self.rewards_matrix_array = np.empty((
                self.sim_dict['max_iters'], self.sim_dict['num_requests']
            ))

        reward = self.locals.get("rewards", 0)[0]
        done = self.locals.get("dones", False)[0]
        self.current_episode_reward += reward
        if self.rewards_matrix_array is not None:
            self.rewards_matrix_array[self.iteration, self.current_step] = reward
        self.current_step += 1

        if done:
            self.episode_rewards_array = np.append(
                self.episode_rewards_array, self.current_episode_reward
            )
            if (self.sim_dict is not None and self.max_iters is not None and
                ((self.iteration % self.sim_dict["save_step"]) == 0 or
                 (self.iteration == self.max_iters - 1))):
                self._save_drl_trial_rewards()

            self.iteration += 1
            self.current_step = 0
            if self.verbose:
                self._callback_logger.info(
                    "Episode %d finished with reward: %f",
                    len(self.episode_rewards_array),
                    self.current_episode_reward
                )
                if (self.max_iters is not None and
                    len(self.episode_rewards_array) == self.max_iters):
                    self.current_episode_reward = 0
                    self.iteration = 0
                    return False

            self.current_episode_reward = 0

        return True


class LearnRateEntCallback(BaseCallback):
    """
    Callback to decay learning rate linearly and entropy coefficient exponentially.

    Uses the same done-based logic as EpisodicRewardCallback to update
    hyperparameters after each episode completion.
    """

    def __init__(self, verbose: int = 1) -> None:
        """
        Initializes the LearnRateEntCallback.

        :param verbose: Verbosity level for logging
        :type verbose: int
        """
        super().__init__(verbose)
        self.sim_dict: dict[str, Any] | None = None
        self._callback_logger = get_logger(__name__)
        self.iteration = 0
        self.trial = 1

        self.current_entropy: float | None = None
        self.current_learning_rate: float | None = None

    # NOTE: Parameters should be reset after trial completion for proper isolation
    # Verify that learning rate and entropy coefficient updates are applied to SB3
    def _on_step(self) -> bool:
        """
        Called at each step to update learning rate and entropy coefficient.

        :return: True to continue training
        :rtype: bool
        """
        done = self.locals.get("dones", [False])[0]
        if done and self.sim_dict is not None:
            if self.current_entropy is None:
                self.current_entropy = self.sim_dict['epsilon_start']
                self.current_learning_rate = self.sim_dict['alpha_start']

            self.iteration += 1

            progress = min(self.iteration / self.sim_dict["max_iters"], 1.0)
            self.current_learning_rate = (
                self.sim_dict["alpha_start"]
                + (self.sim_dict["alpha_end"] - self.sim_dict["alpha_start"])
                * progress
            )

            if self.sim_dict["path_algorithm"] in ("ppo", "a2c"):
                if self.current_entropy is not None:
                    self.current_entropy = max(
                        self.sim_dict["epsilon_end"],
                        self.current_entropy * self.sim_dict["decay_rate"],
                    )
                if hasattr(self.model, 'ent_coef'):
                    self.model.ent_coef = self.current_entropy
            self.model.learning_rate = self.current_learning_rate

            if self.verbose > 0:
                if self.sim_dict["path_algorithm"] in ("ppo", "a2c"):
                    self._callback_logger.info(
                        "[LearnRateEntCallback] Episode %d finished. "
                        "LR: %.6f, EntCoef: %.6f",
                        self.iteration,
                        self.current_learning_rate,
                        self.current_entropy
                    )
                else:
                    self._callback_logger.info(
                        "[LearnRateEntCallback] Episode %d finished. LR: %.6f",
                        self.iteration,
                        self.current_learning_rate
                    )

        return True
