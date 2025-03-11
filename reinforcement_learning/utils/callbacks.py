import os

import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from helper_scripts.os_helpers import create_dir


class GetModelParams(BaseCallback):
    """
    Handles all methods related to custom callbacks in the StableBaselines3 library.
    """

    def __init__(self, verbose: int = 0):
        super(GetModelParams, self).__init__(verbose)  # pylint: disable=super-with-arguments

        self.model_params = None
        self.value_estimate = 0.0

    def _on_step(self) -> bool:
        """
        Every step of the model this method is called. Retrieves the estimated value function for the PPO algorithm.
        """
        self.model_params = self.model.get_parameters()

        obs = self.locals['obs_tensor']
        self.value_estimate = self.model.policy.predict_values(obs=obs)[0][0].item()
        return True


class EpisodicRewardCallback(BaseCallback):
    """
    Finds rewards episodically for any DRL algorithm from SB3.
    """

    def __init__(self, verbose=0):
        super(EpisodicRewardCallback, self).__init__(verbose)  # pylint: disable=super-with-arguments
        self.episode_rewards = np.array([])
        self.current_episode_reward = 0
        self.max_iters = None
        self.sim_dict = None

        self.iter = 0
        self.trial = 0

    def _save_drl_trial_rewards(self):
        erlang = float(self.sim_dict['erlang_start'])
        cores = int(self.sim_dict['cores_per_link'])
        file_path = os.path.join('logs', self.sim_dict['path_algorithm'], self.sim_dict['network'],
                                 self.sim_dict['date'], self.sim_dict['sim_start'])
        create_dir(file_path=file_path)

        file_name = os.path.join(file_path, f'rewards_e{erlang}_routes_c{cores}_t{self.trial}_iter_{self.iter}.npy')
        np.save(file_name, self.episode_rewards)

    def _on_step(self) -> bool:
        reward = self.locals.get("rewards", 0)[0]
        done = self.locals.get("dones", False)[0]
        self.current_episode_reward += reward

        if done:
            self.episode_rewards = np.append(self.episode_rewards, self.current_episode_reward)
            if ((self.iter % self.sim_dict['save_step']) == 0) or (self.iter == self.max_iters - 1):
                self._save_drl_trial_rewards()

            self.iter += 1
            if self.verbose:
                print(f"Episode {len(self.episode_rewards)} finished with reward: {self.current_episode_reward}")
                if len(self.episode_rewards) == self.max_iters:
                    self.current_episode_reward = 0
                    self.iter = 0
                    return False

            self.current_episode_reward = 0

        return True
