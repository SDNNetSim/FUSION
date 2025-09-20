import os

import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from fusion.utils.os import create_dir


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

        self.iteration = 0
        self.trial = 1
        self.current_step = 0

        self.rewards_matrix = None

    def _save_drl_trial_rewards(self):
        """
        Save the average rewards for the current DRL trial to a NumPy file.
        
        Creates the necessary directory structure and saves rewards with
        descriptive filename including Erlang load, core count, trial number,
        and iteration information.
        """
        erlang = float(self.sim_dict['erlang_start'])
        cores = int(self.sim_dict['cores_per_link'])
        file_path = os.path.join('logs', self.sim_dict['path_algorithm'], self.sim_dict['network'],
                                 self.sim_dict['date'], self.sim_dict['sim_start'])
        create_dir(file_path=file_path)

        file_name = os.path.join(file_path, f'rewards_e{erlang}_routes_c{cores}_t{self.trial}_iter_{self.iteration}.npy')
        rewards_matrix = self.rewards_matrix[:self.iteration + 1, :].mean(axis=0)
        np.save(file_name, rewards_matrix)

    def _on_step(self) -> bool:
        if self.rewards_matrix is None:
            self.rewards_matrix = np.empty((self.sim_dict['max_iters'], self.sim_dict['num_requests']))

        reward = self.locals.get("rewards", 0)[0]
        done = self.locals.get("dones", False)[0]
        self.current_episode_reward += reward
        self.rewards_matrix[self.iteration, self.current_step] = reward
        self.current_step += 1

        if done:
            self.episode_rewards = np.append(self.episode_rewards, self.current_episode_reward)
            if ((self.iteration % self.sim_dict['save_step']) == 0) or (self.iteration == self.max_iters - 1):
                self._save_drl_trial_rewards()

            self.iteration += 1
            self.current_step = 0
            if self.verbose:
                print(f"Episode {len(self.episode_rewards)} finished with reward: {self.current_episode_reward}")
                if len(self.episode_rewards) == self.max_iters:
                    self.current_episode_reward = 0
                    self.iteration = 0
                    return False

            self.current_episode_reward = 0

        return True


class LearnRateEntCallback(BaseCallback):
    """
    Callback to decay learning rate linearly and entropy coefficient exponentially
    after each episode, using the same done-based logic as EpisodicRewardCallback.
    """

    def __init__(self, verbose=1):
        super(LearnRateEntCallback, self).__init__(verbose)  # pylint: disable=super-with-arguments
        self.sim_dict = None
        self.iteration = 0
        self.trial = 1

        self.current_entropy = None
        self.current_learning_rate = None

    # NOTE: Parameters should be reset after trial completion for proper isolation
    # Verify that learning rate and entropy coefficient updates are applied to the SB3 model
    def _on_step(self) -> bool:
        done = self.locals.get("dones", [False])[0]
        if done:
            if self.current_entropy is None:
                self.current_entropy = self.sim_dict['epsilon_start']
                self.current_learning_rate = self.sim_dict['alpha_start']

            self.iteration += 1

            progress = min(self.iteration / self.sim_dict['max_iters'], 1.0)
            self.current_learning_rate = self.sim_dict['alpha_start'] + (
                    self.sim_dict['alpha_end'] - self.sim_dict['alpha_start']) * progress

            if self.sim_dict['path_algorithm'] in ('ppo', 'a2c'):
                self.current_entropy = max(self.sim_dict['epsilon_end'], self.current_entropy * self.sim_dict['decay_rate'])
                self.model.ent_coef = self.current_entropy
            self.model.learning_rate = self.current_learning_rate

            if self.verbose > 0:
                if self.sim_dict['path_algorithm'] in ('ppo', 'a2c'):
                    print(f"[LearnRateEntCallback] Episode {self.iteration} finished. "
                          f"LR: {self.current_learning_rate:.6f}, EntCoef: {self.current_entropy:.6f}")
                else:
                    print(f"[LearnRateEntCallback] Episode {self.iteration} finished. "
                          f"LR: {self.current_learning_rate:.6f}")

        return True
