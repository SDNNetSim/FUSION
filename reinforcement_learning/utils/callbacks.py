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
        self.trial = 1
        self.curr_step = 0

        self.rewards_matrix = None

    def _save_drl_trial_rewards(self):
        erlang = float(self.sim_dict['erlang_start'])
        cores = int(self.sim_dict['cores_per_link'])
        file_path = os.path.join('logs', self.sim_dict['path_algorithm'], self.sim_dict['network'],
                                 self.sim_dict['date'], self.sim_dict['sim_start'])
        create_dir(file_path=file_path)

        file_name = os.path.join(file_path, f'rewards_e{erlang}_routes_c{cores}_t{self.trial}_iter_{self.iter}.npy')
        rewards_matrix = self.rewards_matrix[:self.iter + 1, :].mean(axis=0)
        np.save(file_name, rewards_matrix)

    def _on_step(self) -> bool:
        if self.rewards_matrix is None:
            self.rewards_matrix = np.empty((self.sim_dict['max_iters'], self.sim_dict['num_requests']))

        reward = self.locals.get("rewards", 0)[0]
        done = self.locals.get("dones", False)[0]
        self.current_episode_reward += reward
        self.rewards_matrix[self.iter, self.curr_step] = reward
        self.curr_step += 1

        if done:
            self.episode_rewards = np.append(self.episode_rewards, self.current_episode_reward)
            if ((self.iter % self.sim_dict['save_step']) == 0) or (self.iter == self.max_iters - 1):
                self._save_drl_trial_rewards()

            self.iter += 1
            self.curr_step = 0
            if self.verbose:
                print(f"Episode {len(self.episode_rewards)} finished with reward: {self.current_episode_reward}")
                if len(self.episode_rewards) == self.max_iters:
                    self.current_episode_reward = 0
                    self.iter = 0
                    return False

            self.current_episode_reward = 0

        return True


class LearnRateEntCallback(BaseCallback):
    """
    Callback to decay learning rate linearly and entropy coefficient exponentially
    after each episode, using the same done-based logic as EpisodicRewardCallback.
    """

    def __init__(self, verbose=1):
        super(LearnRateEntCallback, self).__init__(verbose)
        self.sim_dict = None
        self.iter = 0
        self.trial = 1

        self.current_ent = None
        self.current_lr = None

    # TODO: (drl_path_agents) Reset after a trial completes
    #   Also, have the parameters actually changed inside the sb3 model?
    def _on_step(self) -> bool:
        done = self.locals.get("dones", [False])[0]
        if done:
            if self.current_ent is None:
                self.current_ent = self.sim_dict['epsilon_start']
                self.current_lr = self.sim_dict['alpha_start']

            self.iter += 1

            progress = min(self.iter / self.sim_dict['max_iters'], 1.0)
            self.current_lr = self.sim_dict['alpha_start'] + (
                        self.sim_dict['alpha_end'] - self.sim_dict['alpha_start']) * progress
            # for param_group in self.model.policy.optimizer.param_groups:
            #     param_group["lr"] = self.current_lr

            if self.sim_dict['path_algorithm'] in ('ppo', 'a2c'):
                self.current_ent = max(self.sim_dict['epsilon_end'], self.current_ent * self.sim_dict['decay_rate'])
                self.model.ent_coef = self.current_ent
            self.model.learning_rate = self.current_lr

            if self.verbose > 0:
                if self.sim_dict['path_algorithm'] in ('ppo', 'a2c'):
                    print(f"[LearnRateEntCallback] Episode {self.iter} finished. "
                          f"LR: {self.current_lr:.6f}, EntCoef: {self.current_ent:.6f}")
                else:
                    print(f"[LearnRateEntCallback] Episode {self.iter} finished. "
                          f"LR: {self.current_lr:.6f}")

        return True
