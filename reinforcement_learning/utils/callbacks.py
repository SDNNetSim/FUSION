import numpy as np

from stable_baselines3.common.callbacks import BaseCallback


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

        self.iter = 0

    def _on_step(self) -> bool:
        reward = self.locals.get("rewards", 0)[0]
        done = self.locals.get("dones", False)[0]
        self.current_episode_reward += reward

        if done:
            self.episode_rewards = np.append(self.episode_rewards, self.current_episode_reward)
            if self.verbose:
                print(f"Episode {len(self.episode_rewards)} finished with reward: {self.current_episode_reward}")
            self.current_episode_reward = 0

        return True
