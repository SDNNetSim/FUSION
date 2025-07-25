import numpy as np
import optuna

import torch.nn as nn  # pylint: disable=consider-using-from-import

from fusion.modules.rl.algorithms.bandits import get_q_table

from fusion.modules.rl.args.general_args import EPISODIC_STRATEGIES


# TODO: (version 5.5-6) Clean up functions to work with shared hyper parameters and cut lines of code (DRL mostly)

class HyperparamConfig:  # pylint: disable=too-few-public-methods
    """
    Controls all hyperparameter starts, ends, and episodic and or time step modifications.
    """

    def __init__(self, engine_props: dict, rl_props: object, is_path: bool):
        self.engine_props = engine_props
        self.total_iters = engine_props['max_iters']
        self.num_nodes = rl_props.num_nodes
        self.is_path = is_path
        if is_path:
            self.n_arms = engine_props['k_paths']
        else:
            self.n_arms = engine_props['cores_per_link']

        self.iteration = 0
        self.curr_reward = None
        self.state_action_pair = None
        self.action_index = None
        self.alpha_strategy = engine_props['alpha_update']
        self.epsilon_strategy = engine_props['epsilon_update']

        if self.alpha_strategy not in EPISODIC_STRATEGIES or self.epsilon_strategy not in EPISODIC_STRATEGIES:
            self.fully_episodic = False
        else:
            self.fully_episodic = True

        self.alpha_start = engine_props['alpha_start']
        self.alpha_end = engine_props['alpha_end']
        self.curr_alpha = self.alpha_start

        self.epsilon_start = engine_props['epsilon_start']
        self.epsilon_end = engine_props['epsilon_end']
        self.curr_epsilon = self.epsilon_start

        self.temperature = None
        self.counts = None
        self.values = None
        self.reward_list = None
        self.decay_rate = engine_props['decay_rate']

        self.alpha_strategies = {
            'softmax': self._softmax_alpha,
            'reward_based': self._reward_based_alpha,
            'state_based': self._state_based_alpha,
            'exp_decay': self._exp_alpha,
            'linear_decay': self._linear_alpha,
        }
        self.epsilon_strategies = {
            'softmax': self._softmax_eps,
            'reward_based': self._reward_based_eps,
            'state_based': self._state_based_eps,
            'exp_decay': self._exp_eps,
            'linear_decay': self._linear_eps,
        }

        if self.iteration == 0:
            self.reset()

    def _softmax(self, q_vals_list: list):
        """
        Compute the softmax probabilities for a given set of Q-values
        """
        exp_values = np.exp(np.array(q_vals_list) / self.temperature)
        probabilities = exp_values / np.sum(exp_values)
        return probabilities

    def _softmax_eps(self):
        """
        Softmax epsilon update rule.
        """
        raise NotImplementedError

    def _softmax_alpha(self):
        """
        Softmax alpha update rule.
        """
        raise NotImplementedError

    def _reward_based_eps(self):
        """
        Reward-based epsilon update.
        """
        if len(self.reward_list) != 2:
            print('Did not update epsilon due to the length of the reward list.')
            return

        curr_reward, last_reward = self.reward_list
        reward_diff = abs(curr_reward - last_reward)
        self.curr_epsilon = self.epsilon_start * (1 / (1 + reward_diff))

    def _reward_based_alpha(self):
        """
        Reward-based alpha update.
        """
        if len(self.reward_list) != 2:
            print('Did not update alpha due to the length of the reward list.')
            return

        curr_reward, last_reward = self.reward_list
        reward_diff = abs(curr_reward - last_reward)
        self.curr_alpha = self.alpha_start * (1 / (1 + reward_diff))

    def _state_based_eps(self):
        """
        State visitation epsilon update.
        """
        self.counts[self.state_action_pair][self.action_index] += 1
        total_visits = self.counts[self.state_action_pair][self.action_index]
        self.curr_epsilon = self.epsilon_start / (1 + total_visits)

    def _state_based_alpha(self):
        """
        State visitation alpha update.
        """
        self.counts[self.state_action_pair][self.action_index] += 1
        total_visits = self.counts[self.state_action_pair][self.action_index]
        self.curr_alpha = 1 / (1 + total_visits)

    def _exp_eps(self):
        """
        Exponential distribution epsilon update.
        """
        self.curr_epsilon = self.epsilon_start * (self.decay_rate ** self.iteration)

    def _exp_alpha(self):
        """
        Exponential distribution alpha update.
        """
        self.curr_alpha = self.alpha_start * (self.decay_rate ** self.iteration)

    def _linear_eps(self):
        """
        Linear decay epsilon update.
        """
        self.curr_epsilon = self.epsilon_end + (
                (self.epsilon_start - self.epsilon_end) * (self.total_iters - self.iteration) / self.total_iters
        )

    def _linear_alpha(self):
        """
        Linear decay alpha update.
        """
        self.curr_alpha = self.alpha_end + (
                (self.alpha_start - self.alpha_end) * (self.total_iters - self.iteration) / self.total_iters
        )

    def update_timestep_data(self, state_action_pair: tuple, action_index: int):
        """
        Updates data structures used for updating alpha and epsilon.
        """
        self.state_action_pair = state_action_pair
        self.action_index = action_index

        if len(self.reward_list) == 2:
            # Moves old current reward to now last reward, current reward always first index
            self.reward_list = [self.curr_reward, self.reward_list[0]]
        elif len(self.reward_list) == 1:
            last_reward = self.reward_list[0]
            self.reward_list = [self.curr_reward, last_reward]
        else:
            self.reward_list.append(self.curr_reward)

    def update_eps(self):
        """
        Update epsilon.
        """
        if self.epsilon_strategy in self.epsilon_strategies:
            self.epsilon_strategies[self.epsilon_strategy]()
        else:
            raise NotImplementedError(f'{self.epsilon_strategy} not in any known strategies: {self.epsilon_strategies}')

    def update_alpha(self):
        """
        Updates alpha.
        """
        if self.alpha_strategy in self.alpha_strategies:
            self.alpha_strategies[self.alpha_strategy]()
        else:
            raise NotImplementedError(f'{self.alpha_strategy} not in any known strategies: {self.alpha_strategies}')

    def reset(self):
        """
        Resets certain class variables.
        """
        self.reward_list = list()
        self.counts, self.values = get_q_table(self=self)


def _get_activation(trial: optuna.Trial):
    name = trial.suggest_categorical("activation_fn", ["relu", "tanh", "elu"])
    return {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}[name]


def _mlp_arch(trial: optuna.Trial, prefix: str, min_layers: int = 2, max_layers: int = 3):
    layer_choices = [32, 64, 128, 256, 512]
    n_layers = trial.suggest_int(f"{prefix}_n_layers", min_layers, max_layers)
    return [trial.suggest_categorical(f"{prefix}_layer_{i + 1}", layer_choices) for i in range(n_layers)]


def _policy_kwargs_actor_critic(trial: optuna.Trial):
    return dict(
        ortho_init=True,
        activation_fn=_get_activation(trial),
        net_arch=dict(pi=_mlp_arch(trial, "pi"), vf=_mlp_arch(trial, "vf"))
    )


def _policy_kwargs_dqn(trial: optuna.Trial, prefix: str):
    return dict(net_arch=_mlp_arch(trial, prefix))


def _ppo_hyperparams(sim_dict: dict, trial: optuna.Trial):
    params = dict()
    params["normalize"] = True
    params["n_timesteps"] = sim_dict["num_requests"] * sim_dict["max_iters"]
    params["policy"] = "MultiInputPolicy"
    params["n_steps"] = trial.suggest_categorical("n_steps", [128, 256, 512, 1024, 2048])
    params["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    params["n_epochs"] = trial.suggest_int("n_epochs", 3, 20)
    params["gamma"] = trial.suggest_float("gamma", 0.90, 0.999)
    params["gae_lambda"] = trial.suggest_float("gae_lambda", 0.8, 1.0)
    params["clip_range"] = trial.suggest_float("clip_range", 0.1, 0.4)
    params["clip_range_vf"] = trial.suggest_float("clip_range_vf", 0.0, 0.4)
    params["normalize_advantage"] = trial.suggest_categorical("normalize_advantage", [True, False])
    params["vf_coef"] = trial.suggest_float("vf_coef", 0.2, 1.0)
    params["max_grad_norm"] = trial.suggest_float("max_grad_norm", 0.3, 1.0)
    params["alpha_start"] = trial.suggest_float("learning_rate_start", 1e-5, 1e-3, log=True)
    params["alpha_end"] = trial.suggest_float("learning_rate_end", 1e-6, params["alpha_start"], log=True)
    params["epsilon_start"] = trial.suggest_float("ent_coef_start", 1e-4, 1e-1, log=True)
    params["epsilon_end"] = trial.suggest_float("ent_coef_end", 1e-5, params["epsilon_start"], log=True)
    params["learning_rate"] = params["alpha_start"]
    params["ent_coef"] = params["epsilon_start"]
    params["decay_rate"] = trial.suggest_float("decay_rate", 0.10, 0.9999, log=True)
    params["use_sde"] = trial.suggest_categorical("use_sde", [False, True])
    params["sde_sample_freq"] = trial.suggest_int("sde_sample_freq", -1, 16)
    params["policy_kwargs"] = _policy_kwargs_actor_critic(trial)
    return params


def _a2c_hyperparams(sim_dict: dict, trial: optuna.Trial):
    params = dict()
    params["normalize"] = True
    params["n_timesteps"] = sim_dict["num_requests"] * sim_dict["max_iters"]
    params["policy"] = "MultiInputPolicy"
    params["n_steps"] = trial.suggest_categorical("n_steps", [5, 16, 32, 64, 128])
    params["gamma"] = trial.suggest_float("gamma", 0.90, 0.999)
    params["gae_lambda"] = trial.suggest_float("gae_lambda", 0.8, 1.0)
    params["vf_coef"] = trial.suggest_float("vf_coef", 0.2, 1.0)
    params["max_grad_norm"] = trial.suggest_float("max_grad_norm", 0.1, 1.0)
    params["alpha_start"] = trial.suggest_float("learning_rate_start", 1e-5, 1e-2, log=True)
    params["alpha_end"] = trial.suggest_float("learning_rate_end", 1e-6, params["alpha_start"], log=True)
    params["epsilon_start"] = trial.suggest_float("ent_coef_start", 1e-4, 1e-1, log=True)
    params["epsilon_end"] = trial.suggest_float("ent_coef_end", 1e-5, params["epsilon_start"], log=True)
    params["learning_rate"] = params["alpha_start"]
    params["ent_coef"] = params["epsilon_start"]
    params["decay_rate"] = trial.suggest_float("decay_rate", 0.10, 0.9999, log=True)
    params["use_rms_prop"] = trial.suggest_categorical("use_rms_prop", [True, False])
    params["rms_prop_eps"] = trial.suggest_float("rms_prop_eps", 1e-6, 1e-4, log=True)
    params["policy_kwargs"] = _policy_kwargs_actor_critic(trial)
    return params


def _dqn_hyperparams(sim_dict: dict, trial: optuna.Trial):
    params = dict()
    params["normalize"] = True
    params["n_timesteps"] = sim_dict["num_requests"] * sim_dict["max_iters"]
    params["policy"] = "MultiInputPolicy"
    params["alpha_start"] = trial.suggest_float("learning_rate_start", 1e-5, 1e-2, log=True)
    params["alpha_end"] = trial.suggest_float("learning_rate_end", 1e-6, params["alpha_start"], log=True)
    params["learning_rate"] = params['alpha_start']
    params["buffer_size"] = trial.suggest_int("buffer_size", 50000, 500000, step=50000)
    params["learning_starts"] = trial.suggest_int("learning_starts", 1000, 10000, step=1000)
    params["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    params["tau"] = trial.suggest_float("tau", 0.005, 1.0, log=True)
    params["gamma"] = trial.suggest_float("gamma", 0.90, 0.999)
    params["train_freq"] = trial.suggest_categorical("train_freq", [1, 2, 4])
    params["gradient_steps"] = trial.suggest_int("gradient_steps", 1, 8)
    params["target_update_interval"] = trial.suggest_int("target_update_interval", 500, 5000, step=500)
    params["exploration_initial_eps"] = 1.0
    params["exploration_fraction"] = trial.suggest_float("exploration_fraction", 0.05, 0.3)
    params["exploration_final_eps"] = trial.suggest_float("exploration_final_eps", 0.01, 0.1)
    params["max_grad_norm"] = trial.suggest_int("max_grad_norm", 5, 20)
    params["policy_kwargs"] = _policy_kwargs_dqn(trial, "dqn")
    return params


def _qr_dqn_hyperparams(sim_dict: dict, trial: optuna.Trial):
    params = _dqn_hyperparams(sim_dict, trial)
    params["n_quantiles"] = trial.suggest_int("n_quantiles", 25, 200, step=25)
    params["policy_kwargs"] = dict(
        net_arch=_mlp_arch(trial, "qr_dqn"),
        n_quantiles=params["n_quantiles"],
    )
    return params


def _drl_hyperparams(sim_dict: dict, trial: optuna.Trial):
    alg = sim_dict["path_algorithm"].lower()
    if alg == "a2c":
        return _a2c_hyperparams(sim_dict, trial)
    if alg == "ppo":
        return _ppo_hyperparams(sim_dict, trial)
    if alg == "dqn":
        return _dqn_hyperparams(sim_dict, trial)
    if alg == "qr_dqn":
        return _qr_dqn_hyperparams(sim_dict, trial)
    raise NotImplementedError(f"Algorithm '{alg}' not supported.")


def get_optuna_hyperparams(sim_dict: dict, trial: optuna.trial):
    """
    Suggests hyperparameters for the Optuna trial.
    """
    resp_dict = dict()

    if sim_dict['path_algorithm'] in ('a2c', 'ppo', 'dqn', 'qr_dqn'):
        resp_dict = _drl_hyperparams(sim_dict=sim_dict, trial=trial)
        return resp_dict

    # There is no alpha in bandit algorithms
    if 'bandit' not in sim_dict['path_algorithm']:
        resp_dict['alpha_start'] = trial.suggest_float('alpha_start', low=0.01, high=0.5, log=False, step=0.01)
        resp_dict['alpha_end'] = trial.suggest_float('alpha_end', low=0.01, high=0.1, log=False, step=0.01)
        resp_dict['cong_cutoff'] = trial.suggest_float('cong_cutoff', low=0.1, high=0.9, log=False, step=0.1)
    else:
        resp_dict['alpha_start'], resp_dict['alpha_end'] = None, None

    if 'ucb' in sim_dict['path_algorithm']:
        resp_dict['conf_param'] = trial.suggest_float('conf_param (c)', low=1.0, high=5.0, log=False, step=0.01)
        resp_dict['epsilon_start'] = None
        resp_dict['epsilon_end'] = None
    else:
        resp_dict['epsilon_start'] = trial.suggest_float('epsilon_start', low=0.01, high=0.5, log=False, step=0.01)
        resp_dict['epsilon_end'] = trial.suggest_float('epsilon_end', low=0.01, high=0.1, log=False, step=0.01)

    if 'q_learning' in (sim_dict['path_algorithm']):
        resp_dict['discount_factor'] = trial.suggest_float('discount_factor', low=0.8, high=1.0, step=0.01)
    else:
        resp_dict['discount_factor'] = None

    if ('exp_decay' in (sim_dict['epsilon_update'], sim_dict['alpha_update']) and
            'ucb' not in sim_dict['path_algorithm']):
        resp_dict['decay_rate'] = trial.suggest_float('decay_rate', low=0.1, high=0.5, step=0.01)
    else:
        resp_dict['decay_rate'] = None

    return resp_dict
