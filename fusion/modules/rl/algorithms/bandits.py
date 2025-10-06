import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from fusion.modules.rl.algorithms.algorithm_props import BanditProps
from fusion.modules.rl.algorithms.persistence import BanditModelPersistence
from fusion.utils.os import create_directory


def _ensure_bandit_attributes(bandit_obj: object) -> None:
    """Ensure bandit object has required attributes for type safety."""
    assert hasattr(bandit_obj, 'engine_props'), "bandit must have engine_props"
    assert hasattr(bandit_obj, 'props'), "bandit must have props"
    assert hasattr(bandit_obj, 'is_path'), "bandit must have is_path"
    assert hasattr(bandit_obj, 'values'), "bandit must have values"
    assert hasattr(bandit_obj, 'counts'), "bandit must have counts"
    assert hasattr(bandit_obj, 'num_nodes'), "bandit must have num_nodes"
    assert hasattr(bandit_obj, 'n_arms'), "bandit must have n_arms"
    assert hasattr(bandit_obj, 'source'), "bandit must have source"
    assert hasattr(bandit_obj, 'dest'), "bandit must have dest"
    assert hasattr(bandit_obj, 'iteration'), "bandit must have iteration"


def load_model(train_fp: str) -> dict[str, Any]:
    """
    Load a pre-trained bandit model.

    :param train_fp: File path the model has been saved on
    :type train_fp: str
    :return: The state-value functions V(s, a)
    :rtype: dict[str, Any]
    """
    full_path = Path('logs') / train_fp
    with open(full_path, encoding='utf-8') as file_obj:
        state_vals_dict: dict[str, Any] = json.load(file_obj)

    return state_vals_dict


def _get_base_fp(is_path: bool, erlang: float, cores_per_link: int) -> str:
    if is_path:
        base_fp = f"e{erlang}_routes_c{cores_per_link}"
    else:
        base_fp = f"e{erlang}_cores_c{cores_per_link}"

    return base_fp


def _save_model(
    state_values_dict: dict[str, Any],
    erlang: float,
    cores_per_link: int,
    save_dir: str,
    is_path: bool,
    trial: int
) -> None:
    """
    Delegate to BanditModelPersistence to avoid code duplication.

    :param state_values_dict: Dictionary of state values to save
    :type state_values_dict: dict[str, Any]
    :param erlang: Erlang traffic value
    :type erlang: float
    :param cores_per_link: Number of cores per link
    :type cores_per_link: int
    :param save_dir: Directory to save the model
    :type save_dir: str
    :param is_path: Whether this is a path agent model
    :type is_path: bool
    :param trial: Trial number
    :type trial: int
    """
    BanditModelPersistence.save_model(
        state_values_dict, erlang, cores_per_link, save_dir, is_path, trial
    )


def save_model(iteration: int, algorithm: str, self: object, trial: int) -> None:
    """
    Save a trained bandit model.

    :param iteration: Current iteration
    :type iteration: int
    :param algorithm: The algorithm used
    :type algorithm: str
    :param self: The object to be saved
    :type self: object
    :param trial: Current trial number
    :type trial: int
    """
    _ensure_bandit_attributes(self)
    bandit = cast(Any, self)

    max_iters = bandit.engine_props['max_iters']
    rewards_matrix = bandit.props.rewards_matrix

    should_save = (
        (iteration % bandit.engine_props['save_step'] == 0 or
         iteration == max_iters - 1) and
        len(bandit.props.rewards_matrix[iteration]) ==
        bandit.engine_props['num_requests']
    )

    if should_save:
        rewards_matrix = np.array(rewards_matrix)
        rewards_arr = rewards_matrix.mean(axis=0)

        date_time_path = (
            Path(bandit.engine_props['network']) /
            bandit.engine_props['date'] /
            bandit.engine_props['sim_start']
        )
        save_dir = Path('logs') / algorithm / date_time_path
        create_directory(directory_path=str(save_dir))

        erlang = bandit.engine_props['erlang']
        cores_per_link = bandit.engine_props['cores_per_link']
        base_fp = _get_base_fp(
            is_path=bandit.is_path, erlang=erlang, cores_per_link=cores_per_link
        )

        rewards_fp = f'rewards_{base_fp}_t{trial + 1}_iter_{iteration}.npy'
        save_fp = Path.cwd() / save_dir / rewards_fp
        np.save(save_fp, rewards_arr)

        BanditModelPersistence.save_model(
            state_values_dict=bandit.values,
            erlang=erlang,
            cores_per_link=cores_per_link,
            save_dir=str(save_dir),
            is_path=bandit.is_path,
            trial=trial
        )


def get_q_table(self: object) -> tuple[dict, dict]:
    """
    Construct the q-table.

    :param self: The current bandit object
    :type self: object
    :return: The initial V(s, a) and N(s, a) values
    :rtype: tuple[dict, dict]
    """
    # Validate essential attributes (not values/counts since we're creating them)
    bandit = cast(Any, self)
    assert hasattr(bandit, 'engine_props'), "bandit must have engine_props"
    assert hasattr(bandit, 'props'), "bandit must have props"
    assert hasattr(bandit, 'is_path'), "bandit must have is_path"
    assert hasattr(bandit, 'num_nodes'), "bandit must have num_nodes"
    assert hasattr(bandit, 'n_arms'), "bandit must have n_arms"

    bandit.counts = {}
    bandit.values = {}
    for source in range(bandit.num_nodes):
        for destination in range(bandit.num_nodes):
            if source == destination:
                continue

            if bandit.is_path:
                bandit.counts[(source, destination)] = np.zeros(bandit.n_arms)
                bandit.values[(source, destination)] = np.zeros(bandit.n_arms)
            else:
                for path_index in range(bandit.engine_props['k_paths']):
                    key = (source, destination, path_index)
                    bandit.counts[key] = np.zeros(bandit.n_arms)
                    bandit.values[key] = np.zeros(bandit.n_arms)

    return bandit.counts, bandit.values


def _update_bandit(
    self: object,
    iteration: int,
    reward: float,
    arm: int,
    algorithm: str,
    trial: int
) -> None:
    """
    Update bandit values with new reward.

    :param self: The bandit object
    :type self: object
    :param iteration: Current iteration
    :type iteration: int
    :param reward: Received reward
    :type reward: float
    :param arm: Selected arm
    :type arm: int
    :param algorithm: Algorithm name
    :type algorithm: str
    :param trial: Trial number
    :type trial: int
    """
    _ensure_bandit_attributes(self)
    bandit = cast(Any, self)

    if bandit.is_path:
        pair: tuple[Any, ...] = (bandit.source, bandit.dest)
    else:
        pair = (bandit.source, bandit.dest, bandit.path_index)

    bandit.counts[pair][arm] += 1
    n_times = bandit.counts[pair][arm]
    value = bandit.values[pair][arm]
    bandit.values[pair][arm] = value + (reward - value) / n_times

    if bandit.iteration >= len(bandit.props.rewards_matrix):
        bandit.props.rewards_matrix.append([])
    bandit.props.rewards_matrix[bandit.iteration].append(reward)

    # Check if we need to save the model
    save_model(iteration=iteration, algorithm=algorithm, self=bandit, trial=trial)


class EpsilonGreedyBandit:
    """
    Epsilon greedy bandit algorithm.

    Considers N(s, a) using counts to update state-action values Q(s, a).
    """

    def __init__(self, rl_props: object, engine_props: dict, is_path: bool) -> None:
        """
        Initialize epsilon greedy bandit.

        :param rl_props: Reinforcement learning properties
        :type rl_props: object
        :param engine_props: Engine properties
        :type engine_props: dict
        :param is_path: Whether this is a path agent
        :type is_path: bool
        """
        self.props = BanditProps()
        self.engine_props = engine_props
        self.rl_props = rl_props
        self.completed_sim = False
        self.iteration = 0
        self.is_path = is_path

        self.source: int | None = None
        self.dest: int | None = None
        self.path_index: int | None = None  # Index of the last chosen path

        if is_path:
            self.n_arms = engine_props['k_paths']
        else:
            self.n_arms = engine_props['cores_per_link']

        self.epsilon: float | None = None
        self.num_nodes = cast(Any, rl_props).num_nodes
        # Amount of times an action has been taken and every V(s,a)
        self.counts, self.values = get_q_table(self=self)

    def _get_action(self, state_action_pair: tuple) -> int:
        """Get action using epsilon-greedy strategy."""
        assert self.epsilon is not None, "epsilon must be set"
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.n_arms))

        return int(np.argmax(self.values[state_action_pair]))

    def select_path_arm(self, source: int, dest: int) -> int:
        """
        Select a bandit's arm for the path agent only.

        :param source: Source node
        :type source: int
        :param dest: Destination node
        :type dest: int
        :return: The action selected
        :rtype: int
        """
        self.source = source
        self.dest = dest
        state_action_pair = (source, dest)
        return self._get_action(state_action_pair=state_action_pair)

    def select_core_arm(self, source: int, dest: int, path_index: int) -> int:
        """
        Select a bandit's arm for the core agent only.

        :param source: Source node
        :type source: int
        :param dest: Destination node
        :type dest: int
        :param path_index: Path index selected prior
        :type path_index: int
        :return: The action selected
        :rtype: int
        """
        self.source = source
        self.dest = dest
        self.path_index = path_index
        state_action_pair = (source, dest, path_index)
        return self._get_action(state_action_pair=state_action_pair)

    def update(self, arm: int, reward: int, iteration: int, trial: int) -> None:
        """
        Make updates to the bandit after each time step or episode.

        :param arm: The arm selected
        :type arm: int
        :param reward: Reward received from R(s, a)
        :type reward: int
        :param iteration: Current episode or iteration
        :type iteration: int
        :param trial: Current trial number
        :type trial: int
        """
        _update_bandit(
            self=self,
            iteration=iteration,
            reward=reward,
            arm=arm,
            algorithm='epsilon_greedy_bandit',
            trial=trial
        )


class UCBBandit:
    """
    Upper Confidence Bound bandit algorithm.

    Considers N(s, a) using counts to update state-action values Q(s, a).
    """

    def __init__(self, rl_props: object, engine_props: dict, is_path: bool) -> None:
        """
        Initialize UCB bandit.

        :param rl_props: Reinforcement learning properties
        :type rl_props: object
        :param engine_props: Engine properties
        :type engine_props: dict
        :param is_path: Whether this is a path agent
        :type is_path: bool
        """
        self.props = BanditProps()
        self.engine_props = engine_props
        self.rl_props = rl_props
        self.completed_sim = False
        self.iteration = 0
        self.is_path = is_path

        self.path_index: int | None = None
        self.source: int | None = None
        self.dest: int | None = None
        self.num_nodes = cast(Any, rl_props).num_nodes

        if is_path:
            self.n_arms = engine_props['k_paths']
        else:
            self.n_arms = engine_props['cores_per_link']

        self.counts, self.values = get_q_table(self=self)

    def _get_action(self, state_action_pair: tuple) -> int:
        """Get action using UCB strategy."""
        if 0 in self.counts[state_action_pair]:
            return int(np.argmin(self.counts[state_action_pair]))

        conf_param = self.engine_props['conf_param']
        total_counts = sum(self.counts[state_action_pair])
        ucb_values = (
            self.values[state_action_pair] +
            np.sqrt(
                conf_param * np.log(total_counts) / self.counts[state_action_pair]
            )
        )
        return int(np.argmax(ucb_values))

    def select_path_arm(self, source: int, dest: int) -> int:
        """
        Select a bandit's arm for the path agent only.

        :param source: Source node
        :type source: int
        :param dest: Destination node
        :type dest: int
        :return: The action selected
        :rtype: int
        """
        self.source = source
        self.dest = dest
        state_action_pair = (source, dest)

        return self._get_action(state_action_pair=state_action_pair)

    def select_core_arm(self, source: int, dest: int, path_index: int) -> int:
        """
        Select a bandit's arm for the core agent only.

        :param source: Source node
        :type source: int
        :param dest: Destination node
        :type dest: int
        :param path_index: Path index selected prior
        :type path_index: int
        :return: The action selected
        :rtype: int
        """
        self.source = source
        self.dest = dest
        self.path_index = path_index
        state_action_pair = (source, dest, path_index)

        return self._get_action(state_action_pair=state_action_pair)

    def update(self, arm: int, reward: int, iteration: int, trial: int) -> None:
        """
        Make updates to the bandit after each time step or episode.

        :param arm: The arm selected
        :type arm: int
        :param reward: Reward received from R(s, a)
        :type reward: int
        :param iteration: Current episode or iteration
        :type iteration: int
        :param trial: Current trial number
        :type trial: int
        """
        _update_bandit(
            iteration=iteration,
            arm=arm,
            reward=reward,
            self=self,
            algorithm='ucb_bandit',
            trial=trial
        )
