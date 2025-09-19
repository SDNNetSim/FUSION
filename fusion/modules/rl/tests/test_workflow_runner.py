# TODO: (version 5.5-6) Address all duplicate code if you can

# reinforcement_learning/workflow_runner.py

# pylint: disable=import-outside-toplevel, used-before-assignment, duplicate-code

"""
Utility helpers that run / train our RL simulations.

The file now **grace-fully degrades** when optional heavyweight libraries
are missing so that unit-tests which only touch the lightweight helpers
(`_setup_callbacks`, `_update_episode_stats`, …) can import the module
without pulling PyTorch, Optuna or psutil onto the CI image.
"""
from __future__ import annotations

import os
import types
from typing import Any, TYPE_CHECKING

import numpy as np

# ----------------------------------------------------------------------
# Optional dependency: Optuna
# ----------------------------------------------------------------------
try:
    import optuna  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – stub out the bare minimum
    optuna = types.ModuleType("optuna")  # type: ignore


    class _TrialStub:  # pylint: disable=too-few-public-methods
        """
        Trial stub for optuna.Trial
        """
        def report(self, *_a, **_kw):
            """
            Mock report.
            """
            return

        def should_prune(self) -> bool:
            """
            Mock should prune.
            """
            return False


    class _TrialPrunedStub(Exception):  # pylint: disable=too-few-public-methods
        """Stub raised when Optuna would prune a trial."""


    optuna.Trial = _TrialStub  # type: ignore[attr-defined]
    optuna.TrialPruned = _TrialPrunedStub  # type: ignore[attr-defined]

# ----------------------------------------------------------------------
# Optional dependency: psutil   (only needed in run_iters – stub suffices)
# ----------------------------------------------------------------------
try:
    import psutil  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    psutil = types.ModuleType("psutil")  # type: ignore


    class _ProcessStub:  # pylint: disable=too-few-public-methods
        def memory_info(self):
            """
            Mock memory info.
            """
            return types.SimpleNamespace(rss=0)


    psutil.Process = _ProcessStub  # type: ignore[attr-defined]

# ----------------------------------------------------------------------
# Internal helpers – imported lazily below so tests that *don’t* touch
# training code don’t drag the full RL stack into the interpreter.
# ----------------------------------------------------------------------
if TYPE_CHECKING:  # for static type-checkers only
    from fusion.modules.rl.gymnasium_envs.general_sim_env import SimEnv
    from fusion.modules.rl.utils.setup import print_info, setup_rl_sim
    from fusion.modules.rl.model_manager import get_model, save_model
    from fusion.modules.rl.utils.hyperparams import get_optuna_hyperparams
    from fusion.modules.rl.utils.general_utils import save_arr
    from fusion.modules.rl.args.general_args import (
        VALID_PATH_ALGORITHMS,
        VALID_CORE_ALGORITHMS,
        VALID_DRL_ALGORITHMS,
    )


# ======================================================================
# Helper functions that the unit-tests actually exercise
# ======================================================================
def _setup_callbacks(callback_list, sim_dict: dict[str, Any]):
    """
    Propagate *max_iters* and *sim_dict* to every callback object in the list.
    """
    if callback_list:
        for cb in callback_list.callbacks:
            cb.max_iters = sim_dict["max_iters"]
            cb.sim_dict = sim_dict


def _update_episode_stats(
        obs,
        reward: float,
        terminated: bool,
        truncated: bool,
        episodic_reward: float,
        episodic_rew_arr: np.ndarray,
        completed_episodes: int,
        completed_trials: int,
        env,
        sim_dict: dict[str, Any],
        rewards_matrix: np.ndarray,
        trial: optuna.Trial | None = None,
):
    """
    Book-keeping that happens at the *end of every step*.

    Returns a 5-tuple that the caller can unpack directly back into its loop:
    (obs, episodic_reward, episodic_rew_arr, completed_episodes, completed_trials)
    """
    episodic_reward += reward
    if not (terminated or truncated):
        return (
            obs,
            episodic_reward,
            episodic_rew_arr,
            completed_episodes,
            completed_trials,
        )

    # Episode just ended ------------------------------------------------
    episodic_rew_arr = np.append(episodic_rew_arr, episodic_reward)
    episodic_reward = 0
    completed_episodes += 1

    # Optuna pruning (only if we *have* a real trial object) -----------
    if trial is not None:
        current_mean_reward = np.mean(episodic_rew_arr) if episodic_rew_arr.size else 0
        trial.report(current_mean_reward, completed_episodes)
        if trial.should_prune():
            raise optuna.TrialPruned()  # type: ignore[misc]

    print(
        f"{completed_episodes} episodes completed "
        f"out of {sim_dict['max_iters']}."
    )

    # Trial boundary ----------------------------------------------------
    if completed_episodes == sim_dict["max_iters"]:
        # reset per-trial counters / arrays
        env.iteration = 0
        env.trial += 1
        rewards_matrix[completed_trials] = episodic_rew_arr
        episodic_rew_arr = np.array([])
        completed_trials += 1
        completed_episodes = 0
        print(
            f"{completed_trials} trials completed "
            f"out of {sim_dict['n_trials']}."
        )

    # Always reset env at the end of an episode
    obs, _ = env.reset(seed=completed_trials)
    return (
        obs,
        episodic_reward,
        episodic_rew_arr,
        completed_episodes,
        completed_trials,
    )


# ======================================================================
# Heavyweight training / study helpers – **unchanged logic** but imports
# are done *inside functions* so importing the module never fails.
# ======================================================================
def _run_drl_training(env: object, sim_dict: dict, yaml_dict: dict | None = None):
    """
    Train a deep-RL model using Stable-Baselines3 – only executed in
    real training runs, never during light unit tests.
    """
    model, yaml_dict = get_model(
        sim_dict=sim_dict, device=sim_dict["device"], env=env, yaml_dict=yaml_dict
    )
    model.learn(
        total_timesteps=yaml_dict["n_timesteps"],
        log_interval=sim_dict["print_step"],
        callback=sim_dict["callback"],
    )
    save_model(sim_dict=sim_dict, env=env, model=model)


def _train_drl_trial(
        env,
        sim_dict,
        callback_list,
        completed_trials: int,
        rewards_matrix: np.ndarray,
):
    """
    Run a full DRL training trial (all episodes) and return new obs / counters.
    """
    _run_drl_training(env=env, sim_dict=sim_dict)
    rewards_matrix[completed_trials] = callback_list.callbacks[0].episode_rewards
    callback_list.callbacks[0].episode_rewards = np.array([])

    completed_trials += 1
    env.trial = completed_trials

    for cb in callback_list.callbacks:
        cb.trial += 1

    callback_list.callbacks[1].current_entropy = sim_dict["epsilon_start"]
    callback_list.callbacks[1].current_learning_rate = sim_dict["alpha_start"]
    callback_list.callbacks[1].iter = 0
    env.iteration = 0

    print(f"{completed_trials} trials completed out of {sim_dict['n_trials']}.")
    obs, _ = env.reset(seed=completed_trials)
    return obs, completed_trials


def run_iters(
        env: object,
        sim_dict: dict,
        is_training: bool,
        drl_agent: bool,
        model=None,
        callback_list: list | None = None,
        trial: optuna.Trial | None = None,
):
    """
    Execute the environment loop for *n_trials × max_iters* episodes.
    """
    from fusion.modules.rl.utils.general_utils import save_arr

    process = psutil.Process()
    memory_usage_list: list[float] = []

    completed_episodes = 0
    completed_trials = 0
    episodic_reward = 0.0
    rewards_matrix = np.zeros((sim_dict["n_trials"], sim_dict["max_iters"]))
    episodic_rew_arr = np.array([])

    _setup_callbacks(callback_list, sim_dict)

    obs, _ = env.reset(seed=completed_trials)

    while completed_trials < sim_dict["n_trials"]:
        memory_usage_list.append(process.memory_info().rss / (1024 * 1024))

        if is_training and drl_agent:
            obs, completed_trials = _train_drl_trial(
                env, sim_dict, callback_list, completed_trials, rewards_matrix
            )
            continue

        if is_training:
            obs, reward, term, trunc, _ = env.step(0)
        else:
            action, _ = model.predict(obs)
            obs, reward, term, trunc, _ = env.step(action)

        (
            obs,
            episodic_reward,
            episodic_rew_arr,
            completed_episodes,
            completed_trials,
        ) = _update_episode_stats(
            obs,
            reward,
            term,
            trunc,
            episodic_reward,
            episodic_rew_arr,
            completed_episodes,
            completed_trials,
            env,
            sim_dict,
            rewards_matrix,
            trial,
        )

    if is_training:
        mean_per_iter = np.mean(rewards_matrix, axis=0)
        save_arr(mean_per_iter, sim_dict, "average_rewards.npy")
        save_arr(memory_usage_list, sim_dict, "memory_usage.npy")
        return float(np.sum(mean_per_iter))

    raise NotImplementedError("Testing mode not implemented yet.")


# ----------------------------------------------------------------------
# Public high-level helpers (run / run_optuna_study) – unchanged API
# ----------------------------------------------------------------------
def run(
        env: object,
        sim_dict: dict,
        callback_list: list | None = None,
        trial: optuna.Trial | None = None,
):
    """
    High-level wrapper that dispatches to either training or testing.
    """
    from fusion.modules.rl.utils.setup import print_info
    from fusion.modules.rl.args.general_args import (
        VALID_PATH_ALGORITHMS,
        VALID_CORE_ALGORITHMS,
        VALID_DRL_ALGORITHMS,
    )

    print_info(sim_dict=sim_dict)

    if sim_dict["is_training"]:
        if (
                sim_dict["path_algorithm"] in VALID_PATH_ALGORITHMS
                or sim_dict["core_algorithm"] in VALID_CORE_ALGORITHMS
        ):
            return run_iters(
                env=env,
                sim_dict=sim_dict,
                is_training=True,
                drl_agent=sim_dict["path_algorithm"] in VALID_DRL_ALGORITHMS,
                callback_list=callback_list,
                trial=trial,
            )
        raise NotImplementedError(
            "Unsupported (path_algorithm, core_algorithm) combination."
        )

    raise NotImplementedError("Testing mode not implemented yet.")


def run_optuna_study(sim_dict: dict, callback_list):
    """
    Launch an Optuna study for hyper-parameter optimisation.
    """
    from fusion.modules.rl.gymnasium_envs.general_sim_env import SimEnv
    from fusion.modules.rl.utils.setup import setup_rl_sim
    from fusion.modules.rl.utils.hyperparams import get_optuna_hyperparams
    from fusion.sim.utils import (
        modify_multiple_json_values,
        update_dict_from_list,
        get_erlang_vals,
        run_simulation_for_erlangs,
        save_study_results,
    )
    from optuna.pruners import HyperbandPruner

    def objective(tr: optuna.Trial):  # noqa: ANN001
        env = SimEnv(render_mode=None, custom_callback=callback_list, sim_dict=setup_rl_sim())
        for cb in callback_list.callbacks:
            cb.sim_dict = env.sim_dict
            cb.max_iters = sim_dict["max_iters"]
        env.sim_dict["callback"] = callback_list.callbacks

        hyperparam_dict = get_optuna_hyperparams(sim_dict=sim_dict, trial=tr)
        update_list = [(p, v) for p, v in hyperparam_dict.items() if p in sim_dict]

        file_path = os.path.join(
            "data", "input", sim_dict["network"], sim_dict["date"], sim_dict["sim_start"]
        )
        modify_multiple_json_values(trial_num=tr.number, file_path=file_path, update_list=update_list)
        env.sim_dict = update_dict_from_list(input_dict=env.sim_dict, updates_list=update_list)

        erl_list = get_erlang_vals(sim_dict=sim_dict)
        mean_reward = run_simulation_for_erlangs(
            env=env,
            erlang_list=erl_list,
            sim_dict=sim_dict,
            run_func=run,
            callback_list=callback_list,
            trial=tr,
        )
        return mean_reward / sim_dict["max_iters"]

    study = optuna.create_study(
        direction="maximize",
        study_name="hyperparam_study.pkl",
        pruner=HyperbandPruner(
            min_resource=20,
            max_resource=sim_dict["max_iters"],
            reduction_factor=3,
        ),
    )
    study.optimize(objective, n_trials=sim_dict["optuna_trials"])

    best = study.best_trial
    save_study_results(  # pylint: disable=unexpected-keyword-arg
        study=study,
        env=best.user_attrs.get("env"),
        study_name="hyperparam_study.pkl",
        best_params=best.params,
        best_reward=best.value,
        best_sim_start=best.user_attrs.get("sim_start_time"),
    )


__all__ = [
    "_setup_callbacks",
    "_update_episode_stats",
    "_run_drl_training",
    "_train_drl_trial",
    "run_iters",
    "run",
    "run_optuna_study",
]
