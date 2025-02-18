# pylint: disable=protected-access

import unittest
from unittest.mock import patch, MagicMock, call

import numpy as np


from rl_scripts.workflow_runner import (
    _run_drl_training, run_iters, run_testing, run, run_optuna_study
)


class TestWorkflowRunner(unittest.TestCase):
    """
    Unit tests for the workflow_runner module.
    """

    def setUp(self):
        """
        Common setup for test cases.
        Initializes mock environment and simulation dictionaries.
        """
        self.mock_env = MagicMock()
        self.mock_env.reset.return_value = (np.array([0]), None)

        self.sim_dict = {
            'n_trials': 3,
            'max_iters': 2,
            'is_training': True,
            'path_algorithm': 'algorithm1',
            'core_algorithm': 'algorithm2',
            'optimize_hyperparameters': False,
            'optimize': False,
            'device': 'cpu',
            'callback': None,
            'print_step': 10
        }

    @patch('workflow_runner.run_rl_zoo')
    @patch('workflow_runner.get_model')
    @patch('workflow_runner.save_model')
    def test_run_drl_training(self, mock_save_model, mock_get_model, mock_run_rl_zoo):
        """
        Tests the _run_drl_training function for training flow.
        """

        # Mock the model behavior
        mock_model = MagicMock()
        mock_model.learn.return_value = None
        mock_get_model.return_value = (mock_model, {'n_timesteps': 100})

        # Test with optimize hyperparameters as False
        _run_drl_training(env=self.mock_env, sim_dict=self.sim_dict)

        mock_get_model.assert_called_once_with(sim_dict=self.sim_dict, device='cpu', env=self.mock_env)
        mock_model.learn.assert_called_once_with(total_timesteps=100, log_interval=10, callback=None)
        mock_save_model.assert_called_once_with(sim_dict=self.sim_dict, env=self.mock_env, model=mock_model)

        # Test with optimize hyperparameters as True
        self.sim_dict['optimize_hyperparameters'] = True
        _run_drl_training(env=self.mock_env, sim_dict=self.sim_dict)
        mock_run_rl_zoo.assert_called_once_with(sim_dict=self.sim_dict)

    def test_run_iters_training_mode(self):
        """
        Tests the run_iters function in training mode.
        """
        rewards_matrix = np.zeros((self.sim_dict['n_trials'], self.sim_dict['max_iters']))
        self.mock_env.step.return_value = (np.array([0]), 1, False, False, {})

        result = run_iters(env=self.mock_env, sim_dict=self.sim_dict, is_training=True, drl_agent=False)

        self.assertEqual(result, None)  # No result expected for training mode

    def test_run_iters_non_training_mode(self):
        """
        Tests the run_iters function in non-training mode.
        """
        rewards_matrix = np.zeros((self.sim_dict['n_trials'], self.sim_dict['max_iters']))
        self.sim_dict['is_training'] = False

        mock_model = MagicMock()
        mock_model.predict.return_value = (1, None)
        self.mock_env.step.return_value = (np.array([0]), 1, True, False, {})

        result = run_iters(env=self.mock_env, sim_dict=self.sim_dict, is_training=False, drl_agent=False,
                           model=mock_model)

        # Arrays should be saved; result should be sum of mean rewards
        self.assertIsNotNone(result)
        mock_model.predict.assert_called()

    @patch('workflow_runner.get_trained_model')
    @patch('workflow_runner.run_iters')
    def test_run_testing(self, mock_run_iters, mock_get_trained_model):
        """
        Tests the run_testing function.
        """
        mock_model = MagicMock()
        mock_get_trained_model.return_value = mock_model

        run_testing(env=self.mock_env, sim_dict=self.sim_dict)

        mock_get_trained_model.assert_called_once_with(env=self.mock_env, sim_dict=self.sim_dict)
        mock_run_iters.assert_called_once_with(env=self.mock_env, sim_dict=self.sim_dict, is_training=False,
                                               model=mock_model)

    @patch('workflow_runner.print_info')
    @patch('workflow_runner.run_iters')
    @patch('workflow_runner.run_testing')
    def test_run(self, mock_run_testing, mock_run_iters, mock_print_info):
        """
        Tests the run function for different scenarios.
        """
        # Testing training flow
        run(env=self.mock_env, sim_dict=self.sim_dict)
        mock_print_info.assert_called_once_with(sim_dict=self.sim_dict)
        mock_run_iters.assert_called_once()

        # Testing testing flow
        self.sim_dict['is_training'] = False
        run(env=self.mock_env, sim_dict=self.sim_dict)
        mock_run_testing.assert_called_once()

    @patch('workflow_runner.optuna.create_study')
    @patch('workflow_runner.save_study_results')
    def test_run_optuna_study(self, mock_save_study_results, mock_create_study):
        """
        Tests the run_optuna_study function.
        """
        mock_study = MagicMock()
        mock_trial = MagicMock()
        mock_create_study.return_value = mock_study

        def mock_objective(trial):
            return 1.0

        mock_study.optimize.return_value = None
        mock_study.best_trial = MagicMock(value=1.0, params={}, user_attrs={})

        run_optuna_study(env=self.mock_env, sim_dict=self.sim_dict)

        mock_create_study.assert_called_once_with(direction='maximize', study_name="hyperparam_study.pkl")
        mock_study.optimize.assert_called()
        mock_save_study_results.assert_called()


if __name__ == '__main__':
    unittest.main()
