# pylint: disable=protected-access

import unittest
import os
from unittest.mock import patch, MagicMock
from plot_scripts import plot_stats


class TestPlotStats(unittest.TestCase):
    """
    Class that contains methods to test plot_stats.py
    """

    def setUp(self):
        self.mock_sims_info_dict = {
            'networks_matrix': [['Network1', 'Network2']],
            'times_matrix': [['Time1', 'Time2']],
            'dates_matrix': [['Date1', 'Date2']],
        }

        # Mock PlotHelpers to avoid side effects
        self.plot_helpers_patch = patch('plot_scripts.plot_stats.PlotHelpers', autospec=True)
        self.mock_plot_helpers = self.plot_helpers_patch.start()
        self.mock_plot_helpers_instance = self.mock_plot_helpers.return_value
        self.mock_plot_helpers_instance.get_file_info = MagicMock()

        # Initialize PlotStats object
        self.plot_stats = plot_stats.PlotStats(sims_info_dict=self.mock_sims_info_dict)

    def tearDown(self):
        self.plot_helpers_patch.stop()

    @patch('plot_scripts.plot_stats.plt')
    def test_setup_plot(self, mock_plt):
        """
        Tests setting up a plot.
        """
        title = "Test Title"
        y_lim = [0.1, 10]
        y_label = "Y Axis"
        x_label = "X Axis"

        # Call the method with all default parameters
        self.plot_stats._setup_plot(title, y_lim, y_label, x_label)

        # Verify that the plot was set up correctly
        mock_plt.figure.assert_called_once_with(figsize=(6.4, 4.8), dpi=100, layout='constrained')
        mock_plt.title.assert_called_once_with(f"{self.plot_stats.props.title_names} {title}")
        mock_plt.ylabel.assert_called_once_with(y_label)
        mock_plt.xlabel.assert_called_once_with(x_label)
        mock_plt.yscale.assert_called_once_with('log')
        mock_plt.xticks.assert_called_once()
        mock_plt.xlim.assert_called_once()
        mock_plt.grid.assert_called_once()

        # Check that ylim was called with the correct arguments
        mock_plt.ylim.assert_any_call(0.1, 10)
        mock_plt.ylim.assert_any_call(1e-05, 1)

        # Reset mocks and call without grid
        mock_plt.reset_mock()
        self.plot_stats._setup_plot(title, y_lim, y_label, x_label, grid=False)
        mock_plt.figure.assert_called_once_with(figsize=(6.4, 4.8), dpi=100, layout='constrained')
        mock_plt.grid.assert_not_called()

    @patch('plot_scripts.plot_stats.plt')
    def test_plot_helper_one(self, mock_plt):
        """
        Tests plot helper one.
        """
        x_vals = 'x'
        y_vals_list = ['y1', 'y2']
        legend_val_list = ['legend1', 'legend2']
        legend_str = False
        file_name = 'test_plot.png'

        # Mock plot_props to simulate realistic plotting scenario
        self.plot_stats.props.plot_dict = {
            'sim1': {
                'info1': {
                    'x': [1, 2, 3],
                    'y1': [4, 5, 6],
                    'y2': [7, 8, 9],
                    'legend1': 'Legend 1',
                    'legend2': 'Legend 2'
                }
            }
        }
        self.plot_stats.props.style_list = ['-', '--']
        self.plot_stats.props.color_list = ['r', 'g']
        self.plot_stats.props.x_tick_list = [1, 2, 3]

        with patch.object(self.plot_stats, '_save_plot', autospec=True) as mock_save_plot:
            self.plot_stats._plot_helper_one(x_vals, y_vals_list, legend_val_list, legend_str, file_name)

            # Ensure plotting calls were made
            self.assertEqual(mock_plt.plot.call_count, len(y_vals_list) * len(legend_val_list))
            self.assertEqual(mock_plt.legend.call_count, 1)
            mock_save_plot.assert_called_once_with(file_name=file_name)

    @patch('plot_scripts.plot_stats.plt')
    def test_plot_helper_two(self, mock_plt):
        """
        Tests plot helper two.
        """
        y_vals_list = ['sum_errors_list']
        erlang = 700
        file_name = 'test_plot.png'

        # Mock plot_props to simulate realistic plotting scenario
        self.plot_stats.props.plot_dict = {
            'sim1': {
                'info1': {
                    'erlang_list': [700, 800],
                    'sum_errors_list': [[1, 2, 3], [4, 5, 6]]
                }
            }
        }
        self.plot_stats.props.style_list = ['-']
        self.plot_stats.props.color_list = ['r']

        with patch.object(self.plot_stats, '_save_plot', autospec=True) as mock_save_plot:
            self.plot_stats._plot_helper_two(y_vals_list, erlang, file_name)

            # Ensure plotting calls were made
            self.assertEqual(mock_plt.plot.call_count, 1)
            self.assertEqual(mock_plt.axhline.call_count, 1)
            mock_save_plot.assert_called_once_with(file_name=file_name)

    @patch('plot_scripts.plot_stats.plt')
    def test_save_plot(self, mock_plt):
        """
        Tests the save plot functionality.
        """
        file_name = 'test_save_plot.png'

        with patch('plot_scripts.plot_stats.create_dir', autospec=True) as mock_create_dir:
            self.plot_stats._save_plot(file_name)

            # Verify that directory creation was called
            mock_create_dir.assert_called_once()

            # Verify that savefig was called with the correct path
            expected_save_path = os.path.join('..', 'data', 'plots', 'Network2', 'Date2', 'Time2', file_name)
            mock_plt.savefig.assert_called_once_with(expected_save_path)


if __name__ == '__main__':
    unittest.main()
