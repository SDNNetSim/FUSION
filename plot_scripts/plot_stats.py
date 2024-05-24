import os

import matplotlib.pyplot as plt

from arg_scripts.plot_args import empty_props
from helper_scripts.os_helpers import create_dir
from helper_scripts.plot_helpers import PlotHelpers, find_times


class PlotStats:
    """
    A class for computing and plotting statistical analysis for simulations.
    """

    def __init__(self, sims_info_dict: dict):
        self.plot_props = empty_props
        self.sims_info_dict = sims_info_dict
        self.plot_help_obj = PlotHelpers(plot_props=self.plot_props, net_names_list=sims_info_dict['networks_matrix'])

        self.plot_help_obj.get_file_info(sims_info_dict=sims_info_dict)

    def _save_plot(self, file_name: str):
        # Default to the earliest time for saving
        time = self.sims_info_dict['times_matrix'][0][-1]
        network = self.sims_info_dict['networks_matrix'][0][-1]
        date = self.sims_info_dict['dates_matrix'][0][-1]
        save_fp = os.path.join('..', 'data', 'plots', network, date, time)
        create_dir(file_path=save_fp)

        save_fp = os.path.join(save_fp, file_name)
        plt.savefig(save_fp)

    def _setup_plot(self, title: str, y_lim: list, y_label: str, x_label: str, grid: bool = True, y_ticks: bool = True,
                    x_ticks: bool = True):
        plt.figure(figsize=(7, 5), dpi=300)
        plt.title(f"{self.plot_props['title_names']} {title}")
        plt.ylabel(y_label)
        plt.xlabel(x_label)

        if y_ticks:
            plt.yticks([10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1])
            plt.ylim(10 ** -5, 1)
            plt.yscale('log')

        if y_lim:
            plt.ylim(y_lim[0], y_lim[1])

        if x_ticks:
            plt.xticks(self.plot_props['x_tick_list'])
            plt.xlim(self.plot_props['x_tick_list'][0], self.plot_props['x_tick_list'][-1])

        if grid:
            plt.grid()

    def _plot_helper_two(self, y_vals_list: list, erlang: float, file_name: str):
        """
        Meant to plot iter stats with erlang each on its own plot.
        """
        color_count = 0
        style_count = 0
        for _, sims_dict in self.plot_props['plot_dict'].items():
            for _, info_dict in sims_dict.items():
                style = self.plot_props['style_list'][style_count]

                for y_val in y_vals_list:
                    color = self.plot_props['color_list'][color_count]
                    index = [index for index, value in enumerate(info_dict['erlang_list']) if value == erlang][0]
                    x_vals = list(range(len(info_dict[y_val][index])))
                    plt.plot(x_vals, info_dict[y_val][index], linestyle=style, markersize=2.3, color=color)
                    color_count += 1

            color_count = 0
            style_count += 1

        self._save_plot(file_name=file_name)
        plt.show()

    def _plot_helper_one(self, x_vals: str, y_vals_list: list, legend_val_list: list, force_legend: bool,
                         file_name: str):
        legend_list = list()
        color_count = 0
        style_count = 0
        for _, sims_dict in self.plot_props['plot_dict'].items():
            for _, info_dict in sims_dict.items():
                style = self.plot_props['style_list'][style_count]

                for y_val, legend_val in zip(y_vals_list, legend_val_list):
                    color = self.plot_props['color_list'][color_count]
                    plt.plot(info_dict[x_vals], info_dict[y_val], linestyle=style, markersize=2.3, color=color)

                    if not force_legend:
                        legend_val = info_dict[legend_val]
                        legend_list.append(legend_val)
                    else:
                        legend_list = legend_val_list
                    color_count += 1

            color_count = 0
            style_count += 1

        plt.legend(legend_list)
        self._save_plot(file_name=file_name)
        plt.show()

    def plot_errors(self, erlang_list: list):
        """
        Plots temporal difference errors.

        :param erlang_list: A list of desired erlangs to plot separately.
        """
        for erlang in erlang_list:
            self._setup_plot(f"Sum of Errors vs. Iteration Erlang {erlang}", y_label='Sum of Errors',
                             x_label='Iteration', y_ticks=False, x_ticks=False, y_lim=[])
            self._plot_helper_two(y_vals_list=['sum_errors_list'], erlang=float(erlang),
                                  file_name=f'sum_errors_{erlang}')

    def plot_rewards(self, erlang_list: list):
        """
        Plot rewards.

        :param erlang_list: A list of desired erlangs to plot separately.
        """
        for erlang in erlang_list:
            self._setup_plot(f"Sum of Rewards vs. Iteration Erlang {erlang}", y_label='Sum of Rewards',
                             x_label='Iteration', y_ticks=False, x_ticks=False, y_lim=[])
            self._plot_helper_two(y_vals_list=['sum_rewards_list'], erlang=float(erlang),
                                  file_name=f'sum_rewards_{erlang}')

    def plot_block_reasons(self):
        """
        Plots the reasons for blocking as a percentage.
        """
        self._setup_plot("Block Reasons w/ Segment Slicing", y_label='Blocking Percentage', x_label='Erlang',
                         y_ticks=False, x_ticks=False, y_lim=[-0.1, 1.1])
        self._plot_helper_one(x_vals='erlang_list', y_vals_list=['cong_block_list', 'dist_block_list'],
                              legend_val_list=['Congestion', 'Distance'], force_legend=True,
                              file_name='block_reasons')

    def plot_hops(self):
        """
        Plots the average number of hops.
        """
        self._setup_plot("Average Hop Count w/ Segment Slicing", y_label='Average Hop Count', x_label='Erlang',
                         y_ticks=False, y_lim=[])
        self._plot_helper_one(x_vals='erlang_list', y_vals_list=['hops_list'], legend_val_list=['QRC', 'k=3', 'k=1'],
                              force_legend=True, file_name='average_hops')

    def plot_path_length(self):
        """
        Plots the average path length.
        """
        self._setup_plot("Average Path Length w/ Segment Slicing", y_label='Average Path Length (KM)', x_label='Erlang',
                         y_ticks=False, y_lim=[])
        self._plot_helper_one(x_vals='erlang_list', y_vals_list=['lengths_list'], legend_val_list=['QRC', 'k=3', 'k=1'],
                              force_legend=True, file_name='average_lengths')

    def plot_blocking(self, ai=False):
        """
        Plots the average blocking probability for each Erlang value.
        """
        self._setup_plot("Average Blocking Prob. vs. Erlang", y_label='Average Blocking Probability',
                         x_label='Time Steps', y_ticks=False, x_ticks=False, y_lim=[])

        if not ai:
            self._plot_helper_one(x_vals='erlang_list', y_vals_list=['blocking_list'],
                                  legend_val_list=['QRC', 'k=3', 'k=1'], force_legend=True, file_name='average_bp')
        else:
            # TODO: Make block per iter a matrix, for each Erlang
            self._plot_helper_two(y_vals_list=['block_per_iter'], erlang=250, file_name='bp_e{250}')


def main():
    """
    Controls this script.
    """
    filter_dict = {
        'and_filter_list': [
        ],
        'or_filter_list': [
        ],
        'not_filter_list': [
            # ['max_segments', 4],
            # ['max_segments', 8],
        ]
    }

    sims_info_dict = find_times(dates_dict={'0523': 'Pan-European'}, filter_dict=filter_dict)
    plot_obj = PlotStats(sims_info_dict=sims_info_dict)

    plot_obj.plot_blocking(ai=False)
    # plot_obj.plot_path_length()
    # plot_obj.plot_hops()
    # plot_obj.plot_block_reasons()
    # plot_obj.plot_rewards(erlang_list=[250])
    # plot_obj.plot_errors(erlang_list=[250])


if __name__ == '__main__':
    main()
