# TODO: (version 5.5-6) No longer supported, to be broken into a module based design

import os

import matplotlib.pyplot as plt

from fusion.utils.os import create_directory
from fusion.visualization.properties import PlotProps
from fusion.visualization.utils import PlotHelpers, find_times


class PlotStats:
    """
    A class for computing and plotting statistical analysis for simulations.
    """

    def __init__(self, sims_info_dict: dict):
        self.props = PlotProps()
        self.sims_info_dict = sims_info_dict
        self.plot_help_obj = PlotHelpers(
            plot_props=self.props, net_names_list=sims_info_dict["networks_matrix"]
        )

        self.plot_help_obj.get_file_info(sims_info_dict=sims_info_dict)

    def _save_plot(self, file_name: str) -> None:
        # Default to the earliest time for saving
        time = self.sims_info_dict["times_matrix"][0][-1]
        network = self.sims_info_dict["networks_matrix"][0][-1]
        date = self.sims_info_dict["dates_matrix"][0][-1]
        save_fp = os.path.join("..", "data", "plots", network, date, time)
        create_directory(directory_path=save_fp)

        save_fp = os.path.join(save_fp, file_name)
        plt.savefig(save_fp)

    def _setup_plot(
        self,
        title: str,
        y_lim: list,
        y_label: str,
        x_label: str,
        grid: bool = True,
        y_ticks: bool = True,
        x_ticks: bool = True,
    ) -> None:
        plt.figure(figsize=(6.4, 4.8), dpi=100, layout="constrained")
        plt.title(f"{self.props.title_names} {title}")
        plt.ylabel(y_label)
        plt.xlabel(x_label)

        if y_ticks:
            plt.yticks([10**-4, 10**-3, 10**-2, 10**-1, 1])
            plt.ylim(10**-5, 1)
            plt.yscale("log")

        if y_lim:
            plt.ylim(y_lim[0], y_lim[1])

        if x_ticks:
            plt.xticks(self.props.x_tick_list)
            plt.xlim(self.props.x_tick_list[0], self.props.x_tick_list[-1])

        if grid:
            plt.grid()

    def _plot_helper_two(
        self, y_vals_list: list, erlang: float, file_name: str
    ) -> None:
        """
        Meant to plot iter stats with erlang each on its own plot.
        """
        color_count = 0
        style_count = 0
        if self.props.plot_dict is not None:
            for sims_dict in self.props.plot_dict.values():
                for info_dict in sims_dict.values():
                    style = self.props.style_list[style_count]

                    for y_val in y_vals_list:
                        color = self.props.color_list[color_count]
                        erlang_list = info_dict.erlang_list
                        index: int = [
                            i for i, value in enumerate(erlang_list) if value == erlang
                        ][0]
                        y_val_data = getattr(info_dict, y_val)
                        x_vals = list(range(len(y_val_data[index])))
                        plt.plot(
                            x_vals,
                            y_val_data[index],
                            linestyle=style,
                            markersize=2.3,
                            color=color,
                        )
                        # color_count += 1

                color_count += 1
                style_count = 0

        plt.axhline(y=0.055, color="r", linestyle="--", linewidth=1.5, label="y=0.055")

        plt.legend(["Here"])
        plt.xlim([-5, 199])
        self._save_plot(file_name=file_name)
        plt.show()

    def _plot_helper_one(
        self,
        x_vals: str,
        y_vals_list: list,
        legend_val_list: list,
        force_legend: bool,
        file_name: str,
    ) -> None:
        legend_list: list[str] = []
        color_count = 0
        style_count = 0
        if self.props.plot_dict is not None:
            for sims_dict in self.props.plot_dict.values():
                for info_dict in sims_dict.values():
                    style = self.props.style_list[style_count]
                    for y_val in y_vals_list:
                        for legend_val in legend_val_list:
                            color = self.props.color_list[color_count]

                            # Plot with each combination of y_val and legend_val
                            x_data = getattr(info_dict, x_vals)
                            y_data = getattr(info_dict, y_val)
                            plt.plot(
                                x_data,
                                y_data,
                                linestyle=style,
                                markersize=2.3,
                                color=color,
                            )

                            if force_legend:
                                legend_list.append(legend_val)
                            else:
                                legend_val_str = str(getattr(info_dict, legend_val))
                                legend_list.append(legend_val_str)
                            color_count += 1

                        color_count = 0
                        style_count = 0

        plt.legend(legend_list)
        self._save_plot(file_name=file_name)
        plt.show()

    def plot_errors(self, erlang_list: list) -> None:
        """
        Plots temporal difference errors.

        :param erlang_list: A list of desired erlangs to plot separately.
        """
        for erlang in erlang_list:
            self._setup_plot(
                f"Sum of Errors vs. Iteration Erlang {erlang}",
                y_label="Sum of Errors",
                x_label="Iteration",
                y_ticks=False,
                x_ticks=False,
                y_lim=[],
            )
            self._plot_helper_two(
                y_vals_list=["sum_errors_list"],
                erlang=float(erlang),
                file_name=f"sum_errors_{erlang}",
            )

    def plot_rewards(self, erlang_list: list) -> None:
        """
        Plot rewards.

        :param erlang_list: A list of desired erlangs to plot separately.
        """
        for erlang in erlang_list:
            self._setup_plot(
                f"Sum of Rewards vs. Iteration Erlang {erlang}",
                y_label="Sum of Rewards",
                x_label="Iteration",
                y_ticks=False,
                x_ticks=False,
                y_lim=[],
            )
            self._plot_helper_two(
                y_vals_list=["sum_rewards_list"],
                erlang=float(erlang),
                file_name=f"sum_rewards_{erlang}",
            )

    def plot_block_reasons(self) -> None:
        """
        Plots the reasons for blocking as a percentage.
        """
        self._setup_plot(
            "Block Reasons w/ Segment Slicing",
            y_label="Blocking Percentage",
            x_label="Erlang",
            y_ticks=False,
            x_ticks=False,
            y_lim=[-0.1, 1.1],
        )
        self._plot_helper_one(
            x_vals="erlang_list",
            y_vals_list=["cong_block_list", "dist_block_list"],
            legend_val_list=["Congestion", "Distance"],
            force_legend=True,
            file_name="block_reasons",
        )

    def plot_hops(self) -> None:
        """
        Plots the average number of hops.
        """
        self._setup_plot(
            "Average Hop Count w/ Segment Slicing",
            y_label="Average Hop Count",
            x_label="Erlang",
            y_ticks=False,
            y_lim=[],
        )
        self._plot_helper_one(
            x_vals="erlang_list",
            y_vals_list=["hops_list"],
            legend_val_list=["QRC", "k=3", "k=1"],
            force_legend=True,
            file_name="average_hops",
        )

    def plot_path_length(self) -> None:
        """
        Plots the average path length.
        """
        self._setup_plot(
            "Average Path Length w/ Segment Slicing",
            y_label="Average Path Length (KM)",
            x_label="Erlang",
            y_ticks=False,
            y_lim=[],
        )
        self._plot_helper_one(
            x_vals="erlang_list",
            y_vals_list=["lengths_list"],
            legend_val_list=["QRC", "k=3", "k=1"],
            force_legend=True,
            file_name="average_lengths",
        )

    def plot_blocking(self, art_int: bool = False) -> None:
        """
        Plots the average blocking probability for each Erlang value.

        :param art_int: Artificial intelligence flag for plotting.
        """
        self._setup_plot(
            "Average Blocking Prob. vs. Erlang",
            y_label="Average Blocking Probability",
            x_label="Erlang",
            y_ticks=False,
            x_ticks=False,
            y_lim=[0, 0.15],
        )

        if art_int:
            self._plot_helper_two(
                y_vals_list=["block_per_iter"], erlang=250, file_name="bp_e250"
            )
        else:
            self._plot_helper_one(
                x_vals="erlang_list",
                y_vals_list=["blocking_list"],
                legend_val_list=[
                    "Trained US",
                    "Trained Euro",
                    "Baseline",
                    "Combined Model",
                ],
                force_legend=True,
                file_name="average_bp",
            )


def main() -> None:
    """
    Controls this script.
    """
    filter_dict: dict[str, list] = {
        "and_filter_list": [
            # ['path_algorithm', 'ppo']
        ],
        "or_filter_list": [
            # ['sim_start', '17_07_02_823817'],
            # ['sim_start', '17_25_05_247014'],
            # ['sim_start', '17_53_02_873894'],
            # ['sim_start', '20_26_35_936142'],
            # ['sim_start', '11_19_35_314638'],
        ],
        "not_filter_list": [
            # ['max_segments', 4],
            # ['max_segments', 8],
        ],
    }

    sims_info_dict = find_times(dates_dict={"0411": "NSFNet"}, filter_dict=filter_dict)
    plot_obj = PlotStats(sims_info_dict=sims_info_dict)

    # plot_obj.plot_blocking(art_int=True)
    plot_obj.plot_path_length()


if __name__ == "__main__":
    main()
