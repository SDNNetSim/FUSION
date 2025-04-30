# ✅ blocking.py (updated for k_shortest_path_X styling)
import matplotlib.pyplot as plt
import numpy as np

def plot_blocking_probabilities(final_result, save_path=None, title=None):
    """
    Plot the blocking probability versus Erlang values using the final_result data
    with a publication-quality style.

    Parameters:
        final_result (dict): Processed simulation data for blocking probabilities.
        save_path (Path or str): Optional path to save the figure.
        title (str): Optional title to display on the plot.
    """
    available_styles = plt.style.available
    if 'seaborn-whitegrid' in available_styles:
        plt.style.use('seaborn-whitegrid')
    elif 'seaborn-white' in available_styles:
        plt.style.use('seaborn-white')
    else:
        plt.style.use('default')

    plt.figure(figsize=(10, 6), dpi=300)
    plotted = False

    for algo_key, traffic_data in final_result.items():
        erlang_values = []
        blocking_probs = []
        for tv_str, sim_block_vectors in traffic_data.items():
            try:
                tv = float(tv_str)
            except ValueError:
                continue
            final_blocks = []

            if isinstance(sim_block_vectors, (int, float)):
                final_blocks.append(sim_block_vectors)
            else:
                for block_vector in sim_block_vectors:
                    if isinstance(block_vector, list) and len(block_vector) > 0:
                        final_blocks.extend(block_vector[-10:])
                    elif isinstance(block_vector, (int, float)):
                        final_blocks.append(block_vector)

            if final_blocks:
                avg_block = np.mean(final_blocks)
                erlang_values.append(tv)
                blocking_probs.append(avg_block)

        if erlang_values:
            sorted_pairs = sorted(zip(erlang_values, blocking_probs), key=lambda x: x[0])
            sorted_erlangs, sorted_blocks = zip(*sorted_pairs)

            # 🔁 Add visual distinction for k_shortest_path variants
            if "k_shortest_path" in algo_key:
                linestyle = '--'
            else:
                linestyle = '-'

            plt.plot(sorted_erlangs, sorted_blocks,
                     marker='o', linewidth=2, linestyle=linestyle, markersize=6, label=algo_key)
            plotted = True

    if not plotted:
        print(f"[plot_blocking] ⚠️ Skipping empty plot: {title}")
        return

    plt.xlabel('Erlang Values', fontsize=14, fontweight='bold')
    plt.ylabel('Blocking Probability', fontsize=14, fontweight='bold')
    plt.title(title or 'Blocking Probability vs Erlang Values', fontsize=16, fontweight='bold')
    plt.yscale('log')
    plt.ylim(10 ** -4, 10 ** -0.5)
    plt.xlim(50, 700)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if save_path:
        plt.savefig(save_path)
        print(f"[plot_blocking] ✅ Saved: {save_path}")
        plt.close()
    else:
        plt.show()
        plt.clf()
