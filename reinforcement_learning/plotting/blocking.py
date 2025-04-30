import matplotlib.pyplot as plt


def plot_blocking_probabilities(final_result, title='Blocking Probability vs Erlang Values', save_path=None):
    """
    Plot the blocking probability versus Erlang values using the final_result data
    with a publication-quality style.

    Parameters:
        final_result (dict): Processed data {algorithm: {traffic_volume: blocking_prob}}
        title (str): Title for the plot.
        save_path (str, optional): If provided, save plot to this file path.
    """
    available_styles = plt.style.available
    if 'seaborn-whitegrid' in available_styles:
        plt.style.use('seaborn-whitegrid')
    elif 'seaborn-white' in available_styles:
        plt.style.use('seaborn-white')
    else:
        plt.style.use('default')

    plt.figure(figsize=(10, 6), dpi=300)

    for algo_key, traffic_data in final_result.items():
        erlang_values = sorted([float(tv) for tv in traffic_data.keys()])
        blocking_probs = [traffic_data[str(tv)] for tv in erlang_values]

        plt.plot(erlang_values, blocking_probs,
                 marker='o', linewidth=2, markersize=6, label=algo_key)

    plt.xlabel('Erlang Values', fontsize=14, fontweight='bold')
    plt.ylabel('Blocking Probability', fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold')

    plt.yscale('log')
    plt.ylim(10 ** -4, 10 ** -0.5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()

    return plt.gcf()
