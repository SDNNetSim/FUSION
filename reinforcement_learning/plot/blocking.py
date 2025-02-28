import matplotlib.pyplot as plt
import numpy as np


def plot_blocking_probabilities(final_result):
    """
    Plot the blocking probability versus Erlang values using the final_result data.

    Parameters:
        final_result (dict): Processed simulation data for blocking probabilities.
    """
    plt.figure(figsize=(8, 6))
    for algo_key, traffic_data in final_result.items():
        erlang_values = []
        blocking_probs = []
        for tv_str, sim_block_vectors in traffic_data.items():
            try:
                tv = float(tv_str)
            except ValueError:
                continue
            final_blocks = []
            for block_vector in sim_block_vectors:
                if isinstance(block_vector, list) and len(block_vector) > 0:
                    final_blocks.append(block_vector[-1])
                elif isinstance(block_vector, (int, float)):
                    final_blocks.append(block_vector)
            if final_blocks:
                avg_block = np.mean(final_blocks)
                erlang_values.append(tv)
                blocking_probs.append(avg_block)
        if erlang_values:
            sorted_pairs = sorted(zip(erlang_values, blocking_probs), key=lambda x: x[0])
            sorted_erlangs, sorted_blocks = zip(*sorted_pairs)
            plt.plot(sorted_erlangs, sorted_blocks, marker='o', label=algo_key)
    plt.xlabel('Erlang Values')
    plt.ylabel('Blocking Probability')
    plt.yscale('log')
    plt.ylim(10 ** -4, 10 ** -0.5)
    plt.title('Blocking Probability vs Erlang Values')
    plt.legend()
    plt.grid(True)
    plt.show()
