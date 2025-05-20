import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

#TODO Add titles to each graph.
#TODO Implement throughput plotting.
#TODO Add method of combining different graphs into one heat map.
def plot_link_usage(final_result, save_path=None, title=None):
    """
    final_result: { algo: { erlang: { link_str: usage_count } } }
    """
    for algo, traffic_dict in final_result.items():
        for tv, usage_dict in traffic_dict.items():
            G = nx.Graph()

            # Build graph from usage_dict
            for link_str, usage in usage_dict.items():
                u, v = link_str.split('-')
                G.add_edge(u, v, usage=usage)

            usage_values = [G[u][v].get('usage', 0) for u, v in G.edges()]
            max_usage = max(usage_values) if usage_values else 1
            edge_colors = [(1.0, 0.5, 0.0, usage / max_usage) for usage in usage_values]
            edge_widths = [1 + 4 * (usage / max_usage) for usage in usage_values]

            pos = nx.spring_layout(G, seed=42)
            fig, ax = plt.subplots(figsize=(10, 7))

            nx.draw(
                G, pos, ax = ax,
                with_labels=True,
                node_color='lightblue',
                edge_color=edge_colors,
                width=edge_widths
            )
            nx.draw_networkx_edge_labels(
                G, pos, ax = ax,
                edge_labels={(u, v): G[u][v].get('usage', 0) for u, v in G.edges()},
                font_color='gray'
            )

            final_title = f"{title or 'Link Usage'} – {algo} – {tv} Erlang"
            print(final_title)

            ax.set_title(final_title, fontsize=16, fontweight='bold')
            ax.axis('off')

            plt.subplots_adjust(top=0.88)  # Leave space for title

            #plt.tight_layout(rect=[0, 0, 1, 0.95])

            if save_path:
                path = Path(save_path)
                filename = f"{path.stem}_{algo}_{tv}{path.suffix}"
                output_path = path.with_name(filename)
                plt.savefig(output_path, bbox_inches='tight')
                print(f"[plot_link_usage] ✅ Saved: {output_path}")
                plt.close(fig)
            else:
                print(f"[plot_link_usage] (no save path) – Showing: {algo}, Erlang {tv}")
                plt.show()
                plt.close(fig)

