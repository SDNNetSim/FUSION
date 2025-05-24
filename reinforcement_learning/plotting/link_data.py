import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import math

#TODO Add titles to each graph.
#TODO Implement throughput plotting.
#TODO Add method of combining different graphs into one heat map.
def plot_link_usage(final_result, save_path=None, title=None):
    """
    Creates a heat map of the averaged usage of links on the final iteration
    across multiple seeds of a simulation.
    """

    usage_data = final_result["usage"]
    for algo, traffic_dict in usage_data.items():
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



def plot_link_throughput(final_result, save_path=None, title=None):
    throughput_data = final_result["throughput"]

    for algo, traffic_dict in throughput_data.items():
        for tv, throughput_dict in traffic_dict.items():
            G = nx.Graph()

            for link_str, throughput in throughput_dict.items():
                u, v = link_str.split('-')
                G.add_edge(u, v, throughput=throughput)

            throughput_values = [G[u][v]["throughput"] for u, v in G.edges()]
            max_throughput = max(throughput_values) if throughput_values else 1
            edge_colors = [(1.0, 0.5, 0.0, v / max_throughput) for v in throughput_values]
            edge_widths = [1 + 4 * (v / max_throughput) for v in throughput_values]

            pos = nx.spring_layout(G, seed=42)

            fig, ax = plt.subplots(figsize=(10, 7))

            nx.draw(
                G, pos, ax=ax,
                with_labels=True,
                node_color='lightblue',
                edge_color=edge_colors,
                width=edge_widths
            )

            draw_rotated_edge_labels(
                G, pos, ax,
                attr="throughput",
                offset=0.02,
                fmt="{:.2f}",
                fontsize=8,
                color="gray"
            )

            final_title = f"{title or 'Link Throughput'} – {algo} – {tv} Erlang"
            ax.set_title(final_title, fontsize=16, fontweight='bold')
            ax.axis('off')
            plt.subplots_adjust(top=0.88)

            if save_path:
                path = Path(save_path)
                filename = f"{path.stem}_{algo}_{tv}{path.suffix}"
                output_path = path.with_name(filename)
                plt.savefig(output_path, bbox_inches='tight')
                print(f"[plot_link_throughput] ✅ Saved: {output_path}")
                plt.close(fig)
            else:
                print(f"[plot_link_throughput] (no save path) – Showing: {algo}, Erlang {tv}")
                plt.show()
                plt.close(fig)




def draw_rotated_edge_labels(G, pos, ax, attr="throughput", offset=0.05, fmt="{:.2f}", **text_kwargs):
    """
    Draws edge labels rotated to follow the edge direction and offset perpendicularly from the edge center.

    Parameters:
        G: networkx.Graph
        pos: dict of node positions {node: (x, y)}
        ax: matplotlib Axes
        attr: edge attribute to label (default 'throughput')
        offset: distance to lift label from edge (default 0.05)
        fmt: format string or callable for label value
        text_kwargs: forwarded to ax.text (e.g., fontsize, color)
    """
    for (u, v) in G.edges():
        if attr not in G[u][v]:
            continue

        x1, y1 = pos[u]
        x2, y2 = pos[v]
        xm, ym = (x1 + x2) / 2, (y1 + y2) / 2

        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length == 0:
            continue  # avoid divide-by-zero

        # Perpendicular unit vector
        perp_dx = -dy / length
        perp_dy = dx / length

        xo = xm + offset * perp_dx
        yo = ym + offset * perp_dy

        # Angle of edge (in degrees)
        angle = math.degrees(math.atan2(dy, dx))

        value = G[u][v][attr]
        label = fmt.format(value) if isinstance(fmt, str) else fmt(value)

        ax.text(
            xo, yo, label,
            rotation=angle,
            rotation_mode='anchor',
            horizontalalignment='center',
            verticalalignment='center',
            **text_kwargs
        )




