import matplotlib.pyplot as plt
import networkx as nx
import json

def plot_link_usage(final_result, save_path=None, title=None):


    for traffic_volume, result in final_result.items():
        link_usage_dict = result["link_usage_dict"]
        topology_edges = result["topology"]

        G = nx.Graph()
        for src, dst in topology_edges:
            G.add_edge(src, dst)

        for link_str, stats in link_usage_dict.items():
            u, v = link_str.split('-')
            if G.has_edge(u, v):
                G[u][v]['usage'] = stats['usage_count']
            else:
                print(f"Warning: Link {u}-{v} not found in topology.")

        usage_values = [G[u][v].get('usage', 0) for u, v in G.edges()]
        max_usage = max(usage_values) if usage_values else 1
        edge_colors = [(1.0, 0.5, 0.0, usage / max_usage) for usage in usage_values]
        edge_widths = [1 + 4 * (usage / max_usage) for usage in usage_values]

        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(10, 7))
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color='lightblue',
            edge_color=edge_colors,
            width=edge_widths
        )
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels={(u, v): G[u][v].get('usage', 0) for u, v in G.edges()},
            font_color='gray'
        )

        final_title = f"{title or 'Link Usage Heatmap'} - Traffic {traffic_volume}"
        plt.title(final_title)
        plt.axis('off')

        if save_path:
            filename = f"{save_path.rstrip('.png')}_{traffic_volume}.png"
            plt.savefig(filename)
            print(f"[plot_link_data] ✅ Saved: {filename}")
            plt.close()
        else:
            plt.show()
            plt.clf()
