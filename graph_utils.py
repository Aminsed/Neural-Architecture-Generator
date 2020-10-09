import networkx as nx
import matplotlib.pyplot as plt

def draw_graph(G):
    fig, ax = plt.subplots(figsize=(20, 10))
    pos = nx.multipartite_layout(G, subset_key="layer")
    nx.draw(G, pos,
            with_labels=True,
            font_weight='bold',
            node_size=500,
            node_color='lightblue',
            font_size=10,
            ax=ax)
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close(fig)