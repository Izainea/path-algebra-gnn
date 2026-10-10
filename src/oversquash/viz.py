"""
Network-drawing helpers for the didactic notebooks.

These render small directed graphs with networkx so a reader can *see the
network* they are defining with aiq-quivers. The theoretical figures in
notebooks/figs-theory/ (hand-drawn SVGs) carry the conceptual explanations; this
module only draws the concrete graphs that the code cells build.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

try:
    import networkx as nx
except Exception:  # pragma: no cover
    nx = None


def draw_graph(edges, n_nodes, src=0, dst=None, pos=None, title="", ax=None):
    """Draw a small directed graph; source blue, target green, rest grey."""
    dst = n_nodes - 1 if dst is None else dst
    G = nx.DiGraph(); G.add_nodes_from(range(n_nodes)); G.add_edges_from(edges)
    pos = pos or nx.spring_layout(G, seed=1)
    cols = ['#34d399' if i == dst else '#60a5fa' if i == src else '#cbd5e1'
            for i in range(n_nodes)]
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(5.5, 3))
    nx.draw(G, pos, ax=ax, node_color=cols, with_labels=True, node_size=620,
            arrowsize=14, edge_color='#94a3b8', font_size=10)
    ax.set_title(title)
    if own:
        plt.show()
    return ax


def draw_bottleneck(data, K, M, title=""):
    """Layered drawing of a bottleneck graph: sources | middle layers | target."""
    G = nx.DiGraph(); G.add_edges_from(data.edge_index.t().tolist())
    N = data.num_nodes; t = int(data.root_mask.nonzero()[0])
    layer = {}
    for n in range(K):
        layer[n] = (0, K / 2 - n)
    mid = list(range(K, N - 1))
    for idx, n in enumerate(mid):
        layer[n] = (1 + idx // M, M / 2 - (idx % M))
    layer[t] = (1 + len(mid) // M + 1, 0)
    cols = ['#60a5fa'] * K + ['#fbbf24'] * (N - 1 - K) + ['#34d399']
    fig, ax = plt.subplots(figsize=(7, 4))
    nx.draw(G, layer, ax=ax, node_color=cols, with_labels=True, node_size=620,
            font_size=8, arrowsize=11, edge_color='#cbd5e1')
    ax.set_title(title)
    plt.show()
    return fig
