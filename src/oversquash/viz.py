"""
Plotting helpers for the didactic notebook series (notebooks/en, notebooks/es).

Every figure used in the tutorials lives here, verified once, so the English and
Spanish notebooks call the *same* functions and stay in sync. Each function takes
an optional `labels` dict so the calling notebook can pass localized titles/axes
without duplicating any logic.

All functions return the Matplotlib figure (and show it). Pure visualization —
the maths is in quiver/ideal_bridge/walk_operators.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

try:
    import networkx as nx
except Exception:  # pragma: no cover
    nx = None


# ---------------------------------------------------------------- graph drawing
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
    import torch  # local; only needed here
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


# ------------------------------------------------------- over-squashing numbers
def plot_message_explosion(depths, msgs, title="", xlabel="", ylabel=""):
    """Semilog plot of messages-at-target vs depth (the explosion)."""
    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    ax.semilogy(depths, msgs, 'o-', color='#dc2626', lw=2)
    for d, m in zip(depths, msgs):
        ax.annotate(str(int(m)), (d, m), textcoords="offset points",
                    xytext=(0, 8), ha='center', fontsize=9)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(alpha=0.3); plt.tight_layout(); plt.show()
    return fig


def plot_multiplicity_heatmap(M, title="", xlabel="source i", ylabel="target j"):
    """Heatmap of a walk-count / weight matrix."""
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(M, cmap='Reds')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.show()
    return fig


# ------------------------------------------------------------- path / algebra
def plot_matrix_pair(A, A2, titles=("A", "A^2")):
    """Show adjacency A and A^2 side by side with integer annotations."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, Mx, ti in zip(axes, (A, A2), titles):
        ax.imshow(Mx, cmap='Blues', vmin=0, vmax=max(2, Mx.max()))
        for i in range(Mx.shape[0]):
            for j in range(Mx.shape[1]):
                ax.text(j, i, int(Mx[i, j]), ha='center', va='center',
                        fontsize=10, color='#1e293b')
        ax.set_title(ti); ax.set_xticks(range(Mx.shape[0])); ax.set_yticks(range(Mx.shape[0]))
    plt.tight_layout(); plt.show()
    return fig


def plot_paths_per_length(lengths, counts, title="", xlabel="", ylabel=""):
    """Bar chart of number of paths src->dst at each length."""
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar([str(l) for l in lengths], counts, color='#6366f1')
    for i, c in enumerate(counts):
        ax.text(i, c + 0.03, str(int(c)), ha='center', fontsize=10, fontweight='bold')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    plt.tight_layout(); plt.show()
    return fig


# -------------------------------------------------- raw vs effective (kQ/I)
def plot_raw_vs_eff(depths, raw_vals, eff_vals, title="", xlabel="",
                    ylabel="", legend=("raw  (A^g)", "effective  (kQ/I)")):
    """Grouped bars: raw vs de-duplicated message counts per depth."""
    fig, ax = plt.subplots(figsize=(6, 3.4))
    x = np.arange(len(depths)); w = 0.38
    ax.bar(x - w/2, raw_vals, w, label=legend[0], color='#f59e0b')
    ax.bar(x + w/2, eff_vals, w, label=legend[1], color='#10b981')
    ax.set_xticks(x); ax.set_xticklabels([str(d) for d in depths])
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(); ax.set_yscale('log'); ax.grid(alpha=0.3, axis='y')
    plt.tight_layout(); plt.show()
    return fig


def plot_two_heatmaps(M1, M2, titles=("raw", "effective")):
    """Two matrices side by side (e.g. A^g vs M_g)."""
    vmax = max(M1.max(), M2.max(), 1)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    for ax, Mx, ti in zip(axes, (M1, M2), titles):
        im = ax.imshow(Mx, cmap='Purples', vmin=0, vmax=vmax)
        ax.set_title(ti); ax.set_xlabel('source i'); ax.set_ylabel('target j')
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025)
    plt.show()
    return fig


# ---------------------------------------------------------------- attention
def plot_mask_grid(raw_ops, n_nodes, title="", subtitle_fmt="g={g}: {k}/{N2}"):
    """Show the walk-reachability mask (attention support) at each range g."""
    G = len(raw_ops)
    fig, axes = plt.subplots(1, G, figsize=(3.2 * G, 3.3))
    if G == 1:
        axes = [axes]
    for g, ax in enumerate(axes):
        mask = (raw_ops[g] > 0).astype(int)
        ax.imshow(mask, cmap='Greens', vmin=0, vmax=1)
        ax.set_title(subtitle_fmt.format(g=g+1, k=int(mask.sum()), N2=n_nodes**2),
                     fontsize=10)
        ax.set_xlabel('source'); ax.set_ylabel('target')
    fig.suptitle(title); plt.tight_layout(); plt.show()
    return fig


def plot_fixed_vs_learned(sources, fixed_w, learned_w, title="",
                          legend=("fixed (walkraw)", "learned (attention)")):
    """Grouped bars: fixed vs learned attention weights into the target."""
    fig, ax = plt.subplots(figsize=(6, 3.4))
    x = np.arange(len(sources)); w = 0.38
    ax.bar(x - w/2, fixed_w, w, label=legend[0], color='#f59e0b')
    ax.bar(x + w/2, learned_w, w, label=legend[1], color='#10b981')
    ax.set_xticks(x); ax.set_xticklabels([f's{int(s)}' for s in sources])
    ax.set_ylabel('weight'); ax.set_title(title); ax.legend()
    plt.tight_layout(); plt.show()
    return fig


# ---------------------------------------------------------------- training
def plot_training_curves(curves: dict, title="", xlabel="epoch", ylabel="accuracy",
                         chance=None):
    """Plot one accuracy-vs-epoch curve per model."""
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for name, c in curves.items():
        ax.plot(range(1, len(c)+1), c, marker='.', label=name)
    if chance is not None:
        ax.axhline(chance, ls='--', color='grey', label='chance')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout(); plt.show()
    return fig


def plot_model_bars(names, accs, title="", ylabel="accuracy", chance=None,
                    colors=None):
    """Final-accuracy bar chart across models."""
    colors = colors or ['#fca5a5', '#fcd34d', '#6ee7b7', '#93c5fd', '#c4b5fd'][:len(names)]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(names, accs, color=colors, edgecolor='#334155')
    for i, v in enumerate(accs):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')
    if chance is not None:
        ax.axhline(chance, ls='--', color='grey', label='chance'); ax.legend()
    ax.set_ylim(0, 1.12); ax.set_ylabel(ylabel); ax.set_title(title)
    plt.tight_layout(); plt.show()
    return fig
